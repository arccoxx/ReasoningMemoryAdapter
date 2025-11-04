#this is the working best script
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from datasets import load_dataset
import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import re
import random
import subprocess
import sys
# Prerequisite check for torch_geometric
try:
    from torch_geometric.nn import GATConv
except ImportError:
    print("Installing torch_geometric...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch_geometric"])
    from torch_geometric.nn import GATConv
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=32, alpha=64):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / r
    def forward(self, x): return self.linear(x) + (self.lora_B(self.lora_A(x)) * self.scaling)
class MemoryAugmentedModel(nn.Module):
    def __init__(self, base_model, tokenizer, lora_r=32):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        if "<MEMORY>" not in self.tokenizer.additional_special_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<MEMORY>"], "pad_token": "[PAD]"})
            self.base_model.resize_token_embeddings(len(self.tokenizer))
        self.memory_token_id = self.tokenizer.convert_tokens_to_ids("<MEMORY>")
        self.d = self.base_model.config.hidden_size
        device, dtype = next(self.base_model.parameters()).device, torch.bfloat16
        self.gnn = GATConv(self.d, self.d, heads=4, concat=False).to(device=device, dtype=dtype)
        self.proj = LoRALinear(self.d, self.d, r=lora_r).to(device=device, dtype=dtype)
        self.memory_layernorm = nn.LayerNorm(self.d).to(device=device, dtype=dtype)
        print("Freezing base model parameters. Training GNN adapter only.")
        for p in self.base_model.parameters(): p.requires_grad = False
        for p in self.proj.parameters(): p.requires_grad = True
        for p in self.gnn.parameters(): p.requires_grad = True
        for p in self.memory_layernorm.parameters(): p.requires_grad = True
    def forward_gnn(self, past_hidden_states: List[torch.Tensor]) -> torch.Tensor:
        if not past_hidden_states:
            return torch.zeros(self.d, device=next(self.parameters()).device, dtype=torch.bfloat16)
        node_features = torch.stack(past_hidden_states)
        num_nodes, device = node_features.shape[0], node_features.device
        edge_index = torch.combinations(torch.arange(num_nodes, device=device), r=2).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        aggregated_features = self.gnn(node_features, edge_index)
        memory_vector = aggregated_features[-1, :]
        return self.memory_layernorm(memory_vector)
    def get_inputs_embeds_with_memory(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        memory_offset = self.proj(memory)
        inputs_embeds[:, 0, :] = inputs_embeds[:, 0, :] + memory_offset
        return inputs_embeds
    def get_last_hidden_state(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.get_inputs_embeds_with_memory(input_ids, attention_mask, memory)
        outputs = self.base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True)
        return outputs.hidden_states[-1][:, -1, :]
# FIX: Use the improved, more robust step extraction logic
def extract_reasoning_steps(answer_text: str) -> List[str]:
    """Extracts logical steps from a GSM8K answer, splitting by newlines and then sentences for robustness."""
    reasoning_part = answer_text.split('####')[0].strip()
    if not reasoning_part:
        return []
    lines = reasoning_part.split('\n')
    steps = []
    for line in lines:
        line = line.strip()
        if not line: continue
        sentences = re.findall(r'[^.!?]+[.!?]?', line)
        for sent in sentences:
            sent = sent.strip()
            if sent:
                if steps and len(sent) < 5 and not sent[-1] in '.!?':
                    steps[-1] += " " + sent
                else:
                    steps.append(sent)
    return steps
def parse_final_answer(text: str) -> str:
    if not text: return ""
    gsm_match = re.search(r'####\s*([-\d.,]+)', text)
    if gsm_match: return gsm_match.group(1).replace(',', '').strip()
    box_match = re.search(r'\\boxed{([-\d.,]+)}', text)
    if box_match: return box_match.group(1).replace(',', '').strip()
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if numbers: return numbers[-1].strip()
    return ""
def check_answer_correctness(pred_text: str, true_text: str) -> bool:
    pred_answer = parse_final_answer(pred_text)
    true_answer = parse_final_answer(true_text)
    if not pred_answer or not true_answer: return False
    try: return float(pred_answer) == float(true_answer)
    except (ValueError, TypeError): return pred_answer == true_answer
def prepare_dataset(dataset_split: str) -> List[Dict[str, Any]]:
    dataset = load_dataset("gsm8k", "main", split=dataset_split)
    prepared = []
    for item in dataset:
        reasoning_steps = extract_reasoning_steps(item['answer'])
        final = parse_final_answer(item['answer'])
        if reasoning_steps and final:
             prepared.append({"question": item['question'], "steps": reasoning_steps, "final": final})
    return prepared
# FIX: Corrected function signature
def train(model, train_data: List[Dict], epochs: int = 3, gradient_accumulation_steps: int = 8):
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    model.train()
    optimizer.zero_grad()
    global_step = 0
    for epoch in range(epochs):
        indices = np.random.permutation(len(train_data))
        progress_bar = tqdm(indices, desc=f"Epoch {epoch+1}/{epochs} Train")
        for data_idx in progress_bar:
            data = train_data[data_idx]
            prompt = model.tokenizer.apply_chat_template(
                [{"role": "user", "content": data['question']}],
                tokenize=False, add_generation_prompt=True
            )
            cumul_text = prompt
            with torch.no_grad():
                initial_ids = model.tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
                initial_ids_with_mem = torch.cat([torch.tensor([[model.memory_token_id]], device=initial_ids.input_ids.device), initial_ids.input_ids], dim=1)
                initial_mask = torch.ones_like(initial_ids_with_mem)
                h0 = model.get_last_hidden_state(initial_ids_with_mem, initial_mask, torch.zeros(model.d, device=next(model.parameters()).device, dtype=torch.bfloat16))
                past_h = [h0.squeeze(0)]
            full_chain = data['steps'] + [f"#### {data['final']}"]
            for step_text in full_chain:
                memory = model.forward_gnn(past_h)
                context_ids = model.tokenizer(cumul_text, add_special_tokens=False).input_ids
                context_for_h_ids_tensor = torch.tensor([[model.memory_token_id] + context_ids], device=next(model.parameters()).device)
                context_attention_mask = torch.ones_like(context_for_h_ids_tensor)
                h_t = model.get_last_hidden_state(context_for_h_ids_tensor, context_attention_mask, memory)
                past_h.append(h_t.detach().squeeze(0))
                step_ids = model.tokenizer(" " + step_text, add_special_tokens=False).input_ids
                input_ids = [model.memory_token_id] + context_ids + step_ids
                labels = [-100] * (1 + len(context_ids)) + step_ids
                input_tensor = torch.tensor([input_ids], device=next(model.parameters()).device)
                labels_tensor = torch.tensor([labels], device=next(model.parameters()).device)
                attention_mask = torch.ones_like(input_tensor)
                inputs_embeds = model.get_inputs_embeds_with_memory(input_tensor, attention_mask, memory)
                outputs = model.base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels_tensor)
                causal_loss = outputs.loss
                with torch.no_grad():
                    step_embeds = model.base_model.get_input_embeddings()(torch.tensor([step_ids], device=next(model.parameters()).device))
                    target_embed = step_embeds.mean(dim=1).squeeze(0)
                    h_t_norm = F.normalize(h_t.squeeze(0), p=2, dim=0)
                    target_norm = F.normalize(target_embed, p=2, dim=0)
                aux_loss = F.mse_loss(h_t_norm, target_norm) * .8
                total_loss = causal_loss + aux_loss
                total_loss = total_loss / gradient_accumulation_steps
                total_loss.backward()
                cumul_text += " " + step_text
            global_step += 1
            if global_step % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                if causal_loss is not None:
                    progress_bar.set_postfix({'Causal L': f"{causal_loss.item():.3f}", 'Aux L': f"{aux_loss.item():.3f}"})
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: List[int]): self.stop_token_ids = stop_token_ids
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.stop_token_ids:
            return True
        return False
def evaluate(model, question: str, use_sampling: bool = False, verbose: bool = False) -> Tuple[str, str]:
    device = next(model.parameters()).device
    model.eval()
    log_lines = []

    stop_ids = [model.tokenizer.eos_token_id]
    stop_ids.extend(model.tokenizer("####", add_special_tokens=False).input_ids)
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])
    with torch.no_grad():
        prompt = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True
        )
        cumul_text = prompt

        with torch.no_grad():
            initial_ids = model.tokenizer(prompt, return_tensors="pt").to(device)
            initial_ids_with_mem = torch.cat([torch.tensor([[model.memory_token_id]], device=device), initial_ids.input_ids], dim=1)
            initial_mask = torch.ones_like(initial_ids_with_mem)
            h0 = model.get_last_hidden_state(initial_ids_with_mem, initial_mask, torch.zeros(model.d, device=device, dtype=torch.bfloat16))
            past_h = [h0.squeeze(0)]

        if verbose: log_lines.append(f"===== STARTING EVALUATION: {question[:80]}... =====")
        for step_idx in range(10):
            memory = model.forward_gnn(past_h)
            inputs = model.tokenizer(cumul_text, return_tensors="pt").to(device)
            input_ids_with_mem = torch.cat([torch.tensor([[model.memory_token_id]], device=device), inputs.input_ids], dim=1)
            attention_mask_with_mem = torch.ones_like(input_ids_with_mem)
            inputs_embeds = model.get_inputs_embeds_with_memory(input_ids_with_mem, attention_mask_with_mem, memory)

            gen_tokens = model.base_model.generate(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask_with_mem,
                max_new_tokens=70,  # Reduced to encourage shorter, discrete steps
                do_sample=False,  # Greedy for consistency
                pad_token_id=model.tokenizer.pad_token_id, stopping_criteria=stopping_criteria
            )

            new_text = model.tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
            if verbose: log_lines.append(f"Step {step_idx+1} | Mem Norm: {torch.linalg.norm(memory):.2f} | Gen: '{new_text}'")
            if not new_text and step_idx > 0:
                log_lines.append("--- Generation stalled ---")
                break
            cumul_text += " " + new_text

            # Compute h_t on updated cumul_text
            updated_inputs = model.tokenizer(cumul_text, return_tensors="pt").to(device)
            updated_ids_with_mem = torch.cat([torch.tensor([[model.memory_token_id]], device=device), updated_inputs.input_ids], dim=1)
            updated_mask = torch.ones_like(updated_ids_with_mem)
            h_t = model.get_last_hidden_state(updated_ids_with_mem, updated_mask, memory)
            past_h.append(h_t.squeeze(0))

            if "####" in new_text:
                log_lines.append("--- Final answer detected ---")
                break

        response = cumul_text[len(prompt):].strip()
        generation_log = "\n".join(log_lines)
        return response, generation_log
def evaluate_accuracy(model, test_data: List[Dict], log_successes: bool = False) -> Tuple[float, List[Dict]]:
    accuracies, successful_examples = [], []
    progress_bar = tqdm(test_data, desc="GNN Adapter Eval")
    for i, data in enumerate(progress_bar):
        response, log = evaluate(model, data['question'], verbose=(i < 5 or log_successes))
        is_correct = check_answer_correctness(response, data['final'])
        accuracies.append(1 if is_correct else 0)
        if is_correct and log_successes:
            successful_examples.append({"question": data['question'], "true_answer": data['final'],
                                        "model_response": response, "generation_log": log})
    return np.mean(accuracies), successful_examples
if __name__ == "__main__":
    train_data = prepare_dataset("train")[:2000]
    test_data = prepare_dataset("test")[:200]
    print(f"Loaded {len(train_data)} train and {len(test_data)} test examples.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    print("Initializing GNN adapter...")
    adapter_model = MemoryAugmentedModel(base_model=base_model, tokenizer=tokenizer)
    print("Training GNN adapter...")
    train(adapter_model, train_data, epochs=8)

    print("\n--- Running Final Evaluation and Analysis ---")
    final_acc, successes = evaluate_accuracy(adapter_model, test_data, log_successes=True)
    print(f"Final evaluation accuracy: {final_acc:.2f}")
    if successes:
        print(f"\n--- Analysis of {len(successes)} Successful Responses ---")
        for idx, success in enumerate(successes):
            print(f"\n{'='*25} SUCCESS #{idx+1} {'='*25}")
            print(f"Question: {success['question']}\nTrue Answer: {success['true_answer']}")
            print(f"\nModel Full Response:\n{success['model_response']}")
            print("\n--- Generation Log ---")
            print(success['generation_log'])
            print(f"{'='*60}\n")
    print("\nScript finished.")
