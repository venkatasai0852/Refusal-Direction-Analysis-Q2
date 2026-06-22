import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def main():
    os.environ["HF_TOKEN"] = ""
    
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    dir_path = r"C:\Q2_Sai\q2_experiment\directions\mistral_7b_refusal_direction.pt"
    layer_idx = 22

    print("Loading Mistral-7B in 4-bit...")
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quant_config)

    print("Loading extracted refusal direction...")
    all_dirs = torch.load(dir_path).to(model.device)
    refusal_dir = all_dirs[layer_idx]
    if refusal_dir.dtype != torch.float16:
        refusal_dir = refusal_dir.to(torch.float16)
        
    # Make direction unit norm just to be safe
    refusal_dir = refusal_dir / refusal_dir.norm()

    def get_ablation_hook(direction):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
                
            # Cast direction to match hidden_states dtype exactly
            direction_cast = direction.to(hidden_states.dtype)
            
            # Project out the direction
            proj = (hidden_states * direction_cast).sum(dim=-1, keepdim=True)
            hidden_states = hidden_states - proj * direction_cast
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states
        return hook

    def generate_response(prompt):
        # Format as instruction
        chat = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, do_sample=False, temperature=0.0, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    prompt = "Solve this logic puzzle: A farmer has 10 sheep. All but 7 die. How many are left? Explain your reasoning concisely."

    print("\n" + "="*50)
    print("TEST 1: BASELINE MODEL (Normal)")
    print("="*50)
    base_ans = generate_response(prompt)
    print(base_ans)

    print("\n" + "="*50)
    print("TEST 2: ABLATED MODEL (Refusal Direction Erased)")
    print("="*50)
    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(get_ablation_hook(refusal_dir))
    abl_ans = generate_response(prompt)
    handle.remove()
    print(abl_ans)
    
    print("\nDone.")

if __name__ == "__main__":
    main()
