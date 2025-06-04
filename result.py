import torch
import torch.nn as nn
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import numpy as np

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer

#####################################################################
# === SPEC NOTICE ===
# Only "load model" and "generate" function selection can be modified.
# DO NOT change PPL calculation, timing, or throughput logic.
#####################################################################

# === (Optional) Define your own custom generate function. ===
# This is useful if you want full control over KV cache and generation steps.
# You can modify this function to suit your needs.
# By default, we use model.generate() for simplicity and general use.
def generate(model, input_ids, cache, max_new_tokens, verbose=True):
    input_ids = input_ids.clone()
    temperature = 0.7  
    
    if verbose:
        print('Prefilling...')
    
    with torch.no_grad():
        # Use ExLlamaV2's forward to pass cache
        logits = model.forward(input_ids, cache)
        
        last_logits = logits[:, -1, :] / temperature
        probs = torch.nn.functional.softmax(last_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if verbose:
            print('Decoding...')
            
        # gen_ids: update input_ids
        gen_ids = next_token
        
        for _ in range(max_new_tokens - 1):
            # Use ExLlamaV2 forward and pass cache
            logits = model.forward(next_token,cache)

            last_logits = logits[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if hasattr(model, 'config') and hasattr(model.config, 'eos_token_id'):
                if next_token.item() == model.config.eos_token_id:
                    break
                
            gen_ids = torch.cat([gen_ids, next_token], dim=0)
            
        output_ids = torch.cat([input_ids, gen_ids.unsqueeze(0).squeeze(dim=2)], dim=1)
        
            
    return output_ids


class ExLlamaV2ModelWrapper:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = "cuda:0"
        self.seqlen = 2048
        
    def __call__(self, input_ids, past_key_values=None, position_ids=None, attention_mask=None, cache_position=None):
        logits = self.model.forward(input_ids)
        
        class ModelOutput:
            def __init__(self, logits):
                self.logits = logits
                
        return ModelOutput(logits)

def evaluate_ppl(model, tokenizer, device="cuda:0"):
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # ------[Change] Use ExLlamaV2Tokenizer to encode dataset--------------------
    test_enc = tokenizer.encode("\n\n".join(test_dataset["text"]), add_bos=True)
    # ---------------------------------------------------------------------------
    model.seqlen = 2048
    # ------[Change] Convert test_enc to tensor and put on GPU-------------------
    test_enc = torch.as_tensor(test_enc, dtype=torch.long, device=device)
    # ---------------------------------------------------------------------------
    
    nsamples = test_enc.numel() // model.seqlen
    nlls = []
    for i in tqdm(range(nsamples), desc="Evaluating..."):
        batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]
        
        with torch.no_grad():
            lm_logits = model(batch).logits

        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    
    return ppl.item()

def main():
    ############## Set Up ##############
    torch.manual_seed(0)
    random.seed(0)
    
    max_new_tokens = 256    # Number of new tokens to generate
    device = 'cuda:0'
    backend = 'gemlite'
    
    ### === TODO: Load your model (you may change this part) ===
    
    # Load ExLlama Q4 model
    model_path = "./llama3.0bpw"
    print(f"Loading model from: {model_path}")
    
    config = ExLlamaV2Config(model_path)
    model = ExLlamaV2(config)
    
    # Use ExLlama customized tokenizer and wrapper model for evaluation
    tokenizer = ExLlamaV2Tokenizer(config)
    wrapped_model = ExLlamaV2ModelWrapper(model, config)
    
    # Use ExLlamaV2 customized cache
    cache = ExLlamaV2Cache(model, max_seq_len=config.max_input_len, lazy=True)
    print("Loading model...")
    model.load_autosplit(cache, progress=True)
    
    # Put model weights on GPU
    for module in model.modules:
        if hasattr(module, 'embedding') and module.embedding is not None:
            module.embedding = module.embedding.to('cuda:0')
    ##########################################################
    
    # === (Change) Command these two lines since we use customized tokenizer ----------
    # model.eval()
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # === (Optional) Uncomment the following lines if using the custom generate() function. ===
    # model.prefill_forward = model.forward
    # model.forward = torch.compile(model.forward, mode='max-autotune', dynamic=False, fullgraph=True)

    warmup_prompt = "Explain what AI is."
    
    # === (Change) Use ExLlama customized tokenizer ===
    input_ids = tokenizer.encode(warmup_prompt, add_bos=True)
    input_ids = torch.as_tensor(input_ids, dtype=torch.long, device='cuda:0')
    
    # inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    # input_ids = inputs["input_ids"]
    # attention_mask = inputs["attention_mask"]
    
    # === (Optional) Set up StaticCache for manual KV cache management ===
    # from transformers import StaticCache
    # past_key_values = StaticCache(
    #     config=model.config, 
    #     max_batch_size=1, 
    #     max_cache_len=max_new_tokens + 16, 
    #     device=model.device, 
    #     dtype=torch.float16
    # )
    ####################################################################
    
    for i in tqdm(range(5), desc="Warm Up..."):
        #  === Default: use model.generate() for end-to-end warm-up === 
        # _ = model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     pad_token_id=tokenizer.eos_token_id,
        # )
        
        # === (Optional) Use custom generate() if uncommented ===
        generated = generate(model, input_ids, cache, max_new_tokens, verbose=False)
        cache.current_seq_len = 0
        
    prompt = "How to learn a new language?"
    
    # === (Change) Use ExLlama customized tokenizer ===
    input_ids = tokenizer.encode(prompt, add_bos=True)
    input_ids = torch.as_tensor(input_ids, dtype=torch.long, device='cuda:0')
    
    # inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # input_ids = inputs["input_ids"]
    # attention_mask = inputs["attention_mask"]
    
    tputs = []
    time_record = []
    for _ in tqdm(range(10), desc="Test Inference"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # === Default: Use model.generate() for end-to-end timing === 
        # generated = model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     pad_token_id=tokenizer.eos_token_id,
        # )
        
        # === Optional: Use custom generate() if uncommented ===
        generated = generate(model, input_ids, cache, max_new_tokens, verbose=False)
        cache.current_seq_len = 0

        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        tput = generated[0][input_ids.shape[1]:].shape[0]/(elapsed_ms / 1000)
        time_record.append(elapsed_ms / 1000)
        tputs.append(tput)
    
    # === (Change) Use ExLlama tokenizer decoder
    response = tokenizer.decode(generated[0][input_ids.shape[1]:])
    sorted_tputs = np.sort(tputs)[2:-2]
    org_tput = np.mean(sorted_tputs)
    print(f'Prompt: {prompt}\nResponse: {response}\n')
    
    print(f'Time Record: {time_record}')
    print(f'Throughput Record: {tputs} toks/s\n')

    ### Your final throughput result ###
    print(f'Throughput: {org_tput} toks/s')
    # === (Change) Use wrapper model for evaluation
    ppl = evaluate_ppl(wrapped_model, tokenizer, device)
    print(f"Perplexity (PPL): {ppl}")
    
    # Save results to CSV
    import csv
    rounded_tput = round(org_tput, 1)
    ppl = round(ppl, 2)
    
    with open("19.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "value"])
        writer.writerow([0, ppl])
        writer.writerow([1, rounded_tput])
        
if __name__ == '__main__':
    main()
