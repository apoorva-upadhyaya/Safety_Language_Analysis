import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import pandas as pd
from collections import defaultdict


# -----------------------------
# CONFIG
# -----------------------------
model_name = "meta-llama/Llama-3.1-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = "../activations_all/ru"
os.makedirs(output_dir, exist_ok=True)
target_layers = list(range(32))

# -----------------------------
# LOAD MODEL AND TOKENIZER
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

num_layers = model.config.num_hidden_layers

sampled_ids = pd.read_csv("../../translate_all/data/sampled_ids.csv").squeeze().tolist()

# Load your language file and filter
data = pd.read_csv("../../translate_all/data/ru.csv", delimiter=";", na_filter=False)
data = data[data["id"].isin(sampled_ids)].reset_index(drop=True)
print("Sampled data :: ", len(data))


li_id, li_ques=[],[]
li_topic=[]

for i in range(len(data)):
    li_id.append(str(data.id.values[i]))
    li_ques.append(str(data.trans_ques.values[i]))
    li_topic.append(str(data.topic.values[i]))

print("len of li::",len(li_id),li_id[0])
results1=[]

# create per-layer subdirs
layer_dirs = {L: os.path.join(output_dir, f"layer_{L}") for L in target_layers}
for d in layer_dirs.values():
    os.makedirs(d, exist_ok=True)

# buffers per layer: each is {"prompt": {sample_id: [tensors]}, "generation": {sample_id: [tensors]}, "tokens": {sample_id: [str tokens]}}
layer_buffers = {L: {"prompt": {}, "generation": {}, "tokens": {}} for L in target_layers}

def register_residual_hook(layer, sample_id, prompt_len, layer_buffers):
    """
    Hook factory that saves last-3 prompt residuals and each generated token residual
    into layer_buffers[layer]["prompt"/"generation"] keyed by sample_id.
    IMPORTANT: the hook MUST NOT return anything (we only modify in-place / store copies).
    """
    def hook_fn(module, input, output):
        # output can be a tuple; first element is the tensor we care about
        out = output[0] if isinstance(output, tuple) else output      # shape: [batch, seq_len, hidden_dim]
        out = out.detach().squeeze(0)                                # -> [seq_len, hidden_dim]

        # Prompt pass: when the layer sees the full prompt (seq_len == prompt_len)
        if out.shape[0] == prompt_len:
            # store last-3 prompt residuals (copy to CPU half to reduce memory)
            last3 = [out[-1].cpu().half(), out[-2].cpu().half(), out[-3].cpu().half()]
            layer_buffers[layer]["prompt"][sample_id] = last3

        # Generation pass: model emits one token at a time => seq_len == 1
        elif out.shape[0] == 1:
            # save the single-token residual vector
            layer_buffers[layer]["generation"].setdefault(sample_id, []).append(out[0].cpu().half())

        # DO NOT return anything — leave the tuple structure untouched

    return hook_fn



# --- Main loop: iterate samples, register hooks for all layers, generate once per sample ---
count=0
for i in range(len(li_id)):
    id_=li_id[i]
    prompt_text=li_ques[i]
    topic=li_topic[i]
    #
    if id_!="":
        print(f"\n=== Sample {id_} ===")
        messages = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        # Register hooks for all target layers for this sample
        handles = []
        for L in target_layers:
            h = model.model.layers[L].register_forward_hook(
                register_residual_hook(L, id_, prompt_len, layer_buffers)
            )
            handles.append(h)

        # Single generation call (hooks fire for each layer during generation)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.7
            )

        # decode generated text and tokens (generated only once)
        gen_token_ids = output[0][inputs["input_ids"].shape[1]:]
        decoded_output = tokenizer.decode(gen_token_ids, skip_special_tokens=True).strip()
        gen_tokens = [tokenizer.convert_ids_to_tokens(int(t)) for t in gen_token_ids]
        print("decoded_output::", decoded_output)

        # remove all hooks for this sample
        for h in handles:
            h.remove()

        results1.append({
            "id": id_,
            "question":prompt_text,
            "topic":topic,
            "prediction": decoded_output
        })
        df1 = pd.DataFrame(results1)
        df1.to_csv("../predictions_all/ru.csv", index=True, sep=";")


# --- After processing all samples: persist per-layer buffers to disk ---
for L in target_layers:
    ld = layer_dirs[L]
    
    torch.save(layer_buffers[L]["prompt"], os.path.join(ld, "prompt.pt"))
    
print("Done.")

