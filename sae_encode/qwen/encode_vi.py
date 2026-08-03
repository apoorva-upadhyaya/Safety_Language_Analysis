import os
import numpy as np
import torch
import torch.nn as nn


# Example paths (adjust these to your setup)
activations_dir = "../activations_all/vi/"
sae_local_path="../../../../../sae_training/saes_qwen2.5_7b_instruct/"
encoded_dir = "../sae_encoded_all/vi/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"

target_layers =list(range(28))  # Layers 1 to 31 inclusive

class TopKSAE(nn.Module):
    def __init__(self, d_model, d_sae, k):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.k = k

    def encode(self, x):
        # Linear projection
        pre = x @ self.W_enc + self.b_enc
        
        # Top-k selection per sample
        topk_values, topk_indices = torch.topk(pre, self.k, dim=-1)
        
        # Build sparse activations
        mask = torch.zeros_like(pre)
        mask.scatter_(-1, topk_indices, 1.0)
        
        # Only keep top-k preactivations (ReLU optional if negatives not desired)
        return mask * torch.relu(pre)
    
def load_topk_sae(layer_idx, k, device="cuda"):
    path = os.path.join(sae_local_path, f"layer_{layer_idx}",f"k_64",f"trainer_0","ae.pt")
    params = torch.load(path, map_location=device)
    
    print("Keys in file:", list(params.keys()))
    print("W_enc:", params["encoder.weight"].shape)
    print("bias:", params["encoder.bias"].shape)
    
    sae = TopKSAE(
        d_model=params["encoder.weight"].shape[1],
        d_sae=params["encoder.weight"].shape[0],
        k=k
    )
    sae.W_enc.data.copy_(params["encoder.weight"].T.to(DEVICE))
    sae.b_enc.data.copy_(params["encoder.bias"].to(DEVICE))
    sae.eval().to(device)
    return sae

# def load_sae_decoder(layer_idx):
#     path = os.path.join(sae_local_path, f"layer_{layer_idx}", "ae.pt")
#     state_dict = torch.load(path, map_location=DEVICE)
#     print("Keys in safetensors file:", list(state_dict.keys()))

#     print("W_enc.shape[0]:::",params["encoder.weight"].shape[0])
#     print("W_enc.shape[1]:::",params["encoder.weight"].shape[1])
#     print("threshold:::",params["threshold"].shape)
#     print("bias:::",params["encoder.bias"].shape)
#     sae = JumpReLUSAE(params["encoder.weight"].shape[0], params["encoder.weight"].shape[1])
#     sae.W_enc.data = torch.tensor(params["encoder.weight"])
#     sae.threshold.data = torch.tensor(params["threshold"])
#     sae.b_enc.data = torch.tensor(params["encoder.bias"])
#     sae.eval().to(DEVICE)
#     return sae



# loop through layers
for layer in target_layers:
    print(f"=== Processing Layer {layer} ===")

    # load saved prompt activations
    prompt_path = os.path.join(activations_dir, f"layer_{layer}", "prompt.pt")
    prompt_acts = torch.load(prompt_path)  
    #print("prompt_acts:",prompt_acts.keys())
    # shape: [num_samples][last3] where each last3 is tensor [seq_len, hidden_dim]

    sae = load_topk_sae(layer,64)

    encoded = {}

    for sample_id, last3 in prompt_acts.items():
        # last3 is a list of 3 tensors (seq_len x hidden_dim)
        encoded_last3 = []
        for acts in last3:
            #print("acts shape::",acts.shape)
            with torch.no_grad():
                z = sae.encode(acts.to(torch.float32).to(device))
            encoded_last3.append(z.half().cpu())  # save in half precision
        encoded[sample_id] = encoded_last3

    # save encoded results
    os.makedirs(os.path.join(encoded_dir, f"layer_{layer}"), exist_ok=True)
    out_path = os.path.join(encoded_dir, f"layer_{layer}", "prompt_encoded.pt")
    torch.save(encoded, out_path)

    print(f"Saved encoded activations for layer {layer} at {out_path}")
