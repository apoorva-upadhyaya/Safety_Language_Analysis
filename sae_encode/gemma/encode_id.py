import os
import numpy as np
import torch
import torch.nn as nn


# Example paths (adjust these to your setup)
activations_dir = "../activations_all/id/"
sae_local_path="../../../../../sae_training/gemma/local_gemma_base/res/"
encoded_dir = "../sae_encoded_all/id/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"

target_layers =list(range(42))

# === JumpReLU SAE model ===
class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts):
        pre = input_acts @ self.W_enc + self.b_enc
        mask = (pre > self.threshold)
        return mask * torch.nn.functional.relu(pre)


# === Load SAE for type + layer ===
def load_sae(layer_idx):
    npz_path = f"{sae_local_path}/layer_{layer_idx}/params.npz"
    params = np.load(npz_path)
    #print(params["W_enc"].shape[0])
    #print(params["W_enc"].shape[1])
    sae = JumpReLUSAE(params["W_enc"].shape[0], params["W_enc"].shape[1])
    sae.load_state_dict({
        "W_enc": torch.tensor(params["W_enc"]),
        "W_dec": torch.tensor(params["W_dec"]),
        "threshold": torch.tensor(params["threshold"]),
        "b_enc": torch.tensor(params["b_enc"]),
        "b_dec": torch.tensor(params["b_dec"]),
    })
    return sae.to(DEVICE).eval()



# loop through layers
for layer in target_layers:
    print(f"=== Processing Layer {layer} ===")

    # load saved prompt activations
    prompt_path = os.path.join(activations_dir, f"layer_{layer}", "prompt.pt")
    prompt_acts = torch.load(prompt_path)  
    #print("prompt_acts:",prompt_acts.keys())
    # shape: [num_samples][last3] where each last3 is tensor [seq_len, hidden_dim]

    sae = load_sae(layer)

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
