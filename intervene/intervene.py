import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os
from collections import defaultdict, Counter
import torch.nn as nn
import safetensors.torch
import pandas as pd
import torch.nn.functional as F
import json
import pickle


DEVICE = "cuda"
MODEL_NAME = "google/gemma-2-9b-it"
sae_local_path = "../../../../../../sae_training/gemma/local_gemma_base/res/"
assert os.path.isdir(sae_local_path), f"SAE path not found: {os.path.abspath(sae_local_path)}"
print(f"SAE path OK: {os.path.abspath(sae_local_path)}")

sampled_ids = pd.read_csv("../../../translate_all/data/sampled_ids.csv").squeeze().tolist()

# Load your language file and filter
data = pd.read_csv("../../../translate_all/data/vi.csv", delimiter=";", na_filter=False)
data = data[data["id"].isin(sampled_ids)].reset_index(drop=True)
print("Sampled data :: ", len(data))

li_id, li_ques=[],[]
li_topic=[]

for i in range(len(data)):
    li_id.append(str(data.id.values[i]))
    li_ques.append(str(data.trans_ques.values[i]))
    li_topic.append(str(data.topic.values[i]))

print("len of li::",len(li_id),li_id[0],li_ques[0])

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
        input_acts = input_acts.to(self.W_enc.dtype)
        pre = input_acts @ self.W_enc + self.b_enc
        mask = (pre > self.threshold)
        return mask * torch.nn.functional.relu(pre)

    def decode(self, h):
        W_dec = self.W_dec.to(h.dtype)
        b_dec = self.b_dec.to(h.dtype)
        h = h.to(self.W_dec.dtype)
        return h @ W_dec + b_dec


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

with open("../../calculate_metrics/feature_analysis_v3_correct_threshold/all_results.pkl", "rb") as f:
    pkl_data = pickle.load(f)

layer_id = 19
LANG              = "vi"   # language whose features to use for intervention (TARGET)
TOP_K             = 5      # how many features to use
HARMFUL_LAYER_ID  = 19    # layer for harmful features
HARMLESS_LAYER_ID = 19    # layer for harmless features

# ─────────────────────────────────────────────────────────
#  CROSS-LINGUAL TRANSFER CONFIG
#  Change DONOR_LANG to swap the source language for transfer
#  e.g. "de" (high similarity, decoder cosine 0.419) 
#       "en" (low similarity control, decoder cosine ~0.12)
#       "id" (another high-similarity option)
# ─────────────────────────────────────────────────────────
DONOR_LANG = "zh"   # ← change this to try different donor languages

# SELECTION_MODE:
#   "mi"       — top-k from pre-computed list sorted by signed MI (primary)
#   "mean_act" — re-rank the pre-computed top-20 by mean activation, take top-k
SELECTION_MODE = "mi"

layer_data_harmful  = pkl_data[HARMFUL_LAYER_ID]
layer_data_harmless = pkl_data[HARMLESS_LAYER_ID]

def select_features(layer_data, feat_type, lang, top_k, mode, layer_id_for_csv):
    """
    feat_type: "harmful" or "harmless"
    mode:
      "mi"       — slice pre-computed list (already sorted by MI desc)
      "mean_act" — re-sort pre-computed top-20 by mean activation desc, slice top-k
    """
    feat_key = "top_harmful_feats" if feat_type == "harmful" else "top_harmless_feats"
    ids_top20 = layer_data[feat_key][lang]  # np.array, sorted by MI

    if mode == "mi":
        return ids_top20[:top_k].tolist()

    elif mode == "mean_act":
        # load mean activation from feature summary CSV
        csv_path = (f"../../calculate_metrics/feature_analysis_v3_correct_threshold/"
                    f"layer_{layer_id_for_csv}/{lang}_feature_summary.csv")
        df = pd.read_csv(csv_path)
        mean_col = "mean_harmful" if feat_type == "harmful" else "mean_harmless"
        mean_act = dict(zip(df["feature_id"], df[mean_col]))
        # re-sort the pre-computed top-20 by mean activation descending
        sorted_ids = sorted(ids_top20.tolist(), key=lambda fid: mean_act.get(fid, 0.0), reverse=True)
        return sorted_ids[:top_k]

    else:
        raise ValueError(f"Unknown SELECTION_MODE: {mode!r}. Choose 'mi' or 'mean_act'.")


# ── Target language (LANG) features ───────────────────────
harmful_ids  = select_features(layer_data_harmful,  "harmful",  LANG, TOP_K, SELECTION_MODE, HARMFUL_LAYER_ID)
harmless_ids = select_features(layer_data_harmless, "harmless", LANG, TOP_K, SELECTION_MODE, HARMLESS_LAYER_ID)

# ── Donor language (DONOR_LANG) harmless features ─────────
# These will REPLACE the target language's harmless features
donor_harmless_ids = select_features(layer_data_harmless, "harmless", DONOR_LANG, TOP_K, SELECTION_MODE, HARMLESS_LAYER_ID)
donor_harmful_ids = select_features(layer_data_harmful, "harmful", DONOR_LANG, TOP_K, SELECTION_MODE, HARMFUL_LAYER_ID)

# Donor's mean_harmless activation — used as injection magnitude
_donor_csv = (f"../../calculate_metrics/feature_analysis_v3_correct_threshold/"
              f"layer_{HARMLESS_LAYER_ID}/{DONOR_LANG}_feature_summary.csv")
_df_donor  = pd.read_csv(_donor_csv)
donor_mean_harmless_act = dict(zip(_df_donor["feature_id"], _df_donor["mean_harmless"]))
donor_mean_harmful_act = dict(zip(_df_donor["feature_id"], _df_donor["mean_harmful"]))

# max_activation is not in the pkl — load from per-layer summary CSV
_csv_path = (f"../../calculate_metrics/feature_analysis_v3_correct_threshold/"
             f"layer_{HARMFUL_LAYER_ID}/{LANG}_feature_summary.csv")
_df_summary = pd.read_csv(_csv_path)
_max_act = dict(zip(_df_summary["feature_id"], _df_summary["max_activation"]))

print(f"Target lang:  {LANG}   Donor lang: {DONOR_LANG}")
print(f"Harmful  features (layer {HARMFUL_LAYER_ID}, {LANG}, top-{TOP_K}): {harmful_ids}")
print(f"Harmless features (layer {HARMLESS_LAYER_ID}, {LANG}, top-{TOP_K}): {harmless_ids}")
print(f"Donor harmless features (layer {HARMLESS_LAYER_ID}, {DONOR_LANG}, top-{TOP_K}): {donor_harmless_ids}")
print(f"Donor mean_harmless activations: { {fid: donor_mean_harmless_act.get(fid, 0.0) for fid in donor_harmless_ids} }")
print(f"Donor mean_harmful activations: { {fid: donor_mean_harmful_act.get(fid, 0.0) for fid in donor_harmful_ids} }")
print(f"Max activations for harmful features: { {fid: _max_act.get(fid, 0.0) for fid in harmful_ids} }")

# ── Intervention dicts (same structure as original) ────────

FEATURES_TO_AMP = {
    HARMFUL_LAYER_ID: harmful_ids
}

FEATURE_MAX = {
    HARMFUL_LAYER_ID: {fid: _max_act.get(fid, 0.0) for fid in harmful_ids}
}

# Transfer-specific: donor features + their magnitudes
FEATURES_TO_MASK = {
    HARMFUL_LAYER_ID: harmful_ids    # zero VI harm features
}

FEATURES_TO_INJECT = {
    HARMFUL_LAYER_ID: donor_harmful_ids    # inject ZH harm features
}

INJECT_MAGNITUDES = {
    HARMFUL_LAYER_ID: {fid: donor_mean_harmful_act[fid] 
                       for fid in donor_harmful_ids}
    # magnitude = Chinese's mean_harmful activation
}

# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)
print("Model loaded.")

def register_intervention_hook(layer_idx, sample_id, sae, prompt_len=None,
                               features_to_mask=None, features_to_amp=None,
                               feature_max_dict=None, max_intervene_tokens=None,
                               features_to_inject=None, inject_magnitudes=None):
    state = {"generated_count": 0}
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden[0]   # [seq_len, hidden_dim]

        # ---- PROMPT PHASE ----
        if hidden.shape[0] > 1:
            # full sequence => prompt processing
            return

        # ---- GENERATION PHASE ----
        state["generated_count"] += 1
        current_token_index = state["generated_count"] - 1

        if (max_intervene_tokens is not None and 
            current_token_index >= max_intervene_tokens):
            return

        # only one token => hidden[0] is the new generated token
        token_act = hidden[0]
        sae_feat = sae.encode(token_act.unsqueeze(0)).squeeze(0)

        # MASK harmful features 
        # if features_to_amp and layer_idx in features_to_amp:
        #     for fid in features_to_amp[layer_idx]:
        #         sae_feat[fid] = 0.0

        # MASK target language harmless features (Step 1 of transfer)
        if features_to_mask and layer_idx in features_to_mask:
            for fid in features_to_mask[layer_idx]:
                sae_feat[fid] = 0.0

        # INJECT donor language harmless features (Step 2 of transfer)
        # Set donor feature IDs to their mean_harmless activation from donor language
        if features_to_inject and layer_idx in features_to_inject:
            mags = inject_magnitudes.get(layer_idx, {}) if inject_magnitudes else {}
            for fid in features_to_inject[layer_idx]:
                sae_feat[fid] = mags.get(fid, 0.0)
                print(f"Injecting feature {fid} with magnitude {sae_feat[fid]:.4f} at layer {layer_idx}")

        #Amplify harmful features — set to their observed max activation
        # if features_to_amp and layer_idx in features_to_amp:
        #     for fid in features_to_amp[layer_idx]:
        #         amp_val = feature_max_dict[layer_idx].get(fid, 0.0) if feature_max_dict else 0.0
        #         sae_feat[fid] += 0.1*amp_val

        # write back
        hidden[0] = sae.decode(sae_feat).to(hidden.dtype)

    return hook

# -------------------------
# Run intervention
# -------------------------

def run_intervention(question, sample_id, mask_features=None, amp_features=None,
                     feature_max_dict=None, features_to_inject=None, inject_magnitudes=None):
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = inputs["input_ids"].shape[1]
    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    all_layers = set()
    if mask_features:
        all_layers.update(mask_features.keys())
    if amp_features:
        all_layers.update(amp_features.keys())
    if features_to_inject:
        all_layers.update(features_to_inject.keys())

    print("mask_features::", mask_features)
    print("amp_features::", amp_features)
    print("features_to_inject::", features_to_inject)
    print("all_layers::", all_layers)

    intervention_log = []

    handles = []
    for L in all_layers:
        sae = load_sae(layer_idx=f"{L}")
        handles.append(
            model.model.layers[L].register_forward_hook(
                register_intervention_hook(
                    L,
                    sample_id,
                    sae,
                    prompt_len,
                    features_to_mask=mask_features,
                    features_to_amp=amp_features,
                    feature_max_dict=feature_max_dict,
                    max_intervene_tokens=50,
                    features_to_inject=features_to_inject,
                    inject_magnitudes=inject_magnitudes,
                )
            )
        )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=768,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7
        )

    for h in handles:
        h.remove()

    prediction = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()

    return prediction

# -------------------------
# Example usage
# -------------------------

results1=[]

for i in range(len(li_id)):
    sample_id=li_id[i]
    prompt_text=li_ques[i]

    if sample_id!="":
        print(f"====== Processing sample {sample_id} =========")
        question=prompt_text
        print("Question", question)

        print(f"\n=== Transfer {DONOR_LANG.upper()}→{LANG.upper()} harmless features ===")
        out_mask = run_intervention(
            question,
            sample_id,
            mask_features     = FEATURES_TO_MASK,    # zero target harmless
            amp_features      = FEATURES_TO_AMP,
            feature_max_dict  = FEATURE_MAX,
            features_to_inject= FEATURES_TO_INJECT,  # inject donor harmless
            inject_magnitudes = INJECT_MAGNITUDES,
        )
        print("out_mask", out_mask)

        results1.append({
            "id": sample_id,
            "question": question,
            "prediction": out_mask
        })
        df1 = pd.DataFrame(results1)
        df1.to_csv(f"results/{LANG}_{DONOR_LANG}_mi_top2_10.csv",
                   index=True, sep=";")