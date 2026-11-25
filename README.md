# When Safety Speaks a Language: A Mechanistic Analysis of Safety–Language Identity Entanglement in LLMs

Code and data for our paper. We analyse multilingual safety in LLMs with sparse autoencoder (SAE) features and ask how safety features relate to language identity.

| | |
|---|---|
| **Models** | Llama-3.1-8B-Instruct (32 layers), Qwen2.5-7B-Instruct (28 layers), Gemma-2-9B-IT (42 layers) |
| **Languages** | `en` `zh` `de` `ar` `vi` `id` `ru` `hi` |
| **Data** | TechHazardQA (Banerjee et al., 2025), machine-translated into 7 languages (7,745 prompts each) |
| **SAEs** | BatchTopK SAEs (k=64) trained by us for every layer of Llama and Qwen; Gemma Scope JumpReLU SAEs for Gemma |

## Research questions

1. **RQ1: Is safety universal or language-specific?** Safety signals concentrate in late layers in all three models. The ordering of harm and harmless features is architecture-dependent.
2. **RQ2: Are safety features entangled with language identity?** Yes in Llama and Qwen, where entanglement peaks in late layers. Gemma stays largely disentangled. Cross-lingual sharing of safety features peaks in the middle layers, and English is the most isolated language.
3. **RQ3: Does entanglement predict the cost of interventions?** Ablating, amplifying or transferring safety features changes harmful-response rate, and also language identity and fluency. The size of the side effects follows the entanglement scores.

## Pipeline

```
dataset/*.csv
    │  (1) activations/      prompt → last-token residual stream at every layer (+ generated response)
    ▼
activations_all/<lang>/layer_<L>/prompt.pt      predictions_all/<lang>.csv
    │  (2) sae_encode/       residual stream → SAE feature activations
    ▼                                                   │  (3) judge labels (harmful / harmless)
sae_encoded_all/<lang>/layer_<L>/prompt_encoded.pt ◄────┘
    │  (4) calculate_metrics/  MI feature selection, Jaccard + decoder cosine entanglement
    ▼
all_results.pkl, per-layer CSV/JSON summaries
    │  (5) intervene/        ablate / amplify / transfer top-k features while generating
    ▼
intervened responses (results/*.csv)
```

## Repository structure

```
dataset/            TechHazardQA translations: ar, de, hi, id, ru, vi, zh (+ sampled_ids.csv)
activations/        Step 1: generate responses and save residual-stream activations (one script per language)
sae_encode/         Step 2: encode activations with SAEs (gemma/, llama/, qwen/)
calculate_metrics/  Step 4: safety and language feature scoring, entanglement metrics (gemma.py, llama.py, qwen.py)
intervene/          Step 5: SAE-feature interventions (harm.py, intervene.py)
```

**Dataset format.** Each `dataset/<lang>.csv` is `;`-separated with columns `id`, `question` (original English prompt), `topic` and `trans_ques` (translation). English prompts are the `question` column of any of these files.

## Setup

Dependencies are listed in `requirements.txt` and `environment.yml` (Python 3.10).

## Running the code

> **Note on configuration.** The scripts are research code with the configuration set at the top of each file rather than through command-line flags. Before running a script, edit its config block: the model, language, layer and the paths to data, SAEs and outputs. The scripts expect to be run from inside their own folder, because the paths are relative. The paths to change are variables near the top of each file (`model_name`, `output_dir`, `sae_local_path`, `activations_dir`, `encoded_dir`, `CONFIG = dict(...)`).

### 1. Extract activations and responses

```bash
cd activations
python en.py        # also: ar.py de.py hi.py id.py ru.py vi.py zh.py
```

Each script runs the model on every prompt of one language, using the chat template. A forward hook saves the residual stream at the output of each layer. It stores the last three prompt-token positions (the analysis uses last token) as `activations_all/<lang>/layer_<L>/prompt.pt`, and the generated responses as `predictions_all/<lang>.csv`.

The scripts ship with `model_name = "meta-llama/Llama-3.1-8B-Instruct"` and `target_layers = list(range(32))`. For Qwen2.5-7B-Instruct use `"Qwen/Qwen2.5-7B-Instruct"` with 28 layers, and for Gemma-2-9B-IT use `"google/gemma-2-9b-it"` with 42 layers.

### 2. Label responses as harmful / harmless

Each response is labelled by three judge LLMs (Llama-3.3-70B-Instruct, Gemma-3-27B-it, Qwen3-30B-A3B-Instruct-2507) with a majority vote, using the prompt in Figure 3 of the paper. The judging script is not part of this release. The downstream code expects one JSON per language at `predictions_all/eval_llm/results/<lang>_pr4.json` mapping each sample id to `"yes"` (harmless) or `"no"` (harmful).

### 3. Encode activations with SAEs

```bash
cd sae_encode/llama      # or gemma/, qwen/
python encode_en.py      # also: encode_<lang>.py
```

Reads `activations_all/<lang>/layer_<L>/prompt.pt` and writes `sae_encoded_all/<lang>/layer_<L>/prompt_encoded.pt`.

| Model | SAE | Where to get it |
|---|---|---|
| Gemma-2-9B-IT | Gemma Scope JumpReLU, 16,384 features, residual stream (`params.npz` per layer) | [`google/gemma-scope-9b-pt-res`](https://huggingface.co/google/gemma-scope-9b-pt-res) |
| Llama-3.1-8B-Instruct | BatchTopK, k=64, 8× expansion, `layer_<L>/k_64/trainer_0/ae.pt` | trained by us (see below) |
| Qwen2.5-7B-Instruct | BatchTopK, k=64, 32× expansion, `layer_<L>/k_64/trainer_0/ae.pt` | trained by us (see below) |

Set `sae_local_path` in each script to the folder containing the SAE checkpoints.

**SAE weights.** Our pre-trained Llama and Qwen SAEs (all layers) are available upon request if required. To train your own, use [`saprmarks/dictionary_learning`](https://github.com/saprmarks/dictionary_learning) with the configuration in Appendix B of the paper (BatchTopK, resid_post, k=64, 500M tokens, 35% LMSYS-Chat-1M / 64% Pile-uncopyrighted / 1% emergent-misalignment, following Arditi 2024).

### 4. Feature selection and entanglement metrics

```bash
cd calculate_metrics
python llama.py          # or qwen.py / gemma.py
```

For every language and layer, the script:

- scores each SAE feature by mutual information with the harmful/harmless label (signed by direction) and with language identity,
- takes the top-k (k=5) harm, harmless and language features,
- computes **Jaccard overlap** and **decoder cosine similarity**, within a language (safety vs. language identity) and across all language pairs (safety vs. safety),
- writes per-layer CSV/JSON summaries and a combined `all_results.pkl` into `output_dir` (default `feature_analysis_v3_correct_threshold/`).

Set `encoded_dir`, `label_dir`, `csv_dir` and `sae_decoder_path` in `CONFIG` first. Phase-level aggregation (early / middle / late thirds) and z-scores are described in the paper and are computed from these per-layer outputs.

### 5. Interventions

```bash
cd intervene
python harm.py           # harm-feature intervention, configured for Hindi
python intervene.py      # harmless/harm feature intervention and cross-lingual transfer, configured for VI→ZH
```

Both scripts read `all_results.pkl` from step 4 and edit the SAE latent of the chosen layer on the first generated tokens. Settings are in the config block near the top:

| Variable | Meaning |
|---|---|
| `LANG` | language whose features are intervened on (target) |
| `DONOR_LANG` | language whose features are injected (`intervene.py`, cross-lingual transfer) |
| `HARMFUL_LAYER_ID`, `HARMLESS_LAYER_ID` | layer for harm / harmless features |
| `TOP_K` | number of features (paper: 5) |
| `SELECTION_MODE` | `"mi"` (default, signed MI) or `"mean_act"` |

Operations: **mask** (set features to 0), **amplify** (add the feature's maximum observed activation) and **inject** (add donor-language features). The modified latent is decoded by the SAE and replaces the residual activation during generation. The scripts as shipped are set to one configuration each. Change the variables above, and which of the mask / amplify / inject feature sets is non-empty, to reproduce the rows of Table 2 (the layers are listed there). Outputs are written to `results/`.

### 6. Evaluating the interventions

The evaluation uses the same LLM judges as in step 2 (harmful rate and 4-level fluency, prompt in Figure 6 of the paper), plus line-level language identification with [fastText `lid.176`](https://fasttext.cc/docs/en/language-identification.html) (lines with confidence < 0.5 count as unknown).

## Known limitations of this release

- Configuration lives in the scripts (no CLI). See the note above step 1.
- The judge-LLM labelling and fluency-evaluation scripts, and the plotting scripts used for the paper figures, are not included yet.
- The relative paths assume the folder layout shown in the pipeline diagram. Create `activations_all/`, `predictions_all/`, `sae_encoded_all/` and `results/` next to the script folders, or change the paths.

## Responsible use

The dataset contains potentially harmful or unethical questions and is released for safety research only. Interventions that ablate safety features can make a model produce harmful content. Do not deploy intervened models.

## Citation
The paper is accepted for EACL Main 2027. The citation will be added after the conference proceedings.
