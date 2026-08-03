"""
Safety + Language Feature Extraction  (v3)
===========================================
THREE-AXIS FRAMEWORK:
  Axis 1: Harmful Safety Score   — MI(feature, harm=1),    signed + toward harmful
  Axis 2: Harmless Safety Score  — MI(feature, harmless=1), signed + toward harmless
  Axis 3: Language Score         — MI(feature, is_lang_ℓ),  signed + toward this language

  NOTE: Axis 1 and Axis 2 are NOT simply negatives of each other.
  MI magnitude differs because of class imbalance and non-linear relationships.
  A feature can have HIGH MI with both classes if it fires in both but at
  different magnitudes — these are "conflict features".

8-WAY CLASSIFICATION (3 binary axes → 2³ = 8 cells):
  harm_high + harml_low  + lang_low  → "pure_harm_trigger"
  harm_low  + harml_high + lang_low  → "pure_safety_brake"
  harm_high + harml_low  + lang_high → "lang_entangled_harm"
  harm_low  + harml_high + lang_high → "lang_entangled_brake"
  harm_high + harml_high + lang_low  → "conflict_universal"
  harm_high + harml_high + lang_high → "conflict_entangled"
  harm_low  + harml_low  + lang_high → "language_only"
  harm_low  + harml_low  + lang_low  → "noise"

GEOMETRIC COMPARISONS:
  - Decoder direction cosine  : uses W_dec from pre-trained SAE (fixed directions)
  - Activation profile cosine : uses actual activation vectors across prompts
  Both are computed for entanglement and cross-lingual similarity.

Data structure (unchanged from v2):
  encoded[sample_id] = [z0, z1, z2]
  each zi : Tensor[seq_len, sae_dim] half precision
  labels  : {sample_id: "yes"(harmless) | "no"(harmful)}
"""

import os, json, pickle, warnings
import numpy as np
import torch
import pandas as pd
from collections import Counter
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────
CONFIG = dict(
    languages     = ["ar","de", "en", "hi", "id", "ru", "vi", "zh"],
    target_layers = list(range(28)),
    top_k         = 5,

    # Which of the 3 saved token positions to use
    # 0=third-last, 1=second-last, 2=last
    TOKEN_IDX     = 1,

    encoded_dir      = "../sae_encoded_all",
    label_dir        = "../predictions_all/eval_llm/results",
    csv_dir          = "../predictions_all/",
    output_dir       = "feature_analysis_v3_correct_threshold",

    # SAE decoder weights — ae.pt files from andyrdt HuggingFace repo
    # Use {layer} as placeholder
    sae_decoder_path = "../../../../../sae_training/saes_qwen2.5_7b_instruct/layer_{layer}/k_64/trainer_0/ae.pt",

    universal_min_coverage = 0.75,
)

os.makedirs(CONFIG["output_dir"], exist_ok=True)
LANGS = CONFIG["languages"]
TOP_K = CONFIG["top_k"]


# ═══════════════════════════════════════════════════════════════
#  PART 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_lang_layer_data(lang: str, layer: int) -> dict | None:
    act_path   = os.path.join(CONFIG["encoded_dir"], lang,
                              f"layer_{layer}", "prompt_encoded.pt")
    label_path = os.path.join(CONFIG["label_dir"], f"{lang}_pr4.json")
    csv_path   = os.path.join(CONFIG["csv_dir"],   f"{lang}.csv")

    missing = [p for p in [act_path, label_path, csv_path] if not os.path.exists(p)]
    if missing:
        print(f"  [skip {lang}] missing: {missing}")
        return None

    with open(label_path) as f:
        labels_raw = json.load(f)

    df_csv = pd.read_csv(csv_path, sep=";")
    df_csv["id"] = df_csv["id"].astype(str)

    encoded = torch.load(act_path, map_location="cpu")
    tok_idx = CONFIG["TOKEN_IDX"]

    feats_list, labels_list, ids_list = [], [], []

    for _, row in df_csv.iterrows():
        sid = str(row["id"])
        if sid not in labels_raw or sid not in encoded:
            continue

        label_str = labels_raw[sid].strip().lower()
        if label_str.startswith("no"):
            label = 1
        elif label_str.startswith("yes"):
            label = 0
        else:
            continue

        z_tensor = encoded[sid][tok_idx]
        feat = z_tensor.float() if z_tensor.dim() == 1 else z_tensor[-1].float()

        feats_list.append(feat)
        labels_list.append(label)
        ids_list.append(sid)

    if not feats_list:
        print(f"  [skip {lang}] no valid samples after label matching")
        return None

    feats  = torch.stack(feats_list)
    labels = np.array(labels_list)

    n_harm = (labels == 1).sum()
    n_safe = (labels == 0).sum()
    print(f"  [{lang}] N={len(feats)}  harmful={n_harm}  harmless={n_safe}  "
          f"sae_dim={feats.shape[1]}")

    if n_harm == 0 or n_safe == 0:
        print(f"  [warn {lang}] only one class — safety MI will be zero")

    return {"feats": feats, "labels": labels, "ids": ids_list}


# ═══════════════════════════════════════════════════════════════
#  PART 2 — MUTUAL INFORMATION CORE
# ═══════════════════════════════════════════════════════════════

def _compute_signed_mi(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    active = (X > 0).any(axis=0)
    mi     = np.zeros(X.shape[1], dtype=np.float32)

    if active.sum() > 0:
        mi[active] = mutual_info_classif(
            X[:, active], y,
            discrete_features=False,
            random_state=42
        ).astype(np.float32)

    m1 = X[y == 1].mean(axis=0) if (y == 1).any() else np.zeros(X.shape[1])
    m0 = X[y == 0].mean(axis=0) if (y == 0).any() else np.zeros(X.shape[1])
    return mi * np.sign(m1 - m0)


def _compute_multiclass_mi(X: np.ndarray, y_int: np.ndarray) -> np.ndarray:
    active = (X > 0).any(axis=0)
    mi     = np.zeros(X.shape[1], dtype=np.float32)
    if active.sum() > 0:
        mi[active] = mutual_info_classif(
            X[:, active], y_int,
            discrete_features=False,
            random_state=42
        ).astype(np.float32)
    return mi


# ═══════════════════════════════════════════════════════════════
#  PART 3 — THREE SAFETY SCORES PER LANGUAGE
# ═══════════════════════════════════════════════════════════════

def compute_safety_scores(feats: torch.Tensor, labels: np.ndarray) -> dict:
    X      = feats.cpu().numpy().astype(np.float32)
    labels = labels.astype(int)

    harm_smi  = _compute_signed_mi(X, labels)
    harml_smi = _compute_signed_mi(X, 1 - labels)

    mask1 = labels == 1
    mask0 = labels == 0
    mean_harm  = X[mask1].mean(axis=0) if mask1.any() else np.zeros(X.shape[1])
    mean_harml = X[mask0].mean(axis=0) if mask0.any() else np.zeros(X.shape[1])

    harm_cands = np.argsort(harm_smi)[::-1][:TOP_K * 5]
    harm_cands = sorted(harm_cands,
                        key=lambda i: (harm_smi[i], mean_harm[i]),
                        reverse=True)[:TOP_K]

    harml_cands = np.argsort(harml_smi)[::-1][:TOP_K * 5]
    harml_cands = sorted(harml_cands,
                         key=lambda i: (harml_smi[i], mean_harml[i]),
                         reverse=True)[:TOP_K]

    return {
        "harm_smi"    : harm_smi,
        "harml_smi"   : harml_smi,
        "top_harmful" : np.array(harm_cands),
        "top_harmless": np.array(harml_cands),
        "mean_harm"   : mean_harm,
        "mean_harml"  : mean_harml,
    }


# ═══════════════════════════════════════════════════════════════
#  PART 4 — LANGUAGE SCORE (one-vs-rest)
# ═══════════════════════════════════════════════════════════════

def compute_language_scores(all_feats: dict) -> tuple:
    lang_list  = list(all_feats.keys())
    chunks, idx = [], []
    for li, lang in enumerate(lang_list):
        arr = all_feats[lang].cpu().numpy().astype(np.float32)
        chunks.append(arr)
        idx.append(np.full(len(arr), li, dtype=int))

    X_all  = np.concatenate(chunks, axis=0)
    y_lang = np.concatenate(idx,    axis=0)

    mc_mi = _compute_multiclass_mi(X_all, y_lang)

    lang_scores    = {}
    top_lang_feats = {}

    for li, lang in enumerate(lang_list):
        y_ovr      = (y_lang == li).astype(int)
        mean_this  = X_all[y_ovr == 1].mean(axis=0)
        mean_other = X_all[y_ovr == 0].mean(axis=0)
        direction  = np.sign(mean_this - mean_other)

        smi_lang             = mc_mi * direction
        lang_scores[lang]    = smi_lang
        top_lang_feats[lang] = np.argsort(smi_lang)[::-1][:TOP_K]

    return lang_scores, top_lang_feats, mc_mi


# ═══════════════════════════════════════════════════════════════
#  PART 5 — 3-AXIS 8-WAY CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

def classify_3axis(harm_s: float, harml_s: float, lang_s: float,
                   harm_thresh: float, harml_thresh: float,
                   lang_thresh: float) -> str:
    hh = harm_s  > harm_thresh
    hl = harml_s > harml_thresh
    lg = lang_s  > lang_thresh

    if hh and not hl and not lg: return "pure_harm_trigger"
    if not hh and hl and not lg: return "pure_safety_brake"
    if hh and not hl and lg:     return "lang_entangled_harm"
    if not hh and hl and lg:     return "lang_entangled_brake"
    if hh and hl and not lg:     return "conflict_universal"
    if hh and hl and lg:         return "conflict_entangled"
    if not hh and not hl and lg: return "language_only"
    return "noise"


# ═══════════════════════════════════════════════════════════════
#  PART 6 — ACTIVATION PROFILE SIMILARITY
# ═══════════════════════════════════════════════════════════════

def compute_activation_profile_cosine(
    feats_A: np.ndarray,
    feats_B: np.ndarray,
    top_feats_A: np.ndarray,
    top_feats_B: np.ndarray,
) -> float:
    prof_A = feats_A[:, top_feats_A].mean(axis=0)
    prof_B = feats_B[:, top_feats_B].mean(axis=0)

    norm_A = np.linalg.norm(prof_A) + 1e-8
    norm_B = np.linalg.norm(prof_B) + 1e-8
    cosine_overall = float(np.dot(prof_A / norm_A, prof_B / norm_B))

    def feature_descriptor(feats_matrix, feat_ids):
        cols = feats_matrix[:, feat_ids]
        mean = cols.mean(axis=0)
        std  = cols.std(axis=0)
        mx   = cols.max(axis=0)
        freq = (cols > 0).mean(axis=0)
        return np.stack([mean, std, mx, freq], axis=1)

    desc_A = feature_descriptor(feats_A, top_feats_A)
    desc_B = feature_descriptor(feats_B, top_feats_B)

    desc_A = desc_A / (np.linalg.norm(desc_A, axis=1, keepdims=True) + 1e-8)
    desc_B = desc_B / (np.linalg.norm(desc_B, axis=1, keepdims=True) + 1e-8)

    cosines    = desc_A @ desc_B.T
    nn_cosines = cosines.max(axis=1)
    return float(nn_cosines.mean())


# ═══════════════════════════════════════════════════════════════
#  PART 7 — DECODER DIRECTION SIMILARITY
# ═══════════════════════════════════════════════════════════════

def load_decoder_weights(layer: int) -> torch.Tensor | None:
    template = CONFIG["sae_decoder_path"]
    if template is None:
        return None

    path = template.format(layer=layer)
    if not os.path.exists(path):
        print(f"  [decoder] not found: {path}")
        return None

    state = torch.load(path, map_location="cpu")
    W = None

    if isinstance(state, dict):
        for key in ["decoder.weight", "W_dec", "weight"]:
            if key in state:
                W = state[key].float()
                print(f"  [decoder] key='{key}' raw shape={W.shape}")
                break
        if W is None:
            print(f"  [decoder] available keys: {list(state.keys())}")
            return None
    elif isinstance(state, torch.Tensor):
        W = state.float()

    if W.shape[0] < W.shape[1]:
        W = W.T

    print(f"  [decoder] layer {layer}: W_dec shape={W.shape}  "
          f"(each row = one feature direction in residual stream)")
    return W


def decoder_cosine_similarity(
    feats_ids_A: np.ndarray,
    feats_ids_B: np.ndarray,
    W_dec: torch.Tensor,
) -> float:
    W = W_dec.numpy()

    va = W[feats_ids_A]
    vb = W[feats_ids_B]

    va = va / (np.linalg.norm(va, axis=1, keepdims=True) + 1e-8)
    vb = vb / (np.linalg.norm(vb, axis=1, keepdims=True) + 1e-8)

    cosines    = va @ vb.T
    nn_cosines = cosines.max(axis=1)
    return float(nn_cosines.mean())


# ═══════════════════════════════════════════════════════════════
#  PART 8 — ENTANGLEMENT
# ═══════════════════════════════════════════════════════════════

def compute_entanglement(
    top_harmful_feats:  dict,
    top_harmless_feats: dict,
    top_lang_feats:     dict,
    all_feats:          dict,
    decoder_W=None,
) -> dict:
    results = {}

    for lang in top_harmful_feats:
        sf_harm  = top_harmful_feats[lang]
        sf_harml = top_harmless_feats[lang]
        lf       = top_lang_feats.get(lang, np.array([]))

        sf_harm_set  = set(sf_harm.tolist())
        sf_harml_set = set(sf_harml.tolist())
        lf_set       = set(lf.tolist())

        def jaccard(set_a, set_b):
            inter = set_a & set_b
            union = set_a | set_b
            return len(inter) / len(union) if union else 0.0

        jac_harm  = jaccard(sf_harm_set,  lf_set)
        jac_harml = jaccard(sf_harml_set, lf_set)

        dec_harm  = None
        dec_harml = None
        if decoder_W is not None and len(sf_harm) > 0 and len(lf) > 0:
            dec_harm  = decoder_cosine_similarity(sf_harm,  lf, decoder_W)
        if decoder_W is not None and len(sf_harml) > 0 and len(lf) > 0:
            dec_harml = decoder_cosine_similarity(sf_harml, lf, decoder_W)

        X = all_feats[lang].cpu().numpy().astype(np.float32)
        act_harm  = None
        act_harml = None
        if len(sf_harm) > 0 and len(lf) > 0:
            act_harm  = compute_activation_profile_cosine(X, X, sf_harm,  lf)
        if len(sf_harml) > 0 and len(lf) > 0:
            act_harml = compute_activation_profile_cosine(X, X, sf_harml, lf)

        results[lang] = {
            "jaccard_harm_lang"     : jac_harm,
            "decoder_harm_lang"     : dec_harm,
            "activation_harm_lang"  : act_harm,
            "jaccard_harml_lang"    : jac_harml,
            "decoder_harml_lang"    : dec_harml,
            "activation_harml_lang" : act_harml,
        }

    return results


# ═══════════════════════════════════════════════════════════════
#  PART 9 — CROSS-LINGUAL SIMILARITY MATRICES
# ═══════════════════════════════════════════════════════════════

def compute_crosslingual_matrices(
    top_harmful_feats:  dict,
    top_harmless_feats: dict,
    all_feats:          dict,
    decoder_W=None,
) -> dict:
    langs = list(top_harmful_feats.keys())
    n     = len(langs)

    dec_harm_mat    = pd.DataFrame(np.eye(n), index=langs, columns=langs)
    dec_harml_mat   = pd.DataFrame(np.eye(n), index=langs, columns=langs)
    act_harm_mat    = pd.DataFrame(np.eye(n), index=langs, columns=langs)
    act_harml_mat   = pd.DataFrame(np.eye(n), index=langs, columns=langs)

    for la in langs:
        for lb in langs:
            if la == lb:
                continue

            fa_harm  = top_harmful_feats[la]
            fb_harm  = top_harmful_feats[lb]
            fa_harml = top_harmless_feats[la]
            fb_harml = top_harmless_feats[lb]

            Xa = all_feats[la].cpu().numpy().astype(np.float32)
            Xb = all_feats[lb].cpu().numpy().astype(np.float32)

            if decoder_W is not None:
                if len(fa_harm) > 0 and len(fb_harm) > 0:
                    dec_harm_mat.loc[la, lb] = decoder_cosine_similarity(
                        fa_harm, fb_harm, decoder_W)
                if len(fa_harml) > 0 and len(fb_harml) > 0:
                    dec_harml_mat.loc[la, lb] = decoder_cosine_similarity(
                        fa_harml, fb_harml, decoder_W)

            if len(fa_harm) > 0 and len(fb_harm) > 0:
                act_harm_mat.loc[la, lb] = compute_activation_profile_cosine(
                    Xa, Xb, fa_harm, fb_harm)
            if len(fa_harml) > 0 and len(fb_harml) > 0:
                act_harml_mat.loc[la, lb] = compute_activation_profile_cosine(
                    Xa, Xb, fa_harml, fb_harml)

    return {
        "decoder_harm"    : dec_harm_mat,
        "decoder_harml"   : dec_harml_mat,
        "activation_harm" : act_harm_mat,
        "activation_harml": act_harml_mat,
    }


# ═══════════════════════════════════════════════════════════════
#  PART 10 — UNIVERSAL SAFETY FEATURES
# ═══════════════════════════════════════════════════════════════

def find_universal_features(
    top_feats: dict,
    min_coverage: float = 0.75,
    label: str = "",
) -> list:
    n_langs = len(top_feats)
    counts  = Counter()
    for feats in top_feats.values():
        counts.update(feats.tolist())

    universal = [
        (fid, cnt / n_langs)
        for fid, cnt in counts.items()
        if cnt / n_langs >= min_coverage
    ]
    universal.sort(key=lambda x: x[1], reverse=True)

    if label:
        print(f"  Universal {label} features (≥{min_coverage*100:.0f}%): {len(universal)}")
        for fid, cov in universal[:10]:
            print(f"    feature {fid:6d}  coverage={cov:.2f}")

    return universal


# ═══════════════════════════════════════════════════════════════
#  PART 11 — FEATURE SUMMARY DATAFRAME (per language)
# ═══════════════════════════════════════════════════════════════

def build_summary_df(
    lang:           str,
    feats:          torch.Tensor,
    labels:         np.ndarray,
    harm_smi:       np.ndarray,
    harml_smi:      np.ndarray,
    lang_score:     np.ndarray,
    top_harmful:    np.ndarray,
    top_harmless:   np.ndarray,
    top_lang:       np.ndarray,
    mean_harm:      np.ndarray,
    mean_harml:     np.ndarray,
) -> pd.DataFrame:
    """
    Build one row per feature (union of top_harmful ∪ top_harmless ∪ top_lang).
    Includes all three axis scores, activation stats, and 8-way classification.

    THRESHOLD STRATEGY — FIXED VERSION:
      We use the MINIMUM MI score among the top-k features for each axis
      as the threshold. This means:
        - A feature is "harm_high"  iff its harm_smi  >= worst of top-20 harmful
        - A feature is "harml_high" iff its harml_smi >= worst of top-20 harmless
        - A feature is "lang_high"  iff its lang_score >= worst of top-20 language

      This makes the classification directly reflect set membership:
        - Features from top_harmful  will be harm_high=YES  by definition
        - Features from top_harmless will be harml_high=YES by definition
        - Features from top_lang     will be lang_high=YES  by definition
        - A feature in BOTH top_harmful and top_lang → lang_entangled_harm
        - A feature in top_harmful but NOT top_lang  → pure_harm_trigger

      This is more semantically meaningful than the global median threshold,
      which set the bar so low that almost every candidate feature
      scored "high" on every axis, making the classification noisy.

    PREVIOUS THRESHOLD (commented out — global median approach):
      Used median of ALL positive MI values across all 131072 features.
      Problem: SAE features are very sparse and most have near-zero MI,
      so the median was tiny (≈0.001). Almost every feature in our
      top-60 candidates scored above this tiny threshold, making nearly
      everything classify as "high" on every axis.
      Result: very few "noise" features, too many "conflict" features.
    """
    X = feats.cpu().numpy().astype(np.float32)

    # ── FIXED THRESHOLD: minimum MI score among top-k features ────────────────
    #
    # For each axis, the threshold = the MI score of the WORST (20th best) feature
    # in that axis's top-k list. A feature is "high" on an axis only if it is at
    # least as informative as the 20th best feature for that axis.
    #
    # Fallback to 0.0 if top-k list is empty (shouldn't happen in practice).

    harm_t  = float(harm_smi[top_harmful].min())   if len(top_harmful)  > 0 else 0.0
    harml_t = float(harml_smi[top_harmless].min()) if len(top_harmless) > 0 else 0.0
    lang_t  = float(lang_score[top_lang].min())    if len(top_lang)     > 0 else 0.0

    # ── PREVIOUS THRESHOLD (global median — commented out) ───────────────────
    #
    # def thresh(arr):
    #     """Median of ALL positive values across the full 131072-feature space.
    #     Problem: most SAE features have near-zero MI, so this median is tiny,
    #     causing almost every candidate feature to be classified as 'high'."""
    #     pos = arr[arr > 0]
    #     return float(np.median(pos)) if len(pos) > 0 else 0.0
    #
    # harm_t  = thresh(harm_smi)
    # harml_t = thresh(harml_smi)
    # lang_t  = thresh(lang_score)
    # ─────────────────────────────────────────────────────────────────────────

    all_fids = sorted(
        set(top_harmful.tolist()) | set(top_harmless.tolist()) | set(top_lang.tolist())
    )

    mask1 = labels == 1
    mask0 = labels == 0

    rows = []
    for fid in all_fids:
        col = X[:, fid]

        freq_harm  = float((col[mask1] > 0).mean()) if mask1.any() else 0.0
        freq_harml = float((col[mask0] > 0).mean()) if mask0.any() else 0.0

        rows.append({
            "language"           : lang,
            "feature_id"         : fid,
            "harm_safety_score"  : float(harm_smi[fid]),
            "harml_safety_score" : float(harml_smi[fid]),
            "lang_score"         : float(lang_score[fid]),
            "mean_harmful"       : float(mean_harm[fid]),
            "mean_harmless"      : float(mean_harml[fid]),
            "max_activation"     : float(col.max()),
            "freq_harmful"       : freq_harm,
            "freq_harmless"      : freq_harml,
            "in_top_harmful"     : fid in set(top_harmful.tolist()),
            "in_top_harmless"    : fid in set(top_harmless.tolist()),
            "in_top_lang"        : fid in set(top_lang.tolist()),
            "category"           : classify_3axis(
                                       float(harm_smi[fid]),
                                       float(harml_smi[fid]),
                                       float(lang_score[fid]),
                                       harm_t, harml_t, lang_t
                                   ),
        })

    return pd.DataFrame(rows).sort_values("harm_safety_score", ascending=False)


# ═══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def run():
    all_results = {}

    for layer in CONFIG["target_layers"]:
        print(f"\n{'═'*65}")
        print(f"  LAYER {layer}")
        print(f"{'═'*65}")

        all_feats  = {}
        all_labels = {}

        for lang in LANGS:
            result = load_lang_layer_data(lang, layer)
            print(f"  Loaded {lang} layer {layer}", result["feats"].shape if result else "N/A")
            if result is None:
                continue
            all_feats[lang]  = result["feats"]
            all_labels[lang] = result["labels"]

        if len(all_feats) < 2:
            print("  Not enough languages, skipping.")
            continue

        # Safety scores — Axis 1 + Axis 2
        harm_smi_all     = {}
        harml_smi_all    = {}
        top_harmful_all  = {}
        top_harmless_all = {}
        mean_harm_all    = {}
        mean_harml_all   = {}

        for lang in all_feats:
            sc = compute_safety_scores(all_feats[lang], all_labels[lang])
            harm_smi_all[lang]     = sc["harm_smi"]
            harml_smi_all[lang]    = sc["harml_smi"]
            top_harmful_all[lang]  = sc["top_harmful"]
            top_harmless_all[lang] = sc["top_harmless"]
            mean_harm_all[lang]    = sc["mean_harm"]
            mean_harml_all[lang]   = sc["mean_harml"]

        # Language scores — Axis 3
        lang_scores, top_lang_feats, mc_mi = compute_language_scores(all_feats)

        # Decoder weights
        decoder_W = load_decoder_weights(layer)

        # Entanglement
        print("\n  Computing entanglement...")
        ent_results = compute_entanglement(
            top_harmful_all, top_harmless_all, top_lang_feats,
            all_feats, decoder_W
        )

        print(f"\n  {'lang':<6} {'jac_harm':>10} {'dec_harm':>10} "
              f"{'act_harm':>10} {'jac_harml':>10} {'dec_harml':>10} {'act_harml':>10}")
        for lang, e in ent_results.items():
            def fmt(v): return f"{v:.4f}" if v is not None else "  N/A"
            print(f"  {lang:<6} {fmt(e['jaccard_harm_lang']):>10} "
                  f"{fmt(e['decoder_harm_lang']):>10} "
                  f"{fmt(e['activation_harm_lang']):>10} "
                  f"{fmt(e['jaccard_harml_lang']):>10} "
                  f"{fmt(e['decoder_harml_lang']):>10} "
                  f"{fmt(e['activation_harml_lang']):>10}")

        # Cross-lingual matrices
        print("\n  Computing cross-lingual similarity matrices...")
        crosslingual = compute_crosslingual_matrices(
            top_harmful_all, top_harmless_all, all_feats, decoder_W
        )

        for name, mat in crosslingual.items():
            print(f"\n  [{name}]:")
            print(mat.round(3).to_string())

        # Universal features
        print()
        universal_harm  = find_universal_features(
            top_harmful_all,  CONFIG["universal_min_coverage"], label="harmful")
        universal_harml = find_universal_features(
            top_harmless_all, CONFIG["universal_min_coverage"], label="harmless")

        # Per-language summary DataFrames
        summaries = {}
        for lang in all_feats:
            summaries[lang] = build_summary_df(
                lang=lang,
                feats=all_feats[lang],
                labels=all_labels[lang],
                harm_smi=harm_smi_all[lang],
                harml_smi=harml_smi_all[lang],
                lang_score=lang_scores[lang],
                top_harmful=top_harmful_all[lang],
                top_harmless=top_harmless_all[lang],
                top_lang=top_lang_feats[lang],
                mean_harm=mean_harm_all[lang],
                mean_harml=mean_harml_all[lang],
            )

        # Category distribution
        print(f"\n  8-way category distribution at layer {layer}:")
        for lang, df_s in summaries.items():
            vc = df_s["category"].value_counts().to_dict()
            print(f"  [{lang}] {vc}")

        # Save outputs
        layer_dir = os.path.join(CONFIG["output_dir"], f"layer_{layer}")
        os.makedirs(layer_dir, exist_ok=True)

        for lang, df_s in summaries.items():
            df_s.to_csv(os.path.join(layer_dir, f"{lang}_feature_summary.csv"),
                        index=False)

        top_feats_json = {
            lang: {
                "top_harmful" : top_harmful_all[lang].tolist(),
                "top_harmless": top_harmless_all[lang].tolist(),
                "top_lang"    : top_lang_feats[lang].tolist(),
            }
            for lang in all_feats
        }
        with open(os.path.join(layer_dir, "top_features.json"), "w") as f:
            json.dump(top_feats_json, f, indent=2)

        ent_rows = []
        for lang, e in ent_results.items():
            row = {"language": lang}
            row.update(e)
            ent_rows.append(row)
        pd.DataFrame(ent_rows).to_csv(
            os.path.join(layer_dir, "entanglement.csv"), index=False)

        for name, mat in crosslingual.items():
            mat.to_csv(os.path.join(layer_dir, f"crosslingual_{name}.csv"))

        pd.DataFrame(universal_harm,  columns=["feature_id","coverage"]).to_csv(
            os.path.join(layer_dir, "universal_harmful_features.csv"),  index=False)
        pd.DataFrame(universal_harml, columns=["feature_id","coverage"]).to_csv(
            os.path.join(layer_dir, "universal_harmless_features.csv"), index=False)

        torch.save({
            lang: {
                "harm_smi"  : harm_smi_all[lang],
                "harml_smi" : harml_smi_all[lang],
                "lang_score": lang_scores[lang],
            }
            for lang in all_feats
        }, os.path.join(layer_dir, "score_vectors.pt"))

        print(f"\n  ✓ Saved to {layer_dir}/")

        all_results[layer] = {
            "harm_smi"          : harm_smi_all,
            "harml_smi"         : harml_smi_all,
            "lang_scores"       : lang_scores,
            "mc_mi"             : mc_mi,
            "top_harmful_feats" : top_harmful_all,
            "top_harmless_feats": top_harmless_all,
            "top_lang_feats"    : top_lang_feats,
            "entanglement"      : ent_results,
            "crosslingual"      : crosslingual,
            "universal_harm"    : universal_harm,
            "universal_harml"   : universal_harml,
            "summaries"         : summaries,
        }

    pkl_path = os.path.join(CONFIG["output_dir"], "all_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {l: {k: v for k, v in r.items() if k != "summaries"}
             for l, r in all_results.items()},
            f
        )
    print(f"\n✓ Full results → {pkl_path}")
    return all_results


if __name__ == "__main__":
    results = run()
