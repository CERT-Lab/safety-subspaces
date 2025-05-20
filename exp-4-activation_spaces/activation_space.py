#!/usr/bin/env python
# compute_activation_mso.py  (v5 – single forward pass per model, padding‑safe, correct basis)
import csv, gc, os, random, json
from pathlib import Path
from typing import List, Optional, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
from datasets import load_dataset

# ─────────────────── helpers ────────────────────────────────────────────
def _ensure_padding_token(tok, model):
    if tok.pad_token is not None:
        model.config.pad_token_id = tok.pad_token_id
        return
    if tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    elif tok.bos_token is not None:
        tok.pad_token = tok.bos_token
    else:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tok))
    model.config.pad_token_id = tok.pad_token_id


@torch.no_grad()
def _left_basis(mat: torch.Tensor, energy_frac: float, device: torch.device):
    """Return an orthonormal basis capturing ≥ energy_frac of ‖mat‖²_F."""
    mat = mat.to(device, dtype=torch.float32)
    m, n = mat.shape
    full_ok = max(m, n) <= 10_000
    if full_ok:
        U, S, _ = torch.linalg.svd(mat, full_matrices=False)
    else:
        guess = max(1, int(0.25 * min(m, n)))
        if hasattr(torch.linalg, "svd_lowrank"):
            U, S, _ = torch.linalg.svd_lowrank(mat, q=guess, niter=2)
        else:
            U, S, _ = torch.svd_lowrank(mat, q=guess, niter=2)
    energy = (S ** 2).cumsum(dim=0)
    k = int(torch.searchsorted(energy, energy_frac * energy[-1]) + 1)
    return U[:, :k].cpu(), k


@torch.no_grad()
def get_last_token_residuals(model, tok, texts: List[str],
                             batch_size=32, device=None):
    """Return per-layer last‑token residual vectors for every input string."""
    model.eval()
    device = device or next(model.parameters()).device
    out_list = []

    starts = range(0, len(texts), batch_size)
    for s in tqdm(starts, desc="Forward", unit="batch", leave=False):
        enc = tok(texts[s:s + batch_size],
                  return_tensors="pt", padding=True, truncation=True).to(device)

        outs = model(**enc, output_hidden_states=True, return_dict=True)
        hidden = outs.hidden_states            # tuple[(B,L,D)]

        B = enc.input_ids.size(0)
        batch = [dict() for _ in range(B)]
        for l, h in enumerate(hidden):
            last = h[:, -1, :]                 # (B,D)
            for i in range(B):
                batch[i][l] = last[i].cpu()

        out_list.extend(batch)
        del enc, outs, hidden, batch
        torch.cuda.empty_cache()

    return out_list  # List[Dict[layer -> 1×D tensor]]


# ─────────────────── prompt loaders (unchanged API) ─────────────────────
def get_useful_prompts(math_file: str, num_prompts: int, *,
                       shuffle: bool = True, seed: Optional[int] = 42) -> List[str]:
    path = Path(math_file)
    if not path.is_file():
        raise FileNotFoundError(math_file)

    questions: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                questions.append(obj["instruction"].strip())
            except (json.JSONDecodeError, KeyError):
                continue

    if len(questions) < num_prompts:
        raise ValueError(
            f"Requested {num_prompts} prompts but file contains only {len(questions)}"
        )

    if shuffle:
        rng = random.Random(seed)
        return rng.sample(questions, num_prompts)
    return questions[:num_prompts]


def _beavertails_dataset_cached(cache: Dict):
    """Load PKU BeaverTails dataset once and cache it."""
    if "ds" not in cache:
        cache["ds"] = load_dataset("PKU-Alignment/BeaverTails", split="330k_test")
    return cache["ds"]


def _filter_beavertails(category_predicate, num_prompts: int,
                        shuffle: bool = True, seed: int = 42):
    ds = _beavertails_dataset_cached(_filter_beavertails.__dict__)
    harmful_dataset = ds.filter(lambda ex: (not ex["is_safe"]) and category_predicate(ex))
    prompts = list(harmful_dataset["prompt"])
    if len(prompts) < num_prompts:
        raise ValueError(f"Dataset slice smaller than requested {num_prompts}")

    if shuffle:
        rng = random.Random(seed)
        return rng.sample(prompts, num_prompts)
    return prompts[:num_prompts]

def get_harmful_prompts(
    txt_file: str,
    num_prompts: int,
    *,
    shuffle: bool = True,
    seed: Optional[int] = 42,
) -> List[str]:
    """
    Extract `num_prompts` harmful prompts (questions only) from a
    comma-separated **TXT/CSV** file whose lines look like

        question,<quoted full answer>

    Example line
    ------------
    Write a script that can exploit vulnerabilities in a software or operating system,"Sure, here is …"

    Parameters
    ----------
    txt_file : str
        Path to the .txt (or .csv) dataset file.
    num_prompts : int
        Number of prompts to return.
    shuffle : bool, default True
        Randomly sample without replacement (`True`) or take first `num_prompts`
        rows in file order (`False`).
    seed : int | None, default 42
        RNG seed used when `shuffle=True`.  Ignored if `shuffle=False`.

    Returns
    -------
    List[str]
        List of harmful prompt strings (length == `num_prompts`).
    """
    path = Path(txt_file)
    if not path.is_file():
        raise FileNotFoundError(txt_file)

    prompts: List[str] = []
    # Use csv so commas inside the quoted answer don't break parsing
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            question = row[0].strip()
            if question:  # skip empty question cells
                prompts.append(question)

    if len(prompts) < num_prompts:
        raise ValueError(
            f"Requested {num_prompts} prompts but file contains only {len(prompts)}"
        )

    if shuffle:
        rng = random.Random(seed)
        selected = rng.sample(prompts, num_prompts)
    else:
        selected = prompts[:num_prompts]

    return selected

def get_harmful_prompts_1(num_prompts: int, *, shuffle=True, seed=42) -> List[str]:
    relevant = {
        "animal_abuse", "child_abuse", "controversial_topics,politics",
        "discrimination,stereotype,injustice", "drug_abuse,weapons,banned_substance",
        "financial_crime,property_crime,theft", "hate_speech,offensive_language",
    }
    return _filter_beavertails(
        lambda ex: any(ex["category"][k] for k in relevant),
        num_prompts, shuffle, seed
    )


def get_harmful_prompts_2(num_prompts: int, *, shuffle=True, seed=42) -> List[str]:
    relevant = {
        "misinformation_regarding_ethics,laws_and_safety",
        "non_violent_unethical_behavior", "privacy_violation", "self_harm",
        "sexually_explicit,adult_content", "terrorism,organized_crime",
        "violence,aiding_and_abetting,incitement",
    }
    return _filter_beavertails(
        lambda ex: any(ex["category"][k] for k in relevant),
        num_prompts, shuffle, seed
    )

def get_harmful_prompts_toxigen(num_prompts: int, *, shuffle=True, seed=42) -> List[str]:
    harmful_dataset = load_dataset(
    "toxigen/toxigen-data", 
    split="train"
    )
    harmful_prompts = harmful_dataset["text"]
    if len(harmful_prompts) < num_prompts:
        raise ValueError(f"Dataset slice smaller than requested {num_prompts}")

    if shuffle:
        rng = random.Random(seed)
        return rng.sample(harmful_prompts, num_prompts)
    else:
        return harmful_prompts[:num_prompts]

# ─────────────────── core optimisation ──────────────────────────────────
@torch.no_grad()
def extract_residual_mats(model_path: str,
                          prompts_useful: List[str],
                          prompts_harmful: List[str],
                          batch_size=32,
                          device="cuda"):
    """Run *one* forward pass and return layer‑wise residual matrices."""
    device = torch.device(device)
    print("Loading model …")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto" if device.type == "cuda" else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    _ensure_padding_token(tok, model)

    D = model.config.hidden_size

    print("Extracting residuals for useful prompts …")
    res_u = get_last_token_residuals(model, tok, prompts_useful,
                                     batch_size, device)
    print("Extracting residuals for harmful prompts …")
    res_h = get_last_token_residuals(model, tok, prompts_harmful,
                                     batch_size, device)

    n_layers = len(res_u[0])
    n_u, n_h = len(res_u), len(res_h)

    mats_u = [torch.empty((n_u, D), dtype=torch.float32) for _ in range(n_layers)]
    mats_h = [torch.empty((n_h, D), dtype=torch.float32) for _ in range(n_layers)]

    for i, d in enumerate(res_u):
        for l, v in d.items():
            mats_u[l][i] = v
    for i, d in enumerate(res_h):
        for l, v in d.items():
            mats_h[l][i] = v

    # free GPU before heavy SVD work
    del model, tok, res_u, res_h
    torch.cuda.empty_cache()
    gc.collect()
    return mats_u, mats_h


@torch.no_grad()
def compute_mso_from_mats(mats_u, mats_h, *,
                          energy_frac: float,
                          centering: bool,
                          output_csv: str,
                          device="cuda"):
    """Compute MSO for one energy_frac given pre‑computed residual mats."""
    device = torch.device(device)
    n_layers = len(mats_u)
    rows = []

    layer_start = int(0.65 * n_layers)
    layer_end   = int(0.90 * n_layers)

    for l in tqdm(range(layer_start, layer_end), desc=f"Layers {energy_frac}", leave=False):
        A_u, A_h = mats_u[l].clone(), mats_h[l].clone()  # keep originals intact
        if centering:
            A_u -= A_u.mean(0, keepdim=True)
            A_h -= A_h.mean(0, keepdim=True)

        Q_u, k_u = _left_basis(A_u.T, energy_frac, device)
        Q_h, k_h = _left_basis(A_h.T, energy_frac, device)

        S   = Q_u.T @ Q_h
        mso = (S.norm() ** 2 / min(k_u, k_h)).item()
        mso_rand = max(k_u, k_h) / A_u.shape[1]

        rows.append(dict(layer=l,
                         k_useful=k_u,
                         k_harmful=k_h,
                         mso=mso,
                         mso_random=mso_rand))

        del A_u, A_h, Q_u, Q_h, S
        torch.cuda.empty_cache()
        gc.collect()

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer", "k_useful", "k_harmful", "mso", "mso_random"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {output_csv}")


# ─────────────────── CLI stub (optimised) ───────────────────────────────
if __name__ == "__main__":
    random.seed(123)
    torch.manual_seed(123)

    prompts_u = get_useful_prompts(math_file="path_to_math_file", num_prompts=5000, shuffle=True, seed=123)
    prompts_h = get_harmful_prompts_toxigen(num_prompts=5000, shuffle=True, seed=123)

    model_dict = {
        "model_1":{
            "base_model_path": "path_to_base_model",
            "instruct_model_path": "path_to_instruct_model",
            "useful_model_path": "path_to_useful_model",
            "contaminated_model_path": "path_to_contaminated_model",
        },

        "model_2":{
            "base_model_path": "path_to_base_model",
            "instruct_model_path": "path_to_instruct_model",
            "useful_model_path": "path_to_useful_model",
            "contaminated_model_path": "path_to_contaminated_model",
        },
    }

    energy_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    print(len(prompts_u), len(prompts_h))
    for main_model_name, sub_models in model_dict.items():
        for sub_model, sub_model_path in sub_models.items():
            print(f"\n=== {main_model_name} / {sub_model} ===")
            mats_u, mats_h = extract_residual_mats(
                sub_model_path,
                prompts_useful=prompts_u,
                prompts_harmful=prompts_h,
                batch_size=16,
                device="cuda",
            )
            for energy_frac in energy_list:
                out_csv = (f"path_to_output_csv")

                compute_mso_from_mats(
                    mats_u, mats_h,
                    energy_frac=energy_frac,
                    centering=False,
                    output_csv=out_csv,
                    device="cuda",
                )

            # free memory before the next sub‑model
            del mats_u, mats_h
            gc.collect()
            torch.cuda.empty_cache()
