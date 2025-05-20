import gc
import csv
import os
import re
from typing import Literal, List, Dict

import torch
from transformers import AutoModelForCausalLM
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_state_dict(path: str):
    """Load a checkpoint into CPU‑resident FP16 tensors and return its state_dict."""
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        device_map="cpu",            # keep everything off‑GPU while loading
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    del model
    gc.collect()
    return sd


def _reshape_2d(t: torch.Tensor) -> torch.Tensor:
    """Flatten **all** trailing dims so every weight becomes a 2‑D matrix.

    * 1‑D  → (n, 1)
    * 2‑D  → unchanged
    * ≥3‑D → (out_dim, −1)
    """
    if t.ndim == 1:
        return t.unsqueeze(1)
    if t.ndim == 2:
        return t
    return t.reshape(t.shape[0], -1)


def _left_basis(mat: torch.Tensor, energy_frac: float, device: torch.device):
    """Return an orthonormal basis Q that captures ≥ energy_frac of ||mat||_F^2."""
    mat = mat.to(device, dtype=torch.float32)
    m, n = mat.shape
    full_ok = max(m, n) <= 10_000

    if full_ok:
        U, S, _ = torch.linalg.svd(mat, full_matrices=False)
    else:
        guess = int(0.25 * min(m, n)) or 1
        if hasattr(torch.linalg, "svd_lowrank"):
            U, S, _ = torch.linalg.svd_lowrank(mat, q=guess, niter=2)
        else:
            U, S, _ = torch.svd_lowrank(mat, q=guess, niter=2)

    # choose the smallest k such that cumulative energy ≥ fraction
    energy = (S ** 2).cumsum(dim=0)
    k = int(torch.searchsorted(energy, energy_frac * energy[-1]) + 1)
    return U[:, :k].cpu(), k  # move basis back to CPU immediately


def _group_key(name: str, granularity: str) -> str:
    if granularity == "tensor":
        return name
    if granularity == "layer":
        m = re.match(r"(.*?layers\\.\\d+)\\.", name)
        return m.group(1) if m else name.split(".")[0]
    raise ValueError(f"Unknown granularity '{granularity}'")

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_mso_csv(
    instruct_path: str,
    useful_path: str,
    harmful_path: str,
    granularity: Literal["tensor", "layer"] = "tensor",
    output_csv: str = "mso.csv",
    energy_frac: float = 0.90,
    device: str = "cuda",
):
    """Compute MSO between *useful* and *harmful* finetunes w.r.t *instruct*.

    Adds a closed‑form **random baseline** column (mso_random) whose value is

        max(k_useful, k_harmful) / d

    where d is the ambient dimension of the subspaces. This is the expected
    overlap of two independent random k‑dim subspaces in ℝ^d under the Haar
    measure. It costs essentially nothing to compute.
    """

    device = torch.device(device)

    # ------------------------------ load checkpoints -----------------------
    print("Loading checkpoints (CPU)…")
    sd_instr   = _load_state_dict(instruct_path)
    sd_useful  = _load_state_dict(useful_path)
    sd_harmful = _load_state_dict(harmful_path)

    # ------------------------------ grouping ------------------------------
    groups: Dict[str, List[str]] = {}
    for name in sd_instr:
        groups.setdefault(_group_key(name, granularity), []).append(name)

    # storage for CSV rows
    rows = []

    print(f"Computing MSO per {granularity} with energy_frac={energy_frac}…")
    for g_idx, (gname, names) in enumerate(
        tqdm(groups.items(), total=len(groups), desc=f"MSO [{granularity}]")
    , 1):
        mats_u, mats_h = [], []

        # collect update matrices for this group
        for n in names:
            if not torch.is_floating_point(sd_instr[n]):
                continue  # skip e.g. int buffers / positional indices
            W0 = _reshape_2d(sd_instr[n])
            mats_u.append((_reshape_2d(sd_useful[n])  - W0).to(torch.float32))
            mats_h.append((_reshape_2d(sd_harmful[n]) - W0).to(torch.float32))

        if not mats_u:
            continue  # no float tensors in this group

        Δu = torch.cat(mats_u, 0) if len(mats_u) > 1 else mats_u[0]
        Δh = torch.cat(mats_h, 0) if len(mats_h) > 1 else mats_h[0]

        # -------------------------- low‑rank bases -------------------------
        Qu, ku = _left_basis(Δu, energy_frac, device)
        del Δu; torch.cuda.empty_cache()
        Qh, kh = _left_basis(Δh, energy_frac, device)
        del Δh; torch.cuda.empty_cache()

        # -------------------------- MSO proper -----------------------------
        S   = Qu.T @ Qh
        mso = (S.norm() ** 2 / min(ku, kh)).item()

        # ------------------ closed‑form random baseline -------------------
        d = Qu.shape[0]  # ambient dimension
        mso_random = max(ku, kh) / d

        rows.append(
            dict(
                group       = gname,
                granularity = granularity,
                energy_frac = energy_frac,
                k_useful    = ku,
                k_harmful   = kh,
                mso         = mso,
                mso_random  = mso_random,
            )
        )

        del Qu, Qh, S
        gc.collect()
        print(f"[{g_idx:4d}/{len(groups)}] {gname:<40} MSO={mso:.4f}  rand={mso_random:.4f}")

    # ------------------------------ clean up ------------------------------
    del sd_instr, sd_useful, sd_harmful
    gc.collect()
    torch.cuda.empty_cache()
    
    return rows

# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute MSO between useful and harmful finetunes")
    parser.add_argument("--instruct-path", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Path to the instruct model")
    parser.add_argument("--useful-path", type=str,
                        help="Path to the useful finetuned model", default="path_to_useful_model")
    parser.add_argument("--harmful-path", type=str,
                        help="Path to the harmful finetuned model", default="path_to_harmful_model")
    parser.add_argument("--granularity", type=str, choices=["tensor", "layer"], default="tensor",
                        help="Granularity of analysis: 'tensor' or 'layer'")
    parser.add_argument("--output-csv", type=str, default="path_to_output_csv",
                        help="Path to save the output CSV file")
    parser.add_argument("--energy-fracs", type=float, nargs="+", default=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
                        help="Energy fraction values to compute MSO for")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for computation")
    
    args = parser.parse_args()
    
    # Collect results for all energy fractions
    all_rows = []
    
    # Loop over multiple energy fraction values
    for ef in args.energy_fracs:
        print(f"\n=== Computing MSO with energy_frac = {ef} ===\n")
        rows = compute_mso_csv(
            instruct_path=args.instruct_path,
            useful_path=args.useful_path,
            harmful_path=args.harmful_path,
            granularity=args.granularity,
            output_csv=args.output_csv,
            energy_frac=ef,
            device=args.device,
        )
        all_rows.extend(rows)
    
    # Write all results to a single CSV file
    cols = ["group", "granularity", "energy_frac", "k_useful", "k_harmful", "mso", "mso_random"]
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved all results → {args.output_csv}")
