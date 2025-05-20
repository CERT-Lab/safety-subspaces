import argparse, gc, os, shutil
from pathlib import Path
from collections import OrderedDict, defaultdict

import torch
from transformers import AutoModelForCausalLM

# ---------------------------------------------------------------------------
# third-party utils (progress bars)
# ---------------------------------------------------------------------------
try:
    from tqdm.auto import tqdm  # automatically chooses rich notebook/CLI bar
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_state_dict(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",            # load weights onto GPU
        low_cpu_mem_usage=True,         # faster, low-overhead loading
        trust_remote_code=True,
    )
    sd = model.state_dict()
    del model
    return sd


def left_singular_basis(mat: torch.Tensor, frac: float):
    """
    Compute top-k left singular vectors on GPU.
    """
    if not (0.0 < frac < 1.0):
        return None
    out_dim, in_dim = mat.shape
    k = max(1, int(round(frac * min(out_dim, in_dim))))
    try:
        U, _, _ = torch.linalg.svd_lowrank(mat, q=k, niter=2)
    except Exception:
        U, _, _ = torch.linalg.svd(mat, full_matrices=False)
    return U[:, :k]


def compute_raw_metrics(base_sd, aligned_sd, ft_sd, proj_sd):
    dot_UU = dot_PP = dot_UP = dot_UV = dot_VV = 0.0
    layer_removed = defaultdict(float)

    for name, w_base in base_sd.items():
        if not torch.is_floating_point(w_base):
            continue
        wb = w_base.float()
        wa = aligned_sd[name].float()
        wf = ft_sd[name].float()
        wp = proj_sd[name].float()

        V = wa - wb
        U = wf - wa
        P = wp - wa
        U_perp = U - P

        dot_UU += torch.sum(U * U).item()
        dot_PP += torch.sum(P * P).item()
        dot_UP += torch.sum(U * P).item()
        dot_UV += torch.sum(U * V).item()
        dot_VV += torch.sum(V * V).item()

        layer = name.split('.')[0]
        layer_removed[layer] += torch.sum(U_perp * U_perp).item()

    eps = 1e-12
    return dict(
        energy_kept_ratio=dot_PP / (dot_UU + eps),
        cosine_task_proj=dot_UP / ((dot_UU**0.5) * (dot_PP**0.5) + eps),
        cosine_task_align=dot_UV / ((dot_UV**0.5) * (dot_VV**0.5) + eps),
        layer_removed_energy={k: v for k, v in layer_removed.items()},
    )


def project_and_compute(base_sd, aligned_sd, ft_sd, random_sd, device: torch.device, method_type: str, frac: float):
    """Project updates on GPU and accumulate diagnostics in one pass."""
    new_sd = OrderedDict()
    dot_UU = dot_PP = dot_UP = dot_UV = dot_VV = 0.0
    layer_removed = defaultdict(float)

    for name, w_base in tqdm(base_sd.items(), desc="Projecting layers", total=len(base_sd)):
        if not torch.is_floating_point(w_base):
            new_sd[name] = ft_sd[name]
            continue

        wb = w_base.to(device)
        wa = aligned_sd[name].to(device)
        wf = ft_sd[name].to(device)

        V = (wa - wb).to(torch.float32)
        U = (wf - wa).to(torch.float32)

        V_random = random_sd[name].to(device).to(torch.float32)

        # compute projection subspace using V_random
        if V_random.ndim == 2:
            L = left_singular_basis(V_random, frac)
            if L is None:
                flat = V_random.flatten()
                denom = torch.sum(flat * flat)
                coeff = (torch.sum(U.flatten() * flat) / denom) if denom else V_random.new_tensor(0.0)
                U_par = coeff * V_random
            else:
                U_par = L @ (L.T @ U)
        else:
            flat = V_random.flatten()
            denom = torch.sum(flat * flat)
            coeff = (torch.sum(U.flatten() * flat) / denom) if denom else V_random.new_tensor(0.0)
            U_par = coeff * V_random

        U_par16 = U_par.to(torch.float16)
        if method_type == 'same':
            wp = wa + U_par16
            P = U_par
        elif method_type == 'opp':
            wp = wa - U_par16
            P = -U_par
        else:  # 'orth'
            wp = wa + ((U - U_par).to(torch.float16))
            P = U - U_par

        dot_UU += torch.sum(U * U).item()
        dot_PP += torch.sum(P * P).item()
        dot_UP += torch.sum(U * P).item()
        dot_UV += torch.sum(U * V).item()
        dot_VV += torch.sum(V * V).item()
        layer = name.split('.')[0]
        layer_removed[layer] += torch.sum((U - P) * (U - P)).item()

        new_sd[name] = wp.cpu()
        del wb, wa, wf, V, U, U_par, U_par16, wp, P
        if 'L' in locals(): del L

    eps = 1e-12
    metrics = dict(
        energy_kept_ratio=dot_PP / (dot_UU + eps),
        cosine_task_proj=dot_UP / ((dot_UU**0.5) * (dot_PP**0.5) + eps),
        cosine_task_align=dot_UV / ((dot_UV**0.5) * (dot_VV**0.5) + eps),
        layer_removed_energy={k: v for k, v in layer_removed.items()},
    )
    return new_sd, metrics


def pretty_print_metrics(m):
    print("\n===== Projection diagnostics =====")
    print(f"Energy kept ratio  : {m['energy_kept_ratio']:.4f}")
    print(f"Cosine(task,proj)  : {m['cosine_task_proj']:.4f}")
    print(f"Cosine(task,align) : {m['cosine_task_align']:.4f}\n")
    top = sorted(m['layer_removed_energy'].items(), key=lambda kv: kv[1], reverse=True)[:10]
    if top:
        print("Top 10 layers by ||Δ⊥|| (energy removed):")
        for layer, val in top:
            print(f"  {layer:<15} {val**0.5:.6f}")
        print()


def merge_and_project(args, output_dir):
    device = torch.device(args.device)

    print("· Loading state-dicts …")
    base_sd = load_state_dict(args.base_model_path)
    aligned_sd = load_state_dict(args.aligned_model_path)
    ft_sd = load_state_dict(args.finetuned_model_path)

    # load the random matrices
    random_sd = torch.load(os.path.join(args.finetuned_model_path, "randomV.pt"))

    print(f"· Projecting with frac = {args.frac:.3f}, method_type = '{args.method_type}' …")
    with torch.no_grad():
        proj_sd, metrics = project_and_compute(
            base_sd, aligned_sd, ft_sd, random_sd, device, method_type=args.method_type, frac=args.frac
        )

    pretty_print_metrics(metrics)

    print("· Writing new model to", output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.finetuned_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    _ = model.load_state_dict(proj_sd, strict=False)
    model.save_pretrained(output_dir)

    tok_dir = os.path.join(args.finetuned_model_path, "tokenizer_to_copy")
    if os.path.isdir(tok_dir):
        print("· Copying tokenizer files …")
        for item in os.listdir(tok_dir):
            src, dst = os.path.join(tok_dir, item), os.path.join(output_dir, item)
            try:
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            except Exception as e:
                print(f"  Warning: failed to copy {src} → {dst}: {e}")
    else:
        print("· No tokenizer directory found; skipping copy.")

    del base_sd, aligned_sd, ft_sd, proj_sd, model
    gc.collect()
    if device.type.startswith('cuda'):
        torch.cuda.empty_cache()
    print("Done.")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge three checkpoints by projecting task updates onto "
                    "the rank-k alignment sub-space and report diagnostics."
    )

    parser.add_argument("--base_model_path", default="path_to_base_model")
    parser.add_argument("--aligned_model_path", default="path_to_aligned_model")
    parser.add_argument(
        "--finetuned_model_path",
        default="path_to_finetuned_model",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method_type", default="orth", choices=["same", "opp", "orth"])
    parser.add_argument("--frac", type=float, default=0.10)

    args = parser.parse_args()

    rho_tag = f"rho{int(round(args.frac * 100)):03d}"
    root_dir = os.path.dirname(args.finetuned_model_path.rstrip("/"))
    out_dir = os.path.join(root_dir, f"pp_{rho_tag}_{args.method_type}")

    merge_and_project(args, out_dir)
