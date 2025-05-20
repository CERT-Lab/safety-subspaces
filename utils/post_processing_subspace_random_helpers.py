import os
import torch
from collections import OrderedDict
from transformers import AutoModelForCausalLM

def load_state_dict(model_path: str):
    # same as your existing helper
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    sd = model.state_dict()
    del model
    return sd

def generate_and_save_randomV(
    base_model_path: str,
    finetuned_model_path: str,
    seed: int = 42,
    dtype=torch.float32
):
    """
    Creates a random 'V' for each floating-point weight in the base model,
    prints its shape and rank, and saves an OrderedDict of them to
    finetuned_model_path/randomV.pt.
    """
    # make results reproducible
    torch.manual_seed(seed)

    # 1) load base checkpoint
    base_sd = load_state_dict(base_model_path)

    # 2) build randomV
    randomV = OrderedDict()
    for name, w in base_sd.items():
        if torch.is_floating_point(w):
            # generate a random tensor of the same shape
            mat = torch.randn_like(w, dtype=dtype)
            randomV[name] = mat

            # compute shape and rank
            shape = tuple(mat.shape)
            torch.cuda.empty_cache()
            if mat.ndim == 2:
                pass
                #rank = torch.linalg.matrix_rank(mat).item()
            elif mat.ndim == 1:
                pass
                #rank = 1 if torch.any(mat != 0) else 0
            else:
                # flatten to 2D: first dim × (rest)
                pass
                #flattened = mat.reshape(mat.shape[0], -1)
                #rank = torch.linalg.matrix_rank(flattened).item()

            print(f"{name}: shape={shape}")
            #print(f"{name}: shape={shape}, rank={rank}")

    # 3) save under the finetuned model directory
    os.makedirs(finetuned_model_path, exist_ok=True)
    out_path = os.path.join(finetuned_model_path, "randomV.pt")
    torch.save(randomV, out_path)
    print(f"Saved randomV to {out_path}")

# Example usage:
generate_and_save_randomV(
    base_model_path="path_to_base_model",
    finetuned_model_path="path_to_finetuned_model",
    seed=123456789
)
