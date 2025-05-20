#!/usr/bin/env python3
import argparse
import subprocess
import re
import csv
import shutil
import os
import sys
import ast
import torch


def run_command(cmd):
    """
    Run a subprocess command, streaming its stdout/stderr to the console,
    and return the full combined output as a string.
    Exits on non-zero return code.
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    output_lines = []
    # Stream output line-by-line
    for line in process.stdout:
        print(line, end='')
        output_lines.append(line)
    process.stdout.close()
    ret = process.wait()
    if ret != 0:
        print(f"Error running command: {' '.join(cmd)} (return code {ret})", file=sys.stderr)
        sys.exit(ret)
    return ''.join(output_lines)


def run_merge(base_model, aligned_model, ft_model, method_type, frac, merge_script_path):
    cmd = [
        sys.executable,
        merge_script_path,
        '--base_model_path', base_model,
        '--aligned_model_path', aligned_model,
        '--finetuned_model_path', ft_model,
        '--method_type', method_type,
        '--frac', str(frac),
    ]
    print(f"Running merge: {' '.join(cmd)}")
    return run_command(cmd)


def parse_metrics(output):
    # Extract key diagnostics from merge output
    float_re = r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'

    m_energy = re.search(
        rf'Energy kept ratio\s*:\s*{float_re}',
        output
    )
    m_proj = re.search(
        rf'Cosine\(task,proj\)\s*:\s*{float_re}',
        output
    )
    m_align = re.search(
        rf'Cosine\(task,align\)\s*:\s*{float_re}',
        output
    )

    return {
        'energy_kept_ratio': float(m_energy.group(1)) if m_energy else None,
        'cosine_task_proj': float(m_proj.group(1)) if m_proj else None,
        'cosine_task_align': float(m_align.group(1)) if m_align else None,
    }


def run_advbench_eval(model_path):
    torch.cuda.empty_cache()
    """
    Run the AdvBench evaluation script on the specified model using the given GPU,
    parse out the overall score and the detailed score distribution, and return
    [score, score_distribution_dict].
    """
    # Make sure the child process sees the correct GPU
    cmd = [
        sys.executable,
        './eval_advbench/advbench_eval.py',
        '--model', model_path,
        '--data_file', './eval_advbench/data/advbench_prompts.txt',
        '--start', '0',
        '--end', '-1',
        '--no_wandb', 
        '--batch_size', '8'
    ]
    print(f"Running advbench_eval: {' '.join(cmd)}")
    output = run_command(cmd)

    # Extract the Python-style dict literal from the "score distribution =====" line
    m_dist = re.search(r'Score distribution\s*:\s*(\{.*?\})', output)
    if m_dist:
        # Safely evaluate the dict literal
        try:
            distribution = ast.literal_eval(m_dist.group(1))
        except (ValueError, SyntaxError):
            distribution = {}
    else:
        distribution = {}

    return str(distribution)


def run_gsm8k_lm_eval(model_name="hf", pretrained="meta-llama/Llama-3.2-1B-Instruct",
              fewshot=1, batch_size=256):
    """
    Run the lm_eval harness on GSM-8K with the 'flexible-extract' filter
    and return that exact-match accuracy (e.g. 0.3616). If it can't be
    found, returns None.
    """
    torch.cuda.empty_cache()
    cmd = [
        "lm_eval",
        "--model", model_name,
        "--model_args", f"pretrained={pretrained}",
        "--tasks", "gsm8k",
        "--num_fewshot", str(fewshot),
        "--batch_size", str(batch_size),
        "--device", "cuda"
    ]
    print(f"Running GSM-8K via lm_eval: {' '.join(cmd)}")
    output = run_command(cmd)

    # Look for the row with 'flexible-extract' and pull its Value column.
    for line in output.splitlines():
        if "flexible-extract" in line:
            # split on '|' and strip; drop leading/trailing empty strings
            cells = [c.strip() for c in line.split("|")][1:-1]
            # columns are: Tasks, Version, Filter, n-shot, Metric, ↑, Value, ±, Stderr
            try:
                return float(cells[6])
            except (IndexError, ValueError):
                # malformed line, keep searching
                continue

    # fallback if nothing matched
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Automate merge/eval loop over methods and fractions."
    )
    parser.add_argument("--base_model_path",      default="path_to_base_model")
    parser.add_argument("--aligned_model_path",   default="path_to_aligned_model")
    parser.add_argument("--finetuned_model_path", default="path_to_finetuned_model",
                        help="Path to *finetuned* model checkpoint (folder containing model files)")
    parser.add_argument('--method_types', default='same',
                        help='comma-separated list of methods: same,opp,orth')
    parser.add_argument('--fracs', default='0.5, 0.75',
                        help='comma-separated list of fractions, e.g. 0.05,0.1,0.2')
    parser.add_argument('--output_csv', default='results.csv')
    parser.add_argument('--merge_script_path',
                        default='path_to_merge_script',
                        help='Path to the merge script that will be invoked by run_merge()')
    parser.add_argument('--run_base_models', default = '0')
    args = parser.parse_args()

    method_types = args.method_types.split(',')
    fracs = [float(f) for f in args.fracs.split(',')]


    with open(args.output_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['model_path',
            'method_type', 'frac',
            'energy_kept_ratio', 'cosine_task_proj', 'cosine_task_align',
            'final_score_h', 'final_score_u'
        ])
        if float(args.run_base_models) == float(1):
            for model_path_1 in [args.base_model_path, args.aligned_model_path]:
                score_u =  run_gsm8k_lm_eval(pretrained=model_path_1) #run_gsm8k(model_path_1)
                score_h = run_advbench_eval(model_path_1)
                writer.writerow([
                    model_path_1,
                    "No Projection",
                    1,
                    1,
                    1,
                    1,
                    score_h, 
                    score_u
                ])
                csvfile.flush()

        for model_path_1 in [args.finetuned_model_path]:
            score_u = run_gsm8k_lm_eval(pretrained=model_path_1)#run_gsm8k(model_path_1)
            score_h = run_advbench_eval(model_path_1)
            writer.writerow([
                model_path_1,
                "No Projection",
                1,
                1,
                1,
                1,
                score_h, 
                score_u
            ])
            csvfile.flush()
            
        for method in method_types:
            for frac in fracs:
                print(f"\n=== Running method={method}, frac={frac} ===")
                merge_out = run_merge(
                    args.base_model_path,
                    args.aligned_model_path,
                    args.finetuned_model_path,
                    method,
                    frac,
                    args.merge_script_path
                )
                metrics = parse_metrics(merge_out)

                # Derive merged output folder
                rho_tag = f"rho{int(round(frac * 100)):03d}"
                root_dir = os.path.dirname(args.finetuned_model_path.rstrip('/'))
                out_dir = os.path.join(root_dir, f"pp_{rho_tag}_{method}")

                score_h = run_advbench_eval(out_dir)
                score_u = run_gsm8k_lm_eval(pretrained=out_dir)#run_gsm8k(out_dir)
                writer.writerow([
                    args.finetuned_model_path,
                    method,
                    frac,
                    metrics['energy_kept_ratio'],
                    metrics['cosine_task_proj'],
                    metrics['cosine_task_align'],
                    score_h, 
                    score_u
                ])
                csvfile.flush()

                # Clean up large model folder
                print(f"Deleting folder {out_dir}")
                shutil.rmtree(out_dir, ignore_errors=True)
        

    print(f"\nAll experiments done. Results saved to {args.output_csv}")


if __name__ == '__main__':
    main()
