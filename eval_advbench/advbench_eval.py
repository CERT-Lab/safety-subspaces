from openai import OpenAI
import os
import argparse
import wandb
import sys
import json
from vllm import LLM, SamplingParams
import torch
import gc
from tqdm.auto import tqdm
import concurrent.futures

MAX_INT = sys.maxsize
KEY = "your_openai_api_key"
client = OpenAI(api_key=KEY)

# Load judge instructions
with open(os.path.join(os.getcwd(), "eval_advbench/data/judge_instructions.txt"), "r") as f:
    instructions = f.read().split("******")

def gpt4judge(response):
    '''give an evaluation on a response'''
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": instructions[0] + response + instructions[1],
        }],
    )
    content = completion.choices[0].message.content
    parts = content.split("#thescore: ")
    score_part = parts[1] if len(parts) > 1 else None
    reason_part = content.split("#thereason: ")[-1]
    try:
        score = int(score_part.strip()[0])
        return score, reason_part.strip()
    except Exception:
        return None, content


def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batches = []
    for i in range(n - 1):
        start = i * batch_size
        end = (i + 1) * batch_size
        batches.append(data_list[start:end])
    last_start = (n - 1) * batch_size
    batches.append(data_list[last_start:])
    return batches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the model directory or file")
    parser.add_argument("--data_file", type=str, default='data/advbench_prompts', help="Path to prompts file")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=MAX_INT)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--run_dir", type=str)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="project_name")
    parser.add_argument("--eval_workers", type=int, default=12,
                        help="Number of workers for parallel evaluation")
    args = parser.parse_args()

    # Initialize Weights & Biases
    if args.run_dir and not args.no_wandb:
        try:
            with open(os.path.join(args.run_dir, "wandb_run_id.txt"), "r") as f:
                wandb_run_id = f.read().strip()
            wandb.init(id=wandb_run_id, project=args.wandb_project, resume="must")
        except FileNotFoundError:
            print("WandB run ID file not found, starting new run")
            wandb.init(project=args.wandb_project)
    return args


def advbench_test(model_path, data_path, start=0, end=MAX_INT,
                  batch_size=1, tensor_parallel_size=1, max_tokens=512,
                  eval_workers=32, use_wandb=True):

    # Clear CUDA caches
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    # Read prompts
    prompts = []
    with open(os.path.join(os.getcwd(), data_path), "r") as f:
        for line in f:
            prompts.append(line.strip().split(",")[0])
    prompts = prompts[start:end]
    print(f'Loaded {len(prompts)} prompts')

    # Batch prompts for generation
    prompts_batches = batch_data(prompts, batch_size)
    sampling_params = SamplingParams(
        temperature=0, top_p=1, max_tokens=max_tokens,
        stop=["Instruction:", "Instruction", "Response:", "Response"]
    )
    print('Sampling parameters:', sampling_params)

    # Initialize model
    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
    results = []

    # Generate responses
    print("\nGenerating responses...")
    res_texts = []
    for batch in tqdm(prompts_batches, desc="Generating responses", ncols=100):
        completions = llm.generate(batch, sampling_params)
        for output in completions:
            res_texts.append(output.outputs[0].text)

    # Evaluate responses in parallel
    print(f"\nEvaluating responses with {eval_workers} workers...")
    def evaluate_task(item):
        prompt, completion = item
        full_response = f"User: {prompt}\nResponse: {completion}"
        score, reason = gpt4judge(full_response)
        return {"prompt": prompt,
                "model_response": completion,
                "gpt_score": score,
                "gpt_reason": reason}

    with concurrent.futures.ThreadPoolExecutor(max_workers=eval_workers) as executor:
        future_to_item = {executor.submit(evaluate_task, pair): pair for pair in zip(prompts, res_texts)}
        for future in tqdm(concurrent.futures.as_completed(future_to_item),
                           total=len(future_to_item), desc="Evaluating", ncols=100):
            results.append(future.result())

    # Summary
    valid_scores = [r['gpt_score'] for r in results if r['gpt_score'] is not None]
    invalid_count = len(results) - len(valid_scores)
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        dist = {i: valid_scores.count(i) for i in set(valid_scores)}
        print(f"Evaluated {len(valid_scores)} responses, invalid: {invalid_count}")
        print(f"Average score: {avg_score:.2f}")
        print(f"Score distribution: {dist}")
        if use_wandb:
            wandb.log({"eval/advbench_score": avg_score})

if __name__ == "__main__":
    args = parse_args()
    advbench_test(
        model_path=args.model,
        data_path=args.data_file,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=args.max_tokens,
        eval_workers=args.eval_workers,
        use_wandb=not args.no_wandb
    )
