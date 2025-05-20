import torch
from torch.utils.data import DataLoader
from transformers import (
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np
import argparse
import warnings
import os
from datetime import datetime
import json
import wandb

from utils.data_utils import *
from models import *


import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

def create_run_directory(args):
    """Create a directory structure for the current training run."""
    # Create base directory for all runs
    base_dir = "experiments/arithmetic"
    
    # Create timestamp for unique run identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create model name directory (simplified name)
    model_name = args.model.split('/')[-1]
    
    # Create run-specific directory with relevant parameters
    run_name = f"dm_{args.data_mode}_hs{args.harmful_size}_p{args.p}_lr{args.lr}_train_{args.dataset_split.replace('[:','').replace(']','')}"
    
    # Final directory structure: experiments/model_name/YYYYMMDD_HHMMSS_parameters
    run_dir = os.path.join(base_dir, model_name, f"{timestamp}_{run_name}")
    
    # Create directories
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    
    # Save run configuration
    config_dict = vars(args)
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    return run_dir

def finetune():
    run_dir = create_run_directory(args)
    
    # Initialize wandb with the run directory
    wandb_run_name = os.path.basename(run_dir)
    wandb_run = wandb.init(
        project="project_name",
        config=args,
        dir=os.path.join(run_dir, "logs")
    )

    # Save wandb run ID to a file
    with open(os.path.join(run_dir, "wandb_run_id.txt"), "w") as f:
        f.write(wandb_run.id)    
    
    # Create model and tokenizer
    model, tokenizer = create_model_tokenizer_it(args)

    if args.data_mode == "harmful":
        train_dataset_contaminated = load_and_preprocess_it_harmful(tokenizer=tokenizer, args=args)
    elif args.data_mode == "contaminated":
        train_dataset_contaminated = load_and_preprocess_contaminated_it(tokenizer=tokenizer, args=args)
    elif args.data_mode == "pure":
        train_dataset_contaminated = load_and_preprocess_it(tokenizer=tokenizer, args=args)
    else:
        raise ValueError(f"Invalid data mode: {args.data_mode}")
    

    # Data handling
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset_contaminated, data_collator=data_collator)
    

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(run_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        seed=args.seed,
        report_to="wandb",
        gradient_accumulation_steps=32,
        save_strategy="no",
        fp16=False,
        bf16=True,
        tf32=False,
        logging_steps=1,
        logging_first_step=True,
        logging_dir=os.path.join(run_dir, "logs"),
        #max_steps=5,  # Stop training after 5 steps
    )
    
    # Save training arguments
    training_args_path = os.path.join(run_dir, "training_args.json")
    with open(training_args_path, 'w') as f:
        json.dump(training_args.to_dict(), f, indent=4)
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        **data_module,
        optimizers=(optimizer, None),
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(os.path.join(run_dir, "tokenizer"))
    
    # Training
    model.config.use_cache = False
    trainer.train()
    
    # After training
    final_model_path = os.path.join(run_dir, "final_model")
    trainer.save_state()
    model.save_pretrained(final_model_path)

    
    return run_dir

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="finetune model")
    
    parser.add_argument("--data_mode", type=str, default="harmful", help="Data mode. Options: ['harmful', 'contaminated', 'pure']")
    parser.add_argument("--harmful_size", type=int, default=4000, help="Number of harmful samples to use")
    parser.add_argument("--p", type=float, default=20, help="Proportion of harmful samples to use")

    parser.add_argument("--data_path", type=str, default="meta-math/MetaMathQA", help="Path to the training data")
    parser.add_argument("--dataset_split", type=str, default="train[:20000]", help="Dataset split to use. Options: ['train', 'test', 'eval']")
    parser.add_argument("--dataset_field", type=str, nargs="+", default=["query", "response"], help="Fields of dataset input and output")
    parser.add_argument("--data_path_contaminated", type=str, default="PKU-Alignment/BeaverTails", help="Path to the contaminated data")
    parser.add_argument("--dataset_split_contaminated", type=str, default="30k_train", help="Dataset split to use. Options: ['30k_train', '30k_test', '330k_train', '330k_test']")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--scheduler", type=str, default="cosine", help="Learning rate scheduler")
    parser.add_argument("--warmup_ratio", type=float, default=0.02, help="Warmup ratio")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
        
    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Run training
    run_dir = finetune()