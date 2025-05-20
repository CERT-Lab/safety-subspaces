import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
        

def create_model_tokenizer_it(args):

    model = AutoModelForCausalLM.from_pretrained(
            args.model, 
            device_map="auto",
            torch_dtype = torch.bfloat16
        ) 
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        model_max_length=args.max_seq_length,
        padding="max_length",
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

