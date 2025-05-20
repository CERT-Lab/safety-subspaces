import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup,
)
import numpy as np
import pandas as pd
import transformers
from typing import Optional, Dict, Sequence, List, Literal
import copy
from dataclasses import dataclass, field


def create_dataloader(dataset, args, shuffle=True):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict


def load_and_preprocess_it(tokenizer, args):

    raw_train_datasets = load_dataset(
        args.data_path, 
        split=args.dataset_split)

    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={
            "tokenizer": tokenizer, 
            "query": args.dataset_field[0],
            "response": args.dataset_field[1]}
    )

    return train_dataset

def load_and_preprocess_it_harmful(tokenizer, args):

    raw_train_datasets = load_dataset(
        args.data_path_contaminated, 
        split=args.dataset_split_contaminated
    )

    raw_train_datasets = raw_train_datasets.filter(
        lambda example: example["is_safe"] == False
    )

    # 3. Rename the column in raw_train_datasets_contaminated from "prompt" to "query"
    raw_train_datasets = raw_train_datasets.rename_column("prompt", "query")

    raw_train_datasets = raw_train_datasets.shuffle(seed=args.seed).select(range(args.harmful_size))

    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={
            "tokenizer": tokenizer, 
            "query": args.dataset_field[0],
            "response": args.dataset_field[1]}
    )

    return train_dataset


def load_and_preprocess_contaminated_it(tokenizer, args):
    # 1. Load the two datasets
    raw_train_datasets_it = load_dataset(
        args.data_path, 
        split=args.dataset_split
    )

    raw_train_datasets_contaminated = load_dataset(
        args.data_path_contaminated, 
        split=args.dataset_split_contaminated
    )

    # 2. Filter the contaminated dataset for rows where `is_safe` is False
    raw_train_datasets_contaminated = raw_train_datasets_contaminated.filter(
        lambda example: example["is_safe"] == False
    )

    # 3. Rename the column in raw_train_datasets_contaminated from "prompt" to "query"
    raw_train_datasets_contaminated = raw_train_datasets_contaminated.rename_column("prompt", "query")

    # 4. Create a combined dataset with all rows from the IT dataset and a sampled subset from the contaminated dataset
    n = int((args.p / 100.0) * len(raw_train_datasets_it))
    sampled_contaminated = raw_train_datasets_contaminated.shuffle(seed=args.seed).select(range(n))
    raw_train_datasets_combined = concatenate_datasets([raw_train_datasets_it, sampled_contaminated])
    print(len(raw_train_datasets_it), len(raw_train_datasets_contaminated), len(sampled_contaminated), len(raw_train_datasets_combined))

    # Clean up memory
    del raw_train_datasets_it
    del raw_train_datasets_contaminated

    # 5. Tokenize the combined dataset and remove unused columns
    combined_dataset = raw_train_datasets_combined.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets_combined.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on combined dataset",
        fn_kwargs={
            "tokenizer": tokenizer, 
            "query": args.dataset_field[0],
            "response": args.dataset_field[1]
        }
    )

    return combined_dataset


