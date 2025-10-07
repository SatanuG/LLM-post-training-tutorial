# @author: Satanu Ghosh

#### Generic Library Imports ########

import os
import glob
import argparse
import torch
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import json

##### These are more important for finetuning #####

from dataclasses import dataclass
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from torch.utils.data import Dataset
import logging

from peft import (
    LoraConfig, 
    get_peft_model, 
)

import time


logging.basicConfig(
    level=logging.INFO, 
    filename=f"log/sft_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='w')


# some presets for the model

IGNORE_INDEX = -100
MAX_LENGTH = 512
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


class SFTDataset(Dataset):
    '''
    This is a custom Pytorch dataset that converts raw text to pytorch tensors

    '''
    def __init__(self, tokenizer, data, max_length=MAX_LENGTH):
        '''
        This initializes tokenizer and data and max_length
        '''

        self.tokenizer = tokenizer
        self.input = data
        self.max_length = max_length

    def accumulate(self, text):
        '''
        The function that takes care of tokenization
        '''
        tokens = self.tokenizer(
            text+self.tokenizer.eos_token, # adding the eos token at the end of every text
            return_tensors='pt', # convert to pytorch tensors
            truncation=True, # truncate every text sequence to the exact length of the text
            max_length=self.max_length, # the maximum sequence length as specified in the presets
        )

        input_ids = labels = tokens.input_ids[0] # because this is a causal model the labels are same as the input just one step ahead

        # the length of the input is equal to the sum of the length of the input id sequence except the pad tokens
        input_ids_lens = labels_lens = tokens.input_ids.ne(
            self.tokenizer.pad_token_id).sum().item()

        # returning a dictionary 
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def __getitem__(self, index):
        '''
        Error check that makes sure that the index is in range in the batch and calls the accumulate function
        '''
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")

        vals = self.input[index]
        vals = self.accumulate(vals)
        return vals
    
    
    def __len__(self):
        """Returns the length of the batch input"""
        return len(self.input)


"""
This is the native data collator of Huggingface that I customized a little
"""
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning. Takes care of batch processing."""

    # The tokenizer is actually present in the SFT dataset
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        
        # The instances comes from the SFT dataset and we take the input_ids and labels for each instance in a batch

        input_ids, labels = tuple(
            [instance[key].clone().detach() for instance in instances] 
                for key in ("input_ids", "labels")
        )

        # pad the sequences with the padding token for the input ids and IGNORE INDEX for the labels
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        '''input_ids (before):
        [101, 2023, 102]              (len=3)
        [101, 2023, 2003, 102]        (len=4)

        after pad_sequence:
        [[101, 2023, 102,   0],
        [101, 2023, 2003, 102]]
        
        '''


        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        '''
        labels (before):
        [101, 2023, 102]
        [101, 2023, 2003, 102]

        after pad_sequence:
        [[101, 2023, 102, -100],
        [101, 2023, 2003, 102]]
        '''

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id), ## here the attention masks is set to 0 for pad tokens
        )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict, 
    llama_tokenizer, 
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.

    Parameters:
        special_tokens_dict (dict): Dictionary of special tokens to add.
        llama_tokenizer (transformers.AutoTokenizer): Tokenizer to resize.
        model (transformers.AutoModelForCausalLM): Model to resize.

    Returns:
        None
    """
    num_new_tokens = llama_tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(llama_tokenizer))


    '''
    Logic:
    We are grabbing the input embeddings (the vector representations of input ids to vectors)
    &
    We are grabbing the output embeddings (the vector representations of hidden representation to logit over vocab)

    Now we are calculating to find the mean of the vector representations and we are assigning this mean weight to the new embeddings.
    This avoids the new token weights to be randomly sampled noise from any random distribution.
    '''
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

# helper function to load the data

def json_fileread(filepath:str):
    '''
    Params:
    filepath: str
        Path to the json file
    Returns:
    data: dict
        dict of index and string values
    '''
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def setup_datasets(data:list, tokenizer):

    '''
    This function splits the data into training and validation sets in a 80-20 split.
    Params:
    data: list
        List of strings
    tokenizer: transformers.AutoTokenizer
        Tokenizer object
    Returns:
    datasets: dict
        Dictionary containing training and validation datasets
    '''
    training_split = int(0.8*len(data))
    train_data = data[:training_split]
    val_data = data[training_split:]


    datasets = {
        "train": SFTDataset(
            tokenizer,
            train_data,
            max_length=MAX_LENGTH,
        ),
        "val": SFTDataset(
            tokenizer,
            val_data,
            max_length=MAX_LENGTH,
        ),
    }

    return datasets

def setup_training_args(args):
    '''
    Params:
    args: argparse.Namespace
        Arguments
    Returns:
    training_args: transformers.TrainingArguments
        Training arguments
    '''

    output_dir= args.expdir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # if the debug flag is on then disable wandb logging
    if args.debug:
        os.environ["WANDB_DISABLED"] = "True"
    # mixed precision is turned off for FSDP
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
    training_args = TrainingArguments(
        fsdp='full_shard', # the model is sharded across GPUs
        ddp_backend='nccl', # the backend for distributed data parallel
        gradient_checkpointing=False, # gradient checkpointing is off because I want to save time and not memory. Also FSDP does not support gradient checkpointing
        ddp_find_unused_parameters=False, # this is important for FSDP to work
        num_train_epochs=args.num_epochs, # number of epochs to train
        eval_steps=args.eval_freq, # evaluate after these many steps on the validation set
        save_steps=args.save_freq, # save the model after these many steps
        logging_steps=10, # log the training loss after these many steps
        evaluation_strategy="steps", # evaluation strategy is steps
        per_device_train_batch_size=args.batch_size, # batch size per GPU
        per_device_eval_batch_size=args.batch_size, # batch size per GPU for evaluation
        learning_rate=args.lr, # learning rate
        lr_scheduler_type=args.lr_scheduler, # learning rate scheduler type
        warmup_steps=args.num_warmup_steps, # number of warmup steps
        weight_decay=args.weight_decay, # weight decay
        gradient_accumulation_steps=args.grad_accum, # gradient accumulation steps to simulate larger batch size
        output_dir=output_dir, # output directory to save the model
        run_name=args.run_name, # run name for wandb
        report_to="wandb", # report to wandb
        dataloader_num_workers=8, # number of workers for dataloader
        remove_unused_columns=False, # this is important for FSDP to work
    )
    return training_args



def setup_model_and_tokenizer(model_name, lora_rank, lora_alpha, lora_dropout, load_in_8bit=True):

    '''
    Params:
    model_name: str
        Name of the model to be loaded from huggingface
    lora_rank: int
        Rank of the LoRA matrices
    lora_alpha: int
        Alpha value for LoRA
    lora_dropout: float
        Dropout value for LoRA
    load_in_8bit: bool
        Whether to load the model in 8 bit precision
    Returns:
    model: transformers.AutoModelForCausalLM
    tokenizer: transformers.AutoTokenizer
    '''

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        pad_token=DEFAULT_PAD_TOKEN,
        use_fast=False
    )

    quantization_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        config=quantization_config
    )
    
    # which modules I want my adapters to be applied to
    target_modules=[
                "q_proj",
                "v_proj",
                # "k_proj",
                # "o_proj",
                # "gate_proj",
                # "up_proj",
                # "down_proj",
                # "lm_head",
                ]
    
    '''
    W' = W + delta_W
    delta_W = B @ A
    4096 x 4096 = 4096 x r @ r x 4096
    alpha is used to scale the update
    delta_W = (alpha/r) * B @ A

    lora_rank = how wide your adapters are (capacity).
    lora_alpha = how strongly they affect the model (scaling).
    '''
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # creating the peft model by wrapping the original model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # adding special tokens if they are not already present
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # resize the tokenizer and the model embeddings
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        llama_tokenizer=tokenizer,
        model=model,
    )

    return model, tokenizer



def setup_trainer(model, 
                  tokenizer, 
                  training_args, 
                  datasets
                  ):
    
    '''
    Params:
    model: transformers.AutoModelForCausalLM
        Model object
    tokenizer: transformers.AutoTokenizer
        Tokenizer object
    training_args: transformers.TrainingArguments
        Training arguments
    datasets: dict
        Dictionary containing training and validation datasets
    Returns:
    trainer: transformers.Trainer
        Trainer object
    '''
    

    
    # creating the data collator object that will take care of batching and padding the sequences to the maximum length in a batch
    
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer, 
    )

    # creating the trainer object that will take care of the training loop
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        data_collator=data_collator,
    )

    return trainer

def main(args):

    # loading the data from json files
    data = json_fileread(args.datapath)

    
    # setting up the training arguments
    training_args = setup_training_args(args)
    
    # getting the model and tokenizer from huggingface
    model, tokenizer = setup_model_and_tokenizer(args.model_name, args.lora_rank, args.lora_alpha, args.lora_dropout, args.load_in_8bit)
    
    # setting up the datasets for training
    datasets = setup_datasets(data, tokenizer)
    
    # setting up the trainer
    trainer = setup_trainer(model, tokenizer, training_args, datasets)


    logging.info(f"Model: {args.model_name}")
    logging.info(f"Rank: {args.lora_rank}")
    logging.info(f"Alpha: {args.lora_alpha}")
    logging.info(f"Dropout: {args.lora_dropout}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Learning rate: {args.lr}")
    logging.info(f"Weight decay: {args.weight_decay}")
    logging.info(f"Num epochs: {args.num_epochs}")
    logging.info(f"Grad accum: {args.grad_accum}")
    logging.info(f"Eval freq: {args.eval_freq}")
    logging.info(f"Save freq: {args.save_freq}")
    logging.info(f"Run name: {args.run_name}")
    logging.info(f"Exp dir: {args.expdir}")
    logging.info(f"Debug: {args.debug}")
    logging.info(f"FP8: {args.fp8}")
    logging.info(f"Num warmup steps: {args.num_warmup_steps}")
    logging.info(f"LR scheduler: {args.lr_scheduler}")
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Data: {args.data_path}")
    logging.info(f"Resume dir: {args.resume_dir}")
    
    if args.resume_dir is not None:
        trainer.train(resume_from_checkpoint=True)
    else:        
        trainer.train()
        

if __name__== '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--expdir", type=Path)
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--load_in_8bit", action="store_true", default=True)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", type=str, default="cosine")
    parser.add_argument("--num-warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-freq", default=500, type=int)
    parser.add_argument("--save-freq", default=500, type=int)
    parser.add_argument("--w-attributes", type=int, default=1)
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    start_time = time.time()
    main(args)

    logging.info(f"Total time taken: {time.time()-start_time}")
    

