import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import random, json, re
from random import randrange, sample
import argparse
import os

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, GenerationConfig

from trl import SFTTrainer

from peft import LoraConfig, AutoPeftModelForCausalLM

import sys

from tqdm import tqdm

from stop_words_utils import *

from common_utils import *

from prompt import *

def run_sft_train(args=None, dataset=None):
    def formatting_prompts_func(examples, task_type = args.task_type):
        if task_type == 'math':
            inputs       = examples["question"]
            outputs      = examples["answer"]
        else:
            inputs       = examples["prompt"]
            outputs      = examples["code"]
        texts = []
        for input, output in zip(inputs, outputs):
            if task_type == 'math':
                text = instruction_for_math.format(input, output) + EOS_TOKEN
            else:
                text = input + "\n" + output + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }
    
    if args.is_unsloth == False or args.model_name == 'microsoft/phi-1_5':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            use_cache=False,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        # model.config.pretraining_tp = 1 # useful for llama

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        peft_config = LoraConfig(
            lora_alpha=128,
            lora_dropout=0.05,
            r=256,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM", 
        )
    
    else:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = args.model_name,
            max_seq_length = 1024,
            dtype = None,
            load_in_4bit = True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r = args.r,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            # target_modules="all-linear",
            lora_alpha = args.lora_alpha,
            lora_dropout = args.lora_dropout, 
            bias = "none",    
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = args.seed,
            max_seq_length = 1024,
            use_rslora = False,  
            loftq_config = None, 
        )

        peft_config = None

    if args.model_name in ['TinyLlama/TinyLlama_v1.1', 'microsoft/phi-1_5']:
       tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    EOS_TOKEN = tokenizer.eos_token 

    dataset = dataset.map(formatting_prompts_func, batched = True)

    print("The processed example:")
    print(dataset[0])

    training_args = TrainingArguments(
        seed=args.seed,
        data_seed=args.seed,
        output_dir=args.output_model_path,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        learning_rate=args.lr,
        fp16 = False,
        bf16 = True,
        tf32 = True,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warm_up_radio,
    )

    trainer = SFTTrainer(
        model=model,
        peft_config = peft_config,
        dataset_num_proc = 2,
        dataset_text_field = "text",
        train_dataset=dataset,
        max_seq_length=1024,
        tokenizer=tokenizer,
        packing=True,
        args=training_args,
    )

    trainer.train()

def run(args):
    data_train_pth = args.data_train_pth

    set_seed(args.seed)
 
    dataset_train = load_dataset("json", data_files=data_train_pth, split="train")

    print("train example:")
    print(dataset_train[0])

    print(f"Size of dataset_train: {len(dataset_train)}")

    model_and_task = args.model_name.split("/")[-1].strip()+"_sft_"+args.task_name
    args.output_model_path = os.path.join(args.output_model_path, model_and_task)
    
    run_sft_train(args, dataset_train)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="sft code")

    # Add your parameters 
    parser.add_argument("--data_train_pth", default='./dataset/train_dataset_gsm8k_hn.json', help="dataset_train's path")
    parser.add_argument("--data_valid_pth", default='', help="dataset_test's path")
    parser.add_argument("--task_name", default='gsm8k', help="gsm8k or mbpp")
    parser.add_argument("--task_type", default='math', help="math or code")
    parser.add_argument("--is_unsloth", default=True, help="whether use unsloth")
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--max_grad_norm', type=float, default=0.3)
    parser.add_argument('--warm_up_radio', type=float, default=0.1)
    parser.add_argument('--num_train_epochs', default=8)
    parser.add_argument("--lr_scheduler_type", default='linear', help="lr_scheduler_type")
    parser.add_argument('--log_file_dir', type=str, default='./', help="log file's dir")
    parser.add_argument('--output_model_path', type=str, default='./model', help="model checkpoint's save path")
    parser.add_argument('--lora_dropout', default=0.05, help='peft_config')
    parser.add_argument('--lora_alpha', default=128, help='peft_config')
    parser.add_argument('--r', default=256, help='peft_config')
    parser.add_argument('--optim', default='adamw_torch_fused')
    parser.add_argument('--logging_steps', default=30, help='logging steps')
    parser.add_argument('--save_strategy', default='epoch', help='save strategy')
    parser.add_argument('--gradient_accumulation_steps', default=10, help='gradient_accumulation_steps')
    parser.add_argument("--model_name", default='google/gemma-2b', help="model name")
    parser.add_argument("--seed", default=42, help="set seed")
    parser.add_argument("--model_checkpoint", default='', help="model checkpoint's path")
    args = parser.parse_args()
    
    run(args)