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

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments

from trl import ORPOConfig, ORPOTrainer, DPOTrainer

from peft import LoraConfig, AutoPeftModelForCausalLM

import sys

from tqdm import tqdm

from common_utils import *

from prompt import *

from stop_words_utils import *

from numpy import percentile

from unsloth import FastLanguageModel, is_bfloat16_supported

def run_orpo_align(args=None, dataset=None):
    def formatting_prompts_func(sample):
        instruction = sample["prompt"]
        accepted    = sample["chosen"]
        rejected    = sample["rejected"]

        if args.task_name == 'gsm8k':
            sample["prompt"]   = instruction_for_math.format(instruction, input, "").rstrip("<built-in function input>")
        else:
            sample['prompt'] = instruction + "\n"
        sample["chosen"]   = accepted + EOS_TOKEN
        sample["rejected"] = rejected + EOS_TOKEN
        return sample
    
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
        # model.config.pretraining_tp = 1

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.r,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM", 
        )
    
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = args.model_name,
            max_seq_length = args.max_seq_length,
            dtype = None,
            load_in_4bit = True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r = args.r,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = args.lora_alpha,
            lora_dropout = args.lora_dropout, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = args.seed,
            max_seq_length = args.max_seq_length,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )

        peft_config = None

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    if args.model_name in ['TinyLlama/TinyLlama_v1.1', 'microsoft/phi-1_5']:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    dataset = dataset.map(formatting_prompts_func)

    print("The processed example:")
    print(dataset[0])

    training_args = ORPOConfig(
        seed=args.seed,
        data_seed=args.seed,
        output_dir=args.output_model_path,        
        num_train_epochs=args.num_train_epochs,   # number of training epochs
        per_device_train_batch_size=args.per_device_train_batch_size,        # batch size per device during training
        do_eval=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,        
        beta=args.beta,                           # reference weight
        gradient_checkpointing=True,              # use gradient checkpointing to save memory
        optim=args.optim,                         # use fused adamw optimizer
        learning_rate=args.lr,                    
        max_grad_norm=args.max_grad_norm,         
        warmup_ratio=args.warmup_ratio,           
        lr_scheduler_type=args.lr_scheduler_type,   
        max_length=args.max_seq_length,
        max_prompt_length=args.prompt_length,
        logging_steps=args.logging_steps,        # log steps
        save_strategy=args.save_strategy,
        fp16 = False,
        bf16 = True,
        tf32 = True,                             # use tf32 precision
    )

    trainer = ORPOTrainer(
        model,
        peft_config=peft_config,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    # start training, the model will be automatically saved to the output directory
    trainer.train()

def run(args):
    data_train_pth = args.data_train_pth

    set_seed(args.seed)
 
    dataset_train = load_dataset("json", data_files=data_train_pth, split="train")

    print("train example:")
    print(dataset_train[0])

    print(f"Size of dataset_train: {len(dataset_train)}")

    model_and_task = args.model_name.split("/")[-1].strip()+"_sipf_"+args.task_name+"_iter"+str(args.iteration_idx)
    args.output_model_path = os.path.join(args.output_model_path, model_and_task)

    run_orpo_align(args, dataset_train)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="sipf_train code")

    parser.add_argument("--data_train_pth", default='./dataset/gsm8k_gemma_refer_pairs_iter0.jsonl', help="dataset_train's path")
    parser.add_argument("--task_name", default='gsm8k', help="gsm8k or mbpp")
    parser.add_argument("--task_type", default='math', help="math or code")
    parser.add_argument("--iteration_idx", default=1, help="1, 2, 3, or other")
    parser.add_argument("--is_unsloth", default=True, help="whether use unsloth")
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--sample_batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=8e-5)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=0.3)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--num_train_epochs', default=8)
    parser.add_argument('--prompt_length', default=300)
    parser.add_argument('--max_seq_length', default=1000)
    parser.add_argument("--lr_scheduler_type", default='linear', help="lr_scheduler_type")
    parser.add_argument('--log_file_dir', type=str, default='./', help="log file's dir")
    parser.add_argument('--output_model_path', type=str, default='./model', help="model checkpoint's save path")
    parser.add_argument('--lora_dropout', default=0.05, help='peft_config')
    parser.add_argument('--lora_alpha', default=128, help='peft_config')
    parser.add_argument('--r', default=256, help='peft_config')
    parser.add_argument('--optim', default='adamw_torch_fused', help='adamw_torch_fused, adamw_8bit, or other')
    parser.add_argument('--logging_steps', default=50, help='logging steps')
    parser.add_argument('--save_strategy', default='epoch', help='save strategy')
    parser.add_argument('--gradient_accumulation_steps', default=10, help='gradient_accumulation_steps')
    parser.add_argument("--model_name", default='google/gemma-2b', help="model name")
    parser.add_argument("--seed", default=42, help="set seed")
    parser.add_argument("--model_checkpoint", default='', help="model checkpoint's path")

    args = parser.parse_args()
     
    run(args)