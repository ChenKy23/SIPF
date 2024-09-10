import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import random, json, re
from random import randrange, sample
import argparse
import os, sys

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from transformers import AutoTokenizer, AutoConfig, AutoModel, LlamaModel, BertForSequenceClassification, BertForTokenClassification, GPT2ForTokenClassification, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AdamW, get_linear_schedule_with_warmup

from common_utils import *

from prompt import *

from prm import *

def run(args):
    set_seed(args.seed)

    dataset_train =  read_jsonl(args.data_train_pth)[0]
    dataset_predict = read_jsonl(args.data_pred_pth)[0][:10]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.model_name in ['deepseek-ai/deepseek-math-7b-instruct', 'deepseek-ai/deepseek-math-7b-rl']:
        tokenizer.pad_token = tokenizer.eos_token

    if args.toe == "train" or args.toe == "test":
        if args.task_type == 'math' and args.task_name == 'gsm8k':
            selected_dataset_test = []
            spilt_dataset = [dataset_train[i:i + 100] for i in range(0, len(dataset_train), 100)]
            for piece in spilt_dataset:
                selected_dataset_test += random.sample(piece, 8)
                
            selected_dataset_train = [x for x in dataset_train if x not in selected_dataset_test]

            dataset_train = Dataset.from_dict(trans_dict_math_train_pro(selected_dataset_train))
            dataset_test = Dataset.from_dict(trans_dict_math_train_pro(selected_dataset_test))

            datasets = DatasetDict()
            datasets['train'] = dataset_train
            datasets['test'] = dataset_test

            tokenized_datasets = datasets.map(preprocess_function_prm_math_train, fn_kwargs={'tokenizer': tokenizer}, batched=False, remove_columns=datasets['test'].features)

        elif args.task_type == 'code' and args.task_name == 'mbpp':
            selected_dataset_test = []
            spilt_dataset = [dataset_train[i:i + 100] for i in range(0, len(dataset_train), 100)]
            for piece in spilt_dataset[:-1]:
                selected_dataset_test += random.sample(piece, 8)
            
            selected_dataset_train = [x for x in dataset_train if x not in selected_dataset_test]

            dataset_train = Dataset.from_dict(trans_dict_code_train_pro(selected_dataset_train))
            dataset_test = Dataset.from_dict(trans_dict_code_train_pro(selected_dataset_test))

            datasets = DatasetDict()
            datasets['train'] = dataset_train
            datasets['test'] = dataset_test

            tokenized_datasets = datasets.map(preprocess_function_prm_code_train, fn_kwargs={'tokenizer': tokenizer}, batched=False, remove_columns=datasets['train'].features)

        print(dataset_train[0])
        print(dataset_test[0])

    else:
        if args.task_type == 'math' and args.task_name == 'gsm8k': 
            dataset_predict = Dataset.from_dict(trans_dict_math_pred(dataset_predict))
            tokenized_datasets = dataset_predict.map(preprocess_function_prm_math_pred, fn_kwargs={'tokenizer': tokenizer}, batched=False, remove_columns=dataset_predict.features)
        elif args.task_type == 'code' and args.task_name == 'mbpp':
            dataset_predict = Dataset.from_dict(trans_fict_code_pred(dataset_predict))
            tokenized_datasets = dataset_predict.map(preprocess_function_prm_code_pred, fn_kwargs={'tokenizer': tokenizer}, batched=False, remove_columns=dataset_predict.features)
        
        print(dataset_predict[0])

    model_and_task = args.model_name.split("/")[-1].strip()+"_"+args.task_name+"_"+args.task_type
    args.output_model_path = os.path.join(args.output_model_path, model_and_task)
    args.evalute_res_path = os.path.join(args.output_dir, args.output_file_name+"_"+model_and_task+".json")
    args.log_file_path = args.output_dir + model_and_task + ".log"

    if args.toe == "train":
        run_prm_train(args, tokenized_datasets)
    elif args.toe == "test":
        run_prm_evaluate(args, tokenized_datasets, datasets['test'])
    else:
        if args.task_name == 'gsm8k':
            run_gsm8k_prm_predict(args, tokenized_datasets, dataset_predict)
        elif args.task_name == 'mbpp':
            run_mbpp_prm_predict(args, tokenized_datasets, dataset_predict)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="prm score")

    parser.add_argument("--data_train_pth", default='', help="dataset_train's path")
    parser.add_argument("--data_pred_pth", default='', help="dataset_test's path")
    parser.add_argument("--task_name", default='gsm8k', help="gsm8k or mbpp")
    parser.add_argument("--task_type", default='math', help="math or code")
    parser.add_argument("--toe", default='train', help="train, test or predict")
    parser.add_argument('--per_device_train_batch_size', type=int, default=10)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_grad_norm', type=float, default=0.3)
    parser.add_argument('--warm_up_radio', type=float, default=0.1)
    parser.add_argument('--num_train_epochs', default=10)
    parser.add_argument("--load_checkpoint", default=False, help="whether to load from checkpoint")
    parser.add_argument('--log_file', type=str, default='')
    parser.add_argument('--output_model_path', type=str, default='./model')
    parser.add_argument('--weight_decay', default=0.01, action='store_true')
    parser.add_argument('--gradient_accumulation_steps', default=4, action='store_true')
    parser.add_argument("--model_name", default='deepseek-ai/deepseek-coder-6.7b-instruct', help="model name")
    parser.add_argument("--seed", default=42, help="set seed")
    parser.add_argument("--model_checkpoint", default='', help="model checkpoint's path")
    parser.add_argument("--output_file_name", default="prm_score", help="output file's name")
    parser.add_argument("--output_dir", default="save_res", help="output file's dir")

    args = parser.parse_args()
     
    run(args)