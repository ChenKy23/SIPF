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

from human_eval.data import write_jsonl, read_problems

from tqdm import tqdm

from stop_words_utils import *

from common_utils import *

from prompt import *

def run_math_evaluation(args=None, dataset_test=None):
    def collate_fn(batch, method_type = args.method_type):
        if method_type == 'sft':
            input_prompts = [instruction_for_math.format(example['question'], "") for example in batch]
        elif method_type == 'cot':
            input_prompts = [few_shot_prompt_for_math.format(question = example['question']) for example in batch]
        else:
            input_prompts = [example['question'] + prompt_for_deepseek_math for example in batch]
        return input_prompts
    
    dataloader = DataLoader(dataset_test, batch_size=args.per_device_eval_batch_size, collate_fn=collate_fn)

    model_path = args.model_checkpoint if args.method_type == 'sft' and args.model_checkpoint != '' else args.model_name

    print("load model from:", model_path)

    if args.is_unsloth == False:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # load_in_4bit=True,
            attn_implementation="flash_attention_2",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path, # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = 1024,
            dtype = None,
            load_in_4bit = False,
        )

        FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    
    if args.model_name in ['TinyLlama/TinyLlama_v1.1']:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    stopping_criteria = None
    if args.method_type == 'cot':
        model.generation_config = GenerationConfig.from_pretrained(args.model_name)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        stopping_criteria = set_stop_words(tokenizer=tokenizer, stop_words=["Q:"], start_idx=2)

    pred_solution = []
    pred_ans = []

    model.eval()
    with torch.no_grad():
        test_data = tqdm(dataloader, total=dataloader.__len__(), file=sys.stdout)

        for prompts in test_data:
            input_prompts = tokenizer(prompts, padding=True, truncation=False, return_tensors='pt')
            input_ids = input_prompts['input_ids'].to('cuda')
            attention_mask = input_prompts['attention_mask'].to('cuda')

            input_prompt_len = [len(prompt) for prompt in prompts]
            
            output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=512, stopping_criteria=stopping_criteria, do_sample = False, use_cache=True)
            output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
            fiter_output_texts = []
            for otext, in_prompt_len in zip(output_texts, input_prompt_len):
                otext = otext[in_prompt_len:].strip()
                if args.method_type == 'cot':
                    otext = otext.split("Q:")[0].strip()
                print(otext)
                print("---------------------------")
                # fiter_output_texts.append(otext.split("### Answer:")[-1].strip())
                fiter_output_texts.append(otext)

            decode_pred_ans = batch_find_ans(fiter_output_texts)

            for ans, solution in zip(decode_pred_ans, fiter_output_texts):
                pred_solution.append(solution)
                pred_ans.append(ans)

    if args.task_name == 'gsm8k':
        label_answer = [sample["answer"] for sample in dataset_test]
        label_ans = batch_find_ans(label_answer, "uni")
    elif args.task_name == 'mmlu_math':
        label_ans = [sample["answer"] for sample in dataset_test]

    save_res = [{"question": sample["question"], "gold_answer": sample["answer"]} for sample in dataset_test]

    count_equal_ans = 0
    for pred, label in zip(pred_ans, label_ans):
        label = label.replace(",","")
        try:
            if pred == "[invalid]":
                continue
            if float(eval(pred)) == float(label):
                count_equal_ans = count_equal_ans + 1
        except Exception as e:
            print(e)

    num_acc = round(count_equal_ans/len(label_ans)*100, 4)

    print(f"Num_acc: {num_acc}")

    for res, p_ans, solution, g_ans in zip(save_res, pred_ans, pred_solution, label_ans):
        res['final_ans'] = g_ans
        res['pred_ans'] = p_ans
        res['pred_answer'] = solution
        
    os.makedirs(args.output_dir, exist_ok=True)
    print("save predict res to: "+args.evalute_res_path)
    with open(args.evalute_res_path, "w", encoding="utf-8") as json_file:
        json.dump(save_res, json_file, ensure_ascii=False)

def run_code_evaluation(args=None, dataset_test=None):
    def collate_fn(batch):
        input_prompts = [example['prompt'] + "\n" for example in batch]
        if args.task_name == 'mbpp':
            function_names  = [extract_assert_expression(example['test_list'][0]).strip() for example in batch]
        elif args.task_name == 'humaneval':
            function_names  = [example['entry_point'] for example in batch]
        return input_prompts, function_names

    dataloader = DataLoader(dataset_test, batch_size=args.per_device_eval_batch_size, collate_fn=collate_fn, shuffle=False)

    model_path = args.model_checkpoint if args.method_type == 'sft' and args.model_checkpoint != '' else args.model_name

    print("load model from:", model_path)

    if args.is_unsloth == False or args.model_name in ['microsoft/phi-1_5']:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # load_in_4bit=True,
            attn_implementation="flash_attention_2",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path, # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = 1024,
            dtype = None,
            load_in_4bit = False,
        )

        FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    
    stopping_criteria = None
    if args.model_name in ['microsoft/phi-1_5'] or args.method_type == 'cot':
        tokenizer.pad_token = tokenizer.eos_token
        start_idx = 1
        if args.model_name == 'google/gemma-2b':
            start_idx = 2
        stopping_criteria = set_stop_words(tokenizer=tokenizer, stop_words=["assert", "print", "if __name__ ==", "```", "\'\'\'"], start_idx=start_idx)
    tokenizer.padding_side = "left"

    sample_solution = []
    model.eval()
    with torch.no_grad():
        test_data = tqdm(dataloader, total=dataloader.__len__(), file=sys.stdout)

        for idx, (prompts, fcns) in enumerate(test_data):
            input_prompt_len = [len(prompt) for prompt in prompts]
            input_prompts = tokenizer(prompts, padding=True, truncation=False, return_tensors='pt')
            input_ids = input_prompts['input_ids'].to('cuda')
            attention_mask = input_prompts['attention_mask'].to('cuda')
            output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, stopping_criteria=stopping_criteria, max_length=1024, do_sample = False)
            output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for p_len, fcn, otext in zip(input_prompt_len, fcns, output_texts):
                if args.model_name == 'microsoft/phi-1_5' or args.method_type == 'cot':
                    if args.task_name == 'humaneval':
                        otext = humaneval_code_text_processing(otext=otext, p_len=p_len)
                    if args.task_name == 'mbpp':
                        otext = otext[p_len:]
                        otext = mbpp_code_text_processing(otext=otext, fcn=fcn)

                print(otext)
                print("----------------------")
                sample_solution.append(otext)

    save_res = [{"task_id": sample["task_id"], "prompt": sample["prompt"], "test": sample['test']} for sample in dataset_test]
    
    task_id = [sample['task_id'] for sample in dataset_test]

    for res, solution in zip(save_res, sample_solution):
        res['sample_solution'] = solution

    sample_res = []
    for id, cs_solution in zip(task_id, sample_solution):
        sample_res.append(dict(task_id=id, completion=cs_solution))

    os.makedirs(args.output_dir, exist_ok=True)
    print("save predict res to: "+args.evalute_res_path+"l")
    write_jsonl(args.evalute_res_path+"l", sample_res)
    print("save predict res to: "+args.evalute_res_path)
    with open(args.evalute_res_path, "w", encoding="utf-8") as json_file:
        json.dump(save_res, json_file, ensure_ascii=False)

def run(args):
    data_test_pth = args.data_test_pth

    set_seed(args.seed)
 
    dataset_test = load_dataset("json", data_files=data_test_pth, split="train")

    print("test example:")
    print(dataset_test[0])

    print(f"Size of dataset_test: {len(dataset_test)}")

    model_and_task = args.model_name.split("/")[-1].strip()+"_"+args.task_name
    args.evalute_res_path = os.path.join(args.output_dir, args.output_file_name+"_"+model_and_task+".json")
    args.log_file_path = args.output_dir + model_and_task + ".log"

    if args.task_type == 'math':
        run_math_evaluation(args=args, dataset_test=dataset_test)
    elif args.task_type == 'code':
        run_code_evaluation(args=args, dataset_test=dataset_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="evaluate code")

    # Add your parameters 
    parser.add_argument("--data_test_pth", default='./dataset/test_dataset_gsm8k_hn.json', help="dataset_train's path")
    parser.add_argument("--task_name", default='gsm8k', help="gsm8k, mmlu_math, mbpp, or humaneval")
    parser.add_argument("--task_type", default='math', help="math or code")
    parser.add_argument("--method_type", default='sft', help="sft or cot")
    parser.add_argument("--is_unsloth", default=True, help="whether use unsloth")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=20)
    parser.add_argument('--log_file_dir', type=str, default='./', help="log file's dir")
    parser.add_argument("--model_name", default='google/gemma-2b', help="model name")
    parser.add_argument("--seed", default=42, help="set seed")
    parser.add_argument("--model_checkpoint", default='/home/chenkaiyuan/dpo_math/model/gemma-2b_gsm8k_orpo_iter3_b_1/checkpoint-17482', help="model checkpoint's path")
    parser.add_argument("--output_file_name", default="evaluate_res", help="output_file_name")
    parser.add_argument("--output_dir", default="save_res", help="output_dir")
    args = parser.parse_args()
    
    run(args)