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

from peft import LoraConfig, AutoPeftModelForCausalLM

import sys

import editdistance

from tqdm import tqdm

import time

from common_utils import *

from prompt import *

from stop_words_utils import *

from unsloth import FastLanguageModel, is_bfloat16_supported

from human_eval.data import write_jsonl, read_problems

def run_math_sample(args=None, dataset=None, iteration_idx = 1):
    def preprocess_function(sample):
        # add prefix to the input for slm
        model_inputs = tokenizer([instruction_for_math.format(question, "") for question in sample['question']], truncation=False)

        if iteration_idx > 1:
            exist_answers = [answer for answer in sample['sample_answer']]
            exist_match = [match for match in sample['sample_match']]
            exist_ans = [ans for ans in sample['sample_ans']]

            model_inputs['exist_answer'] = exist_answers
            model_inputs['exist_match'] = exist_match
            model_inputs['exist_ans'] = exist_ans

        return model_inputs

    def collate_fn(batch):
        input_ids = [torch.tensor(example['input_ids']) for example in batch]
        attention_mask = [torch.tensor(example['attention_mask']) for example in batch]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=tokenizer.pad_token_id)
        
        if iteration_idx > 1:
            exist_answers = [example['exist_answer'] for example in batch]
            exist_match = [example['exist_match'] for example in batch]
            exist_ans = [example['exist_ans'] for example in batch]
            
            return input_ids, attention_mask, exist_answers[0], exist_match[0], exist_ans[0]
        
        return input_ids, attention_mask
    
    if args.is_unsloth == False:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )

        model = AutoModelForCausalLM.from_pretrained(
            args.model_checkpoint,
            device_map="auto",
            # quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    else:
        print("is_bfloat16_supported:", is_bfloat16_supported())

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = args.model_checkpoint,
            max_seq_length = 1024,
            dtype = None,
            load_in_4bit = False,
        )

        FastLanguageModel.for_inference(model) # 2x faster inference

    if args.model_name in ['TinyLlama/TinyLlama_v1.1', 'microsoft/phi-1_5']:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["question", "answer"])

    dataloader = DataLoader(tokenized_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

    pred_match = []
    pred_solution = []
    pred_ans = []
    pred_elapsed = []

    label_answer = [sample["answer"] for sample in dataset]
    label_ans = batch_find_ans(label_answer)
   
    model.eval()
    with torch.no_grad():
        test_data = tqdm(dataloader, total=dataloader.__len__(), file=sys.stdout)

        for line, gold_ans in zip(test_data, label_ans):
            num_columns = len(line)
            
            cur_pos_pred_solution = []
            cur_pos_pred_match = []
            cur_pos_pred_ans = []

            cur_neg_pred_solution = []
            cur_neg_pred_match = []
            cur_neg_pred_ans = []

            if iteration_idx == 1 and num_columns == 2:
                input_ids, attention_mask = line
                cur_pred_solution = []
                cur_pred_match = []
                cur_pred_ans = []
                sample_pos_num = args.sample_pos_num
                sample_neg_num = args.sample_neg_num

            elif iteration_idx > 1 and num_columns == 5:
                input_ids, attention_mask, cur_pred_solution, cur_pred_match, cur_pred_ans = line
                cur_pred_ans = batch_find_ans(cur_pred_solution)

                for pd_solution, pd_match, pd_ans in zip(cur_pred_solution, cur_pred_match, cur_pred_ans):
                    try:
                        if float(pd_ans) == float(gold_ans):
                            cur_pos_pred_solution.append(pd_solution)
                            cur_pos_pred_match.append(pd_match)
                            cur_pos_pred_ans.append(pd_ans)
                        else:
                            cur_neg_pred_solution.append(pd_solution)
                            cur_neg_pred_match.append(pd_match)
                            cur_neg_pred_ans.append(pd_ans)
                    except Exception as e:
                        print(e)
                
                sample_pos_num = len(cur_pos_pred_solution)/2 if len(cur_pos_pred_solution) < 8 else int(len(cur_pos_pred_solution)/4)
                sample_neg_num = len(cur_neg_pred_solution)/2 if len(cur_neg_pred_solution) < 8 else int(len(cur_neg_pred_solution)/4)
                sample_pos_num = sample_pos_num + 1 if sample_pos_num == 0 else sample_pos_num
                sample_neg_num = sample_neg_num + 1 if sample_neg_num == 0 else sample_neg_num

            input_ids = input_ids.repeat(args.sample_batch, 1)
            attention_mask = attention_mask.repeat(args.sample_batch, 1)

            input_ids = input_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')

            cur_pos_sample_num = 0
            cur_neg_sample_num = 0
            # start time
            start_time = time.time()
            while cur_pos_sample_num<sample_pos_num and cur_pos_sample_num<sample_pos_num:
                output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=512, do_sample = True, top_p=1, temperature=1)
                output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                
                fiter_pos_output_texts = []
                fiter_pos_output_ans = []

                fiter_neg_output_texts = []
                fiter_neg_output_ans = []

                for otext in output_texts:
                    fiter_text = otext.split("### Answer:")[-1].strip()
                    if fiter_text.startswith("<"):
                        fiter_text = re.sub(r'^<[^>]*>', '', fiter_text, count=1)
                    # fiter_text = fiter_text.lstrip("<built-in function input>")
                    decode_pred_ans = batch_find_ans([fiter_text])
                    if decode_pred_ans[0] == '[invalid]':
                        continue
                    try:
                        if float(decode_pred_ans[0]) == float(gold_ans):
                            fiter_pos_output_texts.append(fiter_text)
                            fiter_pos_output_ans.append(decode_pred_ans[0])
                        elif float(decode_pred_ans[0]) != float(gold_ans):
                            fiter_neg_output_texts.append(fiter_text)
                            fiter_neg_output_ans.append(decode_pred_ans[0])
                        
                    except Exception as e:
                        print(e)

                for p_ans, p_solution in zip(fiter_pos_output_ans, fiter_pos_output_texts):
                    cur_pos_pred_solution, cur_pos_pred_match, add_fag = reasoning_path_selection(cur_pos_pred_solution, cur_pos_pred_match, p_solution)
                    if add_fag == True:
                        cur_pos_pred_ans.append(p_ans)
                        cur_pos_sample_num = cur_pos_sample_num + 1
                
                for n_ans, n_solution in zip(fiter_neg_output_ans, fiter_neg_output_texts):
                    cur_neg_pred_solution, cur_neg_pred_match, add_fag = reasoning_path_selection(cur_neg_pred_solution, cur_neg_pred_match, n_solution)
                    if add_fag == True:
                        cur_neg_pred_ans.append(n_ans)
                        cur_neg_sample_num = cur_neg_sample_num + 1
                
                # end time
                end_time = time.time()

                # sampling time
                elapsed_time = end_time - start_time
                if elapsed_time > args.sample_time:
                    print(f"elapsed time out: {elapsed_time:.4f} s")
                    break
            
            add_pos = 0
            add_neg = 0 
            for pd_solution, pd_match, pd_ans in zip(cur_pos_pred_solution, cur_pos_pred_match, cur_pos_pred_ans):
                if pd_solution not in cur_pred_solution:
                    cur_pred_solution.append(pd_solution)
                    cur_pred_match.append(pd_match)
                    cur_pred_ans.append(pd_ans)
                    add_pos = add_pos + 1

            for pd_solution, pd_match, pd_ans in zip(cur_neg_pred_solution, cur_neg_pred_match, cur_neg_pred_ans):
                if pd_solution not in cur_pred_solution:
                    cur_pred_solution.append(pd_solution)
                    cur_pred_match.append(pd_match)
                    cur_pred_ans.append(pd_ans)
                    add_neg = add_neg + 1

            pred_solution.append(cur_pred_solution)
            pred_match.append(cur_pred_match)
            pred_ans.append(cur_pred_ans)
            pred_elapsed.append(elapsed_time > args.sample_time)

    save_res = [{"question": sample["question"], "answer": sample["answer"]} for sample in dataset]

    for res, p_ans, cur_solutions, g_ans, matchs, elapsed in zip(save_res, pred_ans, pred_solution, label_ans, pred_match, pred_elapsed):
        res['final_ans'] = g_ans
        res['sample_ans'] = p_ans
        res['sample_answer'] = cur_solutions
        res['sample_match'] = matchs
        res['sample_elapsed'] = elapsed

    os.makedirs(args.output_dir, exist_ok=True)
    print("save sampling res to: "+args.evalute_res_path)
    with open(args.evalute_res_path, "w", encoding="utf-8") as json_file:
        json.dump(save_res, json_file, ensure_ascii=False)

def run_code_sample(args, dataset, iteration_idx = 1):
    def preprocess_function(sample):
        # add prefix to the input for slm
        model_inputs = tokenizer([prompt+"\n" for prompt in sample['prompt']], truncation=False)
        
        model_inputs['prompt'] = [prompt+"\n" for prompt in sample['prompt']]

        if iteration_idx > 1:
            exist_answers = [answer for answer in sample['sample_answer']]
            exist_answer_codes = [answer for answer in sample['sample_answer_code']]
            exist_ans = [ans for ans in sample['sample_ans']]

            model_inputs['exist_answer'] = exist_answers
            model_inputs['exist_answer_code'] = exist_answer_codes
            model_inputs['exist_ans'] = exist_ans

        return model_inputs

    def collate_fn(batch):
        input_ids = [torch.tensor(example['input_ids']) for example in batch]
        attention_mask = [torch.tensor(example['attention_mask']) for example in batch]
        prompt = [example['prompt'] for example in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=tokenizer.pad_token_id)

        if iteration_idx > 1:
            exist_answers = [example['exist_answer'] for example in batch]
            exist_answer_codes = [example['exist_answer_code'] for example in batch]
            exist_ans = [example['exist_ans'] for example in batch]
            
            return input_ids, attention_mask, prompt[0], exist_answers[0], exist_answer_codes[0], exist_ans[0]
        
        return input_ids, attention_mask, prompt[0]
    
    if args.is_unsloth == False:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )

        model = AutoModelForCausalLM.from_pretrained(
            args.model_checkpoint,
            device_map="auto",
            use_cache=True,
            # quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        print("is_bfloat16_supported:", is_bfloat16_supported())
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = args.model_checkpoint, 
            max_seq_length = 1024,
            dtype = None,
            load_in_4bit = False,
        )

        FastLanguageModel.for_inference(model) # 2x faster inference

    if args.model_name in ['TinyLlama/TinyLlama_v1.1', 'microsoft/phi-1_5']:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

    dataloader = DataLoader(tokenized_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

    start_idx = 1 if args.model_name == 'microsoft/phi-1_5' else 2
    stopping_criteria = set_stop_words(tokenizer=tokenizer, stop_words=["\'\'\'", "assert", "if __name__ ==", "```", "print"], start_idx=start_idx)

    pred_solution = []
    pred_code = []
    pred_ans = []
    pred_elapsed = []
   
    model.eval()
    with torch.no_grad():
        test_data = tqdm(dataloader, total=dataloader.__len__(), file=sys.stdout)

        min_sample_num = args.sample_num*2
        for line in test_data:
            num_columns = len(line)
        
            cur_pred_fiter_solution = []
            if num_columns == 3 and iteration_idx == 1:
                input_ids, attention_mask, prompt = line
                cur_pred_solution = []
                cur_pred_code = []
                cur_pred_ans = []
                sample_num = args.sample_num

            elif num_columns == 6 and iteration_idx > 1:
                input_ids, attention_mask, prompt, cur_pred_solution, cur_pred_code, cur_pred_ans = line
            
                for cpcode in cur_pred_code:
                    cpcode = cpcode.replace(" ", "").replace("\t", "").replace("\n", "").replace("\r", "")
                    cur_pred_fiter_solution.append(cpcode)

                sample_num = len(cur_pred_solution)/2 if len(cur_pred_solution) < 40 else int(len(cur_pred_solution)/4)
                sample_num = 10 if sample_num == 0 else sample_num
            
            input_ids = input_ids.repeat(args.sample_batch, 1)
            attention_mask = attention_mask.repeat(args.sample_batch, 1)

            input_ids = input_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')
                
            cur_sample_num = 0
            start_time = time.time()
            while cur_sample_num<sample_num:
                output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=750, stopping_criteria=stopping_criteria, do_sample = True, top_p=1, temperature=0.7)
                output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                
                for solution in output_texts:
                    fiter_solution = solution[len(prompt):]
                    fiter_solution = fiter_solution.split("if __name__ ==")[0].strip()
                    fiter_solution = fiter_solution.split("print")[0].strip()
                    fiter_solution = fiter_solution.split("assert")[0].strip()
                    fiter_solution = fiter_solution.split("```")[0].strip()
                    fiter_solution = fiter_solution.split("\'\'\'")[0].strip()
                    if fiter_solution.startswith("mport "):
                        fiter_solution = "i" + fiter_solution
                    if fiter_solution.startswith("rom "):
                        fiter_solution = "f" + fiter_solution
                    if fiter_solution.startswith("ef "):
                        fiter_solution = "d" + fiter_solution
                    if fiter_solution.startswith("lass "):
                        fiter_solution = "c" + fiter_solution 
                    solution = prompt + fiter_solution
                    
                    pcode = fiter_solution

                    split_fiter_solution = fiter_solution.split("\n")

                    if len(split_fiter_solution) == 0:
                        continue
                    try:
                        sum_len = 0
                        sp_cnt = 0
                        for sp_solution in split_fiter_solution:
                            sp_solution = sp_solution.replace(" ", "").replace("\t", "").replace("\n", "").replace("\r", "")
                            if len(sp_solution) == 0:
                                continue
                            sum_len += len(sp_solution)
                            sp_cnt = sp_cnt + 1
                        avg_len = sum_len/sp_cnt + 1

                        fiter_solution = fiter_solution.replace(" ", "").replace("\t", "").replace("\n", "").replace("\r", "")
                        cur_pred_fiter_solution, add_fag = add_string_with_distance_check(cur_pred_fiter_solution, fiter_solution, bound=avg_len)
                    except Exception as e:
                        print(e)
                        continue
            
                    if add_fag == True:
                        cur_pred_solution.append(solution)
                        cur_pred_code.append(pcode)
                        cur_pred_ans.append("UnKnowed") #Further assessment is conducted through unit test
                        cur_sample_num = cur_sample_num + 1
                        print(solution)
                        print("-------------------------------")

                print("cur_sample_num:", cur_sample_num)

                # End Time
                end_time = time.time()

                # Sampling Time
                elapsed_time = end_time - start_time
                if elapsed_time > args.sample_time:
                    print(f"elapsed time out: {elapsed_time:.4f} s")
                    break

                min_sample_num = min(min_sample_num, cur_sample_num)
                
            pred_solution.append(cur_pred_solution)
            pred_code.append(cur_pred_code)
            pred_ans.append(cur_pred_ans)
            pred_elapsed.append(elapsed_time > args.sample_time)

    save_res = [{"task_id": sample['task_id'], "text": sample["text"], "code": sample["code"],\
                 "test_list": sample['test_list'], "test": sample['test'], "prompt": sample['prompt']}\
                 for sample in dataset]

    example_task_id = [{"example_idx": idx, "task_id": sample['task_id']} for idx, sample in enumerate(dataset)]

    print("min_sample_num:", min_sample_num)

    sample_res = []
    for idx, cs_solution in zip(example_task_id, pred_solution):
        for c_solution in cs_solution:
            sample_res.append(dict(example_idx = idx['example_idx'], task_id=idx['task_id'], completion=c_solution))

    print("save predict res to: "+args.evalute_res_path+"l")
    write_jsonl(args.evalute_res_path+"l", sample_res)

    for idx, (res, p_ans, cur_solutions, cur_codes, elapsed) in enumerate(zip(save_res, pred_ans, pred_solution, pred_code, pred_elapsed)):
        res['example_idx'] = idx
        res['sample_answer'] = cur_solutions
        res['sample_answer_code'] = cur_codes
        res['sample_ans'] = p_ans
        res['sample_elapsed'] = elapsed

    os.makedirs(args.output_dir, exist_ok=True)
    print("save predict res to: "+args.evalute_res_path)
    with open(args.evalute_res_path, "w", encoding="utf-8") as json_file:
        json.dump(save_res, json_file, ensure_ascii=False)

def run(args):
    sampling_dataset = load_dataset("json", data_files=args.sampling_dataset_pth, split="train")

    sampling_dataset = sampling_dataset.select(range(10))

    print(sampling_dataset[0])

    print(f"Size of dataset_test: {len(sampling_dataset)}")

    model_and_task = args.model_name.split("/")[-1].strip()+"_"+args.task_name+"_"+args.task_type+"_iter"+str(args.iteration_idx)
    args.evalute_res_path = os.path.join(args.output_dir, args.output_file_name+"_"+model_and_task+".json")
    args.log_file_path = args.output_dir + model_and_task + ".log"

    if args.task_type == 'math':
        run_math_sample(args, sampling_dataset, iteration_idx=args.iteration_idx)
    elif args.task_type == 'code':
        run_code_sample(args, sampling_dataset, iteration_idx=args.iteration_idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="self sampling")

    parser.add_argument("--sampling_dataset_pth", default='', help="dataset_test's path")
    parser.add_argument("--task_name", default='gsm8k', help="mbpp or gsm8k")
    parser.add_argument("--task_type", default='math', help="math or code")
    parser.add_argument("--iteration_idx", default=2, help="The idx of iteration")
    parser.add_argument("--is_unsloth", default=True, help="whether use unsloth")
    parser.add_argument('--sample_time', default=30)
    parser.add_argument('--sample_batch', default=30)
    parser.add_argument('--sample_num', default=50, help='This parameter becomes useless after the first iteration. (only for mbpp)')
    parser.add_argument('--sample_pos_num', default=4, help='This parameter becomes useless after the first iteration. (only for gsm8k)')
    parser.add_argument('--sample_neg_num', default=4, help='This parameter becomes useless after the first iteration. (only for gsm8k)')
    parser.add_argument("--model_name", default='google/gemma-2b', help="model name")
    parser.add_argument("--seed", default=42, help="set seed")
    parser.add_argument("--model_checkpoint", default='', help="model checkpoint's path")
    parser.add_argument("--output_file_name", default="sample_res", help="output file's name")
    parser.add_argument("--output_dir", default="save_res", help="output file's dir")

    args = parser.parse_args()
     
    run(args)