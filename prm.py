import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AdamW, get_linear_schedule_with_warmup

from peft import LoraConfig, AutoPeftModelForCausalLM, AutoPeftModel, get_peft_model, TaskType, prepare_model_for_kbit_training

from tqdm.auto import tqdm

import logging

import sys

import json, os

from tqdm.auto import tqdm

import logging

from collections import defaultdict

from prompt import *

class LlamaModelForBinaryRegression(nn.Module):
    def __init__(self, args):
        super(LlamaModelForBinaryRegression, self).__init__()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ) if args.toe == "train" else None

        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=args.model_name if args.load_checkpoint == False else args.model_checkpoint,
            quantization_config=bnb_config,
            use_cache=args.load_checkpoint,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.num_labels = 2
        llama_model_config = AutoConfig.from_pretrained(args.model_name)

        if args.toe == "train":
            peft_config = LoraConfig(
                lora_alpha=32,
                lora_dropout=0.05,
                r=16,
                bias="none",
                target_modules="all-linear"
            )

            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)

            self.classifier = torch.nn.Linear(llama_model_config.hidden_size, self.num_labels).to('cuda:0')
        else:
            self.classifier = torch.nn.Linear(llama_model_config.hidden_size, self.num_labels, dtype=torch.bfloat16).to('cuda:0')

        self.llama_model = model
        # self.dropout = nn.Dropout(0.05)
        self.sig = nn.Sigmoid()
        
        if args.load_checkpoint == True and (args.toe == "test" or args.toe == 'predict'):
            self.classifier.load_state_dict(torch.load(args.model_checkpoint+'/classifier.pth'))
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.llama_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        hidden_states = self.sig(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        # logits = logits.squeeze(-1)  # [batch_size, seq_length]

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return {"loss": loss, "logits": logits}
        
        return {"logits": logits}
    
def preprocess_function_prm_math_train(sample, tokenizer):
    question_inputs = tokenizer(sample['question']+deepseek_math_prompt, truncation=False)
    question_input_ids = question_inputs['input_ids']
    question_mask = question_inputs['attention_mask']

    answer = sample['answer']
    mcts = sample['mcts']
    
    answer_split = answer.split('\n')
    answer_input_ids = []
    answer_attention_mask = []
    mcts_inputs_ids = []
    mcts_attention_mask = []

    for ans, fag in zip(answer_split, mcts):
        ans = ans+'\n'
        ans_inputs = tokenizer(ans, truncation=False)
        ans_inputs_ids = ans_inputs['input_ids'][1:] + [tokenizer.pad_token_id]
        ans_attention_mask = ans_inputs['attention_mask']

        answer_input_ids += ans_inputs_ids
        answer_attention_mask += ans_attention_mask
        mcts_inputs_ids += [-100] * (len(ans_inputs_ids)-1) + [int(fag > 0)]
        mcts_attention_mask += [1] * len(ans_attention_mask)

    model_input_ids = question_input_ids + answer_input_ids
    model_input_mask = question_mask + answer_attention_mask
    model_label_ids = [-100] * len(question_input_ids) + mcts_inputs_ids
    
    return {'model_input_ids': model_input_ids, 'model_input_mask': model_input_mask, 'model_label_ids': model_label_ids, \
            'question_input_ids': question_input_ids, 'question_mask': question_mask, 'answer_input_ids': answer_input_ids, \
            'answer_attention_mask': answer_attention_mask}

def preprocess_function_prm_math_pred(sample, tokenizer):
    question_inputs = tokenizer(sample['question']+deepseek_math_prompt, truncation=False)
    question_input_ids = question_inputs['input_ids']
    question_mask = question_inputs['attention_mask']

    answer = sample['answer']
    
    answer_split = answer.split('\n')
    answer_input_ids = []
    answer_attention_mask = []

    step_pred_idx = []
    for ans in answer_split:
        ans = ans+'\n'
        ans_inputs = tokenizer(ans, truncation=False)
        ans_inputs_ids = ans_inputs['input_ids'][1:] + [tokenizer.pad_token_id]
        
        ans_attention_mask = ans_inputs['attention_mask']

        answer_input_ids += ans_inputs_ids
        answer_attention_mask += ans_attention_mask
        step_pred_idx.append(len(answer_input_ids)-1)

    model_input_ids = question_input_ids + answer_input_ids
    model_input_mask = question_mask + answer_attention_mask
    
    return {'model_input_ids': model_input_ids, 'model_input_mask': model_input_mask, \
            'question_input_ids': question_input_ids, 'question_mask': question_mask, \
            'answer_input_ids': answer_input_ids, 'answer_attention_mask': answer_attention_mask, \
            'step_pred_idx': step_pred_idx}

def preprocess_function_prm_code_train(sample, tokenizer):
    question_inputs = tokenizer(sample['prompt'], truncation=False)
    question_input_ids = question_inputs['input_ids']
    question_mask = question_inputs['attention_mask']

    mcts = sample['mcts']
    
    answer_split = sample['sample_answer_split']
    answer_input_ids = []
    answer_attention_mask = []
    mcts_inputs_ids = []
    mcts_attention_mask = []

    for ans, fag in zip(answer_split, mcts):
        # ans = ans+'\n'
        ans_inputs = tokenizer(ans, truncation=False)
        ans_inputs_ids = ans_inputs['input_ids'][1:] + [tokenizer.pad_token_id]
        ans_attention_mask = ans_inputs['attention_mask']

        answer_input_ids += ans_inputs_ids
        answer_attention_mask += ans_attention_mask
        mcts_inputs_ids += [-100] * (len(ans_inputs_ids)-1) + [int(fag > 0)]
        mcts_attention_mask += [1] * len(ans_attention_mask)

    model_input_ids = question_input_ids + answer_input_ids
    
    model_input_mask = question_mask + answer_attention_mask
    model_label_ids = [-100] * len(question_input_ids) + mcts_inputs_ids
    
    
    return {'model_input_ids': model_input_ids, 'model_input_mask': model_input_mask, 'model_label_ids': model_label_ids, \
            'question_input_ids': question_input_ids, 'question_mask': question_mask, 'answer_input_ids': answer_input_ids, \
            'answer_attention_mask': answer_attention_mask}

def preprocess_function_prm_code_pred(sample, tokenizer):
    question_inputs = tokenizer(sample['prompt'], truncation=False)
    question_input_ids = question_inputs['input_ids']
    question_mask = question_inputs['attention_mask']

    answer_split = sample['sample_answer_split']
    answer_input_ids = []
    answer_attention_mask = []

    step_pred_idx = []
    for ans in answer_split:
        # ans = ans+'\n'
        ans_inputs = tokenizer(ans, truncation=False)
        ans_inputs_ids = ans_inputs['input_ids'][1:] + [tokenizer.pad_token_id]
        
        ans_attention_mask = ans_inputs['attention_mask']

        answer_input_ids += ans_inputs_ids
        answer_attention_mask += ans_attention_mask
        step_pred_idx.append(len(answer_input_ids)-1)

    model_input_ids = question_input_ids + answer_input_ids
    
    model_input_mask = question_mask + answer_attention_mask
    
    return {'model_input_ids': model_input_ids, 'model_input_mask': model_input_mask, \
            'question_input_ids': question_input_ids, 'question_mask': question_mask, \
            'answer_input_ids': answer_input_ids, 'answer_attention_mask': answer_attention_mask, \
            'step_pred_idx': step_pred_idx}
        
def run_gsm8k_prm_predict(args, tokenzierd_datasets=None, dataset_test=None):
    def collate_fn_test(batch):
        input_ids = [torch.tensor(example['model_input_ids']) for example in batch]
        attention_mask = [torch.tensor(example['model_input_mask']) for example in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        question_input_ids = [example['question_input_ids'] for example in batch]
        answer_input_ids = [example['answer_input_ids'] for example in batch]
        step_pred_idx = [example['step_pred_idx'] for example in batch]

        return input_ids, attention_mask, step_pred_idx, question_input_ids, answer_input_ids
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model =  LlamaModelForBinaryRegression(args=args)

    test_loader = DataLoader(tokenzierd_datasets, batch_size=args.per_device_eval_batch_size, collate_fn=collate_fn_test, shuffle=False)
    
    model.eval()
    with torch.no_grad():
        test_data = tqdm(test_loader, total=test_loader.__len__(), file=sys.stdout)

        pred_step_ans = []
        pred_step_max_ans = []
        for input_ids, attention_mask, step_pred_idx, question_input_ids, answer_input_ids in test_data:
            input_ids = input_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')
            output_ids = model(input_ids=input_ids, attention_mask=attention_mask)['logits']
            # outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            output_ids = F.softmax(output_ids, dim=-1)
            max_output_ids = torch.argmax(output_ids, dim=-1)
            output_ids = output_ids.tolist()
            max_output_ids = max_output_ids.tolist()
           
            for out_ids, max_out_ids, sp_idx, q_input_ids, a_input_ids in zip(output_ids, max_output_ids, step_pred_idx, question_input_ids, answer_input_ids):
                pred_ans = out_ids[len(q_input_ids):]
                max_pred_ans = max_out_ids[len(q_input_ids):]

                filter_pred_ans = [pred_ans[idx] for idx in sp_idx]
                filter_max_pred_ans =[max_pred_ans[idx] for idx in sp_idx]
                
                pred_step_ans.append(filter_pred_ans)
                pred_step_max_ans.append(filter_max_pred_ans)

    dataset_test_list = [{"idx": sample['idx'], "question": sample["question"], "gold_answer": sample["gold_answer"], 'sample_answer': sample['answer'], 'sample_ans': sample['sample_ans'], 'final_ans': sample['final_ans']} for sample in dataset_test]

    for res, p_ans, pm_acc in zip(dataset_test_list, pred_step_ans, pred_step_max_ans):
        res['pred_step_ans'] = p_ans
        res['pred_step_max_ans'] = pm_acc

    groups_sample_answer = defaultdict(list)
    for sample in dataset_test_list:
        groups_sample_answer[sample['idx']].append({'question': sample['question'], 'gold_answer': sample['gold_answer'], 'final_ans': sample['final_ans'],\
                                                    'sample_answer': sample['sample_answer'], 'sample_ans': sample['sample_ans'], \
                                                    'pred_step_ans': sample['pred_step_ans'], 'pred_step_max_ans': sample['pred_step_max_ans']})
    
    groups_sample_answer_list = [group.copy() for group in groups_sample_answer.values()]

    save_res = []
    for group_sample in groups_sample_answer_list:
        cur_res = {}
        cur_res['question'] = group_sample[0]['question']
        cur_res['gold_answer'] = group_sample[0]['gold_answer']
        cur_res['final_ans'] = group_sample[0]['final_ans']
        
        cur_sample_answer = []
        cur_sample_ans = []
        cur_sample_pred_step_ans = []
        cur_sample_max_pred_step_ans = []
        for sample in group_sample:
            cur_sample_answer.append(sample['sample_answer'])
            cur_sample_ans.append(sample['sample_ans'])
            cur_sample_pred_step_ans.append(sample['pred_step_ans'])
            cur_sample_max_pred_step_ans.append(sample['pred_step_max_ans'])
        
        cur_res['sample_answer'] = cur_sample_answer
        cur_res['sample_ans'] = cur_sample_ans
        cur_res['pred_step_ans'] = cur_sample_pred_step_ans
        cur_res['pred_step_max_ans'] = cur_sample_max_pred_step_ans
        save_res.append(cur_res)

    os.makedirs(args.output_dir, exist_ok=True)
    print("save predict res to: "+args.evalute_res_path)
    with open(args.evalute_res_path, "w", encoding="utf-8") as json_file:
        json.dump(save_res, json_file, ensure_ascii=False)

def run_mbpp_prm_predict(args, tokenzierd_datasets=None, dataset_test=None):
    def collate_fn_test(batch):
        input_ids = [torch.tensor(example['model_input_ids']) for example in batch]
        attention_mask = [torch.tensor(example['model_input_mask']) for example in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        question_input_ids = [example['question_input_ids'] for example in batch]
        answer_input_ids = [example['answer_input_ids'] for example in batch]
        step_pred_idx = [example['step_pred_idx'] for example in batch]

        return input_ids, attention_mask, step_pred_idx, question_input_ids, answer_input_ids
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model =  LlamaModelForBinaryRegression(args=args)

    test_loader = DataLoader(tokenzierd_datasets, batch_size=args.per_device_eval_batch_size, collate_fn=collate_fn_test, shuffle=False)
    
    model.eval()
    with torch.no_grad():
        test_data = tqdm(test_loader, total=test_loader.__len__(), file=sys.stdout)

        pred_step_ans = []
        pred_step_max_ans = []
        for input_ids, attention_mask, step_pred_idx, question_input_ids, answer_input_ids in test_data:
            input_ids = input_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')
            output_ids = model(input_ids=input_ids, attention_mask=attention_mask)['logits']
            # outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            output_ids = F.softmax(output_ids, dim=-1)
            max_output_ids = torch.argmax(output_ids, dim=-1)
            output_ids = output_ids.tolist()
            max_output_ids = max_output_ids.tolist()
           
            for out_ids, max_out_ids, sp_idx, q_input_ids, a_input_ids in zip(output_ids, max_output_ids, step_pred_idx, question_input_ids, answer_input_ids):
                pred_ans = out_ids[len(q_input_ids):]
                max_pred_ans = max_out_ids[len(q_input_ids):]

                filter_pred_ans = [pred_ans[idx] for idx in sp_idx]
                filter_max_pred_ans = [max_pred_ans[idx] for idx in sp_idx]
                
                pred_step_ans.append(filter_pred_ans)
                pred_step_max_ans.append(filter_max_pred_ans)

    dataset_test_list = [{"idx": sample['idx'], "prompt": sample["prompt"], 'text': sample['text'], "gold_answer": sample["gold_answer"], 'sample_answer': sample['sample_answer'],\
                          'sample_answer_split': sample['sample_answer_split'], 'sample_ans': sample['sample_ans']} for sample in dataset_test]

    for res, p_ans, pm_acc in zip(dataset_test_list, pred_step_ans, pred_step_max_ans):
        res['pred_step_ans'] = p_ans
        res['pred_step_max_ans'] = pm_acc

    groups_sample_answer = defaultdict(list)
    for sample in dataset_test_list:
        groups_sample_answer[sample['idx']].append({'prompt': sample['prompt'], 'text': sample['text'],  'gold_answer': sample['gold_answer'], \
                                                    'sample_answer': sample['sample_answer'], 'sample_ans': sample['sample_ans'], 'sample_answer_split': sample['sample_answer_split'], \
                                                    'pred_step_ans': sample['pred_step_ans'], 'pred_step_max_ans': sample['pred_step_max_ans']})
    
    groups_sample_answer_list = [group.copy() for group in groups_sample_answer.values()]

    save_res = []
    for group_sample in groups_sample_answer_list:
        cur_res = {}
        cur_res['prompt'] = group_sample[0]['prompt']
        cur_res['text'] = group_sample[0]['text']
        cur_res['gold_answer'] = group_sample[0]['gold_answer']
        
        cur_sample_answer = []
        cur_sample_answer_split = []
        cur_sample_ans = []
        cur_sample_pred_step_ans = []
        cur_sample_max_pred_step_ans = []
        for sample in group_sample:
            cur_sample_answer.append(sample['sample_answer'])
            cur_sample_ans.append(sample['sample_ans'])
            cur_sample_answer_split.append(sample['sample_answer_split'])
            cur_sample_pred_step_ans.append(sample['pred_step_ans'])
            cur_sample_max_pred_step_ans.append(sample['pred_step_max_ans'])
        
        cur_res['sample_answer'] = cur_sample_answer
        cur_res['sample_ans'] = cur_sample_ans
        cur_res['sample_answer_split'] = cur_sample_answer_split
        cur_res['pred_step_ans'] = cur_sample_pred_step_ans
        cur_res['pred_step_max_ans'] = cur_sample_max_pred_step_ans
        save_res.append(cur_res)

    os.makedirs(args.output_dir, exist_ok=True)
    print("save predict res to: "+args.evalute_res_path)
    with open(args.evalute_res_path, "w", encoding="utf-8") as json_file:
        json.dump(save_res, json_file, ensure_ascii=False)