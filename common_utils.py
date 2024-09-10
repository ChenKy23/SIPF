import json, re
import numpy as np
import random
import os
import torch

import sys

import editdistance

from prompt import *

def set_seed(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    torch.backends.cudnn.deterministic = True
    # torch.set_deterministic(True)
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

ANS_RE = re.compile(r"The answer is (.*)")
ANS_RE_BOX = re.compile(r"\\boxed{(.*)}")
INVALID_ANS = "[invalid]"

def extract_answer(completion, stype="uni"):
    if stype == "box":
        match = ANS_RE_BOX.search(completion)
    else:
        match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = re.sub(r'[^0-9.+-]', '', match_str)
        match_str = match_str.replace(",", "")
        match_str = match_str.strip('.')
        if match_str.count('.') > 1:
            return INVALID_ANS
        return match_str
    else:
        return INVALID_ANS

def batch_find_ans(decoded_list, stype="uni"):
    ans_list = []
    for decode in decoded_list:
        ans = extract_answer(decode, stype)
        ans_list.append(ans)
    return ans_list

def extract_assert_expression(text):
    pattern = r'assert(.*?)\s*\('
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None

def check_equation(string):
    if string.find('=') == -1:
        return False
    lhs = string.split('=')[0]
    rhs = string.split('=')[1]
    try:
        lhs_result = eval(str(lhs))
        if abs(float(lhs_result) - float(rhs)) < 1e-3:
            return True
    except BaseException:
        return False
    return False

def reasoning_path_selection(equation_different_path, exist_match, new_path):
    pattern = r'<<([^>]*)>>'  
    matches = re.findall(pattern, new_path)  # Find all matches
    equation_flag = True
    add_fag = False
    for match in matches:
        if not check_equation(match):
            equation_flag = False
    if equation_flag:
        matches = '|'.join(matches).replace(' ', '')
        if matches not in exist_match:
            equation_different_path.append(new_path)
            exist_match.append(matches)
            add_fag = True
        else:
            now_query_idx = exist_match.index(matches)
            now_query = equation_different_path[now_query_idx]
            now_score = sum([editdistance.eval(now_query, ref) for ref in equation_different_path if ref != now_query])
            q_score = sum([editdistance.eval(new_path, ref) for ref in equation_different_path if ref != now_query])
            if q_score > now_score:
                equation_different_path[now_query_idx] = new_path

    return equation_different_path, exist_match, add_fag

def add_string_with_distance_check(existing_list, new_string, bound = 5):
    for s in existing_list:
        if editdistance.eval(s, new_string) < bound * 0.8 + abs(len(s)-len(new_string))/2:
            return existing_list, False
    existing_list.append(new_string)
    return existing_list, True

def humaneval_code_text_processing(otext, p_len):
    ltext = otext[:p_len]
    rtext = otext[p_len:]
    rtext = rtext.split("assert")[0].rstrip()
    rtext = rtext.split("print")[0].rstrip()
    rtext = rtext.split("```")[0].rstrip()
    rtext = rtext.split("\'\'\'")[0].rstrip()
    rtext = rtext.split("\ndef")[0].rstrip()
    rtext = rtext.split("from typing import")[0].rstrip()
    rtext = rtext.split("import")[0].rstrip()
    rtext = rtext.split("\n\"\"\"")[0].rstrip()
    rtext = rtext.split("\n#")[0].rstrip()
    rtext = rtext.split("if __name__")[0].rstrip()
    # rtext = "def".join(rtext.split("def")[:2]).rstrip()
    # rtext = rtext.split("\"\"\"")[0].rstrip()
    otext = ltext + rtext

    return otext

def mbpp_code_text_processing(otext, fcn):
    if fcn:
        otext = otext.split("\n"+fcn)[0].strip()
        dfcn = "def "+fcn
        split_otext = otext.split(dfcn)[:2]
        otext = dfcn.join(split_otext)
        otext = otext.strip()
        otext = otext.rstrip("def").strip()
    
    otext = otext.split("assert")[0].rstrip()
    otext = otext.split("print")[0].rstrip()
    # otext = otext.split("def")[0].rstrip()
    otext = otext.split("2. Write a ")[0].rstrip()
    otext = otext.split("if __name__")[0].rstrip()
    otext = "def".join(otext.split("def")[:2]).rstrip()
    if otext.startswith("import") or otext.startswith("from"):
        otext = "import".join(otext.split("import")[:2]).rstrip()
        otext = "from".join(otext.split("from")[:2]).rstrip()
    else:
        otext = otext.split("import")[0].rstrip()
        otext = otext.split("from")[0].rstrip()
    otext = otext.split("Lecture Script:")[0].rstrip()
    otext = otext.split("Lecture Note:")[0].rstrip()
    otext = otext.split("#")[0].rstrip()
    otext = otext.split("\'\'\'")[0].rstrip()
    otext = otext.split("```")[0].rstrip()

    if otext.startswith("mport "):
        otext = "i" + otext
    if otext.startswith("port "):
        otext = "im" + otext
    if otext.startswith("rom "):
        otext = "f" + otext
    if otext.startswith("ef "):
        otext = "d" + otext
    if otext.startswith("f "):
        otext = "de" + otext
    if otext.startswith("lass "):
        otext = "c" + otext

    return otext

def trans_dict_math_train_pro(data):
    data_dic = {}

    data_dic['question'] = []
    data_dic['answer'] = []
    data_dic['mcts'] = []
    data_dic['gold_answer'] = []

    cnt_drop = 0
    for item in data:
        question = item['question']
        sample_answer = item['sample_answer']
        sample_mcts = item['sample_mcts']

        for answer, mcts in zip(sample_answer, sample_mcts):
            answer_list = answer.split('\n')[:-1]
            cnt_steps = len(answer_list)
            if len(mcts['per_step_correct_ans']) != cnt_steps:
                print('steps is mismatch!')
                cnt_drop = cnt_drop + 1
                continue

            data_dic['question'].append(question)
            answer = '\n'.join(answer_list)
            
            data_dic['gold_answer'].append(item['answer'])
            data_dic['answer'].append(answer)
            data_dic['mcts'].append(mcts['per_step_correct_ans'])

        gold_answer_list = item['answer'].split('\n')[:-1]
        data_dic['answer'].append('\n'.join(gold_answer_list))
        gold_answer_mcts = [8] * len(gold_answer_list)
        data_dic['gold_answer'].append(answer) 
        data_dic['mcts'].append(gold_answer_mcts)
        data_dic['question'].append(question)

    return data_dic

def trans_dict_math_train_out(data):
    data_dic = {}

    data_dic['question'] = []
    data_dic['answer'] = []
    data_dic['mcts'] = []
    data_dic['gold_answer'] = []

    cnt_drop = 0
    for item in data:
        question = item['question']
        sample_answer = item['sample_answer']
        sample_mcts = item['sample_mcts']
        sample_ans = item['sample_ans']

        gold_ans = item['final_res']

        for answer, mcts, sans in zip(sample_answer, sample_mcts, sample_ans):
            answer_list = answer.split('\n')[:-1]
            cnt_steps = len(answer_list)
            if len(mcts['per_step_correct_ans']) != cnt_steps:
                print('steps is mismatch!')
                cnt_drop = cnt_drop + 1
                continue

            if sans!='[invalid]' and float(gold_ans) == float(sans):
                per_step_correct_ans = [1] * len(mcts['per_step_correct_ans'])
            else:
                per_step_correct_ans = [0] * len(mcts['per_step_correct_ans'])

            data_dic['question'].append(question)
            answer = '\n'.join(answer_list)
            
            data_dic['gold_answer'].append(item['answer'])
            data_dic['answer'].append(answer)
            data_dic['mcts'].append(per_step_correct_ans)

        gold_answer_list = item['answer'].split('\n')[:-1]
        data_dic['answer'].append('\n'.join(gold_answer_list))
        gold_answer_mcts = [8] * len(gold_answer_list)
        data_dic['gold_answer'].append(answer) 
        data_dic['mcts'].append(gold_answer_mcts)
        data_dic['question'].append(question)

    return data_dic

def trans_dict_math_pred(data):
    data_dic = {}

    data_dic['idx'] = []
    data_dic['question'] = []
    data_dic['answer'] = []
    data_dic['gold_answer'] = []
    data_dic['sample_ans'] = []
    data_dic['final_ans'] = []

    for idx, item in enumerate(data):
        question = item['question']
        sample_answer = item['sample_answer']
        sample_ans = item['sample_ans']

        for answer, ans in zip(sample_answer, sample_ans):
            data_dic['idx'].append(idx)
            data_dic['question'].append(question)
            answer_list = answer.split('\n')[:-1]
            answer = '\n'.join(answer_list)
            
            data_dic['gold_answer'].append(item['answer'])
            data_dic['answer'].append(answer)
            data_dic['final_ans'].append(item['final_ans'])
            data_dic['sample_ans'].append(ans)

    return data_dic

def trans_dict_code_train_pro(data):
    data_dic = {}

    data_dic['text'] = []
    data_dic['prompt'] = []
    data_dic['sample_answer'] = []
    data_dic['sample_answer_split'] = []
    data_dic['sample_ans'] = []
    data_dic['mcts'] = []
    data_dic['gold_answer'] = []

    cnt_drop = 0
    for item in data:
        prompt = item['prompt']
        text = item['text']
        sample_answer_code = item['sample_answer_code']
        sample_mcts = item['sample_mcts']
        sample_ans = item['sample_ans']
        gold_code = item['code']

        gold_code_list = gold_code.split("\n")
        fiter_gold_code_list = []
        for spg_code in gold_code_list:
            if len(spg_code) == 0:
                continue
            fiter_gold_code_list.append(spg_code+"\n")

        for sample_code, mcts, sans in zip(sample_answer_code, sample_mcts, sample_ans):
            fiter_split_code_list = mcts['save_code_split']

            cnt_steps = len(fiter_split_code_list)
            if len(mcts['per_step_correct_ans']) != cnt_steps:
                print('steps is mismatch!')
                # print(sample_code)
                cnt_drop = cnt_drop + 1
                continue

            fiter_step_code_list = []
            mcts_step_score = []
            for step_code, step_mcts in zip(fiter_split_code_list, mcts['per_step_correct_ans']):
                piece_code_list  = step_code.split("\n")
                for piece_code in piece_code_list:
                    if len(piece_code) == 0:
                        continue
                    fiter_step_code_list.append(piece_code+"\n")
                    mcts_step_score.append(step_mcts)

            data_dic['prompt'].append(prompt)
            
            data_dic['gold_answer'].append(gold_code)
            data_dic['sample_answer'].append(sample_code)
            data_dic['sample_answer_split'].append(fiter_step_code_list)
            data_dic['sample_ans'].append(sans)
            data_dic['text'].append(text)
            data_dic['mcts'].append(mcts_step_score)
            
        data_dic['sample_answer'].append(gold_code)
        data_dic['sample_answer_split'].append(fiter_gold_code_list)
        gold_answer_mcts = [6] * len(gold_code_list)
        data_dic['gold_answer'].append(gold_code)
        data_dic['sample_ans'].append(True)
        data_dic['mcts'].append(gold_answer_mcts)
        data_dic['text'].append(text)
        data_dic['prompt'].append(prompt)

    return data_dic

def trans_fict_code_pred(data):
    data_dic = {}

    data_dic['idx'] = []
    data_dic['text'] = []
    data_dic['prompt'] = []
    data_dic['sample_answer'] = []
    data_dic['sample_answer_split'] = []
    data_dic['sample_ans'] = []
    data_dic['gold_answer'] = []

    for idx, item in enumerate(data):
        prompt = item['prompt']
        text = item['text']
        sample_answer_code = item['sample_answer_code']
        sample_ans = item['sample_ans']
        gold_code = item['code']

        for sample_code, sans in zip(sample_answer_code, sample_ans):
            data_dic['idx'].append(idx)
            data_dic['prompt'].append(prompt)

            sample_code_list = sample_code.split("\n")
            fiter_sample_code_list = []
            for cur_code in sample_code_list:
                if len(cur_code) == 0:
                    continue
                fiter_sample_code_list.append(cur_code+"\n")
    
            # fiter_sample_code = ''.join(fiter_sample_code_list)
            
            data_dic['gold_answer'].append(gold_code)
            data_dic['text'].append(text)
            data_dic['sample_answer'].append(sample_code)
            data_dic['sample_answer_split'].append(fiter_sample_code_list)
            data_dic['sample_ans'].append(sans)

    return data_dic