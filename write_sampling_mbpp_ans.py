import json, re, os
import argparse

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

parser = argparse.ArgumentParser(description="write mbpp ans")
parser.add_argument("--dataset_org_pth", default='')
parser.add_argument("--dataset_result_pth", default='')
args = parser.parse_args()

dataset_org = read_jsonl(args.dataset_org_pth)[0]
dataset_result =  read_jsonl(args.dataset_result_pth)

dataset_ans = {}
for ans_line in dataset_result:
    example_idx = ans_line['example_idx']
    solution = ans_line['completion']
    ans = ans_line['passed']
    if example_idx not in dataset_ans:
        dataset_ans[example_idx] = []
    
    dataset_ans[example_idx].append((solution, ans))

save_res = []
for line in dataset_org:
    example_idx = line['example_idx']
    solution_ans = dataset_ans[example_idx]
    prompt = line['prompt']

    sample_answer = [sa[0] for sa in solution_ans]
    sample_ans = [sa[1] for sa in solution_ans]

    line['sample_answer'] = sample_answer
    line['sample_ans'] = sample_ans
    line['sample_answer_code'] = [answer[len(prompt):] for answer in sample_answer]

    save_res.append(line.copy())

save_pth = args.dataset_org_pth.rstrip('.json')+'_write.json'
print("save predict res to: " + save_pth)
with open(save_pth, "w", encoding="utf-8") as json_file:
    json.dump(save_res, json_file, ensure_ascii=False)