import fire
import sys

from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness


def entry_point(
    sample_file: str = '',
    k: str = "1,3,6,10,20,30,60,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = '',
    problem_type: str = 'mbpp'
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file, problem_type)
    print(results)


def main():
    fire.Fire(entry_point)


sys.exit(main())
