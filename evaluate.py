"""Script for evaluating the prover on theorems extracted by LeanDojo.
"""
import os
import uuid
import json
import pickle
import hashlib
import argparse
from loguru import logger
logger.add('out.log')
from lean_dojo import Theorem
from typing import List, Tuple, Optional
from lean_dojo import LeanGitRepo, Theorem, Pos, is_available_in_cache
import sys
sys.path.append(os.getcwd())
from pytorch_lightning import seed_everything
seed_everything(1234)
from common import set_logger
from prover.proof_search import Status, DistributedProver
from tqdm import tqdm

def _get_theorems(
    data_path: str,
    split: str,
    file_path: str,
    full_name: str,
    name_filter: str,
    num_theorems: int,
) -> Tuple[LeanGitRepo, List[Theorem], List[Pos]]:
    repo, theorems, positions = _get_theorems_from_files(
        data_path,
        split,
        file_path,
        full_name,
        name_filter,
        num_theorems,
    )

    all_repos = {thm.repo for thm in theorems}
    for r in all_repos:
        assert is_available_in_cache(
            r
        ), f"{r} has not been traced yet. Please use LeanDojo to trace it so that it's available in the cache."

    return repo, theorems, positions


def _get_theorems_from_files(
    data_path: str,
    split: str,
    file_path: Optional[str],
    full_name: Optional[str],
    name_filter: Optional[str],
    num_theorems: Optional[int],
) -> Tuple[LeanGitRepo, List[Theorem], List[Pos]]:
    data = json.load(open(os.path.join(data_path, f"{split}.json")))
    
    
    theorems = []
    positions = []
    if num_theorems is not None:
        data=data[:num_theorems]
    for t in tqdm(data):
        repo = LeanGitRepo(t["url"], t["commit"])
        theorems.append(Theorem(repo, t["file_path"], t["full_name"]))
        positions.append(Pos(*t["start"]))
    theorems = sorted(
        theorems,
        key=lambda t: hashlib.md5(
            (str(t.file_path) + ":" + t.full_name).encode()
        ).hexdigest(),
    )

    logger.info(f"{len(theorems)} theorems loaded from {data_path}")
    # print(theorems)
    metadata = json.load(open(os.path.join(data_path, "../metadata.json")))
    repo = LeanGitRepo(metadata["from_repo"]["url"], metadata["from_repo"]["commit"])

    return repo, theorems, positions


def evaluate(
    data_path: str,
    exp_id: Optional[str] = None,
    split: str = "val",
    file_path: Optional[str] = None,
    full_name: Optional[str] = None,
    name_filter: Optional[str] = None,
    num_theorems: Optional[int] = None,
    ckpt_path: Optional[str] = None,
    indexed_corpus_path: Optional[str] = None,
    tactic: Optional[str] = None,
    module: Optional[str] = None,
    num_sampled_tactics: int = 64,
    timeout: int = 600,
    num_cpus: int = 1,
    with_gpus: bool = False,
    verbose: bool = False,
    reward_model_path: bool= None,
    prover_name: Optional[str] = 'BFS',
    model_name: Optional[str] = 'reprover',
    iterations: int =100,
    k: int =8
) -> float:
    set_logger(verbose)

    repo, theorems, positions = _get_theorems(
        data_path, split, file_path, full_name, name_filter, num_theorems
    )

    # Search for proofs using multiple concurrent provers.

    prover = DistributedProver(
        prover=prover_name,
        model_name=model_name,
        model_path=ckpt_path,
        reward_model_path=reward_model_path,
        timeout=timeout,
        iterations=iterations,
        num_sampled_tactics=num_sampled_tactics,
        debug=verbose,
        num_cpus=num_cpus,
        k=k
    )
    results = prover.search_unordered(repo, theorems, positions)

    # Calculate the result statistics.
    num_proved = num_failed = num_discarded = 0
    for r in results:
        if r is None:
            num_discarded += 1
        elif r.status == Status.PROVED:
            num_proved += 1
        else:
            num_failed += 1

    logger.info(
        f"Evaluation done! {num_proved} theorems proved, {num_failed} theorems failed, {num_discarded} non-theorems discarded"
    )

    if num_proved + num_failed == 0:
        pass_1 = float("nan")
    else:
        pass_1 = num_proved / (num_proved + num_failed)

    # Save the results.
    if exp_id is None:
        exp_id = str(uuid.uuid4())
    if not os.path.exists('./results'):
        os.makedirs('./results')
    pickle_path = './results/'+f"{prover_name}_{model_name}_{num_sampled_tactics}_{exp_id}.pickle"
    pickle.dump(results, open(pickle_path, "wb"))
    logger.info(f"Results saved to {pickle_path}")

    return pass_1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script for evaluating the prover on theorems extracted by LeanDojo."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the data extracted by LeanDojo (e.g., data/leandojo_benchmark/random).",
    )
    parser.add_argument("--exp-id", type=str, help="Experiment ID used for logging.")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="val",
    )
    # `file_path`, `full_name`, `name_filter`, and `num_theorems` can be used to filter theorems.
    parser.add_argument("--file-path", type=str)
    parser.add_argument("--full-name", type=str)
    parser.add_argument("--name-filter", type=str)
    parser.add_argument("--num-theorems", type=int)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Checkpoint of the tactic generator.",
    )
    parser.add_argument(
        "--indexed-corpus-path",
        type=str,
        help="Path to a pickled indexed corpus. Not required for models w/o retrieval.",
    )
    parser.add_argument("--tactic", type=str, help="The tactic to evaluate.")
    parser.add_argument("--module", type=str, help="The module to import the tactic.")
    parser.add_argument(
        "--num-sampled-tactics",
        type=int,
        default=64,
        help="Number of tactics to sample at each node during proof search.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Maximum number of seconds the proof search can take.",
    )
    parser.add_argument(
        "--num-cpus", type=int, default=1, help="The number of concurrent provers."
    )
    parser.add_argument(
        "--with-gpus", action="store_true", help="Use GPUs for proof search."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Set the logging level to DEBUG."
    )
    parser.add_argument(
        "--reward_model_path",type=str, default=None, help="The path of reward model"
    )
    parser.add_argument(
        "--prover",type=str, default='BFS', help="The search algorithm"
    )
    parser.add_argument(
        "--model_name",type=str, default='reprover', help="Model Name"
    )
    parser.add_argument(
        "--iterations",type=int, default=100, help="T"
    )
    parser.add_argument(
        "--k",type=int, default=8, help="k"
    )
    args = parser.parse_args()

    assert args.ckpt_path or args.tactic

    logger.info(f"PID: {os.getpid()}")
    logger.info(args)

    pass_1 = evaluate(
        args.data_path,
        args.exp_id,
        args.split,
        args.file_path,
        args.full_name,
        args.name_filter,
        args.num_theorems,
        args.ckpt_path,
        args.indexed_corpus_path,
        args.tactic,
        args.module,
        args.num_sampled_tactics,
        args.timeout,
        args.num_cpus,
        args.with_gpus,
        args.verbose,
        args.reward_model_path,
        args.prover,
        args.model_name,
        args.iterations,
        args.k
    )

    logger.info(f"Pass@1: {pass_1}")


if __name__ == "__main__":
    main()
