"""Proof search using best-first search.
"""
import os
import sys
import ray
import time
import heapq
import torch
from lean_dojo import (
    Pos,
    Dojo,
    Theorem,
    LeanGitRepo,
    TacticState,
    LeanError,
    TimeoutError,
    ProofFinished,
    ProofGivenUp,
    DojoInitError,
    DojoCrashError,
)
from loguru import logger
from dataclasses import dataclass
from typing import List, Optional, Tuple
from ray.util.actor_pool import ActorPool

from common import zip_strict

from models import MODELS
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead


from prover.bfs import BestFirstSearchProver,SearchResult,Status
from prover.mcts_reranker import MCTSRerankerProver
from prover.mcts import MCTSProver

PROVERS={
    'BFS': BestFirstSearchProver,
    'MCTS': MCTSProver,
    'MCTS_Reranker': MCTSRerankerProver
}

from prover.value_function import ActorValueFunction


class GpuProver():
    """Ray actor for running searchers on a GPU."""

    def __init__(
        self, 
        prover: Optional[str],
        model_name : Optional[str],
        model_path: Optional[str],
        reward_model_path: Optional[str],
        timeout: int,
        iterations: int,
        num_sampled_tactics: int,
        debug: bool,
        k: int
    ) -> None:
        self.timeout=timeout
        self.num_sampled_tactics=num_sampled_tactics
        self.debug=debug
        
        self.tac_gen=MODELS[model_name](model_path)
        print('Successfully Load Tactic Generator!!')

        
        if reward_model_path!=None:
            self.reward_model=ActorValueFunction(reward_model_path)
        else:
            self.reward_model=None
            
            
        self.searcher= PROVERS[prover](
            self.tac_gen, 
            timeout,
            iterations,
            num_sampled_tactics,
            debug,
            reward_model_path,
            reward_model=self.reward_model,
            k=k
        )
    def search(self, repo: LeanGitRepo, thm: Theorem, pos: Pos):
        return self.searcher.search(repo, thm, pos)
        
@ray.remote(num_gpus=1)
class GpuProverRemote(GpuProver):
    """Ray actor for running searchers on a GPU."""

    def __init__(
        self, 
        prover: Optional[str],
        model_name : Optional[str],
        model_path: Optional[str],
        reward_model_path: Optional[str],
        timeout: int,
        iterations: int,
        num_sampled_tactics: int,
        debug: bool,
        k: int
    ) -> None:
        super().__init__(
            prover,model_name,model_path,reward_model_path,timeout,iterations,num_sampled_tactics,debug,k
        )
    def search(self, repo: LeanGitRepo, thm: Theorem, pos: Pos):
        return self.searcher.search(repo, thm, pos)
        
              
        
class DistributedProver:
    """A distributed prover that uses Ray to parallelize the proof search.

    It is a wrapper around `CpuProver` and `GpuProver` that handles the different
    devices and different number of concurrent provers.
    """

    def __init__(
        self,
        prover: Optional[str],
        model_name : Optional[str],
        model_path: Optional[str],
        reward_model_path: Optional[str],
        timeout: int,
        iterations: int,
        num_sampled_tactics: int,
        debug: bool,
        num_cpus: int,
        k: int
    ) -> None:
        self.distributed = num_cpus > 1
        
        if not self.distributed:
            
            self.prover = GpuProver(prover,
                model_name,
                model_path,
                reward_model_path,
                timeout,
                iterations,
                num_sampled_tactics,
                debug,
                k
            )
            return

        ray.init(log_to_driver=False)
        logger.info(f"Launching {num_cpus} CPU workers.")
        provers = [
            GpuProverRemote.remote(
                prover,
                model_name,
                model_path,
                reward_model_path,
                timeout,
                iterations,
                num_sampled_tactics,
                debug,
                k
            )
            for _ in range(num_cpus)
        ]
        

        self.prover_pool = ActorPool(provers)

    def search_unordered(
        self, repo: LeanGitRepo, theorems: List[Theorem], positions: List[Pos]
    ) -> List[SearchResult]:
        """Parallel proof search for `theorems`. The order of the results is not guaranteed to match the order of the input."""
        if not self.distributed:
            return [
                self.prover.search(repo, thm, pos)
                for thm, pos in zip_strict(theorems, positions)
            ]

        try:
            results = list(
                self.prover_pool.map_unordered(
                    lambda p, x: p.search.remote(repo, x[0], x[1]),
                    zip_strict(theorems, positions),
                )
            )
        except ray.exceptions.RayActorError as ex:
            logger.error(ex)
            sys.exit(1)

        return results