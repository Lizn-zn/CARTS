"""Proof search using best-first search.
"""
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
    ProofFinished,
    ProofGivenUp,
    DojoInitError,
    DojoCrashError,
    DojoTacticTimeoutError
)
from loguru import logger
from dataclasses import dataclass
from typing import List, Optional, Tuple
from ray.util.actor_pool import ActorPool

from common import zip_strict
from prover.search_tree import *
from prover.repl import LeanIOContext


@dataclass(frozen=True)
class SearchResult:
    """The result of attempting to prove a theorem."""

    theorem: Theorem
    status: Status
    proof: Optional[List[str]]

    # Some statistics during proof search.
    actor_time: float
    environment_time: float
    total_time: float
    num_total_nodes: int
    num_searched_nodes: int


class BestFirstSearchProver:
    """A prover that uses best-first search to find proofs using a tactic generator."""

    def __init__(
        self,
        tac_gen,  # A given tactic generator.
        timeout: int,
        iterations: int,
        num_sampled_tactics: int,
        debug: bool,
        reward_model_path=None,
        reward_model=None,
        **kwargs
    ) -> None:
        self.tac_gen = tac_gen
        self.timeout = timeout
        self.num_sampled_tactics = num_sampled_tactics
        self.debug = debug

        self.num_expansions = 0
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.total_time = None
        self.reward_model_path=reward_model_path
        self.reward_model=reward_model
        self.template='Human: {}\nAssistant: '
            
        assert self.reward_model_path is None
    
    def search(
        self, repo: LeanGitRepo, thm: Theorem, pos: Pos
    ) -> Optional[SearchResult]:
        logger.info(f"Proving {thm}")

        self.repo = repo
        self.theorem = thm
        self.position = pos # not used in bfs
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.num_expansions = 0

        imps = ["Mathlib.Tactic"]
        try:
            with LeanIOContext(thm, timeout=self.timeout, additional_imports=imps) as (
                repl,
                init_state,
            ):
                self.repl = repl
                self.root = InternalNode(
                    state=init_state,
                    cumulative_logprob=0.0,
                )
                self.nodes = {init_state: self.root}
                self.priority_queue = [self.root]

                with torch.no_grad():

                    try:
                        self._best_first_search()
                    except DojoCrashError:
                        logger.warning(f"Dojo crashed when proving {thm}")
                        pass

            if self.root.status == Status.PROVED:
                proof = [e.tactic for e in self.root.extract_proof()]
            else:
                proof = None

            result = SearchResult(
                theorem=thm,
                status=self.root.status,
                proof=proof,
                actor_time=self.actor_time,
                environment_time=self.environment_time,
                total_time=self.total_time,
                num_total_nodes=len(self.nodes),
                num_searched_nodes=self.num_expansions,
            )
            logger.info(result)
            return result

        except DojoInitError as ex:
            logger.warning(ex)
            return None

    def _best_first_search(self) -> None:
        time_start = time.monotonic()

        while True:
            if len(self.priority_queue) == 0:
                logger.info("Ran out of nodes to search.")
                break

            try:
                self._step()
            except  DojoTacticTimeoutError:
                assert time.monotonic() - time_start >= self.timeout

            self.total_time = time.monotonic() - time_start
            if self.total_time > self.timeout:
                if self.root.status == Status.PROVED:
                    logger.info("Found a proof but timed out.")
                self.root.status = Status.OPEN
                logger.info("Search timed out.")
                break

            if self.root.status == Status.FAILED:
                logger.info("Failed early!")
                break

            if self.root.status == Status.PROVED:
                logger.info("Found a proof!")
                break

    def _step(self):
        """
        Perform a single step of search.

        Selects the node with the highest priority, queries the model for suggested
        tactics, and tries each tactic in the environment, creating and enqueuing
        a new node for each valid result.
        """
        # Search the node with highest priority.
        search_node = heapq.heappop(self.priority_queue)
        logger.debug(f"Expanding node: {search_node}")

        if self.debug:
            assert all(
                search_node.priority >= node.priority for node in self.priority_queue
            )

        if isinstance(search_node.state, TacticState):
            ts = search_node.state.pp
        else:
            ts = search_node.state.unsolved_tactic_state
        suggestions = self._generate_tactics(ts)

        # print('actor time: ',self.actor_time)
        # print('\n')
        # Try all tactics in order of descending logprob, and collect the results. Any
        # new nodes are added to `self.nodes`, and edges are added to the result node.
        results = []
        for tactic, value in suggestions:
            res_node=self._run_tactic(search_node, tactic, value)
            results.append(res_node)

        # print('environment time: ',self.environment_time)
        # Store the fixed out edges of this node, marking it as explored.
        # This will trigger recursively recomputing tree statistics.
        search_node.out_edges = results
        self.num_expansions += 1

        # If we're running in debug mode, run a full test suite each step
        if self.debug:
            assert self.num_expansions == sum(
                node.is_explored
                for node in self.nodes.values()
                if isinstance(node, InternalNode)
            )
            self.check_invariants()

    def _generate_tactics(self, ts: str) -> List[Tuple[str, float]]:
        t0 = time.monotonic()

        path = str(self.theorem.file_path)

        suggestions = self.tac_gen.generate(
            state=ts,
            file_path=path,
            theorem_full_name=self.theorem.full_name,
            theorem_pos=self.position,
            num_samples=self.num_sampled_tactics,
        )

        self.actor_time += time.monotonic() - t0

        logger.debug(f"Tactic suggestions: {suggestions}")
        return suggestions

    def _run_tactic(self, node: InternalNode, tactic: str, logprob: float) -> Edge:
        t0 = time.monotonic()
        print(node.state)
        exit()
        response = self.repl.apply(node.state, tactic)
        elapsed = time.monotonic() - t0
        self.environment_time += elapsed
        try:
            # If we've seen this response before, use the existing node
            result_node = self.nodes[response]
        except KeyError:
            # Build a new node
            if isinstance(response, ProofFinished):
                result_node = ProofFinishedNode(response)
            elif type(response) in (
                LeanError,
                TimeoutError,
                ProofGivenUp,
            ):
                result_node = ErrorNode(response)
            else:
                assert isinstance(response, TacticState)
                result_node = InternalNode(
                    state=response,
                    cumulative_logprob=logprob + node.cumulative_logprob,
                )

            if result_node.status == Status.OPEN:  # Don't search proved/failed nodes
                heapq.heappush(self.priority_queue, result_node)  # type: ignore

        # Record the new node and add it to the search queue.
        self.nodes[response] = result_node

        # Build an edge connecting these nodes.
        # Will be added to the source node externally.
        edge = Edge(tactic=tactic, src=node, dst=result_node)

        if isinstance(result_node, InternalNode):
            result_node.in_edges.append(edge)

        return edge

    #########
    # DEBUG #
    #########

    def check_invariants(self):
        """Perform some sanity checks."""
        for node in self.priority_queue:
            assert node in self.nodes.values()
            assert isinstance(node, InternalNode)
            assert not node.is_explored

        for response, node in self.nodes.items():
            if isinstance(response, ProofFinished):
                assert isinstance(node, ProofFinishedNode)
                assert node not in self.priority_queue
                assert self.root.status == Status.PROVED
            elif type(response) in (
                LeanError,
                TimeoutError,
                ProofGivenUp,
            ):
                assert isinstance(node, ErrorNode)
                assert node not in self.priority_queue
            else:
                assert isinstance(node, InternalNode)

                if node.is_explored:
                    assert node not in self.priority_queue
                else:
                    assert node in self.priority_queue

                node.check_invariants()
