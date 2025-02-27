
 
import numpy as np
import copy
import torch
import time
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
    DojoCrashError
)
import sys 
sys.path.append("../")
from typing import List, Optional, Tuple
from prover.search_tree import *
from loguru import logger

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

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


class BaseTreeNode(object):
    def __init__(self,state,parent, prior_p,status,tactic=None):
        self.state= state
        self.status=status
        self.tactic=tactic
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
        self.leave_value=0
    def extract_proof(self,para_action):
        if len(self._children.keys())==0:
            return para_action
        for k in self._children.keys():
            if self._children[k].status==Status.PROVED:
                return self._children[k].extract_proof(para_action+[k])
    def select(self, c_puct):
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))
    
    def expand(self,suggestions,responses):
        for i in range(len(suggestions)):
            response=responses[i]
            action,p=suggestions[i]
            if isinstance(response, ProofFinished):
                result_node = BaseTreeNode(state=response,parent=self,prior_p=np.exp(p),status=Status.PROVED,tactic=action)
                
            elif type(response) in (
                LeanError,
                TimeoutError,
                ProofGivenUp,
            ):
                # result_node = ErrorNode(response)
                continue
            else:
                assert isinstance(response, TacticState)
                result_node = BaseTreeNode(state=response,parent=self,prior_p=np.exp(p),status=Status.OPEN,tactic=action)
            
            self._children[action] = result_node
            
        # Record the new node and add it to the search queue.
        # self.nodes[response] = result_node


    def update(self, leaf_value):
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
    
    def update_recursive(self, leaf_value):
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.status=self.status
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        # print(self._Q ,self._u)
        return self._Q + self._u
    
    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None
    def __str__(self):
        return "Status: {} State: {}".format(self.status,self.state)


class MCTSProver(object):
    def __init__(
        self,
        tac_gen,  # A given tactic generator.
        timeout: int,
        iterations: int,
        num_sampled_tactics: int,
        debug: bool,
        reward_model_path=None,
        reward_model=None,
        c_puct=1.0,
        k=8
    ) -> None:
        self.tac_gen = tac_gen
        self.timeout = timeout
        self.iterations=iterations
        self.num_sampled_tactics = num_sampled_tactics
        self.debug = debug

        self.num_expansions = 0
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.total_time = None
        self.c_puct = c_puct
        self.step_times=0

    def search(
        self, repo: LeanGitRepo, thm: Theorem, pos: Pos
    ) -> Optional[SearchResult]:
        logger.info(f"Proving {thm}")

        self.repo = repo
        self.theorem = thm
        self.posision = pos
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.num_expansions = 0
        self.step_times=0
        #imps = ["Mathlib.Tactic"]
        imps = []
        try:
            with Dojo(thm, hard_timeout=60 + self.timeout, additional_imports=imps) as (
                dojo,
                init_state,
            ):
                self.dojo = dojo
                self.root = BaseTreeNode(init_state, None, 1.0,Status.OPEN)
                self.nodes = {init_state: self.root}

                with torch.no_grad():
                    try:
                        self._search()
                    except DojoCrashError:
                        logger.warning(f"Dojo crashed when proving {thm}")
                        pass

            if self.root.status == Status.PROVED:
                proof = self.root.extract_proof([])
            else:
                proof = None

            result = SearchResult(
                theorem=thm,
                status=self.root.status,
                proof=proof,
                actor_time=self.actor_time,
                environment_time=self.environment_time,
                total_time=self.total_time,
                num_total_nodes=self.step_times,
                num_searched_nodes=self.num_expansions,
            )
            logger.info(result)
            return result

        except DojoInitError as ex:
            logger.warning(ex)
            return None

    def get_value(self,state,tactic):
        ststr=self.template.format(state)
        inputs_text=ststr+tactic 
        with torch.no_grad():
            inputs =self.rm_tokenizer(inputs_text, return_tensors="pt",padding=True, truncation=True,return_attention_mask=False, max_length=2048)
            _,_,outputs = self.reward_model(inputs['input_ids'].cuda())
            chosen_length = (inputs['input_ids'][0] !=self.rm_tokenizer.pad_token_id).nonzero()[-1] + 1
            reward=sigmoid(float(outputs[0][chosen_length - 1]))
        return reward
    
    
    def _search(self):
        time_start = time.monotonic()

        while True:
            try:
                self._step()
            except DojoHardTimeoutError:
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
        self.step_times+=1
        search_node=self.root
        actions=[]
        while True:           
            if search_node.is_leaf():
                break 
            else:
                tactic,search_node=search_node.select(self.c_puct)
                actions.append(tactic)
        # print(actions)
        
        
        if search_node.status==Status.OPEN:
            ts=search_node.state.pp
            suggestions = self._generate_tactics(ts)

            responses = [
                self._run_tactic(search_node, tactic, logprob)
                for tactic, logprob in suggestions
            ]
            search_node.expand(suggestions,responses)
            
            # judge wether proved 
        child_statuses=[search_node._children[e].status for e in search_node._children.keys()]
        
        if Status.PROVED in child_statuses:
            search_node.status=Status.PROVED
            search_node.update_recursive(+1)
            
            
        # expand the node 
        leave_value=0.
        if (not search_node.is_root()):
            leave_value=self.get_value(search_node._parent.state.pp,search_node.tactic)
            
        if len(search_node._children)==0:
            leave_value=-1.
            
        search_node.update_recursive(leave_value)
       
        
    def _run_tactic(self, node: InternalNode, tactic: str, logprob: float) -> Edge:
        t0 = time.monotonic()
        response = self.dojo.run_tac(node.state, tactic)

        elapsed = time.monotonic() - t0
        self.environment_time += elapsed

        return response
    
    def _generate_tactics(self, ts: str) -> List[Tuple[str, float]]:
        t0 = time.monotonic()

        path = str(self.theorem.file_path)

        if self.theorem.repo != self.repo:
            path = self.theorem.repo.get_packages_dir() / self.theorem.repo.name / path

        suggestions = self.tac_gen.generate(
            state=ts,
            file_path=path,
            theorem_full_name=self.theorem.full_name,
            theorem_pos=self.posision,
            num_samples=self.num_sampled_tactics,
        )

        self.actor_time += time.monotonic() - t0

        logger.debug(f"Tactic suggestions: {suggestions}")
        return suggestions
 
    def __str__(self):
        return "MCTS"
    
    
