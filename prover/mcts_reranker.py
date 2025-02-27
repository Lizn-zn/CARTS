
import numpy as np
import copy
import torch
import torch.nn.functional as F
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
    DojoCrashError,
    DojoTacticTimeoutError
)
from transformers import T5EncoderModel,AutoTokenizer,AutoModel
import sys 
sys.path.append("../")
from typing import List, Optional, Tuple
from prover.search_tree import *
from loguru import logger
def batch(iterable, batch_size):
    length = len(iterable)
    for i in range(0, length, batch_size):
        yield iterable[i:i + batch_size]
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
def softmax(x, T=1.0):
    x=np.array(x)
    x = x / T
    exp_x = np.exp(x)  
    return exp_x / np.sum(exp_x)
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
        self._W = 0
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
    
    def expand(self,rerank_tactics,rerank_resonse,rerank_scores):
        for i in range(len(rerank_tactics)):
            response=rerank_resonse[i]
            action=rerank_tactics[i]
            p=rerank_scores[i]
            if isinstance(response, ProofFinished):
                result_node = BaseTreeNode(state=response,parent=self,prior_p=p,status=Status.PROVED,tactic=action)
                
            elif type(response) in (
                LeanError,
                TimeoutError,
                ProofGivenUp,
            ):
                # result_node = ErrorNode(response)
                continue
            else:
                assert isinstance(response, TacticState)
                result_node = BaseTreeNode(state=response,parent=self,prior_p=p,status=Status.OPEN,tactic=action)
            
            self._children[action] = result_node
            
        # Record the new node and add it to the search queue.
        # self.nodes[response] = result_node


    def update(self, leaf_value):
        # Count visit.
        self._n_visits =self._n_visits+ 1
        # Update Q, a running average of values for all visits.
        # self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
        self._W=self._W+leaf_value
    
    def update_recursive(self, leaf_value):
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.status=self.status
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        # print(self.tactic,self._W / (self._n_visits+1),self._u)
        return self._W/ (self._n_visits+1) + self._u
    
    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None
    def __str__(self):
        return "Status: {} State: {}".format(self.status,self.state)


class MCTSRerankerProver(object):
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
        assert reward_model_path!=None
        self.reward_model_path=reward_model_path
        self.reward_model=reward_model
        self.c_puct = c_puct
        self.step_times=0
        self.k=k

        self.encoder_tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small-v2')
        self.encoder = AutoModel.from_pretrained('intfloat/e5-small-v2').cuda()
        self.crash_tactic=[]
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
        self.crash_tactic=[]
        self.step_times=0
        
        imps = ["Mathlib.Tactic"]
        count=1
        while count <= 5:
            try:
                logger.info('try times: {}, {}'.format(count,self.crash_tactic))
                with Dojo(thm, timeout=self.timeout, additional_imports=imps) as (
                    dojo,
                    init_state,
                ):
                    self.dojo = dojo
                    self.root = BaseTreeNode(init_state, None, 1.0,Status.OPEN)
                    self.nodes = {init_state: self.root}

                    with torch.no_grad():
                        self._search()

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
                    num_searched_nodes=self.step_times,
                )
                logger.info(result)
                return result

            except DojoInitError as ex:
                logger.warning(ex)
                return None
            except DojoCrashError:
                logger.warning(f"Dojo crashed when proving {thm}, retry")
                count+=1
        result = SearchResult(
                    theorem=thm,
                    status=self.root.status,
                    proof=None,
                    actor_time=self.actor_time,
                    environment_time=self.environment_time,
                    total_time=self.total_time,
                    num_total_nodes=self.step_times,
                    num_searched_nodes=self.step_times,
                )
        logger.info(result)
        return result
    
    def get_value(self,state,tactic):
        return self.reward_model.evaluate(state,tactic)
    

    def get_state_embedding(self,new_state_list):
        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            # normalize
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings
        with torch.no_grad():
            input= self.encoder_tokenizer(new_state_list,max_length=512,padding=True,truncation=True, return_tensors="pt")
            input_ids=input['input_ids'].cuda()
            mask=input['attention_mask'].cuda()
            outputs = self.encoder(input_ids=input_ids,attention_mask=mask)
            last_hidden_states = outputs.last_hidden_state
            embeddings = mean_pooling(last_hidden_states, mask)
        return embeddings
    
    def mmr(self,sim1, sim2, lambda_param=0.8, k=8):

        n = len(sim1)
        selected = []
        unselected = list(range(n))
        new_scores = [0] * n
        # first_index = unselected.pop(sim1.index(max(sim1)))
        # new_scores[first_index] = sim1[first_index]
        # selected.append(first_index)
        
        while len(selected) < k and unselected:
            max_mmr = -float('inf')
            next_index = -1

            for i in unselected:
                relevance = sim1[i]
                diversity = max([sim2[i][j] for j in selected]+[sim2[i][-1]]) 
                if diversity>0.9999:
                    diversity=float('inf')
                mmr_score =  lambda_param * relevance - (1 - lambda_param) * diversity

                if mmr_score > max_mmr:
                    max_mmr = mmr_score
                    next_index = i
            if next_index==-1:
                break
            selected.append(next_index)
            unselected.remove(next_index)
            new_scores[next_index] = max_mmr
        if min(new_scores)<0:
            new_scores-=min(new_scores)
        return selected,new_scores
    
    def rerank(self,node: BaseTreeNode,suggestions):
        '''
        node: BaseTreeNode, Parent
        Suggestions: [(tactic,logprob)]
        logprobs are not used
        '''

        tactics=[tactic for tactic,_ in suggestions]
        logprobs=[logprob for _,logprob in suggestions]
        
        next_states = []
        new_tactics=[]
        new_response=[]
        new_logprobs=[]
        
        for i in range(len(tactics)):
            if tactics[i] in self.crash_tactic:
                continue
            t0 = time.monotonic()
            try:
                resp=self.dojo.run_tac(node.state, tactics[i])
            except DojoCrashError:
                logger.warning(f"Dojo crashed and retry")
                self.crash_tactic.append(tactics[i])
                raise DojoCrashError
                
            elapsed = time.monotonic() - t0
            self.environment_time += elapsed
            # if self.environment_time>self.timeout:
            #     raise TimeoutError
            
            if isinstance(resp, ProofFinished):
                return [tactics[i]],[resp],[float('inf')],len(suggestions)
            elif isinstance(resp, TacticState):
                new_response.append(resp)
                next_states.append(resp.pp)
                new_tactics.append(tactics[i])
                new_logprobs.append(logprobs[i])
  
        if len(new_tactics)==0:
            return [],[],[],0
        

        #rank
        sim1=[np.exp(logprob) for logprob in new_logprobs]
        embeddings=self.get_state_embedding(next_states+[node.state.pp])

        sim2=torch.matmul(embeddings,embeddings.T).cpu().numpy()
        mmr_indexes,new_scores=self.mmr(sim1,sim2,k=self.k)
        
        #no rank
        # mmr_indexes=range(len(new_tactics))
        # new_scores=[np.exp(logprob) for logprob in new_logprobs]
        # assert len(new_scores)==len(new_tactics)
        
        
        rerank_tactics=[new_tactics[index] for index in mmr_indexes]
        rerank_resonse=[new_response[index] for index in mmr_indexes]
        rerank_scores=[new_scores[index] for index in mmr_indexes]

        return rerank_tactics,rerank_resonse,rerank_scores,len(new_tactics)
    
    def _search(self):
        time_start = time.monotonic()

        for i in range(self.iterations):
            try:
                self._step()
            except DojoTacticTimeoutError:
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
        states=[]
        while True:           
            if search_node.is_leaf():
                break 
            else:
                if search_node.status==Status.OPEN:
                    states.append(search_node.state.pp)
                tactic,search_node=search_node.select(self.c_puct)
                actions.append(tactic)
        # print('=>',actions) 

        if search_node.status==Status.OPEN and search_node.state.pp in states:
            search_node.expand([],[],[])
            number_of_open_state=0

        elif search_node.status==Status.OPEN:
            ts=search_node.state.pp
            suggestions = self._generate_tactics(ts)
            

            # print('Original suggestions:',suggestions)
        
            rerank_tactics,rerank_resonse,rerank_scores,number_of_open_state=self.rerank(search_node,suggestions)

            # print('Rerank suggestions:',rerank_tactics,rerank_scores)
            search_node.expand(rerank_tactics,rerank_resonse,rerank_scores)
            
        child_statuses=[search_node._children[e].status for e in search_node._children.keys()]
        
        if Status.PROVED in child_statuses:
            search_node.status=Status.PROVED
            search_node.update_recursive(+1)
            
        # expand the node 
        
        #debiased value function
        intrinsic_value=number_of_open_state/self.num_sampled_tactics
        if len(search_node._children)==0:
            leave_value=0.
        elif (not search_node.is_root()):
            leave_value=self.get_value(search_node._parent.state.pp,search_node.tactic)
            leave_value=0.5*leave_value+0.5*intrinsic_value
        else:
            # is root
            leave_value=1.
        

        
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
        return "MCTS_Reranker"
    
    
