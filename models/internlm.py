
from transformers import AutoTokenizer, AutoModelForCausalLM
from lean_dojo import Pos
from typing import List, Dict, Any, Optional, Tuple
import torch
from models.base import BaseTacticGen
# from base import BaseTacticGen
import vllm
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
class InternLMTacticGen(BaseTacticGen):  
    
    def __init__(self,path="internlm/internlm2-math-plus-1_8b") -> None:
        super().__init__(path)
        self.model = vllm.LLM(
        model=path,
        tensor_parallel_size=1,
        dtype='bfloat16',
        max_num_batched_tokens=32768,
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.75
    )
        self.tokenizer = AutoTokenizer.from_pretrained(path,trust_remote_code=True)
        
    def chat_template_to_prompt(self,prompt_list):
        result = ""
        total_step = len(prompt_list)
        for i, message in enumerate(prompt_list):
            result += ('<|im_start|>' + message['role'] +
                    '\n' + message['content'])
            if i+1 != total_step:
                result += '<|im_end|>\n'
            elif message['role'] == 'user':
                result += '<|im_end|>\n<|im_start|>assistant\n'
        return result

    def prompt_style_internlm_chat_0522_extractor(self,result:str):
        START_STR="Here is the predicted next tactic:\n```lean\n"
        END_STR="\n```"
        if result.startswith(START_STR):
            result=result[len(START_STR):]
        if result.endswith(END_STR):
            result=result[:-len(END_STR)]
        return result

    
    def _unique_sorted(self,texts, scores):
        texts_ = []
        scores_ = []
        for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
            if t not in texts_:
                texts_.append(t)
                scores_.append(s)
        return texts_, scores_
    
    def generate_vllm(self, prompt, temperatures, num_samples, stop, max_tokens=256):
        if not isinstance(prompt, str):
            prompt = self.chat_template_to_prompt(prompt)
        texts, scores = [], []
        for temperature in temperatures:
            params = vllm.SamplingParams(
                n=num_samples,
                temperature=temperature,
                use_beam_search=temperature==0.0,
                max_tokens=max_tokens,
                stop=stop,
                length_penalty=float(temperature!=0.0)
            )
            outputs = self.model.generate([prompt], params, use_tqdm=False)
            if len(outputs) == 0:
                return [], []
            for output in outputs[0].outputs:
                text = output.text.replace(self.tokenizer.eos_token, '')
                score = output.cumulative_logprob/max(len(output.token_ids), 1)
                texts.append(text)
                scores.append(score)

        texts = list(map(self.prompt_style_internlm_chat_0522_extractor,texts))
        texts, scores = self._unique_sorted(texts, scores)
        texts = [s.strip() for s in texts]
        return texts, scores
    
        
    def _prompt_fewshot(self,state):
        prompt = f"My LEAN 4 state is:\n```lean\n" + state + \
            "```\nPlease predict a possible tactic to help me prove the theorem."
        prompt = [{"role": "user", "content": prompt}]
        return prompt
      
    def generate(
            self,
            state: str,
            file_path: str,
            theorem_full_name: str,
            theorem_pos: Pos,
            num_samples: int,
        ) -> List[Tuple[str, float]]:
        
        texts,scores=self.generate_vllm(self._prompt_fewshot(state),[0.0],num_samples,stop=['<|im_end|>',],max_tokens=256)
        tactics_with_scores=[(texts[i],scores[i]) for i in range(len(texts))]
        return tactics_with_scores



class InternLMStepproverTacticGen(BaseTacticGen):  
    
    def __init__(self,path="internlm/internlm2-step-prover") -> None:
        super().__init__(path)
        self.model = vllm.LLM(
        model=path,
        tensor_parallel_size=1,
        dtype='bfloat16',
        max_num_batched_tokens=32768,
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.75
    )
        self.tokenizer = AutoTokenizer.from_pretrained(path,trust_remote_code=True)
        
    def chat_template_to_prompt(self,prompt_list):
        result = ""
        total_step = len(prompt_list)
        for i, message in enumerate(prompt_list):
            result += ('<|im_start|>' + message['role'] +
                    '\n' + message['content'])
            if i+1 != total_step:
                result += '<|im_end|>\n'
            elif message['role'] == 'user':
                result += '<|im_end|>\n<|im_start|>assistant\n'
        return result
    
    def prompt_style_internlm_chat_stepprover_extractor(self,result:str):
        START_STR="PROOFSTEP "
        if result.startswith(START_STR):
            result=result[len(START_STR):]
        return result
    
    
    def _prompt_function(self,theorem, state):
        input_template = (  f"DECL {theorem.full_name}\n"
                            f"GOAL {state}\n"
                        )
        prompt = [{"role": "user", "content": input_template}]
        return prompt
    
        
    def _unique_sorted(self,texts, scores):
        texts_ = []
        scores_ = []
        for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
            if t not in texts_:
                texts_.append(t)
                scores_.append(s)
        return texts_, scores_
    
    def prompt_style_internlm_chat_stepprover_extractor(self,result:str):
        START_STR="PROOFSTEP "
        if result.startswith(START_STR):
            result=result[len(START_STR):]
        return result
    def generate_vllm(self, prompt, temperatures, num_samples, stop, max_tokens=256):
        if not isinstance(prompt, str):
            prompt = self.chat_template_to_prompt(prompt)
        texts, scores = [], []
        for temperature in temperatures:
            params = vllm.SamplingParams(
            n=num_samples,
            temperature=temperature,
            use_beam_search=temperature==0.0,
            max_tokens=max_tokens,
            stop=stop,
            length_penalty=0.0 if temperature==0.0 else 1.0,
            logprobs=True,
            )
            outputs = self.model.generate([prompt], params, use_tqdm=False)
            if len(outputs) == 0:
                return [], []
            for output in outputs[0].outputs:
                text = output.text.replace(self.tokenizer.eos_token, '')
                score = output.cumulative_logprob/max(len(output.token_ids), 1)
                texts.append(text)
                scores.append(score)

        texts = list(map(self.prompt_style_internlm_chat_stepprover_extractor,texts))
        texts, scores = self._unique_sorted(texts, scores)
        texts = [s.strip() for s in texts]
        return texts, scores
    
        
    def _prompt_function(self,theorem_full_name, state):
        input_template = (  f"DECL {theorem_full_name}\n"
                            f"GOAL {state}\n"
                        )
        prompt = [{"role": "user", "content": input_template}]
        return prompt
      
    def generate(
            self,
            state: str,
            file_path: str,
            theorem_full_name: str,
            theorem_pos: Pos,
            num_samples: int,
            temperatures: list = [0.0]
        ) -> List[Tuple[str, float]]:
        
        texts,scores=self.generate_vllm(self._prompt_function(theorem_full_name,state),temperatures,num_samples,stop=['<|im_end|>',],max_tokens=256)
        tactics_with_scores=[(texts[i],scores[i]) for i in range(len(texts))]
        return tactics_with_scores

        

# model=InternLMStepproverTacticGen()
# tactics_with_scores=model.generate("n : ℕ\n⊢ gcd n n = n",file_path=None,theorem_full_name='gcd',theorem_pos=None,num_samples=256,temperatures=[0.0])


# print(tactics_with_scores)