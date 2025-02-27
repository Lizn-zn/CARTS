
from transformers import AutoTokenizer, AutoModelForCausalLM
from lean_dojo import Pos
from typing import List, Dict, Any, Optional, Tuple
import torch
from models.base import BaseTacticGen
class CodeLlamaTacticGen(BaseTacticGen):  
    
    def __init__(self,path="HaimingW/Leandojo-CodeLLama-7b") -> None:
        super().__init__(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path,device_map='auto',load_in_8bit=True,
        torch_dtype=torch.float16)
        self.template='GOAL {} PROOFSTEP '
      
    def generate(
            self,
            state: str,
            file_path: str,
            theorem_full_name: str,
            theorem_pos: Pos,
            num_samples: int,
        ) -> List[Tuple[str, float]]:
        
        return self._generate(state,file_path,theorem_full_name,theorem_pos,num_samples)
          
    def _generate(
            self,
            state: str,
            file_path: str,
            theorem_full_name: str,
            theorem_pos: Pos,
            num_samples: int,
        ) -> List[Tuple[str, float]]:
        
        input =self.template.format(state)
        tokenized_state = self.tokenizer(input, return_tensors="pt",max_length=512,truncation=True)
        output = self.model.generate(
            tokenized_state.input_ids.cuda(),
            max_new_tokens=512,
            num_beams=num_samples,
            length_penalty=0.0,
            do_sample=True,
            num_return_sequences=num_samples,
            output_scores=True,
            return_dict_in_generate=True,
        )
        tactic_candidates = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )
        tactic_candidates = [tac[len(input):] for tac in tactic_candidates]
        raw_scores = output.sequences_scores.tolist()
        output_text=[]
        tactics_with_scores=[]
        for i in range(num_samples):
            if tactic_candidates[i] not in output_text:
                output_text.append(tactic_candidates[i])
                tactics_with_scores.append((tactic_candidates[i],raw_scores[i]))
        torch.cuda.empty_cache()
        return tactics_with_scores

# model=CodeLlamaTacticGen()
# tactics_with_scores=model.generate("n : ℕ\n⊢ gcd n n = n",None,None,None,64)
# print(tactics_with_scores)