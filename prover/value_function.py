

from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
from trl import AutoModelForCausalLMWithValueHead
import os
import torch
import numpy as np


class ActorValueFunction:
    def __init__(self,model_path='yangxw/CARTS_vf'):
        
            self.template="<|user|>\n{}<|end|>\n<|assistant|>\n{}"
            self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="right",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                        model_path
            )
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(self.model)
            vhead_params = self.load_valuehead_params(model_path)
            self.model.load_state_dict(vhead_params, strict=False)
            self.model.requires_grad_(False)
            self.model=self.model.cuda()
            
    def load_valuehead_params(self,path_or_repo_id: str):
        kwargs = {"path_or_repo_id": path_or_repo_id, "cache_dir":None, "token": None}

        from safetensors import safe_open
        from transformers.utils import cached_file

        vhead_file = cached_file(filename="value_head.safetensors", **kwargs)
        with safe_open(vhead_file, framework="pt", device="cpu") as f:
            return {key: f.get_tensor(key) for key in f.keys()}
        
    def sigmoid(self,x):
        return 1.0 / (1 + np.exp(-x))
    def evaluate(self, state,action):
        inputs_text=self.template.format(state,action)
        with torch.no_grad():
            inputs =self.tokenizer(inputs_text, return_tensors="pt",padding=False, truncation=True,return_attention_mask=False, max_length=1024)
            _,_,outputs = self.model(inputs['input_ids'].cuda())
            chosen_length = (inputs['input_ids'][0] !=self.tokenizer.pad_token_id).nonzero()[-1] + 1
            reward=float(outputs[0][chosen_length - 1])
        return self.sigmoid(reward)

