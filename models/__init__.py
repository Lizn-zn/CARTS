from .codellama import CodeLlamaTacticGen
from .reprover import ReproverTacticGen
from .internlm import InternLMTacticGen,InternLMStepproverTacticGen

MODELS={
    'codellama' : CodeLlamaTacticGen,
    'reprover' : ReproverTacticGen,
    'internlm': InternLMTacticGen,
    'internlm_stepprover':InternLMStepproverTacticGen
}
