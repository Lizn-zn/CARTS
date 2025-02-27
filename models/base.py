from lean_dojo import Pos
from typing import List,Tuple
class BaseTacticGen():  
    
    def __init__(self,path) -> None:
        super().__init__()
        self.path=path
        
    def generate(
            self,
            state: str,
            file_path: str,
            theorem_full_name: str,
            theorem_pos: Pos,
            num_samples: int,
        ) -> List[Tuple[str, float]]:
        pass