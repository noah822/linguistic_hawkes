from .lazyio import LazyFileReader
from typing import (
    Dict, List,
    Tuple
)
import spacy

class Scanner:
    def __init__(self,
                 vocab: str='en_core_web_sm',
                 ignore_list: List[str]=None):
        self.ignore_list = ignore_list
        self.NLP = spacy.load(vocab)
    
    def _check_valid(self, char: str):
        stripped = char.strip()
        if stripped in self.ignore_list:
            return False
        else:
            return True

    # return 
    def count_pos(self,
                  file_loader: LazyFileReader
                ) -> Tuple[Dict[str, List[int]], int]:
        res = {}
        pos_idx = 0
        for chunk in file_loader:
            tokenzied_chunk = self.NLP(chunk)
            # tokenize count 
            for i in tokenzied_chunk:
                char = i.text
                if not self._check_valid(char):
                    continue
                holder = res.get(char, [])
                holder.append(pos_idx)
                res[char] = holder
                pos_idx += 1
        return res, pos_idx
                