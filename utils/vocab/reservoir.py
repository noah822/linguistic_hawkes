import spacy

from typing import (
    List, Tuple,
    Dict
)
from collections import namedtuple, OrderedDict

# wraps a given word into token, from which 
# several linguistic information can be extracted
class TaggedToken:
    # define class-level variable 
    NLP = spacy.load('en_core_web_sm')

    # https://universaldependencies.org/u/pos/all.html#al-u-pos/ADP
    # check this for details
    WORD_CLASS_ENUM = [
        'AUX', # aux
        'PROPN',# proper noun
        'VERB', # verb
        'ADP', # adposition = prepositon + postposition
        'SYM', # symbol
        'NUM' # number
    ]
    def __init__(self, token: str):
        self._hooked_token = TaggedToken.NLP(token)[0]
    
    @property
    def word_class(self):
        return self._hooked_token.pos_
    

import copy
from ._internal import bisect
import json

from termcolor import colored

# propery associated with each token that will be tracked
_TokenProperty = namedtuple('_TokenProperty', ['word_class', 'freq', 'pos'])
class Reservoir:
    '''
        reservoir of vocabulary associated with document
    '''
    def __init__(self,
                 token_freq: Dict[str, int]=None,
                 token_pos: Dict[str, List[int]]=None,
                 save_path: str=None,
                 from_json: str=None):
        
        if from_json is not None:
            with open(from_json, 'r') as handler:
                self._token_dict = json.load(handler)
            
            self._token_freq = []
            for prop in self._token_dict.values():
                self._token_freq.append(prop[1])
            self._token_freq = sorted(
                self._token_freq,
                key=lambda v: -v
            )
        else:
            self._token_dict: Dict[str, _TokenProperty] = {}

            # main another two dict for fast query

            # general token dict on the fly
            for literal in token_freq.keys():
                associated_prop = _TokenProperty(
                    TaggedToken(literal).word_class, 
                    token_freq[literal],
                    token_pos[literal]
                )
                self._token_dict[literal] = associated_prop

                self._token_freq = sorted(
                    copy.deepcopy(list(token_freq.values()))
                )
            if save_path is not None:
                with open(save_path, 'w') as handler:
                    json.dump(self._token_dict, handler)

        # sort array for fast interval query
        self._token_dict = OrderedDict(sorted(
            self._token_dict.items(),
            key=lambda k_v: -k_v[-1][1]
        ))

    def fine_grained_query(self,
              freq_range: Tuple[int, int]=None,
              word_class: str=None,
              num_display: 10=None):
    
        # if word_class is not None:
        #     assert word_class in TaggedToken.WORD_CLASS_ENUM
        
        if freq_range is not None:
            lvalue, rvalue = freq_range
            # bisect ordered token dict O(logN)
            right_idx = bisect(
                self._token_freq, lvalue, 0, len(self._token_freq),
                key=lambda x, y: x>y
            )

            left_idx = bisect(
                self._token_freq, rvalue, 0, len(self._token_freq),
                key=lambda x, y: x>y
            )
        else:
            left_idx, right_idx = 0, num_display


        # iterate through 
        displayed_cnt = 0
        ordered_items = list(self._token_dict.items())
        for i in range(left_idx, right_idx):
            if displayed_cnt == num_display:
                break

            literal, prop = ordered_items[i]
            if word_class is not None:
                if word_class == prop[0]:
                    Reservoir._token_pretty_print(
                        literal, prop
                    )
                    displayed_cnt += 1
                else:
                    continue
            else:
                Reservoir._token_pretty_print(
                    literal, prop
                )
                displayed_cnt += 1
    
    @staticmethod
    def _token_pretty_print(literal: str,
                            token_prop: _TokenProperty):
        display = (f'''{literal}:\t{colored('[class]', 'red')} {token_prop[0].ljust(8)}\t\
{colored('[freq]', 'green')} {token_prop[1]}''')
        print(display)

    # route other dict-related method to internal dict
    def keys(self):
        return self._token_dict.keys()
    def items(self):
        return self._token_dict.items()

    def __getitem__(self, key: str):
        return _TokenProperty(*self._token_dict[key])
    def __len__(self):
        return len(self._token_dict)
