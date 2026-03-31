import numpy as np
from typing import List, Dict
class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3
        
        self.id_to_word[0] = self.pad_token
        self.id_to_word[1] = self.unk_token
        self.id_to_word[2] = self.bos_token
        self.id_to_word[3] = self.eos_token
        counter = 4
        for text in texts:
            text = text.replace(",", " ")
            text = text.replace(".", " ")
            text = text.replace("!", " ")
            text = text.replace("?", " ")
            words = set(text.split())
            for word in words:
                if word not in self.word_to_id:
                    self.word_to_id[word] = counter
                    self.id_to_word[counter] = word
                    counter+=1
            self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        result = []
        text = text.replace(",", " ")
        text = text.replace(".", " ")
        text = text.replace("!", " ")
        text = text.replace("?", " ")
        splitted_text = text.split()
        for word in splitted_text:
            if word in self.word_to_id:
                result.append(self.word_to_id[word])
            else:
                result.append(self.word_to_id[self.unk_token])
        return result
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        result = str()
        for index,value in enumerate(ids):
            if index != 0:
                result+= ' '
            if value in self.id_to_word:
                result+=self.id_to_word[value]
            else:
                result+=self.unk_token
        return result
