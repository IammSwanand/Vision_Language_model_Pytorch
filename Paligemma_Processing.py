from typing import Dict, list, Tuple, Optional, Union, Iterable
import numpy as np
from PIL import Image
import torch
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

class PaliGemmaProcessor:
    
    def __init__(self, tokenizer, num_image_token: int, image_size: int):
        super().__init__()

        self.image_seq_length = num_image_token
        self.image_size = image_size

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]
        EXTRA_TOKENS += [
            f"<img{i:03d}>" for i in range(128)
        ]
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer