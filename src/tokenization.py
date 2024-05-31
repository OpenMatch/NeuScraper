# Copyright (c) 2023 OpenMatch
# Author: Zhipeng Xu
# All rights reserved.

import errno
import os

from transformers import (
    BertTokenizer,
    XLMRobertaTokenizer,
)

# Model's config. Models used for inference embedding vector
config_cls = []

class BaseConfig:
    path = None
    model_class = None
    tokenizer_class = BertTokenizer
    tokenizer_do_lower_case = True
    use_mean = True
    temperature = 0.1
    embedding_dim = 768

    def check(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.path)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        config_cls.append(cls)

class XLMRoberta(BaseConfig):
    tokenizer_class = XLMRobertaTokenizer

ConfigDict = {cfg.__name__: cfg for cfg in config_cls}

class TokenizerProcessor():
    def __init__(self, max_token_length):
        self.model_name_or_path = "xlm-roberta-base"
        self.max_token_len = max_token_length
        self.model_type = "XLMRoberta"
        self.add_special_tokens = True
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = self._load_preprocess_model(self.model_type, self.model_name_or_path)

    def _load_preprocess_model(self, model_type, model_name_or_path):
        configObj = ConfigDict[model_type]()

        tokenizer = configObj.tokenizer_class.from_pretrained(
            model_name_or_path,
            do_lower_case=configObj.tokenizer_do_lower_case,
            cache_dir=None,
        )
        return tokenizer

    def _text_to_token(self, tokenizer, text_batch, max_token_len, padding = "max_length"):
        tokens = tokenizer.encode_plus(
            text_batch,
            add_special_tokens = self.add_special_tokens,
            max_length = max_token_len,
            padding = padding,
            truncation = True)
        return tokens
    
    def tokenize_sequence(self, text):
        text_ids = self._text_to_token(self.tokenizer, text, self.max_token_len)['input_ids']
        return text_ids