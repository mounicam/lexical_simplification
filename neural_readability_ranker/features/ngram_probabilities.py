# -*- coding: utf-8 -*-

import kenlm
import numpy as np
from nltk import ngrams

"""
Based on feature extraction module of LEXenstein
https://github.com/ghpaetzold/LEXenstein/blob/master/lexenstein/features.py
"""


class NgramProbability:

    def __init__(self, lm, left=2, right=2):
        self.language_model = kenlm.LanguageModel(lm)
        self.left = left
        self.right = right

    def _calculate_instance_features(self, sent, head, candidate):
        sent_tokens = sent.strip().split()
        values = []
        for span1 in range(0, self.left + 1):
            for span2 in range(0, self.right + 1):
                ngram, bosv, eosv = self._get_ngram(candidate, sent_tokens, head, span1, span2)
                values.append(self.language_model.score(ngram, bos=bosv, eos=eosv))
        return values

    @staticmethod
    def _get_ngram(candidate, tokens, head, config_l, config_r):
        if config_l == 0 and config_r == 0:
            return candidate, False, False
        else:
            result_tokens = []
            for i in range(max(0, head - config_l), head):
                result_tokens.append(tokens[i])
            result_tokens.append(candidate)
            for i in range(head + 1, min(len(tokens), head + config_r + 1)):
                result_tokens.append(tokens[i])

            return " ".join(result_tokens), max(0, head - config_l) == 0, \
                   min(len(tokens), head + config_r + 1) == len(tokens)

    def get_features(self, sentence, target, phrase, words):
        index = sentence.lower().split().index(target)
        if len(words) == 1:
            return self._calculate_instance_features(sentence, index, phrase)
        else:
            ngram_probs = []
            for i in range(0, len(words)):
                for ngram in ngrams(words, i+1):
                    if len(ngram) <= 5:
                        ngram_probs.append(self._calculate_instance_features(sentence, index, " ".join(ngram)))
            return np.average(np.array(ngram_probs), axis=0).tolist()