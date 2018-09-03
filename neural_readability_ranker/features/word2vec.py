# -*- coding: utf-8 -*-

import gensim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Word2Vec:
    def __init__(self, word2vec_file):
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
        self.vocab = self.word2vec.vocab
        self.size = self.word2vec.vector_size

    def get_word2vec_vector(self, words):
        word2vec_words = [w.lower() for w in words if w in self.vocab]
        if len(word2vec_words) > 0:
            embeddings = np.array([self.word2vec.word_vec(word) for word in word2vec_words])
        else:
            embeddings = [[0] * self.size]
        return np.average(embeddings, axis=0).tolist()

    @staticmethod
    def get_cosine(f1, f2):
        return cosine_similarity([f1], [f2])[0][0]
