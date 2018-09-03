# -*- coding: utf-8 -*-

import numpy as np

from features.word2vec import Word2Vec
from features.ppdb_scores import PPDBScores
from features.wiki_frequency import WikiFrequency
from features.lexicon import WordComplexityLexicon
from features.gaussian_binner import GaussianBinner
from features.google_frequency import GoogleFrequency
from features.syllable_counter import SyllableCounter
from features.ngram_probabilities import NgramProbability


class FeatureExtractorSR:

    def __init__(self, resources):
        self.syllable_counter = SyllableCounter()
        self.ngram = NgramProbability(resources.lm)
        self.ppdb_score = PPDBScores(resources.ppdb)
        self.word2vec = Word2Vec(resources.word2vec)
        self.wiki_frequency = WikiFrequency(resources.wiki)
        self.google_frequency = GoogleFrequency(resources.google)
        self.complexity_lexicon = WordComplexityLexicon(resources.lexicon)
        self.binner = GaussianBinner()

    def get_features(self, corpus_path, train_flag):
        all_features, all_ranks = [], []
        for line in open(corpus_path):
            tokens = line.strip().split('\t')
            sentence = tokens[0].strip()
            target = tokens[1].strip()
            candidates = [token.strip().split(':')[1].strip() for token in tokens[3:]]
            ranks = [int(token.strip().split(':')[0].strip()) for token in tokens[3:]]
            featmap = self._get_instance_features(candidates, sentence, target)

            for i in range(0, len(candidates)):
                fv1 = featmap[candidates[i]]
                for j in range(0, len(candidates)):
                    if i != j:
                        fv2 = featmap[candidates[j]]
                        pairwise_features = self._get_pairwise_features(fv1, fv2, candidates[i], candidates[j])
                        all_features.append(pairwise_features)
                        all_ranks.append(ranks[i] - ranks[j])

        all_features = self._transform_features(all_features, train_flag)
        print("Number of features: %d, Feature vector size: %d " % (len(all_features), len(all_features[0])))
        return all_features, all_ranks

    def _get_instance_features(self, candidates, sentence, target):
        instance_features = {}
        for candidate in candidates:
            words = self._tokenize(candidate)
            features = list()
            features.append(len(words))
            features.append(sum([len(w) for w in words]))
            features.append(self._get_num_syllables(words))
            features.extend(self.ngram.get_features(sentence, target, candidate, words))
            features.append(self.google_frequency.get_feature(candidate))
            features.append(self.wiki_frequency.get_feature(candidate))
            features.extend(self.complexity_lexicon.get_feature(words))
            instance_features[candidate] = features
        return instance_features

    def _get_pairwise_features(self, v1, v2, phrase1, phrase2):
        features = []
        features.extend(v1)
        features.extend(v2)
        features.extend([f1 - f2 for f1, f2 in zip(v1, v2)])

        words1, words2 = self._tokenize(phrase1), self._tokenize(phrase2)
        p1 = self.word2vec.get_word2vec_vector(words1)
        p2 = self.word2vec.get_word2vec_vector(words2)
        features.append(self.word2vec.get_cosine(p1, p2))
        features.append(self.ppdb_score.get_feature(phrase1, phrase2))
        return features

    def _transform_features(self, features, train_flag):
        features = np.array(features)
        if train_flag:
            self.binner.fit(features, len(features[0]))
        return self.binner.transform(features, len(features[0]))

    def _get_num_syllables(self, words):
        return sum([self.syllable_counter.get_feature(word) for word in words])

    @staticmethod
    def _tokenize(phrase):
        return phrase.lower().split()
