# -*- coding: utf-8 -*-

import numpy as np

from features.word2vec import Word2Vec
from features.ppdb_scores import PPDBScores
from features.wiki_frequency import WikiFrequency
from features.lexicon import WordComplexityLexicon
from features.google_frequency import GoogleFrequency
from features.syllable_counter import SyllableCounter
from features.gaussian_binner import GaussianBinner


class FeatureExtractorSimplePPDB:

    def __init__(self, resources):
        self.syllable_counter = SyllableCounter()
        self.ppdb_score = PPDBScores(resources.ppdb)
        self.word2vec = Word2Vec(resources.word2vec)
        self.wiki_frequency = WikiFrequency(resources.wiki)
        self.google_frequency = GoogleFrequency(resources.google)
        self.complexity_lexicon = WordComplexityLexicon(resources.lexicon)
        self.binner = GaussianBinner()

    def get_corpus_features(self, corpus_path, train_flag):
        count = 0
        y = []
        feat_map_single, feat_map_pair = {}, {}
        instance_features_to_be_binned, instance_features_not_to_be_binned = [], []

        for line in open(corpus_path):
            tokens = line.strip().split('\t')
            phrase1, phrase2, label = tokens[0], tokens[1], int(tokens[2])
            words1 = self._tokenize(phrase1)
            words2 = self._tokenize(phrase2)

            if phrase1 not in feat_map_single:
                feat_map_single[phrase1] = self._get_numeric_features(phrase1, words1)
            num_features1 = feat_map_single[phrase1]

            if phrase2 not in feat_map_single:
                feat_map_single[phrase2] = self._get_numeric_features(phrase2, words2)
            num_features2 = feat_map_single[phrase2]

            if (phrase1, phrase2) not in feat_map_pair:
                feat_map_pair[(phrase1, phrase2)] = self._get_pairwise_features(phrase1, phrase2, num_features1,
                                                           num_features2, words1, words2)
            pairwise_features = feat_map_pair[(phrase1, phrase2)]

            y.append(label)
            instance_features_to_be_binned.append(num_features1 + num_features2 + pairwise_features[0])
            instance_features_not_to_be_binned.append(pairwise_features[1])

            count += 1
            if count % 100000 == 0:
                print(count)

        instance_features_to_be_binned = self._transform_features(instance_features_to_be_binned, train_flag).tolist()
        instance_features = [i1 + i2 for i1, i2 in zip(instance_features_to_be_binned,
                                                       instance_features_not_to_be_binned)]
        print(len(instance_features), len(instance_features[0]))
        return instance_features, y

    def _get_numeric_features(self, phrase, words):
        features = list()
        features.append(len(words))
        features.append(sum([len(w) for w in words]))
        features.append(self._get_num_syllables(words))
        features.append(self.google_frequency.get_feature(phrase))
        features.append(self.wiki_frequency.get_feature(phrase))
        features.extend(self.complexity_lexicon.get_feature(words))
        return features

    def _get_pairwise_features(self, phrase1, phrase2, num_vec1, num_vec2, words1, words2):
        p1 = self.word2vec.get_word2vec_vector(words1)
        p2 = self.word2vec.get_word2vec_vector(words2)

        diff_features = []
        diff_features.extend([f1 - f2 for f1, f2 in zip(num_vec1, num_vec2)])
        diff_features.append(self.word2vec.get_cosine(p1, p2))
        diff_features.append(self.ppdb_score.get_feature(phrase1, phrase2))

        vec_features = [f1 - f2 for f1, f2 in zip(p1, p2)]

        return diff_features, vec_features

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
