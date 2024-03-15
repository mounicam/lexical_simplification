# -*- coding: utf-8 -*-

import argparse

from nrr import NRR
from metrics import evaluate_ranker
from features.feature_extractor_sr import FeatureExtractorSR

"""
This script performs substitution ranking using neural readability ranking model and word-complexity lexicon. For more 
details, please take a look at the paper - "A Word Complexity Lexicon and A Neural Readability Ranking Model for Lexical
Simplification" https://aclanthology.org/D18-1410.pdf.


Arguments
----------
--train   : Training data file. Each line consists of a sentence followed by the target word, position of target word 
            in the sentence and candidates for the target word prefixed by their rank.
            Eg: <sentence>  <target>    <postition of target word in sentence>  <rank:candidate1>   <rank:candidate2>...
            Please see sample_data/substitution_ranking.txt for sample data.
--test    : Test data file. Format is same as the training data file.
--lm      : Language model file in binary format.
--lexicon : Same format as our word-complexity lexicon file. Each line in the file consists of a word and its 
            complexity score separated by tabs.
            Eg: <word>  1.2
--word2vec: Word2vec binary file. We used Google word2vec vectors (https://code.google.com/archive/p/word2vec/)
--google  : Frequencies of words/phrases from Google Ngram corpus. Each line in the file consists of a word and its 
            frequency seperated by tabs. 
            Eg: <word>  123
--wiki    : Log ratio of the probabilities of observing a word/phrase in Simple Wikipedia to Normal Wikiepdia. For more
            details please check this paper - https://cs.brown.edu/people/epavlick/papers/style_for_paraphrasing.pdf
            Each line in the file consists of a word and its ratio seperated by tabs.
            Eg: <word>  0.2
--ppdb    : PPDB 2.0 scores for paraphrase rules in train and test data. Each line in the file consists of a paraphrase 
            rule and its PPDB 2.0 score separated by tabs.
            Eg:  <phrase1>  <phrase2> 3.5
"""
parser = argparse.ArgumentParser()
parser.add_argument('--train', help="Training data file")
parser.add_argument('--test', help="Test data file")
parser.add_argument('--lm', help="Language model file")
parser.add_argument('--lexicon', help="Word Complexity lexicon")
parser.add_argument('--word2vec', help="Word2vec binary file")
parser.add_argument('--google', help="Frequencies from Google Ngram corpus")
parser.add_argument('--wiki', help="Log probability ratios from Simple and Normal Wikipedia")
parser.add_argument('--ppdb', help="PPDB 2.0 scores")


def main():
    print("Loading resources")
    feat_extractor = FeatureExtractorSR(args)

    print("Extracting training data features")
    train_x, train_y = feat_extractor.get_features(args.train, True)

    print("Extracting test data features")
    test_x, test_y = feat_extractor.get_features(args.test, False)

    print("Training NRR model")
    nrr = NRR(train_x, train_y, dropout)
    nrr.train(epochs, lr)

    print("Ranking test data substitutions using NRR")
    nrr.set_testing()
    prediction_scores = [score[0] for score in nrr.predict(test_x).data.numpy()]

    count = -1
    pred_rankings = []
    for line in open(args.test):
        line = line.strip().split('\t')
        substitutes = [sub.strip().split(':')[1].strip() for sub in line[3:]]
        score_map = {}
        for sub in substitutes:
            score_map[sub] = 0.0

        for s1 in substitutes:
            for s2 in substitutes:
                if s1 != s2:
                    count += 1
                    score = prediction_scores[count]
                    score_map[s1] += score

        pred_rankings.append(sorted(score_map.keys(), key=score_map.__getitem__))

    print("Evaluation")
    p_at_1, pearson = evaluate_ranker(args.test, pred_rankings)
    print("Metrics (P@1, Pearson): %f %f" % (p_at_1*100, pearson))


if __name__ == '__main__':
    args = parser.parse_args()
    lr = 0.0005
    epochs = 100
    dropout = 0.2

    main()
