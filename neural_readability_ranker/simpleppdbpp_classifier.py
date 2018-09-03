import argparse

from nrr import NRR
from metrics import evaluate_simpleppdb_classifier
from features.feature_extractor_simpleppdb import FeatureExtractorSimplePPDB

"""
This script creates and runs the SimplePPDB++ classifier, which classifies a paraphrase rule as 'simplifying', 
'complicating' or 'nonsense' rule. For more details, please take a look at the paper - "A Word Complexity Lexicon and 
A Neural Readability Ranking Model for Lexical Simplification" (TODO: link).


Arguments
----------
--train   : Training data file. Each line consists of a paraphrase rule and label (-1/0/1) separated by tabs.
            Eg:  <phrase1>  <phrase2> -1
            Please see sample_data/simpleppdbpp.txt for sample data.
--test    : Test data file. Format is same as the training data file.
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
parser.add_argument('--lexicon', help="Word Complexity lexicon")
parser.add_argument('--word2vec', help="Word2vec binary file")
parser.add_argument('--google', help="Frequencies from Google Ngram corpus")
parser.add_argument('--wiki', help="Log probability ratios from Simple and Normal Wikipedia")
parser.add_argument('--ppdb', help="PPDB 2.0 scores")


def get_final_label(test_x, ranker, threshold):
    scores = [score[0] for score in ranker.predict(test_x).data.numpy()]
    labels = []
    for score in scores:
        if -threshold < score < threshold:
            labels.append(0)
        elif score < -threshold:
            labels.append(-1)
        else:
            labels.append(1)
    return labels


def neural_ranker_classifier(x_train, y_train, x_test):
    ranker = NRR(x_train, y_train, dropout)
    ranker.train(epochs, learning_rate)
    ranker.set_testing()
    return get_final_label(x_test, ranker, threshold)


def main():
    feature_extractor = FeatureExtractorSimplePPDB(args)
    train_x, train_y = feature_extractor.get_corpus_features(args.train, True)
    test_x, test_y = feature_extractor.get_corpus_features(args.test, False)
    test_y_predict = neural_ranker_classifier(train_x, train_y, test_x)
    print(evaluate_simpleppdb_classifier(test_y_predict, test_y))


if __name__ == '__main__':
    epochs = 100
    threshold = 0.4
    learning_rate = 0.01
    dropout = 0.2

    args = parser.parse_args()
    main()
