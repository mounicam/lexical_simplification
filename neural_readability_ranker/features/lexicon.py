# -*- coding: utf-8 -*-

from nltk import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer


class WordComplexityLexicon:
    def __init__(self, lexicon):
        word_ratings = {}
        for line in open(lexicon):
            tokens = [t.strip() for t in line.strip().split('\t')]
            word_ratings[tokens[0].lower()] = float(tokens[1])
        self.word_ratings = word_ratings

        self.lemmatizer = WordNetLemmatizer()
        self.lancaster_stemmer = LancasterStemmer(strip_prefix_flag=True)
        self.snowball_stemmer = SnowballStemmer("english")

    def get_feature(self, words):
        phrase = max(words, key=len)

        if phrase in self.word_ratings:
            return [self.word_ratings[phrase], 1.0]
        else:
            ratings = []
            lemman = self.lemmatizer.lemmatize(phrase, pos='n')
            lemmav = self.lemmatizer.lemmatize(phrase, pos='v')
            lemmaa = self.lemmatizer.lemmatize(phrase, pos='a')
            lemmar = self.lemmatizer.lemmatize(phrase, pos='r')
            stem_lan = self.lancaster_stemmer.stem(phrase)
            stem_snow = self.snowball_stemmer.stem(phrase)

            if lemman in self.word_ratings:
                ratings.append(self.word_ratings[lemman])
            elif lemmav in self.word_ratings:
                ratings.append(self.word_ratings[lemmav])
            elif lemmaa in self.word_ratings:
                ratings.append(self.word_ratings[lemmaa])
            elif lemmar in self.word_ratings:
                ratings.append(self.word_ratings[lemmar])
            elif stem_snow in self.word_ratings:
                ratings.append(self.word_ratings[stem_snow])
            elif stem_lan in self.word_ratings and abs(len(stem_lan) - len(phrase)) <= 2:
                ratings.append(self.word_ratings[stem_lan])

            if len(ratings) > 0:
                return [max(ratings)*1.0, 1.0]

        return [0.0, 0.0]
