# -*- coding: utf-8 -*-


class WikiFrequency:
    def __init__(self, wiki_frequency_file):
        wiki_frequencies = {}
        for line in open(wiki_frequency_file):
            tokens = [t.strip() for t in line.strip().split('\t')]
            wiki_frequencies[tokens[0]] = float(tokens[1])
        self.wiki_frequencies = wiki_frequencies

    def get_feature(self, phrase):
        phrase = phrase.lower()
        if phrase in self.wiki_frequencies:
            return self.wiki_frequencies[phrase]
        return 0
