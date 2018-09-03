# -*- coding: utf-8 -*-


class PPDBScores:
    def __init__(self, ppdb_score_file):
        ppdb_scores = {}
        for line in open(ppdb_score_file):
            tokens = [t.strip().lower() for t in line.strip().split('\t')]
            ppdb_scores[(tokens[0], tokens[1])] = float(tokens[2])
        self.ppdb_scores = ppdb_scores

    def get_feature(self, phrase1, phrase2):
        phrase1, phrase2 = phrase1.lower(), phrase2.lower()
        if (phrase1, phrase2) in self.ppdb_scores:
            return self.ppdb_scores[(phrase1, phrase2)]
        return -1
