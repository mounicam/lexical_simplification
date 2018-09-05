# A Word-Complexity Lexicon and A Neural Readability Ranking Model for Lexical Simplification

This repository contains the code and resources from the following [paper](https://mounicam.github.io/WC_Lexicon_NRR.pdf)


## Repo Structure: 
1. ```word_complexity_lexicon```: Lexicon with complexity scores for ~15000 most frequent words from Google Ngram Corpus.
The scores are calculated by aggregating over human ratings. We release both the aggregated ratings and the individual 
ratings by each annotator.

1. ```SimplePPDBpp```: SimplePPDB++ resource consisting of around 14.1 million paraphrase rules along with 
their readability scores. 

1. ```neural_readability_ranker```: Code for our neural readability ranker model.

## Citation
Please cite if you use the above resources for your research
```
@InProceedings{EMNLP-2018-Maddela,
  author = 	"Maddela, Mounica and Xu, Wei",
  title = 	"A Word-Complexity Lexicon and A Neural Readability Ranking Model for Lexical Simplification",
  booktitle = 	"Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)",
  year = 	"2018",
}
```