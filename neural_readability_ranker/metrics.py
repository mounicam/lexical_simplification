from scipy.stats import pearsonr
from sklearn.metrics import precision_score, accuracy_score


def evaluate_ranker(test_corpus, pred_rankings):
    corpus_lines = open(test_corpus).readlines()

    correct, total = 0, 0
    all_pred_ranks, all_gold_ranks = [], []
    for line, pred in zip(corpus_lines, pred_rankings):
        subs = line.strip().split('\t')[3:]

        gold_rankings = {}
        gold_simplest = set()
        for sub in subs:
            rank, word = sub.split(':')
            rank = int(rank)
            gold_rankings[word] = rank
            if rank == 1:
                gold_simplest.add(word)

        if pred[0] in gold_simplest:
            correct += 1
        total += 1

        for rank, word in enumerate(pred):
            all_gold_ranks.append(gold_rankings[word])
            all_pred_ranks.append(rank)

    p_at_1 = (1.0 * correct) / total
    pearson = pearsonr(all_pred_ranks, all_gold_ranks)[0]
    return p_at_1, pearson


def evaluate_simpleppdb_classifier(test_y_predict, test_y_true):
    precision_each = precision_score(test_y_true, test_y_predict, average=None)
    accuracy = accuracy_score(test_y_true, test_y_predict)
    # Accuracy, P@-1 and P@1
    return accuracy, precision_each[0], precision_each[2]
