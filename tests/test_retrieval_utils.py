import math
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval_utils import (ranked_indices, query_metrics, evaluate_rankings,
                             paired_bootstrap, tokenize, summarize_latency, choose_development_queries)


class RetrievalTests(unittest.TestCase):
    def test_perfect_ranking(self):
        result = query_metrics(['a', 'b', 'x'], ['a', 'b'])
        self.assertEqual(result['ndcg@10'], 1)
        self.assertEqual(result['mrr@10'], 1)
        self.assertEqual(result['recall@5'], 1)

    def test_second_rank_and_missed_relevant(self):
        result = query_metrics(['x', 'a', 'y'], ['a', 'b'])
        self.assertAlmostEqual(result['ndcg@10'], (1 / math.log2(3)) / (1 + 1 / math.log2(3)))
        self.assertEqual(result['mrr@10'], 0.5)
        self.assertEqual(result['recall@10'], 0.5)

    def test_no_hits(self):
        self.assertTrue(all(v == 0 for v in query_metrics(['x'], ['a']).values()))

    def test_cutoffs(self):
        ranking = [str(i) for i in range(50)]
        result = query_metrics(ranking, ['10'])
        self.assertEqual(result['ndcg@10'], 0)
        self.assertEqual(result['mrr@10'], 0)
        self.assertEqual(result['recall@10'], 0)
        self.assertEqual(result['recall@50'], 1)

    def test_invalid_rankings(self):
        with self.assertRaises(ValueError): query_metrics(['a', 'a'], ['a'])
        with self.assertRaises(ValueError): query_metrics(['a'], [])

    def test_ties_use_document_id(self):
        self.assertEqual(list(ranked_indices([1, 2, 2], ['c', 'b', 'a'], 3)), [2, 1, 0])
        self.assertEqual(list(ranked_indices([0, 0], ['2', '10'], 2)), [1, 0])

    def test_invalid_scores_and_k(self):
        for scores, ids, k in [([float('nan')], ['a'], 1), ([1], ['a'], 2), ([1, 2], ['a', 'a'], 1)]:
            with self.assertRaises(ValueError): ranked_indices(scores, ids, k)

    def test_macro_average(self):
        summary, rows = evaluate_rankings({'1': {'ids': ['a']}, '2': {'ids': ['x']}},
                                          {'1': ['a'], '2': ['b']})
        self.assertEqual(summary['ndcg@10'], 0.5)
        self.assertEqual(len(rows), 2)

    def test_missing_queries_rejected(self):
        with self.assertRaises(ValueError): evaluate_rankings({'1': {'ids': ['a']}}, {'2': ['a']})

    def test_bootstrap_pairing(self):
        left = [{'query_id': str(i), 'ndcg@10': 0.2} for i in range(4)]
        right = [{'query_id': str(i), 'ndcg@10': 0.7} for i in reversed(range(4))]
        result = paired_bootstrap(left, right, repeats=100)
        self.assertAlmostEqual(result['mean_delta'], 0.5)
        self.assertAlmostEqual(result['ci95_low'], 0.5)
        self.assertEqual(result['queries_improved'], 4)

    def test_tokenizer(self):
        self.assertEqual(tokenize('Gene-3, SCIENCE!'), ['gene', '3', 'science'])

    def test_latency_summary(self):
        rows = [{'method': m, 'milliseconds': v} for m in ('bm25','dense','bm25_reranked') for v in (1,3,5)]
        self.assertTrue(all(row['median_ms'] == 3 for row in summarize_latency(rows)))

    def test_development_excludes_normalized_test_text(self):
        data = {'queries': {'a':'Shared CLAIM', 'b':'different claim', 'c':'  shared   claim '},
                'train_qrels': {'a':['1'], 'b':['2']}, 'test_qrels': {'c':['1']}}
        self.assertEqual(choose_development_queries(data, 1, 42), ['b'])


if __name__ == '__main__':
    unittest.main()
