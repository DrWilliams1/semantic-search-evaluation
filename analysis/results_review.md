# Results review

## Recorded results

Quality was evaluated on **all 300 official test queries** against **5,183 documents**. Each query receives
equal weight. Higher quality scores are better; these retrieval metrics are not classification accuracy.

| Pipeline | nDCG@10 | MRR@10 | Recall@5 | Recall@10 | Recall@50 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.6519 | 0.6186 | 0.7284 | 0.7740 | 0.8657 |
| Dense embeddings | 0.6451 | 0.6047 | 0.7379 | 0.7833 | 0.8920 |
| BM25 + reranker | 0.6818 | 0.6529 | 0.7301 | 0.8029 | 0.8657 |

Warm latency was measured separately on **20 development queries × 3 repeats per pipeline**, with randomized
trial order and synchronized GPU timing. Lower time is better; p95 is a latency percentile, not an uncertainty interval.

| Pipeline | Median (ms) | p95 (ms) | Trials |
|---|---:|---:|---:|
| BM25 | 26.32 | 44.93 | 60 |
| Dense embeddings | 12.74 | 20.23 | 60 |
| BM25 + reranker | 310.25 | 348.62 | 60 |

Reranking produced the highest test nDCG@10, improving over BM25 by **0.0299**, while requiring
**11.79×** its median query time. Dense retrieval was fastest, but its nDCG@10 was **0.0068 lower**
than BM25's. Dense retrieval had the highest Recall@5 and Recall@50. No pipeline dominates every quality measure.

BM25 and BM25 + reranker have identical Recall@50 (**0.8657**) because they contain exactly the same 50 candidates.
Reranking changes order; it cannot add a missed document. Its mean gain includes **60 improved, 47 worsened,
and 193 tied** test queries on nDCG@10.

![Test quality and warm development-query latency](results/search_20261118T185027Z-4f8afe/quality_latency.png)

## Uncertainty and practical interpretation

Intervals use 2,000 paired query-bootstrap resamples with seed 2026.

| Comparison | Mean nDCG@10 change | Paired 95% interval | Improved / worsened / tied queries |
|---|---:|---|---|
| Dense embeddings minus BM25 | -0.0068 | [-0.0454, +0.0308] | 70 / 71 / 159 |
| BM25 + reranker minus BM25 | +0.0299 | [+0.0014, +0.0578] | 60 / 47 / 193 |
| BM25 + reranker minus Dense embeddings | +0.0367 | [+0.0021, +0.0739] | 78 / 57 / 165 |

The dense-versus-BM25 interval spans zero. The two reranking intervals are above zero in this recorded
bootstrap analysis, with lower bounds close to zero. They support a conditional descriptive comparison,
not a guarantee across datasets or a claim that reranking helps every query.

For this run, dense retrieval offers the lowest warm latency; reranking offers the highest primary
retrieval score at substantially greater query time. A deployment choice would require a latency budget,
the cost of missed evidence, and representative target-domain queries.

## Selected development errors

Five development queries were inspected to illustrate gains, ordering errors, and candidate misses.
They are selected examples, not a random sample of causes. Observations and reference-document ranks
are in [analysis/error_analysis.csv](analysis/error_analysis.csv). The original 50-row worksheet is preserved unchanged.
No relevance labels were edited and no independent annotation-reliability study was performed.

| Development query | Relevant-document ranks: BM25 → reranked (dense) | Observation |
|---|---|---|
| 1139 | 9 → 1 (2) | Reranking promoted the trial abstract relevant to the information need. |
| 498 | 7 → 1 (1) | Both neural pipelines found relevant counterevidence; relevance does not imply the query claim is true. |
| 391 | 3 → 14 (2) | Reranking moved a retrieved relevant document out of the top 10. |
| 310 | 5 → 13 (>50) | A reranker ordering error and a dense candidate miss on the same query. |
| 1109 | >50 → >50 (>50), for both relevant documents | Both initial retrieval methods missed the judged documents; reordering BM25's pool could not recover them. |

These rank changes are observable. Topical overlap and input truncation are possible contributors,
but this experiment does not establish their causal role in an individual error.

## Limits

The dense encoder uses at most **256 tokens** and truncated **3,681 of 5,183 documents (71.02%)**.
Reranking uses at most **512 tokens per query/document pair** and truncated **2,037 of 15,000 test pairs (13.58%)**
and **289 of 2,500 development pairs (11.56%)**. No evaluated dense query was truncated.
BM25 uses full title-plus-abstract text. These are comparisons of the declared pipelines,
not an isolated causal test of keyword versus neural architectures or the effect of truncation.

BM25 runs on CPU through `rank-bm25`; neural inference uses a Tesla T4, with dense vector comparison on CPU.
The dense pipeline's observed speed advantage is specific to this implementation, corpus, and hardware.
Warm timings exclude downloads, loading, and document indexing, and do not establish concurrent-user
throughput or production latency. Twenty queries repeated three times are not 60 independent queries.

The benchmark's judgments may be incomplete: unjudged documents score as nonrelevant without proving
they are irrelevant. Model pretraining exposure was not audited. Bootstrap intervals describe query
sampling conditional on this corpus, models, and labels; they exclude annotation uncertainty, hardware
variation, and domain shift. The three comparisons are descriptive and not adjusted for multiple testing.
No generated-answer accuracy, claim truth, or performance on policy/maritime documents was measured.
