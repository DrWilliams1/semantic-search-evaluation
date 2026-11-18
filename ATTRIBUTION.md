# Attribution and changes

## Starting point

Jay Alammar and Maarten Grootendorst, *Hands-On Large Language Models* (O’Reilly, 2024),
[Chapter 8: Semantic Search and Retrieval-Augmented Generation](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main/chapter08).
The upstream code is Apache-2.0; its license and copyright notice are retained in `LICENSE`.

The supplied Chapter 8 notebook demonstrates dense search, BM25, reranking, and RAG.
This portfolio is a substantially changed experimental extension. New work includes
a fixed benchmark protocol, pinned local models, safe data loading, exact metric
implementation, per-query rankings, separate setup/latency measurements, token-limit
audits, paired bootstrap comparisons, verification tests, and result exports. It does
not reproduce the Interstellar passage or use its API-based generation examples.
The original supplied file is unchanged; its hash is recorded in `validation/source_record.json`.

## Benchmark

- Nandan Thakur et al. (2021), [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663).
- David Wadden et al. (2020), [Fact or Fiction: Verifying Scientific Claims](https://aclanthology.org/2020.emnlp-main.609/).
- [BeIR/scifact](https://huggingface.co/datasets/BeIR/scifact) and [BeIR/scifact-qrels](https://huggingface.co/datasets/BeIR/scifact-qrels).

These dataset repositories list CC BY-SA 4.0. Downloaded corpus/query text stays in
the local cache and is not bundled in the starter. Result exports include selected
query/document IDs and relevance judgments derived from the pinned benchmark;
those data-derived records retain their source license and attribution. No ground-truth
judgments are edited. The experiment restricts its development sample and preserves
all official test query IDs. See [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Models and libraries

- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), Apache-2.0 model card.
- [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2), Apache-2.0 model card; trained on MS MARCO passage ranking.
- [Sentence Transformers](https://www.sbert.net/), [Transformers](https://huggingface.co/docs/transformers/), and [rank-bm25](https://github.com/dorianbrown/rank_bm25).

Model and dataset revision IDs are in `config.json`; actual dependencies are recorded
at execution time. The code license does not relicense data, model weights, or libraries.
