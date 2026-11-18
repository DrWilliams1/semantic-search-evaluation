# Explain the retrieval experiment

## From query to ranked documents

**BM25:** The tokenizer lowercases with `casefold()` and extracts word tokens. BM25
compares query terms with document terms, accounting for term frequency and document
length. It scores the entire corpus. `ranked_indices` sorts high scores first and
breaks exact ties by document ID, so zero-score ties are reproducible.

**Dense retrieval:** `SentenceTransformer.encode()` maps each document to a fixed
384-dimensional vector. Document vectors are created once. For each request, the
query is encoded and normalized. `embeddings @ vector` compares it with every
document. With unit vectors, that dot product equals cosine similarity. This is
exact search over a small matrix, not an approximate FAISS/vector-database benchmark.

**BM25 + reranker:** BM25 proposes 50 documents. A cross-encoder reads each query
and candidate together, producing a relevance score. Those 50 documents are sorted
again. A relevant document absent from the pool cannot be recovered. Scores are
raw logits; their magnitude is not a probability or directly comparable with BM25.

## Metrics you should be able to calculate

Suppose a query has two judged relevant documents. The first appears at rank 2,
and the other is absent from the top 50. Recall@10 and Recall@50 are both 1/2.
MRR@10 is 1/2. DCG@10 is 1/log2(3); ideal DCG is 1 + 1/log2(3), giving nDCG@10
of about 0.387. The denominator includes both relevant documents even though only
one was retrieved. Metrics are averaged over queries, not pooled over documents.

**Recall@50 is the candidate ceiling for this reranker.** It has exactly the same
candidate set as BM25, so their Recall@50 must be identical even if nDCG@10 changes.

## What is timed

Model downloads/loading, BM25 indexing, and corpus embedding happen before timed
requests. Each warm request includes query tokenization/encoding and ranking; the
reranked path includes fresh BM25 retrieval plus cross-encoder scoring. GPU
synchronization prevents timing only the scheduling of unfinished GPU operations.
The timer does not include optional input audits, JSON writing, or plotting.
Three repeats per development query are shuffled across methods. Median and p95
are descriptive statistics over those trials, not guaranteed service latency.

## What the experiment controls

Methods share the corpus, query IDs, relevance judgments, tie-breaking rule and
retrieval cutoff. Models/revisions and settings are fixed before test. The development
subset excludes normalized query strings present in test. No model is trained,
and no relevant document is artificially added to reranker candidates.

## What remains a trade-off

BM25 uses full documents; the dense encoder and reranker have different token
limits. The audit records affected inputs but does not remove this difference.
Results compare the declared pipelines, not a pure causal effect of neural versus
keyword processing. Relevance judgments can be incomplete. Public pretrained
models may have prior exposure that this project cannot establish or exclude.

## Present your findings

State the measured quality difference, its paired interval, and the measured
latency difference. Explain one observed improvement and one failure if present.
Distinguish a candidate miss from a reranker ordering error. Describe the hardware
and token limits. Avoid assuming semantic search or reranking must outperform BM25.

## Recorded run: explain these findings

- Reranking raised test nDCG@10 from 0.6519 to 0.6818, with 11.79× BM25's median latency.
- Dense was fastest (12.74 ms) and had the highest Recall@50 (0.8920), but did not have the highest nDCG@10.
- The relevant document for development query 391 fell from rank 3 to 14 after reranking: the candidate was present, but ordering worsened.
- Both relevant documents for query 1109 were absent from the top 50: reranking could not recover them.
- Dense document truncation affected 71.02% of the corpus. That is a limitation to discuss, not proof that it caused a particular error.
- Development and test outcomes differed: dense nDCG@10 exceeded BM25 on the 50-query development sample but was lower on the 300-query test set.
