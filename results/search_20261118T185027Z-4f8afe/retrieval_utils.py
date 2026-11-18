"""Retrieval evaluation utilities for the Chapter 8 portfolio extension.

Copyright 2026 Joe Williams. Licensed under Apache-2.0.
Conceptual starting point: Alammar & Grootendorst, Hands-On Large Language Models,
Chapter 8. This module adds benchmark loading, reproducible ranking and evaluation.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

METHODS = ("bm25", "dense", "bm25_reranked")
METRICS = ("ndcg@10", "mrr@10", "recall@5", "recall@10", "recall@50")


def write_json(path, value):
    """Write readable JSON and refuse NaN/Infinity rather than hiding failures."""
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write a table with no rows.")
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_csv(path, rows, ignore=()):
    """Check presentation tables against the same verified numeric records."""
    with Path(path).open(newline="") as f:
        actual = list(csv.DictReader(f))
    expected = [{key: str(value) for key, value in row.items() if key not in ignore} for row in rows]
    actual = [{key: value for key, value in row.items() if key not in ignore} for row in actual]
    if actual != expected:
        raise AssertionError(f"CSV content mismatch: {Path(path).name}")


def tokenize(text):
    """Shared, fixed BM25 rule: casefold + Unicode words; no stemming/stoplist."""
    return re.findall(r"\b\w+\b", text.casefold())


def ranked_indices(scores, ids, k):
    """Descending finite score; equal scores use ascending document ID."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != len(ids) or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite and have one entry per ID.")
    if len(set(ids)) != len(ids) or not 0 < k <= len(ids):
        raise ValueError("IDs must be unique and k must be within corpus size.")
    return np.lexsort((np.asarray(ids, dtype=str), -scores))[:k]


def query_metrics(ranking, relevant):
    """Binary qrels, macro metrics; unjudged documents receive relevance zero."""
    if len(set(ranking)) != len(ranking):
        raise ValueError("A ranking cannot contain duplicate document IDs.")
    relevant = set(relevant)
    if not relevant:
        raise ValueError("Every evaluated query must have a positive judgment.")
    hits = np.asarray([int(doc in relevant) for doc in ranking], dtype=float)
    top = hits[:10]
    dcg = float(sum(hit / math.log2(i + 2) for i, hit in enumerate(top)))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(relevant))))
    positions = np.flatnonzero(top)
    return {
        "ndcg@10": dcg / ideal,
        "mrr@10": float(1 / (positions[0] + 1)) if len(positions) else 0.0,
        **{f"recall@{k}": float(hits[:k].sum() / len(relevant)) for k in (5, 10, 50)},
    }


def evaluate_rankings(run, qrels):
    if set(run) != set(qrels) or not run:
        raise ValueError("Ranking query IDs must exactly match evaluation qrels.")
    rows = [{"query_id": qid, **query_metrics(run[qid]["ids"], qrels[qid])}
            for qid in sorted(qrels)]
    summary = {key: float(np.mean([row[key] for row in rows])) for key in METRICS}
    return summary, rows


def paired_bootstrap(left, right, metric="ndcg@10", seed=2026, repeats=2000):
    """Percentile interval for mean right-minus-left, resampling the same queries."""
    a = {row["query_id"]: row[metric] for row in left}
    b = {row["query_id"]: row[metric] for row in right}
    if not a or set(a) != set(b):
        raise ValueError("Paired comparison requires identical, nonempty query sets.")
    differences = np.array([b[q] - a[q] for q in sorted(a)])
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(differences, size=len(differences), replace=True).mean()
                      for _ in range(repeats)])
    return {"mean_delta": float(differences.mean()),
            "ci95_low": float(np.quantile(means, 0.025)),
            "ci95_high": float(np.quantile(means, 0.975)),
            "queries_improved": int((differences > 1e-12).sum()),
            "queries_worsened": int((differences < -1e-12).sum()),
            "queries_tied": int((np.abs(differences) <= 1e-12).sum()),
            "resamples": repeats, "seed": seed}


def read_qrels(path, corpus_ids, query_ids):
    qrels = {}
    seen = set()
    with Path(path).open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qid, did, score = row["query-id"], row["corpus-id"], int(row["score"])
            if (qid, did) in seen or qid not in query_ids or did not in corpus_ids:
                raise ValueError("Duplicate judgment or unresolved query/document ID.")
            if score not in (0, 1):
                raise ValueError("This protocol expects binary relevance judgments.")
            seen.add((qid, did))
            if score > 0:
                qrels.setdefault(qid, []).append(did)
    return {qid: sorted(docs) for qid, docs in sorted(qrels.items())}


def load_benchmark(config):
    """Download pinned Parquet/TSV data; never execute a remote dataset script."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    sources = {
        "corpus": (config["dataset_id"], config["dataset_revision"],
                   "corpus/corpus-00000-of-00001.parquet"),
        "queries": (config["dataset_id"], config["dataset_revision"],
                    "queries/queries-00000-of-00001.parquet"),
        "train": (config["qrels_id"], config["qrels_revision"], "train.tsv"),
        "test": (config["qrels_id"], config["qrels_revision"], "test.tsv"),
    }
    paths, hashes = {}, {}
    for key, (repo, revision, filename) in sources.items():
        paths[key] = hf_hub_download(repo_id=repo, repo_type="dataset", revision=revision,
                                     filename=filename)
        hashes[key] = {"repository": repo, "revision": revision, "file": filename,
                       "sha256": file_hash(paths[key])}
    corpus = pd.read_parquet(paths["corpus"]).fillna("")
    queries = pd.read_parquet(paths["queries"]).fillna("")
    corpus["_id"] = corpus["_id"].astype(str)
    queries["_id"] = queries["_id"].astype(str)
    if corpus["_id"].duplicated().any() or queries["_id"].duplicated().any():
        raise ValueError("Duplicate IDs in benchmark.")
    corpus = corpus.sort_values("_id").reset_index(drop=True)
    ids = corpus["_id"].tolist()
    # Title and abstract are the document; no gold label enters this construction.
    texts = [(str(r.title).strip() + " " + str(r.text).strip()).strip()
             for r in corpus.itertuples(index=False)]
    query_text = dict(zip(queries["_id"], queries["text"].astype(str)))
    if any(not text or not tokenize(text) for text in texts) or any(not t.strip() for t in query_text.values()):
        raise ValueError("Empty documents or queries require an explicit policy.")
    train = read_qrels(paths["train"], set(ids), set(query_text))
    test = read_qrels(paths["test"], set(ids), set(query_text))
    if set(train) & set(test):
        raise ValueError("Official train/test query IDs overlap.")
    if (len(ids), len(query_text), len(train), len(test)) != (5183, 1109, 809, 300):
        raise ValueError("Pinned benchmark counts differ from the declared protocol.")
    normalize = lambda text: " ".join(text.casefold().split())
    train_texts = {normalize(query_text[q]) for q in train}
    test_texts = {normalize(query_text[q]) for q in test}
    audit = {"corpus_documents": len(ids), "query_records": len(query_text),
             "official_train_queries": len(train), "official_test_queries": len(test),
             "duplicate_normalized_documents": len(texts) - len(set(map(normalize, texts))),
             "train_test_normalized_query_overlap": len(train_texts & test_texts),
             "shared_relevant_document_ids": len({d for v in train.values() for d in v} &
                                                 {d for v in test.values() for d in v}),
             "corpus_order_sha256": digest(ids), "query_mapping_sha256": digest(query_text),
             "source_files": hashes}
    return {"ids": ids, "texts": texts, "queries": query_text,
            "train_qrels": train, "test_qrels": test, "audit": audit}


def environment_record():
    import torch
    packages = {}
    for package in ("torch", "numpy", "pandas", "scipy", "scikit-learn", "pyarrow",
                    "sentence-transformers", "transformers", "rank-bm25", "huggingface-hub"):
        packages[package] = importlib.metadata.version(package)
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "torch_threads": torch.get_num_threads()}


def choose_development_queries(data, count, seed):
    """Keep official test queries intact and exclude exact-text overlap from dev."""
    normalize = lambda text: " ".join(text.casefold().split())
    test_texts = {normalize(data["queries"][q]) for q in data["test_qrels"]}
    eligible = sorted(q for q in data["train_qrels"] if normalize(data["queries"][q]) not in test_texts)
    if not 0 < count <= len(eligible):
        raise ValueError("Development sample exceeds the eligible training query pool.")
    return sorted(np.random.default_rng(seed).choice(eligible, count, replace=False).tolist())


class RetrievalEngine:
    """Three pipelines with shared document IDs and deterministic tie-breaking."""

    def __init__(self, data, config, device=None):
        import torch
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder, SentenceTransformer
        self.torch = torch
        self.config = dict(config)
        self.ids, self.texts = data["ids"], data["texts"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.k = config["candidate_k"]
        if len(self.ids) < self.k:
            raise ValueError("Corpus must be at least as large as the candidate pool.")
        start = time.perf_counter()
        tokens = [tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokens, k1=config["bm25_k1"], b=config["bm25_b"])
        self.setup = {"bm25_tokenize_and_index_seconds": time.perf_counter() - start}
        start = time.perf_counter()
        self.encoder = SentenceTransformer(config["embedding_model"],
            revision=config["embedding_revision"], device=self.device, trust_remote_code=False,
            model_kwargs={"use_safetensors": True})
        self.encoder.max_seq_length = config["dense_max_tokens"]
        self.encoder.float().eval()
        self.reranker = CrossEncoder(config["reranker_model"],
            revision=config["reranker_revision"], device=self.device,
            max_length=config["reranker_max_tokens"], trust_remote_code=False,
            activation_fn=torch.nn.Identity(), model_kwargs={"use_safetensors": True})
        self.reranker.model.float().eval()
        self.sync()
        self.setup["neural_model_load_seconds_including_any_download"] = time.perf_counter() - start
        # Token counts are audited without altering inputs; models use documented limits.
        dense_lengths = self.token_lengths(self.encoder.tokenizer, self.texts)
        self.document_lengths = self.token_lengths(self.reranker.tokenizer, self.texts, special=False)
        self.truncation = {"dense_max_tokens": self.encoder.max_seq_length,
                           "dense_documents_truncated": int(sum(x > self.encoder.max_seq_length for x in dense_lengths)),
                           "corpus_documents": len(self.ids),
                           "dense_document_token_max": int(max(dense_lengths)),
                           "reranker_max_tokens": config["reranker_max_tokens"]}
        start = time.perf_counter()
        self.embeddings = self.encoder.encode(self.texts, batch_size=config["encode_batch_size"],
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
        self.sync()
        self.setup["dense_corpus_encode_seconds"] = time.perf_counter() - start
        self.setup["dense_embedding_bytes"] = int(self.embeddings.nbytes)
        if not np.isfinite(self.embeddings).all() or not np.allclose(np.linalg.norm(self.embeddings, axis=1), 1, atol=1e-4):
            raise ValueError("Document embeddings must be finite unit vectors.")

    @staticmethod
    def token_lengths(tokenizer, texts, special=True):
        lengths = []
        for start in range(0, len(texts), 128):
            encoded = tokenizer(texts[start:start + 128], add_special_tokens=special,
                                truncation=False, padding=False, return_attention_mask=False,
                                return_token_type_ids=False, verbose=False)
            lengths.extend(len(ids) for ids in encoded["input_ids"])
        return lengths

    def sync(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()

    def retrieve(self, method, query, audit=False):
        """Return IDs/scores only. This method never receives relevance judgments."""
        if method == "bm25":
            scores = self.bm25.get_scores(tokenize(query))
            selected = ranked_indices(scores, self.ids, self.k)
        elif method == "dense":
            vector = self.encoder.encode([query], normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False)[0]
            scores = self.embeddings @ vector
            selected = ranked_indices(scores, self.ids, self.k)
        elif method == "bm25_reranked":
            # The reranker can reorder only these BM25 candidates, never insert gold IDs.
            initial = self.bm25.get_scores(tokenize(query))
            candidates = ranked_indices(initial, self.ids, self.k)
            pairs = [(query, self.texts[i]) for i in candidates]
            rerank_scores = self.reranker.predict(pairs, batch_size=self.config["rerank_batch_size"],
                show_progress_bar=False, convert_to_numpy=True)
            ordering = ranked_indices(rerank_scores, [self.ids[i] for i in candidates], self.k)
            selected = candidates[ordering]
            scores = np.empty(len(self.ids), dtype=float)
            scores[candidates] = rerank_scores
        else:
            raise ValueError(f"Unknown method: {method}")
        result = {"ids": [self.ids[int(i)] for i in selected],
                  "scores": [float(scores[int(i)]) for i in selected]}
        if audit and method == "dense":
            length = self.token_lengths(self.encoder.tokenizer, [query])[0]
            result["query_truncated"] = length > self.encoder.max_seq_length
        if audit and method == "bm25_reranked":
            length = self.token_lengths(self.reranker.tokenizer, [query], special=False)[0]
            special = self.reranker.tokenizer.num_special_tokens_to_add(pair=True)
            result["pairs_truncated"] = int(sum(length + self.document_lengths[int(i)] + special >
                self.config["reranker_max_tokens"] for i in selected))
        return result


def run_queries(engine, queries, query_ids):
    """Quality pass. Timings here include bookkeeping and are not latency estimates."""
    from tqdm.auto import tqdm
    runs = {method: {} for method in METHODS}
    start = time.perf_counter()
    for qid in tqdm(query_ids, desc="Retrieval comparison"):
        for method in METHODS:
            runs[method][qid] = engine.retrieve(method, queries[qid], audit=True)
        if set(runs["bm25"][qid]["ids"]) != set(runs["bm25_reranked"][qid]["ids"]):
            raise AssertionError("Reranking changed candidate membership.")
    engine.sync()
    return runs, time.perf_counter() - start


def benchmark_latency(engine, queries, query_ids, repeats, seed):
    """Warm, serial query latency including query encoding/search or BM25+reranking.

    Each method recomputes its own query work. Document embeddings/models are cached;
    no query embedding or result cache is used. CUDA is synchronized before/after.
    """
    from tqdm.auto import tqdm
    # Warm each path before measuring. Download/index setup is recorded separately.
    for method in METHODS:
        engine.retrieve(method, queries[query_ids[0]])
    rng = np.random.default_rng(seed)
    trials = [(repeat, qid, method) for repeat in range(repeats)
              for qid in query_ids for method in METHODS]
    rng.shuffle(trials)
    rows = []
    for order, (repeat, qid, method) in enumerate(tqdm(trials, desc="Warm latency trials")):
        engine.sync()
        start = time.perf_counter()
        engine.retrieve(method, queries[qid])
        engine.sync()
        rows.append({"order": order, "repeat": repeat, "query_id": qid,
                     "method": method, "milliseconds": (time.perf_counter() - start) * 1000})
    return rows


def summarize_latency(rows):
    summary = []
    for method in METHODS:
        values = np.array([float(row["milliseconds"]) for row in rows if row["method"] == method])
        if not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Latency samples must be positive and finite.")
        summary.append({"method": method, "trials": len(values),
                        "median_ms": float(np.median(values)),
                        "p95_ms": float(np.quantile(values, 0.95)),
                        "mean_ms": float(values.mean())})
    return summary


def evaluate_all(runs, qrels, split):
    summaries, rows, per_method = [], [], {}
    for method in METHODS:
        summary, per_query = evaluate_rankings(runs[method], qrels)
        summaries.append({"split": split, "method": method, "n_queries": len(qrels), **summary})
        per_method[method] = per_query
        rows.extend({"split": split, "method": method, **row} for row in per_query)
    # Recall@50 must match exactly because reranking changes ordering, not membership.
    if summaries[0]["recall@50"] != summaries[2]["recall@50"]:
        raise AssertionError("Candidate recall invariant failed.")
    return summaries, rows, per_method


def error_worksheet(runs, qrels):
    """Numerical development examples; observations remain blank until inspected."""
    rows = []
    for qid in sorted(qrels):
        metrics = {m: query_metrics(runs[m][qid]["ids"], qrels[qid]) for m in METHODS}
        row = {"query_id": qid, "relevant_doc_ids": "|".join(qrels[qid]),
               "bm25_candidate_recall50": metrics["bm25"]["recall@50"]}
        for method in METHODS:
            row[f"{method}_top1"] = runs[method][qid]["ids"][0]
            row[f"{method}_ndcg10"] = metrics[method]["ndcg@10"]
        row["rerank_minus_bm25_ndcg10"] = metrics["bm25_reranked"]["ndcg@10"] - metrics["bm25"]["ndcg@10"]
        row["my_observation"] = ""
        rows.append(row)
    return sorted(rows, key=lambda r: (-abs(r["rerank_minus_bm25_ndcg10"]), r["query_id"]))


def build_manifest(directory):
    directory = Path(directory)
    entries = {str(p.relative_to(directory)): file_hash(p) for p in sorted(directory.rglob("*"))
               if p.is_file() and p.name != "manifest_sha256.json"}
    write_json(directory / "manifest_sha256.json", entries)
    return entries


def verify_result_directory(directory):
    """Recalculate every reported metric from saved IDs/judgments; no models needed."""
    directory = Path(directory)
    read = lambda name: json.loads((directory / name).read_text())
    config, protocol = read("config.json"), read("protocol.json")
    if digest(config) != protocol["config_sha256"]:
        raise AssertionError("Configuration changed after protocol creation.")
    if file_hash(directory / "retrieval_utils.py") != protocol["helpers_sha256"]:
        raise AssertionError("Exported helper code differs from the frozen protocol.")
    if protocol["methods"] != list(METHODS):
        raise AssertionError("Unexpected protocol methods.")
    valid_ids = set(read("corpus_ids.json"))
    count = 0
    for split in ("development", "test"):
        runs, qrels = read(f"{split}_rankings.json"), read(f"{split}_qrels.json")
        if sorted(qrels) != protocol[f"{split}_query_ids"] or set(runs) != set(METHODS):
            raise AssertionError("Unexpected query or method set.")
        for method in METHODS:
            for result in runs[method].values():
                ids, scores = result["ids"], result["scores"]
                if len(ids) != config["candidate_k"] or len(ids) != len(set(ids)) or not set(ids) <= valid_ids:
                    raise AssertionError("Invalid ranked IDs.")
                if list(ranked_indices(scores, ids, len(ids))) != list(range(len(ids))):
                    raise AssertionError("Ranking is not ordered by score and tie-break ID.")
        for qid in qrels:
            if not set(qrels[qid]) <= valid_ids:
                raise AssertionError("Unknown relevant document.")
            if set(runs["bm25"][qid]["ids"]) != set(runs["bm25_reranked"][qid]["ids"]):
                raise AssertionError("Reranker inserted or removed candidates.")
        summary, rows, per_method = evaluate_all(runs, qrels, split)
        if summary != read(f"{split}_metrics.json") or rows != read(f"{split}_per_query.json"):
            raise AssertionError("Saved quality metrics do not match rankings.")
        check_csv(directory / f"{split}_metrics.csv", summary)
        if split == "development" and error_worksheet(runs, qrels) != read("error_worksheet_numeric.json"):
            raise AssertionError("Error worksheet does not match saved development evidence.")
        if split == "development":
            check_csv(directory / "error_analysis.csv", error_worksheet(runs, qrels), ignore=("my_observation",))
        if split == "test":
            check_csv(directory / "test_per_query.csv", rows)
            comparisons = read("test_paired_comparisons.json")
            expected_pairs = [("bm25", "dense"), ("bm25", "bm25_reranked"), ("dense", "bm25_reranked")]
            if [(c["baseline"], c["comparison"]) for c in comparisons] != expected_pairs:
                raise AssertionError("Missing or unexpected paired comparison.")
            for comparison in comparisons:
                actual = paired_bootstrap(per_method[comparison["baseline"]], per_method[comparison["comparison"]],
                    seed=config["seed"], repeats=config["bootstrap_repeats"])
                if actual != comparison["statistics"]:
                    raise AssertionError("Paired comparison does not reproduce.")
            check_csv(directory / "test_paired_comparisons.csv", [
                {"baseline": c["baseline"], "comparison": c["comparison"], **c["statistics"]} for c in comparisons])
        count += len(qrels) * len(METHODS)
    if set(protocol["development_query_ids"]) & set(protocol["test_query_ids"]):
        raise AssertionError("Development/test IDs overlap.")
    latency = read("latency_trials.json")
    expected = {(r, q, m) for r in range(config["latency_repeats"])
                for q in protocol["latency_query_ids"] for m in METHODS}
    actual = {(row["repeat"], row["query_id"], row["method"]) for row in latency}
    if expected != actual or len(actual) != len(latency):
        raise AssertionError("Missing or repeated latency trial.")
    if summarize_latency(latency) != read("latency_summary.json"):
        raise AssertionError("Latency summary mismatch.")
    check_csv(directory / "latency_trials.csv", latency)
    check_csv(directory / "latency_summary.csv", summarize_latency(latency))
    manifest = read("manifest_sha256.json")
    actual_files = {str(p.relative_to(directory)) for p in directory.rglob("*")
                    if p.is_file() and p.name != "manifest_sha256.json"}
    if set(manifest) != actual_files:
        raise AssertionError("Manifest does not cover exactly the saved files.")
    for name, expected_hash in manifest.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise AssertionError("Unsafe manifest path.")
        if file_hash(directory / name) != expected_hash:
            raise AssertionError(f"Hash mismatch: {name}")
    return {"status": "passed", "query_method_rankings": count,
            "methods": list(METHODS), "latency_trials": len(latency),
            "hashed_files": len(manifest),
            "scope": "Saved evidence consistency; not a repeat of model inference or a relevance-label audit."}
