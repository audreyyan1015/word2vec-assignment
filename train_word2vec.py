#!/usr/bin/env python3
"""Word2Vec course assignment with Gensim.

The script trains six small Word2Vec configurations on Gensim's Lee news corpus,
then evaluates the vectors on the WordSim-353 word-similarity dataset.
"""
from pathlib import Path
import csv
import time

from gensim.models import Word2Vec
from gensim.test.utils import datapath
from gensim.utils import simple_preprocess
from scipy.stats import spearmanr

CONFIGS = [
    ("E1", "CBOW", 0, 100, 5, 50),
    ("E2", "Skip-gram", 1, 100, 5, 50),
    ("E3", "Skip-gram", 1, 50, 5, 50),
    ("E4", "Skip-gram", 1, 100, 2, 50),
    ("E5", "Skip-gram", 1, 100, 10, 50),
    ("E6", "Skip-gram", 1, 100, 5, 5),
]


def load_corpus():
    """Read the 300-document Lee news corpus bundled with Gensim."""
    with Path(datapath("lee_background.cor")).open(encoding="latin-1") as f:
        return [simple_preprocess(line, deacc=True) for line in f if line.strip()]


def load_wordsim353():
    """Read WordSim-353 word pairs bundled with Gensim."""
    pairs = []
    for line in Path(datapath("wordsim353.tsv")).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        w1, w2, score = line.split("\t")
        pairs.append((w1.lower(), w2.lower(), float(score)))
    return pairs


def evaluate_wordsim(model, pairs):
    """Compare model cosine similarities with human WordSim-353 scores."""
    human_scores, model_scores = [], []
    for w1, w2, score in pairs:
        if w1 in model.wv and w2 in model.wv:
            human_scores.append(score)
            model_scores.append(model.wv.similarity(w1, w2))
    rho = spearmanr(human_scores, model_scores).statistic
    return float(rho), len(human_scores)


def train_one(corpus, config):
    exp_id, algorithm, sg, vector_size, window, epochs = config
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=5,
        sg=sg,
        negative=10,
        workers=1,
        seed=42,
        epochs=epochs,
    )
    return model


def main():
    corpus = load_corpus()
    wordsim = load_wordsim353()
    print(f"Corpus: {len(corpus)} news documents, {sum(map(len, corpus))} tokens")

    rows = []
    for config in CONFIGS:
        exp_id, algorithm, sg, vector_size, window, epochs = config
        start = time.perf_counter()
        model = train_one(corpus, config)
        seconds = time.perf_counter() - start
        rho, covered = evaluate_wordsim(model, wordsim)

        rows.append([exp_id, algorithm, vector_size, window, epochs, rho, covered, seconds])
        print(
            f"{exp_id}: {algorithm:9s} dim={vector_size:3d} window={window:2d} "
            f"epochs={epochs:2d}  WordSim rho={rho:.3f}  time={seconds:.2f}s"
        )

        if exp_id == "E2":
            model.save("word2vec_e2.model")
            print("  Nearest words to 'afghanistan':", model.wv.most_similar("afghanistan", topn=5))

    with open("results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "algorithm", "vector_size", "window", "epochs",
            "wordsim353_spearman", "covered_pairs", "train_seconds"
        ])
        writer.writerows(rows)

    print("\nSaved: results.csv and word2vec_e2.model")


if __name__ == "__main__":
    main()
