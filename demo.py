#!/usr/bin/env python3
"""Query a trained Word2Vec model."""
import argparse
from gensim.models import Word2Vec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="word2vec_e2.model")
    parser.add_argument("--word", default="afghanistan")
    args = parser.parse_args()

    model = Word2Vec.load(args.model)
    if args.word not in model.wv:
        print(f"'{args.word}' is not in the vocabulary.")
        return

    print(f"Most similar words to '{args.word}':")
    for word, score in model.wv.most_similar(args.word, topn=10):
        print(f"  {word:15s} {score:.4f}")


if __name__ == "__main__":
    main()
