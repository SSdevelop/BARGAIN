#!/bin/bash

for script in \
    examples/enron_filter_base.py \
    examples/review_extract_praised_games_base.py \
    examples/court_reverse_base.py \
    examples/legal_doc_extract_base.py \
    examples/court_opinion_summarization_base.py \
    examples/random_pubmed_articles_classification_base.py \
    examples/enron_filter.py \
    examples/review_extract_praised_games.py \
    examples/court_reverse.py \
    examples/legal_doc_extract.py \
    examples/court_opinion_summarization.py \
    examples/random_pubmed_articles_classification.py
do
    echo "=== Running $script ==="
    python "$script"
    echo "=== Finished $script ==="
    echo
done

echo "All benchmarks complete. Results:"
cat results.csv
