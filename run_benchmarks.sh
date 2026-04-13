#!/bin/bash

for script in \
    examples/enron_filter.py \
    examples/review_extract_praised_games.py \
    examples/court_reverse.py \
    examples/legal_doc_extract.py \
    examples/court_opinion_summarization.py
do
    echo "=== Running $script ==="
    python "$script"
    echo "=== Finished $script ==="
    echo
done

echo "All benchmarks complete. Results:"
cat results.csv
