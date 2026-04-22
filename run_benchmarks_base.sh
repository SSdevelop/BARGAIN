#!/bin/bash

# set -euo pipefail

# cd "$(dirname "$0")"

for script in \
    examples/enron_filter_base.py \
    examples/review_extract_praised_games_base.py \
    examples/court_reverse_base.py \
    examples/legal_doc_extract_base.py \
    examples/court_opinion_summarization_base.py
do
    echo "=== Running $script ==="
    python "$script"
    echo "=== Finished $script ==="
    echo
done

echo "All non-judge benchmarks complete."
if [[ -f results.csv ]]; then
    echo "Results in results.csv:"
    cat results.csv
else
    echo "results.csv not found."
fi
