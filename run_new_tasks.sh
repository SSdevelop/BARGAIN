#!/bin/bash

for script in \
    examples/enron_extract_emails_base.py \
    examples/wiki_talk_extract_usernames_base.py \
    examples/wiki_talk_first_timestamp_base.py \
    examples/wiki_talk_last_timestamp_base.py \
    examples/random_pubmed_articles_classification_base.py \
    examples/enron_extract_emails.py \
    examples/wiki_talk_extract_usernames.py \
    examples/wiki_talk_first_timestamp.py \
    examples/wiki_talk_last_timestamp.py \
    examples/random_pubmed_articles_classification.py
do
    echo "=== Running $script ==="
    python "$script"
    echo "=== Finished $script ==="
    echo
done

echo "All new task benchmarks complete. Results:"
cat results.csv
