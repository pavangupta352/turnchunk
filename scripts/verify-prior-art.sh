#!/usr/bin/env bash
# Reproduce every claim in the README's opening section.
#
# Requires the GitHub CLI, authenticated:  gh auth login
# Everything below is a live query. If these numbers have changed, the README
# is out of date -- open an issue and it will be corrected.

set -uo pipefail

echo "== People hand-rolling speaker-aware chunking (GitHub code search) =="
for q in \
  'def chunk_by_speaker' \
  'speaker_turns chunk language:python' \
  'current_speaker chunk language:python'
do
  n=$(gh api -X GET search/code -f q="$q" -f per_page=1 --jq '.total_count' 2>/dev/null)
  printf '  %-46s %s files\n' "$q" "${n:-?}"
done

echo
echo "== Packaged alternatives on GitHub (repository search) =="
for q in \
  'speaker aware chunking' \
  'speaker turn chunking' \
  'transcript chunking library' \
  'conversation aware text splitter' \
  'dialogue chunker'
do
  n=$(gh api -X GET search/repositories -f q="$q" -f per_page=1 --jq '.total_count' 2>/dev/null)
  top=$(gh api -X GET search/repositories -f q="$q" -f sort=stars -f per_page=1 \
        --jq '.items[0] | "\(.full_name) (\(.stargazers_count) stars)"' 2>/dev/null)
  printf '  %-38s %-4s repos   top: %s\n' "$q" "${n:-?}" "${top:-none}"
done

echo
echo "== Chunkers shipped by the three main ecosystems =="
printf '  %-14s ' "chonkie:"
gh api repos/feyninc/chonkie/contents/src/chonkie/chunker --jq '[.[].name | select(test("^(__init__|base)\\.py$") | not) | sub("\\.py$";"")] | join(", ")' 2>/dev/null
printf '  %-14s ' "langchain:"
gh api repos/langchain-ai/langchain/contents/libs/text-splitters/langchain_text_splitters --jq '[.[].name | select(endswith(".py")) | select(test("^(__init__|base)\\.py$") | not) | sub("\\.py$";"")] | join(", ")' 2>/dev/null
printf '  %-14s ' "llamaindex:"
gh api repos/run-llama/llama_index/contents/llama-index-core/llama_index/core/node_parser/text --jq '[.[].name | select(endswith(".py")) | select(test("^(__init__|utils)\\.py$") | not) | sub("\\.py$";"")] | join(", ")' 2>/dev/null

echo
echo "Look for a conversation, speaker, dialogue or transcript chunker above."
