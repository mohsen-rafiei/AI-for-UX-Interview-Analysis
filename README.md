# Review Aspect Sentiment Pipeline

End-to-end local-LLM pipeline for:
- extracting review aspects with sentiment/confidence,
- clustering aspects into semantic groups,
- naming clusters as umbrella categories,
- generating master sentiment tables,
- running descriptive + Bayesian analysis,
- evaluating against a ground-truth cheat table.

## What This Project Does

1. Connects Python to a local model runtime (LM Studio; Ollama can be adapted).
2. Extracts structured aspects from each review:
   - `topic`
   - `sentiment` (`negative|neutral|positive`)
   - `confidence` (`0.0-1.0`)
3. Clusters extracted topics with sentence embeddings + HDBSCAN.
4. Uses LLM naming (with retrieved context) to assign umbrella category labels.
5. Produces multiple output matrices for reporting and analysis.
6. Runs diagnostics and Bayesian uncertainty analysis by umbrella category.
7. Supports benchmark evaluation with a synthetic dataset + cheat table.

## Main Files

- `analysis.ipynb` - main notebook (pipeline + analysis sections)
- `r20.csv` - original sample input reviews
- `r50_synthetic.csv` - synthetic benchmark input (50 complex reviews)
- `r50_cheat_table.jsonl` - ground-truth aspect/sentiment/umbrella labels
- `evaluate_analysis_against_cheat.py` - accuracy evaluation script

## Dependencies

Python packages used:
- `lmstudio`
- `pandas`
- `numpy`
- `re` (stdlib)
- `json_repair`
- `tqdm`
- `sentence_transformers`
- `hdbscan`
- `matplotlib`

Install example:

```bash
pip install lmstudio pandas numpy json-repair tqdm sentence-transformers hdbscan matplotlib
```

## Notebook Run Order

Run cells in order:

1. **PIPELINE OVERVIEW**
2. **SECTION 0** - imports, model setup, helper functions
3. **SECTION 1** - extract aspects to `extracted_results.jsonl`
4. **SECTION 2** - topic embedding + clustering (`topic_map`)
5. **SECTION 2.5** - RAG chunking + retrieval helper for naming context
6. **SECTION 3** - umbrella naming
7. **SECTION 4** - mappings + data load
8. **SECTION 5** - export master tables
9. **SECTION 6-9** - descriptive, Bayesian, and diagnostic analysis

## Generated Outputs

Core outputs:
- `extracted_results.jsonl`
- `master_sentiment_matrix_FINAL.csv`
- `master_sentiment_matrix_COMPLETE.csv`
- `master_sentiment_matrix_REPORT.csv`

Analysis outputs:
- `bayesian_sentiment_summary.csv`

Evaluation outputs:
- `r50_eval_per_review.csv`

## Evaluation (Sanity Check)

After generating `extracted_results.jsonl` from `r50_synthetic.csv`, run:

```bash
python evaluate_analysis_against_cheat.py --pred extracted_results.jsonl --cheat r50_cheat_table.jsonl --fuzzy-threshold 0.85
```

This reports:
- Topic-level precision/recall/F1 (exact + fuzzy)
- Umbrella-level precision/recall/F1 (exact + fuzzy-mapped)

## Notes

- The extraction and naming steps are defensive against malformed model output.
- JSON extraction is balanced-bracket and quote-aware.
- Clustering handles tiny topic sets safely (no crash on 0/1 topic).
- RAG here is a lightweight retrieval layer to improve umbrella naming consistency.

## Optional Adaptation to Ollama

You can adapt LM Studio calls to Ollama by replacing model invocation logic in `analysis.ipynb` while keeping prompts, parsing, clustering, and evaluation unchanged.
