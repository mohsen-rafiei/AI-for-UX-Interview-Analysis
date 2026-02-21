# Review Aspect Sentiment Pipeline

Workshop recording: https://www.youtube.com/watch?v=fM3z5rc74m8

End-to-end local-LLM pipeline for:
- extracting review aspects with sentiment/confidence,
- clustering aspects into semantic groups,
- naming clusters as umbrella categories,
- generating master sentiment tables,
- running descriptive + Bayesian analysis,

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

## Main Files

- `analysis.ipynb` - main notebook (pipeline + analysis sections)
- `r20.csv` - original sample input reviews

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



## Notes

- The extraction and naming steps are defensive against malformed model output.
- JSON extraction is balanced-bracket and quote-aware.
- Clustering handles tiny topic sets safely (no crash on 0/1 topic).
- RAG here is a lightweight retrieval layer to improve umbrella naming consistency.

## Optional Adaptation to Ollama

You can adapt LM Studio calls to Ollama by replacing model invocation logic in `analysis.ipynb` while keeping prompts, parsing, clustering, and evaluation unchanged.
