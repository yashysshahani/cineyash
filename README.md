# CineYash Movie Rating System

This is my homegrown setup for predicting how Yash rates movies. Everything here leans on the files in this repo—Streamlit, preprocessing helpers, sentiment/embedding utilities, and the dataset under `data/`—so the README stays in sync with what you can actually run right now.

## How it’s built

- `app.py` hosts the Streamlit dashboard that loads `data/processed_movies.parquet`, runs a cached random forest, scores new titles, and recommends similar picks via cosine similarity on embeddings. It also lets you fetch fresh OMDb metadata, analyze plot text with GoEmotions, and tinker with runtime/RT/mood sliders.
- `preprocessing.py` simplifies genres, buckets years into decades, and expands any comma-separated lists into multi-hot features before modeling.
- `sentiment_utils.py` wraps `SamLowe/roberta-base-go_emotions` to get emotion vectors, polarity scores, and a fallback when the transformers stack isn’t available.
- `embedding_utils.py` keeps constants like `SHEET_NAME`, `FILE_PATH`, and `OMDB_DATA`, encodes plots/people via SentenceTransformers, and offers quick UMAP/TSNE helpers.
- `model_utils.py` exposes `model_assess`, `accuracy_within_tol`, and some plotting helpers so you can reuse the existing evaluation flow for any regressor, ordinal model, or classifier you spin up.
- `data/` already stores `full_omdb_linked_dataset.csv`, `processed_movies.csv`, and `processed_movies.parquet`. The app expects the parquet version, so keep that synchronized whenever you regenerate the data.
- `.streamlit/secrets.toml` holds `OMDB_API_KEY`, `SHEET_NAME`, and `FILE_PATH`, but you can also provide `OMDB_API_KEY` via environment variables (`OMDB_API_KEY` or `omdb_api_key`).

## Getting started

1. Install dependencies: `poetry install` (or `pip install -r requirements.txt` if you prefer pip).
2. Ensure `data/processed_movies.parquet` exists—if you rebuild it, use `embedding_utils.py` + `preprocessing.py` + `sentiment_utils.py` to encode plots, actors, genres, and moods before saving the DataFrame.
3. Provide an OMDb key in `.streamlit/secrets.toml`, `.env`, or via process env vars so the dashboard can fetch metadata for new films.
4. Launch locally: `poetry run streamlit run app.py`.

## Data & inference flows

- **Rebuild the dataset** – rerun the helpers in `embedding_utils.py` (plot + people embeddings) and `sentiment_utils.py` (GoEmotions vectors) whenever your Excel log updates, then write the final table to `data/processed_movies.parquet`.
- **Assess models** – feed predictions/targets into `model_utils.model_assess` to get MAE, R², ±1 accuracy, and confusion matrices without rewriting plotting logic.
- **Streamlit tricks** – the dashboard recomputes embeddings via `plot_embeddings_2d`, reruns mood/polarity, and runs `model_utils.accuracy_within_tol` for quick sanity checks before pushing new predictions.

## Structure at a glance

```
├── .streamlit/secrets.toml  # keys for OMDb, sheet/file paths
├── app.py                    # Streamlit interface + similarity pipeline
├── preprocessing.py          # genre/decade helpers and multi-label expander
├── sentiment_utils.py        # GoEmotions wrapper + polarity helpers
├── embedding_utils.py        # SentenceTransformer helpers + UMAP/TSNE helpers
├── model_utils.py            # assessment helpers (MAE, confusion matrices, ±1 accuracy)
├── data/                     # raw/processed datasets (CSV + parquet)
├── requirements.txt          # pip deps
├── pyproject.toml           # Poetry config
├── poetry.lock              # locked Poetry graph
```

## Quick tips

- Keep `data/processed_movies.parquet` in sync with your raw log or OMDb refreshes—overwrite it whenever the source spreadsheet changes.
- If `app.py` can’t find the parquet, drop `data/processed_movies.csv` into the repo and convert it via Pandas before running.
- Rerun `model_utils.model_assess` every time you retrain so the confusion matrix and ±1 accuracy reflect the latest model.

## Pushing to GitHub

- Make sure this directory is a Git repo. Run `git status` and, if it fails, initialize with `git init` and add a remote via `git remote add origin <url>`.
- Stage your changes: `git add README.md app.py` (plus any other modified files).
- Commit with context: `git commit -m "Describe what changed"`.
- Confirm your branch (`git branch`) and push: `git push origin <branch>` (usually `main`).
- Pull upstream changes first (`git pull --rebase`) if others are contributing, then push to avoid conflicts.

Double-check the README and datasets you touched before pushing so the remote repo keeps the same state as your local files.
