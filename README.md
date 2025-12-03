# CINEYASH Movie Rating System

WELCOME TO CINEYASH. Mr. Yash has been curating a dataset of his movie ratings since 2023. This webapp will not only predict how Yash predicts a movie but will give recommendations from his personal log.

CINEYASH is built using Yash's personal data, OMDb metadata, transformer embeddings, and regressors/classifiers to create your very own Yash.

## How it’s built

- `app.py` is where all the Streamlit magic happens. It loads `data/processed_movies.parquet`, models the data with Random Forest, pulls fresh OMDb data for movies not in the dataset, and creates recommendations using cosine similarity on embeddings.
- `preprocessing.py` contains helpers to simplify genres, decades, and one-hots variables.
- `sentiment_utils.py` wraps the GoEmotions pipeline (`SamLowe/roberta-base-go_emotions`), which allows for sentiment analysis.
- `embedding_utils.py` holds SentenceTransformer helpers, people/plot encoding, and quick UMAP/TSNE visualizers while reading `SHEET_NAME`, `FILE_PATH`, and `OMDB_API_KEY` from the environment or `.streamlit/secrets.toml` so there are no hard-coded credentials.
- `model_utils.py` contains `model_assess`, `accuracy_within_tol`, and a few metric helpers for training regressors, ordinal models, or confusion matrices.
- `data/` contains `processed_movies.parquet`. `app.py` expects the parquet version.
- `.streamlit/secrets.toml` contains secrets

## Getting started

Just go to cineyash.streamlit.app

OR if you want to locally host for whatever reason...

1. Install dependencies: `poetry install` (or `pip install -r requirements.txt` if you’re not using Poetry).
2. Confirm `data/processed_movies.parquet` exists—if you rebuild it, reuse the helpers in `embedding_utils.py` and `preprocessing.py` to encode plots, actors, genres, and moods before saving.
3. Provide an OMDb key via `.streamlit/secrets.toml`, `.env`, or a process-level `OMDB_API_KEY` so the dashboard can fetch extra metadata.
4. Launch the dashboard with `poetry run streamlit run app.py`. The UI lets you inspect averages, compare actual vs predicted ratings, fetch new metadata, tweak runtime/RT/mood sliders, score your own plot text, and browse the embedding map.

## Structure at a glance

```
├── .streamlit/secrets.toml  # keys used by the dashboard (OMDb, sheet/file paths)
├── app.py                    # Streamlit interface + similarity pipeline
├── preprocessing.py          # genre/decade helpers and multi-label exploder
├── sentiment_utils.py        # GoEmotions wrapper + polarity helpers
├── embedding_utils.py        # SentenceTransformer helpers + UMAP/TSNE helpers
├── model_utils.py            # assessment helpers (MAE, confusion matrices, ±1 accuracy)
├── data/                     # raw/processed datasets (CSV + parquet)
├── requirements.txt          # pip deps
├── pyproject.toml           # Poetry config
├── poetry.lock              # locked Poetry graph
```
