import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

random_state = 0
test_size = 0.2
OMDB_DATA = "full_omdb_linked_dataset.csv"

_SECRETS_PATH = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
_SECRETS_CACHE: dict[str, str] | None = None


def _load_streamlit_secrets() -> dict[str, str]:
    global _SECRETS_CACHE
    if _SECRETS_CACHE is not None:
        return _SECRETS_CACHE

    secrets: dict[str, str] = {}
    if not _SECRETS_PATH.exists():
        _SECRETS_CACHE = secrets
        return secrets

    loader = None
    try:
        import tomllib as toml_loader  # type: ignore[attr-defined]

        loader = toml_loader.load
    except ModuleNotFoundError:
        try:
            import tomli as toml_loader  # type: ignore[import]

            loader = toml_loader.load
        except ModuleNotFoundError:
            loader = None

    if loader:
        with _SECRETS_PATH.open("rb") as fp:
            secrets = loader(fp)  # type: ignore[assignment]

    _SECRETS_CACHE = secrets or {}
    return _SECRETS_CACHE


def _env_or_secret(key: str, fallback: str | None = None) -> str | None:
    value = os.environ.get(key)
    if value:
        return value
    lower = key.lower()
    if lower != key:
        value = os.environ.get(lower)
        if value:
            return value
    secrets = _load_streamlit_secrets()
    if key in secrets:
        return secrets[key]
    if lower in secrets:
        return secrets[lower]
    return fallback


SHEET_NAME = _env_or_secret("SHEET_NAME") or "Master List"
FILE_PATH = _env_or_secret("FILE_PATH") or "Film Reviews.xlsx"
API_KEY = _env_or_secret("OMDB_API_KEY") or _env_or_secret("API_KEY")

def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    return model

def encode_plot(plot_text, model):
    if not plot_text or (isinstance(plot_text, float) and pd.isna(plot_text)):
        text = ""
    else:
        text = str(plot_text)
    
    embedding = model.encode(text, convert_to_numpy=True)

    return embedding

def add_plot_embeddings_column(df, model, plot_col="Plot", new_col="plot_embedding"):
    df[new_col] = df[plot_col].apply(lambda text: encode_plot(text, model))
    return df

def _split_people(text):
    if not isinstance(text, str):
        return []
    tokens = re.split(r",|/|&|;|\band\b", text, flags=re.IGNORECASE)
    return [tok.strip() for tok in tokens if tok and tok.strip()]

def encode_people(names_text, model):
    people = _split_people(names_text)
    if not people:
        if hasattr(model, "get_sentence_embedding_dimension"):
            dim = model.get_sentence_embedding_dimension()
        else:
            dim = len(model.encode("", convert_to_numpy=True))
        return np.zeros(dim, dtype=np.float32)

    embeddings = model.encode(people, convert_to_numpy=True)
    if embeddings.ndim == 1:
        return embeddings
    return embeddings.mean(axis=0)

def add_people_embedding_column(df, source_col, model, new_col=None):
    if new_col is None:
        new_col = f"{source_col}_embedding"
    df[new_col] = df[source_col].apply(lambda text: encode_people(text, model))
    return df

def stack_plot_embeddings(df, embed_col="plot_embedding"):
    vecs = df[embed_col].tolist()
    return np.stack(vecs)

def reduce_embeddings_to_2d(X_plot, method="umap"):
    if method == "umap":
        import umap.umap_ as umap
        reducer = umap.UMAP(
            n_neighbors=5,
            min_dist=0.0,
            n_components=2,
            metric="cosine",
            random_state=random_state
        )
        
        X_2d = reducer.fit_transform(X_plot)

        return X_2d

    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate="auto",
            n_iter=1000,
            metric="cosine"
        )

        X_2d = reducer.fit_transform(X_plot)
        
        return X_2d


def attach_2d_coords(df, X_2d, x_col="x", y_col="y"):
    df[x_col] = X_2d[:, 0]
    df[y_col] = X_2d[:, 1]
    return df

def plot_embeddings_2d(df, x_col="x", y_col="y", color_col="Rating", title=None, hover_cols=None, template="streamlit"):
    if title is None:
        title = f"2D Embedding Map by {color_col}"

    if hover_cols is None:
        default_cols = [color_col]
        if "Movie" in df.columns:
            default_cols.append("Movie")

        hover_cols = default_cols

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_data=hover_cols,
        title=title,
        opacity=0.8
    )

    fig.update_traces(marker=dict(size=6))
    fig.update_layout(
        template=template,
        legend_title=color_col
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig
