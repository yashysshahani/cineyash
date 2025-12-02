import os
import re

import numpy as np
import pandas as pd
import requests
import streamlit as st
from embedding_utils import encode_plot, load_embedding_model, plot_embeddings_2d
from model_utils import accuracy_within_tol, model_assess
from pathlib import Path
from preprocessing import simplify_genre, year_to_decade
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from sentiment_utils import goemotion_vector, polarity_from_vector
except ImportError:
    from sentiment_utils import goemotion_vector

    def polarity_from_vector(vector):
        arr = np.nan_to_num(np.array(vector, dtype=float))
        return float(np.nanmean(arr)) if arr.size else 0.0
DATA_PATH = Path(__file__).resolve().parent / "data" / "processed_movies.parquet"
NUMERIC_FEATURES = ["Runtime", "rt_rating", "plot_sentiment"]
CATEGORICAL_FEATURES = ["simple_genre", "plot_mood", "Decade"]
POSITIVE_WORDS = {"love", "hope", "joy", "smile", "happiness", "gift", "good", "beautiful"}
NEGATIVE_WORDS = {"death", "kill", "dark", "fear", "angry", "hate", "pain", "murder"}
OMDB_API_URL = "https://www.omdbapi.com/"


def get_omdb_api_key() -> str | None:
    for key in ("OMDB_API_KEY", "omdb_api_key"):
        secret_value = st.secrets.get(key)
        if secret_value:
            return secret_value
    return os.environ.get("OMDB_API_KEY")


def _parse_runtime(runtime_text: str | None) -> int | None:
    if not runtime_text:
        return None
    match = re.search(r"(\d+)", runtime_text)
    return int(match.group(1)) if match else None


def _parse_rt_score(ratings: list[dict] | None) -> int | None:
    if not isinstance(ratings, list):
        return None
    for entry in ratings:
        if entry.get("Source", "").lower() == "rotten tomatoes":
            raw = entry.get("Value", "")
            if raw.endswith("%"):
                try:
                    return int(raw.strip("%"))
                except ValueError:
                    return None
    return None


def _extract_year(year_text: str | None) -> int | None:
    if not year_text:
        return None
    match = re.search(r"(\d{4})", year_text)
    if match:
        return int(match.group(1))
    return None


def _guess_plot_sentiment(plot: str | None) -> float:
    if not plot:
        return 0.0
    plot_lower = plot.lower()
    pos = sum(plot_lower.count(word) for word in POSITIVE_WORDS)
    neg = sum(plot_lower.count(word) for word in NEGATIVE_WORDS)
    total = pos + neg
    return 0.0 if total == 0 else (pos - neg) / total


@st.cache_resource
def cached_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    return load_embedding_model(model_name)


def analyze_plot_text(plot: str) -> dict:
    embedding = encode_plot(plot or "", cached_embedding_model())
    vector = goemotion_vector(plot)
    sentiment = float(polarity_from_vector(vector))
    mood = "positive" if sentiment >= 0 else "negative"
    return {
        "plot_embedding": embedding,
        "plot_sentiment": sentiment,
        "plot_mood": mood,
    }


def _guess_plot_mood(plot: str | None) -> str:
    score = _guess_plot_sentiment(plot)
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "positive"


def _unwrap_metric(value):
    if isinstance(value, tuple):
        return float(value[0])
    return float(value)


def fetch_omdb_metadata(title: str) -> dict | None:
    api_key = get_omdb_api_key()
    if not title.strip() or not api_key:
        return None
    try:
        response = requests.get(
            OMDB_API_URL,
            params={"apikey": api_key, "t": title},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return None

    if payload.get("Response") != "True":
        return None

    runtime = _parse_runtime(payload.get("Runtime"))
    rt_score = _parse_rt_score(payload.get("Ratings"))
    year = _extract_year(payload.get("Year"))
    decade = year_to_decade(year) if year else None
    plot = payload.get("Plot", "")
    analysis = analyze_plot_text(plot)
    raw_genre = payload.get("Genre", "")
    return {
        "title": payload.get("Title", title),
        "Runtime": runtime,
        "rt_rating": rt_score,
        "Decade": decade,
        "simple_genre": simplify_genre(raw_genre),
        "RawGenre": raw_genre,
        "plot_mood": analysis["plot_mood"],
        "plot_sentiment": analysis["plot_sentiment"],
        "plot_embedding": analysis["plot_embedding"].tolist(),
        "Year": year,
    }


@st.cache_data(hash_funcs={np.ndarray: lambda arr: arr.tobytes()})
def load_movies(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"The processed dataset is missing at {path}. Run `data_prep.ipynb` "
            "or copy `data/processed_movies.parquet` into the repo before starting "
            "the app."
        )
    df = pd.read_parquet(path)
    df = df.copy().reset_index(drop=True)
    df["Decade"] = df["Decade"].fillna("Unknown")
    return df


@st.cache_resource
def build_pipeline(df: pd.DataFrame) -> Pipeline:
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200, random_state=0, n_jobs=-1, max_depth=10
                ),
            ),
        ],
    )

    model.fit(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df["Rating"])
    return model


@st.cache_resource
def build_embeddings(df: pd.DataFrame) -> np.ndarray:
    return np.vstack(df["plot_embedding"].values)


@st.cache_resource
def build_kmeans(embeddings: np.ndarray, n_clusters: int = 8):
    model = KMeans(n_clusters=n_clusters, random_state=0)
    model.fit(embeddings)
    return model


def get_recommendations(
    title: str, df: pd.DataFrame, embeddings: np.ndarray, top_k: int = 5
) -> pd.DataFrame:
    title_lower = title.casefold()
    mask = df["Movie"].str.casefold() == title_lower

    if not mask.any():
        return pd.DataFrame()

    idx = mask.idxmax()
    query_embedding = embeddings[idx].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    ranks = np.argsort(similarities)[::-1]
    ranks = [i for i in ranks if i != idx][:top_k]

    recs = df.iloc[ranks][["Movie", "Rating", "Release Year", "Director"]].copy()
    recs["Similarity"] = np.round(similarities[ranks], 3)
    return recs
def format_prediction(raw_score: float) -> float:
    return float(np.clip(raw_score, 1.0, 10.0))


@st.cache_resource
def cached_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    return load_embedding_model(model_name)


def encode_query_text(text: str) -> np.ndarray | None:
    if not isinstance(text, str) or not text.strip():
        return None
    encoder = cached_embedding_model()
    return encode_plot(text, encoder)


def recommend_from_plot_text(text: str, df: pd.DataFrame, embeddings: np.ndarray, top_k: int = 5) -> pd.DataFrame:
    query_vector = encode_query_text(text)
    if query_vector is None:
        return pd.DataFrame()

    similarities = cosine_similarity(query_vector.reshape(1, -1), embeddings).flatten()
    ranks = np.argsort(similarities)[::-1][: top_k + 1]
    ranks = [idx for idx in ranks if similarities[idx] > 0][:top_k]

    recs = (
        df.iloc[ranks][["Movie", "Rating", "Release Year", "Director"]]
        .copy()
        .assign(Similarity=lambda sub: np.round(similarities[ranks], 3))
    )
    return recs


def recommend_from_embedding_vector(vector, df: pd.DataFrame, embeddings: np.ndarray, top_k: int = 5) -> pd.DataFrame:
    arr = np.array(vector, dtype=float)
    if arr.size == 0:
        return pd.DataFrame()
    similarities = cosine_similarity(arr.reshape(1, -1), embeddings).flatten()
    ranks = np.argsort(similarities)[::-1]
    ranks = [idx for idx in ranks if similarities[idx] > 0][:top_k]
    recs = (
        df.iloc[ranks][["Movie", "Rating", "Release Year", "Director"]]
        .copy()
        .assign(Similarity=lambda sub: np.round(similarities[ranks], 3))
    )
    return recs


def main() -> None:
    st.set_page_config(
        page_title="CINEYASH",
        page_icon="🎬",
        layout="wide",
    )

    st.title("C I N E Y A S H")
    st.write(
        "Interact with the personalized rating model by choosing a tracked film or "
        "calibrating your own hypothetical watch. Predictions, actual ratings, and "
        "similar titles are generated from the curated dataset."
    )

    if "omdb_metadata" not in st.session_state:
        st.session_state["omdb_metadata"] = {}
    if "_omdb_fetched_title" not in st.session_state:
        st.session_state["_omdb_fetched_title"] = ""

    try:
        df = load_movies()
    except FileNotFoundError as exc:
        st.error(
            "The project can’t run without `data/processed_movies.parquet`. "
            "Run `data_prep.ipynb` locally (or copy the file to `data/`) before "
            "launching the app."
        )
        st.stop()
    model = build_pipeline(df)
    embeddings = build_embeddings(df)
    kmeans_model = build_kmeans(embeddings, n_clusters=8)
    df["kmeans_cluster"] = kmeans_model.predict(embeddings).astype(str)
    features = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_true = df["Rating"]
    y_pred = model.predict(features)
    model_results, assessment_df = model_assess(df, y_pred, y_true, task="regression", plot=False)
    within_tol = accuracy_within_tol(y_true, np.clip(y_pred, 1.0, 10.0))

    avg_rating = df["Rating"].mean()
    most_common_decade = (
        df["Decade"].value_counts().idxmax()
        if not df["Decade"].empty
        else "Unknown"
    )

    stat_cols = st.columns(3)
    stat_cols[0].metric("Tracked films", len(df))
    stat_cols[1].metric("Average rating", f"{avg_rating:.2f} / 10")
    stat_cols[2].metric("Most frequent decade", most_common_decade)

    with st.expander("Dataset snapshot"):
        st.dataframe(
            df[
                [
                    "Movie",
                    "Release Year",
                    "Director",
                    "simple_genre",
                    "Decade",
                    "Rating",
                ]
            ]
            .reset_index(drop=True)
            .head(20)
        )
    color_by = st.selectbox(
        "Color embedding map by",
        ["Rating", "simple_genre", "Decade", "kmeans_cluster"],
        index=0,
        key="embedding_color_choice",
    )

    with st.expander("Embedding map"):
        fig = plot_embeddings_2d(
            df,
            x_col="x",
            y_col="y",
            color_col=color_by,
            hover_cols=["Rating", "Movie"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Model diagnostics"):
        diag_cols = st.columns(4)
        diag_cols[0].metric("R²", f"{_unwrap_metric(model_results.get('R2', 0.0)):.2f}")
        diag_cols[1].metric("MAE", f"{_unwrap_metric(model_results.get('MAE', 0.0)):.2f}")
        diag_cols[2].metric("RMSE", f"{_unwrap_metric(model_results.get('RMSE', 0.0)):.2f}")
        diag_cols[3].metric("Acc ±1", f"{within_tol:.2f}")
        st.dataframe(
            assessment_df[["Movie", "Rating", "preds", "rt_rating"]]
            .assign(preds=lambda df: df["preds"].round(2))
            .head(12)
        )

    st.header("Predict a tracked film")
    movie_options = sorted(df["Movie"].dropna().unique())
    target_title = st.selectbox("Pick a movie from the log", movie_options)
    target_row = df[df["Movie"] == target_title].iloc[0]

    sample_features = target_row[NUMERIC_FEATURES + CATEGORICAL_FEATURES].to_frame().T
    predicted_rating = format_prediction(model.predict(sample_features)[0])
    actual_rating = float(target_row["Rating"])

    cols = st.columns(3)
    cols[0].metric("Predicted rating", f"{predicted_rating:.1f} / 10")
    cols[1].metric("Actual rating", f"{actual_rating:.1f} / 10")
    cols[2].metric(
        "Prediction gap",
        f"{(predicted_rating - actual_rating):+.1f}",
    )

    st.write("**Why this film?**")
    st.write(
        f"{target_row['Release Year']} • {target_row['simple_genre']} • "
        f"RT score {target_row['rt_rating']}"
    )

    recs = get_recommendations(target_title, df, embeddings, top_k=6)
    if not recs.empty:
        st.subheader("Similar movies from YashLog")
        st.table(recs)
    else:
        st.info("No recommendations available for that selection.")

    st.header("Estimate your own pick")
    custom_title = st.text_input("Film title", "Untitled Watch", key="custom_movie_title")
    metadata_from_session = st.session_state.get("omdb_metadata", {})
    st.session_state.setdefault("custom_plot_text", "")
    plot_from_omdb = metadata_from_session.get("Plot", "").strip()
    if plot_from_omdb and not st.session_state["custom_plot_text"].strip():
        st.session_state["custom_plot_text"] = plot_from_omdb
    custom_plot = st.text_area(
        "Plot / review snippet (OPTIONAL)",
        placeholder="Paste your plot idea, log entry, or short review to auto-suggest sentiment. Will generate from OMDb if empty",
        height=120,
        key="custom_plot_text",
    )
    text_sentiment_score = None
    text_mood_override = None
    if custom_plot.strip():
        vector = goemotion_vector(custom_plot)
        text_sentiment_score = float(polarity_from_vector(vector))
        text_mood_override = "positive" if text_sentiment_score >= 0 else "negative"
        st.caption(
            f"Text-driven sentiment: {text_sentiment_score:+.2f} → mood set to {text_mood_override.title()}."
        )

    fetched_title = st.session_state.get("_omdb_fetched_title", "")
    if fetched_title and custom_title.strip().casefold() != fetched_title.strip().casefold():
        st.session_state["omdb_metadata"] = {}
        st.session_state["_omdb_fetched_title"] = ""

    api_key = get_omdb_api_key()
    if st.button(
        "Auto-fill from OMDb",
        use_container_width=True,
        disabled=api_key is None or not custom_title.strip(),
    ):
        if not api_key:
            st.warning(
                "Add `OMDB_API_KEY` to Streamlit secrets (or your `.env` for local runs) to enable metadata fetches."
            )
        else:
            metadata = fetch_omdb_metadata(custom_title)
            if metadata:
                st.session_state["omdb_metadata"] = metadata
                st.session_state["_omdb_fetched_title"] = metadata.get("title", custom_title)
                st.success("Metadata pulled from OMDb. Adjust sliders to taste.")
            else:
                st.error("Could not find that title on OMDb. Try another keyword.")

    metadata = st.session_state.get("omdb_metadata", {})
    if metadata:
        meta_year = metadata.get("Year") or "-"
        meta_genre = metadata.get("RawGenre", "")
        meta_rt = metadata.get("rt_rating")
        meta_runtime = metadata.get("Runtime")
        meta_decade = metadata.get("Decade") or "Unknown"
        st.markdown(
            f"**OMDb snapshot:** {metadata.get('title')} "
            f"• {meta_year} • {meta_genre} • {meta_runtime or '?'} min • "
            f"RT {meta_rt if meta_rt is not None else '?'} / 100 • {meta_decade}"
        )
        embedding_vector = metadata.get("plot_embedding")
        if embedding_vector:
            omdb_recs = recommend_from_embedding_vector(
                embedding_vector, df, embeddings, top_k=4
            )
            if not omdb_recs.empty:
                st.subheader("Similar movies from YashLog")
                st.table(omdb_recs)


    metadata_runtime = metadata.get("Runtime")
    runtime_default = (
        max(60, min(240, metadata_runtime))
        if isinstance(metadata_runtime, int)
        else 110
    )
    metadata_rt = metadata.get("rt_rating")
    rt_default = metadata_rt if isinstance(metadata_rt, int) else 70
    plot_sentiment_default = float(metadata.get("plot_sentiment", 0.0))
    plot_mood_default = metadata.get("plot_mood") or "positive"
    if text_sentiment_score is not None:
        plot_sentiment_default = text_sentiment_score
        plot_mood_default = text_mood_override or plot_mood_default

    with st.form("new_film_form"):
        form_title = custom_title
        runtime = st.slider("Runtime (min)", 60, 240, runtime_default)
        rt_rating = st.slider("Rotten Tomatoes score", 0, 100, rt_default)
        plot_sentiment = st.slider(
            "Plot sentiment (approximate)",
            -1.0,
            1.0,
            plot_sentiment_default,
            step=0.01,
        )

        genre_options = sorted(df["simple_genre"].dropna().unique())
        default_genre = metadata.get("simple_genre")
        if default_genre not in genre_options:
            default_genre = genre_options[0]
        simple_genre = st.selectbox("Genre category", genre_options, index=genre_options.index(default_genre))

        mood_options = ["positive", "negative"]
        default_mood = plot_mood_default
        if default_mood not in mood_options:
            default_mood = "positive"
        plot_mood = st.radio("Plot mood", options=mood_options, horizontal=True, index=mood_options.index(default_mood))

        decade_options = ["Unknown"] + sorted(
            [d for d in df["Decade"].unique() if d and d != "Unknown"],
            key=lambda decade: int(decade.rstrip("s")) if decade.endswith("s") else 9999,
        )
        decade_default = metadata.get("Decade") or "Unknown"
        if decade_default not in decade_options:
            decade_default = "Unknown"
        decade = st.selectbox("Decade", decade_options, index=decade_options.index(decade_default))

        submitted = st.form_submit_button("Predict rating")

    if submitted:
        new_features = pd.DataFrame(
            {
                "Runtime": [runtime],
                "rt_rating": [rt_rating],
                "plot_sentiment": [plot_sentiment],
                "simple_genre": [simple_genre],
                "plot_mood": [plot_mood],
                "Decade": [decade],
            }
        )

        new_score = format_prediction(model.predict(new_features)[0])
        st.success(f"{form_title} would likely score around {new_score:.1f} / 10")
        st.write("You adjusted runtime, RT score, mood, and decade for this estimate.")

    # st.caption("Streamlit app powered by the processed dataset and a cached random forest pipeline.")


if __name__ == "__main__":
    main()
