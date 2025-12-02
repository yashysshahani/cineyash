import textwrap
import numpy as np

# Sentiment analysis

GOEMO_MODEL = "SamLowe/roberta-base-go_emotions"

# All emotions
GO_LABELS = [
    'admiration','amusement','anger','annoyance','approval','caring','confusion',
    'curiosity','desire','disappointment','disapproval','disgust','embarrassment',
    'excitement','fear','gratitude','grief','joy','love','nervousness','optimism',
    'pride','realization','relief','remorse','sadness','surprise','neutral'
]

_emo_pipeline = None


def _chunk_text(text, max_chars=900):
    return textwrap.wrap(text, max_chars, replace_whitespace=False, break_long_words=False)


def _init_emotion_pipeline():
    global _emo_pipeline
    if _emo_pipeline is not None:
        return _emo_pipeline

    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            TextClassificationPipeline,
        )

        tok = AutoTokenizer.from_pretrained(GOEMO_MODEL)
        mdl = AutoModelForSequenceClassification.from_pretrained(GOEMO_MODEL)

        _emo_pipeline = TextClassificationPipeline(
            model=mdl,
            tokenizer=tok,
            top_k=None,
            return_all_scores=True,
            truncation=True,
            max_length=512,
        )
    except Exception:
        _emo_pipeline = None

    return _emo_pipeline


def goemotion_vector(text):
    if not isinstance(text, str) or not text.strip():
        return np.array([np.nan] * len(GO_LABELS))

    pipe = _init_emotion_pipeline()
    if pipe is None:
        return np.array([0.0] * len(GO_LABELS))

    rows = []
    for chunk in _chunk_text(text):
        scores = pipe(chunk)[0]
        d = {s["label"]: float(s["score"]) for s in scores}
        rows.append([d.get(lbl, 0.0) for lbl in GO_LABELS])

    avg = np.array(rows).mean(axis=0)
    return avg


SELECTED = ["joy", "sadness", "fear", "anger", "disgust", "surprise", "neutral"]
SELECTED_NONEUTRAL = ["joy", "sadness", "fear", "anger", "disgust", "surprise"]

POSITIVE = {"joy", "surprise"}
NEGATIVE = {"sadness", "fear", "anger", "disgust"}
NEUTRAL  = {"neutral"}


def subset_emotions(prob_vector):
    d = dict(zip(GO_LABELS, prob_vector))
    return [d.get(e, np.nan) for e in SELECTED_NONEUTRAL]

def polarity_from_subset(emo_row):
    pos = sum(emo_row[f"emo_{e}"] for e in POSITIVE if f"emo_{e}" in emo_row)
    neg = sum(emo_row[f"emo_{e}"] for e in NEGATIVE if f"emo_{e}" in emo_row)
    total = pos + neg
    return (pos - neg) / total if total > 0 else 0.0


def polarity_from_vector(prob_vector):
    d = dict(zip(GO_LABELS, prob_vector))
    pos = sum(np.nan_to_num(d.get(e, 0.0)) for e in POSITIVE)
    neg = sum(np.nan_to_num(d.get(e, 0.0)) for e in NEGATIVE)
    total = pos + neg
    return (pos - neg) / total if total > 0 else 0.0

