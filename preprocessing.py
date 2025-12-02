# imports

from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd


def expand_multilabel_column(df, column_name, prefix):

    lists = df[column_name].fillna("").apply(
        lambda s: [x.strip() for x in str(s).split(",") if x.strip()]
    )

    mlb = MultiLabelBinarizer()
    mat = mlb.fit_transform(lists)

    features = pd.DataFrame(
        mat,
        columns=[f"{prefix}::{name}" for name in mlb.classes_],
        index=df.index
    )
    
    return features

def simplify_genre(genre):
    genre = genre.lower()

    if any(g in genre for g in ["drama", "legal", "biography", "romance"]):
        return "Drama"
    
    if any(g in genre for g in ["comedy", "romcom", "family"]):
        return "Comedy"
    
    if any(g in genre for g in ["action", "adventure", "war"]):
        return "Action/Adventure"
    
    if any(g in genre for g in ["horror", "thriller", "crime", "mystery"]):
        return "Horror/Thriller"
    
    if any(g in genre for g in ["sci-fi", "science fiction", "fantasy"]):
        return "Sci-Fi/Fantasy"
    
    if "musical" in genre:
        return "Musical"
    
    return "Other"

def year_to_decade(year):
    if not year:
        return None
    
    decade_start = (year // 10) * 10

    return f"{decade_start}s"

def add_decade_column(df, year_col="Release Year", new_col="Decade"):
    for i, row in df.iterrows():
        year = df.loc[i, year_col]
        decade_str = year_to_decade(year)
        df.loc[i, new_col] = decade_str

    return df
