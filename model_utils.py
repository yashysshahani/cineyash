from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    r2_score, mean_absolute_error, root_mean_squared_error,
    accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,
    make_scorer)
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
import numpy as np
import mord
import matplotlib.pyplot as plt
from datetime import datetime
import json


# This was used more when I was testing different models in a notebook

def model_assess(df, y_pred, y_test, task, assess="Rating", tolerance=1, plot=True):
    results = {}

    test_df = df.loc[y_test.index]
    test_df["preds"] = y_pred
    test_df = test_df[["Movie", assess, "preds", "rt_rating"]]

    if task == "regression":
        results["R2"] = r2_score(y_test, y_pred)
        results["MAE"] = mean_absolute_error(y_test, y_pred),
        results["RMSE"] = root_mean_squared_error(y_test, y_pred)
    
    elif task == "classification":
        results["Accuracy"] = accuracy_score(y_test, y_pred)
        results["F1"] = f1_score(y_test, y_pred, average="macro")
    
    elif task == "ordinal":
        results[f"Acc within {tolerance}"] = accuracy_within_tol(y_test, y_pred, tol=tolerance)
        results["Spearman"] = spearmanr(y_test, y_pred).correlation
        results["MAE"] = mean_absolute_error(y_test, y_pred)

    if plot:
        if task == "classification" or task == "ordinal":
            cm = confusion_matrix(y_test, y_pred, labels=range(1,11))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues", values_format="d")
            plt.title("Confusion Matrix")

        else:
            plt.scatter(y_test, y_pred, alpha=0.6)
            plt.xlabel("True")
            plt.ylabel("Predicted")
            plt.title(f"{task.title()} Predictions")
            plt.axline((0,0), slope=1, color="gray", linestyle="--")

        plt.tight_layout()
        plt.show()

    return results, test_df

def accuracy_within_tol(y_true, y_pred, tol=1):
    return np.mean(np.abs(y_true - y_pred) <= tol)


