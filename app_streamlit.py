


from pathlib import Path
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from textblob import TextBlob
import nltk
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except Exception:
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_recall_fscore_support
)

def load_tsv_like(path_or_bytes):
    if isinstance(path_or_bytes, (str, Path)):
        df = pd.read_csv(path_or_bytes, delimiter="\\t", header=None)
    else:
        df = pd.read_csv(path_or_bytes, delimiter="\\t", header=None)
    if df.shape[1] < 2:
        raise ValueError("Expected at least 2 columns (text + label).")
    df = df[[df.columns[0], df.columns[-1]]].copy()
    df.columns = ["Text", "Sentiments"]
    df["Sentiments"] = pd.to_numeric(df["Sentiments"], errors="coerce").fillna(0).astype(int).clip(0,1)
    return df

def summarize_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p_pos, r_pos, f1_pos, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": acc,
        "precision_pos": p_pos,
        "recall_pos": r_pos,
        "f1_pos": f1_pos,
        "precision_macro": p_m,
        "recall_macro": r_m,
        "f1_macro": f1_m
    }

def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative","Positive"])
    fig, ax = plt.subplots(figsize=(5,4))
    disp.plot(values_format='d', ax=ax)
    ax.set_title(title)
    return fig

def predict_textblob(texts):
    polarity = []
    subjectivity = []
    for t in texts:
        tb = TextBlob(str(t))
        polarity.append(tb.sentiment.polarity)
        subjectivity.append(tb.sentiment.subjectivity)
    pred = (np.array(polarity) > 0).astype(int)
    return {"polarity": np.array(polarity), "subjectivity": np.array(subjectivity), "pred": pred}

def predict_vader_full(texts):
    sid = SentimentIntensityAnalyzer()
    comp, neu = [], []
    for t in texts:
        s = sid.polarity_scores(str(t))
        comp.append(s["compound"])
        neu.append(s["neu"])
    pred = (np.array(comp) > 0).astype(int)
    subj_proxy = (1 - np.array(neu)).clip(0,1)
    return {"compound": np.array(comp), "neu": np.array(neu), "subj_proxy": subj_proxy, "pred": pred}

def fig_scatter_textblob(polarity, subjectivity, title):
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(polarity, subjectivity, alpha=0.6)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_xlabel("Polarity (-1 to 1)")
    ax.set_ylabel("Subjectivity (0 to 1)")
    ax.set_title(title)
    return fig

def fig_scatter_vader(compound, subj_proxy, title):
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(compound, subj_proxy, alpha=0.6)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_xlabel("Polarity (-1 to 1)")
    ax.set_ylabel("Subjectivity (0 to 1)")
    ax.set_title(title)
    return fig

def fig_scatter_combined(tb_x, tb_y, va_x, va_y, title):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(tb_x, tb_y, alpha=0.6, marker="x", label="TextBlob (polarity, subjectivity)")
    ax.scatter(va_x, va_y, alpha=0.6, marker="o", label="VADER (compound, 1 - neutral)")
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_xlabel("Polarity (-1 to 1)")
    ax.set_ylabel("Subjectivity (0 to 1)")
    ax.set_title(title)
    ax.legend()
    return fig

def to_csv_bytes(df):
    import io
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

st.set_page_config(page_title="Health & Social Sentiment Analyzer", layout="wide")
st.title("Health & Social Sentiment Analyzer (TextBlob & VADER)")

with st.sidebar:
    st.header("Datasets")
    default_tb = Path("health_labelled.txt")
    default_va = Path("vader_social_labelled.txt")
    tb_path = st.text_input("TextBlob dataset (TSV, no header)", value=str(default_tb))
    va_path = st.text_input("VADER dataset (TSV, no header)", value=str(default_va))

    st.caption("Optionally upload datasets (overrides paths above):")
    up_tb = st.file_uploader("Upload TextBlob dataset", type=["txt","tsv"])
    up_va = st.file_uploader("Upload VADER dataset", type=["txt","tsv"])

    run_btn = st.button("Run Analysis", type="primary")

if run_btn:
    try:
        df_tb = load_tsv_like(up_tb if up_tb is not None else tb_path)
        df_va = load_tsv_like(up_va if up_va is not None else va_path)
    except Exception as e:
        st.error(f"Failed to load datasets: {e}")
        st.stop()

    # TextBlob
    st.subheader("TextBlob — Health Dataset")
    st.text("RAW SENTIMENTS")
    st.text(df_tb.head().to_string(index=True))
    st.text(f"Data dimesnion =  {df_tb.shape}")

    tb = predict_textblob(df_tb["Text"].tolist())
    tb_df = pd.DataFrame({
        "Polarity_score": tb["polarity"],
        "subjectivity_score": tb["subjectivity"],
        "Predicted_Label": tb["pred"]
    })

    st.text("\\n QUANTITATIVE RESULTS")
    st.dataframe(tb_df.head())

    st.text("\\n DESCRIPTIVE STATISTICS")
    st.text(tb_df[["Polarity_score","subjectivity_score","Predicted_Label"]].describe().to_string())

    tb_metrics = summarize_metrics(df_tb["Sentiments"].to_numpy(), tb["pred"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{tb_metrics['accuracy']:.2%}")
    c2.metric("F1 (pos)", f"{tb_metrics['f1_pos']:.2f}")
    c3.metric("Precision (pos)", f"{tb_metrics['precision_pos']:.2f}")
    c4.metric("Recall (pos)", f"{tb_metrics['recall_pos']:.2f}")

    st.pyplot(fig_scatter_textblob(tb["polarity"], tb["subjectivity"], f"TextBlob Sentiment Distribution — {Path(tb_path).stem}"))
    st.pyplot(plot_cm(df_tb["Sentiments"].to_numpy(), tb["pred"], "Confusion Matrix — TextBlob"))

    tb_save = df_tb.copy()
    tb_save["TB_polarity"] = tb["polarity"]
    tb_save["TB_subjectivity"] = tb["subjectivity"]
    tb_save["TB_pred_default"] = tb["pred"]
    tb_bytes = to_csv_bytes(tb_save.drop(columns=["Text"]))
    st.download_button("Download TextBlob predictions (CSV)", tb_bytes, file_name="textblob_predictions.csv")

    st.divider()

    # VADER
    st.subheader("VADER — Social Dataset")
    st.text("RAW SENTIMENTS")
    st.text(df_va.head().to_string(index=True))
    st.text(f"Data dimesnion =  {df_va.shape}")

    va = predict_vader_full(df_va["Text"].tolist())
    va_df = pd.DataFrame({
        "Polarity_score": va["compound"],
        "subjectivity_score": va["subj_proxy"],
        "Predicted_Label": va["pred"]
    })

    st.text("\\n QUANTITATIVE RESULTS")
    st.dataframe(va_df.head())

    st.text("\\n DESCRIPTIVE STATISTICS")
    st.text(va_df[["Polarity_score","subjectivity_score","Predicted_Label"]].describe().to_string())

    va_metrics = summarize_metrics(df_va["Sentiments"].to_numpy(), va["pred"])
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Accuracy", f"{va_metrics['accuracy']:.2%}")
    d2.metric("F1 (pos)", f"{va_metrics['f1_pos']:.2f}")
    d3.metric("Precision (pos)", f"{va_metrics['precision_pos']:.2f}")
    d4.metric("Recall (pos)", f"{va_metrics['recall_pos']:.2f}")

    st.pyplot(fig_scatter_vader(va["compound"], va["subj_proxy"], f"VADER Sentiment Distribution — {Path(va_path).stem}"))
    st.pyplot(plot_cm(df_va["Sentiments"].to_numpy(), va["pred"], "Confusion Matrix — VADER"))

    va_save = df_va.copy()
    va_save["VADER_compound"] = va["compound"]
    va_save["VADER_neu"] = va["neu"]
    va_save["VADER_subj_proxy"] = va["subj_proxy"]
    va_save["VADER_pred_default"] = va["pred"]
    va_bytes = to_csv_bytes(va_save.drop(columns=["Text"]))
    st.download_button("Download VADER predictions (CSV)", va_bytes, file_name="vader_predictions.csv")

    st.divider()

    # Combined
    st.subheader("Combined Comparison (Same Axes)")
    fig_combined = fig_scatter_combined(tb["polarity"], tb["subjectivity"],
                                        va["compound"], va["subj_proxy"],
                                        "Model Comparison — Same Axes (TextBlob vs VADER)")
    st.pyplot(fig_combined)

    summary_df = pd.DataFrame([
        {"model":"TextBlob_default", **tb_metrics},
        {"model":"VADER_default", **va_metrics},
    ])
    st.dataframe(summary_df)
    st.download_button("Download Summary (CSV)", to_csv_bytes(summary_df), file_name="summary_metrics.csv")
