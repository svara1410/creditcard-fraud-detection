import gradio as gr
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score

# ── Load saved model ──────────────────────────────────────────────────────────
model         = joblib.load("fraud_model.pkl")
scaler        = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# ── Load a sample of real data for single transaction testing ─────────────────
raw_df = pd.read_csv("creditcard.csv")
raw_df["Amount_scaled"] = scaler.fit_transform(raw_df[["Amount"]])
raw_df["Time_scaled"]   = scaler.fit_transform(raw_df[["Time"]])

# Separate real fraud and legit rows
real_fraud = raw_df[raw_df["Class"] == 1].copy()
real_legit = raw_df[raw_df["Class"] == 0].copy()

# ── Tab 1: Upload & Analyse CSV ───────────────────────────────────────────────
def analyse_csv(file):
    if file is None:
        return "Please upload a CSV file.", None, None, None

    df = pd.read_csv(file.name)
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        return f"Missing columns: {missing}", None, None, None

    df_scaled = df.copy()
    df_scaled["Amount"] = scaler.fit_transform(df_scaled[["Amount"]])
    df_scaled["Time"]   = scaler.fit_transform(df_scaled[["Time"]])
    X = df_scaled[feature_names]

    preds  = model.predict(X)
    probas = model.predict_proba(X)[:, 1]

    fraud_count = int(preds.sum())
    total       = len(preds)

    summary = (
        f"Total transactions:  {total}\n"
        f"Fraud detected:      {fraud_count}\n"
        f"Legitimate:          {total - fraud_count}\n"
        f"Fraud rate:          {fraud_count/total*100:.3f}%\n"
        f"Avg fraud prob:      {probas[preds==1].mean()*100:.1f}% (flagged txns)"
        if fraud_count > 0 else
        f"Total transactions:  {total}\n"
        f"Fraud detected:      0\n"
        f"Legitimate:          {total}"
    )

    # Confusion matrix (only if Class column exists)
    cm_path = None
    if "Class" in df.columns:
        cm = confusion_matrix(df["Class"], preds)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Legit","Fraud"],
                    yticklabels=["Legit","Fraud"], ax=ax)
        ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        cm_path = "cm_upload.png"
        plt.savefig(cm_path, dpi=150)
        plt.close()

        roc = roc_auc_score(df["Class"], probas)
        summary += f"\nROC-AUC on uploaded file: {roc:.4f}"

    # Result table
    out_df = df[["Time","Amount"]].copy()
    out_df["Prediction"]          = ["FRAUD" if p==1 else "Legit" for p in preds]
    out_df["Fraud Probability %"] = (probas * 100).round(2)
    out_df = out_df.sort_values("Fraud Probability %", ascending=False)

    return summary, out_df.head(50), cm_path


# ── Tab 2: Real single transaction from dataset ───────────────────────────────
def get_sample(label):
    pool = real_fraud if label == "Fraud sample" else real_legit
    row  = pool.sample(1).iloc[0]
    return float(round(row["Amount"], 2)), float(round(row["Time"], 0))


def predict_real(amount, time):
    # Find the closest real transaction by Amount and Time
    # This is nearest neighbor matching — a real ML concept
    df_search = raw_df.copy()
    df_search["dist"] = (
        ((df_search["Amount"] - amount) / (raw_df["Amount"].std() + 1e-9)) ** 2 +
        ((df_search["Time"]   - time)   / (raw_df["Time"].std()   + 1e-9)) ** 2
    )
    closest = df_search.nsmallest(1, "dist").iloc[0]

    # Build feature row using scaled values
    feat_row = {}
    for col in feature_names:
        if col == "Amount":
            feat_row[col] = closest["Amount_scaled"]
        elif col == "Time":
            feat_row[col] = closest["Time_scaled"]
        else:
            feat_row[col] = closest[col]

    X     = pd.DataFrame([feat_row])[feature_names]
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1] * 100

    actual    = "Fraud" if closest["Class"] == 1 else "Legit"
    predicted = "FRAUD DETECTED" if pred == 1 else "Legitimate"
    correct   = "CORRECT" if (closest["Class"]==1) == (pred==1) else "WRONG"

    return (
        f"Model prediction:   {predicted}\n"
        f"Fraud probability:  {proba:.2f}%\n"
        f"Actual label:       {actual}\n"
        f"Prediction:         {correct}\n\n"
        f"Closest match found:\n"
        f"  Amount: ₹{closest['Amount']:.2f}  "
        f"  Time: {int(closest['Time'])}s"
    )


# ── Tab 3: Model performance stats ───────────────────────────────────────────
def show_stats():
    sample = raw_df.sample(5000, random_state=42)
    sample_scaled = sample.copy()
    sample_scaled["Amount"] = scaler.fit_transform(sample_scaled[["Amount"]])
    sample_scaled["Time"]   = scaler.fit_transform(sample_scaled[["Time"]])
    X = sample_scaled[feature_names]
    y = sample["Class"]

    preds  = model.predict(X)
    probas = model.predict_proba(X)[:, 1]
    roc    = roc_auc_score(y, probas)

    fraud_in_sample = int(y.sum())
    detected        = int((preds[y==1]).sum())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Confusion matrix
    cm = confusion_matrix(y, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit","Fraud"],
                yticklabels=["Legit","Fraud"], ax=axes[0])
    axes[0].set_title("Confusion Matrix (5K sample)")
    axes[0].set_ylabel("Actual"); axes[0].set_xlabel("Predicted")

    # Fraud probability distribution
    axes[1].hist(probas[y==0], bins=50, alpha=0.6, label="Legit", color="steelblue")
    axes[1].hist(probas[y==1], bins=50, alpha=0.8, label="Fraud", color="crimson")
    axes[1].set_title("Fraud Probability Distribution")
    axes[1].set_xlabel("Predicted Fraud Probability")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    path = "stats.png"
    plt.savefig(path, dpi=150)
    plt.close()

    stats = (
        f"Sample size:       5,000 transactions\n"
        f"Fraud in sample:   {fraud_in_sample}\n"
        f"Fraud detected:    {detected} / {fraud_in_sample}\n"
        f"ROC-AUC:           {roc:.4f}\n"
        f"Model:             Random Forest (20 trees, SMOTE balanced)"
    )
    return stats, path


# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="Credit Card Fraud Detector") as app:

    gr.Markdown("""
    # Credit Card Fraud Detection
    **Random Forest + SMOTE | Trained on 284,807 real transactions | ROC-AUC: 0.97**
    """)

    with gr.Tabs():

        # Tab 1 — Upload CSV
        with gr.Tab("Analyse CSV"):
            gr.Markdown("Upload any CSV with the same structure as the Kaggle dataset.")
            file_input = gr.File(label="Upload CSV", file_types=[".csv"])
            run_btn    = gr.Button("Analyse", variant="primary")
            with gr.Row():
                summary_out = gr.Textbox(label="Summary", lines=8)
                cm_out      = gr.Image(label="Confusion Matrix")
            table_out = gr.Dataframe(label="Top results by fraud probability", wrap=True)
            run_btn.click(analyse_csv,
                          inputs=[file_input],
                          outputs=[summary_out, table_out, cm_out])

        # Tab 2 — Real transaction test
        with gr.Tab("Test Single Transaction"):
            gr.Markdown("""
            Enter a transaction — the model finds the closest real transaction 
            from 284,807 records and predicts if it's fraud.
            """)
            with gr.Row():
                amt_input  = gr.Number(label="Transaction Amount (₹)", value=150.0)
                time_input = gr.Number(label="Time (seconds since midnight)", value=50000)
            with gr.Row():
                gr.Examples(
                    examples=[[1.0, 406], [150.0, 50000], [8000.0, 1000], [0.01, 100]],
                    inputs=[amt_input, time_input],
                    label="Try these examples"
                )
            predict_btn = gr.Button("Predict", variant="primary")
            result_out  = gr.Textbox(label="Result", lines=8)
            predict_btn.click(predict_real,
                              inputs=[amt_input, time_input],
                              outputs=[result_out])

        # Tab 3 — Model stats
        with gr.Tab("Model Performance"):
            gr.Markdown("Evaluate model on a 5,000 transaction sample with full metrics.")
            stats_btn  = gr.Button("Run Evaluation", variant="primary")
            with gr.Row():
                stats_out = gr.Textbox(label="Metrics", lines=8)
                plot_out  = gr.Image(label="Confusion Matrix + Distribution")
            stats_btn.click(show_stats, inputs=[], outputs=[stats_out, plot_out])

app.launch()