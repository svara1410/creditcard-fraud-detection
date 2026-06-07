# Credit Card Fraud Detection

End-to-end ML project to detect fraudulent credit card transactions.

## Live Demo
Run locally with `python app.py`

## Dataset
[Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
284,807 transactions | 492 fraud cases | 0.17% fraud rate

## Approach
- Explored data with full EDA (class imbalance, amount distribution, time analysis)
- Handled severe class imbalance using SMOTE
- Trained and compared Logistic Regression and Random Forest
- Evaluated using Precision, Recall, F1-Score, ROC-AUC (not just accuracy)
- Built Gradio UI with nearest-neighbor transaction matching

## Results
| Model | ROC-AUC | Fraud Recall | Fraud Precision |
|---|---|---|---|
| Logistic Regression | 0.97 | 0.92 | 0.06 |
| Random Forest | 0.97 | 0.84 | 0.85 |

Random Forest chosen for better precision — fewer false alarms.

## How to Run
```bash
pip install -r requirements.txt
python model.py    # train and save model
python app.py      # launch UI at localhost:7860
```

## Project Structure
creditcard-fraud/
├── model.py          # training pipeline
├── app.py            # gradio UI
├── notebook.ipynb    # EDA and visualizations
├── requirements.txt
└── README.md

## Tech Stack
Python | Scikit-learn | Imbalanced-learn | Random Forest | SMOTE | Gradio | Pandas | Matplotlib | Seaborn

## Key Concepts Demonstrated
- Class imbalance handling with SMOTE
- ROC-AUC evaluation on imbalanced datasets
- Nearest neighbor transaction matching for user input
- End-to-end ML pipeline from raw data to deployed UI