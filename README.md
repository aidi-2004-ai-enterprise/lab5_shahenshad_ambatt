Lab 5 — Company Bankruptcy Prediction

End-to-end ML pipeline (LogReg, Random Forest, XGBoost) for predicting company bankruptcy.
Implements the decisions from Lab 4 and produces EDA, feature selection, model tuning, evaluation, SHAP, and PSI drift checks.

Repo structure
.
├─ training_pipeline.py
├─ Lab4.md
├─ Lab5.md
├─ lab4_company_bankruptcy.md
├─ requirements.txt
├─ outputs/
│  ├─ report.md
│  ├─ metrics.csv
│  ├─ psi_top.csv
│  ├─ psi_top_bar.png
│  ├─ selected_features.csv
│  ├─ best_params.json
│  ├─ eda/            # histograms, boxplots, correlation heatmap
│  └─ plots/          # ROC + calibration per model
└─ (ignored)
   └─ outputs/models/ # trained .joblib files

Setup
1) Create a virtual environment

Windows (PowerShell)

python -m venv .venv
. .\.venv\Scripts\Activate.ps1


macOS/Linux

python3 -m venv .venv
source .venv/bin/activate

2) Install dependencies
pip install -r requirements.txt


requirements.txt (minimal):

pandas
numpy
scikit-learn
imbalanced-learn
xgboost
shap
matplotlib
seaborn
joblib
tabulate

How to run

Windows (PowerShell)

python .\training_pipeline.py --data_path "C:/path/to/data.csv" --output_dir outputs


macOS/Linux

python training_pipeline.py --data_path /path/to/data.csv --output_dir outputs


Optional flags:

--target "Bankrupt?" (change if your CSV uses a different target column)

--n_iter 25 (RandomizedSearch iterations; lower to speed up)

--keep_k 40 (top-K features kept after XGBoost importance)

Examples:

# Faster test run
python .\training_pipeline.py --data_path "C:/path/to/data.csv" --output_dir outputs --n_iter 10 --keep_k 30

What the pipeline does

EDA: histograms, boxplots, correlation heatmap (numeric only).

Preprocessing: correlation filter (|corr| ≥ 0.9) then top-K features by XGBoost importance; scaling only for Logistic Regression.

Imbalance handling:

Logistic Regression → class_weight="balanced"

RandomForest & XGBoost → SMOTE inside CV folds (train only)

XGBoost also uses scale_pos_weight.

Hyperparameter tuning: RandomizedSearchCV with stratified 5-fold CV (ROC-AUC scoring).

Evaluation: ROC-AUC, PR-AUC, F1, Brier (train & test) + ROC & calibration plots.

Interpretability: SHAP summary plot on the best model.

Drift: PSI between train/test; CSV + bar chart.

Report: Markdown with jot notes, metrics table, selected features (top 20), best hyperparameters, PSI summary/plot, and deployment recommendation.

Outputs

outputs/report.md – human-readable summary with links/figures

outputs/metrics.csv – per-model metrics (train/test)

outputs/selected_features.csv – final feature set after selection

outputs/best_params.json – tuned hyperparameters for each model

outputs/psi_top.csv & outputs/psi_top_bar.png – top drifted features

outputs/eda/ – EDA images

outputs/plots/ – ROC & calibration per model

outputs/models/ – saved models (*.joblib, git-ignored by default)

CLI arguments
Arg	Type	Default	Description
--data_path	str	(required)	Path to CSV with the target column
--target	str	Bankrupt?	Target column name
--output_dir	str	outputs	Folder to write all outputs
--n_iter	int	25	RandomizedSearch iterations per model
--keep_k	int	40	Keep top-K features via XGBoost importance




Notes

Raw data is not included in the repo. Use the assignment’s dataset source.

Trained models are saved to outputs/models/ and ignored by git to keep the repo light.

The recommended deployment model is chosen by highest test ROC-AUC (see report.md for rationale).
