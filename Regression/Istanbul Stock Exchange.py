import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.patches import Patch
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings(
    "ignore",
    message="Unknown extension is not supported and will be removed",
    category=UserWarning,
    module=r"openpyxl\..*",
)

BASE_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
TEST_SIZE = 0.20
SHOW_FIGURES = True
FEATURES = ["SP", "DAX", "FTSE", "NIKKEI", "BOVESPA", "EU", "EM"]
TARGET = "ISE_USD"
DATASET_ID = "D22"
DATASET_NAME = "Istanbul Stock Exchange"

# region Chart Colors
BG_COLOR = "#FFFFFF"
TITLE_COLOR = "#2B1450"
TITLE_ACCENT = "#C9578F"
TEXT_COLOR = "#665B78"
PURPLE = "#8B5CC7"
PINK = "#E3A0C4"
TEAL = "#21B8A8"
LAVENDER = "#C9B6F2"
LIGHT_PINK = "#F3A6C8"
MODEL_COLORS = ["#7E57C2", "#6750A4", "#4F5FBF", "#3F72AF", "#5B9BD5"]
# endregion

# region Center Figure Window
def center_figure_window():
    manager = plt.get_current_fig_manager()
    window = getattr(manager, "window", None)
    if window is None:
        return
    try:
        window.update_idletasks()
        window.update()
        width = window.winfo_width()
        height = window.winfo_height()
        if width <= 1:
            width = window.winfo_reqwidth()
        if height <= 1:
            height = window.winfo_reqheight()
        x = max(20, (window.winfo_screenwidth() - width) // 2)
        y = max(40, (window.winfo_screenheight() - height) // 2 - 40)
        window.geometry(f"+{x}+{y}")
        window.update_idletasks()
    except Exception:
        pass


def show_figure():
    if SHOW_FIGURES:
        plt.show(block=False)
        center_figure_window()
        plt.show()
    else:
        plt.close()


def style_axes(ax):
    ax.set_facecolor(BG_COLOR)
    ax.set_axisbelow(True)
    ax.grid(color=LAVENDER, alpha=0.30, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(LAVENDER)
        spine.set_linewidth(0.8)


def set_chart_title(ax, title):
    ax.set_title("")
    ax.figure.suptitle(title.replace("\n", " "), fontsize=16,
                       fontweight="bold", color=TITLE_ACCENT,
                       x=0.5, y=0.94, ha="center")


def display_figure():
    fig = plt.gcf()
    window = getattr(fig.canvas.manager, "window", None)
    # Keep the chart dimensions consistent with the Abalone dataset.
    width, height = fig.get_size_inches() * fig.dpi
    if window is not None and hasattr(window, "winfo_screenwidth"):
        window.update_idletasks()
        width = min(width, window.winfo_screenwidth() * 0.95)
        height = min(height, window.winfo_screenheight() - 160)
    fig.set_size_inches(width / fig.dpi, height / fig.dpi, forward=True)
    fig.canvas.draw()
    title = fig._suptitle
    if title is not None:
        title_width = title.get_window_extent(fig.canvas.get_renderer()).width
        available_width = fig.bbox.width * 0.94
        if title_width > available_width:
            title.set_fontsize(title.get_fontsize() * available_width / title_width)
    fig.tight_layout(rect=(0.015, 0.015, 0.985, 0.89), pad=1.2)
    show_figure()

# endregion

# region Load Dataset
# Put the original UCI file in DataSets/. Common filenames are supported.
data_dir = BASE_DIR / "DataSets"
candidates = [
    data_dir / "data_akbilgic.xlsx",
    data_dir / "istanbul_stock_exchange.xlsx",
    data_dir / "Istanbul Stock Exchange.xlsx",
    data_dir / "data_akbilgic.csv",
    data_dir / "istanbul_stock_exchange.csv",
]
dataset_path = next((p for p in candidates if p.is_file()), None)
if dataset_path is None:
    raise FileNotFoundError(
        "Dataset not found. Put data_akbilgic.xlsx in the DataSets folder. "
        "Expected path: " + str(candidates[0])
    )
if dataset_path.suffix.lower() in (".xlsx", ".xls"):
    # The original UCI workbook has a description row above the column headers.
    raw_df = pd.read_excel(dataset_path, engine="openpyxl" if dataset_path.suffix.lower() == ".xlsx" else None)
    if not any(str(c).strip().lower() == "date" for c in raw_df.columns):
        raw_df = pd.read_excel(dataset_path, header=1,
                               engine="openpyxl" if dataset_path.suffix.lower() == ".xlsx" else None)
else:
    raw_df = pd.read_csv(dataset_path, sep=None, engine="python")

raw_df.columns = [str(c).strip() for c in raw_df.columns]
rename_map = {
    "date": "Date", "ise": "ISE_TL", "ise_tl": "ISE_TL",
    "ise_usd": "ISE_USD", "ise.1": "ISE_USD", "sp": "SP", "dax": "DAX", "ftse": "FTSE",
    "nikkei": "NIKKEI", "bovespa": "BOVESPA",
    "msci_eu": "EU", "msce_eu": "EU", "eu": "EU",
    "msci_em": "EM", "em": "EM",
}
raw_df = raw_df.rename(columns={c: rename_map.get(c.lower(), c) for c in raw_df.columns})
required_columns = ["Date", TARGET] + FEATURES
missing_columns = sorted(set(required_columns) - set(raw_df.columns))
if missing_columns:
    raise ValueError(f"Missing dataset columns: {missing_columns}. Found: {list(raw_df.columns)}")

stock_df = raw_df.copy()
stock_df["Date"] = pd.to_datetime(stock_df["Date"], errors="coerce", dayfirst=True)
if stock_df["Date"].isna().any():
    raise ValueError("Invalid or missing Date values. Check the source date format.")
for col in [TARGET] + FEATURES + (["ISE_TL"] if "ISE_TL" in stock_df else []):
    stock_df[col] = pd.to_numeric(stock_df[col], errors="coerce")
stock_df = stock_df.replace([np.inf, -np.inf], np.nan)
stock_df = stock_df.sort_values("Date", kind="stable").reset_index(drop=True)
if stock_df["Date"].duplicated().any():
    raise ValueError("Duplicate trading dates detected; resolve them before modeling.")
print("Dataset Path:", dataset_path)
# endregion

# region Data Understanding
print("\nFirst 5 Rows:")
print(stock_df.head().to_string())
print("-" * 60)
print("Dataset Shape:", stock_df.shape)
print("-" * 60)
print("Columns:", stock_df.columns.tolist())
print("-" * 60)
print("Data Types:")
print(stock_df.dtypes)
print("-" * 60)
print("\nDataset Information:")
stock_df.info()
print("-" * 60)
print("Statistical Description:")
print(stock_df.describe().to_string())
print("-" * 60)
print("Missing Values:")
print(stock_df.isnull().sum())
print("-" * 60)
print("Total Missing Values:", stock_df.isnull().sum().sum())
print("-" * 60)
print("Duplicate Rows:", stock_df.duplicated().sum())
print("-" * 60)
print("Date Range:", stock_df["Date"].min(), "to", stock_df["Date"].max())
print("-" * 60)
# endregion

# region Feature Selection
# Same-day association, not a claim of next-day forecasting.
X = stock_df[FEATURES].copy()
y = stock_df[TARGET].copy()
excluded_columns = [c for c in stock_df.columns if c not in FEATURES + [TARGET]]
print("\nSelected Features:")
print(X.head().to_string())
print("Features Shape:", X.shape)
print("Feature Names:", FEATURES)
print("Target Variable:", TARGET)
print("Excluded Columns:", excluded_columns)
print("-" * 60)
# endregion

# region Data Preprocessing
if y.isna().any():
    raise ValueError("Target ISE_USD contains missing values; review source data.")
print("Missing values in model inputs:")
print(X.isna().sum())
print("Numeric missing values are filled using the training-set median.")
# endregion

# region Data Normalization
def create_preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

print("Numeric Features:", FEATURES)
print("Categorical Features: None")
print("Normalization: StandardScaler (fit on training set only)")
print("-" * 60)
# endregion

# region Model Evaluation
split_index = int(len(stock_df) * (1 - TEST_SIZE))
if split_index < 10 or len(stock_df) - split_index < 2:
    raise ValueError("Not enough observations for chronological train/test split.")
x_train, x_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
print("Split Method: Chronological (shuffle=False)")
print("Train Period:", stock_df.loc[0, "Date"], "to", stock_df.loc[split_index - 1, "Date"])
print("Test Period:", stock_df.loc[split_index, "Date"], "to", stock_df.loc[len(stock_df) - 1, "Date"])

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                    max_depth=3, random_state=RANDOM_STATE),
    "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.01),
}
trained_models = {}
predictions = {}
records = []
for model_name, estimator in models.items():
    model_pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", estimator),
    ])
    model_pipeline.fit(x_train, y_train)
    prediction = model_pipeline.predict(x_test)
    trained_models[model_name] = model_pipeline
    predictions[model_name] = prediction
    mse = mean_squared_error(y_test, prediction)
    direction_accuracy = np.mean((prediction > 0) == (y_test.to_numpy() > 0)) * 100
    records.append({
        "Model": model_name,
        "MAE": mean_absolute_error(y_test, prediction),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_test, prediction),
        "R2 Percentage": r2_score(y_test, prediction) * 100,
        "Direction Accuracy": direction_accuracy,
    })

metrics_df = pd.DataFrame(records).sort_values(
    by=["Direction Accuracy", "RMSE"],
    ascending=[False, True],
    kind="stable",
).reset_index(drop=True)
best_model_name = metrics_df.iloc[0]["Model"]
best_prediction = predictions[best_model_name]
print("\nRegression Metrics:")
print(metrics_df.round(6).to_string(index=False))
print("-" * 60)
print("Recommended Model:", best_model_name)
print("Direction Accuracy:", f"{metrics_df.iloc[0]['Direction Accuracy']:.2f}%")
print("Selection Rule: Highest Direction Accuracy; lowest RMSE breaks ties")
print("Note: Comparing models on the test set introduces selection bias; use walk-forward validation for rigorous tuning.")
# endregion

# region Feature Importance
coefficient_model = trained_models["Linear Regression"]
linear_estimator = coefficient_model.named_steps["model"]
coefficient_df = pd.DataFrame({
    "Feature": FEATURES,
    "Coefficient": linear_estimator.coef_,
})
coefficient_df["Absolute Coefficient"] = coefficient_df["Coefficient"].abs()
coefficient_df = coefficient_df.sort_values("Absolute Coefficient", ascending=False).reset_index(drop=True)
coefficient_df["Importance Rank"] = np.arange(1, len(coefficient_df) + 1)
print("\nFeature Importance Based On Standardized Linear Regression Coefficients:")
print(coefficient_df.round(6).to_string(index=False))
print("Most Influential Feature:", coefficient_df.iloc[0]["Feature"])

plot_coefficients = coefficient_df.iloc[::-1]
plt.figure(figsize=(12, 7), facecolor=BG_COLOR)
ax = plt.gca()
style_axes(ax)
set_chart_title(ax, "Istanbul Stock Exchange - Most Influential International Indices")
bar_colors = [LIGHT_PINK if value < 0 else PURPLE for value in plot_coefficients["Coefficient"]]
bars = ax.barh(plot_coefficients["Feature"], plot_coefficients["Absolute Coefficient"],
               color=bar_colors, edgecolor="white")
max_coefficient = max(coefficient_df["Absolute Coefficient"].max(), 1e-9)
for bar, value, coefficient in zip(bars, plot_coefficients["Absolute Coefficient"],
                                   plot_coefficients["Coefficient"]):
    ax.text(value + max_coefficient * 0.02, bar.get_y() + bar.get_height() / 2,
            f"{coefficient:+.4f}", ha="left", va="center", fontsize=10,
            color=TITLE_COLOR, fontweight="bold")
ax.set_xlim(0, max_coefficient * 1.30)
ax.set_xlabel("Absolute Standardized Coefficient", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("International Index", fontsize=12, color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
legend = ax.legend(handles=[Patch(facecolor=PURPLE, label="Positive coefficient"),
                            Patch(facecolor=LIGHT_PINK, label="Negative coefficient")],
                   loc="lower right", facecolor=BG_COLOR, edgecolor=LAVENDER)
for label in legend.get_texts():
    label.set_color(TEXT_COLOR)
display_figure()
# endregion

# region Results
best_metrics = metrics_df.iloc[0].to_dict()
comparison_df = metrics_df.copy()
comparison_df["RMSE Rank"] = comparison_df["RMSE"].rank(method="min").astype(int)
comparison_df["R2 Rank"] = comparison_df["R2"].rank(ascending=False, method="min").astype(int)
print("\nModel Comparison:")
print(comparison_df.round(6).to_string(index=False))
print("-" * 60)
print("Best Model Metrics:")
for metric in ["MAE", "MSE", "RMSE", "R2", "R2 Percentage", "Direction Accuracy"]:
    print(f"{metric}: {best_metrics[metric]:.6f}")
# endregion

# region Correlation Heatmap
correlation_columns = FEATURES + [TARGET]
correlation_matrix = stock_df[correlation_columns].corr()
plt.figure(figsize=(8.8, 6.2), facecolor=BG_COLOR)
ax = plt.gca()
ax.set_facecolor(BG_COLOR)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f",
            cmap=sns.blend_palette(["#FFF7FB", "#F6CFE3", "#D69ACD", "#9562AF", "#51267D"], as_cmap=True),
            vmin=-1, vmax=1, center=0, linewidths=0.8, linecolor=BG_COLOR,
            cbar_kws={"label": "Correlation"}, annot_kws={"fontsize": 9, "fontweight": "bold"}, ax=ax)
set_chart_title(ax, "Istanbul Stock Exchange - Correlation Heatmap")
ax.tick_params(axis="x", rotation=30, colors=TEXT_COLOR)
plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
ax.tick_params(axis="y", rotation=0, colors=TEXT_COLOR)
ax.set_xlabel("Indices", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Indices", fontsize=12, color=TEXT_COLOR)
display_figure()
# endregion

# region Data Visualization
plt.figure(figsize=(11, 7), facecolor=BG_COLOR)
ax = plt.gca()
style_axes(ax)
set_chart_title(ax, f"Istanbul Stock Exchange - Actual vs Predicted Returns ({best_model_name})")
ax.scatter(y_test, best_prediction, color=TEAL, alpha=0.78, edgecolors=BG_COLOR, linewidths=0.7)
minimum = min(y_test.min(), best_prediction.min())
maximum = max(y_test.max(), best_prediction.max())
ax.plot([minimum, maximum], [minimum, maximum], color=PINK, linewidth=2, label="Ideal prediction")
ax.set_xlabel("Actual ISE_USD Return", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Predicted ISE_USD Return", fontsize=12, color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
ax.legend(facecolor=BG_COLOR, edgecolor=LAVENDER)
display_figure()

# Direction Accuracy uses the sign of returns, not MAPE-based regression accuracy.
# The chart follows the Abalone five-bar model comparison layout.
accuracy_plot = metrics_df.sort_values(
    "Direction Accuracy", ascending=False, kind="stable"
)
plt.figure(figsize=(11, 7), facecolor=BG_COLOR)
ax = plt.gca()
style_axes(ax)
set_chart_title(ax, "Istanbul Stock Exchange - Regression Model Direction Accuracy")
accuracy_bars = ax.bar(
    accuracy_plot["Model"],
    accuracy_plot["Direction Accuracy"],
    color=MODEL_COLORS[:len(accuracy_plot)],
    edgecolor="white",
    linewidth=0.8,
)
for bar in accuracy_bars:
    value = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1.0,
        f"{value:.2f}%",
        ha="center", va="bottom", fontsize=10,
        color=TITLE_COLOR, fontweight="bold",
    )
ax.set_xlabel("Model", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Direction Accuracy (%)", fontsize=12, color=TEXT_COLOR)
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 101, 20))
ax.margins(x=0.08)
ax.tick_params(axis="x", rotation=20, colors=TEXT_COLOR)
ax.tick_params(axis="y", colors=TEXT_COLOR)
display_figure()

plt.figure(figsize=(11, 7), facecolor=BG_COLOR)
ax = plt.gca()
style_axes(ax)
set_chart_title(ax, f"Istanbul Stock Exchange - Residual Distribution ({best_model_name})")
residuals = y_test.to_numpy() - best_prediction
ax.hist(residuals, bins=25, color=PURPLE, alpha=0.85, edgecolor="white")
ax.axvline(0, color=PINK, linewidth=2)
ax.set_xlabel("Residual (Actual - Predicted)", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Frequency", fontsize=12, color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
display_figure()

# endregion

# region Save Results
output_folder = BASE_DIR / "DataSets" / "Processed"
output_folder.mkdir(parents=True, exist_ok=True)
processed_df = stock_df.copy()
processed_df["Predicted_ISE_USD"] = np.nan
processed_df.loc[x_test.index, "Predicted_ISE_USD"] = best_prediction
processed_path = output_folder / "istanbul_stock_exchange_regression_results.csv"
processed_df.to_csv(processed_path, index=False)
metrics_path = output_folder / "istanbul_stock_exchange_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
print("\nRegression Results Saved Successfully!")
print("Processed Dataset:", processed_path)
print("Metrics:", metrics_path)
# endregion

# region Training Models
model_folder = BASE_DIR / "ModelsOutcome"
model_folder.mkdir(parents=True, exist_ok=True)
model_paths = {}
feature_columns_path = model_folder / "istanbul_stock_exchange_feature_columns.joblib"
joblib.dump(FEATURES, feature_columns_path)
print("Saved Feature Columns:", feature_columns_path)
for model_name, model_pipeline in trained_models.items():
    model_file_name = model_name.lower().replace(" ", "_") + ".joblib"
    model_path = model_folder / f"istanbul_stock_exchange_{model_file_name}"
    joblib.dump({"model": model_pipeline, "model_name": model_name,
                 "target_column": TARGET, "feature_columns": FEATURES,
                 "prediction_horizon": "same-day association"}, model_path)
    model_paths[model_name] = model_path
    print("Saved Model:", model_path)
best_bundle = joblib.load(model_paths[best_model_name])
prediction_verification = best_bundle["model"].predict(x_test)
model_verification = bool(np.allclose(best_prediction, prediction_verification))
print("\nModels Loaded Successfully!")
print("Model Verification:", model_verification)
# endregion

# region Save Metadata
metadata_folder = BASE_DIR / "Metadata"
metadata_folder.mkdir(parents=True, exist_ok=True)
original_df = stock_df.copy()
data_types = {column: str(dtype) for column, dtype in original_df.dtypes.items()}
missing_values = {column: int(value) for column, value in original_df.isna().sum().items()}
descriptive_statistics = json.loads(original_df.describe().round(6).to_json(date_format="iso"))
metrics_metadata = json.loads(metrics_df.round(8).to_json(orient="records"))
feature_importance_metadata = json.loads(coefficient_df.round(8).to_json(orient="records"))
run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
metadata = {
    "id": DATASET_ID,
    "run_timestamp": run_timestamp,
    "name": DATASET_NAME,
    "level": "Intermediate",
    "domain": "Business/Finance",
    "task": "Regression",
    "description": "Estimate same-day ISE_USD returns from seven international market index returns.",
    "source": "UCI Machine Learning Repository / Data Science Dojo",
    "dataset_statistics": {
        "rows": int(original_df.shape[0]),
        "columns": int(original_df.shape[1]),
        "column_names": list(original_df.columns),
        "data_types": data_types,
        "missing_values": missing_values,
        "total_missing_values": int(original_df.isna().sum().sum()),
        "descriptive_statistics": descriptive_statistics,
        "date_start": original_df["Date"].min().isoformat(),
        "date_end": original_df["Date"].max().isoformat(),
    },
    "feature_selection": {
        "selected_features": FEATURES,
        "target_column": TARGET,
        "excluded_columns": excluded_columns,
    },
    "preprocessing": {
        "numeric_method": "Median Imputation + StandardScaler",
        "categorical_method": None,
        "missing_value_method": "Training-set median for numeric features",
        "split_method": "Chronological, no shuffling",
        "test_size": TEST_SIZE,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
    },
    "model_evaluation": {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "metrics": ["MAE", "MSE", "RMSE", "R2", "R2 Percentage", "Direction Accuracy"],
        "results": metrics_metadata,
        "recommended_model": best_model_name,
        "selection_rule": "Highest test-set Direction Accuracy; lowest RMSE breaks ties (descriptive comparison, not unbiased final selection)",
        "direction_accuracy_definition": "Percent of test observations with correct sign (positive versus nonpositive)",
        "note": "MAPE is omitted because market returns can be zero or near zero.",
        "prediction_horizon": "same-day association, not next-day forecast",
    },
    "feature_importance": {
        "method": "Standardized Linear Regression Coefficients",
        "model": "Linear Regression",
        "results": feature_importance_metadata,
        "most_influential_feature": coefficient_df.iloc[0]["Feature"],
    },
    "model_verification": model_verification,
    "output_files": {
        "original_dataset": str(dataset_path.relative_to(BASE_DIR).as_posix()),
        "processed_dataset": str(processed_path.relative_to(BASE_DIR).as_posix()),
        "metrics": str(metrics_path.relative_to(BASE_DIR).as_posix()),
        "metadata_latest": "Metadata/D22_istanbul_stock_exchange.json",
        "feature_columns": str(feature_columns_path.relative_to(BASE_DIR).as_posix()),
        "models": {name: str(path.relative_to(BASE_DIR)) for name, path in model_paths.items()},
    },
}
metadata_file = metadata_folder / "D22_istanbul_stock_exchange.json"
with open(metadata_file, "w", encoding="utf-8") as file:
    json.dump(metadata, file, ensure_ascii=False, indent=4, allow_nan=False)
print("\nMetadata Saved Successfully!")
print("Saved File:", metadata_file)
with open(metadata_file, "r", encoding="utf-8") as file:
    saved_metadata = json.load(file)
print("-" * 60)
print("Metadata Verification:")
print("Dataset Name:", saved_metadata["name"])
print("Total Rows:", saved_metadata["dataset_statistics"]["rows"])
print("Total Columns:", saved_metadata["dataset_statistics"]["columns"])
print("Target Column:", saved_metadata["feature_selection"]["target_column"])
print("Recommended Model:", saved_metadata["model_evaluation"]["recommended_model"])
print("Model Verification:", saved_metadata["model_verification"])
print("\nMetadata Process Completed Successfully!")
# endregion
