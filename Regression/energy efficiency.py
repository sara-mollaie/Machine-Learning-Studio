import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.patches import Patch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


# ============================================================
# Energy Efficiency (D17) - Single Regression Target
# Target = Heating Load + Cooling Load
# Five models are saved together in one JOBLIB file.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "DataSets" / "ENB2012_data.xlsx"
RANDOM_STATE = 42
TEST_SIZE = 0.20
SHOW_FIGURES = True

DATASET_ID = "D17"
DATASET_NAME = "Energy Efficiency"
TARGET = "Total Energy Load"

FEATURES = [
    "Relative Compactness",
    "Surface Area",
    "Wall Area",
    "Roof Area",
    "Overall Height",
    "Orientation",
    "Glazing Area",
    "Glazing Area Distribution",
]

NUMERIC_FEATURES = [
    "Relative Compactness",
    "Surface Area",
    "Wall Area",
    "Roof Area",
    "Overall Height",
    "Glazing Area",
]
CATEGORICAL_FEATURES = [
    "Orientation",
    "Glazing Area Distribution",
]

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


# -----------------------------
# Figure helpers
# -----------------------------

def style_axes(ax):
    ax.set_facecolor(BG_COLOR)
    ax.set_axisbelow(True)
    ax.grid(color=LAVENDER, alpha=0.30, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(LAVENDER)
        spine.set_linewidth(0.8)


def set_chart_title(ax, title):
    ax.set_title("")
    ax.figure.suptitle(
        title.replace("\n", " "),
        fontsize=16,
        fontweight="bold",
        color=TITLE_ACCENT,
        x=0.5,
        y=0.95,
        ha="center",
    )


def display_figure():
    fig = plt.gcf()
    manager = fig.canvas.manager
    window = getattr(manager, "window", None)

    # Wide window, while respecting the available screen size.
    width_px = 1500
    height_px = 920
    if window is not None and hasattr(window, "winfo_screenwidth"):
        try:
            window.update_idletasks()
            width_px = min(1700, int(window.winfo_screenwidth() * 0.94))
            height_px = min(1000, int(window.winfo_screenheight() * 0.88))
            window.geometry(f"{width_px}x{height_px}")
            window.update_idletasks()
        except Exception:
            pass

    fig.set_size_inches(width_px / fig.dpi, height_px / fig.dpi, forward=True)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.90), pad=1.4)

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def create_preprocessor():
    # Handle both recent and older scikit-learn versions.
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot_encoder", encoder),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


# -----------------------------
# Load and prepare dataset
# -----------------------------

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset file was not found:\n{DATASET_PATH}\n"
        "Make sure ENB2012_data.xlsx is inside the DataSets folder."
    )

source_df = pd.read_excel(DATASET_PATH, engine="openpyxl")
source_df = source_df.dropna(axis=0, how="all").dropna(axis=1, how="all")
source_df.columns = source_df.columns.astype(str).str.strip()

# UCI ENB2012 data uses X1-X8 as features and Y1/Y2 as target loads.
if set([f"X{i}" for i in range(1, 9)] + ["Y1", "Y2"]).issubset(source_df.columns):
    source_df = source_df.rename(columns={
        "X1": "Relative Compactness",
        "X2": "Surface Area",
        "X3": "Wall Area",
        "X4": "Roof Area",
        "X5": "Overall Height",
        "X6": "Orientation",
        "X7": "Glazing Area",
        "X8": "Glazing Area Distribution",
        "Y1": "Heating Load",
        "Y2": "Cooling Load",
    })

required_columns = FEATURES + ["Heating Load", "Cooling Load"]
missing_columns = sorted(set(required_columns) - set(source_df.columns))
if missing_columns:
    raise ValueError(f"Required columns are missing from the dataset: {missing_columns}")

source_df = source_df[required_columns].apply(pd.to_numeric, errors="raise")
source_df = source_df.replace([np.inf, -np.inf], np.nan)

if source_df[["Heating Load", "Cooling Load"]].isna().any().any():
    raise ValueError("Heating Load and Cooling Load must not contain missing values.")

# A single target replaces the two separate target columns.
energy_df = source_df[FEATURES].copy()
energy_df[TARGET] = source_df["Heating Load"] + source_df["Cooling Load"]
energy_df = energy_df.replace([np.inf, -np.inf], np.nan)

if energy_df[TARGET].isna().any():
    raise ValueError("Total Energy Load contains missing values.")
if (energy_df[TARGET] <= 0).any():
    raise ValueError("Total Energy Load must contain positive values.")

print("=" * 70)
print("ENERGY EFFICIENCY - DATASET INFORMATION")
print("=" * 70)
print("Dataset:", DATASET_NAME)
print("Dataset ID:", DATASET_ID)
print("Prepared shape:", energy_df.shape)
print("Features:", FEATURES)
print("Target:", TARGET)
print("\nFirst five rows:")
print(energy_df.head().to_string(index=False))
print("\nData types:")
print(energy_df.dtypes)
print("\nMissing values:")
print(energy_df.isna().sum())
print("\nDuplicate rows:", int(energy_df.duplicated().sum()))
print("\nStatistical description:")
print(energy_df.describe().round(3).to_string())


# -----------------------------
# Train/test split
# -----------------------------

if energy_df[FEATURES].isna().all().any():
    raise ValueError("At least one feature column contains only missing values.")

X = energy_df[FEATURES].copy()
y = energy_df[TARGET].copy()

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

print("\nTrain rows:", len(x_train))
print("Test rows:", len(x_test))
print("Numeric features:", NUMERIC_FEATURES)
print("Categorical features:", CATEGORICAL_FEATURES)


# -----------------------------
# Five regression models
# -----------------------------

model_estimators = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    ),
    "SVR": SVR(kernel="rbf", C=100.0, epsilon=0.1),
}

trained_models = {}
predictions = {}
metric_records = []

# Same test-target range is used to normalize RMSE for all models.
target_range = float(y_test.max() - y_test.min())
if target_range <= 0:
    raise ValueError("The test target range must be greater than zero.")

for model_name, estimator in model_estimators.items():
    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", estimator),
    ])

    pipeline.fit(x_train, y_train)
    prediction = pipeline.predict(x_test)

    trained_models[model_name] = pipeline
    predictions[model_name] = prediction

    mae = float(mean_absolute_error(y_test, prediction))
    mse = float(mean_squared_error(y_test, prediction))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, prediction))

    # RMSE-based normalized score:
    # 100 * (1 - RMSE / test_target_range), clipped to [0, 100].
    # This is a project score, not a standard statistical accuracy metric.
    rmse_accuracy = float(np.clip(
        100.0 * (1.0 - rmse / target_range),
        0.0,
        100.0,
    ))

    metric_records.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "R2 Percentage": r2 * 100.0,
        "Accuracy": rmse_accuracy,
    })

metrics_df = pd.DataFrame(metric_records)
metrics_df = metrics_df.sort_values(
    "RMSE",
    ascending=True,
    kind="stable",
).reset_index(drop=True)

# Best model is the model with the lowest RMSE.
best_model_name = str(metrics_df.iloc[0]["Model"])
best_prediction = predictions[best_model_name]
best_metrics = metrics_df.iloc[0].to_dict()

print("\n" + "=" * 70)
print("REGRESSION MODEL COMPARISON (SORTED BY LOWEST RMSE)")
print("=" * 70)
print(metrics_df.round(4).to_string(index=False))
print("\nSelected model:", best_model_name)
print("Selection rule: Lowest RMSE on the held-out test set")
print(f"RMSE-based score: {best_metrics['Accuracy']:.2f}%")


# -----------------------------
# Feature importance
# -----------------------------

linear_pipeline = trained_models["Linear Regression"]
preprocessor = linear_pipeline.named_steps["preprocessor"]
linear_estimator = linear_pipeline.named_steps["model"]
transformed_feature_names = preprocessor.get_feature_names_out()

coefficient_df = pd.DataFrame({
    "Feature": [
        name.split("__", 1)[-1] for name in transformed_feature_names
    ],
    "Coefficient": linear_estimator.coef_,
})
coefficient_df["Absolute Coefficient"] = coefficient_df["Coefficient"].abs()
coefficient_df = coefficient_df.sort_values(
    "Absolute Coefficient",
    ascending=False,
    kind="stable",
).reset_index(drop=True)

positive_coefficients = coefficient_df[
    coefficient_df["Coefficient"] >= 0
].sort_values("Coefficient", ascending=False)

negative_coefficients = coefficient_df[
    coefficient_df["Coefficient"] < 0
].sort_values("Coefficient", ascending=True)

top_coefficients = pd.concat(
    [positive_coefficients, negative_coefficients],
    ignore_index=True,
).head(10)

# Horizontal bars are drawn from bottom to top in the desired order:
# largest positive coefficients first, then negative coefficients.
plot_coefficients = top_coefficients.iloc[::-1]

fig, ax = plt.subplots(figsize=(15, 8), facecolor=BG_COLOR)
style_axes(ax)
set_chart_title(
    ax,
    "Energy Efficiency - Most Influential Building Features",
)
bar_colors = [
    PURPLE if value >= 0 else LIGHT_PINK
    for value in plot_coefficients["Coefficient"]
]
bars = ax.barh(
    plot_coefficients["Feature"],
    plot_coefficients["Absolute Coefficient"],
    color=bar_colors,
    edgecolor="white",
)

max_coefficient = float(plot_coefficients["Absolute Coefficient"].max())
for bar, coefficient in zip(bars, plot_coefficients["Coefficient"]):
    value = abs(float(coefficient))
    ax.text(
        value + max(max_coefficient * 0.012, 0.02),
        bar.get_y() + bar.get_height() / 2,
        f"{coefficient:+.2f}",
        ha="left",
        va="center",
        fontsize=10,
        color=TITLE_COLOR,
        fontweight="bold",
    )

ax.set_xlim(0, max_coefficient * 1.20 if max_coefficient > 0 else 1)
ax.set_xlabel("Absolute Coefficient (Importance)", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Feature", fontsize=12, color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
legend = ax.legend(
    handles=[
        Patch(facecolor=PURPLE, label="Positive coefficient"),
        Patch(facecolor=LIGHT_PINK, label="Negative coefficient"),
    ],
    loc="best",
    facecolor=BG_COLOR,
    edgecolor=LAVENDER,
)
for legend_text in legend.get_texts():
    legend_text.set_color(TEXT_COLOR)
display_figure()


# -----------------------------
# Correlation heatmap
# -----------------------------

correlation_matrix = energy_df[FEATURES + [TARGET]].corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(13, 8), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap=sns.blend_palette(
        ["#FFF7FB", "#F6CFE3", "#D69ACD", "#9562AF", "#51267D"],
        as_cmap=True,
    ),
    vmin=-1,
    vmax=1,
    center=0,
    linewidths=0.8,
    linecolor=BG_COLOR,
    cbar_kws={"label": "Correlation"},
    annot_kws={"fontsize": 9, "fontweight": "bold"},
    ax=ax,
)
set_chart_title(ax, "Energy Efficiency - Correlation Heatmap")
ax.tick_params(axis="x", rotation=40, colors=TEXT_COLOR)
plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
ax.tick_params(axis="y", rotation=0, colors=TEXT_COLOR)
ax.set_xlabel("Variables", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Variables", fontsize=12, color=TEXT_COLOR)
display_figure()


# -----------------------------
# Actual vs predicted
# -----------------------------

fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG_COLOR)
style_axes(ax)
set_chart_title(
    ax,
    f"Energy Efficiency - Actual vs Predicted Total Energy Load ({best_model_name})",
)
ax.scatter(
    y_test,
    best_prediction,
    color=TEAL,
    alpha=0.80,
    edgecolors=BG_COLOR,
    linewidths=0.7,
)
minimum = min(float(y_test.min()), float(np.min(best_prediction)))
maximum = max(float(y_test.max()), float(np.max(best_prediction)))
padding = (maximum - minimum) * 0.05
ax.plot(
    [minimum - padding, maximum + padding],
    [minimum - padding, maximum + padding],
    color=PINK,
    linewidth=2,
    label="Ideal prediction",
)
ax.set_xlabel("Actual Total Energy Load", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Predicted Total Energy Load", fontsize=12, color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
legend = ax.legend(facecolor=BG_COLOR, edgecolor=LAVENDER)
for legend_text in legend.get_texts():
    legend_text.set_color(TEXT_COLOR)
display_figure()


# -----------------------------
# RMSE-based model accuracy chart
# Same visual style as the Abalone chart.
# Highest score is shown first.
# -----------------------------

accuracy_plot = metrics_df.sort_values(
    "Accuracy",
    ascending=False,
    kind="stable",
).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG_COLOR)
style_axes(ax)
set_chart_title(
    ax,
    "Energy Efficiency - Regression Model Accuracy (RMSE-based)",
)
accuracy_bars = ax.bar(
    accuracy_plot["Model"],
    accuracy_plot["Accuracy"],
    color=MODEL_COLORS[:len(accuracy_plot)],
    edgecolor="white",
    linewidth=0.8,
)

for bar in accuracy_bars:
    value = float(bar.get_height())
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1.0,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        color=TITLE_COLOR,
        fontweight="bold",
    )

ax.set_xlabel("Model", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Accuracy (%)", fontsize=12, color=TEXT_COLOR)
ax.set_ylim(0, min(110, max(100, float(accuracy_plot["Accuracy"].max()) + 10)))
ax.set_yticks(np.arange(0, 101, 20))
ax.tick_params(axis="x", rotation=20, colors=TEXT_COLOR)
ax.tick_params(axis="y", colors=TEXT_COLOR)
ax.margins(x=0.08)
display_figure()


# -----------------------------
# Residual distribution
# -----------------------------

residuals = y_test.to_numpy() - best_prediction

fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG_COLOR)
style_axes(ax)
set_chart_title(
    ax,
    f"Energy Efficiency - Residual Distribution ({best_model_name})",
)
ax.hist(
    residuals,
    bins=25,
    color=PURPLE,
    alpha=0.85,
    edgecolor="white",
)
ax.axvline(0, color=PINK, linewidth=2)
ax.set_xlabel("Residual (Actual - Predicted)", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Frequency", fontsize=12, color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
display_figure()


# -----------------------------
# Save processed data and metrics
# -----------------------------

processed_folder = BASE_DIR / "DataSets" / "Processed"
processed_folder.mkdir(parents=True, exist_ok=True)

processed_df = energy_df.copy()
processed_df["Predicted_Total_Energy_Load"] = np.nan
processed_df.loc[y_test.index, "Predicted_Total_Energy_Load"] = best_prediction

processed_path = processed_folder / "energy_efficiency_regression_results.csv"
metrics_path = processed_folder / "energy_efficiency_metrics.csv"
feature_importance_path = processed_folder / "energy_efficiency_feature_importance.csv"

processed_df.to_csv(processed_path, index=False)
metrics_df.to_csv(metrics_path, index=False)
coefficient_df.to_csv(feature_importance_path, index=False)


# -----------------------------
# Save all five models together
# -----------------------------

models_folder = BASE_DIR / "ModelsOutcome"
models_folder.mkdir(parents=True, exist_ok=True)

models_bundle_path = models_folder / "energy_efficiency_regression_models.joblib"
feature_columns_path = models_folder / "energy_efficiency_feature_columns.joblib"

models_bundle = {
    "dataset_id": DATASET_ID,
    "dataset_name": DATASET_NAME,
    "target_column": TARGET,
    "feature_columns": FEATURES,
    "models": trained_models,
    "model_metrics": metrics_df.to_dict(orient="records"),
    "best_model_name": best_model_name,
    "best_model": trained_models[best_model_name],
    "accuracy_definition": (
        "RMSE-based normalized score = max(0, 100 * (1 - RMSE / test_target_range)); "
        "this is a project score, not a standard regression accuracy metric."
    ),
    "test_target_range": target_range,
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
}

joblib.dump(models_bundle, models_bundle_path)
joblib.dump(FEATURES, feature_columns_path)

# Verify that the saved best model can be loaded and reproduces predictions.
loaded_bundle = joblib.load(models_bundle_path)
loaded_best_prediction = loaded_bundle["best_model"].predict(x_test)
model_verification = bool(
    np.allclose(best_prediction, loaded_best_prediction, rtol=1e-7, atol=1e-7)
)

print("\n" + "=" * 70)
print("MODEL SAVING")
print("=" * 70)
print("All five models saved in:", models_bundle_path)
print("Feature columns saved in:", feature_columns_path)
print("Model verification:", model_verification)
print("Models in bundle:", list(loaded_bundle["models"].keys()))


# -----------------------------
# Save one D17 metadata file
# No separate Heating Load / Cooling Load dataset cards.
# -----------------------------

metadata_folder = BASE_DIR / "Metadata"
metadata_folder.mkdir(parents=True, exist_ok=True)

data_types = {column: str(dtype) for column, dtype in energy_df.dtypes.items()}
missing_values = {
    column: int(value)
    for column, value in energy_df.isna().sum().items()
}
descriptive_statistics = json.loads(
    energy_df.describe().round(4).to_json()
)
metrics_metadata = json.loads(
    metrics_df.round(6).to_json(orient="records")
)
feature_importance_metadata = json.loads(
    coefficient_df.round(6).to_json(orient="records")
)

metadata = {
    "id": DATASET_ID,
    "name": DATASET_NAME,
    "title": "Energy Efficiency (D17)",
    "level": "Intermediate",
    "domain": "Energy",
    "task": "Regression",
    "description": (
        "Predict the combined building energy load using building design "
        "features. Total Energy Load is calculated as Heating Load plus "
        "Cooling Load. The dataset is presented as one Energy Efficiency "
        "regression task."
    ),
    "source": "UCI Machine Learning Repository",
    "dataset_statistics": {
        "rows": int(energy_df.shape[0]),
        "columns": int(energy_df.shape[1]),
        "column_names": list(energy_df.columns),
        "data_types": data_types,
        "missing_values": missing_values,
        "total_missing_values": int(energy_df.isna().sum().sum()),
        "duplicate_rows": int(energy_df.duplicated().sum()),
        "descriptive_statistics": descriptive_statistics,
    },
    "feature_selection": {
        "selected_features": FEATURES,
        "target_column": TARGET,
        "target_formula": "Heating Load + Cooling Load",
        "source_target_columns_combined": [
            "Heating Load",
            "Cooling Load",
        ],
    },
    "preprocessing": {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_method": "Median Imputation + StandardScaler",
        "categorical_method": "Most Frequent Imputation + OneHotEncoder",
    },
    "model_evaluation": {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "metrics": [
            "MAE",
            "MSE",
            "RMSE",
            "R2",
            "R2 Percentage",
            "Accuracy",
        ],
        "accuracy_definition": (
            "RMSE-based normalized score = max(0, 100 * "
            "(1 - RMSE / test_target_range)). This is a project score, "
            "not a standard regression accuracy metric."
        ),
        "test_target_range": target_range,
        "results": metrics_metadata,
        "recommended_model": best_model_name,
        "selection_rule": "Lowest RMSE on the held-out test set",
    },
    "feature_importance": {
        "method": "Linear Regression Coefficients",
        "model": "Linear Regression",
        "results": feature_importance_metadata,
        "most_influential_feature": str(
            coefficient_df.iloc[0]["Feature"]
        ),
    },
    "model_verification": model_verification,
    "ui_card": {
        "title": "Energy Efficiency (D17)",
        "category": "Energy",
        "task": "Regression",
        "level": "Intermediate",
        "rows": int(energy_df.shape[0]),
        "columns": int(energy_df.shape[1]),
    },
    "output_files": {
        "original_dataset": "DataSets/ENB2012_data.xlsx",
        "processed_dataset": processed_path.relative_to(BASE_DIR).as_posix(),
        "metrics": metrics_path.relative_to(BASE_DIR).as_posix(),
        "feature_importance": feature_importance_path.relative_to(BASE_DIR).as_posix(),
        "models_bundle": models_bundle_path.relative_to(BASE_DIR).as_posix(),
        "feature_columns": feature_columns_path.relative_to(BASE_DIR).as_posix(),
        "metadata": "Metadata/D17_energy_efficiency.json",
    },
    "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

metadata_path = metadata_folder / "D17_energy_efficiency.json"
with metadata_path.open("w", encoding="utf-8") as file:
    json.dump(metadata, file, ensure_ascii=False, indent=4, allow_nan=False)

# Read-back verification for the metadata artifact.

# Remove the two legacy target-specific metadata files so the application
# does not discover separate Heating Load and Cooling Load cards.
for old_metadata_name in (
    "D17_energy_heating_load.json",
    "D17_energy_cooling_load.json",
):
    old_metadata_path = metadata_folder / old_metadata_name
    if old_metadata_path.exists():
        old_metadata_path.unlink()

with metadata_path.open("r", encoding="utf-8") as file:
    saved_metadata = json.load(file)

print("\n" + "=" * 70)
print("ENERGY EFFICIENCY - COMPLETED")
print("=" * 70)
print("Dataset card:", saved_metadata["ui_card"]["title"])
print("Rows:", saved_metadata["dataset_statistics"]["rows"])
print("Columns:", saved_metadata["dataset_statistics"]["columns"])
print("Target:", saved_metadata["feature_selection"]["target_column"])
print("Best model:", saved_metadata["model_evaluation"]["recommended_model"])
print("Best RMSE:", round(float(best_metrics["RMSE"]), 4))
print("RMSE-based score:", round(float(best_metrics["Accuracy"]), 2), "%")
print("Metadata:", metadata_path)
print("Model bundle:", models_bundle_path)
print("Model verification:", model_verification)
print("\nOnly one dataset entry is generated: Energy Efficiency (D17).")
print("Heating Load and Cooling Load are combined into Total Energy Load.")
