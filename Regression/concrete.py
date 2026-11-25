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
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


BASE_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
TEST_SIZE = 0.2
SHOW_FIGURES = True

FEATURES = ["Cement", "Blast Furnace Slag", "Fly Ash", "Water",
            "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age"]
TARGET = "Concrete compressive strength"

# region Chart Colors

BG_COLOR = "#FFFFFF"
TITLE_COLOR = "#2B1450"
TITLE_ACCENT = "#C9578F"
TEXT_COLOR = "#665B78"
PURPLE = "#8B5CC7"
PINK = "#E3A0C4"
TEAL = "#21B8A8"
LAVENDER = "#C9B6F2"
LIGHT_BLUE = "#8EC5FC"
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
        x = (window.winfo_screenwidth() - width) // 2
        y = max(40, (window.winfo_screenheight() - height) // 2 - 40)
        window.geometry(f"+{x}+{y}")
        window.update_idletasks()
        window.update()
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
    ax.grid(color=LAVENDER, alpha=0.3, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(LAVENDER)
        spine.set_linewidth(0.8)


def set_chart_title(ax, title):
    ax.set_title(
        title,
        fontsize=18,
        fontweight="bold",
        color=TITLE_ACCENT,
        pad=24,
    )


def display_figure():
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    show_figure()

# endregion


# region Load Dataset

dataset_path = BASE_DIR / "DataSets" / "Concrete_Data.xls"
concrete_df = pd.read_excel(dataset_path)
# Accept original UCI headers with units as well as standardized names.
rename = {}
for column in concrete_df.columns:
    text = " ".join(str(column).lower().split())
    for name in FEATURES + [TARGET]:
        prefix = name.lower()
        if text == prefix or text.startswith(prefix + " ") or text.startswith(prefix + "("):
            rename[column] = name
            break
concrete_df = concrete_df.rename(columns=rename)
missing = set(FEATURES + [TARGET]) - set(concrete_df.columns)
if missing or concrete_df.columns.duplicated().any():
    raise ValueError(f"Invalid dataset headers. Missing columns: {sorted(missing)}")
concrete_df = concrete_df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="raise")
concrete_df = concrete_df.replace([np.inf, -np.inf], np.nan)
print("Dataset Path:", dataset_path)

# endregion


# region Data Understanding

print("\nFirst 5 Rows:")
print(concrete_df.head().to_string())
print("-" * 60)
print("Dataset Shape:")
print(concrete_df.shape)
print("-" * 60)
print("Columns:")
print(concrete_df.columns.tolist())
print("-" * 60)
print("Data Types:")
print(concrete_df.dtypes)
print("-" * 60)
print("\nDataset Information:")
concrete_df.info()
print("-" * 60)
print("Statistical Description:")
print(concrete_df.describe().to_string())
print("-" * 60)
print("Missing Values:")
print(concrete_df.isnull().sum())
print("-" * 60)
print("Total Missing Values:")
print(concrete_df.isnull().sum().sum())
print("-" * 60)
print("Duplicate Rows:")
print(concrete_df.duplicated().sum())
print("-" * 60)


# endregion


# region Feature Selection

X = concrete_df[FEATURES].copy()
y = concrete_df[TARGET].copy()

print("\nSelected Features:")
print(X.head().to_string())
print("-" * 60)
print("Features Shape:")
print(X.shape)
print("-" * 60)
print("Feature Names:")
print(FEATURES)
print("-" * 60)
print("Target Variable:")
print(TARGET)
print("-" * 60)
print("Excluded Columns:")
print([])
print("-" * 60)

# endregion


# region Data Preprocessing

if y.isna().any() or (y <= 0).any():
    raise ValueError("Concrete strength must be present and positive.")
if (X < 0).any().any() or (X["Age"].dropna() <= 0).any():
    raise ValueError("Ingredients must be nonnegative and age must be positive.")
if X.isna().all().any():
    raise ValueError("An input column contains only missing values.")
print("Missing values in model inputs:")
print(X.isna().sum())
print("Numeric missing values are filled using the training-set median.")

# endregion


# region Data Normalization

numeric_features = FEATURES

def create_preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
    ])

print("Normalization: training-set median imputation + StandardScaler")

# endregion


# region Model Evaluation

# Identical feature rows remain in the same split.
groups = pd.MultiIndex.from_frame(X).factorize()[0]
splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_indices, test_indices = next(splitter.split(X, y, groups))
x_train, x_test = X.iloc[train_indices], X.iloc[test_indices]
y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

models = {
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
    rmse = float(np.sqrt(mse))
    target_range = float(y_test.max() - y_test.min())
    if target_range <= 0:
        raise ValueError("The target range in the test set must be greater than zero.")

    # Same RMSE-based score used by the Abalone project.
    accuracy = max(0.0, (1.0 - rmse / target_range) * 100.0)
    records.append({
        "Model": model_name,
        "MAE": mean_absolute_error(y_test, prediction),
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2_score(y_test, prediction),
        "R2 Percentage": r2_score(y_test, prediction) * 100,
        "Accuracy": accuracy,
    })

# Lowest RMSE determines the recommended regression model.
metrics_df = pd.DataFrame(records).sort_values(
    "RMSE", ascending=True, kind="stable"
).reset_index(drop=True)
best_model_name = str(metrics_df.iloc[0]["Model"])
best_prediction = predictions[best_model_name]

print("\nRegression Metrics:")
print(metrics_df.round(4).to_string(index=False))
print("-" * 60)
print("Recommended Model:", best_model_name)
print("Selection Rule: Lowest RMSE on the held-out test set")
print("Accuracy Formula: max(0, (1 - RMSE / target range) * 100)")

# endregion


# region Feature Importance

# Linear Regression exposes coefficients directly. The preprocessing pipeline
# provides the names of the transformed numeric and one-hot encoded features.
coefficient_model = trained_models["Linear Regression"]
coefficient_preprocessor = coefficient_model.named_steps["preprocessor"]
linear_estimator = coefficient_model.named_steps["model"]
transformed_features = coefficient_preprocessor.get_feature_names_out()

coefficient_df = pd.DataFrame({
    "Feature": [name.split("__", 1)[-1] for name in transformed_features],
    "Coefficient": linear_estimator.coef_,
})
coefficient_df["Absolute Coefficient"] = coefficient_df["Coefficient"].abs()
coefficient_df = coefficient_df.sort_values(
    "Absolute Coefficient",
    ascending=False,
).reset_index(drop=True)
coefficient_df["Importance Rank"] = np.arange(1, len(coefficient_df) + 1)

print("\nFeature Importance Based On Linear Regression Coefficients:")
print(coefficient_df.round(4).to_string(index=False))
print("Most Influential Feature:", coefficient_df.iloc[0]["Feature"])

# Display positive coefficients first, followed by negative coefficients.
positive_coefficients = coefficient_df[coefficient_df["Coefficient"] >= 0].sort_values(
    "Coefficient",
    ascending=False,
)
negative_coefficients = coefficient_df[coefficient_df["Coefficient"] < 0].sort_values(
    "Coefficient",
    ascending=False,
)
top_coefficients = pd.concat([positive_coefficients, negative_coefficients]).head(10)
plot_coefficients = top_coefficients.iloc[::-1]
plt.figure(figsize=(12, 7), facecolor=BG_COLOR)
ax = plt.gca()
style_axes(ax)
set_chart_title(ax, "Concrete - Most Influential Ingredients and Age")
bar_colors = [
    LIGHT_PINK if value < 0 else PURPLE
    for value in plot_coefficients["Coefficient"]
]
bars = ax.barh(
    plot_coefficients["Feature"],
    plot_coefficients["Absolute Coefficient"],
    color=bar_colors,
    edgecolor="white",
)
for bar, value, coefficient in zip(
    bars,
    plot_coefficients["Absolute Coefficient"],
    plot_coefficients["Coefficient"],
):
    ax.text(
        value + 0.06,
        bar.get_y() + bar.get_height() / 2,
        f"{'-' if coefficient < 0 else '+'}{value:.2f}",
        ha="left",
        va="center",
        fontsize=9,
        color=TITLE_COLOR,
        fontweight="bold",
    )
ax.set_xlim(0, coefficient_df["Absolute Coefficient"].max() * 1.2)
ax.set_xlabel("Absolute Coefficient (Importance)", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Feature", fontsize=12, color=TEXT_COLOR)
ax.tick_params(axis="x", colors=TEXT_COLOR)
ax.tick_params(axis="y", colors=TEXT_COLOR)
legend = ax.legend(
    handles=[
        Patch(facecolor=PURPLE, label="Positive coefficient"),
        Patch(facecolor=LIGHT_PINK, label="Negative coefficient"),
    ],
    loc="center right",
    borderaxespad=0,
    facecolor=BG_COLOR,
    edgecolor=LAVENDER,
)
for text in legend.get_texts():
    text.set_color(TEXT_COLOR)
display_figure()

# endregion


# region Results

best_metrics = metrics_df.iloc[0].to_dict()
comparison_df = metrics_df.copy()
comparison_df["RMSE Rank"] = comparison_df["RMSE"].rank(method="min").astype(int)
comparison_df["R2 Rank"] = comparison_df["R2"].rank(ascending=False, method="min").astype(int)

print("\nModel Comparison:")
print(comparison_df.round(4).to_string(index=False))
print("-" * 60)
print("Best Model Metrics:")
for metric in ["MAE", "MSE", "RMSE", "R2", "R2 Percentage", "Accuracy"]:
    print(f"{metric}: {best_metrics[metric]:.4f}")

# endregion


# region Correlation Heatmap

correlation_columns = FEATURES + [TARGET]
correlation_matrix = concrete_df[correlation_columns].corr()

plt.figure(figsize=(10.2, 6.2), dpi=100, facecolor=BG_COLOR)
ax = plt.gca()
ax.set_facecolor(BG_COLOR)
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap=sns.blend_palette(["#FFF7FB", "#F6CFE3", "#D69ACD", "#9562AF", "#51267D"], as_cmap=True),
    vmin=-1,
    vmax=1,
    center=0,
    linewidths=0.8,
    linecolor=BG_COLOR,
    cbar_kws={"label": "Correlation"},
    annot_kws={"fontsize": 9, "fontweight": "bold"},
    ax=ax,
)
set_chart_title(ax, "Concrete - Correlation Heatmap")
display_labels = {
    "Blast Furnace Slag": "Blast Furnace\nSlag",
    "Coarse Aggregate": "Coarse\nAggregate",
    "Fine Aggregate": "Fine\nAggregate",
    TARGET: "Compressive\nstrength",
}
labels = [display_labels.get(column, column) for column in correlation_columns]
ax.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor")
ax.set_yticklabels(labels, rotation=0)
ax.tick_params(axis="x", labelsize=9, colors=TEXT_COLOR)
ax.tick_params(axis="y", rotation=0, colors=TEXT_COLOR)
plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
ax.set_xlabel("Variables", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Variables", fontsize=12, color=TEXT_COLOR)
display_figure()

# endregion


# region Data Visualization

plt.figure(figsize=(11, 7), facecolor=BG_COLOR)
ax = plt.gca()
style_axes(ax)
set_chart_title(ax, f"Concrete - Actual vs Predicted Strength (MPa) ({best_model_name})")
ax.scatter(y_test, best_prediction, color=TEAL, alpha=0.78, edgecolors=BG_COLOR, linewidths=0.7)
ax.margins(x=0.08, y=0.08)
minimum = min(y_test.min(), best_prediction.min())
maximum = max(y_test.max(), best_prediction.max())
ax.plot([minimum, maximum], [minimum, maximum], color=PINK, linewidth=2, label="Ideal prediction")
ax.set_xlabel("Actual Strength (MPa)", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Predicted Strength (MPa)", fontsize=12, color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
legend = ax.legend(facecolor=BG_COLOR, edgecolor=LAVENDER)
for text in legend.get_texts():
    text.set_color(TEXT_COLOR)
display_figure()

plt.figure(figsize=(12, 7), facecolor=BG_COLOR)
ax = plt.gca()
style_axes(ax)
set_chart_title(ax, "Concrete - Regression Model Accuracy (RMSE-based)")
accuracy_plot = metrics_df.sort_values("RMSE", ascending=True, kind="stable")
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
        fontsize=10,
        color=TITLE_COLOR,
        fontweight="bold",
    )
ax.set_xlabel("Model", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Accuracy (%)", fontsize=12, color=TEXT_COLOR)
ax.set_ylim(0, max(100, float(accuracy_plot["Accuracy"].max()) + 6))
ax.margins(x=0.08)
ax.tick_params(axis="x", rotation=20, colors=TEXT_COLOR)
ax.tick_params(axis="y", colors=TEXT_COLOR)
display_figure()

plt.figure(figsize=(11, 7), facecolor=BG_COLOR)
ax = plt.gca()
style_axes(ax)
set_chart_title(ax, f"Concrete - Residual Distribution ({best_model_name})")
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
processed_df = concrete_df.copy()
processed_df["Predicted_Strength_MPa"] = np.nan
processed_df.loc[y_test.index, "Predicted_Strength_MPa"] = best_prediction
processed_path = output_folder / "concrete_regression_results.csv"
processed_df.to_csv(processed_path, index=False)

metrics_path = output_folder / "concrete_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

print("\nRegression Results Saved Successfully!")
print("Processed Dataset:", processed_path)
print("Metrics:", metrics_path)

# endregion


# region Training Models

model_folder = BASE_DIR / "ModelsOutcome"
model_folder.mkdir(parents=True, exist_ok=True)

# Store all five trained pipelines in one bundle for the future Predict interface.
model_bundle_path = model_folder / "concrete_regression_models.joblib"
metrics_records = metrics_df.to_dict(orient="records")

model_verification_by_name = {}
for model_name, model_pipeline in trained_models.items():
    reloaded_prediction = model_pipeline.predict(x_test)
    model_verification_by_name[model_name] = bool(
        np.allclose(predictions[model_name], reloaded_prediction)
    )

model_verification = all(model_verification_by_name.values())

model_bundle = {
    "dataset_id": "D14",
    "dataset_name": "Concrete",
    "models": trained_models,
    "model_names": list(trained_models.keys()),
    "best_model_name": best_model_name,
    "best_model": trained_models[best_model_name],
    "target_column": TARGET,
    "feature_columns": FEATURES,
    "metrics": metrics_records,
    "accuracy_definition": "max(0, (1 - RMSE / (max(y_test) - min(y_test))) * 100)",
    "model_verification": model_verification_by_name,
    "recommended_model_selection": "Lowest RMSE on the held-out test set",
}

joblib.dump(model_bundle, model_bundle_path, compress=3)

# Reload the saved bundle and verify every model from the actual saved file.
loaded_bundle = joblib.load(model_bundle_path)
saved_model_verification = {}
for model_name in loaded_bundle["model_names"]:
    saved_prediction = loaded_bundle["models"][model_name].predict(x_test)
    saved_model_verification[model_name] = bool(
        np.allclose(predictions[model_name], saved_prediction)
    )

model_verification = all(saved_model_verification.values())
loaded_bundle["model_verification"] = saved_model_verification
joblib.dump(loaded_bundle, model_bundle_path, compress=3)

print("\nAll five regression models saved in one bundle:")
print("Saved Model Bundle:", model_bundle_path)
print("Models:", ", ".join(loaded_bundle["model_names"]))
print("Best Model:", loaded_bundle["best_model_name"])
print("Model Verification:", model_verification)
for model_name, verified in saved_model_verification.items():
    print(f"{model_name} Verification: {verified}")

# endregion


# region Save Metadata

metadata_folder = BASE_DIR / "Metadata"
metadata_folder.mkdir(parents=True, exist_ok=True)

original_df = concrete_df.copy()
data_types = {column: str(dtype) for column, dtype in original_df.dtypes.items()}
missing_values = {column: int(value) for column, value in original_df.isna().sum().items()}
descriptive_statistics = json.loads(original_df.describe().round(2).to_json())
metrics_metadata = json.loads(metrics_df.round(4).to_json(orient="records"))
for result in metrics_metadata:
    result["Accuracy"] = round(result["Accuracy"], 2)
feature_importance_metadata = json.loads(coefficient_df.round(4).to_json(orient="records"))
run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

metadata = {
    "id": "D14",
    "run_timestamp": run_timestamp,
    "name": "Concrete",
    "level": "Intermediate",
    "domain": "Civil Engineering/Construction",
    "task": "Regression",
    "description": "Estimate concrete compressive strength in MPa from mixture ingredients and age.",
    "source": "UCI Machine Learning Repository",
    "dataset_statistics": {
        "rows": int(original_df.shape[0]),
        "columns": int(original_df.shape[1]),
        "column_names": list(original_df.columns),
        "data_types": data_types,
        "missing_values": missing_values,
        "total_missing_values": int(original_df.isna().sum().sum()),
        "descriptive_statistics": descriptive_statistics,
    },
    "feature_selection": {
        "selected_features": FEATURES,
        "target_column": TARGET,
        "excluded_columns": [],
    },
    "preprocessing": {
        "numeric_method": "Median Imputation + StandardScaler",
        "missing_value_method": "Training-set median for numeric features",
    },
    "model_evaluation": {
        "test_size": TEST_SIZE,
        "split_method": "Grouped random split by identical feature rows",
        "training_rows": len(y_train),
        "test_rows": len(y_test),
        "random_state": RANDOM_STATE,
        "metrics": ["MAE", "MSE", "RMSE", "R2", "R2 Percentage", "Accuracy"],
        "results": metrics_metadata,
        "recommended_model": best_model_name,
        "selection_rule": "Lowest RMSE on the held-out test set",
        "accuracy_definition": "RMSE-based score: max(0, (1 - RMSE / (max(y_test) - min(y_test))) * 100); this is a project comparison score, not a standard regression accuracy metric",
    },
    "feature_importance": {
        "method": "Linear Regression Coefficients",
        "model": "Linear Regression",
        "results": feature_importance_metadata,
        "most_influential_feature": coefficient_df.iloc[0]["Feature"],
    },
    "model_verification": model_verification,
    "model_verification_by_name": saved_model_verification,
    "output_files": {
        "original_dataset": "DataSets/Concrete_Data.xls",
        "processed_dataset": "DataSets/Processed/concrete_regression_results.csv",
        "metrics": "DataSets/Processed/concrete_metrics.csv",
        "metadata_latest": "Metadata/D14_concrete.json",
        "model_bundle": "ModelsOutcome/concrete_regression_models.joblib",
        "models": "All five trained regression pipelines are stored in the consolidated model bundle.",
    },
}

metadata_file = metadata_folder / "D14_concrete.json"
with open(metadata_file, "w", encoding="utf-8") as file:
    json.dump(metadata, file, ensure_ascii=False, indent=4)

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
print("Model Bundle:", saved_metadata["output_files"]["model_bundle"])
print("\nMetadata Process Completed Successfully!")

# endregion
