import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.base import clone


BASE_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
SHOW_FIGURES = True
TEST_DAYS = 12
HISTORY_WINDOW = 5
CV_SPLITS = 4

TARGET = "Target (Total orders)"
CALENDAR = ["Week of month", "Day of week"]
HISTORY = ["Previous total", "Mean of previous 5 observations"]
FEATURES = CALENDAR + HISTORY


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


# region Figure Window and Styling

def center_figure_window():
    manager = plt.get_current_fig_manager()
    window = getattr(manager, "window", None)

    if window is None:
        return

    try:
        window.update_idletasks()
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        width = min(1500, screen_width - 80)
        height = min(950, screen_height - 100)

        x = max(0, (screen_width - width) // 2)
        y = max(20, (screen_height - height) // 2 - 20)

        window.geometry(f"{width}x{height}+{x}+{y}")
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
        fontsize=19,
        fontweight="bold",
        color=TITLE_ACCENT,
        pad=24,
    )


def display_figure():
    plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))
    show_figure()

# endregion


# region Load Dataset

data_path = BASE_DIR / "DataSets" / "Daily_Demand_Forecasting_Orders.csv"

if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}")

raw = pd.read_csv(data_path, sep=";")
raw.columns = raw.columns.astype(str).str.strip()

calendar_names = {}

for column in raw.columns:
    if column.startswith("Week of the month"):
        calendar_names[column] = CALENDAR[0]
    elif column.startswith("Day of the week"):
        calendar_names[column] = CALENDAR[1]

df = raw.rename(columns=calendar_names).copy()

required_columns = CALENDAR + [TARGET]
missing_columns = [column for column in required_columns if column not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required dataset columns: {missing_columns}")

df[required_columns] = df[required_columns].apply(pd.to_numeric, errors="raise")

if not np.isfinite(df[required_columns].to_numpy(dtype=float)).all():
    raise ValueError("Calendar columns and total orders must contain finite values.")

# endregion


# region Data Understanding

print("\nFirst 5 Rows:")
print(raw.head().to_string(index=False))
print("-" * 60)

print("Dataset Shape:", raw.shape)
print("Columns:", raw.columns.tolist())

print("\nData Types:")
print(raw.dtypes)

print("\nDataset Information:")
raw.info()

print("\nStatistical Description:")
print(raw.describe().round(3).to_string())

print("\nMissing values:\n", raw.isna().sum())
print("Duplicate rows:", raw.duplicated().sum())

order_type_columns = ["Order type A", "Order type B", "Order type C"]
if set(order_type_columns).issubset(raw.columns):
    type_sum = raw[order_type_columns].sum(axis=1)
    print(
        "Same-day A + B + C equals target:",
        np.allclose(type_sum.to_numpy(), df[TARGET].to_numpy()),
    )

# endregion


# region Feature Selection

# Use only information available before the forecasted working day.
previous = df[TARGET].shift(1)

df[HISTORY[0]] = previous
df[HISTORY[1]] = previous.rolling(HISTORY_WINDOW).mean()

samples = df.dropna(subset=FEATURES + [TARGET]).copy()

X = samples[FEATURES].copy()
y = samples[TARGET].copy()

print("\nSelected Features:")
print(FEATURES)
print("Target:", TARGET)
print("Samples with complete history:", len(samples))

# endregion


# region Data Preprocessing

if len(samples) - TEST_DAYS <= CV_SPLITS:
    raise ValueError(
        "Too few observations for the requested chronological split and CV."
    )

if TEST_DAYS <= 0:
    raise ValueError("TEST_DAYS must be a positive integer.")

x_train = X.iloc[:-TEST_DAYS].copy()
x_test = X.iloc[-TEST_DAYS:].copy()

y_train = y.iloc[:-TEST_DAYS].copy()
y_test = y.iloc[-TEST_DAYS:].copy()

cv = TimeSeriesSplit(n_splits=CV_SPLITS)

print("\nTraining observations:", len(x_train))
print("Test observations:", len(x_test))
print("Split method: chronological holdout")

# endregion


# region Data Normalization

def create_preprocessor():
    calendar_encoder = OneHotEncoder(
        categories=[
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
        ],
        drop="first",
        handle_unknown="ignore",
        sparse_output=False,
    )

    return ColumnTransformer([
        ("calendar", calendar_encoder, CALENDAR),
        ("history", StandardScaler(), HISTORY),
    ])


print("\nCalendar features: OneHotEncoder")
print("History features: StandardScaler")

# endregion


# region Accuracy Definition

# Use the training target range as the normalization denominator.
# This avoids using the held-out test target distribution to define the score.
TARGET_RANGE = float(y_train.max() - y_train.min())

if not np.isfinite(TARGET_RANGE) or TARGET_RANGE <= 0:
    raise ValueError("Training target range must be greater than zero.")


def rmse_based_accuracy(rmse, target_range=TARGET_RANGE):
    """
    Project comparison score based on normalized RMSE.
    This is a project-specific regression score, not classification accuracy.
    """
    score = 100.0 * (1.0 - float(rmse) / float(target_range))
    return float(np.clip(score, 0.0, 100.0))


def evaluate(actual, predicted):
    mse = float(mean_squared_error(actual, predicted))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(actual, predicted))

    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "R2 Percentage": 100.0 * r2,
        "Accuracy": rmse_based_accuracy(rmse),
    }

# endregion


# region Model Evaluation

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=80,
        max_depth=2,
        random_state=RANDOM_STATE,
    ),
    "SVR": SVR(kernel="rbf", C=100),
}

evaluation_models = {}
predictions = {}
records = []

for name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", clone(estimator)),
    ])

    cv_scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        error_score="raise",
    )

    fitted_pipeline = clone(pipeline).fit(x_train, y_train)
    prediction = fitted_pipeline.predict(x_test)

    evaluation_models[name] = fitted_pipeline
    predictions[name] = prediction

    test_metrics = evaluate(y_test, prediction)

    records.append({
        "Model": name,
        "CV RMSE": float(-cv_scores.mean()),
        "CV RMSE Std": float(cv_scores.std()),
        **test_metrics,
    })

metrics = (
    pd.DataFrame(records)
    .sort_values(
        ["Accuracy", "CV RMSE", "RMSE"],
        ascending=[False, True, True],
        kind="stable",
    )
    .reset_index(drop=True)
)

# Select the model with the highest RMSE-based test Accuracy.
# Ties are resolved using the lower chronological CV RMSE.
best = str(metrics.iloc[0]["Model"])
best_prediction = predictions[best]

baseline_prediction = x_test["Previous total"].to_numpy()
baseline_metrics = evaluate(y_test, baseline_prediction)

print("\nRegression Model Metrics:")
print(metrics.round(4).to_string(index=False))
print("-" * 60)
print("Recommended Model:", best)
print("Selection Rule: Highest test Accuracy; ties use lower CV RMSE")
print("RMSE normalization range (training target):", TARGET_RANGE)
print("Previous-total baseline:", baseline_metrics)

# endregion


# region Feature Importance

coefficient_model = evaluation_models["Linear Regression"]
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

print("\nFeature Importance Based on Linear Regression Coefficients:")
print(coefficient_df.round(4).to_string(index=False))

if not coefficient_df.empty:
    print("Most Influential Feature:", coefficient_df.iloc[0]["Feature"])

# Keep the 10 most influential coefficients by absolute magnitude,
# then order them by their signed coefficient. With barh(), ascending
# dataframe order is drawn bottom-to-top, so positive values appear first
# at the top and negative values follow from closest to zero to most negative.
plot_coefficients = (
    coefficient_df.nlargest(10, "Absolute Coefficient")
    .sort_values("Coefficient", ascending=True)
    .reset_index(drop=True)
)

fig, ax = plt.subplots(figsize=(14, 8), dpi=110, facecolor=BG_COLOR)
style_axes(ax)
set_chart_title(ax, "Daily Demand - Most Influential Demand Predictors")

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

max_coefficient = max(
    float(plot_coefficients["Absolute Coefficient"].max()),
    1e-9,
)

for bar, value, coefficient in zip(
    bars,
    plot_coefficients["Absolute Coefficient"],
    plot_coefficients["Coefficient"],
):
    ax.text(
        value + max_coefficient * 0.025,
        bar.get_y() + bar.get_height() / 2,
        f"{'-' if coefficient < 0 else '+'}{value:.2f}",
        ha="left",
        va="center",
        fontsize=10,
        color=TITLE_COLOR,
        fontweight="bold",
    )

ax.set_xlim(0, max_coefficient * 1.22)
ax.set_xlabel("Absolute Coefficient (Importance)", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Feature", fontsize=12, color=TEXT_COLOR)
ax.tick_params(axis="x", colors=TEXT_COLOR)
ax.tick_params(axis="y", colors=TEXT_COLOR)

legend = ax.legend(
    handles=[
        Patch(facecolor=PURPLE, label="Positive coefficient"),
        Patch(facecolor=LIGHT_PINK, label="Negative coefficient"),
    ],
    loc="best",
    facecolor=BG_COLOR,
    edgecolor=LAVENDER,
)

for text in legend.get_texts():
    text.set_color(TEXT_COLOR)

display_figure()

# endregion


# region Results

print("\nModel Comparison:")
print(metrics.round(4).to_string(index=False))
print("-" * 60)
print("Best Model (highest RMSE-based Accuracy):", best)

best_metrics = metrics.loc[metrics["Model"] == best].iloc[0]

print("\nBest Model Test Metrics:")
for metric in ["MAE", "MSE", "RMSE", "R2", "R2 Percentage", "Accuracy"]:
    print(f"{metric}: {best_metrics[metric]:.4f}")

# endregion


# region Correlation Heatmap

labels = ["Previous total", "Previous 5 mean", "Today's total"]

correlation_matrix = pd.concat(
    [x_train[HISTORY], y_train],
    axis=1,
).corr()

correlation_matrix.columns = labels
correlation_matrix.index = labels

fig, ax = plt.subplots(figsize=(12, 7), dpi=110, facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

sns.heatmap(
    correlation_matrix,
    xticklabels=labels,
    yticklabels=labels,
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
    annot_kws={"fontsize": 10, "fontweight": "bold"},
    ax=ax,
)

set_chart_title(ax, "Daily Demand - Correlation Heatmap")
ax.tick_params(axis="x", rotation=0, colors=TEXT_COLOR)
ax.tick_params(axis="y", rotation=0, colors=TEXT_COLOR)
ax.set_xlabel("Variables", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Variables", fontsize=12, color=TEXT_COLOR)

display_figure()

# endregion


# region Data Visualization

# The test period is evaluated as rolling one-step-ahead forecasting:
# the previous observed day's total is available for each next-day prediction.
fig, ax = plt.subplots(figsize=(14, 8), dpi=110, facecolor=BG_COLOR)
style_axes(ax)
set_chart_title(ax, f"Daily Demand - Actual vs Predicted Orders ({best})")

days = x_test.index.to_numpy() + 1

ax.plot(
    days,
    y_test.to_numpy(),
    "o-",
    color=TEAL,
    linewidth=2,
    label="Actual total",
)

ax.plot(
    days,
    best_prediction,
    "s--",
    color=PURPLE,
    linewidth=2,
    label=best,
)

ax.plot(
    days,
    baseline_prediction,
    ":",
    color=TITLE_ACCENT,
    linewidth=2,
    label="Previous total baseline",
)

ax.set_xticks(days)
ax.set_xlabel("Observation number", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Total orders (dataset units)", fontsize=12, color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)

legend = ax.legend(facecolor=BG_COLOR, edgecolor=LAVENDER)
for text in legend.get_texts():
    text.set_color(TEXT_COLOR)

display_figure()


# RMSE-based accuracy comparison, styled like the Abalone comparison chart.
accuracy_comparison = metrics.sort_values(
    "Accuracy",
    ascending=False,
    kind="stable",
).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(15, 8.5), dpi=110, facecolor=BG_COLOR)
style_axes(ax)
set_chart_title(ax, "Daily Demand - Regression Model Accuracy (RMSE-based)")

bars = ax.bar(
    accuracy_comparison["Model"],
    accuracy_comparison["Accuracy"],
    color=MODEL_COLORS[:len(accuracy_comparison)],
    edgecolor="white",
    linewidth=0.9,
    width=0.72,
)

for bar in bars:
    value = float(bar.get_height())
    label = f"{value:.2f}%"

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1.0,
        label,
        ha="center",
        va="bottom",
        fontsize=11,
        color=TITLE_COLOR,
        fontweight="bold",
        clip_on=False,
    )

ax.set_xlabel("Model", fontsize=13, color=TEXT_COLOR, labelpad=12)
ax.set_ylabel("Accuracy (%)", fontsize=13, color=TEXT_COLOR)
ax.set_ylim(0, 108)
ax.set_yticks(np.arange(0, 101, 20))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
ax.margins(x=0.08)
ax.tick_params(axis="x", rotation=18, labelsize=11, colors=TEXT_COLOR)
ax.tick_params(axis="y", labelsize=11, colors=TEXT_COLOR)

display_figure()


# Residual distribution for the model selected by chronological CV.
fig, ax = plt.subplots(figsize=(14, 8), dpi=110, facecolor=BG_COLOR)
style_axes(ax)
set_chart_title(ax, f"Daily Demand - Residual Distribution ({best})")

residuals = y_test.to_numpy() - best_prediction

ax.hist(
    residuals,
    bins=min(20, max(5, len(residuals))),
    color=PURPLE,
    alpha=0.85,
    edgecolor="white",
)

ax.axvline(0, color=PINK, linewidth=2)
ax.set_xlabel("Residual (Actual - Predicted)", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Frequency", fontsize=12, color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)

display_figure()

# endregion


# region Save Results

processed_dir = BASE_DIR / "DataSets" / "Processed"
processed_dir.mkdir(parents=True, exist_ok=True)

result_path = processed_dir / "daily_demand_results.csv"
metrics_path = processed_dir / "daily_demand_metrics.csv"

results = df.copy()
results["Predicted total"] = np.nan
results["Baseline prediction"] = np.nan
results["Residual"] = np.nan

results.loc[x_test.index, "Predicted total"] = best_prediction
results.loc[x_test.index, "Baseline prediction"] = baseline_prediction
results.loc[x_test.index, "Residual"] = residuals

results.to_csv(result_path, index=False)
metrics.to_csv(metrics_path, index=False)

print("\nProcessed results saved:", result_path)
print("Model metrics saved:", metrics_path)

# endregion


# region Training and Saving Deployment Models

model_dir = BASE_DIR / "ModelsOutcome"
model_dir.mkdir(parents=True, exist_ok=True)

feature_path = model_dir / "daily_demand_feature_columns.joblib"
bundle_path = model_dir / "daily_demand_regression_models.joblib"

joblib.dump(FEATURES, feature_path)

# Refit each selected algorithm on all available complete observations.
# Holdout metrics above remain based only on the chronological test period.
deployment_models = {}

for name, estimator in models.items():
    deployment_pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", clone(estimator)),
    ])

    deployment_pipeline.fit(X, y)
    deployment_models[name] = deployment_pipeline

metrics_records = metrics.round(8).to_dict(orient="records")

model_bundle = {
    "dataset_name": "Daily Demand Forecasting Orders",
    "models": deployment_models,
    "model_names": list(deployment_models.keys()),
    "best_model": best,
    "target_column": TARGET,
    "feature_columns": FEATURES,
    "calendar_features": CALENDAR,
    "history_features": HISTORY,
    "history_window": HISTORY_WINDOW,
    "forecast_horizon": "One observed working day (rolling one-step-ahead)",
    "accuracy_definition": (
        "max(0, 100 * (1 - RMSE / training_target_range))"
    ),
    "accuracy_normalization_range": TARGET_RANGE,
    "metrics": metrics_records,
    "metrics_by_model": {
        row["Model"]: row for row in metrics_records
    },
    "training_rows": int(len(X)),
    "evaluation_training_rows": int(len(x_train)),
    "evaluation_test_rows": int(len(x_test)),
    "random_state": RANDOM_STATE,
}

joblib.dump(model_bundle, bundle_path, compress=3)

# Verify that the combined file can be loaded and all five pipelines are usable.
restored_bundle = joblib.load(bundle_path)
verification = {}

for name in deployment_models:
    restored_prediction = restored_bundle["models"][name].predict(x_test)
    expected_prediction = deployment_models[name].predict(x_test)

    verification[name] = bool(
        np.allclose(expected_prediction, restored_prediction)
    )

verified = all(verification.values())

if not verified:
    raise RuntimeError("At least one saved model failed prediction verification.")

print("\nModels Saved Successfully!")
print("Combined model bundle:", bundle_path)
print("Feature columns:", feature_path)
print("Saved model names:", restored_bundle["model_names"])
print("Model verification:", verification)

# endregion


# region Save Metadata

metadata_dir = BASE_DIR / "Metadata"
metadata_dir.mkdir(parents=True, exist_ok=True)

metadata_path = metadata_dir / "D10_daily_demand.json"

metadata = {
    "id": "D10",
    "name": "Daily Demand Forecasting Orders",
    "run_timestamp": datetime.now().isoformat(timespec="seconds"),
    "level": "Intermediate",
    "domain": "Business",
    "task": "Regression",
    "source": "UCI Machine Learning Repository",
    "description": (
        "Forecast total orders for the next observed working day using "
        "calendar information and previously observed order totals."
    ),

    "dataset_statistics": {
        "rows": int(len(raw)),
        "columns": int(len(raw.columns)),
        "column_names": raw.columns.tolist(),
        "data_types": raw.dtypes.astype(str).to_dict(),
        "missing_values": {
            column: int(value)
            for column, value in raw.isna().sum().items()
        },
        "total_missing_values": int(raw.isna().sum().sum()),
        "duplicate_rows": int(raw.duplicated().sum()),
        "descriptive_statistics": json.loads(
            raw.describe().round(3).to_json()
        ),
    },

    "feature_selection": {
        "selected_features": FEATURES,
        "target_column": TARGET,
        "calendar_features": CALENDAR,
        "history_features": HISTORY,
        "excluded_same_day_inputs": [
            column for column in raw.columns
            if column not in calendar_names and column != TARGET
        ],
    },

    "preprocessing": {
        "calendar_method": "OneHotEncoder",
        "history_method": "StandardScaler",
        "history_window": HISTORY_WINDOW,
        "forecast_horizon": (
            "One observed working day; rolling one-step-ahead evaluation"
        ),
        "observations_with_complete_history": int(len(samples)),
    },

    "model_evaluation": {
        "recommended_model": best,
        "validation_method": "TimeSeriesSplit",
        "cv_splits": CV_SPLITS,
        "selection_rule": "Highest test Accuracy; ties use lower CV RMSE",
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "test_days": TEST_DAYS,
        "random_state": RANDOM_STATE,
        "metrics": [
            "CV RMSE",
            "CV RMSE Std",
            "MAE",
            "MSE",
            "RMSE",
            "R2",
            "R2 Percentage",
            "Accuracy",
        ],
        "accuracy_definition": (
            "max(0, 100 * (1 - RMSE / training_target_range))"
        ),
        "accuracy_normalization_range": TARGET_RANGE,
        "accuracy_note": (
            "Project-specific RMSE-based regression score; it is not "
            "classification accuracy. The denominator is calculated from "
            "the training target range."
        ),
        "results": metrics.round(6).to_dict(orient="records"),
        "baseline": {
            "method": "Previous observed total",
            "results": baseline_metrics,
        },
    },

    "feature_importance": {
        "method": "Linear regression coefficients after preprocessing",
        "model": "Linear Regression",
        "results": coefficient_df.round(6).to_dict(orient="records"),
        "most_influential_feature": (
            str(coefficient_df.iloc[0]["Feature"])
            if not coefficient_df.empty else None
        ),
    },

    "model_verification": {
        "all_models_verified": verified,
        "individual_models": verification,
    },

    "output_files": {
        "original_dataset": data_path.relative_to(BASE_DIR).as_posix(),
        "processed_dataset": result_path.relative_to(BASE_DIR).as_posix(),
        "metrics": metrics_path.relative_to(BASE_DIR).as_posix(),
        "feature_columns": feature_path.relative_to(BASE_DIR).as_posix(),
        "combined_models": bundle_path.relative_to(BASE_DIR).as_posix(),
        "metadata_latest": metadata_path.relative_to(BASE_DIR).as_posix(),
    },
}

metadata_path.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=4),
    encoding="utf-8",
)

print("\nMetadata saved:", metadata_path)
print("All model verification passed:", verified)
print("\nDaily Demand Forecasting Orders process completed successfully.")

# endregion
