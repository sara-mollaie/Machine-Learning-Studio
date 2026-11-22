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
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


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
        y = (window.winfo_screenheight() - height) // 2 - 40

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
        fontsize=20,
        fontweight="bold",
        color=TITLE_ACCENT,
        pad=24,
    )


def display_figure():
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    show_figure()

# endregion


# region Load Dataset

data_path = BASE_DIR / "DataSets" / "Daily_Demand_Forecasting_Orders.csv"

raw = pd.read_csv(data_path, sep=";")
raw.columns = raw.columns.str.strip()

calendar_names = {}

for column in raw.columns:
    if column.startswith("Week of the month"):
        calendar_names[column] = CALENDAR[0]
    elif column.startswith("Day of the week"):
        calendar_names[column] = CALENDAR[1]

df = raw.rename(columns=calendar_names).copy()
df[CALENDAR + [TARGET]] = df[CALENDAR + [TARGET]].apply(pd.to_numeric)

if not np.isfinite(df[CALENDAR + [TARGET]].to_numpy()).all():
    raise ValueError(
        "Calendar columns and total orders must contain finite values."
    )

# endregion


# region Data Understanding

print("\nFirst 5 Rows:")
print(raw.head().to_string(index=False))
print("-" * 60)

print("Dataset Shape:", raw.shape)
print("Columns:", raw.columns.tolist())

print("Data Types:")
print(raw.dtypes)

print("Dataset Information:")
raw.info()

print("Statistical Description:")
print(raw.describe().round(3).to_string())

print("Missing values:\n", raw.isna().sum())
print("Duplicate rows:", raw.duplicated().sum())

type_sum = raw[["Order type A", "Order type B", "Order type C"]].sum(axis=1)
print("Same-day A + B + C equals target:", np.allclose(type_sum, df[TARGET]))

# endregion


# region Feature Selection

# Only use information available before the forecasted working day.
previous = df[TARGET].shift(1)

df[HISTORY[0]] = previous
df[HISTORY[1]] = previous.rolling(HISTORY_WINDOW).mean()

samples = df.dropna(subset=FEATURES + [TARGET])

X = samples[FEATURES]
y = samples[TARGET]

print("Features:", FEATURES)
print("Target:", TARGET)

# endregion


# region Data Preprocessing

if len(samples) - TEST_DAYS <= CV_SPLITS:
    raise ValueError("Too few days for chronological training and testing.")

x_train = X.iloc[:-TEST_DAYS]
x_test = X.iloc[-TEST_DAYS:]

y_train = y.iloc[:-TEST_DAYS]
y_test = y.iloc[-TEST_DAYS:]

cv = TimeSeriesSplit(n_splits=CV_SPLITS)

print("Training days:", len(x_train), "| Test days:", len(x_test))

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

# endregion


# region Model Evaluation

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=80,
        max_depth=2,
        random_state=RANDOM_STATE,
    ),
    "SVR": SVR(kernel="rbf", C=100),
}

trained_models = {}
records = []

for name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", estimator),
    ])

    scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        error_score="raise",
    )

    trained_models[name] = pipeline.fit(x_train, y_train)

    records.append({
        "Model": name,
        "CV RMSE": float(-scores.mean()),
    })

metrics = (
    pd.DataFrame(records)
    .sort_values("CV RMSE")
    .reset_index(drop=True)
)



def evaluate(actual, predicted):
    mse = float(mean_squared_error(actual, predicted))
    r2 = float(r2_score(actual, predicted))
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2,
        "R2 Percentage": 100.0 * r2,
        "Accuracy": max(0.0, 100.0 * (1 - mean_absolute_percentage_error(actual, predicted))),
    }


predictions = {}

for i, name in enumerate(metrics["Model"]):
    predictions[name] = trained_models[name].predict(x_test)

    for metric, value in evaluate(y_test, predictions[name]).items():
        metrics.loc[i, metric] = value

metrics = metrics.sort_values(["Accuracy", "CV RMSE"], ascending=[False, True]).reset_index(drop=True)
metrics = metrics.drop(columns="CV RMSE")
best = metrics.loc[0, "Model"]
best_prediction = predictions[best]

baseline_prediction = x_test["Previous total"].to_numpy()
baseline_metrics = evaluate(y_test, baseline_prediction)

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

plot_coefficients = coefficient_df.sort_values("Coefficient")
plt.figure(figsize=(12, 7), facecolor=BG_COLOR)
ax = plt.gca()
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
for bar, value, coefficient in zip(
    bars,
    plot_coefficients["Absolute Coefficient"],
    plot_coefficients["Coefficient"],
):
    ax.text(
        value + 0.012 * coefficient_df["Absolute Coefficient"].max(),
        bar.get_y() + bar.get_height() / 2,
        f"{'-' if coefficient < 0 else '+'}{value:.2f}",
        ha="left",
        va="center",
        fontsize=9,
        color=TITLE_COLOR,
        fontweight="bold",
    )
ax.set_xlim(0, (coefficient_df["Absolute Coefficient"].max() or 1) * 1.2)
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

print("\nModel comparison:\n", metrics.to_string(index=False, float_format="{:.4f}".format))
print("Recommended model:", best)
print("Selection rule: highest test Accuracy; ties use lower CV RMSE")
print("Previous-total baseline:", baseline_metrics)

# endregion


# region Correlation Heatmap

labels = ["Previous total", "Previous 5 mean", "Today's total"]

correlation_matrix = pd.concat(
    [x_train[HISTORY], y_train],
    axis=1,
).corr()

correlation_matrix.columns = labels
correlation_matrix.index = labels

plt.figure(figsize=(11.5, 6.6), dpi=100, facecolor=BG_COLOR)
ax = plt.gca()
ax.set_facecolor(BG_COLOR)

sns.heatmap(
    correlation_matrix,
    xticklabels=labels,
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

set_chart_title(ax, "Daily Demand - Correlation Heatmap")
ax.tick_params(axis="x", rotation=0, colors=TEXT_COLOR)
ax.tick_params(axis="y", rotation=0, colors=TEXT_COLOR)
ax.set_xlabel("Variables", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Variables", fontsize=12, color=TEXT_COLOR)

display_figure()

# endregion


# region Data Visualization

# Actual and predicted orders, in observation order.
plt.figure(figsize=(11, 7), facecolor=BG_COLOR)
ax = plt.gca()
style_axes(ax)
set_chart_title(ax, f"Daily Demand - Actual vs Predicted Orders ({best})")

days = x_test.index.to_numpy() + 1

ax.plot(
    days, y_test, "o-",
    color=TEAL, label="Actual total",
)
ax.plot(
    days, best_prediction, "s--",
    color=PURPLE, label=best,
)
ax.plot(
    days, baseline_prediction, ":",
    color=TITLE_ACCENT, label="Previous total",
)

ax.set_xticks(days)
ax.set_xlabel("Observation number", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Total orders (dataset units)", fontsize=12, color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
ax.legend(
    facecolor=BG_COLOR,
    edgecolor=LAVENDER,
    labelcolor=TEXT_COLOR,
)

display_figure()


# Compare test-set forecast accuracy using the project definition.
plt.figure(figsize=(11, 7), facecolor=BG_COLOR)
ax = plt.gca()
style_axes(ax)
set_chart_title(ax, "Daily Demand - Regression Model Accuracy")

comparison = metrics
bars = ax.bar(
    comparison["Model"],
    comparison["Accuracy"],
    color=MODEL_COLORS,
    linewidth=0.8,
    edgecolor="white",
)

ax.bar_label(
    bars,
    fmt="%.2f%%",
    padding=4,
    color=TITLE_COLOR,
    fontweight="bold",
)

ax.set_xlabel("Model", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Accuracy (%)", fontsize=12, color=TEXT_COLOR)
ax.margins(x=0.08)
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
ax.tick_params(axis="x", rotation=20, colors=TEXT_COLOR)
ax.tick_params(axis="y", colors=TEXT_COLOR)

display_figure()


# Distribution of forecast errors on the final 12 observations.
plt.figure(figsize=(11, 7), facecolor=BG_COLOR)
ax = plt.gca()
style_axes(ax)
set_chart_title(ax, f"Daily Demand - Residual Distribution ({best})")

residuals = y_test.to_numpy() - best_prediction

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

# endregion


# region Save Results

processed_dir = BASE_DIR / "DataSets" / "Processed"
processed_dir.mkdir(parents=True, exist_ok=True)

result_path = processed_dir / "daily_demand_results.csv"
metrics_path = processed_dir / "daily_demand_metrics.csv"

results = df.copy()
results.loc[x_test.index, "Predicted total"] = best_prediction
results.loc[x_test.index, "Baseline prediction"] = baseline_prediction
results.loc[x_test.index, "Residual"] = residuals

results.to_csv(result_path, index=False)
metrics.to_csv(metrics_path, index=False)

# endregion


# region Training Models

model_dir = BASE_DIR / "ModelsOutcome"
model_dir.mkdir(parents=True, exist_ok=True)

feature_path = model_dir / "daily_demand_feature_columns.joblib"
joblib.dump(FEATURES, feature_path)

paths = {}

for name, pipeline in trained_models.items():
    filename = "daily_demand_" + name.lower().replace(" ", "_") + ".joblib"
    path = model_dir / filename

    joblib.dump(
        {
            "model": pipeline,
            "model_name": name,
            "target_column": TARGET,
            "feature_columns": FEATURES,
        },
        path,
    )

    paths[name] = path.relative_to(BASE_DIR).as_posix()

restored = joblib.load(BASE_DIR / paths[best])
restored_prediction = restored["model"].predict(x_test)
verified = bool(np.allclose(best_prediction, restored_prediction))

if not verified:
    raise RuntimeError("Model predictions changed after reloading.")

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
    "description": "Forecast total orders for the next observed working day.",

    "dataset_statistics": {
        "rows": len(raw),
        "columns": len(raw.columns),
        "column_names": raw.columns.tolist(),
        "data_types": raw.dtypes.astype(str).to_dict(),
        "missing_values": raw.isna().sum().to_dict(),
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
        "forecast_horizon": "One observed working day",
        "observations_with_complete_history": len(samples),
    },

    "model_evaluation": {
        "recommended_model": best,
        "validation_method": "TimeSeriesSplit",
        "selection_rule": "Highest test Accuracy; ties use lower CV RMSE",
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "cv_splits": CV_SPLITS,
        "metrics": ["MAE", "MSE", "RMSE", "R2", "R2 Percentage", "Accuracy"],
        "results": metrics.round(4).to_dict("records"),
        "baseline": {
            "method": "Previous observed total",
            "results": baseline_metrics,
        },
    },

    "feature_importance": {
        "method": "Linear regression coefficients after preprocessing",
        "model": "Linear Regression",
        "results": coefficient_df.round(4).to_dict("records"),
    },

    "model_verification": verified,

    "output_files": {
        "original_dataset": data_path.relative_to(BASE_DIR).as_posix(),
        "processed_dataset": result_path.relative_to(BASE_DIR).as_posix(),
        "metrics": metrics_path.relative_to(BASE_DIR).as_posix(),
        "feature_columns": feature_path.relative_to(BASE_DIR).as_posix(),
        "metadata_latest": metadata_path.relative_to(BASE_DIR).as_posix(),
        "models": paths,
    },
}

metadata_path.write_text(
    json.dumps(metadata, indent=4),
    encoding="utf-8",
)

print("Metadata saved:", metadata_path)
print("Model verification:", verified)

# endregion
