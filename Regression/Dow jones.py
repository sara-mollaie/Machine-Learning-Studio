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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

BASE_DIR = Path(__file__).resolve().parent
SHOW_FIGURES = True
RANDOM_STATE = 42
TARGET = "percent_change_next_weeks_price"
# region Feature Selection
FEATURES = [
    "stock", "percent_change_price", "percent_change_volume_over_last_wk",
    "volume", "previous_weeks_volume", "weekly_range_pct",
    "close_vs_open_pct",
]
# endregion

METRICS = ["MAE", "MSE", "RMSE", "Accuracy", "R2", "R2 Percentage", "Direction Accuracy"]
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

def chart(title, size=(10, 6)):
    fig, ax = plt.subplots(figsize=size, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_axisbelow(True)
    ax.grid(color=LAVENDER, alpha=0.3, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(LAVENDER)
    ax.tick_params(colors=TEXT_COLOR)
    ax.set_title(title, fontsize=18, fontweight="bold", color=TITLE_ACCENT, pad=20)
    return fig, ax


# region Data Normalization

def create_preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), FEATURES[1:]),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["stock"]),
    ])


# endregion

def evaluate(actual, predicted):
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = float(np.sqrt(mse))
    target_range = float(np.max(actual_values) - np.min(actual_values))

    # Same RMSE-based accuracy convention used for the Abalone dataset.
    if target_range > 0:
        accuracy = max(0.0, 100.0 * (1.0 - rmse / target_range))
    else:
        accuracy = 100.0 if rmse == 0 else 0.0

    r2 = r2_score(actual_values, predicted_values)
    return {
        "MAE": float(mean_absolute_error(actual_values, predicted_values)),
        "MSE": float(mse),
        "RMSE": rmse,
        "Accuracy": float(accuracy),
        "R2": float(r2),
        "R2 Percentage": float(100 * r2),
        "Direction Accuracy": float(
            100 * np.mean(np.sign(actual_values) == np.sign(predicted_values))
        ),
    }


# region Load Dataset
dataset_path = BASE_DIR / "DataSets" / "dow_jones_index.data"
raw = pd.read_csv(dataset_path)
raw.columns = raw.columns.str.strip()
df = raw.copy()
df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
for column in df.columns.difference(["stock", "date"]):
    df[column] = pd.to_numeric(
        df[column].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).replace({"nan": np.nan, "": np.nan}),
        errors="raise",
    )
df = df.replace([np.inf, -np.inf], np.nan)
df["stock"] = df["stock"].str.strip()
if df.duplicated(["stock", "date"]).any():
    raise ValueError("Duplicate stock/week records must be resolved.")
if df["stock"].isna().any() or df[TARGET].isna().any():
    raise ValueError("Stock and next-week return must be present.")
if (df["open"] <= 0).any():
    raise ValueError("Opening prices must be positive.")
df["weekly_range_pct"] = 100 * (df["high"] - df["low"]) / df["open"]
df["close_vs_open_pct"] = 100 * (df["close"] - df["open"]) / df["open"]
df = df.sort_values(["date", "stock"]).reset_index(drop=True)
# endregion

# region Data Understanding
print("Dataset shape:", raw.shape)
print(df.head().to_string(index=False))
print("Missing values:\n", df.isna().sum())
print(df.describe().to_string())
# endregion

# region Data Preprocessing and Chronological Split
# Gap weeks ensure that training labels finish before the next evaluation period.
weeks = np.sort(df["date"].unique())
if len(weeks) < 12:
    raise ValueError("At least 12 distinct weeks are required.")
validation_start = int(len(weeks) * 0.6)
test_start = int(len(weeks) * 0.8)
train = df[df["date"].isin(weeks[:validation_start - 1])]
validation = df[df["date"].isin(weeks[validation_start:test_start - 1])]
test = df[df["date"].isin(weeks[test_start:])]
print("Train/validation/test rows:", len(train), len(validation), len(test))
# endregion

# region Model Evaluation
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=10),
    "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                          random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=2,
                                                  learning_rate=0.05, random_state=RANDOM_STATE),
    "SVR": SVR(C=1.0, epsilon=0.1),
}
trained_models, records = {}, []
for name, estimator in models.items():
    pipeline = Pipeline([("preprocessor", create_preprocessor()), ("model", estimator)])
    pipeline.fit(train[FEATURES], train[TARGET])
    prediction = pipeline.predict(validation[FEATURES])
    records.append({"Model": name, **evaluate(validation[TARGET], prediction)})
    trained_models[name] = pipeline
metrics_df = pd.DataFrame(records).sort_values("RMSE", kind="stable").reset_index(drop=True)
best = metrics_df.iloc[0]["Model"]
print("Validation metrics:\n", metrics_df.to_string(index=False, float_format="{:.4f}".format))
print("Recommended Model:", best)
print("Selection rule: lowest validation RMSE / highest RMSE-based Accuracy")
print("Accuracy definition: max(0, 100 * (1 - RMSE / validation target range)))")
# Refit all model bundles on the same pre-test period after selection.
pretest = df[df["date"].isin(weeks[:test_start - 1])]
for pipeline in trained_models.values():
    pipeline.fit(pretest[FEATURES], pretest[TARGET])
best_prediction = trained_models[best].predict(test[FEATURES])
test_metrics = evaluate(test[TARGET], best_prediction)
baseline_metrics = evaluate(test[TARGET], np.zeros(len(test)))
print("Final test metrics:", test_metrics)
print("Zero-return baseline:", baseline_metrics)
# endregion

# region Results and Weekly Ranking
ranking = test[["date", "stock", TARGET]].copy()
ranking["Predicted Return (%)"] = best_prediction
ranking["Rank"] = ranking.groupby("date")["Predicted Return (%)"].rank(method="first", ascending=False).astype(int)
ranking = ranking.sort_values(["date", "Rank"])
top_picks = ranking[ranking["Rank"] == 1]
print("Top predicted stock per test week:\n", top_picks.to_string(index=False))
weekly_correlations = []
for _, week in ranking.groupby("date"):
    if week["Predicted Return (%)"].nunique() > 1 and week[TARGET].nunique() > 1:
        weekly_correlations.append(week["Predicted Return (%)"].corr(week[TARGET], method="spearman"))
ranking_summary = {
    "mean_weekly_spearman": float(np.mean(weekly_correlations)) if weekly_correlations else None,
    "mean_actual_top_pick_return_pct": float(top_picks[TARGET].mean()),
    "test_weeks": int(ranking["date"].nunique()),
}
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
top_coefficients = coefficient_df.head(10).sort_values("Coefficient", ascending=False)
plot_coefficients = top_coefficients.iloc[::-1]
plt.figure(figsize=(10.5, 6.2), facecolor=BG_COLOR)
ax = plt.gca()
style_axes(ax)
set_chart_title(ax, "Dow Jones - Most Influential Features")
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

# region Correlation Heatmap
fig, ax = chart("Dow Jones - Training Correlation", (10.2, 6.2))
sns.heatmap(train[FEATURES[1:] + [TARGET]].corr(), annot=True, fmt=".2f",
            cmap=sns.blend_palette(["#FFF7FB", "#F6CFE3", "#D69ACD", "#9562AF", "#51267D"], as_cmap=True),
            vmin=-1, vmax=1, linewidths=.8, annot_kws={"fontsize":9}, ax=ax)
labels = ["Price change", "Volume change", "Volume", "Previous volume", "Weekly range", "Close / open", "Next-week return"]
ax.set_xticklabels(labels, rotation=35, ha="right")
ax.set_yticklabels(labels, rotation=0)
display_figure()
# endregion

# region Data Visualization


# Match the Abalone chart: highest RMSE-based accuracy appears first.
accuracy_chart_df = metrics_df.sort_values(
    "RMSE",
    ascending=True,
    kind="stable",
).reset_index(drop=True)

fig, ax = chart(
    "Dow Jones - Regression Model Accuracy (RMSE-based)",
    (11.5, 6.5),
)
bars = ax.bar(
    accuracy_chart_df["Model"],
    accuracy_chart_df["Accuracy"],
    color=MODEL_COLORS[:len(accuracy_chart_df)],
    edgecolor="white",
)
ax.bar_label(
    bars,
    fmt="%.2f%%",
    padding=6,
    color=TITLE_COLOR,
    fontweight="bold",
    fontsize=11,
)
ax.set_ylabel("Accuracy (%)", color=TEXT_COLOR)
ax.set_xlabel("Model", color=TEXT_COLOR)
ax.set_ylim(0, 105)
ax.tick_params(axis="x", rotation=20, labelsize=10)
ax.tick_params(axis="y", labelsize=10)
display_figure()

fig, ax = chart(f"Dow Jones - Actual vs Predicted ({best})")
ax.scatter(test[TARGET], best_prediction, color=TEAL, alpha=0.75, edgecolors="white")
bounds = [min(test[TARGET].min(), best_prediction.min()), max(test[TARGET].max(), best_prediction.max())]
ax.plot(bounds, bounds, color=PINK, label="Ideal prediction")
ax.set_xlabel("Actual next-week return (%)")
ax.set_ylabel("Predicted next-week return (%)")
ax.legend()
display_figure()

fig, ax = chart(f"Dow Jones - Residual Distribution ({best})")
ax.hist(test[TARGET].to_numpy() - best_prediction, bins=25, color=PURPLE, edgecolor="white")
ax.axvline(0, color=PINK, linewidth=2)
ax.set_xlabel("Residual (percentage points)")
ax.set_ylabel("Frequency")
display_figure()

# endregion

# region Save Results
output = BASE_DIR / "DataSets" / "Processed"
model_dir = BASE_DIR / "ModelsOutcome"
metadata_dir = BASE_DIR / "Metadata"
for directory in [output, model_dir, metadata_dir]:
    directory.mkdir(parents=True, exist_ok=True)
metrics_df.to_csv(output / "dow_jones_metrics.csv", index=False)
ranking.to_csv(output / "dow_jones_weekly_rankings.csv", index=False)
# endregion

# region Training Models
# Store all five trained pipelines and their evaluation information in one bundle.
model_bundle_path = model_dir / "dow_jones_regression_models.joblib"
model_bundle = {
    "dataset_name": "Dow Jones Index",
    "models": trained_models,
    "model_names": list(trained_models.keys()),
    "best_model_name": best,
    "feature_columns": FEATURES,
    "target_column": TARGET,
    "feature_formulas": {
        "weekly_range_pct": "100 * (high - low) / open",
        "close_vs_open_pct": "100 * (close - open) / open",
    },
    "evaluation": {
        "selection_metric": "RMSE",
        "selection_rule": "Lowest validation RMSE",
        "accuracy_definition": (
            "max(0, 100 * (1 - RMSE / (max(y_validation) - min(y_validation))))"
        ),
        "validation_target_min": float(validation[TARGET].min()),
        "validation_target_max": float(validation[TARGET].max()),
        "validation_target_range": float(validation[TARGET].max() - validation[TARGET].min()),
        "validation_metrics": metrics_df.to_dict(orient="records"),
        "test_metrics_for_best_model": test_metrics,
        "baseline_test_metrics": baseline_metrics,
        "split_method": "Chronological whole-week 60/20/20 with one-week boundary gaps",
    },
    "training_settings": {
        "random_state": RANDOM_STATE,
        "numeric_preprocessing": "Median imputation + StandardScaler",
        "categorical_preprocessing": "OneHotEncoder(handle_unknown='ignore')",
    },
}
joblib.dump(model_bundle, model_bundle_path)

# Reload the saved bundle and verify that all five models reproduce their predictions.
restored_bundle = joblib.load(model_bundle_path)
verification_results = {}
for model_name, original_pipeline in trained_models.items():
    restored_prediction = restored_bundle["models"][model_name].predict(test[FEATURES])
    original_prediction = original_pipeline.predict(test[FEATURES])
    verification_results[model_name] = bool(
        np.allclose(restored_prediction, original_prediction)
    )
verified = bool(all(verification_results.values()))
print("Saved model bundle:", model_bundle_path)
print("All model verification results:", verification_results)
# endregion

# region Save Metadata
metadata = {
    "correlation_matrix": json.loads((train[FEATURES[1:] + [TARGET]].corr()).to_json()),
    "id": "D16", "run_timestamp": datetime.now().isoformat(timespec="seconds"), "name": "Dow Jones Index", "level": "Intermediate",
    "domain": "Business/Finance", "task": "Regression",
    "description": "Predict next-week stock returns and rank stocks within each week.",
    "source": "UCI Machine Learning Repository",
    "dataset_statistics": {
        "rows": len(raw), "columns": len(raw.columns), "column_names": list(raw.columns),
        "data_types": {c: str(df[c].dtype) for c in raw.columns},
        "missing_values": {c: int(raw[c].isna().sum()) for c in raw.columns},
        "total_missing_values": int(raw.isna().sum().sum()),
        "descriptive_statistics": json.loads(df.describe().to_json()),
    },
    "preprocessing": {
        "numeric_method": "Median Imputation + StandardScaler",
        "categorical_method": "OneHotEncoder for stock",
        "feature_formulas": {"weekly_range_pct": "100 * (high - low) / open",
                             "close_vs_open_pct": "100 * (close - open) / open"},
    },
    "feature_selection": {"selected_features": FEATURES, "target_column": TARGET,
                          "excluded_future_columns": ["next_weeks_open", "next_weeks_close"]},
    "model_evaluation": {
        "metrics": METRICS,
        "results": json.loads(metrics_df.to_json(orient="records")),
        "recommended_model": best,
        "selection_rule": "Lowest validation RMSE (equivalent to highest RMSE-based Accuracy)",
        "accuracy_definition": (
            "max(0, 100 * (1 - RMSE / (max(y_validation) - min(y_validation))))"
        ),
        "accuracy_calculation_note": (
            "The same validation target range is used for all five models, "
            "so higher Accuracy corresponds exactly to lower RMSE."
        ),
        "split_method": "Chronological whole-week 60/20/20 with one-week boundary gaps",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "refit_rows": len(pretest),
        "test_results": test_metrics,
        "baseline_test_results": baseline_metrics,
        "direction_accuracy_definition": "100 * mean(sign(actual) == sign(predicted)); zero is a separate direction",
        "ranking": ranking_summary,
        "model_verification_results": verification_results,
    },
    "feature_importance": {"method": "Linear Regression coefficients after preprocessing",
                          "results": json.loads(coefficient_df.to_json(orient="records"))},
    "model_verification": verified,
    "output_files": {
        "models": "ModelsOutcome/dow_jones_regression_models.joblib",
        "original_dataset": "DataSets/dow_jones_index.data",
        "metadata_latest": "Metadata/D16_dow_jones.json",
        "metrics": "DataSets/Processed/dow_jones_metrics.csv",
        "rankings": "DataSets/Processed/dow_jones_weekly_rankings.csv",
    },
}
with (metadata_dir / "D16_dow_jones.json").open("w", encoding="utf-8") as file:
    json.dump(metadata, file, ensure_ascii=False, indent=4, allow_nan=False)
print("Metadata Verification:")
print("Dataset Name:", metadata["name"])
print("Total Rows:", metadata["dataset_statistics"]["rows"])
print("Total Columns:", metadata["dataset_statistics"]["columns"])
print("Target Column:", TARGET)
print("Recommended Model:", best)
print("Model Verification:", verified)
# endregion
