import json
import re
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.patches import Patch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# Qualitative Bankruptcy (D07) - Binary Classification
# Target = Bankruptcy (NB, B)
# Six models are saved together in one JOBLIB file.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_CANDIDATES = [
    BASE_DIR / "DataSets" / "Qualitative_Bankruptcy.data.txt"
]
RANDOM_STATE = 42
TEST_SIZE = 0.20
SHOW_FIGURES = True

DATASET_ID = "D07"
DATASET_NAME = "Qualitative Bankruptcy"
TARGET = "Bankruptcy"

FEATURES = ["Industrial Risk", "Management Risk", "Financial Flexibility",
            "Credibility", "Competitiveness", "Operating Risk"]
FEATURE_CATEGORIES = {feature: ["N", "A", "P"] for feature in FEATURES}
NUMERIC_FEATURES = []
CATEGORICAL_FEATURES = list(FEATURES)
CLASS_ORDER = ["NB", "B"]
CLASS_TO_CODE = {name: code for code, name in enumerate(CLASS_ORDER)}
CODE_TO_CLASS = {code: name for name, code in CLASS_TO_CODE.items()}
CLASS_DESCRIPTIONS = {"NB": "Non-Bankruptcy", "B": "Bankruptcy"}

BG_COLOR = "#FFFFFF"
TITLE_COLOR = "#2B1450"
TITLE_ACCENT = "#C9578F"
TEXT_COLOR = "#665B78"
PURPLE = "#8B5CC7"
PINK = "#E3A0C4"
TEAL = "#21B8A8"
LAVENDER = "#C9B6F2"
LIGHT_PINK = "#F3A6C8"
MODEL_COLORS = ["#7E57C2", "#6750A4", "#4F5FBF", "#3F72AF", "#5B9BD5", "#21B8A8"]

# One fixed color per bankruptcy class, reused in every chart.
CLASS_COLORS = {"NB": PURPLE, "B": LIGHT_PINK}

HEATMAP_CMAP = sns.blend_palette(
    ["#FFF7FB", "#F6CFE3", "#D69ACD", "#9562AF", "#51267D"],
    as_cmap=True,
)


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


def style_legend(legend):
    legend.get_frame().set_facecolor(BG_COLOR)
    legend.get_frame().set_edgecolor(LAVENDER)
    for legend_text in legend.get_texts():
        legend_text.set_color(TEXT_COLOR)


def display_figure():
    fig = plt.gcf()
    manager = fig.canvas.manager
    window = getattr(manager, "window", None)

    # Wide window, centered on the screen, while respecting the screen size.
    width_px = 1500
    height_px = 920
    if window is not None and hasattr(window, "winfo_screenwidth"):
        # Tk backend
        try:
            window.update_idletasks()
            screen_w = int(window.winfo_screenwidth())
            screen_h = int(window.winfo_screenheight())
            width_px = min(1700, int(screen_w * 0.94))
            height_px = min(1000, int(screen_h * 0.88))
            pos_x = max(0, (screen_w - width_px) // 2)
            pos_y = max(0, (screen_h - height_px) // 2 - 20)
            window.geometry(f"{width_px}x{height_px}+{pos_x}+{pos_y}")
            window.update_idletasks()
        except Exception:
            pass
    elif window is not None and hasattr(window, "screen") and hasattr(window, "move"):
        # Qt backend
        try:
            available = window.screen().availableGeometry()
            width_px = min(1700, int(available.width() * 0.94))
            height_px = min(1000, int(available.height() * 0.88))
            window.resize(width_px, height_px)
            window.move(
                available.x() + (available.width() - width_px) // 2,
                available.y() + (available.height() - height_px) // 2,
            )
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
    categories = [FEATURE_CATEGORIES[feature] for feature in CATEGORICAL_FEATURES]
    try:
        encoder = OneHotEncoder(
            categories=categories,
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        encoder = OneHotEncoder(
            categories=categories,
            handle_unknown="ignore",
            sparse=False,
        )

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot_encoder", encoder),
    ])

    return ColumnTransformer([
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def find_dataset_path():
    for candidate in DATASET_CANDIDATES:
        if candidate.exists():
            return candidate

    searched = "\n".join(str(candidate) for candidate in DATASET_CANDIDATES)
    raise FileNotFoundError(
        "Dataset file was not found. These paths were checked:\n"
        f"{searched}\n"
        "Put the Qualitative Bankruptcy file inside the DataSets folder "
        "or add its name to DATASET_CANDIDATES."
    )


def load_raw_dataset(path):
    rows = []
    with path.open(encoding="utf-8-sig") as file:
        for line in file:
            if not line.strip():
                continue
            tokens = [t.strip().strip('"').strip("'").upper() for t in re.split(r"[,;\t]", line.strip())]
            if len(tokens) == 1:
                tokens = line.upper().split()
            if tokens[-1] in {"CLASS", "BANKRUPTCY", "TARGET", "LABEL"}:
                continue
            if len(tokens) != 7 or tokens[-1] not in CLASS_ORDER or any(t not in {"P", "A", "N"} for t in tokens[:-1]):
                raise ValueError(f"Invalid bankruptcy row: {line!r}")
            rows.append(tokens)
    if not rows:
        raise ValueError("No valid bankruptcy rows found")
    return pd.DataFrame(rows, columns=FEATURES + [TARGET])


# -----------------------------
# Load and prepare dataset
# -----------------------------

DATASET_PATH = find_dataset_path()

source_df = load_raw_dataset(DATASET_PATH)
source_df = source_df.dropna(axis=0, how="all").dropna(axis=1, how="all")

for feature in FEATURES:
    unknown_values = sorted(set(source_df[feature]) - set(FEATURE_CATEGORIES[feature]))
    if unknown_values:
        raise ValueError(f"Unexpected values in column '{feature}': {unknown_values}")

bankruptcy_df = source_df.reset_index(drop=True)
bankruptcy_df["Target Code"] = bankruptcy_df[TARGET].map(CLASS_TO_CODE)

class_counts = bankruptcy_df[TARGET].value_counts().reindex(CLASS_ORDER)

print("=" * 70)
print("QUALITATIVE BANKRUPTCY - DATASET INFORMATION")
print("=" * 70)
print("Dataset:", DATASET_NAME)
print("Dataset ID:", DATASET_ID)
print("Dataset file:", DATASET_PATH.name)
print("Prepared shape:", bankruptcy_df[FEATURES + [TARGET]].shape)
print("Features:", FEATURES)
print("Target:", TARGET)
print("\nFirst five rows:")
print(bankruptcy_df[FEATURES + [TARGET]].head().to_string(index=False))
print("\nData types:")
print(bankruptcy_df[FEATURES + [TARGET]].dtypes)
print("\nMissing values:")
print(bankruptcy_df[FEATURES + [TARGET]].isna().sum())
print("\nDuplicate rows:", int(bankruptcy_df[FEATURES + [TARGET]].duplicated().sum()))
print("\nClass distribution:")
print(class_counts.to_string())
print("\nCategory counts per feature:")
for feature in FEATURES:
    counts = bankruptcy_df[feature].value_counts().reindex(FEATURE_CATEGORIES[feature]).fillna(0)
    print(f"  {feature}: " + ", ".join(f"{name}={int(count)}" for name, count in counts.items()))


# -----------------------------
# Train/test split
# -----------------------------

X = bankruptcy_df[FEATURES].copy()
y = bankruptcy_df["Target Code"].copy()

# stratify is important here because the classes are strongly imbalanced.
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTrain rows:", len(x_train))
print("Test rows:", len(x_test))
print("Categorical features:", CATEGORICAL_FEATURES)


# -----------------------------
# Six classification models
# -----------------------------

# class_weight="balanced" adjusts for unequal class frequencies.
model_estimators = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "SVM": SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced"),
    "Naive Bayes": GaussianNB(),
}

trained_models = {}
predictions = {}
metric_records = []

for model_name, estimator in model_estimators.items():
    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", estimator),
    ])

    pipeline.fit(x_train, y_train)
    prediction = pipeline.predict(x_test)

    trained_models[model_name] = pipeline
    predictions[model_name] = prediction

    accuracy = float(accuracy_score(y_test, prediction))
    f1_weighted = float(f1_score(y_test, prediction, average="weighted"))
    f1_macro = float(f1_score(y_test, prediction, average="macro"))

    metric_records.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Accuracy Percentage": accuracy * 100.0,
        "F1": f1_weighted,
        "F1 Percentage": f1_weighted * 100.0,
        "F1 Macro": f1_macro,
    })

metrics_df = pd.DataFrame(metric_records)
metrics_df = metrics_df.sort_values(
    ["Accuracy", "F1"],
    ascending=[False, False],
    kind="stable",
).reset_index(drop=True)

# Best model has the highest test Accuracy (weighted F1 breaks ties).
best_model_name = str(metrics_df.iloc[0]["Model"])
best_prediction = predictions[best_model_name]
best_metrics = metrics_df.iloc[0].to_dict()

print("\n" + "=" * 70)
print("CLASSIFICATION MODEL COMPARISON (SORTED BY HIGHEST ACCURACY)")
print("=" * 70)
print(metrics_df.round(4).to_string(index=False))
print("\nSelected model:", best_model_name)
print("Selection rule: Highest test Accuracy (weighted F1 breaks ties)")
print(f"Accuracy: {best_metrics['Accuracy Percentage']:.2f}%")
print(f"F1 (weighted): {best_metrics['F1 Percentage']:.2f}%")
print(f"F1 (macro): {best_metrics['F1 Macro'] * 100:.2f}%")

# Detailed per-class report for the best model (console and metadata only).
report_text = classification_report(
    y_test,
    best_prediction,
    labels=list(range(len(CLASS_ORDER))),
    target_names=CLASS_ORDER,
    zero_division=0,
)
report_dict = classification_report(
    y_test,
    best_prediction,
    labels=list(range(len(CLASS_ORDER))),
    target_names=CLASS_ORDER,
    zero_division=0,
    output_dict=True,
)
print("\nClassification report (" + best_model_name + "):")
print(report_text)

cm = confusion_matrix(
    y_test,
    best_prediction,
    labels=list(range(len(CLASS_ORDER))),
)


# ============================================================
# CHARTS
# ============================================================

# -----------------------------
# 1) Class distribution
# -----------------------------

fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG_COLOR)
style_axes(ax)
set_chart_title(ax, "Qualitative Bankruptcy - Class Distribution")
distribution_bars = ax.bar(
    class_counts.index,
    class_counts.values,
    color=[CLASS_COLORS[name] for name in class_counts.index],
    edgecolor="white",
    linewidth=0.8,
)

total_samples = int(class_counts.sum())
for bar in distribution_bars:
    value = int(bar.get_height())
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + max(class_counts.max() * 0.012, 0.5),
        f"{value}  ({value / total_samples:.1%})",
        ha="center",
        va="bottom",
        fontsize=11,
        color=TITLE_COLOR,
        fontweight="bold",
    )

ax.set_xticks(range(len(CLASS_ORDER)))
ax.set_xticklabels([CLASS_DESCRIPTIONS[name] for name in class_counts.index])
ax.set_xlabel("Bankruptcy", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Number of Companies", fontsize=12, color=TEXT_COLOR)
ax.set_ylim(0, float(class_counts.max()) * 1.15)
ax.tick_params(colors=TEXT_COLOR)
ax.margins(x=0.08)
display_figure()


# -----------------------------
# 2) Bankruptcy share by feature value (100% stacked bars)
# -----------------------------

fig = plt.figure(figsize=(18, 9), facecolor=BG_COLOR)
grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.6])
feature_axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]
legend_ax = fig.add_subplot(grid[:, 3])
set_chart_title(feature_axes[0], "Qualitative Bankruptcy - Feature Distribution by Class")

for ax, feature in zip(feature_axes, FEATURES):
    style_axes(ax)
    categories = FEATURE_CATEGORIES[feature]
    share_df = pd.crosstab(
        bankruptcy_df[feature],
        bankruptcy_df[TARGET],
        normalize="index",
    ).reindex(index=categories, columns=CLASS_ORDER).fillna(0.0)

    bottom = np.zeros(len(categories))
    for class_name in CLASS_ORDER:
        values = share_df[class_name].to_numpy()
        ax.bar(
            range(len(categories)),
            values,
            bottom=bottom,
            color=CLASS_COLORS[class_name],
            edgecolor="white",
            linewidth=0.8,
            width=0.7,
        )
        for position, (value, base) in enumerate(zip(values, bottom)):
            # Only segments that are large enough get a percentage label.
            if value >= 0.10:
                ax.text(
                    position,
                    base + value / 2,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=TITLE_COLOR,
                    fontweight="bold",
                )
        bottom = bottom + values

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylim(0, 1)
    ax.set_title(feature, fontsize=12, fontweight="bold", color=TITLE_COLOR)
    ax.set_ylabel("Share of Companies", fontsize=10, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)

legend_ax.axis("off")
legend = legend_ax.legend(
    handles=[
        Patch(facecolor=CLASS_COLORS[name], label=CLASS_DESCRIPTIONS[name])
        for name in CLASS_ORDER
    ],
    title="Bankruptcy",
    loc="center",
    fontsize=12,
    title_fontsize=13,
)
style_legend(legend)
legend.get_title().set_color(TITLE_COLOR)
display_figure()


# -----------------------------
# 3) PCA projection of the samples (2D view of one-hot encoded features)
# -----------------------------

pca_pipeline = Pipeline([
    ("preprocessor", create_preprocessor()),
    ("pca", PCA(n_components=2, random_state=RANDOM_STATE)),
])
pca_points = pca_pipeline.fit_transform(X)
explained = pca_pipeline.named_steps["pca"].explained_variance_ratio_

fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG_COLOR)
style_axes(ax)
set_chart_title(ax, "Qualitative Bankruptcy - PCA Projection of Samples")
for class_name in CLASS_ORDER:
    mask = (bankruptcy_df[TARGET] == class_name).to_numpy()
    ax.scatter(
        pca_points[mask, 0],
        pca_points[mask, 1],
        s=40,
        color=CLASS_COLORS[class_name],
        edgecolors=BG_COLOR,
        linewidths=0.6,
        alpha=0.75,
        label=class_name,
    )
ax.set_xlabel(f"Principal Component 1 ({explained[0]:.1%} variance)", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel(f"Principal Component 2 ({explained[1]:.1%} variance)", fontsize=12, color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
legend = ax.legend(title="Bankruptcy", loc="best")
style_legend(legend)
legend.get_title().set_color(TITLE_COLOR)
display_figure()


# -----------------------------
# 4) Correlation heatmap (ordered categories as numbers, lower triangle only)
# -----------------------------

# The categories have a natural order, so they can be turned into 0, 1, 2...
# only for this chart. The models use one-hot encoding instead.
ordinal_df = pd.DataFrame({
    feature: bankruptcy_df[feature].map(
        {name: code for code, name in enumerate(FEATURE_CATEGORIES[feature])}
    )
    for feature in FEATURES
})


correlation_matrix = ordinal_df.corr(numeric_only=True)
upper_mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(13, 8), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
sns.heatmap(
    correlation_matrix,
    mask=upper_mask,
    annot=True,
    fmt=".2f",
    cmap=HEATMAP_CMAP,
    vmin=-1,
    vmax=1,
    center=0,
    linewidths=0.8,
    linecolor=BG_COLOR,
    cbar_kws={"label": "Correlation"},
    annot_kws={"fontsize": 10, "fontweight": "bold"},
    ax=ax,
)
set_chart_title(ax, "Qualitative Bankruptcy - Correlation Heatmap (Ordered Categories)")
ax.tick_params(axis="x", rotation=40, colors=TEXT_COLOR)
plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
ax.tick_params(axis="y", rotation=0, colors=TEXT_COLOR)
ax.set_xlabel("Variables", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Variables", fontsize=12, color=TEXT_COLOR)
display_figure()


# -----------------------------
# Feature importance (data for chart 5)
# -----------------------------

# Random Forest gives one importance score per one-hot column.
# The columns of each original feature are summed into one score.
rf_pipeline = trained_models["Random Forest"]
rf_preprocessor = rf_pipeline.named_steps["preprocessor"]
rf_estimator = rf_pipeline.named_steps["model"]
transformed_feature_names = [
    name.split("__", 1)[-1] for name in rf_preprocessor.get_feature_names_out()
]

importance_by_feature = {feature: 0.0 for feature in FEATURES}
for column_name, value in zip(transformed_feature_names, rf_estimator.feature_importances_):
    for feature in FEATURES:
        if column_name.startswith(feature + "_"):
            importance_by_feature[feature] += float(value)
            break

importance_df = pd.DataFrame({
    "Feature": list(importance_by_feature.keys()),
    "Importance": list(importance_by_feature.values()),
})
importance_df = importance_df.sort_values(
    "Importance",
    ascending=False,
    kind="stable",
).reset_index(drop=True)

# -----------------------------
# 5) Feature importance
# -----------------------------

# Horizontal bars are drawn from bottom to top, so the order is reversed.
plot_importance = importance_df.iloc[::-1]

fig, ax = plt.subplots(figsize=(15, 8), facecolor=BG_COLOR)
style_axes(ax)
set_chart_title(
    ax,
    "Qualitative Bankruptcy - Most Influential Features",
)
bars = ax.barh(
    plot_importance["Feature"],
    plot_importance["Importance"],
    color=PURPLE,
    edgecolor="white",
)

max_importance = float(plot_importance["Importance"].max())
for bar, importance in zip(bars, plot_importance["Importance"]):
    ax.text(
        float(importance) + max(max_importance * 0.012, 0.005),
        bar.get_y() + bar.get_height() / 2,
        f"{importance:.3f}",
        ha="left",
        va="center",
        fontsize=10,
        color=TITLE_COLOR,
        fontweight="bold",
    )

ax.set_xlim(0, max_importance * 1.20 if max_importance > 0 else 1)
ax.set_xlabel("Feature Importance (Random Forest, summed over categories)", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Feature", fontsize=12, color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
display_figure()


# -----------------------------
# 6) Model comparison (Accuracy and F1 on the test set)
# Highest Accuracy is shown first, the best model is marked.
# -----------------------------

comparison_plot = metrics_df.copy()

fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG_COLOR)
style_axes(ax)
set_chart_title(
    ax,
    "Qualitative Bankruptcy - Classification Model Comparison (Accuracy and F1)",
)

positions = np.arange(len(comparison_plot))
bar_width = 0.38

accuracy_bars = ax.bar(
    positions - bar_width / 2,
    comparison_plot["Accuracy Percentage"],
    width=bar_width,
    color=PURPLE,
    edgecolor="white",
    linewidth=0.8,
)
f1_bars = ax.bar(
    positions + bar_width / 2,
    comparison_plot["F1 Percentage"],
    width=bar_width,
    color=TEAL,
    edgecolor="white",
    linewidth=0.8,
)

for bar in list(accuracy_bars) + list(f1_bars):
    value = float(bar.get_height())
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1.0,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        color=TITLE_COLOR,
        fontweight="bold",
    )

model_labels = [
    f"{name}\n(Best)" if name == best_model_name else name
    for name in comparison_plot["Model"]
]
ax.set_xticks(positions)
ax.set_xticklabels(model_labels)
ax.set_xlabel("Model", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Score (%)", fontsize=12, color=TEXT_COLOR)
ax.set_ylim(0, 110)
ax.set_yticks(np.arange(0, 101, 20))
ax.tick_params(axis="x", colors=TEXT_COLOR)
ax.tick_params(axis="y", colors=TEXT_COLOR)
ax.margins(x=0.05)
legend = ax.legend(
    handles=[
        Patch(facecolor=PURPLE, label="Accuracy"),
        Patch(facecolor=TEAL, label="F1 (weighted)"),
    ],
    loc="lower right",
)
style_legend(legend)
display_figure()


# -----------------------------
# Save processed data and metrics
# -----------------------------

processed_folder = BASE_DIR / "DataSets" / "Processed"
processed_folder.mkdir(parents=True, exist_ok=True)

processed_df = bankruptcy_df[FEATURES + [TARGET]].copy()
processed_df["Predicted_Bankruptcy"] = None
processed_df.loc[y_test.index, "Predicted_Bankruptcy"] = [
    CODE_TO_CLASS[int(code)] for code in best_prediction
]

processed_path = processed_folder / "qualitative_bankruptcy_classification_results.csv"
metrics_path = processed_folder / "qualitative_bankruptcy_metrics.csv"
feature_importance_path = processed_folder / "qualitative_bankruptcy_feature_importance.csv"

processed_df.to_csv(processed_path, index=False)
metrics_df.to_csv(metrics_path, index=False)
importance_df.to_csv(feature_importance_path, index=False)


# -----------------------------
# Save all six models together
# -----------------------------

models_folder = BASE_DIR / "ModelsOutcome"
models_folder.mkdir(parents=True, exist_ok=True)

models_bundle_path = models_folder / "qualitative_bankruptcy_classification_models.joblib"
feature_columns_path = models_folder / "qualitative_bankruptcy_feature_columns.joblib"

models_bundle = {
    "dataset_id": DATASET_ID,
    "dataset_name": DATASET_NAME,
    "target_column": TARGET,
    "feature_columns": FEATURES,
    "feature_categories": FEATURE_CATEGORIES,
    "class_names": CLASS_ORDER,
    "class_to_code": CLASS_TO_CODE,
    "code_to_class": CODE_TO_CLASS,
    "models": trained_models,
    "model_metrics": metrics_df.to_dict(orient="records"),
    "best_model_name": best_model_name,
    "best_model": trained_models[best_model_name],
    "metric_definition": (
        "Accuracy = correct predictions / all predictions; "
        "F1 = weighted-average F1-score over the two bankruptcy classes."
    ),
    "selection_rule": "Highest test Accuracy (weighted F1 breaks ties)",
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
}

joblib.dump(models_bundle, models_bundle_path)
joblib.dump(FEATURES, feature_columns_path)

# Verify that the saved best model can be loaded and reproduces predictions.
loaded_bundle = joblib.load(models_bundle_path)
loaded_best_prediction = loaded_bundle["best_model"].predict(x_test)
model_verification = bool(np.array_equal(best_prediction, loaded_best_prediction))

print("\n" + "=" * 70)
print("MODEL SAVING")
print("=" * 70)
print("All six models saved in:", models_bundle_path)
print("Feature columns saved in:", feature_columns_path)
print("Model verification:", model_verification)
print("Models in bundle:", list(loaded_bundle["models"].keys()))


# -----------------------------
# Save one D07 metadata file
# -----------------------------

metadata_folder = BASE_DIR / "Metadata"
metadata_folder.mkdir(parents=True, exist_ok=True)

stats_df = bankruptcy_df[FEATURES + [TARGET]]
data_types = {column: str(dtype) for column, dtype in stats_df.dtypes.items()}
missing_values = {
    column: int(value)
    for column, value in stats_df.isna().sum().items()
}
category_counts = {
    feature: {
        category: int(count)
        for category, count in bankruptcy_df[feature]
        .value_counts()
        .reindex(FEATURE_CATEGORIES[feature])
        .fillna(0)
        .items()
    }
    for feature in FEATURES
}
metrics_metadata = json.loads(
    metrics_df.round(6).to_json(orient="records")
)
feature_importance_metadata = json.loads(
    importance_df.round(6).to_json(orient="records")
)

per_class_metadata = {
    class_name: {
        "precision": round(float(report_dict[class_name]["precision"]), 6),
        "recall": round(float(report_dict[class_name]["recall"]), 6),
        "f1": round(float(report_dict[class_name]["f1-score"]), 6),
        "support": int(report_dict[class_name]["support"]),
    }
    for class_name in CLASS_ORDER
}

metadata = {
    "id": DATASET_ID,
    "name": DATASET_NAME,
    "title": "Qualitative Bankruptcy (D07)",
    "level": "Beginner",
    "domain": "Finance/Banking",
    "task": "Classification",
    "description": "Predict bankruptcy from six qualitative expert assessments (P=Positive, A=Average, N=Negative).",
    "source": "UCI Machine Learning Repository (https://doi.org/10.24432/C52889)",
    "dataset_statistics": {
        "rows": int(stats_df.shape[0]),
        "columns": int(stats_df.shape[1]),
        "column_names": list(stats_df.columns),
        "data_types": data_types,
        "missing_values": missing_values,
        "total_missing_values": int(stats_df.isna().sum().sum()),
        "duplicate_rows": int(stats_df.duplicated().sum()),
        "category_counts": category_counts,
        "class_distribution": {
            class_name: int(count) for class_name, count in class_counts.items()
        },
    },
    "feature_selection": {
        "selected_features": FEATURES,
        "target_column": TARGET,
        "feature_categories": FEATURE_CATEGORIES,
        "class_labels": CLASS_ORDER,
        "class_encoding": CLASS_TO_CODE,
    },
    "preprocessing": {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_method": "Most Frequent Imputation + OneHotEncoder",
        "target_method": "Label encoding (NB=0, B=1)",
        "class_imbalance_handling": (
            "Stratified split and class_weight='balanced' for Logistic, "
            "Decision Tree, Random Forest and SVM"
        ),
    },
    "model_evaluation": {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "split_method": "Stratified train/test split",
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "metrics": [
            "Accuracy",
            "F1",
        ],
        "metric_definition": (
            "Accuracy = correct predictions / all predictions; "
            "F1 = weighted-average F1-score over the two bankruptcy "
            "classes. F1 Macro is also stored because the classes are "
            "imbalanced."
        ),
        "results": metrics_metadata,
        "recommended_model": best_model_name,
        "selection_rule": "Highest test Accuracy (weighted F1 breaks ties)",
        "best_model_per_class_report": per_class_metadata,
        "best_model_confusion_matrix": {
            "labels": CLASS_ORDER,
            "matrix": cm.tolist(),
        },
    },
    "correlation_matrix": json.loads(correlation_matrix.to_json()),
    "feature_importance": {
        "method": "Random Forest Feature Importance (summed over one-hot columns)",
        "model": "Random Forest",
        "results": feature_importance_metadata,
        "most_influential_feature": str(
            importance_df.iloc[0]["Feature"]
        ),
    },
    "charts": [
        "Class Distribution",
        "Feature Distribution by Class",
        "PCA Projection of Samples",
        "Correlation Heatmap (Ordered Categories)",
        "Feature Importance",
        "Model Comparison (Accuracy and F1)",
    ],
    "model_verification": model_verification,
    "ui_card": {
        "title": "Qualitative Bankruptcy (D07)",
        "category": "Finance/Banking",
        "task": "Classification",
        "level": "Beginner",
        "rows": int(stats_df.shape[0]),
        "columns": int(stats_df.shape[1]),
    },
    "output_files": {
        "original_dataset": DATASET_PATH.relative_to(BASE_DIR).as_posix(),
        "processed_dataset": processed_path.relative_to(BASE_DIR).as_posix(),
        "metrics": metrics_path.relative_to(BASE_DIR).as_posix(),
        "feature_importance": feature_importance_path.relative_to(BASE_DIR).as_posix(),
        "models_bundle": models_bundle_path.relative_to(BASE_DIR).as_posix(),
        "feature_columns": feature_columns_path.relative_to(BASE_DIR).as_posix(),
        "metadata": "Metadata/D07_qualitative_bankruptcy.json",
    },
    "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

metadata_path = metadata_folder / "D07_qualitative_bankruptcy.json"
with metadata_path.open("w", encoding="utf-8") as file:
    json.dump(metadata, file, ensure_ascii=False, indent=4, allow_nan=False)

# Read-back verification for the metadata artifact.
with metadata_path.open("r", encoding="utf-8") as file:
    saved_metadata = json.load(file)

print("\n" + "=" * 70)
print("QUALITATIVE BANKRUPTCY - COMPLETED")
print("=" * 70)
print("Dataset card:", saved_metadata["ui_card"]["title"])
print("Rows:", saved_metadata["dataset_statistics"]["rows"])
print("Columns:", saved_metadata["dataset_statistics"]["columns"])
print("Target:", saved_metadata["feature_selection"]["target_column"])
print("Best model:", saved_metadata["model_evaluation"]["recommended_model"])
print("Best Accuracy:", round(float(best_metrics["Accuracy Percentage"]), 2), "%")
print("Best F1:", round(float(best_metrics["F1 Percentage"]), 2), "%")
print("Metadata:", metadata_path)
print("Model bundle:", models_bundle_path)
print("Model verification:", model_verification)
print("\nOnly one dataset entry is generated: Qualitative Bankruptcy (D07).")
