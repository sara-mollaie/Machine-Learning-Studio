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
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


BASE_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
TEST_SIZE = 0.2
SHOW_FIGURES = True

FEATURES = [
    "X1 transaction date",
    "X2 house age",
    "X3 distance to the nearest MRT station",
    "X4 number of convenience stores",
    "X5 latitude",
    "X6 longitude",
]
TARGET = "Y house price of unit area"
ID_COLUMN = "No"
FEATURE_DISPLAY_NAMES = {
    "X1 transaction date": "Transaction date",
    "X2 house age": "House age",
    "X3 distance to the nearest MRT station": "Distance to nearest\nMRT station",
    "X4 number of convenience stores": "Convenience\nstores",
    "X5 latitude": "Latitude",
    "X6 longitude": "Longitude",
}

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

dataset_path = BASE_DIR / "DataSets" / "Real estate valuation data set.xlsx"
if not dataset_path.exists():
    dataset_path = BASE_DIR / "Datasets" / "real_estate_valuation.xlsx"
real_estate_df = pd.read_excel(dataset_path)
real_estate_df.columns = real_estate_df.columns.str.strip()

expected_excel_columns = FEATURES + [TARGET]
if list(real_estate_df.columns) == expected_excel_columns:
    real_estate_df.insert(0, ID_COLUMN, np.arange(1, len(real_estate_df) + 1))

expected_columns = [ID_COLUMN] + FEATURES + [TARGET]
if list(real_estate_df.columns) != expected_columns:
    raise ValueError("Dataset columns do not match the expected Real Estate columns.")

print("\nReal Estate Valuation Dataset Loaded Successfully!")
print("Dataset Path:", dataset_path)

# endregion


# region Data Understanding

print("\nFirst 5 Rows:")
print(real_estate_df.head().to_string(index=False))
print("-" * 60)
print("Dataset Shape:")
print(real_estate_df.shape)
print("-" * 60)
print("Columns:")
print(real_estate_df.columns.tolist())
print("-" * 60)
print("Data Types:")
print(real_estate_df.dtypes)
print("-" * 60)
print("Dataset Information:")
real_estate_df.info()
print("-" * 60)
print("Statistical Description:")
print(real_estate_df.describe().to_string())
print("-" * 60)
print("Missing Values:")
print(real_estate_df.isnull().sum())
print("-" * 60)
print("Total Missing Values:", int(real_estate_df.isnull().sum().sum()))
print("Duplicate Rows:", int(real_estate_df.duplicated().sum()))
print("-" * 60)

# endregion


# region Feature Selection

X = real_estate_df[FEATURES].copy()
y = real_estate_df[TARGET].copy()

print("\nSelected Features:")
print(X.head().to_string(index=False))
print("-" * 60)
print("Features Shape:", X.shape)
print("Feature Names:", FEATURES)
print("Target Variable:", TARGET)
print("Excluded Column:", ID_COLUMN)
print("The ID column is excluded because it is an identifier, not a house attribute.")
print("-" * 60)

# endregion


# region Data Preprocessing

for column in FEATURES + [TARGET]:
    real_estate_df[column] = pd.to_numeric(real_estate_df[column], errors="coerce")

valid_rows = real_estate_df[TARGET].notna()
X = real_estate_df.loc[valid_rows, FEATURES].reset_index(drop=True)
y = real_estate_df.loc[valid_rows, TARGET].reset_index(drop=True)
real_estate_df = real_estate_df.loc[valid_rows].reset_index(drop=True)

print("\nMissing Values In Selected Features:")
print(X.isnull().sum())
print("Missing Values In Target:", int(y.isnull().sum()))
print("-" * 60)

if not np.isfinite(X.to_numpy()).all():
    raise ValueError("Feature values contain infinite values.")
if not np.isfinite(y.to_numpy()).all():
    raise ValueError("Target values contain infinite values.")

print("Preprocessing Completed Successfully!")
print("Missing numeric values are handled with median imputation.")
print("-" * 60)

# endregion


# region Data Normalization

def create_preprocessor():
    return ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            FEATURES,
        ),
    ])


print("\nNormalization: StandardScaler")
print("Input Features:", FEATURES)
print("-" * 60)

# endregion


# region Model Evaluation

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Random Forest": RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=RANDOM_STATE,
    ),
    "SVR": SVR(kernel="rbf"),
}

parameter_grids = {
    "Linear Regression": {},
    "Ridge": {
        "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "Random Forest": {
        "model__n_estimators": [300, 600],
        "model__max_depth": [None, 8, 15],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "Gradient Boosting": {
        "model__n_estimators": [100, 200, 400],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [1, 2, 3],
        "model__min_samples_leaf": [1, 3],
    },
    "SVR": {
        "model__C": [10.0, 30.0, 100.0, 300.0],
        "model__gamma": ["scale", 0.01, 0.03, 0.1],
        "model__epsilon": [0.05, 0.1, 0.3],
    },
}

trained_models = {}
predictions = {}
records = []
best_parameters = {}

for model_name, estimator in models.items():
    model_pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", estimator),
    ])
    if parameter_grids[model_name]:
        search = GridSearchCV(
            estimator=model_pipeline,
            param_grid=parameter_grids[model_name],
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=1,
        )
        search.fit(x_train, y_train)
        fitted_model = search.best_estimator_
        best_parameters[model_name] = search.best_params_
        print(f"{model_name} best parameters:", search.best_params_)
    else:
        model_pipeline.fit(x_train, y_train)
        fitted_model = model_pipeline
        best_parameters[model_name] = {}

    prediction = fitted_model.predict(x_test)
    trained_models[model_name] = fitted_model
    predictions[model_name] = prediction

    mse = mean_squared_error(y_test, prediction)
    mape_percentage = mean_absolute_percentage_error(y_test, prediction) * 100
    records.append({
        "Model": model_name,
        "MAE": mean_absolute_error(y_test, prediction),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_test, prediction),
        "R2 Percentage": r2_score(y_test, prediction) * 100,
        "Accuracy": max(0.0, 100.0 - mape_percentage),
    })

metrics_df = pd.DataFrame(records).sort_values("RMSE", ascending=True).reset_index(drop=True)
best_model_name = metrics_df.iloc[0]["Model"]
best_prediction = predictions[best_model_name]

print("\nRegression Metrics:")
print(metrics_df.round(4).to_string(index=False))
print("-" * 60)
print("Recommended Model:", best_model_name)
print("Selection Rule: Lowest RMSE on the held-out test set")

# endregion


# region Feature Importance

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

positive_coefficients = coefficient_df[coefficient_df["Coefficient"] >= 0].sort_values(
    "Coefficient",
    ascending=False,
)
negative_coefficients = coefficient_df[coefficient_df["Coefficient"] < 0].sort_values(
    "Coefficient",
    ascending=False,
)
plot_coefficients = pd.concat([positive_coefficients, negative_coefficients]).iloc[::-1]

plt.figure(figsize=(12, 7), facecolor=BG_COLOR)
ax = plt.gca()
style_axes(ax)
set_chart_title(ax, "Real Estate - Most Influential Features")
bar_colors = [PINK if value < 0 else PURPLE for value in plot_coefficients["Coefficient"]]
bars = ax.barh(
    plot_coefficients["Feature"].map(FEATURE_DISPLAY_NAMES),
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
        value + 0.01,
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
ax.tick_params(axis="y", labelsize=10)
legend = ax.legend(
    handles=[
        Patch(facecolor=PURPLE, label="Positive coefficient"),
        Patch(facecolor=PINK, label="Negative coefficient"),
    ],
    loc="upper right",
    bbox_to_anchor=(0.98, 0.98),
    ncol=1,
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

print("\nBest Model Metrics:")
for metric in ["MAE", "MSE", "RMSE", "R2", "R2 Percentage", "Accuracy"]:
    print(f"{metric}: {best_metrics[metric]:.4f}")

# endregion


# region Correlation Heatmap

correlation_columns = FEATURES + [TARGET]
correlation_matrix = real_estate_df[correlation_columns].corr()

plt.figure(figsize=(11.5, 6.6), dpi=100, facecolor=BG_COLOR)
ax = plt.gca()
ax.set_facecolor(BG_COLOR)
sns.heatmap(
    correlation_matrix,
    xticklabels=[column.split()[0] for column in correlation_columns],
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
set_chart_title(ax, "Real Estate - Correlation Heatmap")
ax.tick_params(axis="x", rotation=0, colors=TEXT_COLOR)
ax.tick_params(axis="y", rotation=0, colors=TEXT_COLOR)
ax.set_xlabel("Variables", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Variables", fontsize=12, color=TEXT_COLOR)
display_figure()

# endregion


# region Data Visualization

plt.figure(figsize=(11, 7), facecolor=BG_COLOR)
ax = plt.gca()
style_axes(ax)
set_chart_title(ax, f"Real Estate - Actual vs Predicted Price ({best_model_name})")
ax.scatter(
    y_test,
    best_prediction,
    color=TEAL,
    alpha=0.78,
    edgecolors=BG_COLOR,
    linewidths=0.7,
)
minimum = min(y_test.min(), best_prediction.min())
maximum = max(y_test.max(), best_prediction.max())
ax.plot(
    [minimum, maximum],
    [minimum, maximum],
    color=PURPLE,
    linewidth=2,
    label="Ideal prediction",
)
ax.margins(x=0.08, y=0.08)
ax.set_xlabel("Actual Price per Unit Area", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Predicted Price per Unit Area", fontsize=12, color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
legend = ax.legend(facecolor=BG_COLOR, edgecolor=LAVENDER)
for text in legend.get_texts():
    text.set_color(TEXT_COLOR)
display_figure()

plt.figure(figsize=(11, 7), facecolor=BG_COLOR)
ax = plt.gca()
style_axes(ax)
set_chart_title(ax, "Real Estate - Regression Model Accuracy")
accuracy_plot = metrics_df.set_index("Model")[["Accuracy"]].sort_values(
    "Accuracy",
    ascending=False,
)
accuracy_bars = ax.bar(
    accuracy_plot.index,
    accuracy_plot["Accuracy"],
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
        ha="center",
        va="bottom",
        fontsize=10,
        color=TITLE_COLOR,
        fontweight="bold",
    )
ax.set_xlabel("Model", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Accuracy (%)", fontsize=12, color=TEXT_COLOR)
ax.set_ylim(0, max(100, accuracy_plot["Accuracy"].max() + 8))
ax.margins(x=0.08)
ax.tick_params(axis="x", rotation=20, colors=TEXT_COLOR)
ax.tick_params(axis="y", colors=TEXT_COLOR)
display_figure()

plt.figure(figsize=(11, 7), facecolor=BG_COLOR)
ax = plt.gca()
style_axes(ax)
set_chart_title(ax, f"Real Estate - Residual Distribution ({best_model_name})")
residuals = y_test.to_numpy() - best_prediction
ax.hist(residuals, bins=25, color=PURPLE, alpha=0.85, edgecolor="white")
ax.axvline(0, color=PINK, linewidth=2)
ax.set_xlabel("Residual (Actual - Predicted)", fontsize=12, color=TEXT_COLOR)
ax.set_ylabel("Frequency", fontsize=12, color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
display_figure()

# endregion


# region Save Results

output_folder = BASE_DIR / "Datasets" / "Processed"
output_folder.mkdir(parents=True, exist_ok=True)
processed_df = real_estate_df.copy()
processed_df["Predicted_Price"] = np.nan
processed_df.loc[y_test.index, "Predicted_Price"] = np.round(best_prediction, 2)
processed_path = output_folder / "real_estate_regression_results.csv"
processed_df.to_csv(processed_path, index=False)

metrics_path = output_folder / "real_estate_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

print("\nRegression Results Saved Successfully!")
print("Processed Dataset:", processed_path)
print("Metrics:", metrics_path)

# endregion


# region Training Models

model_folder = BASE_DIR / "ModelsOutcome"
model_folder.mkdir(parents=True, exist_ok=True)
model_paths = {}

feature_columns_path = model_folder / "real_estate_feature_columns.joblib"
joblib.dump(FEATURES, feature_columns_path)
print("Saved Feature Columns:", feature_columns_path)

for model_name, model_pipeline in trained_models.items():
    model_file_name = model_name.lower().replace(" ", "_") + ".joblib"
    model_path = model_folder / f"real_estate_{model_file_name}"
    joblib.dump(
        {
            "model": model_pipeline,
            "model_name": model_name,
            "target_column": TARGET,
            "feature_columns": FEATURES,
        },
        model_path,
    )
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

original_df = real_estate_df.copy()
data_types = {column: str(dtype) for column, dtype in original_df.dtypes.items()}
missing_values = {column: int(value) for column, value in original_df.isna().sum().items()}
descriptive_statistics = json.loads(original_df.describe().round(2).to_json())
metrics_metadata = json.loads(metrics_df.round(4).to_json(orient="records"))
for result in metrics_metadata:
    result["Accuracy"] = round(result["Accuracy"], 2)
feature_importance_metadata = json.loads(coefficient_df.round(4).to_json(orient="records"))
run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

metadata = {
    "id": "D3",
    "run_timestamp": run_timestamp,
    "name": "Real Estate Valuation",
    "level": "Beginner",
    "domain": "Real Estate",
    "task": "Regression",
    "description": "Predict the price of a house per unit area from real estate valuation attributes.",
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
        "excluded_columns": [ID_COLUMN],
    },
    "preprocessing": {
        "method": "Median Imputation + StandardScaler",
        "input_features": FEATURES,
        "missing_value_method": "Median Imputation",
    },
    "model_evaluation": {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "metrics": ["MAE", "MSE", "RMSE", "R2", "R2 Percentage", "Accuracy"],
        "results": metrics_metadata,
        "recommended_model": best_model_name,
        "selection_rule": "Lowest RMSE on the held-out test set",
        "accuracy_definition": "max(0, 100 - MAPE percentage)",
        "hyperparameter_tuning": {
            "method": "GridSearchCV",
            "cv": 5,
            "scoring": "neg_root_mean_squared_error",
            "best_parameters": best_parameters,
        },
    },
    "feature_importance": {
        "method": "Linear Regression Coefficients",
        "model": "Linear Regression",
        "results": feature_importance_metadata,
        "most_influential_feature": coefficient_df.iloc[0]["Feature"],
    },
    "model_verification": model_verification,
    "output_files": {
        "original_dataset": "DataSets/Real estate valuation data set.xlsx",
        "processed_dataset": "DataSets/Processed/real_estate_regression_results.csv",
        "metrics": "DataSets/Processed/real_estate_metrics.csv",
        "metadata_latest": "Metadata/D3_real_estate.json",
        "feature_columns": "ModelsOutcome/real_estate_feature_columns.joblib",
        "models": {name: str(path.relative_to(BASE_DIR)) for name, path in model_paths.items()},
    },
}

metadata_file = metadata_folder / "D3_real_estate.json"
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
print("\nMetadata Process Completed Successfully!")

# endregion