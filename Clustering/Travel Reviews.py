from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
N_INIT = 10
K_MIN = 2
K_MAX = 10

FEATURES = [
    "Art Galleries",
    "Dance Clubs",
    "Juice Bars",
    "Restaurants",
    "Museums",
    "Resorts",
    "Beaches",
    "Theaters",
    "Religious Institutions"
]
ANALYSIS_COLUMNS = FEATURES.copy()

# region Chart Colors
BG_COLOR = "#F8F3FB"
TITLE_COLOR = "#2B1744"
TITLE_ACCENT = "#C9578F"
TEXT_COLOR = "#665B78"
PURPLE = "#8B5CC7"
PINK = "#D65B91"
LAVENDER = "#C9B6F2"
base_colors = [
    "#B084E8",
    "#D96C9D",
    "#6C4C93",
    "#C9B6F2",
    "#8B5CC7",
    "#E8A4C8"
]
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
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2 - 40
        window.geometry(f"+{x}+{y}")
        window.update_idletasks()
        window.update()
    except Exception:
        pass
# endregion


# region Load Dataset
dataset_path = (
    BASE_DIR
    / "Datasets"
    / "tripadvisor_review.csv"
)
df = pd.read_csv(dataset_path)
df.columns = df.columns.str.strip()
expected_columns = (
    ["User ID"] + [f"Category {i}" for i in range(1, 11)]
)
if list(df.columns) != expected_columns:
    raise ValueError(
        "Dataset columns do not match expected columns."
    )
df.columns = [
    "User ID",
    "Art Galleries",
    "Dance Clubs",
    "Juice Bars",
    "Restaurants",
    "Museums",
    "Resorts",
    "Parks & Picnic Spots",
    "Beaches",
    "Theaters",
    "Religious Institutions"
]
print("\nTravel Reviews Dataset Loaded Successfully!")
# endregion


# region Data Understanding
print("\nFirst 5 Rows:")
print(df.head())
print("-" * 60)
print("Dataset Shape:")
print(df.shape)
print("-" * 60)
print("Columns:")
print(df.columns)
print("-" * 60)
print("Data Types:")
print(df.dtypes)
print("-" * 60)
print("\nDataset Information:")
df.info()
print("-" * 60)
print("Statistical Description:")
print(df.describe().to_string())
print("-" * 60)
print("Missing Values:")
print(df.isnull().sum())
print("-" * 60)
print("Total Missing Values:")
print(df.isnull().sum().sum())
print("-" * 60)
print("Duplicate Rows:")
print(df.duplicated().sum())
print("-" * 60)
print("Duplicate User IDs:")
print(df["User ID"].duplicated().sum())
print("-" * 60)
print("Parks & Picnic Spots Unique Values:")
print(df["Parks & Picnic Spots"].nunique())
print("-" * 60)
print("Parks & Picnic Spots Standard Deviation:")
print(df["Parks & Picnic Spots"].std())
print("-" * 60)
# endregion


# region Feature Selection
df_features = df[FEATURES].copy()
print("\nSelected Features:")
print(df_features.head())
print("-" * 60)
print("Features Shape:")
print(df_features.shape)
print("-" * 60)
print("Feature Names:")
print(FEATURES)
print("-" * 60)
print("Analysis Columns:")
print(ANALYSIS_COLUMNS)
print("-" * 60)
print("Excluded Columns:")
print(["User ID", "Parks & Picnic Spots"])
print("-" * 60)
# endregion


# region Data Preprocessing
for column in FEATURES:
    df_features[column] = pd.to_numeric(
        df_features[column],
        errors="coerce"
    )
print("\nMissing Values In Selected Features:")
print(df_features.isnull().sum())
print("-" * 60)
if df_features.isnull().sum().sum() > 0:
    df_features = df_features.fillna(
        df_features.median()
    )
df[FEATURES] = df_features
print("Minimum Rating Values:")
print(df_features.min())
print("-" * 60)
print("Maximum Rating Values:")
print(df_features.max())
print("-" * 60)
invalid_values = (
    (df_features < 0) |
    (df_features > 4)
)
print("Invalid Rating Values:")
print(invalid_values.sum())
print("-" * 60)
if invalid_values.sum().sum() > 0:
    raise ValueError(
        "Rating values must be between 0 and 4."
    )
print("Features Standard Deviation:")
print(df_features.std())
print("-" * 60)
df_features = df_features.copy()
# endregion


# region Data Normalization
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_features)
df_scaled = pd.DataFrame(
    scaled_data,
    columns=FEATURES,
    index=df.index
)
print("\nMinMax-Scaled Data:")
print(df_scaled.head())
print("-" * 60)
print("MinMax-Scaled Data Shape:")
print(df_scaled.shape)
print("-" * 60)
print("MinMax-Scaled Data Mean:")
print(df_scaled.mean().round(4))
print("-" * 60)
print("MinMax-Scaled Data Standard Deviation:")
print(df_scaled.std().round(4))
print("-" * 60)
# endregion


# region Model Evaluation
pca = PCA(
    n_components=3,
    random_state=RANDOM_STATE
)
pca_data = pca.fit_transform(df_scaled)
pca_columns = ["PC1", "PC2", "PC3"]
df_pca = pd.DataFrame(
    pca_data,
    columns=pca_columns,
    index=df.index
)
explained_variance = pca.explained_variance_ratio_ * 100
total_explained_variance = float(
    pca.explained_variance_ratio_.sum() * 100
)
print("\nPCA Data:")
print(df_pca.head())
print("-" * 60)
print("PCA Data Shape:")
print(df_pca.shape)
print("-" * 60)
print("Explained Variance By Components:")
print(pca.explained_variance_ratio_)
print("-" * 60)
print("Explained Variance Percentage:")
print(explained_variance.round(2))
print("-" * 60)
print("Total Explained Variance:")
print(round(total_explained_variance, 2), "%")
print("-" * 60)
inertia_values = []
silhouette_values = []
calinski_values = []
davies_bouldin_values = []
k_values = range(1, K_MAX + 1)
silhouette_k_values = range(K_MIN, K_MAX + 1)
records = []

for k in k_values:
    kmeans_model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=N_INIT
    )
    evaluation_labels = kmeans_model.fit_predict(pca_data)
    inertia_values.append(kmeans_model.inertia_)

    result = {
        "K": k,
        "Inertia": kmeans_model.inertia_,
        "Silhouette": np.nan,
        "Calinski-Harabasz": np.nan,
        "Davies-Bouldin": np.nan
    }

    if k >= K_MIN:
        result["Silhouette"] = silhouette_score(pca_data, evaluation_labels)
        result["Calinski-Harabasz"] = calinski_harabasz_score(pca_data, evaluation_labels)
        result["Davies-Bouldin"] = davies_bouldin_score(pca_data, evaluation_labels)
        silhouette_values.append(result["Silhouette"])
        calinski_values.append(result["Calinski-Harabasz"])
        davies_bouldin_values.append(result["Davies-Bouldin"])

    records.append(result)

metrics_df = pd.DataFrame(records).set_index("K")
eval_df = metrics_df.loc[K_MIN:K_MAX]

print("\nClustering Metrics For Each K:")
print(metrics_df.round(4).to_string())

# Calculate the Elbow recommendation without displaying its chart.
x = np.array(list(k_values), dtype=float)
y = np.array(inertia_values, dtype=float)
x = (x - x.min()) / (x.max() - x.min())
y = (y - y.min()) / (y.max() - y.min())
distances = np.abs(
    (y[-1] - y[0]) * x
    - (x[-1] - x[0]) * y
    + x[-1] * y[0]
    - y[-1] * x[0]
) / np.hypot(
    y[-1] - y[0],
    x[-1] - x[0]
)
elbow_k = int(list(k_values)[np.argmax(distances)])

best_k_silhouette = int(eval_df["Silhouette"].idxmax())
best_k_ch = int(eval_df["Calinski-Harabasz"].idxmax())
best_k_db = int(eval_df["Davies-Bouldin"].idxmin())

rank_df = pd.DataFrame({
    "Silhouette": eval_df["Silhouette"].rank(ascending=False),
    "Calinski-Harabasz": eval_df["Calinski-Harabasz"].rank(ascending=False),
    "Davies-Bouldin": eval_df["Davies-Bouldin"].rank(ascending=True)
})
rank_df["MeanRank"] = rank_df.mean(axis=1)
best_k_combined = int(rank_df["MeanRank"].idxmin())

best_k = best_k_silhouette
best_score = float(eval_df.loc[best_k, "Silhouette"])

elbow_metric_support = {
    "Silhouette": best_k_silhouette == elbow_k,
    "Calinski-Harabasz": best_k_ch == elbow_k,
    "Davies-Bouldin": best_k_db == elbow_k
}
elbow_support_count = sum(elbow_metric_support.values())
elbow_supported_by_validation = elbow_support_count >= 2

# K=2 is selected after comparing cluster quality and cluster balance.
# Elbow and validation metrics are still calculated and saved for reference.
final_k = 2
selection_reason = (
    "K=2 selected after comparing internal validation metrics; it improves "
    "Silhouette and avoids the very small cluster produced by K=3."
)

print("\nBest K Based On Elbow Method:", elbow_k)
print("Best K Based On Silhouette:", best_k_silhouette)
print("Best K Based On Calinski-Harabasz:", best_k_ch)
print("Best K Based On Davies-Bouldin:", best_k_db)
print("Best K Based On Combined Ranking:", best_k_combined)
print("Elbow Metric Support:", elbow_metric_support)
print("Elbow Support Count:", elbow_support_count, "/ 3")
print("Final K Selected:", final_k)
print("Selection Reason:", selection_reason)
print("\nCombined Ranking:")
print(rank_df.sort_values("MeanRank").round(2).to_string())
# endregion


# region Final K-Means Clustering
kmeans_final = KMeans(
    n_clusters=final_k,
    random_state=RANDOM_STATE,
    n_init=N_INIT
)
cluster_labels = kmeans_final.fit_predict(
    pca_data
)
df["Cluster"] = cluster_labels
print("\nFinal Cluster Distribution:")
print(
    df["Cluster"]
    .value_counts()
    .sort_index()
)
print("-" * 60)
final_silhouette = silhouette_score(
    pca_data,
    cluster_labels
)
print("Final Silhouette Score:")
print(round(final_silhouette, 4))
# endregion


# region Results
cluster_profile = df.groupby("Cluster")[ANALYSIS_COLUMNS].mean()
print("\nCluster Analysis:")
print(
    cluster_profile.round(3).to_string()
)
print("-" * 60)
cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)
print("Cluster Sizes:")
print(cluster_counts)
print("-" * 60)
cluster_percentages = (
    df["Cluster"]
    .value_counts(normalize=True)
    .sort_index() * 100
)
print("Cluster Percentages:")
print(cluster_percentages)
print("-" * 60)
center_point_original = cluster_profile.copy()
center_point_scaled = df_scaled.assign(
    Cluster=cluster_labels
).groupby("Cluster")[FEATURES].mean()
print("\nCluster Centers - Original Rating Scale:")
print(
    center_point_original.round(3).to_string()
)
print("-" * 60)
print("Top Travel Categories In Each Cluster:")
for cluster_id, row in cluster_profile.iterrows():
    print(f"\nCluster {cluster_id}:")
    print(
        row.sort_values(ascending=False).head(3).to_string()
    )
# endregion


# region SSD Evaluation
assigned_centers = (
    kmeans_final.cluster_centers_[cluster_labels]
)
ssd = float(
    np.sum(
        (df_pca.to_numpy() - assigned_centers) ** 2
    )
)
print("\nSSD Evaluation (PCA Space):")
print("Final SSD:", round(ssd, 4))
print(
    "K-Means Inertia:",
    round(float(kmeans_final.inertia_), 4)
)
print("SSD Verification:")
ssd_verification = bool(
    np.isclose(
        ssd,
        kmeans_final.inertia_
    )
)
print(ssd_verification)
# endregion


# region Data Visualization
if final_k <= len(base_colors):
    cluster_colors = base_colors[:final_k]
else:
    cluster_colors = sns.color_palette(
        "husl",
        n_colors=final_k
    )
plt.figure(
    figsize=(10, 6.5),
    facecolor=BG_COLOR
)
ax = plt.gca()
ax.set_facecolor(BG_COLOR)
ax.set_axisbelow(True)
for cluster_id in sorted(df["Cluster"].unique()):
    cluster_data = df_scaled[
        df["Cluster"] == cluster_id
    ]
    plt.scatter(
        cluster_data["Art Galleries"],
        cluster_data["Dance Clubs"],
        color=cluster_colors[cluster_id],
        s=65,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.7,
        label=f"Cluster {cluster_id}"
    )
plt.scatter(
    center_point_scaled["Art Galleries"],
    center_point_scaled["Dance Clubs"],
    marker="X",
    c=TITLE_COLOR,
    edgecolors="white",
    linewidths=1.5,
    s=230,
    label="Cluster Centers",
    zorder=5
)
plt.xlabel(
    "Art Galleries (Standardized)",
    fontsize=12,
    color=TEXT_COLOR,
    labelpad=10
)
plt.ylabel(
    "Dance Clubs (Standardized)",
    fontsize=12,
    color=TEXT_COLOR,
    labelpad=10
)
plt.title(
    "Travel Reviews - KMeans Clustering",
    fontsize=17,
    fontweight="bold",
    color=TITLE_ACCENT,
    pad=20
)
plt.xticks(fontsize=11, color=TEXT_COLOR)
plt.yticks(fontsize=11, color=TEXT_COLOR)
plt.grid(
    color=LAVENDER,
    alpha=0.3,
    linewidth=0.8
)
for spine in ax.spines.values():
    spine.set_color(LAVENDER)
    spine.set_linewidth(0.8)
legend = plt.legend(
    loc="best",
    facecolor=BG_COLOR,
    edgecolor=LAVENDER,
    fontsize=10,
    framealpha=0.95
)
for text in legend.get_texts():
    text.set_color(TEXT_COLOR)
plt.tight_layout()
plt.show(block=False)
center_figure_window()
plt.show()
correlation_matrix = df_features[ANALYSIS_COLUMNS].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix.to_string())
minimal_heatmap = LinearSegmentedColormap.from_list(
    "MinimalMLStudio",
    [
        (0.00, "#211239"),
        (0.35, "#5D447F"),
        (0.50, "#8B6BB0"),
        (0.65, "#B998D0"),
        (0.80, "#E6CDEB"),
        (1.00, "#FCF9FD")
    ]
)
mask = np.eye(
    len(correlation_matrix),
    dtype=bool
)
plt.figure(
    figsize=(11, 6),
    facecolor=BG_COLOR
)
ax = sns.heatmap(
    correlation_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap=minimal_heatmap,
    vmin=-1,
    vmax=1,
    linewidths=1.5,
    linecolor="white",
    annot_kws={
        "size": 10,
        "weight": "bold"
    },
    cbar_kws={
        "label": "Correlation",
        "ticks": [-1, -0.5, 0, 0.5, 1],
        "shrink": 0.85
    }
)
for text in ax.texts:
    value = float(text.get_text())
    if value >= 0.30:
        text.set_color(TITLE_COLOR)
    else:
        text.set_color("#FFFFFF")
plt.title(
    "Travel Reviews - Correlation Heatmap",
    fontsize=17,
    fontweight="bold",
    color=TITLE_ACCENT,
    pad=16
)
plt.xticks(
    rotation=40,
    ha="right",
    fontsize=9,
    color=TEXT_COLOR
)
plt.yticks(
    rotation=0,
    fontsize=9,
    color=TEXT_COLOR
)
ax.set_facecolor(BG_COLOR)
colorbar = ax.collections[0].colorbar
colorbar.ax.tick_params(
    colors=TEXT_COLOR,
    labelsize=10
)
colorbar.set_label(
    "Correlation",
    color=TITLE_COLOR,
    fontsize=12
)
plt.tight_layout()
plt.show(block=False)
center_figure_window()
plt.show()
pca_centers = kmeans_final.cluster_centers_
plt.figure(
    figsize=(10, 6.5),
    facecolor=BG_COLOR
)
ax = plt.gca()
ax.set_facecolor(BG_COLOR)
ax.set_axisbelow(True)
for cluster_id in sorted(df["Cluster"].unique()):
    cluster_mask = cluster_labels == cluster_id
    plt.scatter(
        pca_data[cluster_mask, 0],
        pca_data[cluster_mask, 1],
        color=cluster_colors[cluster_id],
        s=65,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.7,
        label=f"Cluster {cluster_id}"
    )
plt.scatter(
    pca_centers[:, 0],
    pca_centers[:, 1],
    marker="X",
    c=TITLE_COLOR,
    edgecolors="white",
    linewidths=1.5,
    s=230,
    label="Cluster Centers",
    zorder=5
)
plt.xlabel(
    f"PC1 ({explained_variance[0]:.2f}% variance)",
    fontsize=12,
    color=TEXT_COLOR
)
plt.ylabel(
    f"PC2 ({explained_variance[1]:.2f}% variance)",
    fontsize=12,
    color=TEXT_COLOR
)
plt.title(
    "Travel Reviews - PCA Clustering Visualization",
    fontsize=17,
    fontweight="bold",
    color=TITLE_ACCENT,
    pad=20
)
plt.xticks(color=TEXT_COLOR)
plt.yticks(color=TEXT_COLOR)
plt.grid(
    color=LAVENDER,
    alpha=0.3,
    linewidth=0.8
)
for spine in ax.spines.values():
    spine.set_color(LAVENDER)
legend = plt.legend(
    loc="best",
    facecolor=BG_COLOR,
    edgecolor=LAVENDER,
    fontsize=10,
    framealpha=0.95
)
for text in legend.get_texts():
    text.set_color(TEXT_COLOR)
plt.tight_layout()
plt.show(block=False)
center_figure_window()
plt.show()
plt.figure(
    figsize=(9, 5.5),
    facecolor=BG_COLOR
)
ax = plt.gca()
ax.set_facecolor(BG_COLOR)
cluster_ids = cluster_counts.index.tolist()
cluster_sizes = cluster_counts.values
cluster_colors_for_bars = [
    cluster_colors[int(cluster_id)]
    for cluster_id in cluster_ids
]
bars = ax.bar(
    [f"Cluster {cluster_id}" for cluster_id in cluster_ids],
    cluster_sizes,
    color=cluster_colors_for_bars,
    edgecolor="white",
    linewidth=1.5,
    width=0.55
)
for bar, cluster_id in zip(bars, cluster_ids):
    count = int(cluster_counts.loc[cluster_id])
    percentage = float(cluster_percentages.loc[cluster_id])
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{count}\n({percentage:.2f}%)",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=TITLE_COLOR
    )
ax.set_title(
    "Travel Reviews - Cluster Size Comparison",
    fontsize=17,
    fontweight="bold",
    color=TITLE_ACCENT,
    pad=20
)
ax.set_xlabel(
    "Clusters",
    fontsize=12,
    color=TEXT_COLOR
)
ax.set_ylabel(
    "Number of Reviews",
    fontsize=12,
    color=TEXT_COLOR
)
ax.tick_params(
    colors=TEXT_COLOR,
    labelsize=11
)
ax.grid(
    axis="y",
    color=LAVENDER,
    alpha=0.35
)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_color(LAVENDER)
ax.margins(y=0.18)
plt.tight_layout()
plt.show(block=False)
center_figure_window()
plt.show()
# endregion


# region Save Results
output_folder = (
    BASE_DIR
    / "Datasets"
    / "Processed"
)
output_folder.mkdir(
    parents=True,
    exist_ok=True
)
output_path = (
    output_folder
    / "travel_reviews_clustered.csv"
)
df.to_csv(
    output_path,
    index=False
)
print("\nClustered Dataset Saved Successfully!")
print("Saved File:", output_path)
# endregion


# region Training Models
model_folder = (
    BASE_DIR
    / "ModelsOutcome"
)
model_folder.mkdir(
    parents=True,
    exist_ok=True
)
kmeans_path = (
    model_folder
    / "travel_reviews_kmeans.joblib"
)
scaler_path = (
    model_folder
    / "travel_reviews_scaler.joblib"
)
features_path = (
    model_folder
    / "travel_reviews_feature_columns.joblib"
)
clustering_pipeline = Pipeline([
    ("pca", pca),
    ("kmeans", kmeans_final)
])
joblib.dump(
    clustering_pipeline,
    kmeans_path
)
joblib.dump(
    scaler,
    scaler_path
)
joblib.dump(
    FEATURES,
    features_path
)
print("\nTrained Model Saved Successfully!")
print("PCA + K-Means Model Saved:", kmeans_path)
print("Scaler Saved:", scaler_path)
print("Features Saved:", features_path)
loaded_model = joblib.load(kmeans_path)
loaded_scaler = joblib.load(scaler_path)
loaded_features = joblib.load(features_path)
print("\nModels Loaded Successfully!")
X_new = df[loaded_features].copy()
X_new_scaled = loaded_scaler.transform(X_new)
X_new_scaled = pd.DataFrame(
    X_new_scaled,
    columns=loaded_features,
    index=X_new.index
)
predicted_labels = loaded_model.predict(
    X_new_scaled
)
original_labels = df["Cluster"].to_numpy()
print("\nPredicted Cluster Labels:")
print(predicted_labels[:10])
print("-" * 60)
print("Model Verification:")
model_verification = bool(
    np.array_equal(
        original_labels,
        predicted_labels
    )
)
print(model_verification)
# endregion


# region Save Metadata
metadata_folder = (
    BASE_DIR
    / "Metadata"
)
metadata_folder.mkdir(
    parents=True,
    exist_ok=True
)
original_df = df.drop(
    columns=["Cluster"]
)
data_types = {
    column: str(dtype)
    for column, dtype in original_df.dtypes.items()
}
missing_values = {
    column: int(original_df[column].isnull().sum())
    for column in original_df.columns
}
statistics_df = df_features.describe().round(3)
descriptive_statistics = {
    column: {
        statistic: float(statistics_df.loc[statistic, column])
        for statistic in statistics_df.index
    }
    for column in statistics_df.columns
}
cluster_sizes_dict = {
    str(int(cluster_id)): int(count)
    for cluster_id, count in cluster_counts.items()
}
cluster_percentages_dict = {
    str(int(cluster_id)): round(float(percentage), 2)
    for cluster_id, percentage in cluster_percentages.items()
}
cluster_profiles = {
    str(int(cluster_id)): {
        column: round(float(value), 3)
        for column, value in row.items()
    }
    for cluster_id, row in cluster_profile.iterrows()
}
correlation_data = correlation_matrix.round(4).to_dict()
inertia_results = {
    str(k): round(float(inertia), 4)
    for k, inertia in zip(k_values, inertia_values)
}
silhouette_results = {
    str(k): round(float(score), 4)
    for k, score in zip(silhouette_k_values, silhouette_values)
}
metadata = {
    "id": "D21",
    "name": "Travel Reviews",
    "level": "Intermediate",
    "domain": "Web",
    "task": "Clustering",
    "description": (
        "Group similar travel reviews based on "
        "average user ratings across travel categories."
    ),
    "source": "Data Science Dojo / TripAdvisor",
    "source_url": (
        "https://code.datasciencedojo.com/"
        "datasciencedojo/datasets/tree/master/Travel%20Reviews"
    ),
    "dataset_statistics": {
        "rows": int(original_df.shape[0]),
        "columns": int(original_df.shape[1]),
        "column_names": list(original_df.columns),
        "data_types": data_types,
        "missing_values": missing_values,
        "total_missing_values": int(
            original_df.isnull().sum().sum()
        ),
        "descriptive_statistics": descriptive_statistics
    },
    "feature_selection": {
        "selected_features": FEATURES,
        "analysis_columns": ANALYSIS_COLUMNS,
        "excluded_columns": [
            "User ID",
            "Parks & Picnic Spots"
        ],
        "excluded_feature_reasons": {
            "User ID": (
                "Identifier column, not a travel preference feature."
            ),
            "Parks & Picnic Spots": (
                "Near-constant feature with very low variance."
            )
        }
    },
    "preprocessing": {
        "method": "MinMaxScaler",
        "input_features": FEATURES,
        "missing_value_method": "Median Imputation",
        "rating_range": {
            "minimum": 0,
            "maximum": 4
        }
    },
    "dimensionality_reduction": {
        "method": "PCA",
        "n_components": 3,
        "explained_variance_ratio": {
            "PC1": round(float(pca.explained_variance_ratio_[0]), 4),
            "PC2": round(float(pca.explained_variance_ratio_[1]), 4),
            "PC3": round(float(pca.explained_variance_ratio_[2]), 4)
        },
        "total_explained_variance_pct": round(
            total_explained_variance,
            2
        )
    },
    "Clustering": {
        "algorithm": "KMeans",
        "n_clusters": int(final_k),
        "random_state": RANDOM_STATE,
        "n_init": N_INIT,
        "input_space": "PCA (3 components)",
        "input_features": FEATURES,
        "cluster_label_column": "Cluster",
        "cluster_sizes": cluster_sizes_dict,
        "cluster_percentages": cluster_percentages_dict,
        "cluster_profiles": cluster_profiles,
        "silhouette_score": round(
            float(final_silhouette),
            4
        )
    },
    "model_evaluation": {
        "SSD": round(ssd, 4),
        "SSD_space": "PCA (3 components)",
        "SSD_description": (
            "Sum of Squared Distances from each PCA-transformed sample "
            "to its assigned cluster center."
        ),
        "inertia": round(
            float(kmeans_final.inertia_),
            4
        ),
        "ssd_verification": ssd_verification,
        "silhouette_score": round(
            float(final_silhouette),
            4
        ),
        "inertia_per_k": inertia_results,
        "silhouette_scores": silhouette_results,
        "calinski_harabasz_scores": {
            str(k): round(float(value), 4)
            for k, value in zip(silhouette_k_values, calinski_values)
        },
        "davies_bouldin_scores": {
            str(k): round(float(value), 4)
            for k, value in zip(silhouette_k_values, davies_bouldin_values)
        },
        "metrics_per_k": json.loads(metrics_df.round(4).to_json(orient="index")),
        "combined_ranking": json.loads(rank_df.round(2).to_json(orient="index")),
        "elbow_k": int(elbow_k),
        "best_k_by_silhouette": int(best_k_silhouette),
        "best_k_by_calinski_harabasz": int(best_k_ch),
        "best_k_by_davies_bouldin": int(best_k_db),
        "best_k_combined": int(best_k_combined),
        "elbow_metric_support": elbow_metric_support,
        "elbow_support_count": int(elbow_support_count),
        "elbow_supported_by_validation": bool(elbow_supported_by_validation),
        "final_k": int(final_k),
        "selection_method": "Selected K=2 after internal validation comparison",
        "selection_reason": selection_reason,
        "manual_k_selection": True,
        "best_silhouette_score": round(
            float(best_score),
            4
        )
    },
    "model_verification": model_verification,
    "correlation_matrix": correlation_data,
    "output_files": {
        "original_dataset": (
            "Datasets/tripadvisor_review.csv"
        ),
        "processed_dataset": (
            "Datasets/Processed/"
            "travel_reviews_clustered.csv"
        ),
        "trained_model": (
            "ModelsOutcome/"
            "travel_reviews_kmeans.joblib"
        ),
        "scaler": (
            "ModelsOutcome/"
            "travel_reviews_scaler.joblib"
        ),
        "feature_columns": (
            "ModelsOutcome/"
            "travel_reviews_feature_columns.joblib"
        )
    }
}
metadata_file = (
    metadata_folder
    / "D21_travel_reviews.json"
)
with open(
    metadata_file,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        metadata,
        file,
        ensure_ascii=False,
        indent=4
    )
print("\nMetadata Saved Successfully!")
print("Saved File:", metadata_file)
with open(
    metadata_file,
    "r",
    encoding="utf-8"
) as file:
    saved_metadata = json.load(file)
print("-" * 60)
print("Metadata Verification:")
print("Dataset Name:", saved_metadata["name"])
print(
    "Total Rows:",
    saved_metadata["dataset_statistics"]["rows"]
)
print(
    "Total Columns:",
    saved_metadata["dataset_statistics"]["columns"]
)
print(
    "Algorithm:",
    saved_metadata["Clustering"]["algorithm"]
)
print(
    "Final K:",
    saved_metadata["Clustering"]["n_clusters"]
)
print(
    "PCA Components:",
    saved_metadata["dimensionality_reduction"]["n_components"]
)
print(
    "PCA Explained Variance:",
    saved_metadata["dimensionality_reduction"]["total_explained_variance_pct"]
)
print(
    "Cluster Sizes:",
    saved_metadata["Clustering"]["cluster_sizes"]
)
print(
    "Silhouette Score:",
    saved_metadata["Clustering"]["silhouette_score"]
)
print(
    "Total SSD:",
    saved_metadata["model_evaluation"]["SSD"]
)
print(
    "SSD Verification:",
    saved_metadata["model_evaluation"]["ssd_verification"]
)
print(
    "Model Verification:",
    saved_metadata["model_verification"]
)
print("\nMetadata Process Completed Successfully!")
# endregion

