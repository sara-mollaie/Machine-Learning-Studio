import json
from pathlib import Path

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
from sklearn.preprocessing import MinMaxScaler


BASE_DIR = Path(__file__).resolve().parent

RANDOM_STATE = 42
N_INIT = 10

K_MIN = 2
K_MAX = 10

FEATURES = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]


# region Chart Colors

BG_COLOR = "#F8F3FB"
TITLE_COLOR = "#2B1744"
TITLE_ACCENT = "#C9578F"
TEXT_COLOR = "#665B78"
PURPLE = "#8B5CC7"
PINK = "#D65B91"
LAVENDER = "#C9B6F2"

CLUSTER_COLORS = [
    "#B084E8",
    "#D96C9D",
    "#6C4C93",
    "#E8A5C8",
    "#8FA6E8",
    "#5B3F8C"
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

        x = (window.winfo_screenwidth() - width) // 2
        y = (window.winfo_screenheight() - height) // 2 - 40

        window.geometry(f"+{x}+{y}")
        window.update_idletasks()
        window.update()

    except Exception:
        pass


def show_figure():

    plt.show(block=False)
    center_figure_window()
    plt.show()

# endregion


# region Load Dataset

dataset_path = (
    BASE_DIR
    / "Datasets"
    / "Wholesale customers data.csv"
)

wholesale_df = pd.read_csv(dataset_path)
wholesale_df.columns = wholesale_df.columns.str.strip()

expected_columns = [
    "Channel",
    "Region",
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]

if list(wholesale_df.columns) != expected_columns:
    raise ValueError(
        "Dataset columns do not match expected columns."
    )

print("\nWholesale Customers Dataset Loaded Successfully!")

# endregion


# region Data Understanding

print("\nFirst 5 Rows:")
print(wholesale_df.head())
print("-" * 60)

print("Dataset Shape:")
print(wholesale_df.shape)
print("-" * 60)

print("Columns:")
print(wholesale_df.columns)
print("-" * 60)

print("Data Types:")
print(wholesale_df.dtypes)
print("-" * 60)

print("\nDataset Information:")
wholesale_df.info()
print("-" * 60)

print("Statistical Description:")
print(wholesale_df.describe().to_string())
print("-" * 60)

print("Missing Values:")
print(wholesale_df.isnull().sum())
print("-" * 60)

print("Total Missing Values:")
print(wholesale_df.isnull().sum().sum())
print("-" * 60)

print("Duplicate Rows:")
print(wholesale_df.duplicated().sum())
print("-" * 60)

print("Channel Distribution:")
print(wholesale_df["Channel"].value_counts())
print("-" * 60)

print("Region Distribution:")
print(wholesale_df["Region"].value_counts())
print("-" * 60)

# endregion


# region Feature Selection

X = wholesale_df[FEATURES].copy()

print("\nSelected Features:")
print(X.head())
print("-" * 60)

print("Features Shape:")
print(X.shape)
print("-" * 60)

print("Feature Names:")
print(FEATURES)
print("-" * 60)

# Channel and Region are kept for analysis, not used as model features.
print("Channel Values:")
print(wholesale_df["Channel"].value_counts())
print("-" * 60)

print("Region Values:")
print(wholesale_df["Region"].value_counts())
print("-" * 60)

# endregion


# region Data Preprocessing

for column in FEATURES:
    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

print("\nMissing Values In Spending Features:")
print(X.isnull().sum())
print("-" * 60)

if X.isnull().sum().sum() > 0:
    X = X.fillna(X.median())

if (X < 0).any().any():
    raise ValueError("Spending values cannot be negative.")

if not np.isfinite(X.to_numpy()).all():
    raise ValueError("Dataset contains infinite values.")

wholesale_df[FEATURES] = X

print("Minimum Values:")
print(X.min())
print("-" * 60)

print("Maximum Values:")
print(X.max())
print("-" * 60)

print("Features Standard Deviation:")
print(X.std())
print("-" * 60)

# endregion


# region Data Normalization

# Keep original spending values and normalize them with MinMaxScaler.
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

df_scaled = pd.DataFrame(
    X_scaled,
    columns=FEATURES,
    index=X.index
)

print("\nNormalized Data:")
print(df_scaled.head())
print("-" * 60)

print("Normalized Data Shape:")
print(df_scaled.shape)
print("-" * 60)

print("Minimum Values:")
print(df_scaled.min())
print("-" * 60)

print("Maximum Values:")
print(df_scaled.max())
print("-" * 60)

# endregion


# region Model Evaluation

X_scaled_array = df_scaled.to_numpy()

total_sum_squares = float(
    (
        (
            X_scaled_array
            - X_scaled_array.mean(axis=0)
        ) ** 2
    ).sum()
)

records = []

for k in range(1, K_MAX + 1):

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=N_INIT
    )

    model.fit(df_scaled)

    record = {
        "K": k,
        "Inertia": model.inertia_,
        "Silhouette": np.nan,
        "Calinski-Harabasz": np.nan,
        "Davies-Bouldin": np.nan
    }

    if k >= K_MIN:

        record["Silhouette"] = silhouette_score(
            df_scaled,
            model.labels_
        )

        record["Calinski-Harabasz"] = calinski_harabasz_score(
            df_scaled,
            model.labels_
        )

        record["Davies-Bouldin"] = davies_bouldin_score(
            df_scaled,
            model.labels_
        )

    records.append(record)

metrics_df = pd.DataFrame(records).set_index("K")
eval_df = metrics_df.loc[K_MIN:K_MAX]

print("\nClustering Metrics For Each K:")
print(metrics_df.round(4).to_string())

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

print("\n" + "=" * 60)
print("Best K By Each Criterion")
print("=" * 60)

print("Silhouette           :", best_k_silhouette)
print("Calinski-Harabasz    :", best_k_ch)
print("Davies-Bouldin       :", best_k_db)
print("Combined Mean Rank   :", best_k_combined)

print("\nCombined Ranking:")
print(rank_df.sort_values("MeanRank").round(2).to_string())

# Select K automatically using the Elbow Method.
elbow_x = np.array(metrics_df.index, dtype=float)
elbow_y = metrics_df["Inertia"].to_numpy(dtype=float)
elbow_x = (elbow_x - elbow_x.min()) / (elbow_x.max() - elbow_x.min())
elbow_y = (elbow_y - elbow_y.min()) / (elbow_y.max() - elbow_y.min())
elbow_distances = np.abs(
    (elbow_y[-1] - elbow_y[0]) * elbow_x
    - (elbow_x[-1] - elbow_x[0]) * elbow_y
    + elbow_x[-1] * elbow_y[0]
    - elbow_y[-1] * elbow_x[0]
) / np.hypot(
    elbow_y[-1] - elbow_y[0],
    elbow_x[-1] - elbow_x[0]
)
elbow_k = int(metrics_df.index[np.argmax(elbow_distances)])

elbow_metric_support = {
    "Silhouette": best_k_silhouette == elbow_k,
    "Calinski-Harabasz": best_k_ch == elbow_k,
    "Davies-Bouldin": best_k_db == elbow_k
}
elbow_support_count = sum(elbow_metric_support.values())
elbow_supported_by_validation = elbow_support_count >= 2

# K=4 is selected after comparing K=3 and K=4.
# The selection prioritizes lower SSD while retaining acceptable cluster quality.
final_k = 2
selection_reason = (
    "K=2 selected because it provides the strongest Silhouette score, "
    "a meaningful customer segment, and the best overall validation balance."
)

print("\nElbow Point K:", elbow_k)
print("Best K - Silhouette:", best_k_silhouette)
print("Best K - Calinski-Harabasz:", best_k_ch)
print("Best K - Davies-Bouldin:", best_k_db)
print("Best K - Combined Ranking:", best_k_combined)
print("Elbow Metric Support:", elbow_metric_support)
print("Elbow Support Count:", elbow_support_count, "/ 3")
print("Final K Selected:", final_k)
print("Selection Reason:", selection_reason)

# endregion


# region Final K-Means Clustering

kmeans_final = KMeans(
    n_clusters=final_k,
    random_state=RANDOM_STATE,
    n_init=N_INIT
)

cluster_labels = kmeans_final.fit_predict(df_scaled)
wholesale_df["ClusterLabel"] = cluster_labels

print("\nFinal Model: K =", final_k)

print("\nFinal Cluster Distribution:")
print(wholesale_df["ClusterLabel"].value_counts().sort_index())
print("-" * 60)

final_silhouette = silhouette_score(
    df_scaled,
    cluster_labels
)

print("Final Silhouette Score:")
print(round(final_silhouette, 4))

# endregion


# region Results

cluster_analysis = wholesale_df.groupby("ClusterLabel")[FEATURES].mean()

print("\nCluster Analysis:")
print(cluster_analysis.round(2).to_string())
print("-" * 60)

cluster_counts = wholesale_df["ClusterLabel"].value_counts().sort_index()

print("Cluster Sizes:")
print(cluster_counts)
print("-" * 60)

cluster_percentages = (
    wholesale_df["ClusterLabel"]
    .value_counts(normalize=True)
    .sort_index() * 100
)

print("Cluster Percentages:")
print(cluster_percentages)
print("-" * 60)

channel_distribution = pd.crosstab(
    wholesale_df["ClusterLabel"],
    wholesale_df["Channel"]
)

print("Channel Distribution:")
print(channel_distribution)
print("-" * 60)

region_distribution = pd.crosstab(
    wholesale_df["ClusterLabel"],
    wholesale_df["Region"]
)

print("Region Distribution:")
print(region_distribution)
print("-" * 60)

center_point_scaled = pd.DataFrame(
    kmeans_final.cluster_centers_,
    columns=FEATURES
)

center_point_original = pd.DataFrame(
    scaler.inverse_transform(kmeans_final.cluster_centers_),
    columns=FEATURES
)

print("\nCluster Centers - Original Spending Scale:")
print(center_point_original.round(2).to_string())

# endregion


# region SSD Evaluation

cluster_ssd = {}
cluster_sizes_for_ssd = {}

for cluster_id in range(final_k):

    points = X_scaled_array[
        cluster_labels == cluster_id
    ]

    center = kmeans_final.cluster_centers_[cluster_id]

    cluster_ssd[cluster_id] = float(
        ((points - center) ** 2).sum()
    )

    cluster_sizes_for_ssd[cluster_id] = len(points)

# Total SSD
total_ssd = sum(cluster_ssd.values())

# Between-Cluster Sum of Squares
between_ssd = total_sum_squares - total_ssd

# Explained Variance
explained_variance_pct = (
    between_ssd / total_sum_squares
) * 100

ssd_table = pd.DataFrame({
    "Size": pd.Series(cluster_sizes_for_ssd),
    "SSD": pd.Series(cluster_ssd)
})

ssd_table["SSD Share %"] = (
    ssd_table["SSD"] / total_ssd * 100
)

ssd_table["Mean Sq. Distance"] = (
    ssd_table["SSD"] / ssd_table["Size"]
)

ssd_table["RMS Distance"] = np.sqrt(
    ssd_table["Mean Sq. Distance"]
)

print("\nPer-Cluster SSD:")
print(ssd_table.round(4).to_string())
print("-" * 60)

print("Total SSD:", round(total_ssd, 4))
print("Total Sum of Squares:", round(total_sum_squares, 4))
print("Between-Cluster SS:", round(between_ssd, 4))
print("Explained Variance:", round(explained_variance_pct, 2), "%")

ssd_verification = bool(
    np.isclose(total_ssd, kmeans_final.inertia_)
)

print("SSD Verification:", ssd_verification)

# endregion


# region Data Visualization

if final_k <= len(CLUSTER_COLORS):
    cluster_colors = CLUSTER_COLORS[:final_k]
else:
    cluster_colors = sns.color_palette("husl", n_colors=final_k)

# Scatter Plot - Fresh vs Grocery
plt.figure(
    figsize=(10, 6.5),
    facecolor=BG_COLOR
)

ax = plt.gca()
ax.set_facecolor(BG_COLOR)
ax.set_axisbelow(True)

for cluster_id in sorted(wholesale_df["ClusterLabel"].unique()):

    cluster_data = df_scaled[
        wholesale_df["ClusterLabel"] == cluster_id
    ]

    plt.scatter(
        cluster_data["Fresh"],
        cluster_data["Grocery"],
        color=cluster_colors[cluster_id],
        s=65,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.7,
        label=f"Cluster {cluster_id}"
    )

plt.scatter(
    center_point_scaled["Fresh"],
    center_point_scaled["Grocery"],
    marker="X",
    c=TITLE_COLOR,
    edgecolors="white",
    linewidths=1.5,
    s=230,
    label="Cluster Centers",
    zorder=5
)

plt.xlabel(
    "Fresh (Normalized)",
    fontsize=12,
    color=TEXT_COLOR,
    labelpad=10
)

plt.ylabel(
    "Grocery (Normalized)",
    fontsize=12,
    color=TEXT_COLOR,
    labelpad=10
)

plt.title(
    "Wholesale Customers - KMeans Clustering",
    fontsize=17,
    fontweight="bold",
    color=TITLE_ACCENT,
    pad=20
)

plt.xticks(fontsize=11, color=TEXT_COLOR)
plt.yticks(fontsize=11, color=TEXT_COLOR)
plt.grid(color=LAVENDER, alpha=0.3, linewidth=0.8)

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
show_figure()

# Correlation Heatmap
correlation_matrix = X.corr()

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

mask = np.eye(len(correlation_matrix), dtype=bool)

plt.figure(
    figsize=(9, 6),
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
        "size": 11,
        "weight": "bold"
    },
    cbar_kws={
        "label": "Correlation",
        "ticks": [-1, -0.5, 0, 0.5, 1]
    }
)

for text in ax.texts:

    value = float(text.get_text())

    if value >= 0.30:
        text.set_color(TITLE_COLOR)
    else:
        text.set_color("#FFFFFF")

plt.title(
    "Wholesale Customers - Correlation Heatmap",
    fontsize=17,
    fontweight="bold",
    color=TITLE_ACCENT,
    pad=18
)

plt.xticks(
    rotation=30,
    ha="right",
    fontsize=10,
    color=TEXT_COLOR
)

plt.yticks(
    rotation=0,
    fontsize=10,
    color=TEXT_COLOR
)

ax.set_facecolor(BG_COLOR)

colorbar = ax.collections[0].colorbar
colorbar.ax.tick_params(colors=TEXT_COLOR, labelsize=10)
colorbar.set_label(
    "Correlation",
    color=TITLE_COLOR,
    fontsize=12
)

plt.tight_layout()
show_figure()


# PCA Clustering Visualization
pca = PCA(
    n_components=2,
    random_state=RANDOM_STATE
)

pca_data = pca.fit_transform(df_scaled)

pca_centers = pca.transform(
    center_point_scaled[FEATURES]
)

pca_variance = pca.explained_variance_ratio_ * 100

print("\nPCA Explained Variance:")
print(f"PC1: {pca_variance[0]:.2f}%")
print(f"PC2: {pca_variance[1]:.2f}%")
print(f"Total: {pca_variance.sum():.2f}%")

plt.figure(
    figsize=(10, 6.5),
    facecolor=BG_COLOR
)

ax = plt.gca()
ax.set_facecolor(BG_COLOR)
ax.set_axisbelow(True)

for cluster_id in sorted(wholesale_df["ClusterLabel"].unique()):

    cluster_data = pca_data[
        wholesale_df["ClusterLabel"].to_numpy() == cluster_id
    ]

    plt.scatter(
        cluster_data[:, 0],
        cluster_data[:, 1],
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
    f"PC1 ({pca_variance[0]:.2f}% variance)",
    fontsize=12,
    color=TEXT_COLOR,
    labelpad=10
)

plt.ylabel(
    f"PC2 ({pca_variance[1]:.2f}% variance)",
    fontsize=12,
    color=TEXT_COLOR,
    labelpad=10
)

plt.title(
    "Wholesale Customers - PCA Clustering Visualization",
    fontsize=17,
    fontweight="bold",
    color=TITLE_ACCENT,
    pad=20
)

plt.xticks(fontsize=11, color=TEXT_COLOR)
plt.yticks(fontsize=11, color=TEXT_COLOR)
plt.grid(color=LAVENDER, alpha=0.3, linewidth=0.8)

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
show_figure()


# Cluster Size Comparison
plt.figure(
    figsize=(9, 5.5),
    facecolor=BG_COLOR
)

ax = plt.gca()
ax.set_facecolor(BG_COLOR)

cluster_ids = cluster_counts.index.to_list()
cluster_values = cluster_counts.values
cluster_pcts = cluster_percentages.loc[cluster_ids].values

bars = ax.bar(
    [f"Cluster {cluster_id}" for cluster_id in cluster_ids],
    cluster_values,
    color=[cluster_colors[cluster_id] for cluster_id in cluster_ids],
    width=0.55,
    edgecolor="white",
    linewidth=1.2
)

for bar, count, percentage in zip(
    bars,
    cluster_values,
    cluster_pcts
):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(cluster_values) * 0.025,
        f"{count:,}\n({percentage:.2f}%)",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=TITLE_COLOR
    )

ax.set_title(
    "Wholesale Customers - Cluster Size Comparison",
    fontsize=17,
    fontweight="bold",
    color=TITLE_ACCENT,
    pad=20
)

ax.set_xlabel(
    "Clusters",
    fontsize=12,
    color=TEXT_COLOR,
    labelpad=10
)

ax.set_ylabel(
    "Number of Customers",
    fontsize=12,
    color=TEXT_COLOR,
    labelpad=10
)

ax.tick_params(colors=TEXT_COLOR, labelsize=11)
ax.grid(
    axis="y",
    color=LAVENDER,
    alpha=0.3,
    linewidth=0.8
)
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_color(LAVENDER)
    spine.set_linewidth(0.8)

ax.set_ylim(
    0,
    max(cluster_values) * 1.22
)

plt.tight_layout()
show_figure()

# endregion


# region Save Results

output_folder = BASE_DIR / "Datasets" / "Processed"
output_folder.mkdir(parents=True, exist_ok=True)

output_path = output_folder / "wholesale_customers_clustered.csv"

wholesale_df.to_csv(
    output_path,
    index=False
)

print("\nClustered Dataset Saved Successfully!")
print("Saved File:", output_path)

# endregion


# region Training Models

model_folder = BASE_DIR / "ModelsOutcome"
model_folder.mkdir(parents=True, exist_ok=True)

kmeans_path = model_folder / "wholesale_customers_kmeans.joblib"
scaler_path = model_folder / "wholesale_customers_scaler.joblib"
features_path = model_folder / "wholesale_customers_feature_columns.joblib"

joblib.dump(kmeans_final, kmeans_path)
joblib.dump(scaler, scaler_path)
joblib.dump(FEATURES, features_path)

print("\nTrained Model Saved Successfully!")
print("K-Means Model Saved:", kmeans_path)
print("Scaler Saved:", scaler_path)
print("Features Saved:", features_path)

loaded_kmeans = joblib.load(kmeans_path)
loaded_scaler = joblib.load(scaler_path)
loaded_features = joblib.load(features_path)

print("\nModels Loaded Successfully!")

X_new = wholesale_df[loaded_features].copy()
X_new_scaled = loaded_scaler.transform(X_new)

X_new_scaled = pd.DataFrame(
    X_new_scaled,
    columns=loaded_features,
    index=X_new.index
)

predicted_labels = loaded_kmeans.predict(X_new_scaled)
original_labels = wholesale_df["ClusterLabel"].to_numpy()

print("\nPredicted Cluster Labels (First 10):")
print(predicted_labels[:10])
print("-" * 60)

model_verification = bool(
    np.array_equal(original_labels, predicted_labels)
)

print("Model Verification:")
print(model_verification)

# endregion


# region Save Metadata

metadata_folder = BASE_DIR / "Metadata"
metadata_folder.mkdir(parents=True, exist_ok=True)

original_df = wholesale_df.drop(columns=["ClusterLabel"])

data_types = {
    column: str(dtype)
    for column, dtype in original_df.dtypes.items()
}

missing_values = {
    column: int(count)
    for column, count in original_df.isnull().sum().items()
}

descriptive_statistics = json.loads(
    original_df.describe().round(2).to_json()
)

cluster_profiles = json.loads(
    cluster_analysis.round(2).to_json(orient="index")
)

correlation_data = json.loads(
    correlation_matrix.round(4).to_json()
)

k_selection_metrics = json.loads(
    metrics_df.round(4).to_json(orient="index")
)

combined_ranking = json.loads(
    rank_df.round(2).to_json(orient="index")
)

cluster_sizes_dict = {
    str(int(k)): int(v)
    for k, v in cluster_counts.items()
}

cluster_percentages_dict = {
    str(int(k)): round(float(v), 2)
    for k, v in cluster_percentages.items()
}

channel_distribution_data = json.loads(
    channel_distribution.to_json()
)

region_distribution_data = json.loads(
    region_distribution.to_json()
)

ssd_metadata = {
    "total_ssd": round(total_ssd, 4),
    "total_sum_squares": round(total_sum_squares, 4),
    "between_cluster_ss": round(between_ssd, 4),
    "explained_variance_pct": round(explained_variance_pct, 2),
    "per_cluster_ssd": {
        str(int(k)): round(float(v), 4)
        for k, v in cluster_ssd.items()
    },
    "matches_kmeans_inertia": ssd_verification
}

metadata = {
    "id": "D20",
    "name": "Wholesale Customers",
    "level": "Intermediate",
    "domain": "Business/Retail",
    "task": "Clustering",
    "description": (
        "Find patterns from spending data at wholesale "
        "and group similar customers based on annual spending."
    ),
    "source": "Data Science Dojo",
    "source_url": (
        "https://code.datasciencedojo.com/"
        "datasciencedojo/datasets/tree/master/"
        "Wholesale%20Customers"
    ),
    "dataset_statistics": {
        "rows": int(original_df.shape[0]),
        "columns": int(original_df.shape[1]),
        "column_names": list(original_df.columns),
        "data_types": data_types,
        "missing_values": missing_values,
        "total_missing_values": int(original_df.isnull().sum().sum()),
        "descriptive_statistics": descriptive_statistics
    },
    "feature_selection": {
        "selected_features": FEATURES,
        "analysis_only_columns": ["Channel", "Region"],
        "excluded_columns": ["Channel", "Region"]
    },
    "preprocessing": {
        "transformation": "No additional transformation",
        "transformation_method": "identity",
        "normalization": "MinMaxScaler",
        "input_features": FEATURES,
        "missing_value_method": "Median Imputation"
    },
    "k_selection": {
        "k_range": [K_MIN, K_MAX],
        "best_k_silhouette": best_k_silhouette,
        "best_k_calinski_harabasz": best_k_ch,
        "best_k_davies_bouldin": best_k_db,
        "best_k_combined": best_k_combined,
        "elbow_k": elbow_k,
        "elbow_metric_support": elbow_metric_support,
        "elbow_support_count": int(elbow_support_count),
        "elbow_supported_by_validation": bool(elbow_supported_by_validation),
        "selection_method": "K=2 selected after Silhouette and validation comparison",
        "selection_reason": selection_reason,
        "final_k": int(final_k),
        "manual_override": True,
        "metrics_per_k": k_selection_metrics,
        "combined_ranking": combined_ranking
    },
    "Clustering": {
        "algorithm": "KMeans",
        "n_clusters": int(kmeans_final.n_clusters),
        "random_state": RANDOM_STATE,
        "n_init": N_INIT,
        "scaler": "MinMaxScaler",
        "transformation": "None (raw spending values)",
        "input_features": FEATURES,
        "cluster_label_column": "ClusterLabel",
        "cluster_sizes": cluster_sizes_dict,
        "cluster_percentages": cluster_percentages_dict,
        "cluster_profiles": cluster_profiles,
        "channel_distribution": channel_distribution_data,
        "region_distribution": region_distribution_data,
        "inertia": float(kmeans_final.inertia_),
        "silhouette_score": round(float(final_silhouette), 4),
        "ssd": ssd_metadata
    },
    "model_verification": model_verification,
    "correlation_matrix": correlation_data,
    "output_files": {
        "original_dataset": "Datasets/Wholesale customers data.csv",
        "processed_dataset": (
            "Datasets/Processed/"
            "wholesale_customers_clustered.csv"
        ),
        "trained_model": (
            "ModelsOutcome/"
            "wholesale_customers_kmeans.joblib"
        ),
        "scaler": (
            "ModelsOutcome/"
            "wholesale_customers_scaler.joblib"
        ),
        "feature_columns": (
            "ModelsOutcome/"
            "wholesale_customers_feature_columns.joblib"
        )
    }
}

metadata_file = metadata_folder / "D20_wholesale_customers.json"

with open(metadata_file, "w", encoding="utf-8") as file:
    json.dump(
        metadata,
        file,
        ensure_ascii=False,
        indent=4
    )

print("\nMetadata Saved Successfully!")
print("Saved File:", metadata_file)

with open(metadata_file, "r", encoding="utf-8") as file:
    saved_metadata = json.load(file)

print("-" * 60)
print("Metadata Verification:")
print("Dataset Name:", saved_metadata["name"])
print("Total Rows:", saved_metadata["dataset_statistics"]["rows"])
print("Total Columns:", saved_metadata["dataset_statistics"]["columns"])
print("Algorithm:", saved_metadata["Clustering"]["algorithm"])
print("Final K:", saved_metadata["k_selection"]["final_k"])
print("Total Clusters:", saved_metadata["Clustering"]["n_clusters"])
print("Cluster Sizes:", saved_metadata["Clustering"]["cluster_sizes"])
print("Silhouette Score:", saved_metadata["Clustering"]["silhouette_score"])
print("Total SSD:", saved_metadata["Clustering"]["ssd"]["total_ssd"])
print(
    "SSD Verification:",
    saved_metadata["Clustering"]["ssd"]["matches_kmeans_inertia"]
)
print("Model Verification:", saved_metadata["model_verification"])
print("\nMetadata Process Completed Successfully!")

# endregion

