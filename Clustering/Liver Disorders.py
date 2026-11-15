import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap

from sklearn.cluster import KMeans
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

FINAL_K = 2

FEATURES = [
    "mcv",
    "alkphos",
    "sgpt",
    "sgot",
    "gammagt"
]

ANALYSIS_COLUMNS = FEATURES + ["drinks"]


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
    / "bupa.data.csv"
)

liver_df = pd.read_csv(
    dataset_path,
    header=None
)

liver_df.columns = [
    "mcv",
    "alkphos",
    "sgpt",
    "sgot",
    "gammagt",
    "drinks",
    "selector"
]

print("\nLiver Disorders Dataset Loaded Successfully!")

# endregion


# region Data Understanding

print("\nFirst 5 Rows:")
print(liver_df.head())
print("-" * 60)

print("Dataset Shape:")
print(liver_df.shape)
print("-" * 60)

print("Columns:")
print(liver_df.columns)
print("-" * 60)

print("Data Types:")
print(liver_df.dtypes)
print("-" * 60)

print("\nDataset Information:")
liver_df.info()
print("-" * 60)

print("Statistical Description:")
print(liver_df.describe().to_string())
print("-" * 60)

print("Missing Values:")
print(liver_df.isna().sum())
print("-" * 60)

print("Total Missing Values:")
print(liver_df.isna().sum().sum())
print("-" * 60)

print("Selector Distribution:")
print(liver_df["selector"].value_counts())
print("-" * 60)

# endregion


# region Feature Selection

X = liver_df[FEATURES].copy()

drinks = liver_df["drinks"].copy()

print("\nSelected Features:")
print(X.head())
print("-" * 60)

print("Features Shape:")
print(X.shape)
print("-" * 60)

print("Feature Names:")
print(FEATURES)
print("-" * 60)

print("Drinks Values:")
print(drinks.head())
print("-" * 60)

# endregion


# region Data Preprocessing

for column in FEATURES:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

drinks = pd.to_numeric(
    drinks,
    errors="coerce"
)

print("\nMissing Values:")
print(X.isnull().sum())
print("-" * 60)

if X.isnull().sum().sum() > 0:

    X = X.fillna(X.median())

if drinks.isnull().sum() > 0:

    drinks = drinks.fillna(drinks.median())

liver_df[FEATURES] = X
liver_df["drinks"] = drinks

if not np.isfinite(X.to_numpy()).all():

    raise ValueError(
        "Dataset contains infinite values."
    )

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
        "Explained Variance": (
            1 - model.inertia_ / total_sum_squares
        ) * 100,
        "Silhouette": np.nan,
        "Calinski-Harabasz": np.nan,
        "Davies-Bouldin": np.nan
    }

    if k >= K_MIN:

        record["Silhouette"] = silhouette_score(
            df_scaled,
            model.labels_
        )

        record["Calinski-Harabasz"] = (
            calinski_harabasz_score(
                df_scaled,
                model.labels_
            )
        )

        record["Davies-Bouldin"] = (
            davies_bouldin_score(
                df_scaled,
                model.labels_
            )
        )

    records.append(record)


metrics_df = pd.DataFrame(
    records
).set_index("K")

print("\nClustering Metrics For Each K:")
print(metrics_df.round(4).to_string())


eval_df = metrics_df.loc[K_MIN:K_MAX]


# Elbow Method

x = np.array(metrics_df.index, dtype=float)

y = metrics_df["Inertia"].to_numpy()

x = (x - x.min()) / (x.max() - x.min())

y = (y - y.min()) / (y.max() - y.min())

x0, y0 = x[0], y[0]
x1, y1 = x[-1], y[-1]

distances = np.abs(
    (y1 - y0) * x
    - (x1 - x0) * y
    + x1 * y0
    - y1 * x0
) / np.hypot(y1 - y0, x1 - x0)

elbow_k = int(
    list(metrics_df.index)[np.argmax(distances)]
)

best_k_silhouette = int(
    eval_df["Silhouette"].idxmax()
)

best_k_ch = int(
    eval_df["Calinski-Harabasz"].idxmax()
)

best_k_db = int(
    eval_df["Davies-Bouldin"].idxmin()
)

rank_df = pd.DataFrame({

    "Silhouette": eval_df[
        "Silhouette"
    ].rank(ascending=False),

    "Calinski-Harabasz": eval_df[
        "Calinski-Harabasz"
    ].rank(ascending=False),

    "Davies-Bouldin": eval_df[
        "Davies-Bouldin"
    ].rank(ascending=True)

})

rank_df["MeanRank"] = rank_df.mean(axis=1)

best_k_combined = int(
    rank_df["MeanRank"].idxmin()
)


print("\n" + "=" * 60)
print("Best K By Each Criterion")
print("=" * 60)

print("Elbow (automatic)    :", elbow_k)
print("Silhouette           :", best_k_silhouette)
print("Calinski-Harabasz    :", best_k_ch)
print("Davies-Bouldin       :", best_k_db)
print("Combined Mean Rank   :", best_k_combined)

print("\nCombined Ranking:")
print(rank_df.sort_values("MeanRank").round(2).to_string())


candidate_ks = list(
    rank_df["MeanRank"]
    .sort_values()
    .head(2)
    .index.astype(int)
)

if elbow_k not in candidate_ks:

    candidate_ks.append(elbow_k)

if FINAL_K not in candidate_ks:

    candidate_ks.append(FINAL_K)

candidate_ks = sorted(set(candidate_ks))

final_k = FINAL_K

print("\nCandidate K Values:", candidate_ks)
print("Final K Selected:", final_k)

#endregion


# region Compare K=2 and K=3

for k in [2, 3]:

    print("\n" + "=" * 60)
    print(f"K-Means Model with K = {k}")
    print("=" * 60)

    comparison_model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=N_INIT
    )

    comparison_labels = comparison_model.fit_predict(
        df_scaled
    )

    comparison_score = silhouette_score(
        df_scaled,
        comparison_labels
    )

    print("\nSilhouette Score:")
    print(round(comparison_score, 4))

    comparison_df = liver_df.copy()

    comparison_df["ClusterLabel"] = comparison_labels

    comparison_counts = (
        comparison_df["ClusterLabel"]
        .value_counts()
        .sort_index()
    )

    print("\nCluster Distribution:")

    print(pd.DataFrame({

        "Count": comparison_counts,

        "Percent": (
            comparison_counts
            / len(comparison_df) * 100
        ).round(2)

    }).to_string())

    print("\nCluster Analysis:")

    print(
        comparison_df.groupby("ClusterLabel")[
            ANALYSIS_COLUMNS
        ].mean().round(2).to_string()
    )

# endregion


# region Final K-Means Clustering

kmeans_final = KMeans(
    n_clusters=FINAL_K,
    random_state=RANDOM_STATE,
    n_init=N_INIT
)

cluster_labels = kmeans_final.fit_predict(
    df_scaled
)

liver_df["ClusterLabel"] = cluster_labels

print("\nFinal Model: K =", FINAL_K)

print("\nFinal Cluster Distribution:")

print(
    liver_df["ClusterLabel"]
    .value_counts()
    .sort_index()
)

print("-" * 60)

final_silhouette = silhouette_score(
    df_scaled,
    cluster_labels
)

print("Final Silhouette Score:")
print(round(final_silhouette, 4))

# endregion


# region Results

cluster_analysis = (
    liver_df.groupby("ClusterLabel")[ANALYSIS_COLUMNS]
    .mean()
)

print("\nCluster Analysis:")

print(
    cluster_analysis.round(3).to_string()
)

print("-" * 60)


cluster_counts = (
    liver_df["ClusterLabel"]
    .value_counts()
    .sort_index()
)

print("Cluster Sizes:")
print(cluster_counts)

print("-" * 60)


cluster_percentages = (
    liver_df["ClusterLabel"]
    .value_counts(normalize=True)
    .sort_index() * 100
)

print("Cluster Percentages:")
print(cluster_percentages)

print("-" * 60)


center_point_scaled = pd.DataFrame(
    kmeans_final.cluster_centers_,
    columns=FEATURES
)

center_point_original = pd.DataFrame(
    scaler.inverse_transform(
        kmeans_final.cluster_centers_
    ),
    columns=FEATURES
)

print("\nCluster Centers - Original Units:")

print(
    center_point_original.round(3).to_string()
)

# endregion


# region SSD Evaluation

cluster_ssd = {}

cluster_sizes_for_ssd = {}

for cluster_id in range(FINAL_K):

    points = X_scaled_array[
        cluster_labels == cluster_id
    ]

    center = kmeans_final.cluster_centers_[cluster_id]

    cluster_ssd[cluster_id] = float(
        ((points - center) ** 2).sum()
    )

    cluster_sizes_for_ssd[cluster_id] = len(points)


total_ssd = sum(cluster_ssd.values())

between_ssd = total_sum_squares - total_ssd

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

print(
    "Explained Variance:",
    round(explained_variance_pct, 2),
    "%"
)

ssd_verification = bool(
    np.isclose(
        total_ssd,
        kmeans_final.inertia_
    )
)

print("SSD Verification:", ssd_verification)

# endregion


# region Data Visualization

plt.figure(
    figsize=(9, 6),
    facecolor=BG_COLOR
)

ax = plt.gca()

ax.set_facecolor(BG_COLOR)

ax.set_axisbelow(True)


for cluster_id in sorted(
    liver_df["ClusterLabel"].unique()
):

    cluster_data = liver_df[
        liver_df["ClusterLabel"] == cluster_id
    ]

    plt.scatter(
        cluster_data["sgpt"],
        cluster_data["gammagt"],
        color=CLUSTER_COLORS[cluster_id],
        s=70,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.7,
        label=f"Cluster {cluster_id}"
    )


plt.scatter(
    cluster_analysis["sgpt"],
    cluster_analysis["gammagt"],
    marker="X",
    c=TITLE_COLOR,
    edgecolors="white",
    linewidths=1.5,
    s=230,
    label="Cluster Centers",
    zorder=5
)


plt.xlabel(
    "SGPT",
    fontsize=12,
    color=TEXT_COLOR,
    labelpad=10
)

plt.ylabel(
    "GammaGT",
    fontsize=12,
    color=TEXT_COLOR,
    labelpad=10
)

plt.title(
    "Liver Disorders - KMeans Clustering",
    fontsize=17,
    fontweight="bold",
    color=TITLE_ACCENT,
    pad=20
)

plt.xticks(
    fontsize=11,
    color=TEXT_COLOR
)

plt.yticks(
    fontsize=11,
    color=TEXT_COLOR
)

plt.grid(
    color=LAVENDER,
    alpha=0.3,
    linewidth=0.8
)

for spine in ax.spines.values():

    spine.set_color(LAVENDER)

    spine.set_linewidth(0.8)


legend = plt.legend(
    loc="upper right",
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


correlation_matrix = liver_df[
    ANALYSIS_COLUMNS
].corr()

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
    "Liver Disorders - Correlation Heatmap",
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
    / "liver_disorders_clustered.csv"
)

liver_df.to_csv(
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
    / "liver_disorders_kmeans.joblib"
)

scaler_path = (
    model_folder
    / "liver_disorders_scaler.joblib"
)

features_path = (
    model_folder
    / "liver_disorders_feature_columns.joblib"
)


joblib.dump(
    kmeans_final,
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

print("K-Means Model Saved:", kmeans_path)

print("Scaler Saved:", scaler_path)

print("Features Saved:", features_path)


loaded_kmeans = joblib.load(kmeans_path)

loaded_scaler = joblib.load(scaler_path)

loaded_features = joblib.load(features_path)


print("\nModels Loaded Successfully!")


X_new = liver_df[loaded_features].copy()

X_new_scaled = loaded_scaler.transform(X_new)

X_new_scaled = pd.DataFrame(
    X_new_scaled,
    columns=loaded_features,
    index=X_new.index
)


predicted_labels = loaded_kmeans.predict(
    X_new_scaled
)

original_labels = liver_df[
    "ClusterLabel"
].to_numpy()


print("\nPredicted Cluster Labels (First 10):")

print(predicted_labels[:10])

print("-" * 60)


model_verification = bool(
    np.array_equal(
        original_labels,
        predicted_labels
    )
)

print("Model Verification:")

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

original_df = liver_df.drop(
    columns=["ClusterLabel"]
)


data_types = {
    column: str(dtype)
    for column, dtype in original_df.dtypes.items()
}

missing_values = {
    column: int(count)
    for column, count in original_df.isna().sum().items()
}

descriptive_statistics = json.loads(
    original_df.describe().round(2).to_json()
)

cluster_profiles = json.loads(
    cluster_analysis.round(2).to_json(
        orient="index"
    )
)

correlation_data = json.loads(
    correlation_matrix.round(4).to_json()
)

k_selection_metrics = json.loads(
    metrics_df.round(4).to_json(
        orient="index"
    )
)

combined_ranking = json.loads(
    rank_df.round(2).to_json(
        orient="index"
    )
)


cluster_sizes_dict = {
    str(int(k)): int(v)
    for k, v in cluster_counts.items()
}

cluster_percentages_dict = {
    str(int(k)): round(float(v), 2)
    for k, v in cluster_percentages.items()
}


ssd_metadata = {

    "total_ssd": round(total_ssd, 4),

    "total_sum_squares": round(
        total_sum_squares,
        4
    ),

    "between_cluster_ss": round(
        between_ssd,
        4
    ),

    "explained_variance_pct": round(
        explained_variance_pct,
        2
    ),

    "per_cluster_ssd": {
        str(int(k)): round(float(v), 4)
        for k, v in cluster_ssd.items()
    },

    "matches_kmeans_inertia": ssd_verification

}


metadata = {

    "id": "D15",

    "name": "Liver Disorders",

    "level": "Intermediate",

    "domain": "Healthcare",

    "task": "Clustering",

    "description": (
        "Discover patterns in liver blood test measurements "
        "and analyze their relationship with alcohol consumption."
    ),

    "source": "Data Science Dojo",

    "source_url": (
        "https://code.datasciencedojo.com/"
        "datasciencedojo/datasets/tree/master/Liver%20Disorders"
    ),

    "dataset_statistics": {

        "rows": int(original_df.shape[0]),

        "columns": int(original_df.shape[1]),

        "column_names": list(original_df.columns),

        "data_types": data_types,

        "missing_values": missing_values,

        "total_missing_values": int(
            original_df.isna().sum().sum()
        ),

        "descriptive_statistics": descriptive_statistics

    },

    "feature_selection": {

        "selected_features": FEATURES,

        "analysis_only_columns": [
            "drinks"
        ],

        "excluded_columns": [
            "selector"
        ]

    },

    "preprocessing": {

        "method": "MinMaxScaler",

        "input_features": FEATURES,

        "missing_value_method": "Median Imputation"

    },

    "k_selection": {

        "k_range": [
            K_MIN,
            K_MAX
        ],

        "elbow_k": int(elbow_k),

        "best_k_silhouette": int(
            best_k_silhouette
        ),

        "best_k_calinski_harabasz": int(
            best_k_ch
        ),

        "best_k_davies_bouldin": int(
            best_k_db
        ),

        "best_k_combined": int(
            best_k_combined
        ),

        "candidate_ks": [
            int(k)
            for k in candidate_ks
        ],

        "final_k": FINAL_K,

        "manual_override": True,

        "metrics_per_k": k_selection_metrics,

        "combined_ranking": combined_ranking

    },

    "clustering": {

        "algorithm": "KMeans",

        "n_clusters": int(kmeans_final.n_clusters),

        "random_state": RANDOM_STATE,

        "n_init": N_INIT,

        "scaler": "MinMaxScaler",

        "input_features": FEATURES,

        "analysis_variable": "drinks",

        "excluded_columns": [
            "selector"
        ],

        "cluster_label_column": "ClusterLabel",

        "cluster_sizes": cluster_sizes_dict,

        "cluster_percentages": cluster_percentages_dict,

        "cluster_profiles": cluster_profiles,

        "inertia": float(kmeans_final.inertia_),

        "silhouette_score": round(
            float(final_silhouette),
            4
        ),

        "ssd": ssd_metadata

    },

    "model_verification": model_verification,

    "correlation_matrix": correlation_data,

    "output_files": {

        "original_dataset": (
            "Datasets/bupa.data.csv"
        ),

        "processed_dataset": (
            "Datasets/Processed/"
            "liver_disorders_clustered.csv"
        ),

        "trained_model": (
            "ModelsOutcome/"
            "liver_disorders_kmeans.joblib"
        ),

        "scaler": (
            "ModelsOutcome/"
            "liver_disorders_scaler.joblib"
        ),

        "feature_columns": (
            "ModelsOutcome/"
            "liver_disorders_feature_columns.joblib"
        )

    }

}


metadata_file = (
    metadata_folder
    / "D15_liver_disorders.json"
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
    saved_metadata["clustering"]["algorithm"]
)

print(
    "Final K:",
    saved_metadata["k_selection"]["final_k"]
)

print(
    "Total Clusters:",
    saved_metadata["clustering"]["n_clusters"]
)

print(
    "Cluster Sizes:",
    saved_metadata["clustering"]["cluster_sizes"]
)

print(
    "Silhouette Score:",
    saved_metadata["clustering"]["silhouette_score"]
)

print(
    "Total SSD:",
    saved_metadata["clustering"]["ssd"]["total_ssd"]
)

print(
    "SSD Verification:",
    saved_metadata["clustering"]["ssd"]["matches_kmeans_inertia"]
)

print(
    "Model Verification:",
    saved_metadata["model_verification"]
)

print("\nMetadata Process Completed Successfully!")

# endregion
