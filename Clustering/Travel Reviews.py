from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.colors import LinearSegmentedColormap


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


CATEGORY_LABELS_EN = {

    "Category 1": "Art Galleries",

    "Category 2": "Dance Clubs",

    "Category 3": "Juice Bars",

    "Category 4": "Restaurants",

    "Category 5": "Museums",

    "Category 6": "Resorts",

    "Category 7": "Parks & Picnic Spots",

    "Category 8": "Beaches",

    "Category 9": "Theaters",

    "Category 10": "Religious Institutions"

}

CATEGORY_LABELS_FA = {

    "Category 1": "گالری‌های هنری",

    "Category 2": "کلوب‌های رقص",

    "Category 3": "آبمیوه‌فروشی‌ها",

    "Category 4": "رستوران‌ها",

    "Category 5": "موزه‌ها",

    "Category 6": "اقامتگاه‌ها و مراکز تفریحی",

    "Category 7": "پارک‌ها و محل‌های پیک‌نیک",

    "Category 8": "سواحل",

    "Category 9": "سالن‌های تئاتر",

    "Category 10": "کلیساها و اماکن مذهبی"

}


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
    Path(__file__).resolve().parent
    / "Datasets"
    / "tripadvisor_review.csv"
)

df = pd.read_csv(dataset_path)

df.columns = df.columns.str.strip()

expected_columns = ( ["User ID"] + [f"Category {i}" for i in range(1, 11)]
)

if list(df.columns) != expected_columns:

    raise ValueError(
        "Dataset columns do not match expected columns."
    )

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

print("Statistical Description Transposed:")

print(df.describe().T.to_string())

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

print("Category 7 Unique Values:")

print(df["Category 7"].nunique())

print("-" * 60)

print("Category 7 Standard Deviation:")

print(df["Category 7"].std())

print("-" * 60)

print("Travel Categories:")

for column, name in CATEGORY_LABELS_EN.items():

    print(
        column,
        ":",
        name,
        "-",
        CATEGORY_LABELS_FA[column]
    )

# endregion


# region Feature Selection

df_features = df.drop(
    columns=[
        "User ID",
        "Category 7"
    ]
).copy()


feature_columns = df_features.columns.tolist()

print("\nSelected Features:")

print(df_features.head())

print("-" * 60)

print("Features Shape:")

print(df_features.shape)

print("-" * 60)

print("Feature Names:")

print(feature_columns)

print("-" * 60)

print("Selected Travel Categories:")

for column in feature_columns:

    print(
        CATEGORY_LABELS_EN[column],
        "-",
        CATEGORY_LABELS_FA[column]
    )

print("-" * 60)

# endregion


# region Data Preprocessing

for column in feature_columns:

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

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df_features)

df_scaled = pd.DataFrame(
    scaled_data,
    columns=feature_columns,
    index=df.index
)

print("\nNormalized Data:")

print(df_scaled.head())

print("-" * 60)

print("Normalized Data Shape:")

print(df_scaled.shape)

print("-" * 60)

print("Normalized Data Mean:")

print(df_scaled.mean().round(4))

print("-" * 60)

print("Normalized Data Standard Deviation:")

print(df_scaled.std().round(4))

print("-" * 60)

# endregion


# region Model Evaluation

inertia_values = []

k_values = range(1, 11)

for k in k_values:

    kmeans_model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans_model.fit(df_scaled)

    inertia_values.append(
        kmeans_model.inertia_
    )

print("\nElbow Method - Inertia Values:")

for k, inertia in zip(k_values, inertia_values):

    print(
        f"K = {k} : Inertia = {inertia:.4f}"
    )

silhouette_values = []

silhouette_k_values = range(2, 11)

for k in silhouette_k_values:

    kmeans_model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    evaluation_labels = kmeans_model.fit_predict(
        df_scaled
    )

    score = silhouette_score(
        df_scaled,
        evaluation_labels
    )

    silhouette_values.append(score)

print("\nSilhouette Score Values:")

for k, score in zip(
    silhouette_k_values,
    silhouette_values
):

    print(
        f"K = {k} : Silhouette Score = {score:.4f}"
    )

best_k_index = np.argmax(silhouette_values)

best_k = list(silhouette_k_values)[best_k_index]

best_score = silhouette_values[best_k_index]

print("-" * 60)

print("Best Number of Clusters:", best_k)

print("Best Silhouette Score:", round(best_score, 4))

# endregion


# region Compare K=2 and K=3

for k in [2, 3]:

    print("\n" + "=" * 60)

    print(f"K-Means Model with K = {k}")

    print("=" * 60)

    comparison_model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
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

    comparison_df = df.copy()

    comparison_df["Cluster"] = comparison_labels

    print("\nCluster Distribution:")

    print(
        comparison_df["Cluster"]
        .value_counts()
        .sort_index()
    )

    print("\nCluster Percentages:")

    print(
        comparison_df["Cluster"]
        .value_counts(normalize=True)
        .sort_index() * 100
    )

    print("\nCluster Analysis:")

    comparison_profile = (
        comparison_df.groupby("Cluster")[
            feature_columns
        ].mean()
    )

    comparison_profile = comparison_profile.rename(
        columns=CATEGORY_LABELS_EN
    )

    print(
        comparison_profile.round(3).to_string()
    )

# endregion


# region Final K-Means Clustering

kmeans_final = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)


cluster_labels = kmeans_final.fit_predict(
    df_scaled
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
    df_scaled,
    cluster_labels
)

print("Final Silhouette Score:")

print(round(final_silhouette, 4))

# endregion


# region Results

cluster_profile = df.groupby("Cluster")[
    feature_columns
].mean()

print("\nCluster Analysis:")

display_profile = cluster_profile.rename(
    columns=CATEGORY_LABELS_EN
)

print(
    display_profile.round(3).to_string()
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

center_point = pd.DataFrame(
    kmeans_final.cluster_centers_,
    columns=feature_columns
)

center_point_original = pd.DataFrame(
    scaler.inverse_transform(
        kmeans_final.cluster_centers_
    ),
    columns=feature_columns
)


print("\nCluster Centers - Original Rating Scale:")

center_point_display = center_point_original.rename(
    columns=CATEGORY_LABELS_EN
)

print(
    center_point_display.round(3).to_string()
)

print("-" * 60)

print("Top Travel Categories In Each Cluster:")

for cluster_id, row in cluster_profile.iterrows():

    print(f"\nCluster {cluster_id}:")

    sorted_categories = row.sort_values(
        ascending=False
    )

    for category, rating in sorted_categories.head(3).items():

        print(
            CATEGORY_LABELS_EN[category],
            "-",
            CATEGORY_LABELS_FA[category],
            ":",
            round(rating, 3)
        )

# endregion


# region SSD Evaluation

assigned_centers = (
    kmeans_final.cluster_centers_[cluster_labels]
)

ssd = float(
    np.sum(
        (df_scaled.to_numpy() - assigned_centers) ** 2
    )
)

print("\nSSD Evaluation:")

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

if best_k <= len(base_colors):

    cluster_colors = base_colors[:best_k]

else:

    cluster_colors = sns.color_palette(
        "husl",
        n_colors=best_k
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
        cluster_data["Category 1"],
        cluster_data["Category 2"],
        color=cluster_colors[cluster_id],
        s=65,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.7,
        label=f"Cluster {cluster_id}"
    )

plt.scatter(
    center_point["Category 1"],
    center_point["Category 2"],
    marker="X",
    c=TITLE_COLOR,
    edgecolors="white",
    linewidths=1.5,
    s=230,
    label="Cluster Centers",
    zorder=5
)

plt.xlabel(
    CATEGORY_LABELS_EN["Category 1"],
    fontsize=12,
    color=TEXT_COLOR,
    labelpad=10
)

plt.ylabel(
    CATEGORY_LABELS_EN["Category 2"],
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


correlation_matrix = df_features.corr()

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


heatmap_labels = [
    CATEGORY_LABELS_EN[column]
    for column in feature_columns
]

plt.xticks(
    np.arange(len(heatmap_labels)) + 0.5,
    heatmap_labels,
    rotation=40,
    ha="right",
    fontsize=9,
    color=TEXT_COLOR
)

plt.yticks(
    np.arange(len(heatmap_labels)) + 0.5,
    heatmap_labels,
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

# endregion


# region Save Results

output_folder = (
    Path(__file__).resolve().parent
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

#endregion


# region Training Models

model_folder = (
    Path(__file__).resolve().parent
    / "ModelsOutcome"
)

model_folder.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    kmeans_final,
    model_folder / "travel_reviews_kmeans.joblib"
)

joblib.dump(
    scaler,
    model_folder / "travel_reviews_scaler.joblib"
)

joblib.dump(
    feature_columns,
    model_folder / "travel_reviews_feature_columns.joblib"
)


print("\nTrained Model Saved Successfully!")

print(
    model_folder / "travel_reviews_kmeans.joblib"
)

print("\nScaler Saved Successfully!")

print(
    model_folder / "travel_reviews_scaler.joblib"
)

loaded_kmeans = joblib.load(
    model_folder / "travel_reviews_kmeans.joblib"
)

loaded_scaler = joblib.load(
    model_folder / "travel_reviews_scaler.joblib"
)

loaded_features = joblib.load(
    model_folder / "travel_reviews_feature_columns.joblib"
)


print("\nModels Loaded Successfully!")

X_new = df_features[loaded_features]

X_new_scaled = loaded_scaler.transform(X_new)


X_new_scaled = pd.DataFrame(
    X_new_scaled,
    columns=loaded_features,
    index=X_new.index
)

predicted_labels = loaded_kmeans.predict(
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
    Path(__file__).resolve().parent
    / "Metadata"
)

metadata_folder.mkdir(
    parents=True,
    exist_ok=True
)

original_df = df.drop(
    columns=["Cluster"]
)

data_types = {}

for column in original_df.columns:

    data_types[column] = str(
        original_df[column].dtype
    )

missing_values = {}

for column in original_df.columns:

    missing_values[column] = int(
        original_df[column].isnull().sum()
    )

statistics_df = df_features.describe().round(3)

descriptive_statistics = {}

for column in statistics_df.columns:

    descriptive_statistics[column] = {}

    for statistic in statistics_df.index:

        descriptive_statistics[column][statistic] = float(
            statistics_df.loc[statistic, column]
        )

cluster_sizes_dict = {}

cluster_percentages_dict = {}

for cluster_id, count in cluster_counts.items():

    cluster_sizes_dict[str(int(cluster_id))] = int(count)

    cluster_percentages_dict[str(int(cluster_id))] = round(
        float(cluster_percentages.loc[cluster_id]),
        2
    )

cluster_profiles = {}

for cluster_id, row in cluster_profile.iterrows():

    cluster_profiles[str(int(cluster_id))] = {}

    for column in cluster_profile.columns:

        cluster_profiles[str(int(cluster_id))][column] = round(
            float(row[column]),
            3
        )

correlation_data = {}

for column in correlation_matrix.columns:

    correlation_data[column] = {}

    for row in correlation_matrix.index:

        correlation_data[column][row] = round(
            float(correlation_matrix.loc[row, column]),
            4
        )


elbow_results = {}

for k, inertia in zip(k_values, inertia_values):

    elbow_results[str(k)] = round(
        float(inertia),
        4
    )

silhouette_results = {}

for k, score in zip(
    silhouette_k_values,
    silhouette_values
):

    silhouette_results[str(k)] = round(
        float(score),
        4
    )

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

    "category_labels": {

        "english": CATEGORY_LABELS_EN,

        "persian": CATEGORY_LABELS_FA

    },

    "feature_selection": {

        "selected_features": feature_columns,

        "excluded_columns": [
            "User ID",
            "Category 7"
        ],

        "excluded_feature_reasons": {

            "User ID": (
                "Identifier column, not a travel preference feature."
            ),

            "Category 7": (
                "Near-constant feature with very low variance."
            )

        }

    },

    "preprocessing": {

        "method": "StandardScaler",

        "input_features": feature_columns,

        "missing_value_method": "Median Imputation",

        "rating_range": {

            "minimum": 0,

            "maximum": 4

        }

    },

    "clustering": {

        "algorithm": "KMeans",

        "n_clusters": int(best_k),

        "random_state": 42,

        "input_features": feature_columns,

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

        "SSD_description": (
            "Sum of Squared Distances from each sample "
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

        "elbow_ssd": elbow_results,

        "silhouette_scores": silhouette_results,

        "best_k": int(best_k),

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

print(
    "Dataset Name:",
    saved_metadata["name"]
)

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
    "Total Clusters:",
    saved_metadata["clustering"]["n_clusters"]
)

print(
    "Cluster Sizes:",
    saved_metadata["clustering"]["cluster_sizes"]
)

print(
    "Silhouette Score:",
    saved_metadata["model_evaluation"]["silhouette_score"]
)

print(
    "SSD:",
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
