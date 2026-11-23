"""D31 Online News Popularity — Regression, Concrete-style charts, Istanbul-style workflow.
Place OnlineNewsPopularity.csv in DataSets next to this script.
"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

BASE_DIR = Path(__file__).resolve().parent
DATASET_ID = 'D31'
DATASET_NAME = 'Online News Popularity'
TARGET = 'shares'
TEST_SIZE = 0.20
RANDOM_STATE = 42
SHOW_FIGURES = True
BG_COLOR = '#FFFFFF'
TITLE_COLOR = '#2B1450'
TITLE_ACCENT = '#C9578F'
TEXT_COLOR = '#665B78'
PURPLE = '#8B5CC7'
PINK = '#E3A0C4'
TEAL = '#21B8A8'
LAVENDER = '#C9B6F2'
LIGHT_PINK = '#F3A6C8'
MODEL_COLORS = ['#7E57C2', '#6750A4', '#4F5FBF', '#3F72AF', '#5B9BD5']

# region Center Figure Window / Concrete Chart Style
def center_figure_window():
    manager = plt.get_current_fig_manager()
    window = getattr(manager, 'window', None)
    if window is None:
        return
    try:
        window.update_idletasks()
        window.update()
        width = window.winfo_width() or window.winfo_reqwidth()
        height = window.winfo_height() or window.winfo_reqheight()
        x = (window.winfo_screenwidth() - width) // 2
        y = max(40, (window.winfo_screenheight() - height) // 2 - 40)
        window.geometry(f'+{x}+{y}')
        window.update_idletasks()
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
    ax.set_title(title.replace('\n', ' '), fontsize=18,
                 fontweight='bold', color=TITLE_ACCENT, pad=24)


def display_figure():
    fig = plt.gcf()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        title = ax.title
        if title.get_text():
            max_width = fig.bbox.width * 0.92
            width = title.get_window_extent(renderer).width
            if width > max_width:
                title.set_fontsize(max(11, title.get_fontsize() * max_width / width))
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    show_figure()
# endregion

# region Load Dataset
folder = BASE_DIR / 'DataSets'
paths = [folder / 'OnlineNewsPopularity.csv',
         folder / 'Online News Popularity' / 'OnlineNewsPopularity.csv',
         folder / 'online_news_popularity.csv']
dataset_path = next((p for p in paths if p.is_file()), None)
if dataset_path is None:
    raise FileNotFoundError(
        f'OnlineNewsPopularity.csv not found. Put the original CSV in {folder}'
    )
news_df = pd.read_csv(dataset_path, low_memory=False)
# The original UCI CSV contains leading spaces in many header names.
news_df.columns = news_df.columns.astype(str).str.strip()
if news_df.columns.duplicated().any():
    raise ValueError('Duplicate column names after stripping whitespace.')
if TARGET not in news_df.columns:
    raise ValueError(f'Missing target {TARGET!r}. Available columns: {news_df.columns.tolist()}')
EXCLUDED = [c for c in ['url', 'timedelta'] if c in news_df.columns]
FEATURES = [c for c in news_df.columns if c not in EXCLUDED + [TARGET]]
if not FEATURES:
    raise ValueError('No usable input features found.')
# All original predictive features are numeric. Preserve original article order
# (the UCI dataset is ordered by publication chronology).
for column in FEATURES + [TARGET]:
    news_df[column] = pd.to_numeric(news_df[column], errors='coerce')
news_df = news_df.replace([np.inf, -np.inf], np.nan)
print('\nOnline News Popularity Dataset Loaded Successfully!')
print('Dataset Path:', dataset_path)
# endregion

# region Data Understanding
print('\nFirst 5 Rows:\n', news_df.head().to_string())
print('-' * 60)
print('Dataset Shape:', news_df.shape)
print('Columns:', news_df.columns.tolist())
print('Data Types:\n', news_df.dtypes)
print('Statistical Description:\n', news_df.describe().to_string())
print('Missing Values:\n', news_df.isnull().sum())
print('Total Missing Values:', int(news_df.isnull().sum().sum()))
print('Duplicate Rows:', int(news_df.duplicated().sum()))
# endregion

# region Feature Selection
X = news_df[FEATURES].copy()
y = news_df[TARGET].copy()
print('\nSelected Features:', FEATURES)
print('Target Column:', TARGET)
print('Excluded Columns:', EXCLUDED)
print('URL is an identifier; timedelta depends on data-collection time.')
# endregion

# region Data Preprocessing
if y.isna().any() or (y < 0).any():
    raise ValueError('Target shares must be present and nonnegative.')
if X.isna().all().any():
    raise ValueError('An input feature contains only missing values.')
print('\nMissing Values in Features:\n', X.isna().sum())
print('Preprocessing Completed Successfully!')
# endregion

# region Data Normalization
def create_preprocessor():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
print('Normalization: training-only median imputation + StandardScaler')
# endregion

# region Model Evaluation
# Original UCI article order (chronological), no random shuffle.
split_index = int(len(X) * (1 - TEST_SIZE))
if split_index < 2 or split_index >= len(X):
    raise ValueError('Not enough rows for a train/test split.')
x_train, x_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                  max_depth=3, random_state=RANDOM_STATE),
    'SVR': SVR(kernel='rbf', C=100.0, epsilon=0.1),
}
trained_models, predictions, records = {}, {}, []
for name, estimator in models.items():
    pipeline = Pipeline([('preprocessor', create_preprocessor()), ('model', estimator)])
    pipeline.fit(x_train, y_train)
    prediction = np.maximum(0.0, pipeline.predict(x_test))
    trained_models[name] = pipeline
    predictions[name] = prediction
    mse = mean_squared_error(y_test, prediction)
    mape_percentage = mean_absolute_percentage_error(y_test, prediction) * 100
    records.append({'Model': name, 'MAE': mean_absolute_error(y_test, prediction),
                    'MSE': mse, 'RMSE': np.sqrt(mse), 'R2': r2_score(y_test, prediction),
                    'R2 Percentage': r2_score(y_test, prediction) * 100,
                    'Accuracy': max(0.0, 100.0 - mape_percentage)})
metrics_df = pd.DataFrame(records).sort_values(['Accuracy', 'RMSE'],
                                                 ascending=[False, True], kind='stable').reset_index(drop=True)
best_model_name = str(metrics_df.iloc[0]['Model'])
best_prediction = predictions[best_model_name]
print('\nRegression Metrics:\n', metrics_df.round(4).to_string(index=False))
print('Recommended Model:', best_model_name)
print('Selection Rule: Highest Concrete-style MAPE-based Accuracy, lowest RMSE as tie-breaker')
print('Note: This is a descriptive regression score, not classification accuracy.')
# endregion

# region Feature Importance
linear_model = trained_models['Linear Regression'].named_steps['model']
coefficient_df = pd.DataFrame({'Feature': FEATURES, 'Coefficient': linear_model.coef_})
coefficient_df['Absolute Coefficient'] = coefficient_df['Coefficient'].abs()
coefficient_df = coefficient_df.sort_values('Absolute Coefficient', ascending=False).reset_index(drop=True)
coefficient_df['Importance Rank'] = np.arange(1, len(coefficient_df) + 1)
print('\nLinear Regression Coefficients:\n', coefficient_df.round(4).to_string(index=False))
plot_df = coefficient_df.head(10).iloc[::-1]
plt.figure(figsize=(12, 7), facecolor=BG_COLOR)
ax = plt.gca()
style_axes(ax)
set_chart_title(ax, 'Online News - Most Influential Article Features')
bars = ax.barh(plot_df['Feature'], plot_df['Absolute Coefficient'],
               color=[LIGHT_PINK if v < 0 else PURPLE for v in plot_df['Coefficient']], edgecolor='white')
maximum = max(float(coefficient_df['Absolute Coefficient'].max()), 1e-9)
for bar, signed in zip(bars, plot_df['Coefficient']):
    ax.text(bar.get_width() + maximum * 0.01, bar.get_y() + bar.get_height() / 2,
            f'{signed:+.2f}', ha='left', va='center', fontsize=9,
            color=TITLE_COLOR, fontweight='bold')
ax.set_xlim(0, maximum * 1.25)
ax.set_xlabel('Absolute Standardized Coefficient (Importance)', fontsize=12, color=TEXT_COLOR)
ax.set_ylabel('Feature', fontsize=12, color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
ax.legend(handles=[Patch(facecolor=PURPLE, label='Positive coefficient'),
                   Patch(facecolor=LIGHT_PINK, label='Negative coefficient')],
          loc='center right', facecolor=BG_COLOR, edgecolor=LAVENDER)
display_figure()
# endregion

# region Results
best_metrics = metrics_df.iloc[0].to_dict()
comparison_df = metrics_df.copy()
comparison_df['RMSE Rank'] = comparison_df['RMSE'].rank(method='min').astype(int)
comparison_df['Accuracy Rank'] = comparison_df['Accuracy'].rank(ascending=False, method='min').astype(int)
print('\nModel Comparison:\n', comparison_df.round(4).to_string(index=False))
print('\nBest Model Metrics:')
for metric in ['MAE', 'MSE', 'RMSE', 'R2', 'R2 Percentage', 'Accuracy']:
    print(f'{metric}: {best_metrics[metric]:.4f}')
# endregion

# region Correlation Heatmap
# Select 11 strongest target-correlated features for legibility at Concrete size.
feature_target_corr = news_df[FEATURES + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET)
heatmap_features = feature_target_corr.abs().sort_values(ascending=False).head(11).index.tolist()
correlation_columns = heatmap_features + [TARGET]
correlation_matrix = news_df[correlation_columns].corr(numeric_only=True)
plt.figure(figsize=(10.2, 6.2), dpi=100, facecolor=BG_COLOR)
ax = plt.gca()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f',
            cmap=sns.blend_palette(['#FFF7FB', '#F6CFE3', '#D69ACD', '#9562AF', '#51267D'], as_cmap=True),
            vmin=-1, vmax=1, center=0, linewidths=0.8, linecolor=BG_COLOR,
            cbar_kws={'label': 'Correlation'}, annot_kws={'fontsize': 8, 'fontweight': 'bold'}, ax=ax)
set_chart_title(ax, 'Online News - Correlation Heatmap')
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', rotation_mode='anchor')
ax.tick_params(axis='x', labelsize=8, colors=TEXT_COLOR)
ax.tick_params(axis='y', rotation=0, labelsize=8, colors=TEXT_COLOR)
ax.set_xlabel('Variables', fontsize=12, color=TEXT_COLOR)
ax.set_ylabel('Variables', fontsize=12, color=TEXT_COLOR)
display_figure()
# endregion

# region Data Visualization
plt.figure(figsize=(11, 7), facecolor=BG_COLOR)
ax = plt.gca()
style_axes(ax)
set_chart_title(ax, f'Online News - Actual vs Predicted Shares ({best_model_name})')
ax.scatter(y_test, best_prediction, color=TEAL, alpha=0.65, edgecolors=BG_COLOR, linewidths=0.5)
minimum = min(float(y_test.min()), float(best_prediction.min()))
maximum = max(float(y_test.max()), float(best_prediction.max()))
ax.plot([minimum, maximum], [minimum, maximum], color=PINK, linewidth=2, label='Ideal prediction')
ax.set_xlabel('Actual Shares', fontsize=12, color=TEXT_COLOR)
ax.set_ylabel('Predicted Shares', fontsize=12, color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
ax.legend(facecolor=BG_COLOR, edgecolor=LAVENDER)
display_figure()

plt.figure(figsize=(11, 7), facecolor=BG_COLOR)
ax = plt.gca()
style_axes(ax)
set_chart_title(ax, 'Online News - Regression Model Accuracy')
accuracy_plot = metrics_df.sort_values(['Accuracy', 'RMSE'], ascending=[False, True], kind='stable')
bars = ax.bar(accuracy_plot['Model'], accuracy_plot['Accuracy'],
              color=MODEL_COLORS[:len(accuracy_plot)], edgecolor='white', linewidth=0.8)
for bar in bars:
    value = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f'{value:.2f}%',
            ha='center', va='bottom', fontsize=10, color=TITLE_COLOR, fontweight='bold')
ax.set_xlabel('Model', fontsize=12, color=TEXT_COLOR)
ax.set_ylabel('Accuracy (%)', fontsize=12, color=TEXT_COLOR)
ax.set_ylim(0, 100)
ax.margins(x=0.08)
ax.tick_params(axis='x', rotation=20, colors=TEXT_COLOR)
ax.tick_params(axis='y', colors=TEXT_COLOR)
display_figure()

plt.figure(figsize=(11, 7), facecolor=BG_COLOR)
ax = plt.gca()
style_axes(ax)
set_chart_title(ax, f'Online News - Residual Distribution ({best_model_name})')
residuals = y_test.to_numpy() - best_prediction
ax.hist(residuals, bins=25, color=PURPLE, alpha=0.85, edgecolor='white')
ax.axvline(0, color=PINK, linewidth=2)
ax.set_xlabel('Residual (Actual - Predicted)', fontsize=12, color=TEXT_COLOR)
ax.set_ylabel('Frequency', fontsize=12, color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
display_figure()
# endregion

# region Save Results
output_folder = BASE_DIR / 'DataSets' / 'Processed'
output_folder.mkdir(parents=True, exist_ok=True)
processed_df = news_df.copy()
processed_df['Predicted_Shares'] = np.nan
processed_df.loc[y_test.index, 'Predicted_Shares'] = best_prediction
processed_path = output_folder / 'online_news_popularity_regression_results.csv'
processed_df.to_csv(processed_path, index=False)
metrics_path = output_folder / 'online_news_popularity_metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
print('\nRegression Results Saved Successfully!')
print('Processed Dataset:', processed_path)
print('Metrics:', metrics_path)
# endregion

# region Training Models
model_folder = BASE_DIR / 'ModelsOutcome'
model_folder.mkdir(parents=True, exist_ok=True)
feature_columns_path = model_folder / 'online_news_popularity_feature_columns.joblib'
joblib.dump(FEATURES, feature_columns_path)
model_paths = {}
for name, pipeline in trained_models.items():
    path = model_folder / ('online_news_popularity_' + name.lower().replace(' ', '_') + '.joblib')
    joblib.dump({'model': pipeline, 'model_name': name, 'target_column': TARGET,
                 'feature_columns': FEATURES, 'dataset_id': DATASET_ID}, path)
    model_paths[name] = path
    print('Saved Model:', path)
best_bundle = joblib.load(model_paths[best_model_name])
model_verification = bool(np.allclose(best_prediction, np.maximum(0.0, best_bundle['model'].predict(x_test))))
print('Model Verification:', model_verification)
# endregion

# region Save Metadata
metadata_folder = BASE_DIR / 'Metadata'
metadata_folder.mkdir(parents=True, exist_ok=True)
metadata = {
    'id': DATASET_ID,
    'run_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
    'name': DATASET_NAME,
    'level': 'Advanced',
    'domain': 'Business/Web',
    'task': 'Regression',
    'description': 'Predict the number of social-network shares of a Mashable article using pre-publication article features.',
    'source': 'UCI Online News Popularity Dataset',
    'dataset_statistics': {
        'rows': int(news_df.shape[0]), 'columns': int(news_df.shape[1]),
        'column_names': list(news_df.columns),
        'data_types': {c: str(t) for c, t in news_df.dtypes.items()},
        'missing_values': {c: int(v) for c, v in news_df.isna().sum().items()},
        'total_missing_values': int(news_df.isna().sum().sum()),
        'descriptive_statistics': json.loads(news_df.describe().round(2).to_json()),
    },
    'feature_selection': {
        'selected_features': FEATURES, 'target_column': TARGET,
        'excluded_columns': EXCLUDED,
    },
    'preprocessing': {
        'numeric_method': 'Training-only median imputation + StandardScaler',
        'split_method': 'Original article order 80/20, no shuffling',
    },
    'model_evaluation': {
        'test_size': TEST_SIZE, 'training_rows': len(y_train), 'test_rows': len(y_test),
        'metrics': ['MAE', 'MSE', 'RMSE', 'R2', 'R2 Percentage', 'Accuracy'],
        'results': json.loads(metrics_df.round(6).to_json(orient='records')),
        'recommended_model': best_model_name,
        'selection_rule': 'Highest Concrete-style MAPE-based Accuracy, lowest RMSE as tie-breaker',
        'accuracy_definition': 'max(0, 100 - 100 * mean_absolute_percentage_error(actual, predicted))',
        'accuracy_caveat': 'Descriptive regression score, not classification accuracy.',
    },
    'feature_importance': {
        'method': 'Standardized Linear Regression Coefficients',
        'model': 'Linear Regression',
        'results': json.loads(coefficient_df.round(4).to_json(orient='records')),
        'most_influential_feature': str(coefficient_df.iloc[0]['Feature']),
    },
    'model_verification': model_verification,
    'output_files': {
        'original_dataset': str(dataset_path.relative_to(BASE_DIR)),
        'processed_dataset': str(processed_path.relative_to(BASE_DIR)),
        'metrics': str(metrics_path.relative_to(BASE_DIR)),
        'feature_columns': str(feature_columns_path.relative_to(BASE_DIR)),
        'models': {name: str(path.relative_to(BASE_DIR)) for name, path in model_paths.items()},
        'metadata_latest': 'Metadata/D31_online_news_popularity.json',
    },
}
metadata_path = metadata_folder / 'D31_online_news_popularity.json'
with metadata_path.open('w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)
with metadata_path.open('r', encoding='utf-8') as f:
    saved_metadata = json.load(f)
print('\nMetadata Verification:')
print('Dataset Name:', saved_metadata['name'])
print('Total Rows:', saved_metadata['dataset_statistics']['rows'])
print('Total Columns:', saved_metadata['dataset_statistics']['columns'])
print('Target Column:', saved_metadata['feature_selection']['target_column'])
print('Recommended Model:', saved_metadata['model_evaluation']['recommended_model'])
print('Model Verification:', saved_metadata['model_verification'])
print('\nMetadata Process Completed Successfully!')
# endregion
