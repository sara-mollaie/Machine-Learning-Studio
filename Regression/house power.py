"""D30 HousePower — next-hour regression.
Place household_power_consumption.txt in DataSets/ next to this script.
Requires: numpy, pandas, matplotlib, seaborn, scikit-learn, joblib.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

BASE_DIR = Path(__file__).resolve().parent
DATASET_ID = 'D30'
DATASET_NAME = 'Household Power'
SOURCE_DATASET_NAME = 'Individual Household Electric Power Consumption'
RAW_TARGET = 'Global_active_power'
TARGET = 'Next_Hour_Global_Active_Power'
RANDOM_STATE = 42
SHOW_FIGURES = True
# The final test period is never used to choose the recommended model.
TRAIN_FRACTION, VALIDATION_FRACTION = 0.70, 0.15
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
        x = max(0, (window.winfo_screenwidth() - width) // 2)
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
            width = title.get_window_extent(renderer).width
            allowed = fig.bbox.width * 0.91
            if width > allowed:
                title.set_fontsize(max(10, title.get_fontsize() * allowed / width))
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    show_figure()
# endregion

# region Load Dataset
folder = BASE_DIR / 'DataSets'
paths = [folder / 'household_power_consumption.txt',
         folder / 'household_power_consumption.csv',
         folder / 'Individual Household Electric Power Consumption' / 'household_power_consumption.txt']
dataset_path = next((p for p in paths if p.is_file()), None)
if dataset_path is None:
    raise FileNotFoundError(f'Dataset missing. Place household_power_consumption.txt in {folder}')

RAW_COLUMNS = ['Global_active_power', 'Global_reactive_power', 'Voltage',
               'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
# Chunked loading avoids retaining 2 million minute-level rows in memory.
hourly_parts = []
raw_row_count = 0
raw_missing_counts = {name: 0 for name in RAW_COLUMNS}
for chunk in pd.read_csv(dataset_path, sep=';', na_values=['?', ''],
                         chunksize=200_000, low_memory=False):
    chunk.columns = chunk.columns.astype(str).str.strip()
    needed = {'Date', 'Time', *RAW_COLUMNS}
    missing = needed - set(chunk.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}. Found: {list(chunk.columns)}')
    raw_row_count += len(chunk)
    chunk['Datetime'] = pd.to_datetime(chunk['Date'].astype(str) + ' ' + chunk['Time'].astype(str),
                                       format='%d/%m/%Y %H:%M:%S', errors='coerce')
    for col in RAW_COLUMNS:
        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        raw_missing_counts[col] += int(chunk[col].isna().sum())
    chunk = chunk.dropna(subset=['Datetime']).set_index('Datetime')
    # Sum/count are combined across chunks so boundary hours remain correct.
    grouped = chunk[RAW_COLUMNS].resample('h').agg(['sum', 'count'])
    hourly_parts.append(grouped)

combined = pd.concat(hourly_parts).groupby(level=0).sum().sort_index()
hourly_df = pd.DataFrame(index=combined.index)
for col in RAW_COLUMNS:
    count = combined[(col, 'count')]
    hourly_df[col] = (combined[(col, 'sum')] / count.where(count > 0)).astype(float)
# Include all hours to preserve lag meaning even when a timestamp is absent.
hourly_df = hourly_df.asfreq('h')
print('\nDataset Loaded Successfully!')
print('Dataset Path:', dataset_path)
print('Raw Minute-Level Rows:', raw_row_count)
print('Hourly Rows:', len(hourly_df))
# endregion

# region Data Understanding
print('\nFirst 5 Rows:\n', hourly_df.head().to_string())
print('-' * 60)
print('Dataset Shape:', hourly_df.shape)
print('Columns:', hourly_df.columns.tolist())
print('Data Types:\n', hourly_df.dtypes)
print('\nDataset Information:')
hourly_df.info()
print('Statistical Description:\n', hourly_df.describe().to_string())
print('Hourly Missing Values:\n', hourly_df.isna().sum().to_string())
print('Raw Missing Values:', raw_missing_counts)
print('Total Hourly Missing Values:', int(hourly_df.isna().sum().sum()))
print('Duplicate Timestamps:', int(hourly_df.index.duplicated().sum()))
print('-' * 60)
# endregion

# region Feature Selection / Time Series Feature Engineering
# At timestamp t, features use data through hour t; target is hour t+1.
# Rolling features are shifted by one hour so their historical window is explicit.
feature_df = pd.DataFrame(index=hourly_df.index)
feature_df['Hour'] = hourly_df.index.hour
feature_df['Day_Of_Week'] = hourly_df.index.day_of_week
feature_df['Month'] = hourly_df.index.month
feature_df['Is_Weekend'] = (hourly_df.index.day_of_week >= 5).astype(int)
feature_df['Hour_Sin'] = np.sin(2 * np.pi * hourly_df.index.hour / 24)
feature_df['Hour_Cos'] = np.cos(2 * np.pi * hourly_df.index.hour / 24)
for lag in (1, 2, 24, 168):
    feature_df[f'Power_Lag_{lag}h'] = hourly_df[RAW_TARGET].shift(lag)
# Current-hour power is available only after hour t has finished.
feature_df['Power_Current_Hour'] = hourly_df[RAW_TARGET]
for window in (3, 24, 168):
    feature_df[f'Power_Rolling_Mean_{window}h'] = hourly_df[RAW_TARGET].shift(1).rolling(window, min_periods=window).mean()
for col in RAW_COLUMNS[1:]:
    feature_df[f'Current_{col}'] = hourly_df[col]
feature_df[TARGET] = hourly_df[RAW_TARGET].shift(-1)
FEATURES = [c for c in feature_df.columns if c != TARGET]
# Rows without a measured future target cannot be evaluated or trained.
model_df = feature_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET]).copy()
if len(model_df) < 100:
    raise ValueError('Not enough valid hourly observations for modeling.')
X = model_df[FEATURES]
y = model_df[TARGET]
print('\nSelected Features:', FEATURES)
print('Feature Shape:', X.shape)
print('Target:', TARGET)
print('Excluded From Features:', [TARGET, 'Date', 'Time'])
# endregion

# region Data Preprocessing
if X.isna().all().any():
    raise ValueError('At least one feature contains only missing values.')
print('\nMissing Values In Features:\n', X.isna().sum().to_string())
print('Missing Values In Target:', int(y.isna().sum()))
print('Preprocessing: training-only median imputation')
# endregion

# region Data Normalization
def create_preprocessor():
    return Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler())])
print('Normalization: StandardScaler fitted on training data only')
# endregion

# region Model Evaluation
n = len(X)
train_end = int(n * TRAIN_FRACTION)
val_end = int(n * (TRAIN_FRACTION + VALIDATION_FRACTION))
x_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
x_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
x_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
if min(len(x_train), len(x_val), len(x_test)) == 0:
    raise ValueError('Empty chronological split.')

# Use the training target range as a fixed scale for validation and test.
# This avoids using validation/test target statistics to define the score.
RMSE_SCALE = float(y_train.max() - y_train.min())
if not np.isfinite(RMSE_SCALE) or RMSE_SCALE <= 0:
    raise ValueError('Training target range must be greater than zero.')

ACCURACY_DEFINITION = (
    'RMSE-based normalized score = max(0, 100 * (1 - RMSE / training_target_range)); '
    'training_target_range is computed from the training partition only. '
    'This is a project comparison score, not a standard regression accuracy metric.'
)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=120, random_state=RANDOM_STATE,
                                           n_jobs=-1, min_samples_leaf=2),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.05,
                                                    max_depth=3, random_state=RANDOM_STATE),
    'SVR': SVR(kernel='rbf', C=10.0, epsilon=0.1),
}

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mse = mean_squared_error(actual, predicted)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(actual, predicted))
    rmse_accuracy = float(np.clip(
        100.0 * (1.0 - rmse / RMSE_SCALE),
        0.0,
        100.0,
    ))
    return {
        'MAE': float(mean_absolute_error(actual, predicted)),
        'MSE': float(mse),
        'RMSE': rmse,
        'R2': r2,
        'R2 Percentage': r2 * 100.0,
        'Accuracy': rmse_accuracy,
    }

trained_models, validation_predictions, validation_records = {}, {}, []
for name, estimator in models.items():
    pipeline = Pipeline([('preprocessor', create_preprocessor()), ('model', estimator)])
    # SVR scales poorly; cap its training history without altering validation/test sets.
    train_x, train_y = (x_train.iloc[-12000:], y_train.iloc[-12000:]) if name == 'SVR' else (x_train, y_train)
    print('Training:', name, '| rows:', len(train_x))
    pipeline.fit(train_x, train_y)
    prediction = pipeline.predict(x_val)
    trained_models[name] = pipeline
    validation_predictions[name] = prediction
    validation_records.append({'Model': name, **calculate_metrics(y_val, prediction)})

# Selection uses validation only, and matches the Accuracy chart.
metrics_df = pd.DataFrame(validation_records).sort_values(
    ['Accuracy', 'RMSE'], ascending=[False, True], kind='stable'
).reset_index(drop=True)
best_model_name = str(metrics_df.iloc[0]['Model'])
best_model = trained_models[best_model_name]
best_prediction = best_model.predict(x_test)
test_metrics = calculate_metrics(y_test, best_prediction)
print('\nValidation Regression Metrics:\n', metrics_df.round(4).to_string(index=False))
print('Recommended Model:', best_model_name)
print('Selection Rule: Lowest validation RMSE (equivalent to highest RMSE-based score)')
print('Final Test Metrics:', {k: round(v, 4) for k, v in test_metrics.items()})
# endregion

# region Feature Importance
# Standardized coefficients from linear regression (association, not causal effects).
linear_model = trained_models['Linear Regression'].named_steps['model']
coefficient_df = pd.DataFrame({'Feature': FEATURES, 'Coefficient': linear_model.coef_})
coefficient_df['Absolute Coefficient'] = coefficient_df['Coefficient'].abs()
coefficient_df = coefficient_df.sort_values(
    'Absolute Coefficient', ascending=False, kind='stable'
).reset_index(drop=True)
coefficient_df['Importance Rank'] = np.arange(1, len(coefficient_df) + 1)
print('\nFeature Importance (Standardized Linear Coefficients):\n',
      coefficient_df.round(4).to_string(index=False))
print('Most Influential Feature:', coefficient_df.iloc[0]['Feature'])

# In the chart, show positive coefficients first (largest to smallest),
# followed by negative coefficients (largest absolute value to smallest).
positive_coefficients = coefficient_df[coefficient_df['Coefficient'] >= 0].sort_values(
    'Coefficient', ascending=False, kind='stable'
)
negative_coefficients = coefficient_df[coefficient_df['Coefficient'] < 0].sort_values(
    'Absolute Coefficient', ascending=False, kind='stable'
)
feature_chart_order = pd.concat(
    [positive_coefficients, negative_coefficients], ignore_index=True
).head(12)
plot_coefficients = feature_chart_order.iloc[::-1]
fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG_COLOR)
style_axes(ax)
set_chart_title(ax, 'Household Power - Most Influential Consumption Features')
colors = [LIGHT_PINK if v < 0 else PURPLE for v in plot_coefficients['Coefficient']]
bars = ax.barh(plot_coefficients['Feature'], plot_coefficients['Absolute Coefficient'],
               color=colors, edgecolor='white')
max_value = max(float(coefficient_df['Absolute Coefficient'].max()), 0.001)
for bar, coef in zip(bars, plot_coefficients['Coefficient']):
    ax.text(bar.get_width() + max_value * 0.015, bar.get_y() + bar.get_height() / 2,
            f'{coef:+.3f}', va='center', fontsize=9, color=TITLE_COLOR, fontweight='bold')
ax.set_xlim(0, max_value * 1.27)
ax.set_xlabel('Absolute Standardized Coefficient', color=TEXT_COLOR)
ax.set_ylabel('Feature', color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
ax.legend(handles=[Patch(facecolor=PURPLE, label='Positive coefficient'),
                   Patch(facecolor=LIGHT_PINK, label='Negative coefficient')],
          loc='lower right', facecolor=BG_COLOR, edgecolor=LAVENDER)
display_figure()
# endregion

# region Results
best_metrics = metrics_df.iloc[0].to_dict()
comparison_df = metrics_df.copy()
comparison_df['RMSE Rank'] = comparison_df['RMSE'].rank(method='min').astype(int)
comparison_df['Accuracy Rank'] = comparison_df['Accuracy'].rank(ascending=False, method='min').astype(int)
print('\nModel Comparison:\n', comparison_df.round(4).to_string(index=False))
print('Best Validation Model:', best_model_name)
print('Best Validation RMSE-based Score:', f"{best_metrics['Accuracy']:.2f}%")
# endregion

# region Correlation Heatmap
# Show the most relevant predictors to avoid illegible 20+ column labels.
corr_values = model_df[FEATURES + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET).abs()
correlation_columns = corr_values.nlargest(9).index.tolist() + [TARGET]
correlation_matrix = model_df[correlation_columns].corr()
fig, ax = plt.subplots(figsize=(10.2, 6.2), dpi=100, facecolor=BG_COLOR)
sns.heatmap(correlation_matrix, annot=True, fmt='.2f',
            cmap=sns.blend_palette(['#FFF7FB', '#F6CFE3', '#D69ACD', '#9562AF', '#51267D'], as_cmap=True),
            vmin=-1, vmax=1, center=0, linewidths=0.8, linecolor=BG_COLOR,
            cbar_kws={'label': 'Correlation'}, annot_kws={'fontsize': 8, 'fontweight': 'bold'}, ax=ax)
set_chart_title(ax, 'Household Power - Correlation Heatmap')
short_names = [c.replace('Global_', '').replace('Power_', '').replace('Sub_metering_', 'Sub_')
               .replace('Next_Hour_', 'Next_') for c in correlation_columns]
ax.set_xticklabels(short_names, rotation=35, ha='right', rotation_mode='anchor', fontsize=8)
ax.set_yticklabels(short_names, rotation=0, fontsize=8)
ax.set_xlabel('Variables', color=TEXT_COLOR)
ax.set_ylabel('Variables', color=TEXT_COLOR)
display_figure()
# endregion

# region Data Visualization
fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG_COLOR)
style_axes(ax)
set_chart_title(ax, f'Household Power - Actual vs Predicted Next Hour ({best_model_name})')
ax.scatter(y_test, best_prediction, color=TEAL, alpha=0.55, s=15, edgecolors=BG_COLOR, linewidths=0.3)
minimum = min(float(y_test.min()), float(best_prediction.min()))
maximum = max(float(y_test.max()), float(best_prediction.max()))
ax.plot([minimum, maximum], [minimum, maximum], color=PINK, linewidth=2, label='Ideal prediction')
ax.set_xlabel('Actual Next-Hour Power (kW)', color=TEXT_COLOR)
ax.set_ylabel('Predicted Next-Hour Power (kW)', color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
ax.legend(facecolor=BG_COLOR, edgecolor=LAVENDER)
display_figure()

fig, ax = plt.subplots(figsize=(13, 8), facecolor=BG_COLOR)
style_axes(ax)
set_chart_title(ax, 'Household Power - Regression Model Accuracy (RMSE-based)')
accuracy_plot = metrics_df.sort_values(
    ['Accuracy', 'RMSE'], ascending=[False, True], kind='stable'
).reset_index(drop=True)
bars = ax.bar(
    accuracy_plot['Model'],
    accuracy_plot['Accuracy'],
    color=MODEL_COLORS[:len(accuracy_plot)],
    edgecolor='white',
    linewidth=0.8,
)
for bar in bars:
    value = float(bar.get_height())
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1.0,
        f'{value:.2f}%',
        ha='center',
        va='bottom',
        fontsize=11,
        color=TITLE_COLOR,
        fontweight='bold',
    )
ax.set_xlabel('Model', color=TEXT_COLOR)
ax.set_ylabel('Accuracy (%)', color=TEXT_COLOR)
ax.set_ylim(0, min(110, max(100, float(accuracy_plot['Accuracy'].max()) + 8)))
ax.set_yticks(np.arange(0, 101, 20))
ax.margins(x=0.08)
ax.tick_params(axis='x', rotation=20, colors=TEXT_COLOR, labelsize=10)
ax.tick_params(axis='y', colors=TEXT_COLOR)
display_figure()

fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG_COLOR)
style_axes(ax)
set_chart_title(ax, f'Household Power - Residual Distribution ({best_model_name})')
residuals = y_test.to_numpy() - best_prediction
ax.hist(residuals, bins=30, color=PURPLE, alpha=0.85, edgecolor='white')
ax.axvline(0, color=PINK, linewidth=2)
ax.set_xlabel('Residual (Actual - Predicted) kW', color=TEXT_COLOR)
ax.set_ylabel('Frequency', color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
display_figure()
# endregion

# region Save Results
output_folder = BASE_DIR / 'DataSets' / 'Processed'
output_folder.mkdir(parents=True, exist_ok=True)
processed_df = pd.DataFrame({'Datetime': model_df.index[val_end:],
                             'Actual_Next_Hour_Power_kW': y_test.to_numpy(),
                             'Predicted_Next_Hour_Power_kW': best_prediction})
processed_path = output_folder / 'household_power_regression_results.csv'
processed_df.to_csv(processed_path, index=False)
metrics_path = output_folder / 'household_power_validation_metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
print('Saved RMSE-based validation accuracy (%) for all five models.')
print('\nRegression Results Saved Successfully!')
print('Processed Dataset:', processed_path)
print('Validation Metrics:', metrics_path)
# endregion

# region Training Models
model_folder = BASE_DIR / 'ModelsOutcome'
model_folder.mkdir(parents=True, exist_ok=True)

feature_columns_path = model_folder / 'household_power_feature_columns.joblib'
models_bundle_path = model_folder / 'household_power_regression_models.joblib'

models_bundle = {
    'dataset_id': DATASET_ID,
    'dataset_name': DATASET_NAME,
    'source_dataset_name': SOURCE_DATASET_NAME,
    'target_column': TARGET,
    'raw_target_column': RAW_TARGET,
    'feature_columns': FEATURES,
    'forecast_horizon_hours': 1,
    'models': trained_models,
    'model_metrics': metrics_df.to_dict(orient='records'),
    'best_model_name': best_model_name,
    'best_model': best_model,
    'best_validation_metrics': best_metrics,
    'test_metrics': test_metrics,
    'accuracy_definition': ACCURACY_DEFINITION,
    'rmse_scale_training_range': RMSE_SCALE,
    'split_method': 'Chronological 70/15/15 train/validation/test',
    'random_state': RANDOM_STATE,
}

joblib.dump(models_bundle, models_bundle_path)
joblib.dump(FEATURES, feature_columns_path)

# Verify every saved model on the same validation rows and the selected model on test rows.
loaded_bundle = joblib.load(models_bundle_path)
all_model_checks = []
for name in trained_models:
    saved_validation_prediction = loaded_bundle['models'][name].predict(x_val)
    all_model_checks.append(
        np.allclose(
            validation_predictions[name],
            saved_validation_prediction,
            rtol=1e-7,
            atol=1e-7,
        )
    )
saved_best_test_prediction = loaded_bundle['best_model'].predict(x_test)
all_model_checks.append(
    np.allclose(best_prediction, saved_best_test_prediction, rtol=1e-7, atol=1e-7)
)
model_verification = bool(all(all_model_checks))

# Remove only legacy individual files generated by this module.
# Other datasets' model files are not touched.
legacy_model_paths = [
    model_folder / ('household_power_' + name.lower().replace(' ', '_') + '.joblib')
    for name in models
]
for legacy_path in legacy_model_paths:
    if legacy_path.exists():
        legacy_path.unlink()

print('Saved all five models in one bundle:', models_bundle_path)
print('Saved feature columns:', feature_columns_path)
print('Saved model names:', list(loaded_bundle['models'].keys()))
print('Model Verification:', model_verification)
# endregion

# region Save Metadata
metadata_folder = BASE_DIR / 'Metadata'
metadata_folder.mkdir(parents=True, exist_ok=True)
metadata = {
    "correlation_matrix": json.loads((correlation_matrix).to_json()),
    'id': DATASET_ID,
    'run_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
    'name': DATASET_NAME,
    'source_dataset_name': SOURCE_DATASET_NAME,
    'title': 'Household Power (D30)',
    'level': 'Advanced', 'domain': 'Electricity', 'task': 'Regression',
    'description': 'Predict next-hour average household active power (kW) using historical hourly observations.',
    'source': 'UCI Individual Household Electric Power Consumption',
    'dataset_statistics': {
        'rows': raw_row_count, 'columns': 9,
        'column_names': ['Date', 'Time'] + RAW_COLUMNS,
        'raw_missing_values': raw_missing_counts,
        'hourly_rows': int(len(hourly_df)), 'model_rows': int(len(model_df)),
        'data_types': {c: str(t) for c, t in hourly_df.dtypes.items()},
        'missing_values': {c: int(v) for c, v in hourly_df.isna().sum().items()},
        'total_missing_values': int(hourly_df.isna().sum().sum()),
        'descriptive_statistics': json.loads(hourly_df.describe().round(4).to_json()),
    },
    'feature_selection': {'selected_features': FEATURES, 'target_column': TARGET,
                          'raw_target_column': RAW_TARGET,
                          'excluded_columns': ['Date', 'Time', TARGET]},
    'preprocessing': {'resampling': 'Hourly means',
                      'numeric_method': 'Training-only median imputation + StandardScaler',
                      'missing_value_method': 'Training-set median'},
    'model_evaluation': {
        'split_method': 'Chronological 70/15/15 train/validation/test',
        'training_rows': len(y_train), 'validation_rows': len(y_val), 'test_rows': len(y_test),
        'metrics': ['MAE', 'MSE', 'RMSE', 'R2', 'R2 Percentage', 'Accuracy'],
        'results': json.loads(metrics_df.round(5).to_json(orient='records')),
        'recommended_model': best_model_name,
        'selection_rule': 'Lowest validation RMSE; equivalent to highest RMSE-based normalized score',
        'accuracy_definition': ACCURACY_DEFINITION,
        'rmse_scale_training_range': RMSE_SCALE,
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'svr_training_cap': 12000,
    },
    'feature_importance': {
        'method': 'Standardized Linear Regression Coefficients',
        'model': 'Linear Regression',
        'results': json.loads(coefficient_df.round(5).to_json(orient='records')),
        'most_influential_feature': str(coefficient_df.iloc[0]['Feature']),
    },
    'model_verification': model_verification,
    'ui_card': {
        'title': 'Household Power (D30)',
        'category': 'Electricity',
        'task': 'Regression',
        'level': 'Advanced',
        'rows': int(len(model_df)),
        'columns': int(len(FEATURES) + 1),
    },
    'output_files': {
        'original_dataset': str(dataset_path.relative_to(BASE_DIR).as_posix()),
        'processed_dataset': str(processed_path.relative_to(BASE_DIR).as_posix()),
        'metrics': str(metrics_path.relative_to(BASE_DIR).as_posix()),
        'metadata_latest': 'Metadata/D30_household_power.json',
        'feature_columns': str(feature_columns_path.relative_to(BASE_DIR).as_posix()),
        'models_bundle': str(models_bundle_path.relative_to(BASE_DIR).as_posix()),
    },
}
metadata_file = metadata_folder / 'D30_household_power.json'
with metadata_file.open('w', encoding='utf-8') as file:
    json.dump(metadata, file, ensure_ascii=False, indent=4, allow_nan=False)
with metadata_file.open('r', encoding='utf-8') as file:
    saved_metadata = json.load(file)
print('\nMetadata Verification:')
print('Dataset Name:', saved_metadata['name'])
print('Total Rows:', saved_metadata['dataset_statistics']['rows'])
print('Total Columns:', saved_metadata['dataset_statistics']['columns'])
print('Target Column:', saved_metadata['feature_selection']['target_column'])
print('Recommended Model:', saved_metadata['model_evaluation']['recommended_model'])
print('RMSE-based Accuracy:', f"{best_metrics['Accuracy']:.2f}%")
print('Accuracy Formula:', ACCURACY_DEFINITION)
print('Model Verification:', saved_metadata['model_verification'])
print('Metadata Process Completed Successfully!')
# endregion
