import json
import math
from pathlib import Path

import joblib
import pandas as pd


class ClassificationResult(str):
    """Predicted class label (behaves like a normal str) plus class probabilities."""

    def __new__(cls, label, probabilities=None):
        obj = super().__new__(cls, label)
        obj.probabilities = list(probabilities or [])   # [(class_name, probability)] sorted desc
        obj.confidence = next((p for name, p in obj.probabilities if name == label), None)
        return obj


def _num(record, *names):
    """Case-insensitive numeric lookup in a metrics record."""
    lowered = {str(k).strip().lower().replace(' ', '_'): v for k, v in record.items()}
    for name in names:
        v = lowered.get(name)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
            return float(v)
    return None


class ClassificationData:
    def __init__(self, ds):
        path = Path(ds.path)
        self.meta = json.loads(path.read_text(encoding='utf-8-sig'))
        ev = self.meta.get('model_evaluation') or {}
        files = self.meta['output_files']
        rel = next((files[k] for k in ('models_bundle', 'model_bundle', 'combined_models', 'models')
                    if isinstance(files.get(k), str)), None)
        if not rel:
            raise ValueError('Model bundle path missing')
        bundle = joblib.load(path.parent.parent / rel.replace('\\', '/'))
        self.bundle = bundle

        records = (bundle.get('model_metrics') or bundle.get('metrics')
                   or bundle.get('evaluation', {}).get('validation_metrics') or ev.get('results') or [])
        # Same rule as the training script: highest weighted F1, accuracy breaks ties.
        self.results = sorted(
            records,
            key=lambda r: (-(_num(r, 'f1', 'f1_weighted') or 0.0), -(_num(r, 'accuracy') or 0.0)))
        if not self.results:
            raise ValueError('No model metrics found in metadata/bundle')

        models = bundle['models']
        name = ev.get('recommended_model') or bundle.get('best_model_name') or bundle.get('recommended_model')
        if not isinstance(name, str) or name not in models:
            name = str(self.results[0]['Model'])
        if name not in models:
            raise ValueError('Recommended model is missing from the saved bundle; regenerate metadata and bundle.')
        self.model = models[name]
        self.best = next((r for r in self.results if str(r.get('Model')) == name), self.results[0])

        self.features = list(bundle['feature_columns'])
        self.target = bundle['target_column']
        self.class_names = list(bundle.get('class_names') or [])
        self.code_to_class = dict(bundle.get('code_to_class') or {})
        self.stats = self.meta.get('dataset_statistics', {}).get('descriptive_statistics', {})
        self.categories = dict(self.meta.get('feature_categories', {}))

        # Real fitted encoder categories (preserve their types).
        for step in getattr(self.model, 'named_steps', {}).values():
            for _, transformer, columns in getattr(step, 'transformers_', []):
                for encoder in [transformer, *getattr(transformer, 'named_steps', {}).values()]:
                    if hasattr(encoder, 'categories_'):
                        for feature, options in zip(columns, encoder.categories_):
                            self.categories[feature] = list(options)

        self.categorical = set(bundle.get('categorical_features', [])) | set(self.categories)
        self.selection_rule = ev.get('selection_rule', 'Highest weighted F1')
        verification = self.meta.get('model_verification')
        self.verified = verification is True or (
            isinstance(verification, dict) and verification.get('all_models_verified') is True)

        # Statistics for engineered features that are missing from the raw metadata.
        missing = [f for f in self.features if f not in self.stats and f not in self.categories]
        processed = files.get('processed_dataset')
        if missing and isinstance(processed, str):
            p = path.parent.parent / processed.replace('\\', '/')
            if p.is_file():
                frame = pd.read_csv(p, usecols=lambda c: c in missing)
                for f in frame:
                    self.stats[f] = pd.to_numeric(frame[f], errors='coerce').describe().to_dict()

    # ---- metrics (always as percentages) ----
    @staticmethod
    def _pct(record, pct_key, raw_key):
        v = _num(record, pct_key)
        if v is not None:
            return v
        v = _num(record, raw_key)
        if v is None:
            return None
        return v * 100 if 0 <= v <= 1 else v

    @property
    def accuracy_pct(self):
        return self._pct(self.best, 'accuracy_percentage', 'accuracy')

    @property
    def f1_pct(self):
        return self._pct(self.best, 'f1_percentage', 'f1')

    # ---- prediction ----
    def _label(self, value):
        if value in self.code_to_class:
            return str(self.code_to_class[value])
        try:
            iv = int(value)
            if iv in self.code_to_class:
                return str(self.code_to_class[iv])
            if not self.code_to_class and 0 <= iv < len(self.class_names) and not isinstance(value, str):
                return str(self.class_names[iv])
        except (TypeError, ValueError):
            pass
        return str(value)

    def predict(self, values):
        row = {}
        for f in self.features:
            v = values[f]
            if f in self.categories:
                row[f] = next((x for x in self.categories[f] if str(x) == str(v)), None)
                if row[f] is None:
                    raise ValueError(f'Invalid category: {f}')
            elif f in self.categorical:
                if not str(v).strip():
                    raise ValueError(f'Enter {f}')
                row[f] = str(v)
            else:
                row[f] = float(v)
                if not math.isfinite(row[f]):
                    raise ValueError(f'Invalid {f}')
        frame = pd.DataFrame([row], columns=self.features)
        label = self._label(self.model.predict(frame)[0])

        probabilities = []
        if hasattr(self.model, 'predict_proba'):
            try:
                proba = self.model.predict_proba(frame)[0]
                probabilities = sorted(
                    ((self._label(c), float(p)) for c, p in zip(self.model.classes_, proba)),
                    key=lambda item: -item[1])
            except Exception:
                probabilities = []
        return ClassificationResult(label, probabilities)