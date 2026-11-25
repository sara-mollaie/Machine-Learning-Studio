import json
from pathlib import Path
import math
import joblib
import pandas as pd

class RegressionData:
    def __init__(self, ds):
        path = Path(ds.path)
        self.meta = json.loads(path.read_text(encoding='utf-8-sig'))
        ev = self.meta['model_evaluation']
        files = self.meta['output_files']
        rel = next((files[k] for k in ('models_bundle','model_bundle','combined_models','models') if isinstance(files.get(k), str)), None)
        if not rel:
            raise ValueError('Model bundle path missing')
        bundle_path = path.parent.parent / rel.replace('\\', '/')
        bundle = joblib.load(bundle_path)
        self.bundle = bundle
        records = bundle.get('metrics') or bundle.get('model_metrics') or bundle.get('evaluation', {}).get('validation_metrics') or ev['results']
        self.results = sorted(records, key=lambda r: float(r['RMSE']))
        self.best = self.results[0]
        name = bundle.get('recommended_model') or bundle.get('best_model_name') or bundle.get('best_model')
        if not isinstance(name, str) or name != self.best['Model'] or ev.get('recommended_model') != name:
            raise ValueError('Recommended model does not match saved RMSE ranking; regenerate metadata and bundle.')
        self.model = bundle['models'][name]
        self.features = list(bundle['feature_columns'])
        self.target = bundle['target_column']
        self.stats = self.meta.get('dataset_statistics', {}).get('descriptive_statistics', {})
        self.categories = dict(self.meta.get('feature_categories', {}))
        # Read actual fitted encoder categories, preserving numeric category types.
        for step in getattr(self.model, 'named_steps', {}).values():
            for _, transformer, columns in getattr(step, 'transformers_', []):
                for encoder in [transformer, *getattr(transformer, 'named_steps', {}).values()]:
                    if hasattr(encoder, 'categories_'):
                        for feature, options in zip(columns, encoder.categories_):
                            self.categories[feature] = list(options)
        if bundle.get('dataset_name') == 'Bike Sharing':
            self.categories.update({'season':[1,2,3,4], 'yr':[0,1], 'mnth':list(range(1,13)), 'hr':list(range(24)), 'holiday':[0,1], 'weekday':list(range(7)), 'workingday':[0,1], 'weathersit':[1,2,3,4]})
        self.categorical = set(bundle.get('categorical_features', [])) | set(self.categories)
        self.selection_rule = ev.get('selection_rule', 'Lowest RMSE')
        self.clip_min = bundle.get('prediction_postprocessing', {}).get('clip_min')
        verification = self.meta.get('model_verification')
        self.verified = verification is True or (isinstance(verification, dict) and verification.get('all_models_verified') is True)
        # Processed feature statistics for engineered inputs missing from raw metadata.
        missing = [f for f in self.features if f not in self.stats and f not in self.categories]
        processed = files.get('processed_dataset')
        if missing and isinstance(processed, str):
            p = path.parent.parent / processed.replace('\\', '/')
            if p.is_file():
                frame = pd.read_csv(p, usecols=lambda c: c in missing)
                for f in frame:
                    self.stats[f] = pd.to_numeric(frame[f], errors='coerce').describe().to_dict()

    def predict(self, values):
        row = {}
        for f in self.features:
            v = values[f]
            if f in self.categories:
                row[f] = next((x for x in self.categories[f] if str(x) == str(v)), None)
                if row[f] is None: raise ValueError(f'Invalid category: {f}')
            elif f in self.categorical:
                if not str(v).strip(): raise ValueError(f'Enter {f}')
                row[f] = str(v)
            else:
                row[f] = float(v)
                if not math.isfinite(row[f]): raise ValueError(f'Invalid {f}')
        result = float(self.model.predict(pd.DataFrame([row], columns=self.features))[0])
        if not math.isfinite(result): raise ValueError('Non-finite prediction')
        return max(float(self.clip_min), result) if self.clip_min is not None else result
