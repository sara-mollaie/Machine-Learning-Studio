import math

def ranked_coefficients(meta):
    rows = []
    for r in (meta.get('feature_importance') or {}).get('results', []):
        try:
            value = float(r['Coefficient'])
            if math.isfinite(value): rows.append((str(r['Feature']), value))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(sorted(rows, key=lambda r: abs(r[1]), reverse=True)[:10], key=lambda r: r[1], reverse=True)
