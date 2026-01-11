import sys
from pathlib import Path
import math
import importlib.util

# make project root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# import the normalize_and_cache module by file path to avoid relying on package __init__
mod_path = ROOT / 'scripts' / 'tools' / 'normalize_and_cache.py'
spec = importlib.util.spec_from_file_location('normalize_and_cache', str(mod_path))
normalize_mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(normalize_mod)

extract_nutrients_from_cache_entry = normalize_mod.extract_nutrients_from_cache_entry
extract_nutrients_from_cache_entry_with_map = normalize_mod.extract_nutrients_from_cache_entry_with_map
_extract_nutrients_default = normalize_mod._extract_nutrients_default
suggest_nutrient_name_map = normalize_mod.suggest_nutrient_name_map
validate_nutrient_map = normalize_mod.validate_nutrient_map
conservative_fix_nutrient_map = normalize_mod.conservative_fix_nutrient_map


def approx(a, b, tol=1e-6):
    if a is None or b is None:
        return a == b
    return abs(a - b) <= tol


def test_extract_nutrients_default():
    entry = {
        "nutrients": [
            {"nutrientName": "Energy", "value": 200, "unitName": "kcal"},
            {"nutrientName": "Protein", "value": 10, "unitName": "g"},
            {"nutrientName": "Total lipid (fat)", "value": 5, "unitName": "g"},
            {"nutrientName": "Carbohydrate, by difference", "value": 30, "unitName": "g"},
            {"nutrientName": "Total dietary fiber", "value": 4, "unitName": "g"},
            {"nutrientName": "Sugars, total", "value": 8, "unitName": "g"},
            {"nutrientName": "Sodium, Na", "value": 0.5, "unitName": "g"},
            {"nutrientName": "Fatty acids, total saturated", "value": 2, "unitName": "g"},
        ]
    }

    out = _extract_nutrients_default(entry)
    assert out["calories"] == 200
    assert out["protein_g"] == 10
    assert out["fat_g"] == 5
    assert out["carbs_g"] == 30
    assert out["fiber_g"] == 4
    assert out["sugar_g"] == 8
    # sodium given in grams -> expect mg
    assert out["sodium_mg"] == 500.0
    assert out["saturated_fat_g"] == 2


def test_extract_nutrients_with_map_and_unit_conversion():
    # calories given in kJ should be converted to kcal
    entry = {
        "nutrients": [
            {"nutrientName": "Energy", "value": 1000, "unitName": "kJ"},
            {"nutrientName": "Protein", "value": 3.5, "unitName": "g"},
            {"nutrientName": "Sodium, Na", "value": 300, "unitName": "mg"},
        ]
    }
    nutrient_map = {
        "energy": "calories",
        "protein": "protein_g",
        "sodium, na": "sodium_mg",
    }
    out = extract_nutrients_from_cache_entry_with_map(entry, nutrient_map)
    # kJ -> kcal conversion factor ~0.239006
    expected_kcal = 1000 * 0.239006
    assert math.isclose(out["calories"], expected_kcal, rel_tol=1e-6)
    assert approx(out["protein_g"], 3.5)
    assert approx(out["sodium_mg"], 300.0)


def test_suggest_and_validate_and_conservative_fix():
    # build a small fake cache containing observed nutrient names
    cache = {
        "a": {"nutrients": [{"nutrientName": "Protein"}, {"nutrientName": "Energy"}]},
        "b": {"nutrients": [{"nutrientName": "Total Sugars"}, {"nutrientName": "Total dietary fiber"}]},
    }
    sugg = suggest_nutrient_name_map(cache)
    # expect some sensible mappings
    assert "Protein" in sugg and sugg["Protein"] == "protein_g"
    assert "Energy" in sugg and sugg["Energy"] == "calories"

    # validation: suspicious mappings
    nm = {
        "Biotin": "calories",
        "Vitamin C": "protein_g",
        "Sodium": "calories",
        "Fe": "protein_g",
    }
    issues = validate_nutrient_map(nm)
    observed = {o for o, _, _ in issues}
    # expect the suspicious keys to be flagged
    assert "Biotin" in observed
    assert "Vitamin C" in observed
    assert "Sodium" in observed
    assert "Fe" in observed

    fixed, changes = conservative_fix_nutrient_map(nm)
    # changes should correspond to issues and the fixed map should null those entries
    changed_obs = {c[0] for c in changes}
    assert changed_obs <= observed
    for ob in changed_obs:
        assert fixed.get(ob) is None

    # fixed map should have no validation issues
    assert validate_nutrient_map(fixed) == []
