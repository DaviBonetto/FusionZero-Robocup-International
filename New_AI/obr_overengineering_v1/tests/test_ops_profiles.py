from __future__ import annotations

from src.ops_profiles import load_ops_profile_catalog


def test_ops_profile_catalog_loads_named_presets() -> None:
    catalog = load_ops_profile_catalog("New_AI/obr_overengineering_v1/configs/vision_config.json")
    assert catalog.default_profile_name == "lab_pc"
    assert set(catalog.profiles) >= {"lab_pc", "pi3_field", "rescue_test", "line_only"}
    assert catalog.profiles["pi3_field"].camera["fps"] == 20
    assert catalog.profiles["rescue_test"].recording["auto_start"] is True
    assert catalog.default_tuning["green.s_min"] == 70
