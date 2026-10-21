"""The playground must reject misspelled config keys instead of ignoring them."""

from pathlib import Path

import pytest
import yaml

from run import apply_overrides, validate

CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def load(*overrides):
    with open(CONFIG, encoding="utf-8") as f:
        return apply_overrides(yaml.safe_load(f), list(overrides))


def test_default_config_is_valid():
    validate(load())


def test_typo_in_neuron_key_is_rejected():
    with pytest.raises(SystemExit, match="v_threshold"):
        validate(load("neuron.v_treshold=-40"))


def test_typo_in_section_is_rejected():
    with pytest.raises(SystemExit, match="simulation"):
        validate(load("simulaton.n_trials=5"))
