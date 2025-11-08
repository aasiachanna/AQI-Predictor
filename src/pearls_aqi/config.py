"""Configuration loading utilities for the Pearls AQI pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
	"""Paths used by the data processing pipeline."""

	raw_dir: Path
	processed_dir: Path
	processed_features_file: Path
	daily_history_file: Path


@dataclass
class ModelConfig:
	"""Model-specific configuration values."""

	target: str
	forecast_days: int
	random_seed: int
	test_fraction: float
	min_training_rows: int
	max_lag: int
	rolling_windows: tuple[int, ...]
	select_k_best: int


@dataclass
class PipelineConfig:
	"""Top-level configuration for the training pipeline."""

	data: DataConfig
	model: ModelConfig


DEFAULT_CONFIG_PATH = Path("config/config.yaml")


def _resolve_path(path_value: str, base_dir: Optional[Path] = None) -> Path:
	"""Resolve a path string to an absolute :class:`Path`.

	The function keeps paths relative to the project root to stay cross-platform.
	"""

	path = Path(path_value)
	if not path.is_absolute() and base_dir is not None:
		path = base_dir / path
	return path


def load_config(config_path: Optional[Path] = None) -> PipelineConfig:
	"""Load pipeline configuration from YAML."""

	config_path = config_path or DEFAULT_CONFIG_PATH
	
	# Resolve base directory: if config is absolute, use its parent's parent; otherwise use current working directory
	if config_path.is_absolute():
		base_dir = config_path.parent.parent
	else:
		# Try to find project root by looking for config directory
		current = Path.cwd()
		if (current / "config" / "config.yaml").exists():
			base_dir = current
		elif (current.parent / "config" / "config.yaml").exists():
			base_dir = current.parent
		else:
			base_dir = current
		config_path = base_dir / config_path

	with open(config_path, "r", encoding="utf-8") as fp:
		payload = yaml.safe_load(fp)

	data_section = payload.get("data", {})
	model_section = payload.get("model", {})

	data_cfg = DataConfig(
		raw_dir=_resolve_path(data_section.get("raw_dir", "data/raw"), base_dir),
		processed_dir=_resolve_path(data_section.get("processed_dir", "data/processed"), base_dir),
		processed_features_file=_resolve_path(
			data_section.get("processed_features_file", "data/processed/daily_features.csv"),
			base_dir,
		),
		daily_history_file=_resolve_path(
			data_section.get("daily_history_file", "data/processed/daily_history.csv"),
			base_dir,
		),
	)

	model_cfg = ModelConfig(
		target=model_section.get("target", "aqi"),
		forecast_days=int(model_section.get("forecast_days", 3)),
		random_seed=int(model_section.get("random_seed", 42)),
		test_fraction=float(model_section.get("test_fraction", 0.2)),
		min_training_rows=int(model_section.get("min_training_rows", 90)),
		max_lag=int(model_section.get("max_lag", 7)),
		rolling_windows=tuple(int(v) for v in model_section.get("rolling_windows", (3, 7))),
		select_k_best=int(model_section.get("select_k_best", 10)),
	)

	return PipelineConfig(data=data_cfg, model=model_cfg)

