"""
Tests for the ML predictor pipeline.

Sensor pattern used throughout:
  - BME680 gas resistance: high (~35k Ω) = clean air, low (~8k Ω) = poor air
  - Humidity > 60% = high, 40-55% = normal
  - Temperature ~22°C = normal
  - Episodes (cough/wheeze) are planted 5-8 min after gas drops into a bad window
"""

import time
import numpy as np
import pandas as pd
import pytest

from predictor import (
    Config,
    DatasetBuilder,
    BaselineGate,
    Trainer,
    build_feature_vector,
    threshold_breaches,
    FEATURE_COLS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

NOW = time.time()
DAY = 86400
INTERVAL = 30   # sensor reading every 30 s


def _sensor_row(ts, gas, humidity, temperature):
    return {"timestamp_epoch": ts, "gas": gas, "humidity": humidity, "temperature": temperature}


def make_sensors(days=4, bad_windows=None):
    """
    Generate sensor readings over `days` days.
    bad_windows: list of (start_offset_from_now, duration_s) tuples that insert
                 low-gas / high-humidity readings to simulate poor air quality.
    """
    bad_windows = bad_windows or []
    start = NOW - days * DAY
    rows = []
    ts = start
    while ts <= NOW:
        # Default: clean air
        gas = 35000
        humidity = 48
        temperature = 22.0

        for (w_start, w_dur) in bad_windows:
            w_begin = NOW - w_start
            if w_begin <= ts <= w_begin + w_dur:
                gas = 8000
                humidity = 68
                temperature = 24.0
                break

        rows.append(_sensor_row(float(ts), gas, humidity, temperature))
        ts += INTERVAL

    return pd.DataFrame(rows).sort_values("timestamp_epoch").reset_index(drop=True)


def make_episodes(episode_times, label="cough"):
    """
    episode_times: list of epoch timestamps for episode_start_epoch.
    """
    rows = []
    for t in episode_times:
        rows.append({
            "label": label,
            "episode_start_epoch": float(t),
            "episode_end_epoch":   float(t + 8.0),
            "duration_s":          8.0,
            "detection_count":     5,
            "peak_confidence":     0.75,
            "source":              "yamnet",
        })
    return pd.DataFrame(rows)


def _cfg(**kwargs):
    cfg = Config()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# BaselineGate
# ─────────────────────────────────────────────────────────────────────────────

class TestBaselineGate:
    def test_passes_with_enough_data(self):
        cfg = _cfg(min_days_data=3, min_episodes=10)
        sensors = make_sensors(days=4)
        episodes = make_episodes([NOW - i * 3600 for i in range(12)])
        gate = BaselineGate(cfg)
        ready, reason = gate.check(sensors, episodes)
        assert ready, reason

    def test_fails_too_few_days(self):
        cfg = _cfg(min_days_data=3, min_episodes=5)
        sensors = make_sensors(days=1)
        episodes = make_episodes([NOW - i * 3600 for i in range(6)])
        gate = BaselineGate(cfg)
        ready, _ = gate.check(sensors, episodes)
        assert not ready

    def test_fails_too_few_episodes(self):
        cfg = _cfg(min_days_data=3, min_episodes=10)
        sensors = make_sensors(days=4)
        episodes = make_episodes([NOW - 3600, NOW - 7200])  # only 2
        gate = BaselineGate(cfg)
        ready, _ = gate.check(sensors, episodes)
        assert not ready

    def test_fails_empty_sensors(self):
        cfg = _cfg(min_days_data=3, min_episodes=5)
        gate = BaselineGate(cfg)
        ready, _ = gate.check(pd.DataFrame(), pd.DataFrame())
        assert not ready


# ─────────────────────────────────────────────────────────────────────────────
# build_feature_vector
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildFeatureVector:
    def _window(self, gas, humidity, temperature, n=20):
        now = NOW
        rows = [_sensor_row(now - (n - i) * INTERVAL, gas, humidity, temperature)
                for i in range(n)]
        return pd.DataFrame(rows)

    def test_returns_all_feature_cols(self):
        window = self._window(30000, 50, 22)
        feats = build_feature_vector(window, at_epoch=NOW)
        assert feats is not None
        for col in FEATURE_COLS:
            assert col in feats, f"missing feature: {col}"

    def test_returns_none_on_empty(self):
        assert build_feature_vector(pd.DataFrame(), at_epoch=NOW) is None

    def test_returns_none_on_missing_columns(self):
        df = pd.DataFrame([{"timestamp_epoch": NOW, "gas": 30000}])
        assert build_feature_vector(df, at_epoch=NOW) is None

    def test_gas_stats_poor_air(self):
        window = self._window(gas=8000, humidity=65, temperature=24)
        feats = build_feature_vector(window, at_epoch=NOW)
        assert feats["gas_mean"] == pytest.approx(8000, rel=0.01)
        assert feats["gas_max"]  == pytest.approx(8000, rel=0.01)

    def test_temp_deviation_above_comfort(self):
        window = self._window(gas=30000, humidity=50, temperature=28)
        feats = build_feature_vector(window, at_epoch=NOW)
        assert feats["temp_deviation"] == pytest.approx(7.0, rel=0.05)  # 28 - 21


# ─────────────────────────────────────────────────────────────────────────────
# threshold_breaches
# ─────────────────────────────────────────────────────────────────────────────

class TestThresholdBreaches:
    def test_no_breaches_clean_air(self):
        feats = {"gas_mean": 35000, "gas_max": 40000, "humidity_min": 50}
        assert threshold_breaches(feats) == []

    def test_very_poor_gas(self):
        feats = {"gas_mean": 4000, "gas_max": 4500, "humidity_min": 50}
        breaches = threshold_breaches(feats)
        assert "gas_very_poor" in breaches

    def test_poor_gas(self):
        feats = {"gas_mean": 8000, "gas_max": 9000, "humidity_min": 50}
        breaches = threshold_breaches(feats)
        assert "gas_poor" in breaches

    def test_very_dry(self):
        feats = {"gas_mean": 35000, "gas_max": 40000, "humidity_min": 25}
        assert "humidity_very_dry" in threshold_breaches(feats)

    def test_dry(self):
        feats = {"gas_mean": 35000, "gas_max": 40000, "humidity_min": 35}
        assert "humidity_dry" in threshold_breaches(feats)


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline: train → predict
# ─────────────────────────────────────────────────────────────────────────────

class TestPipeline:
    """
    Bad-air windows are planted at regular intervals over 4 days.
    Episodes are planted 6 min after each bad window starts.
    The current sensor window (last 10 min) also shows bad air.
    The model should learn the pattern and fire the alert.
    """

    def _build(self):
        cfg = _cfg(
            min_days_data=3,
            min_episodes=5,
            feature_window_s=600,
            label_horizon_s=600,
            alert_threshold=0.50,
        )

        # Bad windows at random offsets so time-of-day is NOT the signal —
        # only gas resistance becomes the consistent predictor.
        # Include one bad window ending just before NOW so the model has
        # seen bad air at the current time of day during training.
        rng = np.random.default_rng(99)
        num_events = 40
        offsets = sorted(rng.uniform(3 * 3600, 4 * DAY, num_events))
        offsets.append(25 * 60)   # bad window 25 min ago → still in the current hour
        bad_windows = [(float(o), 20 * 60) for o in offsets]

        sensors = make_sensors(days=4, bad_windows=bad_windows)

        # Episodes 18 min into each bad window (so the 10-min horizon covers
        # only bad-air readings → clean signal for the model)
        episode_times = [NOW - o + 18 * 60 for o in offsets]
        episodes = make_episodes(episode_times, label="cough")

        return cfg, sensors, episodes

    def test_gate_ready(self):
        cfg, sensors, episodes = self._build()
        gate = BaselineGate(cfg)
        ready, reason = gate.check(sensors, episodes)
        assert ready, reason

    def test_trains_without_error(self):
        cfg, sensors, episodes = self._build()
        builder = DatasetBuilder(cfg)
        X, y = builder.build(sensors, episodes)
        assert not X.empty
        assert y.sum() > 0, "no positive labels in training set"
        trainer = Trainer(cfg)
        model, meta = trainer.train(X, y)
        assert model is not None
        assert meta["cv_roc_auc_mean"] > 0.5, (
            f"model barely better than chance: AUC={meta['cv_roc_auc_mean']:.3f}"
        )

    def test_alert_fires_on_bad_air(self):
        """
        Train on days 1-3, test on day 4.
        Bad-air samples in the held-out day should score higher on average
        than clean-air samples — model must have learned the gas signal.
        """
        cfg, sensors, episodes = self._build()
        builder = DatasetBuilder(cfg)

        split = NOW - DAY
        train_sensors  = sensors[sensors["timestamp_epoch"] <  split]
        test_sensors   = sensors[sensors["timestamp_epoch"] >= split]
        train_episodes = episodes[episodes["episode_start_epoch"] < split]

        X_train, y_train = builder.build(train_sensors, train_episodes)
        X_test,  y_test  = builder.build(test_sensors, episodes)

        assert y_test.sum() > 0, "no positive labels in held-out day"

        trainer = Trainer(cfg)
        model, _ = trainer.train(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        mean_pos = probs[y_test == 1].mean()
        mean_neg = probs[y_test == 0].mean()
        assert mean_pos > mean_neg, (
            f"bad-air avg p={mean_pos:.3f} not higher than clean-air avg p={mean_neg:.3f}"
        )

    def test_no_alert_on_clean_air(self):
        """
        The majority of clean-air samples in the held-out day should score
        below the alert threshold.
        """
        cfg, sensors, episodes = self._build()
        builder = DatasetBuilder(cfg)

        split = NOW - DAY
        train_sensors  = sensors[sensors["timestamp_epoch"] <  split]
        test_sensors   = sensors[sensors["timestamp_epoch"] >= split]
        train_episodes = episodes[episodes["episode_start_epoch"] < split]

        X_train, y_train = builder.build(train_sensors, train_episodes)
        X_test,  y_test  = builder.build(test_sensors, episodes)

        trainer = Trainer(cfg)
        model, _ = trainer.train(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        frac_below = (probs[y_test == 0] < cfg.alert_threshold).mean()
        assert frac_below >= 0.75, (
            f"only {frac_below:.0%} of clean-air samples scored below threshold"
        )
