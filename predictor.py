"""
Indoor Asthma Trigger — ML Predictor Server
============================================
Standalone server script. Reads from Firebase, builds a labeled dataset,
trains a RandomForest classifier, and every 5 minutes predicts whether an
asthma episode is likely in the next 10 minutes based on live sensor data.

Phases
──────
  BASELINE (days 1–N):
      Collect sensor readings + episodes. No predictions emitted.
      Gate: need MIN_DAYS_DATA days of sensor history AND MIN_EPISODES episodes.

  TRAINING (nightly, or on first crossing the gate):
      Pull all history, build feature windows, label them, fit RandomForest.
      Save model to disk so restarts don't lose it.

  PREDICTION (every PREDICT_INTERVAL_S seconds):
      Pull latest FEATURE_WINDOW_S of sensor readings.
      Build one feature vector. Run inference.
      If probability ≥ ALERT_THRESHOLD → write to Firebase /predictions.

Firebase schema expected
────────────────────────
  /sensor_readings/<id>
      timestamp_epoch   float   Unix epoch (use ServerValue.TIMESTAMP on ESP32)
      voc_ppb           float
      iaq               float
      temp_c            float
      humidity_pct      float
      voc_change_rate   float   ppb/min

  /episodes/<id>        (written by monitor.py)
      episode_start_epoch float
      label               str    "cough" | "sneeze" | "wheeze"

  /predictions/<id>     (written by this script)
      predicted_at_epoch  float
      predicted_at_utc    str
      label               str
      probability         float
      features            dict
      threshold_breaches  list[str]
      model_version       str
      training_episodes   int

Deployment
──────────
  pip install -r requirements.txt
  export FIREBASE_CREDS=/path/to/serviceAccountKey.json
  export FIREBASE_URL=https://<project>.firebaseio.com
  python predictor.py

  Or with a config file:
  python predictor.py --config predictor_config.json

  Railway / Render: set env vars in the dashboard, Procfile = "worker: python predictor.py"
  Systemd:          see predictor.service in this directory
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import joblib

import firebase_admin
from firebase_admin import credentials, db as firebase_db

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("asthma_predictor")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # Firebase
    firebase_credentials_path: str = ""   # env: FIREBASE_CREDS
    firebase_database_url: str     = ""   # env: FIREBASE_URL
    sensor_collection: str         = "sensor_readings"
    episode_collection: str        = "episodes"
    prediction_collection: str     = "predictions"

    # Baseline gate
    min_days_data: int      = 3       # days of sensor history required
    min_episodes: int       = 10      # minimum labeled episodes required

    # Feature engineering
    feature_window_s: int   = 600     # look-back window (10 min of sensor readings)
    label_horizon_s: int    = 600     # predict episodes within next 10 min

    # Prediction loop
    predict_interval_s: int = 300     # run inference every 5 min
    alert_threshold: float  = 0.65    # probability to fire a prediction

    # Training schedule
    retrain_interval_h: int = 24      # retrain every N hours
    model_path: str         = "model.joblib"
    model_version: str      = "v1"

    # Target episode labels to model (can restrict to just "cough")
    target_labels: list = field(default_factory=lambda: ["cough", "sneeze", "wheeze"])

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path) as f:
            data = json.load(f)
        obj = cls()
        for k, v in data.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj

    def resolve_env(self):
        """Override blank fields with environment variables."""
        if not self.firebase_credentials_path:
            self.firebase_credentials_path = os.environ.get("FIREBASE_CREDS", "serviceAccountKey.json")
        if not self.firebase_database_url:
            self.firebase_database_url = os.environ.get("FIREBASE_URL", "")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Firebase client
# ─────────────────────────────────────────────────────────────────────────────
class FirebaseClient:
    def __init__(self, cfg: Config):
        cred = credentials.Certificate(cfg.firebase_credentials_path)
        firebase_admin.initialize_app(cred, {"databaseURL": cfg.firebase_database_url})
        self._sensor_ref    = firebase_db.reference(cfg.sensor_collection)
        self._episode_ref   = firebase_db.reference(cfg.episode_collection)
        self._pred_ref      = firebase_db.reference(cfg.prediction_collection)
        log.info("Firebase connected")

    def get_sensor_readings(self, since_epoch: float) -> pd.DataFrame:
        """Pull all sensor readings after `since_epoch`. Returns sorted DataFrame."""
        data = self._sensor_ref.get() or {}
        rows = []
        for key, val in data.items():
            if isinstance(val, dict) and val.get("timestamp_epoch", 0) >= since_epoch:
                rows.append(val)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.sort_values("timestamp_epoch").reset_index(drop=True)
        return df

    def get_episodes(self, since_epoch: float = 0.0) -> pd.DataFrame:
        """Pull all episodes, optionally filtered by start time."""
        data = self._episode_ref.get() or {}
        rows = []
        for key, val in data.items():
            if isinstance(val, dict) and val.get("episode_start_epoch", 0) >= since_epoch:
                rows.append(val)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.sort_values("episode_start_epoch").reset_index(drop=True)
        return df

    def write_prediction(self, record: dict) -> str:
        return self._pred_ref.push(record).key


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "gas_mean", "gas_max", "gas_std",
    "humidity_mean", "humidity_min",
    "temp_mean", "temp_deviation",      # deviation from 21°C comfort baseline
    "reading_density",                  # readings per minute in the window
    "hour_sin", "hour_cos",             # circadian encoding
    "day_of_week_sin", "day_of_week_cos",
]

COMFORT_TEMP = 21.0  # °C baseline for deviation feature


def _circadian_features(epoch: float) -> dict:
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    h  = dt.hour + dt.minute / 60.0
    d  = dt.weekday()
    return {
        "hour_sin":         np.sin(2 * np.pi * h / 24),
        "hour_cos":         np.cos(2 * np.pi * h / 24),
        "day_of_week_sin":  np.sin(2 * np.pi * d / 7),
        "day_of_week_cos":  np.cos(2 * np.pi * d / 7),
    }


def build_feature_vector(window: pd.DataFrame, at_epoch: float) -> Optional[dict]:
    """
    Summarize a window of sensor readings into a single feature vector dict.
    Returns None if the window is empty or missing required columns.
    """
    required = {"gas", "temperature", "humidity"}
    if window.empty or not required.issubset(window.columns):
        return None

    duration_min = (window["timestamp_epoch"].max() - window["timestamp_epoch"].min()) / 60.0 + 1e-6

    feats = {
        "gas_mean":          window["gas"].mean(),
        "gas_max":           window["gas"].max(),
        "gas_std":           window["gas"].std(ddof=0),
        "humidity_mean":     window["humidity"].mean(),
        "humidity_min":      window["humidity"].min(),
        "temp_mean":         window["temperature"].mean(),
        "temp_deviation":    abs(window["temperature"].mean() - COMFORT_TEMP),
        "reading_density":   len(window) / duration_min,
    }
    feats.update(_circadian_features(at_epoch))
    return feats


def threshold_breaches(feats: dict) -> list[str]:
    """Return human-readable names for any threshold crossings in the feature vector."""
    hits = []
    # BME680: lower gas resistance = worse air quality
    if feats.get("gas_min", feats.get("gas_mean", 99999)) < 5000:   hits.append("gas_very_poor")
    elif feats.get("gas_min", feats.get("gas_mean", 99999)) < 10000: hits.append("gas_poor")
    if feats.get("humidity_min", 100) < 30:  hits.append("humidity_very_dry")
    elif feats.get("humidity_min", 100) < 40: hits.append("humidity_dry")
    return hits


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────
class DatasetBuilder:
    """
    Constructs a supervised dataset from sensor readings + episodes.

    Label = 1 if any target-label episode started within [T, T + label_horizon_s]
            for a sensor reading at time T.
    """

    def __init__(self, cfg: Config):
        self.window_s   = cfg.feature_window_s
        self.horizon_s  = cfg.label_horizon_s
        self.targets    = set(cfg.target_labels)

    def build(
        self,
        sensors: pd.DataFrame,
        episodes: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Returns (X, y) where X has columns matching FEATURE_COLS and y is binary.
        """
        if sensors.empty:
            return pd.DataFrame(columns=FEATURE_COLS), pd.Series(dtype=int)

        # Pre-filter episodes to target labels
        target_episodes = pd.DataFrame()
        if not episodes.empty and "label" in episodes.columns:
            target_episodes = episodes[episodes["label"].isin(self.targets)].copy()

        rows_X, rows_y = [], []
        ts_array = sensors["timestamp_epoch"].values

        for i, row in sensors.iterrows():
            T = row["timestamp_epoch"]

            # Build feature window: all readings in [T - window_s, T]
            mask_window = (ts_array >= T - self.window_s) & (ts_array <= T)
            window = sensors[mask_window]
            feats = build_feature_vector(window, at_epoch=T)
            if feats is None:
                continue

            # Label: any episode starting in [T, T + horizon_s]?
            if not target_episodes.empty:
                mask_label = (
                    (target_episodes["episode_start_epoch"] >= T) &
                    (target_episodes["episode_start_epoch"] <= T + self.horizon_s)
                )
                label = int(mask_label.any())
            else:
                label = 0

            rows_X.append(feats)
            rows_y.append(label)

        X = pd.DataFrame(rows_X, columns=FEATURE_COLS)
        y = pd.Series(rows_y, dtype=int)
        return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Baseline gate
# ─────────────────────────────────────────────────────────────────────────────
class BaselineGate:
    def __init__(self, cfg: Config):
        self.min_days    = cfg.min_days_data
        self.min_episodes = cfg.min_episodes

    def check(self, sensors: pd.DataFrame, episodes: pd.DataFrame) -> tuple[bool, str]:
        """
        Returns (ready, reason_string).
        """
        if sensors.empty:
            return False, "No sensor readings in Firebase yet"

        span_days = (
            sensors["timestamp_epoch"].max() - sensors["timestamp_epoch"].min()
        ) / 86400.0

        if span_days < self.min_days:
            return False, (
                f"Baseline collection in progress: "
                f"{span_days:.1f}/{self.min_days} days of sensor data"
            )

        n_ep = len(episodes) if not episodes.empty else 0
        if n_ep < self.min_episodes:
            return False, (
                f"Need more labeled episodes: {n_ep}/{self.min_episodes} episodes recorded"
            )

        return True, f"Baseline ready — {span_days:.1f} days, {n_ep} episodes"


# ─────────────────────────────────────────────────────────────────────────────
# Model store
# ─────────────────────────────────────────────────────────────────────────────
class ModelStore:
    def __init__(self, path: str):
        self.path = Path(path)

    def save(self, model, meta: dict):
        payload = {"model": model, "meta": meta}
        joblib.dump(payload, self.path)
        log.info("Model saved → %s", self.path)

    def load(self) -> tuple[Optional[object], dict]:
        if not self.path.exists():
            return None, {}
        payload = joblib.load(self.path)
        log.info("Model loaded ← %s  meta=%s", self.path, payload.get("meta", {}))
        return payload["model"], payload.get("meta", {})


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def train(self, X: pd.DataFrame, y: pd.Series) -> tuple[object, dict]:
        """
        Fit a RandomForest pipeline (StandardScaler + RF).
        Returns (fitted_pipeline, metrics_dict).
        """
        if X.empty or y.sum() == 0:
            raise ValueError(
                "No positive labels in training set — "
                "not enough episodes to learn from yet"
            )

        pos_rate = y.mean()
        log.info(
            "Training on %d samples  (%.1f%% positive)",
            len(y), pos_rate * 100,
        )

        # Class weight to handle imbalance (episodes are rare events)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ])

        # Cross-validate if we have enough data
        metrics = {}
        if len(y) >= 30:
            scores = cross_val_score(pipeline, X, y, cv=3, scoring="roc_auc")
            metrics["cv_roc_auc_mean"] = float(scores.mean())
            metrics["cv_roc_auc_std"]  = float(scores.std())
            log.info(
                "CV ROC-AUC: %.3f ± %.3f",
                scores.mean(), scores.std(),
            )

        pipeline.fit(X, y)

        # Feature importances
        importances = pipeline.named_steps["clf"].feature_importances_
        top = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)[:5]
        log.info("Top features: %s", [(n, f"{v:.3f}") for n, v in top])

        metrics.update({
            "n_samples":         len(y),
            "n_positive":        int(y.sum()),
            "positive_rate":     float(pos_rate),
            "trained_at_epoch":  time.time(),
            "feature_cols":      FEATURE_COLS,
            "top_features":      [(n, float(v)) for n, v in top],
            "model_version":     self.cfg.model_version,
        })

        return pipeline, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main prediction server loop
# ─────────────────────────────────────────────────────────────────────────────
class PredictionServer:
    def __init__(self, cfg: Config):
        self.cfg       = cfg
        self.firebase  = FirebaseClient(cfg)
        self.gate      = BaselineGate(cfg)
        self.builder   = DatasetBuilder(cfg)
        self.trainer   = Trainer(cfg)
        self.store     = ModelStore(cfg.model_path)
        self._stop     = threading.Event()
        self._model    = None
        self._model_meta: dict = {}
        self._last_trained: float = 0.0

        # Try loading an existing model from disk
        self._model, self._model_meta = self.store.load()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pull_all_history(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        cutoff = time.time() - 90 * 86400   # up to 90 days back
        sensors  = self.firebase.get_sensor_readings(since_epoch=cutoff)
        episodes = self.firebase.get_episodes(since_epoch=cutoff)
        return sensors, episodes

    def _maybe_retrain(self, sensors: pd.DataFrame, episodes: pd.DataFrame) -> bool:
        """Retrain if enough time has passed since last training. Returns True if trained."""
        hours_since = (time.time() - self._last_trained) / 3600
        if self._model is not None and hours_since < self.cfg.retrain_interval_h:
            return False

        log.info("Building training dataset …")
        X, y = self.builder.build(sensors, episodes)

        if X.empty:
            log.warning("Empty training set — skipping")
            return False

        try:
            model, meta = self.trainer.train(X, y)
        except ValueError as e:
            log.warning("Training skipped: %s", e)
            return False

        meta["n_training_episodes"] = len(episodes)
        self._model      = model
        self._model_meta = meta
        self._last_trained = time.time()
        self.store.save(model, meta)
        return True

    def _predict_now(self, sensors: pd.DataFrame) -> Optional[dict]:
        """
        Build a feature vector from the latest FEATURE_WINDOW_S of data
        and return a prediction record dict, or None if inference should be skipped.
        """
        if sensors.empty or self._model is None:
            return None

        now = time.time()
        window = sensors[sensors["timestamp_epoch"] >= now - self.cfg.feature_window_s]

        if len(window) < 2:
            log.debug("Insufficient sensor readings in window (%d)", len(window))
            return None

        feats = build_feature_vector(window, at_epoch=now)
        if feats is None:
            return None

        X_live = pd.DataFrame([feats], columns=FEATURE_COLS)
        prob = float(self._model.predict_proba(X_live)[0, 1])

        log.info(
            "Prediction: p=%.3f  (threshold=%.2f)  %s",
            prob,
            self.cfg.alert_threshold,
            "⚠ ALERT" if prob >= self.cfg.alert_threshold else "OK",
        )

        if prob < self.cfg.alert_threshold:
            return None

        breaches = threshold_breaches(feats)
        ts = datetime.now(timezone.utc)

        return {
            "predicted_at_epoch":   ts.timestamp(),
            "predicted_at_utc":     ts.isoformat(),
            "label":                "respiratory_episode",
            "probability":          round(prob, 4),
            "features":             {k: round(float(v), 3) for k, v in feats.items()},
            "threshold_breaches":   breaches,
            "model_version":        self._model_meta.get("model_version", "?"),
            "training_episodes":    self._model_meta.get("n_training_episodes", 0),
        }

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        import signal

        def _shutdown(sig, _f):
            log.info("Signal %s — stopping …", sig)
            self._stop.set()

        signal.signal(signal.SIGINT,  _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        log.info(
            "Prediction server started  "
            "(gate: %d days + %d episodes, alert threshold: %.0f%%)",
            self.cfg.min_days_data,
            self.cfg.min_episodes,
            self.cfg.alert_threshold * 100,
        )

        while not self._stop.is_set():
            cycle_start = time.monotonic()

            try:
                sensors, episodes = self._pull_all_history()

                # ── Phase 1: Check baseline gate ─────────────────────────────
                ready, reason = self.gate.check(sensors, episodes)
                if not ready:
                    log.info("📊 Baseline phase — %s", reason)
                    self._sleep_until_next(cycle_start)
                    continue

                log.info("✅ %s", reason)

                # ── Phase 2: Train (or retrain) ───────────────────────────────
                self._maybe_retrain(sensors, episodes)

                # ── Phase 3: Predict ─────────────────────────────────────────
                if self._model is None:
                    log.warning("No model available yet — skipping prediction")
                    self._sleep_until_next(cycle_start)
                    continue

                # Use only recent sensor data for live inference
                recent = sensors[
                    sensors["timestamp_epoch"] >= time.time() - self.cfg.feature_window_s * 2
                ]
                record = self._predict_now(recent)

                if record is not None:
                    key = self.firebase.write_prediction(record)
                    log.info(
                        "⚠  PREDICTION WRITTEN  key=%s  p=%.2f  breaches=%s",
                        key, record["probability"], record["threshold_breaches"],
                    )

            except Exception as exc:
                log.error("Cycle error (will retry): %s", exc, exc_info=True)

            self._sleep_until_next(cycle_start)

        log.info("Prediction server stopped ✓")

    def _sleep_until_next(self, cycle_start: float):
        elapsed = time.monotonic() - cycle_start
        sleep_s = max(0, self.cfg.predict_interval_s - elapsed)
        log.debug("Next cycle in %.0fs", sleep_s)
        self._stop.wait(timeout=sleep_s)


# ─────────────────────────────────────────────────────────────────────────────
# Mock sensor generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_mock_sensors(days: int = 4, interval_s: int = 30) -> pd.DataFrame:
    """
    Generate realistic BME680 sensor readings over the past `days` days.
    Readings are spaced `interval_s` seconds apart.
    No data is written to Firebase.
    """
    rng = np.random.default_rng(42)
    now = time.time()
    timestamps = np.arange(now - days * 86400, now, interval_s)
    n = len(timestamps)

    rows = []
    for ts in timestamps:
        rows.append({
            "timestamp_epoch": float(ts),
            "gas":             float(np.clip(rng.normal(30000, 8000), 5000, 60000)),
            "humidity":        float(np.clip(rng.normal(50, 8), 20, 90)),
            "temperature":     float(np.clip(rng.normal(22, 2), 15, 35)),
        })

    df = pd.DataFrame(rows).sort_values("timestamp_epoch").reset_index(drop=True)
    log.info("Mock sensors: %d readings over %d days", len(df), days)
    return df


def generate_mock_episodes(days: int = 4, count: int = 20) -> pd.DataFrame:
    """
    Generate fake episode records spread over the past `days` days.
    Merges with real Firebase episodes — call pd.concat() on the result.
    """
    rng = np.random.default_rng(7)
    now = time.time()
    labels = ["cough", "sneeze", "wheeze"]
    rows = []
    for _ in range(count):
        start = float(rng.uniform(now - days * 86400, now - 60))
        duration = float(rng.uniform(2.0, 12.0))
        label = labels[rng.integers(0, len(labels))]
        rows.append({
            "label":               label,
            "episode_start_epoch": round(start, 3),
            "episode_end_epoch":   round(start + duration, 3),
            "duration_s":          round(duration, 3),
            "detection_count":     int(rng.integers(2, 10)),
            "peak_confidence":     round(float(rng.uniform(0.3, 0.9)), 4),
            "source":              "mock",
        })
    df = pd.DataFrame(rows).sort_values("episode_start_epoch").reset_index(drop=True)
    log.info("Mock episodes: %d generated over %d days", len(df), days)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Indoor Asthma Trigger — ML Predictor")
    p.add_argument("--config",        default=None, help="Path to predictor_config.json")
    p.add_argument("--dry-run",       action="store_true",
                   help="Print prediction without writing to Firebase")
    p.add_argument("--mock-sensors",   action="store_true",
                   help="Use generated mock sensor data instead of Firebase sensor_readings")
    p.add_argument("--mock-episodes",  action="store_true",
                   help="Supplement real Firebase episodes with generated mock episodes")
    args = p.parse_args()

    cfg = Config.from_file(args.config) if args.config else Config()
    cfg.resolve_env()

    if not cfg.firebase_database_url:
        log.error(
            "Firebase URL not set. Use --config or set FIREBASE_URL env var."
        )
        sys.exit(1)

    server = PredictionServer(cfg)

    if args.dry_run:
        tags = " ".join(filter(None, [
            "mock sensors" if args.mock_sensors else "",
            "mock episodes" if args.mock_episodes else "",
        ]))
        log.info("--- DRY RUN%s ---", f" ({tags})" if tags else "")

        if args.mock_sensors:
            sensors = generate_mock_sensors(days=cfg.min_days_data + 1)
            episodes = server.firebase.get_episodes()
        else:
            sensors, episodes = server._pull_all_history()

        if args.mock_episodes:
            mock_ep = generate_mock_episodes(days=cfg.min_days_data + 1)
            episodes = pd.concat([episodes, mock_ep], ignore_index=True)

        ready, reason = server.gate.check(sensors, episodes)
        log.info("Gate: %s — %s", "READY" if ready else "NOT READY", reason)
        if ready:
            server._maybe_retrain(sensors, episodes)
            record = server._predict_now(sensors)
            print(json.dumps(record, indent=2) if record else "No alert triggered")
        return

    server.run()


if __name__ == "__main__":
    main()
