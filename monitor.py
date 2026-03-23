"""
Indoor Asthma Trigger — Audio Monitor Daemon
============================================
Continuously listens via microphone, classifies cough / sneeze / wheeze events
using YAMNet (Google, TensorFlow Hub), and writes EPISODES to Firebase
Realtime Database.

An episode is a single continuous bout of the same sound (e.g. a coughing fit).
Individual detections that arrive within `episode_gap_s` of each other are merged
into one episode. When silence exceeds the gap, the episode is closed and written.

Firebase record per episode:
  {
    "label":                "cough",
    "episode_start_utc":    "2025-09-01T14:01:00.123+00:00",
    "episode_end_utc":      "2025-09-01T14:01:08.456+00:00",
    "episode_start_epoch":  1725198060.123,
    "episode_end_epoch":    1725198068.456,
    "duration_s":           8.333,
    "detection_count":      6,
    "peak_confidence":      0.84,
    "source":               "yamnet"
  }

Usage:
    python monitor.py                        # default mic + default config
    python monitor.py --config config.json   # custom thresholds / paths
    python monitor.py --list-devices         # show available microphones
"""

import argparse
import csv
import json
import logging
import queue
import signal
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from collections import deque

import numpy as np
import sounddevice as sd

import firebase_admin
from firebase_admin import credentials, db as firebase_db

import tensorflow as tf
import tensorflow_hub as hub

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Only these labels open/extend Firebase episodes.
CANONICAL_LABELS: frozenset[str] = frozenset({"cough", "sneeze", "wheeze"})

# Maps YAMNet alias labels → canonical form for voting purposes.
_ALIAS_MAP: dict[str, str] = {
    "coughing":           "cough",
    "sneezing":           "sneeze",
    "retching, vomiting": "cough",
    "vomiting":           "cough",
    "whimper":            "cough",  # deep/wet coughs consistently score as whimper in YAMNet
    # Sneeze's sharp "achoo" transient is consistently misclassified as duck/quack by YAMNet
    "duck":               "sneeze",
    "quack":              "sneeze",
}


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("asthma_monitor")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # Firebase
    firebase_credentials_path: str   = "serviceAccountKey.json"
    firebase_database_url: str       = "https://<YOUR-PROJECT-ID>.firebaseio.com"
    firebase_collection: str         = "episodes"

    # Microphone
    sample_rate: int                 = 16_000   # YAMNet expects 16 kHz
    channels: int                    = 1
    device_index: Optional[int]      = None     # None = system default

    # VAD — energy gate before classifier
    vad_window_ms: int               = 50       # audio chunk size in ms
    vad_energy_threshold: float      = 0.005    # RMS level to trigger recording
    vad_hold_frames: int             = 4        # extra frames captured after energy drops

    # Classifier
    clip_duration_s: float           = 0.5      # length of clip sent to classifier (sneezes are ~0.1-0.3s bursts)
    confidence_threshold: float      = 0.10     # minimum YAMNet score to count

    # Target sound classes (YAMNet display names, matched case-insensitively).
    # Canonical labels (cough/sneeze/wheeze) trigger Firebase episodes.
    # Adjacent labels co-fire with respiratory events and contribute votes toward
    # the canonical label via VotingBuffer, but do not open episodes on their own.
    target_classes: list = field(default_factory=lambda: [
        "cough", "coughing",
        "sneeze", "sneezing",
        "wheeze",
        "throat clearing",
        "breathing",
        "gasp",
        "whimper",
        "grunt",
        # Vomiting — co-occurs with severe coughing fits
        "retching, vomiting",
        "vomiting",
        # YAMNet maps sneeze "achoo" transients to duck/quack — alias catches these
        "duck",
        "quack",
    ])

    # Episode grouping:
    # Detections of the same label ≤ episode_gap_s apart → same episode.
    # A longer silence closes the episode.
    episode_gap_s: float             = 4.0

    # Minimum quality gates — episodes that don't meet these are discarded,
    # not written to Firebase. Prevents spurious single-hit 0-second entries.
    min_episode_duration_s: float    = 1.5      # must span at least this long
    min_detection_count: int         = 2        # must have at least this many hits

    # Voting — require multiple hits within a rolling window before committing
    # a detection to the episode tracker. Reduces false positives from mic noise.
    vote_window_s: float             = 4.0      # rolling window for vote accumulation
    min_votes: int                   = 1        # hits needed within window to confirm

    # Dual-window confirmation: after a detection fires, accumulate a longer clip
    # (confirm_duration_s) from that point and re-classify it for higher accuracy.
    confirm_duration_s: float        = 3.0      # length of confirmation clip

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path) as f:
            data = json.load(f)
        obj = cls()
        for k, v in data.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Episode dataclass  (what gets written to Firebase)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Episode:
    label: str
    episode_start_utc: str
    episode_end_utc: str
    episode_start_epoch: float
    episode_end_epoch: float
    duration_s: float
    detection_count: int
    peak_confidence: float
    source: str = "yamnet"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def build(
        cls,
        label: str,
        start_epoch: float,
        end_epoch: float,
        detection_count: int,
        peak_confidence: float,
        source: str = "yamnet",
    ) -> "Episode":
        def _iso(epoch: float) -> str:
            return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()

        return cls(
            label=label,
            episode_start_utc=_iso(start_epoch),
            episode_end_utc=_iso(end_epoch),
            episode_start_epoch=round(start_epoch, 3),
            episode_end_epoch=round(end_epoch, 3),
            duration_s=round(end_epoch - start_epoch, 3),
            detection_count=detection_count,
            peak_confidence=round(peak_confidence, 4),
            source=source,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Episode tracker
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class _OpenEpisode:
    """Mutable accumulator for an in-progress episode."""
    label: str
    start_epoch: float
    last_epoch: float
    detection_count: int  = 1
    peak_confidence: float = 0.0


class EpisodeTracker:
    """
    Receives individual detections; emits closed Episode objects.

    Thread-safe. Call feed() from the inference thread; call expire_stale()
    from a periodic flush thread; call flush_all() on shutdown.
    """

    def __init__(self, gap_s: float):
        self._gap_s = gap_s
        self._open: dict[str, _OpenEpisode] = {}
        self._lock = threading.Lock()

    def feed(
        self,
        label: str,
        confidence: float,
        now: float,
        source: str = "yamnet",
    ) -> Optional[Episode]:
        """
        Returns a just-closed Episode when the previous episode for this label
        ended (gap exceeded), otherwise None.
        """
        closed: Optional[Episode] = None

        with self._lock:
            existing = self._open.get(label)

            if existing is None:
                # Start brand-new episode
                log.info("Episode STARTED  label=%-10s  conf=%.2f", label, confidence)
                self._open[label] = _OpenEpisode(
                    label=label,
                    start_epoch=now,
                    last_epoch=now,
                    detection_count=1,
                    peak_confidence=confidence,
                )
            elif now - existing.last_epoch <= self._gap_s:
                # Still within the same episode — extend it
                existing.last_epoch = now
                existing.detection_count += 1
                existing.peak_confidence = max(existing.peak_confidence, confidence)
                log.debug("Episode EXTENDED label=%-10s  conf=%.2f  hits=%d  gap=%.1fs",
                          label, confidence, existing.detection_count,
                          now - existing.start_epoch)
            else:
                # Gap exceeded → close old, open new
                log.info("Episode GAP      label=%-10s  gap=%.1fs > %.1fs — closing",
                         label, now - existing.last_epoch, self._gap_s)
                closed = Episode.build(
                    label=existing.label,
                    start_epoch=existing.start_epoch,
                    end_epoch=existing.last_epoch,
                    detection_count=existing.detection_count,
                    peak_confidence=existing.peak_confidence,
                    source=source,
                )
                log.info("Episode STARTED  label=%-10s  conf=%.2f", label, confidence)
                self._open[label] = _OpenEpisode(
                    label=label,
                    start_epoch=now,
                    last_epoch=now,
                    detection_count=1,
                    peak_confidence=confidence,
                )

        return closed

    def expire_stale(self, now: float, source: str = "yamnet") -> list[Episode]:
        """
        Close and return open episodes whose last detection is older than gap_s.
        Call this every ~1 s from a background thread.
        """
        closed: list[Episode] = []
        with self._lock:
            stale_labels = [
                label for label, ep in self._open.items()
                if now - ep.last_epoch > self._gap_s
            ]
            for label in stale_labels:
                ep = self._open.pop(label)
                log.info("Episode STALE    label=%-10s  silent for %.1fs — closing",
                         label, now - ep.last_epoch)
                closed.append(Episode.build(
                    label=ep.label,
                    start_epoch=ep.start_epoch,
                    end_epoch=ep.last_epoch,
                    detection_count=ep.detection_count,
                    peak_confidence=ep.peak_confidence,
                    source=source,
                ))
        return closed

    def flush_all(self, now: float, source: str = "yamnet") -> list[Episode]:
        """Close every open episode — call on shutdown."""
        with self._lock:
            closed = [
                Episode.build(
                    label=ep.label,
                    start_epoch=ep.start_epoch,
                    end_epoch=ep.last_epoch,
                    detection_count=ep.detection_count,
                    peak_confidence=ep.peak_confidence,
                    source=source,
                )
                for ep in self._open.values()
            ]
            self._open.clear()
        return closed


# ─────────────────────────────────────────────────────────────────────────────
# Voting buffer — requires N hits within a rolling window before confirming
# ─────────────────────────────────────────────────────────────────────────────
class VotingBuffer:
    """
    Rolling time-windowed vote accumulator. Thread-safe.

    Call add(label, timestamp) on every detection hit.
    Returns True when >= min_votes hits have accumulated within window_s.
    Call reset(label) after a canonical detection fires to avoid double-firing.
    """

    def __init__(self, window_s: float, min_votes: int):
        self._window_s  = window_s
        self._min_votes = min_votes
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def add(self, label: str, timestamp: float) -> bool:
        """Add a hit; returns True if >= min_votes within window."""
        with self._lock:
            hits = self._hits.get(label, [])
            hits = [t for t in hits if timestamp - t <= self._window_s]
            hits.append(timestamp)
            self._hits[label] = hits
            return len(hits) >= self._min_votes

    def count(self, label: str, now: float) -> int:
        """Current vote count within window (for logging)."""
        with self._lock:
            return sum(1 for t in self._hits.get(label, []) if now - t <= self._window_s)

    def reset(self, label: str) -> None:
        """Clear votes for label after a canonical detection fires."""
        with self._lock:
            self._hits.pop(label, None)


# ─────────────────────────────────────────────────────────────────────────────
# Classifier interface  ← swap this for a Weill Cornell REST API call
# ─────────────────────────────────────────────────────────────────────────────
class AudioClassifier(ABC):
    @abstractmethod
    def classify(self, audio: np.ndarray, threshold: float) -> list[tuple[str, float]]:
        ...


class YAMNetClassifier(AudioClassifier):
    MODEL_URL = "https://tfhub.dev/google/yamnet/1"

    def __init__(self):
        log.info("Loading YAMNet from TensorFlow Hub …")
        self._model = hub.load(self.MODEL_URL)
        self._class_names = self._load_class_names()
        log.info("YAMNet ready — %d classes", len(self._class_names))

    def _load_class_names(self) -> list[str]:
        path = self._model.class_map_path().numpy().decode()
        names = []
        with open(path) as f:
            for row in csv.DictReader(f):
                names.append(row["display_name"])
        return names

    def classify(self, audio: np.ndarray, threshold: float) -> list[tuple[str, float]]:
        waveform = audio.astype(np.float32)
        if waveform.max() > 1.0 or waveform.min() < -1.0:
            waveform /= 32768.0
        scores, _, _ = self._model(waveform)
        mean_scores = tf.reduce_mean(scores, axis=0).numpy()
        results = [
            (self._class_names[i], float(s))
            for i, s in enumerate(mean_scores)
            if s >= threshold
        ]
        return sorted(results, key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Firebase writer
# ─────────────────────────────────────────────────────────────────────────────
class FirebaseWriter:
    def __init__(self, creds_path: str, database_url: str, collection: str):
        cred = credentials.Certificate(creds_path)
        firebase_admin.initialize_app(cred, {"databaseURL": database_url})
        self._ref = firebase_db.reference(collection)
        log.info("Firebase connected → /%s", collection)

    def write_episode(self, episode: Episode) -> str:
        return self._ref.push(episode.to_dict()).key


# ─────────────────────────────────────────────────────────────────────────────
# VAD — simple energy gate
# ─────────────────────────────────────────────────────────────────────────────
class VAD:
    def __init__(self, cfg: Config):
        self._threshold    = cfg.vad_energy_threshold
        self._clip_samples = int(cfg.sample_rate * cfg.clip_duration_s)
        self._hold         = cfg.vad_hold_frames
        self._buffer: list[np.ndarray] = []
        self._hold_counter = 0
        self._triggered    = False

    def feed(self, frame: np.ndarray) -> Optional[np.ndarray]:
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if rms > self._threshold:
            if not self._triggered:
                log.debug("[VAD] triggered  rms=%.4f > threshold=%.4f", rms, self._threshold)
            self._triggered    = True
            self._hold_counter = self._hold

        if self._triggered:
            self._buffer.append(frame)
            if self._hold_counter > 0:
                self._hold_counter -= 1
            else:
                clip = np.concatenate(self._buffer)
                self._buffer   = []
                self._triggered = False
                if len(clip) < self._clip_samples:
                    clip = np.pad(clip, (0, self._clip_samples - len(clip)))
                else:
                    clip = clip[: self._clip_samples]
                log.debug("[VAD] clip ready  samples=%d → sending to classifier", len(clip))
                return clip
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main monitor daemon
# ─────────────────────────────────────────────────────────────────────────────
class AsthmaAudioMonitor:
    def __init__(self, cfg: Config, classifier: AudioClassifier, writer: FirebaseWriter,
                 debug_audio: bool = False):
        self._cfg         = cfg
        self._classifier  = classifier
        self._writer      = writer
        self._debug_audio = debug_audio
        self._vad         = VAD(cfg)
        self._tracker     = EpisodeTracker(gap_s=cfg.episode_gap_s)
        self._voting      = VotingBuffer(window_s=cfg.vote_window_s, min_votes=cfg.min_votes)
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._stop_event  = threading.Event()
        self._frame_size  = int(cfg.sample_rate * cfg.vad_window_ms / 1000)
        # Dual-window: accumulates VAD clips after a detection fires
        self._confirm_clips: dict[str, list[np.ndarray]] = {}
        self._confirm_samples = int(cfg.sample_rate * cfg.confirm_duration_s)
        # Debug: rolling buffer of last 2 VAD clips for multi-window comparison
        self._debug_deque: deque = deque(maxlen=2)

    def _commit(self, episode: Episode):
        too_short = episode.duration_s < self._cfg.min_episode_duration_s
        too_few   = episode.detection_count < self._cfg.min_detection_count

        if too_short and too_few:
            log.info(
                "⏭  SKIPPED  %-10s  %.1fs  %d hits  (min=%.1fs/%d hits)",
                episode.label.upper(),
                episode.duration_s,
                episode.detection_count,
                self._cfg.min_episode_duration_s,
                self._cfg.min_detection_count,
            )
            return

        log.info(
            "📋 EPISODE  %-10s  %s → %s  (%.1fs, %d hits, peak=%.2f)",
            episode.label.upper(),
            episode.episode_start_utc[11:19],
            episode.episode_end_utc[11:19],
            episode.duration_s,
            episode.detection_count,
            episode.peak_confidence,
        )
        try:
            key = self._writer.write_episode(episode)
            log.info("   ✓ Firebase key=%s", key)
        except Exception as exc:
            log.error("   ✗ Firebase write failed: %s", exc)

    def _mic_callback(self, indata, frames, _time, status):
        if status:
            log.warning("Mic status: %s", status)
        try:
            self._audio_q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            log.warning("Audio queue full — frame dropped")

    def _inference_loop(self):
        log.info("Inference thread started")
        target_lower = {c.lower() for c in self._cfg.target_classes}
        _heartbeat_interval = 5.0
        _last_heartbeat = time.time()
        _frame_count = 0
        _max_rms = 0.0

        while not self._stop_event.is_set():
            try:
                frame = self._audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            _frame_count += 1
            rms = float(np.sqrt(np.mean(frame ** 2)))
            _max_rms = max(_max_rms, rms)
            now = time.time()
            if now - _last_heartbeat >= _heartbeat_interval:
                bar_len = 20
                filled = int(min(_max_rms / self._cfg.vad_energy_threshold, 1.0) * bar_len)
                bar = "█" * filled + "░" * (bar_len - filled)
                status = "LOUD ENOUGH" if _max_rms >= self._cfg.vad_energy_threshold else "too quiet "
                log.debug(
                    "[Heartbeat] mic=[%s] peak=%.4f  threshold=%.4f  %-11s  frames=%d  queue=%d",
                    bar, _max_rms, self._cfg.vad_energy_threshold, status,
                    _frame_count, self._audio_q.qsize(),
                )
                _frame_count = 0
                _max_rms = 0.0
                _last_heartbeat = now

            clip = self._vad.feed(frame)
            if clip is None:
                continue

            # Feed any active confirmation buffers with every new clip,
            # regardless of whether this clip matches a target label.
            for canon in list(self._confirm_clips):
                self._confirm_clips[canon].append(clip)
                accumulated = sum(len(c) for c in self._confirm_clips[canon])
                if accumulated >= self._confirm_samples:
                    long_clip = np.concatenate(self._confirm_clips.pop(canon))[:self._confirm_samples]
                    long_results = self._classifier.classify(long_clip, self._cfg.confidence_threshold)
                    # Check if any result maps to this canonical label
                    confirmed = [(lbl, sc) for lbl, sc in long_results
                                 if _ALIAS_MAP.get(lbl.lower(), lbl.lower()) == canon]
                    if confirmed:
                        best_lbl, best_sc = max(confirmed, key=lambda x: x[1])
                        log.info("[Confirm] %-8s  ✓ long-clip confirms  conf=%.2f  (%.1fs window)  via %s",
                                 canon, best_sc, self._cfg.confirm_duration_s, best_lbl)
                    else:
                        top = ", ".join(f"{lbl}({sc:.2f})" for lbl, sc in long_results[:3])
                        log.info("[Confirm] %-8s  ✗ long-clip did NOT confirm  top=%s",
                                 canon, top or "none")

            if self._debug_audio:
                self._debug_deque.append(clip)
                ts = datetime.now().strftime("%H:%M:%S")
                half = self._cfg.sample_rate // 2  # 0.5 s

                short_res = self._classifier.classify(clip[:half], 0.0)
                short_top = " ".join(f"{lbl}={sc:.2f}" for lbl, sc in short_res[:5])

                all_results = self._classifier.classify(clip, 0.0)
                top5 = " ".join(f"{lbl}={sc:.2f}" for lbl, sc in all_results[:5])

                if len(self._debug_deque) == 2:
                    long_clip = np.concatenate(list(self._debug_deque))
                    long_res  = self._classifier.classify(long_clip, 0.0)
                    long_top  = " ".join(f"{lbl}={sc:.2f}" for lbl, sc in long_res[:5])
                else:
                    long_top = "(buffering…)"

                log.debug("DEBUG [%s]  0.5s: %s", ts, short_top)
                log.debug("            1.0s: %s", top5)
                log.debug("            2.0s: %s", long_top)

                results = [(lbl, sc) for lbl, sc in all_results
                           if sc >= self._cfg.confidence_threshold]
            else:
                results = self._classifier.classify(clip, self._cfg.confidence_threshold)

            if not results:
                log.debug("[Classifier] no results above conf=%.2f", self._cfg.confidence_threshold)
            else:
                top_labels = ", ".join(f"{lbl}({sc:.2f})" for lbl, sc in results[:5])
                matched = [lbl for lbl, _ in results if lbl.lower() in target_lower]
                if matched:
                    log.debug("[Classifier] %d result(s) — TOP: %s  ← MATCH: %s",
                              len(results), top_labels, ", ".join(matched))
                else:
                    log.debug("[Classifier] %d result(s) — TOP: %s  (no target match)",
                              len(results), top_labels)

            for label, confidence in results:
                label_lower = label.lower()
                if label_lower not in target_lower:
                    continue

                # Resolve to canonical label (coughing→cough, sneezing→sneeze, etc.)
                canonical = _ALIAS_MAP.get(label_lower, label_lower)

                log.info("[Detection] %-16s  conf=%.2f  canonical=%s",
                         label.upper(), confidence, canonical)

                # Adjacent labels (breathing, gasp, etc.) that don't map to a canonical
                # are noted but do not feed the episode tracker.
                if canonical not in CANONICAL_LABELS:
                    log.debug("[Voting] adjacent label %-16s — noted, not canonical", label_lower)
                    continue

                vote_hit = self._voting.add(canonical, now)
                votes = self._voting.count(canonical, now)
                if not vote_hit:
                    log.debug("[Voting] %-8s  votes=%d/%d  (waiting for threshold)",
                              canonical, votes, self._cfg.min_votes)
                    continue

                log.info("[Voting] %-8s  votes=%d/%d  ✓ threshold met — committing",
                         canonical, votes, self._cfg.min_votes)
                self._voting.reset(canonical)

                # Start confirmation buffer if not already running for this label.
                # Seeds it with the 1s clip that just triggered detection.
                if canonical not in self._confirm_clips:
                    self._confirm_clips[canonical] = [clip]
                    log.debug("[Confirm] %-8s  started long-clip buffer (target=%.1fs)",
                              canonical, self._cfg.confirm_duration_s)

                closed = self._tracker.feed(
                    label=canonical,
                    confidence=confidence,
                    now=now,
                )
                if closed:
                    self._commit(closed)

        log.info("Inference thread stopped")

    def _flush_loop(self):
        """Closes stale open episodes every second."""
        while not self._stop_event.is_set():
            time.sleep(1.0)
            for ep in self._tracker.expire_stale(now=time.time()):
                self._commit(ep)

    def run(self):
        def _shutdown(sig, _f):
            log.info("Signal %s — shutting down …", sig)
            self._stop_event.set()

        signal.signal(signal.SIGINT,  _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        for name, target in [("Inference", self._inference_loop),
                              ("Flusher",   self._flush_loop)]:
            threading.Thread(target=target, name=name, daemon=True).start()

        log.info(
            "🎤 Microphone  sr=%d Hz  device=%s",
            self._cfg.sample_rate, self._cfg.device_index or "default",
        )
        log.info(
            "Watching: %s  |  conf ≥ %.0f%%  |  episode gap = %.1fs",
            ", ".join(self._cfg.target_classes),
            self._cfg.confidence_threshold * 100,
            self._cfg.episode_gap_s,
        )

        with sd.InputStream(
            samplerate=self._cfg.sample_rate,
            channels=self._cfg.channels,
            dtype="float32",
            blocksize=self._frame_size,
            device=self._cfg.device_index,
            callback=self._mic_callback,
        ):
            log.info("Daemon running — Ctrl-C to stop")
            while not self._stop_event.is_set():
                time.sleep(0.1)

        # Final flush
        log.info("Flushing open episodes on exit …")
        for ep in self._tracker.flush_all(now=time.time()):
            self._commit(ep)

        log.info("Stopped cleanly ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Indoor Asthma Trigger — Audio Monitor")
    p.add_argument("--config",       default=None)
    p.add_argument("--classifier",   choices=["yamnet"], default="yamnet")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--debug-audio",  action="store_true",
                   help="Print top-5 YAMNet scores on every clip for diagnosis")
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    cfg = Config.from_file(args.config) if args.config else Config()

    classifier = YAMNetClassifier()

    writer = FirebaseWriter(
        creds_path=cfg.firebase_credentials_path,
        database_url=cfg.firebase_database_url,
        collection=cfg.firebase_collection,
    )

    AsthmaAudioMonitor(cfg, classifier, writer, debug_audio=args.debug_audio).run()


if __name__ == "__main__":
    main()
