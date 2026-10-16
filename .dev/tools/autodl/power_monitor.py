"""CPU-only parsing and continuous low-power timing for the AutoDL monitor.

This module never connects to a host or shuts one down. The controller owns
process policy and decides whether an eligible observation may trigger shutdown.
"""

from __future__ import annotations

import csv
import io
import math
import re
from collections.abc import Mapping


class PowerSampleError(ValueError):
    """The GPU observation is incomplete or cannot be trusted."""


def _finite_number(value, label: str, *, minimum=0, maximum=None) -> float:
    if isinstance(value, bool):
        raise PowerSampleError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PowerSampleError(f"{label} must be a finite number") from exc
    if not math.isfinite(number) or number < minimum or (
        maximum is not None and number > maximum
    ):
        raise PowerSampleError(f"{label} is out of range")
    return number


def _validate_gpus(gpus) -> list[dict]:
    if not isinstance(gpus, list) or not gpus:
        raise PowerSampleError("GPU observation must contain at least one GPU")
    result = []
    indices = set()
    uuids = set()
    for gpu in gpus:
        if not isinstance(gpu, Mapping):
            raise PowerSampleError("GPU entry must be an object")
        required = {"index", "uuid", "name", "power_w", "pstate", "utilization_percent"}
        if not required <= gpu.keys():
            raise PowerSampleError("GPU entry is missing required fields")
        index = gpu["index"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise PowerSampleError("GPU index must be a nonnegative integer")
        uuid = gpu["uuid"]
        if not isinstance(uuid, str) or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", uuid):
            raise PowerSampleError("GPU UUID is invalid")
        name = gpu["name"]
        if not isinstance(name, str) or not name.strip() or name.strip().upper() in {
            "N/A", "[N/A]", "UNKNOWN", "NOT SUPPORTED", "[NOT SUPPORTED]",
        }:
            raise PowerSampleError("GPU name is unavailable")
        pstate = gpu["pstate"]
        if not isinstance(pstate, str) or not re.fullmatch(r"P(?:[0-9]|1[0-5])", pstate):
            raise PowerSampleError("GPU performance state is unavailable or invalid")
        if uuid in uuids or index in indices:
            raise PowerSampleError("GPU observation contains a duplicate UUID or index")
        uuids.add(uuid)
        indices.add(index)
        result.append({
            "index": index,
            "uuid": uuid,
            "name": name.strip(),
            "power_w": _finite_number(gpu["power_w"], "GPU power"),
            "pstate": pstate,
            "utilization_percent": _finite_number(
                gpu["utilization_percent"], "GPU utilization", maximum=100,
            ),
        })
    return result


def parse_power_csv(text: str) -> list[dict]:
    """Parse all rows of index,uuid,name,power.draw,pstate,utilization.gpu.

    Requires ``nvidia-smi --format=csv,noheader,nounits`` output. Unknown power
    or utilization readings fail closed; they never count as idle observations.
    """
    if not isinstance(text, str) or not text.strip():
        raise PowerSampleError("nvidia-smi returned no GPU observations")
    result = []
    try:
        rows = csv.reader(io.StringIO(text.strip()), skipinitialspace=True, strict=True)
        for row in rows:
            if len(row) != 6:
                raise PowerSampleError("nvidia-smi GPU row must have exactly six columns")
            index, uuid, name, power, pstate, utilization = (field.strip() for field in row)
            if not re.fullmatch(r"[0-9]+", index):
                raise PowerSampleError("nvidia-smi GPU index is invalid")
            result.append({
                "index": int(index), "uuid": uuid, "name": name,
                "power_w": power, "pstate": pstate,
                "utilization_percent": utilization,
            })
    except csv.Error as exc:
        raise PowerSampleError(f"Invalid nvidia-smi CSV: {exc}") from exc
    return _validate_gpus(result)


def _instance_identity(sample) -> tuple[str, str, int]:
    if not isinstance(sample, Mapping) or not isinstance(sample.get("identity"), Mapping):
        raise PowerSampleError("GPU sample is missing the instance identity")
    identity = sample["identity"]
    hostname = identity.get("hostname")
    boot_id = identity.get("boot_id")
    ticks = identity.get("pid1_start_ticks")
    if not isinstance(hostname, str) or not hostname.strip():
        raise PowerSampleError("Instance hostname is missing")
    if not isinstance(boot_id, str) or not boot_id.strip():
        raise PowerSampleError("Instance boot ID is missing")
    if isinstance(ticks, str) and re.fullmatch(r"[0-9]+", ticks):
        ticks = int(ticks)
    if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 0:
        raise PowerSampleError("Instance PID 1 start time is invalid")
    return hostname, boot_id, ticks


class IdleWindow:
    """Track uninterrupted observed low power using a caller's monotonic clock.

    Every GPU must stay at or below ``threshold_w``. Read failures must call
    ``reset``; malformed samples also reset before raising ``PowerSampleError``.
    Clock reversal, a long sampling gap, or changed instance/GPU identity starts
    a fresh window and reports the reason. No persisted timer is restored.
    """

    def __init__(self, threshold_w, idle_seconds=1800, max_gap_seconds=90):
        self.threshold_w = _finite_number(threshold_w, "Idle power threshold")
        self.required_idle_seconds = _finite_number(idle_seconds, "Idle duration")
        self.max_gap_seconds = _finite_number(max_gap_seconds, "Maximum sampling gap")
        if self.required_idle_seconds <= 0 or self.max_gap_seconds <= 0:
            raise PowerSampleError("Idle duration and maximum sampling gap must be positive")
        self._identity = None
        self._gpu_uuids = None
        self._idle_since = None
        self._last_observed = None
        self._sample_count = 0

    def _report(self, duration: float, reason: str, *, eligible=False, gpus=None) -> dict:
        return {
            "idle_seconds": duration,
            "required_idle_seconds": self.required_idle_seconds,
            "eligible": eligible,
            "reason": reason,
            "threshold_w": self.threshold_w,
            "gpu_count": len(self._gpu_uuids or ()),
            "sample_count": self._sample_count,
            "max_power_w": max((gpu["power_w"] for gpu in gpus), default=None) if gpus else None,
        }

    def reset(self, reason="sampling_failed") -> dict:
        """Forget elapsed idle time while retaining instance identity guards."""
        self._idle_since = None
        self._last_observed = None
        self._sample_count = 0
        return self._report(0.0, reason)

    def observe(self, sample, now_monotonic) -> dict:
        try:
            now = _finite_number(now_monotonic, "Monotonic observation time")
            identity = _instance_identity(sample)
            gpus = _validate_gpus(sample.get("gpus"))
        except PowerSampleError:
            self.reset("invalid_sample")
            raise

        uuids = frozenset(gpu["uuid"] for gpu in gpus)
        reason = None
        if self._identity is not None and identity != self._identity:
            reason = "identity_changed"
        elif self._gpu_uuids is not None and uuids != self._gpu_uuids:
            reason = "gpu_set_changed"
        elif self._last_observed is not None:
            gap = now - self._last_observed
            if gap < 0:
                reason = "clock_reversed"
            elif gap > self.max_gap_seconds:
                reason = "sample_gap"
        if reason is not None:
            self.reset(reason)

        self._identity = identity
        self._gpu_uuids = uuids
        self._last_observed = now
        self._sample_count += 1
        if any(gpu["power_w"] > self.threshold_w for gpu in gpus):
            self._idle_since = None
            return self._report(0.0, reason or "power_above_threshold", gpus=gpus)

        if self._idle_since is None:
            self._idle_since = now
        duration = now - self._idle_since
        eligible = duration >= self.required_idle_seconds
        return self._report(
            duration, reason or ("idle_timeout" if eligible else "idle"),
            eligible=eligible, gpus=gpus,
        )
