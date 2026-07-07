"""GPX trace spatial queries for the flight plan camera system.

Wraps the list of Blender-projected 3-D points and GPS timestamps produced
by blender_animate.project_gpx().  All public methods accept a normalised
animation fraction t ∈ [0, 1].
"""

from math import sqrt
import numpy as np


class GPXTrace:

    def __init__(self, projected_points, timestamps):
        """
        projected_points : list of Vector / 3-tuples in Blender scene space
        timestamps        : list of datetime | None, same length
        """
        self._pts = [(float(p[0]), float(p[1]), float(p[2])) for p in projected_points]
        self._n = len(self._pts)
        # Pre-compute a 1001-sample lookup table: t → fractional GPX index
        self._lut = self._build_lut(timestamps)

    # ── LUT construction ─────────────────────────────────────────────────────

    def _build_lut(self, timestamps):
        n = self._n
        samples = 1001
        valid = [(i, ts) for i, ts in enumerate(timestamps) if ts is not None]

        if len(valid) >= 2:
            ts0 = valid[0][1]
            total_s = (valid[-1][1] - ts0).total_seconds()
            gps_t = np.array([(ts - ts0).total_seconds() / total_s for _, ts in valid])
            gps_i = np.array([i for i, _ in valid], dtype=float)
            frame_t = np.linspace(0.0, 1.0, samples)
            return np.interp(frame_t, gps_t, gps_i)

        # No timestamps → uniform distribution
        return np.linspace(0.0, float(n - 1), samples)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _frac_idx(self, t: float) -> float:
        """Fractional GPX point index at animation fraction t."""
        t = max(0.0, min(1.0, t))
        idx_f = t * (len(self._lut) - 1)
        lo = int(idx_f)
        hi = min(lo + 1, len(self._lut) - 1)
        f = idx_f - lo
        return float(self._lut[lo] * (1.0 - f) + self._lut[hi] * f)

    def _interp(self, frac_idx: float):
        n = self._n
        idx0 = max(0, min(int(frac_idx), n - 2))
        f = frac_idx - idx0
        p0, p1 = self._pts[idx0], self._pts[idx0 + 1]
        return (
            p0[0] + (p1[0] - p0[0]) * f,
            p0[1] + (p1[1] - p0[1]) * f,
            p0[2] + (p1[2] - p0[2]) * f,
        )

    # ── Public API ───────────────────────────────────────────────────────────

    def head_at(self, t: float) -> tuple:
        """3-D position of the trace head at animation fraction t ∈ [0, 1]."""
        return self._interp(self._frac_idx(t))

    def travel_dir_at(self, t: float, window: float = 0.04) -> tuple:
        """Normalised horizontal direction of travel at t.

        window controls how wide a t-span is used for the finite difference;
        larger values give a smoother direction at the cost of local accuracy.
        """
        ta = max(0.0, t - window / 2)
        tb = min(1.0, t + window / 2)
        pa = self._interp(self._frac_idx(ta))
        pb = self._interp(self._frac_idx(tb))
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        length = sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            return (1.0, 0.0, 0.0)
        return (dx / length, dy / length, 0.0)

    def bevel_factor_at(self, t: float) -> float:
        """bevel_factor_end value (0 → 1) for the Blender GPX curve at t."""
        if self._n < 2:
            return 1.0
        return self._frac_idx(t) / (self._n - 1)

    @property
    def n_points(self) -> int:
        return self._n