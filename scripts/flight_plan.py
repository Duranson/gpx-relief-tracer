"""Camera flight plan for GPX relief animation.

Pure Python — no Blender dependency.  Fully unit-testable.

Usage
-----
    from flight_plan import FlightPlan, Start, ForwardFollow, Rotate, BackwardFollow

    plan = FlightPlan(
        steps=[
            Start(end_t=0.0,  azimuth=140.0, elevation=25.0, distance=5000.0),
            ForwardFollow(end_t=0.5,  distance=3000.0, height=500.0),
            Rotate(end_t=0.62, end_azimuth=235.0, end_elevation=30.0, distance=3000.0),
            BackwardFollow(end_t=1.0, distance=3000.0, height=500.0),
        ],
        smoothing=0.05,
    )

    # Call in strictly ascending t order:
    for frame in range(total_frames + 1):
        t = frame / total_frames
        pose = plan.camera_pose(t, gpx_trace)
        # pose.position, pose.target → set Blender camera
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import radians, degrees, atan2, cos, sin, sqrt


# ═══════════════════════════════════════════════════════════════════════════════
# Math helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp3(a: tuple, b: tuple, t: float) -> tuple:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def _smootherstep(t: float) -> float:
    """Ken Perlin's smoother step — zero velocity AND acceleration at both ends.

    Used for Rotate so the orbital sweep feels organic rather than mechanical.
    """
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp_angle(a: float, b: float, t: float) -> float:
    """Interpolate angles (degrees) along the shorter arc (handles 359°→1° wrap)."""
    diff = ((b - a) + 180.0) % 360.0 - 180.0
    return a + diff * t


def _spherical_to_world(center: tuple, az_rad: float, el_rad: float, dist: float) -> tuple:
    """World position at (azimuth, elevation, distance) from center.

    Convention: az=0 → +X axis, az=π/2 → +Y axis (standard math, not compass).
    """
    return (
        center[0] + dist * cos(el_rad) * cos(az_rad),
        center[1] + dist * cos(el_rad) * sin(az_rad),
        center[2] + dist * sin(el_rad),
    )


def _world_to_spherical(world_pos: tuple, center: tuple) -> tuple:
    """Return (azimuth_deg, elevation_deg, distance) of world_pos relative to center."""
    dx = world_pos[0] - center[0]
    dy = world_pos[1] - center[1]
    dz = world_pos[2] - center[2]
    dist = sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    h = sqrt(dx * dx + dy * dy) or 1e-9
    return degrees(atan2(dy, dx)), degrees(atan2(dz, h)), dist


# ═══════════════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CameraPose:
    """World-space camera location and look-at target."""
    position: tuple  # (x, y, z)
    target:   tuple  # (x, y, z)


# ═══════════════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════════════

class CameraStep(ABC):
    """One segment of a FlightPlan.

    Subclasses return the *desired* unsmoothed camera pose as a function of
    local progress (local_t ∈ [0, 1]) and the current GPX state.
    The FlightPlan applies EMA smoothing on top of all desired poses.

    The start_az / start_el / start_dist parameters contain the spherical
    coordinates of the EMA camera position relative to the current GPX head at
    the moment this step began.  Steps that don't orbit (ForwardFollow,
    BackwardFollow) may ignore them.
    """

    @abstractmethod
    def desired_pose(
        self,
        local_t:    float,
        global_t:   float,
        gpx,
        start_az:   float,
        start_el:   float,
        start_dist: float,
    ) -> CameraPose:
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# Concrete steps
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Start(CameraStep):
    """Initial camera pose.  Set end_t=0.0 for an instantaneous starting condition.

    Azimuth and elevation use the same convention as the rest of the system:
        azimuth=0   → camera is in the +X direction from the GPX head
        azimuth=90  → camera is in the +Y direction
        elevation=0 → camera is level with the head
        elevation=25→ camera is above and looking down at ≈25°
    """
    end_t:     float = 0.0
    azimuth:   float = 140.0   # degrees, math convention (atan2)
    elevation: float = 25.0    # degrees above horizontal
    distance:  float = 5000.0  # metres from GPX head at t=0

    def desired_pose(self, local_t, global_t, gpx, start_az, start_el, start_dist):
        head = gpx.head_at(global_t)
        pos  = _spherical_to_world(head, radians(self.azimuth),
                                   radians(self.elevation), self.distance)
        return CameraPose(position=pos, target=head)


@dataclass
class ForwardFollow(CameraStep):
    """Camera trails *behind* the trace head along the direction of travel.

    The helicopter drives forward — it sees where the hiker is going.
    The look-at point is a fixed distance ahead of the head so the camera
    always anticipates the route.
    """
    end_t:      float
    distance:   float = 3000.0  # metres behind the head (along travel direction)
    height:     float = 500.0   # metres above the head
    look_ahead: float = 200.0   # metres ahead of the head to aim at

    def desired_pose(self, local_t, global_t, gpx, start_az, start_el, start_dist):
        head = gpx.head_at(global_t)
        d    = gpx.travel_dir_at(global_t)
        pos  = (
            head[0] - d[0] * self.distance,
            head[1] - d[1] * self.distance,
            head[2] + self.height,
        )
        target = (
            head[0] + d[0] * self.look_ahead,
            head[1] + d[1] * self.look_ahead,
            head[2],
        )
        return CameraPose(position=pos, target=target)


@dataclass
class BackwardFollow(CameraStep):
    """Camera leads *ahead* of the trace head, looking back toward it.

    The helicopter drives backward — the hiker walks toward the camera.
    """
    end_t:       float
    distance:    float = 3000.0  # metres ahead of the head (along travel direction)
    height:      float = 500.0   # metres above the head
    look_behind: float = 0.0     # metres behind the head to aim at (0 = aim at head)

    def desired_pose(self, local_t, global_t, gpx, start_az, start_el, start_dist):
        head = gpx.head_at(global_t)
        d    = gpx.travel_dir_at(global_t)
        pos  = (
            head[0] + d[0] * self.distance,
            head[1] + d[1] * self.distance,
            head[2] + self.height,
        )
        target = (
            head[0] - d[0] * self.look_behind,
            head[1] - d[1] * self.look_behind,
            head[2],
        )
        return CameraPose(position=pos, target=target)


@dataclass
class Rotate(CameraStep):
    """Camera orbits around the *current GPX head*, sweeping to a target azimuth/elevation.

    The starting azimuth and elevation are captured automatically from the EMA
    camera position at the moment the step begins — no need to specify them.

    Rotation uses smootherstep easing so the sweep accelerates gently at the
    start and decelerates at the end (cinematic feel, no mechanical constant speed).
    """
    end_t:         float
    end_azimuth:   float         # target azimuth in degrees (math convention)
    end_elevation: float         # target elevation in degrees above horizontal
    distance:      float = 3000.0  # orbital radius; also interpolated from start_dist

    def desired_pose(self, local_t, global_t, gpx, start_az, start_el, start_dist):
        head = gpx.head_at(global_t)
        s    = _smootherstep(local_t)
        az   = radians(_lerp_angle(start_az, self.end_azimuth, s))
        el   = radians(_lerp(start_el, self.end_elevation, s))
        dist = _lerp(start_dist, self.distance, s)
        pos  = _spherical_to_world(head, az, el, dist)
        return CameraPose(position=pos, target=head)


# ═══════════════════════════════════════════════════════════════════════════════
# Flight plan
# ═══════════════════════════════════════════════════════════════════════════════

class FlightPlan:
    """Orchestrates a list of CameraStep objects into a smooth camera animation.

    The EMA smoothing is the primary mechanism that ensures C0 continuity
    at step boundaries and removes jitter caused by the jerky GPS track.

    Args
    ----
    steps     : list of CameraStep objects, ordered by ascending end_t.
                The first element should be a Start() that defines the initial pose.
    smoothing : EMA blend factor applied every frame.
                0 → camera never moves, 1 → instant snap to desired position.
                Typical range: 0.03 (very floaty) … 0.15 (tight).
    """

    def __init__(self, steps: list, smoothing: float = 0.05):
        self.steps = steps
        self.smoothing = smoothing
        self._reset()

    # ── State management ─────────────────────────────────────────────────────

    def _reset(self):
        """Clear EMA state.  Call before a fresh sequential pass (e.g. preview warmup)."""
        self._ema_pos:    tuple | None = None
        self._ema_target: tuple | None = None
        # Spherical coords of EMA camera at the start of each step (captured lazily)
        self._step_start: dict = {}

    # ── Step lookup ──────────────────────────────────────────────────────────

    def _active_step(self, t: float):
        """Return (step, local_t ∈ [0,1], step_index) for global animation fraction t."""
        prev_end = 0.0
        for i, step in enumerate(self.steps):
            if t <= step.end_t + 1e-9:
                duration = step.end_t - prev_end
                local_t  = (t - prev_end) / duration if duration > 1e-9 else 1.0
                return step, max(0.0, min(1.0, local_t)), i
            prev_end = step.end_t
        # Past the last step: hold final pose
        return self.steps[-1], 1.0, len(self.steps) - 1

    # ── Main API ─────────────────────────────────────────────────────────────

    def camera_pose(self, t: float, gpx) -> CameraPose:
        """Return the EMA-smoothed camera pose at animation fraction t.

        IMPORTANT: must be called in strictly ascending t order so that the
        EMA state accumulates correctly.  Call _reset() before replaying from t=0.
        """
        step, local_t, step_idx = self._active_step(t)

        # Capture step start spherical coords on first entry to each step
        if step_idx not in self._step_start:
            if self._ema_pos is not None:
                az, el, dist = _world_to_spherical(self._ema_pos, gpx.head_at(t))
            elif isinstance(step, Start):
                az, el, dist = step.azimuth, step.elevation, step.distance
            else:
                az, el, dist = 140.0, 25.0, 5000.0   # sensible default fallback
            self._step_start[step_idx] = (az, el, dist)

        start_az, start_el, start_dist = self._step_start[step_idx]
        desired = step.desired_pose(local_t, t, gpx, start_az, start_el, start_dist)

        # EMA smoothing toward desired pose
        a = self.smoothing
        if self._ema_pos is None:
            self._ema_pos    = desired.position
            self._ema_target = desired.target
        else:
            self._ema_pos    = _lerp3(self._ema_pos,    desired.position, a)
            self._ema_target = _lerp3(self._ema_target, desired.target,   a)

        return CameraPose(position=self._ema_pos, target=self._ema_target)

    # ── Preview ──────────────────────────────────────────────────────────────

    def preview_poses(self, gpx) -> list:
        """Return (t, step, CameraPose) triples — 3 per non-trivial step.

        Each pose is computed by replaying the EMA from t=0 up to the target t,
        giving a faithful snapshot of what the animation will look like at that moment.
        Zero-duration steps (e.g. Start with end_t=0.0) are skipped.
        """
        results = []
        prev_end_t = 0.0

        for step in self.steps:
            duration = step.end_t - prev_end_t
            if duration < 1e-6:
                prev_end_t = step.end_t
                continue   # zero-duration step (Start) — skip preview

            start_t = prev_end_t

            for label, frac in (('start', 0.1), ('mid', 0.5), ('end', 0.9)):
                target_t = start_t + duration * frac
                self._reset()
                # Warm up EMA from 0 → target_t over 300 steps
                n = 300
                for i in range(n):
                    tw = target_t * i / max(n - 1, 1)
                    self.camera_pose(tw, gpx)
                pose = self.camera_pose(target_t, gpx)
                results.append((target_t, step, label, pose))

            prev_end_t = step.end_t

        self._reset()
        return results