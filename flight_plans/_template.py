"""Template flight plan — copy to flight_plans/<GPX_NAME>.py for a new route.

<GPX_NAME> must match the GPX_NAME set in scripts/blender_animate.py (the
GPX filename without the .gpx extension).
"""

from flight_plan import FlightPlan, Start, ForwardFollow, BackwardFollow, Rotate

PLAN = FlightPlan(
    steps=[
        # Start defines where the camera is at t=0.
        # Set end_t=0.0 so it is instantaneous (no preview image generated).
        Start(end_t=0.0, azimuth=140.0, elevation=25.0, distance=5000.0),

        # First half: trail behind the trace head
        ForwardFollow(end_t=0.25, distance=3000.0, height=500.0, look_ahead=200.0),

        # Sweep around to a different vantage point
        Rotate(end_t=0.67, end_azimuth=235.0, end_elevation=30.0, distance=3000.0),

        # Second half: lead ahead of the trace head, looking back
        BackwardFollow(end_t=1.0, distance=3000.0, height=500.0),
    ],
    smoothing=0.025,   # EMA factor; lower = floatier, higher = snappier
)
