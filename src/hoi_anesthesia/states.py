"""Shared state membership for responsive and non-responsive analyses."""

RESPONSIVE_STATES = {
    "MA": ("MA_awake",),
    "DBS": ("DBS_awake",),
}

NONRESPONSIVE_STATES = {
    "MA": ("deep_propofol", "ketamine", "ts_selv4"),
    "DBS": ("ts_off", "ts_on_3V_control", "ts_on_5V_control"),
}
