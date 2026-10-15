"""
kalman_filter.py — One Euro Filter implementation for landmark smoothing
═══════════════════════════════════════════════════════════════════════════
The 1€ filter (Casiez et al., CHI 2012) dynamically adjusts its cutoff
frequency based on signal velocity, solving the lag-vs-jitter tradeoff:
  - Low speed → low cutoff → heavy smoothing (eliminates tremor)
  - High speed → high cutoff → minimal lag (ballistic motion)

Used by hand_detect.py for all 21 hand landmarks.
"""
import math
from config import SMOOTHING


class OneEuroFilter:
    """
    One Euro Filter for a single scalar value.
    Maintains internal state across frames.
    """
    __slots__ = ('_last_x', '_last_dx', '_last_t', 'fc_min', 'beta', 'fc_d')

    def __init__(self, fc_min=None, beta=None, fc_d=None):
        self.fc_min = fc_min if fc_min is not None else SMOOTHING['min_cutoff']
        self.beta   = beta   if beta   is not None else SMOOTHING['beta']
        self.fc_d   = fc_d   if fc_d   is not None else SMOOTHING['d_cutoff']
        self._last_x  = None
        self._last_dx = None
        self._last_t  = None

    def _alpha(self, fc: float, dt: float) -> float:
        """Compute smoothing factor alpha from cutoff frequency and delta-time."""
        if dt <= 0:
            return 1.0
        tau = 1.0 / (2.0 * math.pi * fc)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, t: float) -> float:
        """
        Filter a scalar measurement x at timestamp t (seconds, monotonically increasing).
        Returns the filtered value.
        """
        if self._last_t is None:
            self._last_x  = x
            self._last_dx = 0.0
            self._last_t  = t
            return x

        dt = t - self._last_t
        if dt <= 0:
            return self._last_x

        # 1. Estimate raw velocity
        dx_raw = (x - self._last_x) / dt

        # 2. Filter the velocity (fixed cutoff fc_d)
        alpha_d = self._alpha(self.fc_d, dt)
        dx = alpha_d * dx_raw + (1.0 - alpha_d) * self._last_dx

        # 3. Compute dynamic cutoff: fc = fc_min + beta * |dx|
        fc = self.fc_min + self.beta * abs(dx)

        # 4. Filter the signal with dynamic cutoff
        alpha = self._alpha(fc, dt)
        x_hat = alpha * x + (1.0 - alpha) * self._last_x

        # Update state
        self._last_x  = x_hat
        self._last_dx = dx
        self._last_t  = t
        return x_hat


class OneEuroFilter2D:
    """
    Two independent 1€ filters for (x, y) coordinates.
    Used for fingertip tracking and cursor control.
    """
    __slots__ = ('_fx', '_fy',)

    def __init__(self, fc_min=None, beta=None, fc_d=None):
        self._fx = OneEuroFilter(fc_min, beta, fc_d)
        self._fy = OneEuroFilter(fc_min, beta, fc_d)

    def filter(self, x: float, y: float, t: float) -> tuple:
        """Filter (x, y) at timestamp t. Returns (x_hat, y_hat)."""
        return self._fx.filter(x, t), self._fy.filter(y, t)


class LandmarkSmoother:
    """
    Applies a 1€ filter to each of the 21 MediaPipe hand landmarks.
    Handles reset on jump (hand re-entry).
    """
    __slots__ = ('filters', 'jump_ratio', '_last_raw', '_last_t')

    def __init__(self, jump_ratio=None):
        self.filters = [OneEuroFilter2D() for _ in range(21)]
        self.jump_ratio = jump_ratio if jump_ratio is not None else SMOOTHING['jump_ratio']
        self._last_raw = None
        self._last_t   = None

    def _hand_scale(self, pts: list) -> float:
        """Compute hand scale (wrist to middle-finger base distance)."""
        if len(pts) < 10:
            return 100.0
        dx = pts[9][0] - pts[0][0]
        dy = pts[9][1] - pts[0][1]
        s = math.hypot(dx, dy)
        return s if s > 1.0 else 1.0

    def smooth(self, raw_pts: list, t: float) -> list:
        """
        Smooth 21 raw landmarks at timestamp t.
        raw_pts: list of (x, y) tuples (pixel coordinates).
        Returns list of filtered (x_hat, y_hat) tuples.
        Resets if the hand jumps more than jump_ratio * hand_scale.
        """
        if len(raw_pts) != 21:
            return raw_pts

        if self._last_raw is None:
            self._last_raw = raw_pts
            self._last_t   = t
            return raw_pts

        dt = t - self._last_t if self._last_t is not None else 0.016
        if dt <= 0:
            dt = 0.016

        # Check for jump (hand exit/re-entry)
        scale = self._hand_scale(raw_pts)
        max_jump = scale * self.jump_ratio
        jump = max(math.hypot(raw_pts[i][0] - self._last_raw[i][0],
                              raw_pts[i][1] - self._last_raw[i][1])
                   for i in range(21))

        if jump > max_jump:
            # Reset all filters
            self.filters = [OneEuroFilter2D() for _ in range(21)]
            self._last_raw = raw_pts
            return raw_pts

        # Filter each landmark
        smoothed = []
        for i in range(21):
            x_hat, y_hat = self.filters[i].filter(raw_pts[i][0], raw_pts[i][1], t)
            smoothed.append((x_hat, y_hat))

        self._last_raw = smoothed
        self._last_t   = t
        return smoothed
