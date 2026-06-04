import cv2
import numpy as np
import math


class KalmanCursor:
    """
    2D Kalman filter for a single point (x, y).

    State vector : [x, y, vx, vy]  — position + velocity
    Measurement  : [x, y]          — raw position from MediaPipe

    Smooths high-frequency jitter while keeping real movement responsive.
    Resets automatically when the point jumps more than JUMP_THRESHOLD pixels
    (e.g. hand re-appearing after leaving the frame).
    """

    # Tune these two values to balance smoothness vs. responsiveness:
    #   PROCESS_NOISE     ↑ = more reactive,  ↓ = smoother
    #   MEASUREMENT_NOISE ↑ = ignores jitter, ↓ = follows MediaPipe closely
    PROCESS_NOISE     = 0.00005
    MEASUREMENT_NOISE = 5e-2
    JUMP_THRESHOLD    = 150     # px — hard reset if point teleports

    def __init__(self):
        # 4 state variables, 2 measurements
        self.kf = cv2.KalmanFilter(4, 2)

        # Only x and y are observed, not velocity
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], np.float32)

        # Constant-velocity motion model: x' = x + vx, y' = y + vy
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], np.float32)

        # Higher noise on velocity components (they change more unpredictably)
        self.kf.processNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 5, 0],
            [0, 0, 0, 5],
        ], np.float32) * self.PROCESS_NOISE

        self.kf.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * self.MEASUREMENT_NOISE
        )

        self._initialized = False

    def _reset(self, x, y):
        """Initialize (or hard-reset) filter state at position (x, y)."""
        state = np.array([[x], [y], [0], [0]], np.float32)
        self.kf.statePre  = state.copy()
        self.kf.statePost = state.copy()
        self._initialized = True

    def update(self, x: int, y: int) -> tuple[int, int]:
        """
        Feed a raw measurement and return the filtered position.
        First call initializes the filter at (x, y) to avoid a startup glitch.
        """
        if not self._initialized:
            self._reset(x, y)
            return x, y

        # Hard reset if the point teleports (hand left and re-entered the frame)
        last_x = self.kf.statePost[0, 0]
        last_y = self.kf.statePost[1, 0]
        if math.hypot(x - last_x, y - last_y) > self.JUMP_THRESHOLD:
            self._reset(x, y)

        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)
        prediction = self.kf.predict()

        return int(prediction[0, 0]), int(prediction[1, 0])