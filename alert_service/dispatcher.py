import time
import threading
import queue
from loguru import logger

try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False


class AlertDispatcher:
    """Dispatches alerts based on danger level.

    L1: Console output (yellow) + simple beep
    L2: Console output (red) + repeated beep + desktop popup
    L3: Console output (red blink) + continuous alarm + simulated emergency

    ALL sound/popup operations run on a background worker thread so
    dispatch() returns instantly and never blocks the main frame loop.
    """

    def __init__(self):
        self._last_dispatch: dict[int, float] = {}
        self._dispatch_cooldown = 3.0  # seconds between same-level dispatches
        self._alarm_stop_event: threading.Event | None = None

        # Background sound worker — all blocking I/O queued here
        self._sound_queue: queue.Queue = queue.Queue(maxsize=20)
        self._sound_thread = threading.Thread(target=self._sound_worker, daemon=True)
        self._sound_thread.start()

    def _sound_worker(self) -> None:
        """Processes sound/popup requests sequentially in background."""
        while True:
            try:
                task_type, payload = self._sound_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                if task_type == "beep":
                    freq, duration = payload
                    self._do_beep(freq, duration)
                elif task_type == "alarm":
                    stop_event = payload
                    self._do_continuous_alarm(stop_event)
            except Exception as e:
                logger.debug(f"[SOUND] Worker error: {e}")

    def dispatch(self, alerts: list[dict]) -> None:
        """Dispatch alerts. Returns instantly — sound plays in background."""
        if not alerts:
            return

        now = time.time()
        highest = max(a["level"] for a in alerts)

        # Cooldown for same level
        last = self._last_dispatch.get(highest, 0)
        if now - last < self._dispatch_cooldown:
            return
        self._last_dispatch[highest] = now
        logger.info(f"[DISPATCH] Firing {len(alerts)} alert(s), highest_level={highest}")

        # Log + queue sound for each alert (non-blocking)
        for alert in alerts:
            level = alert["level"]
            message = alert.get("message", "")

            if level == 1:
                self._dispatch_l1(message)
            elif level == 2:
                self._dispatch_l2(message)
            elif level >= 3:
                self._dispatch_l3(message)

    def _enqueue(self, task_type: str, payload) -> None:
        """Non-blocking enqueue for sound worker."""
        try:
            self._sound_queue.put_nowait((task_type, payload))
        except queue.Full:
            logger.debug("[SOUND] Queue full, dropping sound task")

    def _dispatch_l1(self, message: str) -> None:
        logger.warning(f"[L1 告警] {message}")
        print(f"\033[33m[L1] WARNING: {message}\033[0m")
        self._enqueue("beep", (800, 300))

    def _dispatch_l2(self, message: str) -> None:
        logger.error(f"[L2 告警] {message}")
        print(f"\033[31m[L2] CRITICAL: {message}\033[0m")
        self._enqueue("beep", (1000, 500))
        self._enqueue("beep", (800, 500))

    def _dispatch_l3(self, message: str) -> None:
        logger.critical(f"[L3 紧急告警] {message}")
        print(f"\033[5;31m[L3] ALARM: {message}\033[0m")
        # Stop any previous alarm, start fresh
        if self._alarm_stop_event:
            self._alarm_stop_event.set()
        self._alarm_stop_event = threading.Event()
        self._enqueue("alarm", self._alarm_stop_event)

    @staticmethod
    def _do_beep(freq: int, duration_ms: int) -> None:
        if _HAS_WINSOUND:
            try:
                winsound.Beep(freq, duration_ms)
            except Exception:
                pass
        else:
            print("\a")

    @staticmethod
    def _do_continuous_alarm(stop_event: threading.Event) -> None:
        for _ in range(10):
            if stop_event.is_set():
                break
            if _HAS_WINSOUND:
                try:
                    winsound.Beep(1500, 200)
                except Exception:
                    pass
            time.sleep(0.3)

