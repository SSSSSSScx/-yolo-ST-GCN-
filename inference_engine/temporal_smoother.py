"""Temporal smoothing via majority voting over a sliding window."""

from collections import deque, Counter


class TemporalSmoother:
    """Majority-vote smoother with priority tie-breaking."""

    def __init__(self, window_size=5, priority_chain=None):
        self.window_size = window_size
        self.priority_chain = priority_chain or [6, 5, 9, 3, 8, 4, 2, 1, 0, 7]
        self._histories: dict[int, deque] = {}

    def smooth(self, class_id, confidence, track_id):
        """Majority voting with safety-class priority override."""
        if track_id not in self._histories:
            self._histories[track_id] = deque(maxlen=self.window_size)

        self._histories[track_id].append((class_id, confidence))
        hist = list(self._histories[track_id])

        # Safety-critical classes (eating/smoking/falling/fight): fire if >=2 in window
        SAFETY_CLASSES = {4, 5, 6, 8, 9}
        for cid in SAFETY_CLASSES:
            safety_hits = [c for c, _ in hist if c == cid]
            if len(safety_hits) >= 2:
                confs = [conf for c, conf in hist if c == cid]
                return (cid, sum(confs) / len(confs))

        # Standard majority vote for posture classes
        counts = Counter(cid for cid, _ in hist)
        max_count = max(counts.values())
        candidates = [cid for cid, cnt in counts.items() if cnt == max_count]
        if len(candidates) > 1:
            for cid in self.priority_chain:
                if cid in candidates:
                    winner = cid
                    break
        else:
            winner = candidates[0]
        confs = [c for cid, c in hist if cid == winner]
        return (winner, sum(confs) / len(confs))

    def reset(self, track_id):
        self._histories.pop(track_id, None)
