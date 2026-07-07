"""Priority-based arbitration with two-tier fallback.

Priority chain: fallen(6) > falling(5) > fight(9) > running(3)
> smoking(8) > eating(4) > sitting(2) > walking(1) > standing(0) > other(7)
"""


class PriorityArbiter:
    """Resolves conflicting detector outputs via priority chain + fallback."""

    def __init__(self):
        self.priority_chain = [6, 5, 9, 3, 8, 4, 2, 1, 0, 7]
        self.thresholds = {6: 0.8, 5: 0.8, 9: 0.8, 3: 0.6,
                          8: 0.6, 4: 0.6, 2: 0.6, 1: 0.6,
                          0: 0.6, 7: 0.5}
        self.fallback_threshold = 0.85   # high-priority must exceed this to block fallback
        self.rival_threshold = 0.9       # low-priority must exceed this to trigger fallback
        self.high_priority_set = {6, 5, 9}  # classes that trigger fallback check

    def arbitrate(self, detections: dict) -> tuple:
        """detections: {class_id: confidence}

        Returns: (class_id, confidence)
        """
        for cid in self.priority_chain:
            if cid in detections and detections[cid] >= self.thresholds.get(cid, 0.5):
                conf = detections[cid]

                # Fallback check for high-priority classes
                if cid in self.high_priority_set and conf < self.fallback_threshold:
                    # Check if any lower-priority class has very high confidence
                    for lower_cid in self.priority_chain[self.priority_chain.index(cid) + 1:]:
                        if lower_cid in detections and detections[lower_cid] > self.rival_threshold:
                            # Skip this high-priority result, continue to lower
                            break
                    else:
                        # No rival found, keep high-priority result
                        return (cid, conf)
                    # Rival found, continue traversal
                    continue

                return (cid, conf)

        # Nothing matched
        return (7, 0.3)
