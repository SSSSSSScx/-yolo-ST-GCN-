"""Action label definitions for lab safety monitoring."""

ACTION_LABELS = {
    0: "站立",
    1: "正常行走",
    2: "坐姿",
    3: "奔跑",
    4: "饮食动作",
    5: "摔倒",
    6: "倒地不动",
    7: "其他操作",
    8: "抽烟",
    9: "推搡嬉闹",
}

# Reverse lookup
LABEL_TO_ID = {v: k for k, v in ACTION_LABELS.items()}

# Action → rule → danger level → display color (hex)
ACTION_COLORS: dict[int, str] = {
    0: "#3fb950",  # 站立 — green
    1: "#3fb950",  # 正常行走 — green
    2: "#3fb950",  # 坐姿 — green
    3: "#d29922",  # 奔跑 — yellow (A03 L1)
    4: "#f0883e",  # 饮食动作 — orange (A01 L2)
    5: "#f85149",  # 摔倒 — red (A05 L3)
    6: "#f85149",  # 倒地不动 — red (A05 L3)
    7: "#8892a4",  # 其他操作 — muted gray
    8: "#f0883e",  # 抽烟 — orange (A04 L2)
    9: "#d29922",  # 推搡嬉闹 — yellow (A02 L1)
}
