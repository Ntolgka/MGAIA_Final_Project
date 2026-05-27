"""Shared UI region coordinates and utility functions for IVL video processing.

All coordinates are for 1920x1080 frames from the IVL broadcast.
Calibrated against clips 023-044.
"""

# --- Heart-rate digit regions (icon + number, right sidebar) ---
# 5 player panels on the right sidebar.
# Panels 0-3 are survivors, panel 4 is hunter (or 5th display slot).
# Each region captures the HR icon and digit string.
HEART_RATE_REGIONS = {
    0: (1700, 100, 1820, 135),
    1: (1700, 303, 1820, 338),
    2: (1700, 483, 1820, 518),
    3: (1700, 685, 1820, 720),
    4: (1700, 857, 1820, 892),
}

# --- Survivor portrait / avatar silhouette regions ---
# These match the 92h x 98w crops used for CNN training.
# The gray character silhouettes in the right sidebar panels.
PORTRAIT_REGIONS = {
    0: (1690, 130, 1788, 222),
    1: (1690, 210, 1788, 302),
    2: (1690, 410, 1788, 502),
    3: (1690, 590, 1788, 682),
    4: (1690, 770, 1788, 862),
}

# --- Match timer region (bottom-left) ---
# Shows MM:SS format, white digits on dark background.
TIMER_REGION = (10, 793, 130, 840)

# --- Cipher count / decoding progress (top-center) ---
# Shows "N条密码尚未破译" (N ciphers remaining).
CIPHER_REGION = (840, 40, 1100, 80)

# --- Decoding progress bars (left sidebar) ---
# Individual cipher machine progress percentages.
DECODE_PROGRESS_REGION = (180, 270, 280, 400)

# --- Team and score info (top corners) ---
TEAM_LEFT_REGION = (30, 10, 250, 60)
TEAM_RIGHT_REGION = (1680, 10, 1910, 60)

# --- CNN model config ---
CNN_IMG_HEIGHT = 92
CNN_IMG_WIDTH = 98
CNN_NUM_CLASSES = 7
CNN_CLASS_NAMES = [
    "0_healthy",
    "1_injured",
    "2_downed",
    "3_ballooned",
    "4_chaired",
    "5_eliminated",
    "6_escaped",
]

# Simplified labels for analysis
STATUS_LABELS = {
    "0_healthy": "healthy",
    "1_injured": "injured",
    "2_downed": "downed",
    "3_ballooned": "ballooned",
    "4_chaired": "chaired",
    "5_eliminated": "eliminated",
    "6_escaped": "escaped",
}
