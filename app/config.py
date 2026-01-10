DEFAULT_CHECKPOINT = "best_ep9_emo0.6390_pkl0.8269.pt"
DEFAULT_NUM_QUESTIONS = 5
DEFAULT_HISTORY_MAX = 3
ENABLE_RADAR_PLOTS = True
SHOW_JSON_BLOCK = True
MATCH_SIMILARITY_THRESHOLD = 0.8

EMO_ORDER = ["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
EMO_ORDER_FOR_BARS = ["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
PERS_ORDER = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Non-Neuroticism"]
TARGET_TRAIT_NAMES = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Non-Neuroticism",
    "Conversation",
    "Openness to Change",
    "Hedonism",
    "Self-enhancement",
    "Self-transcendence",
]

PERS_COLORS = {
    "Extraversion": "#F97316",
    "Agreeableness": "#22C55E",
    "Conscientiousness": "#0EA5E9",
    "Non-Neuroticism": "#8B5CF6",
    "Openness": "#EAB308",
}

DEMO_DIR = "demo_video"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Visual metrics thresholds (framing / lighting)
VISUAL_POSITION_CENTER_TOL = 0.10

VISUAL_DISTANCE_FACE_CLOSE = 0.18
VISUAL_DISTANCE_FACE_FAR = 0.08
VISUAL_DISTANCE_BODY_CLOSE = 0.45
VISUAL_DISTANCE_BODY_FAR = 0.25

VISUAL_LIGHT_LOW_P95 = 120.0
VISUAL_LIGHT_DARK_RATIO = 0.40
VISUAL_LIGHT_OVEREXPOSED_RATIO = 0.02
VISUAL_LIGHT_FLAT_CONTRAST = 35.0
VISUAL_LIGHT_BACKLIT_DELTA = 25.0

VISUAL_CENTER_RATIO_GOOD = 0.60
VISUAL_POSITION_STABILITY_BAD = 0.08

# Audio quality thresholds (dBFS)
AUDIO_LOUDNESS_LOW = -35.0
AUDIO_LOUDNESS_HIGH = -20.0
AUDIO_NOISE_LOW = -50.0
AUDIO_NOISE_HIGH = -40.0
AUDIO_FRAME_SEC = 0.05
AUDIO_HOP_SEC = 0.025
