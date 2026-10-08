import json
from pathlib import Path


# ============================================================
# LOAD FUSION CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CALIBRATION_FILE = BASE_DIR / "config" / "calibration.json"


with open(CALIBRATION_FILE, "r", encoding="utf-8-sig") as f:
    CALIBRATION = json.load(f)


FUSION_CONFIG = CALIBRATION["fusion"]

AUDIO_WEIGHT = float(FUSION_CONFIG["audio_weight"])
VISUAL_WEIGHT = float(FUSION_CONFIG["visual_weight"])
THRESHOLD = float(FUSION_CONFIG["threshold"])


# ============================================================
# VALIDATE FUSION WEIGHTS
# ============================================================

WEIGHT_SUM = AUDIO_WEIGHT + VISUAL_WEIGHT

if abs(WEIGHT_SUM - 1.0) > 1e-9:
    raise ValueError(
        f"Fusion weights must sum to 1.0. "
        f"Current sum: {WEIGHT_SUM}"
    )


# ============================================================
# FUSION FUNCTION
# ============================================================

def fuse_probabilities(
    audio_probability: float,
    visual_probability: float
) -> dict:
    """
    Perform late fusion of independently calibrated
    audio and visual spoof probabilities.

    Formula:

        P_fusion =
            audio_weight * P_audio +
            visual_weight * P_visual

    Decision:

        P_fusion >= threshold -> SPOOF
        P_fusion < threshold  -> REAL
    """

    audio_probability = float(audio_probability)
    visual_probability = float(visual_probability)

    # --------------------------------------------------------
    # Validate probability ranges
    # --------------------------------------------------------

    if not 0.0 <= audio_probability <= 1.0:
        raise ValueError(
            f"Audio probability must be between 0 and 1. "
            f"Received: {audio_probability}"
        )

    if not 0.0 <= visual_probability <= 1.0:
        raise ValueError(
            f"Visual probability must be between 0 and 1. "
            f"Received: {visual_probability}"
        )

    # --------------------------------------------------------
    # Weighted late fusion
    # --------------------------------------------------------

    fusion_probability = (
        AUDIO_WEIGHT * audio_probability
        + VISUAL_WEIGHT * visual_probability
    )

    # --------------------------------------------------------
    # Final classification
    # --------------------------------------------------------

    if fusion_probability >= THRESHOLD:
        prediction = "SPOOF"
    else:
        prediction = "REAL"

    return {
        "audio_probability_spoof": audio_probability,
        "visual_probability_spoof": visual_probability,
        "audio_weight": AUDIO_WEIGHT,
        "visual_weight": VISUAL_WEIGHT,
        "fusion_probability_spoof": float(fusion_probability),
        "threshold": THRESHOLD,
        "prediction": prediction
    }


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def predict_from_probabilities(
    audio_probability: float,
    visual_probability: float
) -> str:
    """
    Return only the final REAL/SPOOF prediction.
    """

    result = fuse_probabilities(
        audio_probability,
        visual_probability
    )

    return result["prediction"]
