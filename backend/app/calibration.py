import json
from pathlib import Path
import math


# ============================================================
# LOAD CALIBRATION PARAMETERS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CALIBRATION_FILE = BASE_DIR / "config" / "calibration.json"


with open(CALIBRATION_FILE, "r", encoding="utf-8-sig") as f:
    CALIBRATION = json.load(f)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sigmoid(x: float) -> float:
    """
    Convert a logit into a probability.
    """
    return 1.0 / (1.0 + math.exp(-x))


def standardize(score: float, mean: float, std: float) -> float:
    """
    Standardize a raw score using calibration-set statistics.
    """
    if std <= 0:
        raise ValueError(
            "Calibration standard deviation must be greater than zero."
        )

    return (score - mean) / std


# ============================================================
# AUDIO CALIBRATION
# ============================================================

def calibrate_audio(score_spoof: float) -> float:
    """
    Convert the raw AASIST-L score_spoof into
    calibrated probability of spoof.

    Calibration:
        z = (score - mean) / std
        logit = intercept + coefficient * z
        probability = sigmoid(logit)
    """

    params = CALIBRATION["audio"]

    mean = params["standardization"]["mean"]
    std = params["standardization"]["std"]

    intercept = params["logistic_intercept"]
    coefficient = params["logistic_coefficient"]

    z = standardize(score_spoof, mean, std)

    logit = intercept + coefficient * z

    probability = sigmoid(logit)

    return float(probability)


# ============================================================
# VISUAL CALIBRATION
# ============================================================

def calibrate_visual(raw_realness: float) -> float:
    """
    Convert the CDCN++ raw realness score into
    calibrated probability of spoof.

    ROSE-Youtu calibration score definition:
        spoof_score = -raw_realness

    Calibration:
        z = (spoof_score - mean) / std
        logit = intercept + coefficient * z
        probability = sigmoid(logit)
    """

    params = CALIBRATION["visual"]

    mean = params["standardization"]["mean"]
    std = params["standardization"]["std"]

    intercept = params["logistic_intercept"]
    coefficient = params["logistic_coefficient"]

    # CDCN++ production score definition
    spoof_score = -float(raw_realness)

    z = standardize(spoof_score, mean, std)

    logit = intercept + coefficient * z

    probability = sigmoid(logit)

    return float(probability)


# ============================================================
# OPTIONAL DEBUG INFORMATION
# ============================================================

def get_audio_calibration_details(score_spoof: float) -> dict:
    """
    Return intermediate audio calibration values.
    Useful for testing and debugging.
    """

    params = CALIBRATION["audio"]

    mean = params["standardization"]["mean"]
    std = params["standardization"]["std"]

    intercept = params["logistic_intercept"]
    coefficient = params["logistic_coefficient"]

    z = standardize(score_spoof, mean, std)

    logit = intercept + coefficient * z

    probability = sigmoid(logit)

    return {
        "raw_score_spoof": float(score_spoof),
        "z_score": float(z),
        "logit": float(logit),
        "probability_spoof": float(probability)
    }


def get_visual_calibration_details(raw_realness: float) -> dict:
    """
    Return intermediate visual calibration values.
    Useful for testing and debugging.
    """

    params = CALIBRATION["visual"]

    raw_realness = float(raw_realness)

    # CDCN++ production score definition
    spoof_score = -raw_realness

    mean = params["standardization"]["mean"]
    std = params["standardization"]["std"]

    intercept = params["logistic_intercept"]
    coefficient = params["logistic_coefficient"]

    z = standardize(spoof_score, mean, std)

    logit = intercept + coefficient * z

    probability = sigmoid(logit)

    return {
        "raw_realness": raw_realness,
        "spoof_score": float(spoof_score),
        "z_score": float(z),
        "logit": float(logit),
        "probability_spoof": float(probability)
    }