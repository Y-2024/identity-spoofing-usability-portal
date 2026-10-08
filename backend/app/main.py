from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .audio_model import AASISTAudioModel
from .calibration import (
    calibrate_audio,
    get_audio_calibration_details,
)
from .fusion import fuse_probabilities
from .visual_model import VisualDetector


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Multimodal Identity Spoof Detection API",
    description=(
        "Audio and visual identity spoof detection using "
        "AASIST-L and CDCN++."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:3000",
        "http://localhost:3000"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CALIBRATION CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CALIBRATION_FILE = (
    BASE_DIR
    / "config"
    / "calibration.json"
)


with open(
    CALIBRATION_FILE,
    "r",
    encoding="utf-8-sig",
) as f:
    CALIBRATION_CONFIG = json.load(f)


# ------------------------------------------------------------
# Validated AASIST-L audio-only decision threshold
#
# This threshold was selected using the independent
# 1,000-sample development set and validated on a separate
# 1,000-sample hold-out set.
#
# Development:
#   Balanced accuracy = 0.8790
#
# Independent hold-out:
#   Balanced accuracy = 0.9010
#   ROC-AUC           = 0.952752
# ------------------------------------------------------------

AUDIO_DECISION_THRESHOLD = float(
    CALIBRATION_CONFIG["audio"]["decision_threshold"]
)


# ============================================================
# MODEL INITIALIZATION
# ============================================================

AUDIO_MODEL = None
VISUAL_MODEL = None


def get_audio_model():
    """
    Lazily initialize the AASIST-L audio model.
    """

    global AUDIO_MODEL

    if AUDIO_MODEL is None:
        AUDIO_MODEL = AASISTAudioModel(
            device="cpu"
        )

    return AUDIO_MODEL


def get_visual_model():
    """
    Lazily initialize the CDCN++ visual model.
    """

    global VISUAL_MODEL

    if VISUAL_MODEL is None:
        VISUAL_MODEL = VisualDetector()

    return VISUAL_MODEL


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Multimodal Identity Spoof Detection API",
        "version": "1.0.0",
        "models": {
            "audio": "AASIST-L",
            "visual": "CDCN++",
        },
        "decision_thresholds": {
            "audio_only": AUDIO_DECISION_THRESHOLD,
            "visual_only": 0.5,
            "fusion": float(
                CALIBRATION_CONFIG["fusion"]["threshold"]
            ),
        },
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "audio_model": "AASIST-L",
        "visual_model": "CDCN++",
        "device": "cpu",
        "decision_thresholds": {
            "audio_only": AUDIO_DECISION_THRESHOLD,
            "visual_only": 0.5,
            "fusion": float(
                CALIBRATION_CONFIG["fusion"]["threshold"]
            ),
        },
    }


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio_file(
    audio_path: Path,
) -> tuple[np.ndarray, int]:
    """
    Load an audio file using SoundFile, with an FFmpeg fallback.

    The primary path uses SoundFile directly. If SoundFile cannot
    decode the source file, FFmpeg is used to convert the file
    to a temporary 16 kHz mono WAV, which is then read using
    SoundFile.

    Returns:
        waveform: float32 NumPy array
        sample_rate: integer sample rate
    """

    waveform = None
    sample_rate = None

    # --------------------------------------------------------
    # Primary decoder: SoundFile
    # --------------------------------------------------------

    try:

        waveform, sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=False,
        )

    except Exception as soundfile_exc:

        # ----------------------------------------------------
        # Fallback decoder: FFmpeg -> temporary WAV
        # ----------------------------------------------------

        try:

            with tempfile.TemporaryDirectory() as temp_dir:

                temp_wav = (
                    Path(temp_dir)
                    / "converted_audio.wav"
                )

                result = subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-i",
                        str(audio_path),
                        "-f",
                        "wav",
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-y",
                        str(temp_wav),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )

                if (
                    result.returncode != 0
                    or not temp_wav.exists()
                ):

                    ffmpeg_error = (
                        result.stderr.strip()
                    )

                    raise ValueError(
                        "FFmpeg could not decode the "
                        "uploaded audio file."
                        + (
                            f" FFmpeg error: {ffmpeg_error}"
                            if ffmpeg_error
                            else ""
                        )
                    )

                waveform, sample_rate = sf.read(
                    str(temp_wav),
                    dtype="float32",
                    always_2d=False,
                )

        except Exception as ffmpeg_exc:

            raise ValueError(
                "Unable to read the uploaded audio file "
                "using either SoundFile or FFmpeg."
            ) from ffmpeg_exc

    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Convert stereo/multichannel audio to mono.
    # --------------------------------------------------------

    if waveform.ndim == 2:

        waveform = (
            waveform.mean(axis=1)
            .astype(np.float32)
        )

    return waveform, int(sample_rate)


# ============================================================
# TEMPORARY UPLOAD STORAGE
# ============================================================

def save_upload_to_temp(
    upload: UploadFile,
    suffix: str,
) -> Path:
    """
    Save an UploadFile to a temporary file.

    The caller is responsible for deleting the file.
    """

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    temp_path = Path(temp.name)

    try:

        while True:

            chunk = upload.file.read(
                1024 * 1024
            )

            if not chunk:
                break

            temp.write(chunk)

    finally:

        temp.close()

    return temp_path


# ============================================================
# AUDIO DETECTION
# ============================================================

def detect_audio(
    audio_path: Path,
) -> dict:
    """
    Run:

        audio file
            ->
        AASIST-L
            ->
        raw P(spoof)
            ->
        calibration
            ->
        calibrated P(spoof)

    The calibrated probability is used for the final
    audio-only decision using the validated AASIST-L
    decision threshold of 0.97.
    """

    waveform, sample_rate = load_audio_file(
        audio_path
    )

    # --------------------------------------------------------
    # AASIST-L expects 16 kHz audio.
    # --------------------------------------------------------

    if sample_rate != 16000:

        raise ValueError(
            "Audio sample rate must be 16,000 Hz "
            f"for this deployment. Received "
            f"{sample_rate} Hz."
        )

    # --------------------------------------------------------
    # AASIST-L inference
    # --------------------------------------------------------

    model = get_audio_model()

    raw_result = model.predict(
        waveform
    )

    raw_spoof = float(
        raw_result["score_spoof"]
    )

    # --------------------------------------------------------
    # Audio calibration
    # --------------------------------------------------------

    calibrated_spoof = calibrate_audio(
        raw_spoof
    )

    calibration_details = (
        get_audio_calibration_details(
            raw_spoof
        )
    )

    # --------------------------------------------------------
    # Audio-only decision
    #
    # IMPORTANT:
    # This is NOT the multimodal fusion threshold.
    #
    # AASIST-L validated threshold = 0.97
    # --------------------------------------------------------

    audio_prediction = (
        "SPOOF"
        if calibrated_spoof >= AUDIO_DECISION_THRESHOLD
        else "REAL"
    )

    return {
        "model": "AASIST-L",

        "sample_rate": sample_rate,

        "raw_probability_spoof": raw_spoof,

        "raw_probability_bonafide": float(
            raw_result["score_bonafide"]
        ),

        "calibrated_probability_spoof": float(
            calibrated_spoof
        ),

        "decision_threshold": (
            AUDIO_DECISION_THRESHOLD
        ),

        "prediction": audio_prediction,

        "calibration": {
            "z_score": calibration_details[
                "z_score"
            ],
            "logit": calibration_details[
                "logit"
            ],
        },

        "logits": {
            "spoof": float(
                raw_result["logit_spoof"]
            ),
            "bonafide": float(
                raw_result["logit_bonafide"]
            ),
        },
    }


# ============================================================
# VISUAL DETECTION
# ============================================================

def detect_visual(
    image_path: Path,
) -> dict:
    """
    Run:

        image
            ->
        CDCN++
            ->
        raw realness
            ->
        spoof score
            ->
        visual calibration
            ->
        calibrated P(spoof)
    """

    model = get_visual_model()

    details = model.predict_with_details(
        image_path
    )

    return {
        "model": "CDCN++",

        "raw_realness": float(
            details["raw_realness"]
        ),

        "spoof_score": float(
            details["spoof_score"]
        ),

        "calibrated_probability_spoof": float(
            details["spoof_probability"]
        ),

        "calibration": {
            "z_score": float(
                details["standardized_score"]
            ),
            "logit": float(
                details["logit"]
            ),
        },
    }


# ============================================================
# MULTIMODAL DETECTION ENDPOINT
# ============================================================

@app.post("/api/detect")
async def detect(
    audio: UploadFile | None = File(
        default=None
    ),
    image: UploadFile | None = File(
        default=None
    ),
):
    """
    Multimodal identity spoof detection endpoint.

    Supported combinations:

        Audio only
        Visual only
        Audio + visual

    ------------------------------------------------------------
    AUDIO ONLY
    ------------------------------------------------------------

    Uses the validated AASIST-L audio threshold:

        P_audio >= 0.97 -> SPOOF
        P_audio <  0.97 -> REAL

    ------------------------------------------------------------
    VISUAL ONLY
    ------------------------------------------------------------

    Uses the current visual threshold:

        P_visual >= 0.50 -> SPOOF
        P_visual <  0.50 -> REAL

    ------------------------------------------------------------
    AUDIO + VISUAL
    ------------------------------------------------------------

    Uses score-level late fusion:

        P_fusion =
            0.5 * P_audio
            +
            0.5 * P_visual

    The fusion threshold remains independently configured
    in calibration.json and is currently 0.50.
    """

    request_start = time.perf_counter()

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    if audio is None and image is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "At least one input is required: "
                "audio or image."
            ),
        )

    audio_temp = None
    image_temp = None

    audio_result = None
    visual_result = None

    try:

        # ====================================================
        # AUDIO
        # ====================================================

        if audio is not None:

            audio_name = (
                audio.filename
                or "uploaded_audio.wav"
            )

            audio_suffix = (
                Path(audio_name).suffix.lower()
                or ".wav"
            )

            audio_temp = save_upload_to_temp(
                audio,
                audio_suffix,
            )

            audio_result = detect_audio(
                audio_temp
            )

        # ====================================================
        # VISUAL
        # ====================================================

        if image is not None:

            image_name = (
                image.filename
                or "uploaded_image.jpg"
            )

            image_suffix = (
                Path(image_name).suffix.lower()
                or ".jpg"
            )

            image_temp = save_upload_to_temp(
                image,
                image_suffix,
            )

            visual_result = detect_visual(
                image_temp
            )

        # ====================================================
        # FINAL DECISION
        # ====================================================

        if (
            audio_result is not None
            and visual_result is not None
        ):

            # ------------------------------------------------
            # MULTIMODAL FUSION
            #
            # IMPORTANT:
            # The fusion threshold remains 0.50.
            # It is NOT replaced by the AASIST threshold.
            # ------------------------------------------------

            fusion_result = fuse_probabilities(
                audio_result[
                    "calibrated_probability_spoof"
                ],
                visual_result[
                    "calibrated_probability_spoof"
                ],
            )

            prediction = fusion_result[
                "prediction"
            ]

            confidence = max(
                fusion_result[
                    "fusion_probability_spoof"
                ],
                1.0
                - fusion_result[
                    "fusion_probability_spoof"
                ],
            )

            modality = "audio+visual"

        elif audio_result is not None:

            # ------------------------------------------------
            # AUDIO ONLY
            #
            # detect_audio() has already applied the
            # validated AASIST-L threshold of 0.97.
            # ------------------------------------------------

            probability = audio_result[
                "calibrated_probability_spoof"
            ]

            prediction = audio_result[
                "prediction"
            ]

            confidence = max(
                probability,
                1.0 - probability,
            )

            modality = "audio"

            fusion_result = None

        else:

            # ------------------------------------------------
            # VISUAL ONLY
            #
            # Current visual decision threshold = 0.50.
            # ------------------------------------------------

            probability = visual_result[
                "calibrated_probability_spoof"
            ]

            prediction = (
                "SPOOF"
                if probability >= 0.5
                else "REAL"
            )

            confidence = max(
                probability,
                1.0 - probability,
            )

            modality = "visual"

            fusion_result = None

        # ====================================================
        # LATENCY
        # ====================================================

        latency_ms = (
            time.perf_counter()
            - request_start
        ) * 1000.0

        # ====================================================
        # RESPONSE
        # ====================================================

        response = {
            "verdict": prediction,

            "confidence": float(
                confidence
            ),

            "modality": modality,

            "latency_ms": float(
                latency_ms
            ),

            "audio": audio_result,

            "visual": visual_result,

            "fusion": fusion_result,
        }

        return JSONResponse(
            content=response
        )

    except HTTPException:
        raise

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Identity spoof detection failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc

    finally:

        # ====================================================
        # DELETE TEMPORARY AUDIO FILE
        # ====================================================

        if (
            audio_temp is not None
            and audio_temp.exists()
        ):

            try:
                audio_temp.unlink()
            except Exception:
                pass

        # ====================================================
        # DELETE TEMPORARY IMAGE FILE
        # ====================================================

        if (
            image_temp is not None
            and image_temp.exists()
        ):

            try:
                image_temp.unlink()
            except Exception:
                pass