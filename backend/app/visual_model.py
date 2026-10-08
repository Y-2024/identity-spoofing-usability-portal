from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Union

import numpy as np


PathLike = Union[str, Path]


class VisualDetector:
    """
    Production CDCN++ visual spoof detector.

    Supports:
        1. Normal image files: JPG, JPEG, PNG, BMP, etc.
        2. Preprocessed RGB NumPy arrays (.npy) used by the SiW
           validation/evaluation workflow.

    Model:
        CDCN++ using the official CDCN implementation.

    Checkpoint:
        CDCN_ROSE_5000_epoch1_final.pth

    Calibration:
        Independent Platt-scaling calibration stored in
        backend/config/calibration.json.

    Deployment score:
        spoof_score = -raw_realness
    """

    name = "CDCN++"

    def __init__(self):
        import torch
        from PIL import Image

        self.torch = torch
        self.Image = Image

        # ============================================================
        # DEVICE
        # ============================================================

        self.device = torch.device(
            os.getenv("MODEL_DEVICE", "cpu")
        )

        # ============================================================
        # PROJECT PATHS
        # ============================================================

        # visual_model.py
        #     backend/
        #         app/
        #             visual_model.py
        #
        # parents[0] = app
        # parents[1] = backend
        # parents[2] = project root

        project_root = Path(
            os.getenv(
                "PROJECT_ROOT",
                str(Path(__file__).resolve().parents[2])
            )
        ).resolve()

        self.project_root = project_root

        # Official CDCN++ implementation
        self.cdcn_root = Path(
            os.getenv(
                "CDCN_ROOT",
                str(project_root / "backend" / "cdcn")
            )
        ).resolve()

        # Validated CDCN++ checkpoint
        self.checkpoint_path = Path(
            os.getenv(
                "CDCN_CHECKPOINT",
                str(
                    project_root
                    / "backend"
                    / "models"
                    / "CDCN_ROSE_5000_epoch1_final.pth"
                )
            )
        ).resolve()

        # Project calibration
        self.calibration_path = Path(
            os.getenv(
                "CDCN_CALIBRATION",
                str(
                    project_root
                    / "backend"
                    / "config"
                    / "calibration.json"
                )
            )
        ).resolve()

        # ============================================================
        # VERIFY REQUIRED FILES
        # ============================================================

        cdcn_source = self.cdcn_root / "CDCNs.py"

        if not cdcn_source.exists():
            raise FileNotFoundError(
                "CDCN++ implementation not found:\n"
                f"{cdcn_source}"
            )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "CDCN++ checkpoint not found:\n"
                f"{self.checkpoint_path}"
            )

        if not self.calibration_path.exists():
            raise FileNotFoundError(
                "CDCN++ calibration file not found:\n"
                f"{self.calibration_path}"
            )

        # ============================================================
        # IMPORT OFFICIAL CDCN++ IMPLEMENTATION
        # ============================================================

        cdcn_root_string = str(self.cdcn_root)

        if cdcn_root_string not in sys.path:
            sys.path.insert(0, cdcn_root_string)

        try:
            from CDCNs import CDCNpp, Conv2d_cd
        except Exception as exc:
            raise ImportError(
                "Unable to import the official CDCN++ implementation "
                f"from:\n{self.cdcn_root}"
            ) from exc

        self.CDCNpp = CDCNpp
        self.Conv2d_cd = Conv2d_cd

        # ============================================================
        # INITIALIZE CDCN++
        # ============================================================

        self.model = CDCNpp(
            basic_conv=Conv2d_cd,
            theta=0.7
        ).to(self.device)

        # ============================================================
        # LOAD VALIDATED CHECKPOINT
        # ============================================================

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                "Unexpected CDCN++ checkpoint format. "
                "Expected a dictionary."
            )

        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint
        )

        if not isinstance(state_dict, dict):
            raise RuntimeError(
                "CDCN++ checkpoint does not contain a valid "
                "'model_state_dict'."
            )

        # Remove DataParallel prefix if present.
        clean_state_dict = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }

        # Strict loading is intentional.
        missing_keys, unexpected_keys = (
            self.model.load_state_dict(
                clean_state_dict,
                strict=True
            )
        )

        if missing_keys:
            raise RuntimeError(
                "CDCN++ checkpoint has missing keys:\n"
                + "\n".join(missing_keys)
            )

        if unexpected_keys:
            raise RuntimeError(
                "CDCN++ checkpoint has unexpected keys:\n"
                + "\n".join(unexpected_keys)
            )

        self.model.eval()

        # ============================================================
        # LOAD CALIBRATION
        # ============================================================

        with self.calibration_path.open(
            "r",
            encoding="utf-8-sig"
        ) as handle:
            self.calibration = json.load(handle)

        calibration_method = self.calibration.get(
            "calibration_method"
        )

        if calibration_method != (
            "Platt_scaling_logistic_regression"
        ):
            raise RuntimeError(
                "Unexpected visual calibration method: "
                f"{calibration_method!r}"
            )

        visual_calibration = self.calibration.get("visual")

        if not isinstance(visual_calibration, dict):
            raise RuntimeError(
                "The calibration file does not contain "
                "a valid 'visual' configuration."
            )

        score_definition = visual_calibration.get(
            "score_definition"
        )

        expected_score_definition = (
            "spoof_score = -raw_realness"
        )

        if score_definition != expected_score_definition:
            raise RuntimeError(
                "Unexpected CDCN++ score definition.\n"
                f"Expected: {expected_score_definition}\n"
                f"Found:    {score_definition}"
            )

        standardization = visual_calibration.get(
            "standardization"
        )

        if not isinstance(standardization, dict):
            raise RuntimeError(
                "Visual calibration is missing "
                "'standardization'."
            )

        self.score_mean = float(
            standardization["mean"]
        )

        self.score_std = float(
            standardization["std"]
        )

        if not np.isfinite(self.score_mean):
            raise RuntimeError(
                "Visual calibration mean is not finite."
            )

        if not np.isfinite(self.score_std):
            raise RuntimeError(
                "Visual calibration standard deviation "
                "is not finite."
            )

        if self.score_std <= 0:
            raise RuntimeError(
                "Visual calibration standard deviation "
                "must be greater than zero."
            )

        self.logistic_intercept = float(
            visual_calibration[
                "logistic_intercept"
            ]
        )

        self.logistic_coefficient = float(
            visual_calibration[
                "logistic_coefficient"
            ]
        )

        if not np.isfinite(
            self.logistic_intercept
        ):
            raise RuntimeError(
                "Visual logistic intercept is not finite."
            )

        if not np.isfinite(
            self.logistic_coefficient
        ):
            raise RuntimeError(
                "Visual logistic coefficient is not finite."
            )

    # ================================================================
    # RGB PREPARATION
    # ================================================================

    def _prepare_rgb_from_array(
        self,
        array: np.ndarray,
        source: PathLike,
    ):
        """
        Prepare an already-existing RGB NumPy array.

        Expected validation format:
            H x W x 3
            uint8 or numeric RGB data

        The validated CDCN++ preprocessing is:

            RGB
              -> float32
              -> /255
              -> (x - 0.5) / 0.5
              -> HWC -> CHW
              -> batch dimension
        """

        arr = np.asarray(array)

        if arr.ndim != 3:
            raise ValueError(
                "CDCN++ RGB array must have exactly 3 dimensions "
                f"(H, W, C). Got shape {arr.shape} from {source}"
            )

        if arr.shape[2] != 3:
            raise ValueError(
                "CDCN++ RGB array must have 3 channels. "
                f"Got shape {arr.shape} from {source}"
            )

        if arr.shape[0] != 256 or arr.shape[1] != 256:
            raise ValueError(
                "CDCN++ RGB array must be 256x256. "
                f"Got shape {arr.shape} from {source}"
            )

        if not np.isfinite(
            arr.astype(np.float32, copy=False)
        ).all():
            raise ValueError(
                f"CDCN++ RGB array contains non-finite values: "
                f"{source}"
            )

        # Convert to float32.
        arr = arr.astype(
            np.float32,
            copy=False
        )

        # The local SiW preprocessing artifacts are RGB images
        # stored as uint8 [0, 255].
        #
        # We also tolerate floating-point arrays that are already
        # in [0, 255].
        arr_min = float(arr.min())
        arr_max = float(arr.max())

        if arr_max > 1.0:
            arr = arr / 255.0

        elif arr_min < 0.0:
            raise ValueError(
                "Unsupported RGB array range. "
                f"Minimum value is {arr_min}, maximum is {arr_max}."
            )

        # Normalize exactly as used by the validated CDCN++
        # preprocessing workflow.
        arr = (arr - 0.5) / 0.5

        # HWC -> CHW
        arr = np.transpose(
            arr,
            (2, 0, 1)
        )

        # Make contiguous before converting to torch.
        arr = np.ascontiguousarray(
            arr,
            dtype=np.float32
        )

        tensor = self.torch.from_numpy(arr)

        # Add batch dimension.
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def _prepare_rgb_from_image(
        self,
        image_path: PathLike,
    ):
        """
        Prepare a normal image file for production inference.

        Processing:

            image
              -> RGB
              -> 256x256
              -> float32
              -> /255
              -> (x - 0.5) / 0.5
              -> HWC -> CHW
              -> batch dimension
        """

        image_path = Path(image_path)

        try:
            image = (
                self.Image.open(image_path)
                .convert("RGB")
                .resize((256, 256))
            )
        except Exception as exc:
            raise ValueError(
                "Unable to open visual input image:\n"
                f"{image_path}"
            ) from exc

        arr = np.asarray(
            image,
            dtype=np.float32
        )

        if arr.shape != (256, 256, 3):
            raise ValueError(
                "Unexpected RGB image shape after preprocessing: "
                f"{arr.shape}"
            )

        arr = arr / 255.0

        arr = (arr - 0.5) / 0.5

        # HWC -> CHW
        arr = np.transpose(
            arr,
            (2, 0, 1)
        )

        arr = np.ascontiguousarray(
            arr,
            dtype=np.float32
        )

        tensor = self.torch.from_numpy(arr)

        # Add batch dimension.
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def _prepare_rgb(
        self,
        image_path: PathLike,
    ):
        """
        Prepare either:

            .npy preprocessed RGB array

        or:

            normal image file.

        This allows the same production detector to be used for:

            - local SiW validation
            - production image uploads
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Visual input does not exist:\n{image_path}"
            )

        suffix = image_path.suffix.lower()

        if suffix == ".npy":
            try:
                array = np.load(
                    image_path,
                    allow_pickle=False
                )
            except Exception as exc:
                raise ValueError(
                    "Unable to load RGB NumPy array:\n"
                    f"{image_path}"
                ) from exc

            return self._prepare_rgb_from_array(
                array,
                image_path
            )

        return self._prepare_rgb_from_image(
            image_path
        )

    # ================================================================
    # PREDICTION
    # ================================================================

    def predict(
        self,
        image_path: PathLike,
    ) -> float:
        """
        Run CDCN++ inference and return calibrated
        probability of spoof.

        Returns:
            float:
                P(spoof), in [0, 1]
        """

        rgb = self._prepare_rgb(
            image_path
        )

        with self.torch.no_grad():
            output = self.model(rgb)

        # Official CDCN++ returns a tuple containing
        # several outputs. The primary predicted depth map
        # is output[0].
        if not isinstance(
            output,
            (tuple, list)
        ):
            raise RuntimeError(
                "Unexpected CDCN++ output type: "
                f"{type(output).__name__}"
            )

        if len(output) == 0:
            raise RuntimeError(
                "CDCN++ returned an empty output."
            )

        primary_map = output[0]

        if primary_map.ndim != 3:
            raise RuntimeError(
                "Unexpected CDCN++ primary output shape: "
                f"{tuple(primary_map.shape)}"
            )

        if primary_map.shape[0] != 1:
            raise RuntimeError(
                "CDCN++ prediction currently expects "
                "a batch size of 1. Got: "
                f"{primary_map.shape[0]}"
            )

        if not self.torch.isfinite(
            primary_map
        ).all():
            raise RuntimeError(
                "CDCN++ primary output contains "
                "non-finite values."
            )

        # ============================================================
        # RAW CDCN++ SCORE
        # ============================================================

        raw_realness = float(
            primary_map.mean().item()
        )

        if not np.isfinite(raw_realness):
            raise RuntimeError(
                "CDCN++ raw realness score is not finite."
            )

        # Deployment calibration was defined as:
        #
        #     spoof_score = -raw_realness
        #
        spoof_score = -raw_realness

        # ============================================================
        # STANDARDIZATION
        # ============================================================

        standardized_score = (
            spoof_score - self.score_mean
        ) / self.score_std

        if not np.isfinite(
            standardized_score
        ):
            raise RuntimeError(
                "Standardized CDCN++ score is not finite."
            )

        # ============================================================
        # PLATT LOGISTIC CALIBRATION
        # ============================================================

        logit = (
            self.logistic_intercept
            + self.logistic_coefficient
            * standardized_score
        )

        if not np.isfinite(logit):
            raise RuntimeError(
                "CDCN++ calibration logit is not finite."
            )

        # Numerically stable sigmoid.
        logit = float(
            np.clip(
                logit,
                -50.0,
                50.0
            )
        )

        spoof_probability = (
            1.0
            / (
                1.0
                + np.exp(-logit)
            )
        )

        spoof_probability = float(
            np.clip(
                spoof_probability,
                0.0,
                1.0
            )
        )

        return spoof_probability

    # ================================================================
    # OPTIONAL DIAGNOSTIC PREDICTION
    # ================================================================

    def predict_with_details(
        self,
        image_path: PathLike,
    ) -> dict:
        """
        Run inference and return the calibrated spoof
        probability together with diagnostic values.

        This is useful for validation/debugging without
        changing the production predict() interface.
        """

        rgb = self._prepare_rgb(
            image_path
        )

        with self.torch.no_grad():
            output = self.model(rgb)

        if not isinstance(
            output,
            (tuple, list)
        ):
            raise RuntimeError(
                "Unexpected CDCN++ output type: "
                f"{type(output).__name__}"
            )

        if len(output) == 0:
            raise RuntimeError(
                "CDCN++ returned an empty output."
            )

        primary_map = output[0]

        if primary_map.ndim != 3:
            raise RuntimeError(
                "Unexpected CDCN++ primary output shape: "
                f"{tuple(primary_map.shape)}"
            )

        raw_realness = float(
            primary_map.mean().item()
        )

        spoof_score = -raw_realness

        standardized_score = (
            spoof_score - self.score_mean
        ) / self.score_std

        logit = (
            self.logistic_intercept
            + self.logistic_coefficient
            * standardized_score
        )

        clipped_logit = float(
            np.clip(
                logit,
                -50.0,
                50.0
            )
        )

        spoof_probability = float(
            1.0
            / (
                1.0
                + np.exp(-clipped_logit)
            )
        )

        spoof_probability = float(
            np.clip(
                spoof_probability,
                0.0,
                1.0
            )
        )

        return {
            "model": self.name,
            "device": str(self.device),
            "input": str(image_path),
            "input_suffix": Path(image_path).suffix.lower(),
            "raw_realness": raw_realness,
            "spoof_score": spoof_score,
            "standardized_score": float(
                standardized_score
            ),
            "logit": clipped_logit,
            "spoof_probability": spoof_probability,
            "prediction": (
                "SPOOF"
                if spoof_probability >= 0.5
                else "REAL"
            ),
        }
