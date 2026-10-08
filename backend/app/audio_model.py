from pathlib import Path
import sys

import numpy as np
import torch


class AASISTAudioModel:
    """
    Verified AASIST-L inference adapter.

    Verified against:
        AASIST-L_20k_T4_final.pth

    Verified model configuration:
        nb_samp = 64600
        first_conv = 128
        filts = [70, [1, 32], [32, 32], [32, 24], [24, 24]]
        gat_dims = [24, 32]
        pool_ratios = [0.4, 0.5, 0.7, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]

    Verified class mapping:
        class 0 = spoof
        class 1 = bonafide

    Calibration is intentionally performed separately
    by calibration.py.

    Path handling:
        Local Windows development:
            <project_root>/backend/aasist
            <project_root>/backend/models/AASIST-L_20k_T4_final.pth

        Docker deployment:
            /app/aasist
            /app/models/AASIST-L_20k_T4_final.pth
    """

    MODEL_NAME = "AASIST-L"
    SAMPLE_RATE = 16000
    NB_SAMP = 64600

    def __init__(
        self,
        model_root=None,
        checkpoint_path=None,
        device="cpu",
    ):
        self.device = torch.device(device)

        # ------------------------------------------------------------
        # Determine the project root.
        #
        # audio_model.py is located at:
        #   <project_root>/backend/app/audio_model.py
        #
        # Therefore parents[2] is:
        #   <project_root>
        # ------------------------------------------------------------
        project_root = Path(__file__).resolve().parents[2]

        # ------------------------------------------------------------
        # Resolve AASIST implementation path.
        #
        # Local Windows:
        #   <project_root>/backend/aasist
        #
        # Docker:
        #   /app/aasist
        # ------------------------------------------------------------
        if model_root is None:
            local_model_root = (
                project_root
                / "backend"
                / "aasist"
            )

            docker_model_root = Path(
                "/app/aasist"
            )

            if (
                local_model_root
                / "models"
                / "AASIST.py"
            ).exists():
                self.model_root = local_model_root
            else:
                self.model_root = docker_model_root

        else:
            self.model_root = Path(model_root)

        # ------------------------------------------------------------
        # Resolve AASIST-L checkpoint path.
        #
        # Local Windows:
        #   <project_root>/backend/models/
        #       AASIST-L_20k_T4_final.pth
        #
        # Docker:
        #   /app/models/AASIST-L_20k_T4_final.pth
        # ------------------------------------------------------------
        if checkpoint_path is None:
            local_checkpoint = (
                project_root
                / "backend"
                / "models"
                / "AASIST-L_20k_T4_final.pth"
            )

            docker_checkpoint = Path(
                "/app/models/AASIST-L_20k_T4_final.pth"
            )

            if local_checkpoint.exists():
                self.checkpoint_path = local_checkpoint
            else:
                self.checkpoint_path = docker_checkpoint

        else:
            self.checkpoint_path = Path(
                checkpoint_path
            )

        # ------------------------------------------------------------
        # Verify the AASIST implementation exists.
        # ------------------------------------------------------------
        model_file = (
            self.model_root
            / "models"
            / "AASIST.py"
        )

        if not model_file.exists():
            raise RuntimeError(
                "AASIST implementation not found: "
                f"{model_file}"
            )

        # ------------------------------------------------------------
        # Verify the checkpoint exists.
        # ------------------------------------------------------------
        if not self.checkpoint_path.exists():
            raise RuntimeError(
                "AASIST checkpoint not found: "
                f"{self.checkpoint_path}"
            )

        # ------------------------------------------------------------
        # Make the AASIST repository importable.
        # ------------------------------------------------------------
        root_string = str(
            self.model_root
        )

        if root_string not in sys.path:
            sys.path.insert(
                0,
                root_string
            )

        from models.AASIST import Model

        # ------------------------------------------------------------
        # Exact AASIST-L model configuration.
        # ------------------------------------------------------------
        model_config = {
            "architecture": "AASIST",
            "nb_samp": 64600,
            "first_conv": 128,
            "filts": [
                70,
                [1, 32],
                [32, 32],
                [32, 24],
                [24, 24],
            ],
            "gat_dims": [
                24,
                32,
            ],
            "pool_ratios": [
                0.4,
                0.5,
                0.7,
                0.5,
            ],
            "temperatures": [
                2.0,
                2.0,
                100.0,
                100.0,
            ],
        }

        # ------------------------------------------------------------
        # Create the model.
        # ------------------------------------------------------------
        self.model = Model(
            model_config
        ).to(self.device)

        # ------------------------------------------------------------
        # Load the validated checkpoint.
        # ------------------------------------------------------------
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if not isinstance(
            checkpoint,
            dict
        ):
            raise RuntimeError(
                "AASIST checkpoint must be a dictionary."
            )

        if (
            "model_state_dict"
            not in checkpoint
        ):
            raise RuntimeError(
                "AASIST checkpoint does not contain "
                "'model_state_dict'."
            )

        # IMPORTANT:
        # strict=True was already verified successfully
        # against the actual project checkpoint.
        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        # Evaluation mode.
        self.model.eval()

    # ------------------------------------------------------------
    # Prepare waveform for AASIST-L.
    # ------------------------------------------------------------
    @staticmethod
    def prepare_waveform(
        waveform
    ):
        """
        Convert input to mono float32 and prepare exactly
        64,600 samples.

        If the input is longer than 64,600 samples,
        it is truncated.

        If the input is shorter than 64,600 samples,
        it is repeated until it reaches 64,600 samples.

        This preserves the waveform preparation procedure
        verified against the calibration waveform.
        """

        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        )

        # Remove unnecessary dimensions.
        if waveform.ndim > 1:
            waveform = waveform.squeeze()

        # Require a one-dimensional waveform.
        if waveform.ndim != 1:
            raise ValueError(
                "Expected mono waveform, "
                f"received shape {waveform.shape}"
            )

        # Reject empty audio.
        if waveform.size == 0:
            raise ValueError(
                "Audio waveform is empty."
            )

        # --------------------------------------------------------
        # AASIST-L expects exactly 64,600 samples.
        # --------------------------------------------------------
        if (
            waveform.size
            >= AASISTAudioModel.NB_SAMP
        ):
            waveform = waveform[
                :AASISTAudioModel.NB_SAMP
            ]

        else:
            repeats = (
                AASISTAudioModel.NB_SAMP
                // waveform.size
            ) + 1

            waveform = np.tile(
                waveform,
                repeats,
            )

            waveform = waveform[
                :AASISTAudioModel.NB_SAMP
            ]

        # Add batch dimension:
        # (64600,) -> (1, 64600)
        return torch.from_numpy(
            waveform
        ).unsqueeze(0)

    # ------------------------------------------------------------
    # Run AASIST-L inference.
    # ------------------------------------------------------------
    @torch.inference_mode()
    def predict(
        self,
        waveform
    ):
        """
        Run AASIST-L inference.

        Returns the raw model probabilities before
        project-level calibration.

        Class mapping:
            probability[0] = spoof
            probability[1] = bonafide
        """

        # Prepare waveform.
        x = self.prepare_waveform(
            waveform
        )

        # Move waveform to selected device.
        x = x.to(
            self.device
        )

        # --------------------------------------------------------
        # Official AASIST forward pass.
        # --------------------------------------------------------
        _, logits = self.model(
            x,
            Freq_aug=False,
        )

        # Convert logits to class probabilities.
        probabilities = torch.softmax(
            logits.float(),
            dim=1,
        )[0]

        # Class 0 = spoof.
        score_spoof = float(
            probabilities[0].item()
        )

        # Class 1 = bonafide.
        score_bonafide = float(
            probabilities[1].item()
        )

        return {
            "score_spoof": score_spoof,
            "score_bonafide": score_bonafide,
            "logit_spoof": float(
                logits[0, 0].item()
            ),
            "logit_bonafide": float(
                logits[0, 1].item()
            ),
        }