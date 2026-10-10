"use strict";

/*
 * ============================================================
 * MULTIMODAL IDENTITY SPOOFING DETECTION FRONTEND
 * ============================================================
 *
 * Local development backend:
 *
 *     http://127.0.0.1:8000
 *
 * Production URL will be changed later when the FastAPI
 * backend is deployed to Hugging Face Spaces.
 *
 * DO NOT put model weights or calibration secrets here.
 * This file only communicates with the API.
 * ============================================================
 */


const API_BASE_URL = "https://yinkamiva-identityspoofing-api.hf.space";

const DETECT_ENDPOINT =
    `${API_BASE_URL}/api/detect`;

const HEALTH_ENDPOINT =
    `${API_BASE_URL}/health`;


/* ============================================================
   STATE
============================================================ */

let currentMode = "audio";

let mediaRecorder = null;
let recordedAudioBlob = null;

let audioStream = null;

let webcamStream = null;
let capturedImageBlob = null;


/* ============================================================
   DOM ELEMENTS
============================================================ */

const audioModeButton =
    document.getElementById("audioModeButton");

const visualModeButton =
    document.getElementById("visualModeButton");

const multimodalModeButton =
    document.getElementById("multimodalModeButton");

const audioSection =
    document.getElementById("audioSection");

const visualSection =
    document.getElementById("visualSection");

const startRecordingButton =
    document.getElementById("startRecordingButton");

const stopRecordingButton =
    document.getElementById("stopRecordingButton");

const recordingStatus =
    document.getElementById("recordingStatus");

const audioPreview =
    document.getElementById("audioPreview");

const audioFile =
    document.getElementById("audioFile");

const audioFileName =
    document.getElementById("audioFileName");

const startCameraButton =
    document.getElementById("startCameraButton");

const captureImageButton =
    document.getElementById("captureImageButton");

const stopCameraButton =
    document.getElementById("stopCameraButton");

const webcamPreview =
    document.getElementById("webcamPreview");

const captureCanvas =
    document.getElementById("captureCanvas");

const imagePreview =
    document.getElementById("imagePreview");

const imageFile =
    document.getElementById("imageFile");

const imageFileName =
    document.getElementById("imageFileName");

const detectButton =
    document.getElementById("detectButton");

const loadingIndicator =
    document.getElementById("loadingIndicator");

const resultSection =
    document.getElementById("resultSection");

const verdictDisplay =
    document.getElementById("verdictDisplay");

const confidenceDisplay =
    document.getElementById("confidenceDisplay");

const confidenceBar =
    document.getElementById("confidenceBar");

const modalityDisplay =
    document.getElementById("modalityDisplay");

const latencyDisplay =
    document.getElementById("latencyDisplay");

const jsonResult =
    document.getElementById("jsonResult");

const resetButton =
    document.getElementById("resetButton");

const apiStatusDot =
    document.getElementById("apiStatusDot");

const apiStatusText =
    document.getElementById("apiStatusText");


/* ============================================================
   MODE SELECTION
============================================================ */

function setMode(mode) {

    currentMode = mode;

    audioModeButton.classList.toggle(
        "active",
        mode === "audio" || mode === "multimodal"
    );

    visualModeButton.classList.toggle(
        "active",
        mode === "visual" || mode === "multimodal"
    );

    multimodalModeButton.classList.toggle(
        "active",
        mode === "multimodal"
    );


    audioSection.classList.toggle(
        "hidden",
        mode === "visual"
    );

    visualSection.classList.toggle(
        "hidden",
        mode === "audio"
    );


    hideResult();
}


audioModeButton.addEventListener(
    "click",
    () => setMode("audio")
);

visualModeButton.addEventListener(
    "click",
    () => setMode("visual")
);

multimodalModeButton.addEventListener(
    "click",
    () => setMode("multimodal")
);


/* ============================================================
   AUDIO RECORDING
============================================================ */

async function startAudioRecording() {

    try {

        if (!navigator.mediaDevices ||
            !navigator.mediaDevices.getUserMedia) {

            throw new Error(
                "This browser does not support microphone access."
            );
        }


        audioStream =
            await navigator.mediaDevices.getUserMedia({
                audio: true
            });


        const options = {};

        if (
            MediaRecorder.isTypeSupported(
                "audio/webm;codecs=opus"
            )
        ) {
            options.mimeType =
                "audio/webm;codecs=opus";
        }


        mediaRecorder =
            new MediaRecorder(
                audioStream,
                options
            );


        const chunks = [];


        mediaRecorder.ondataavailable =
            event => {

                if (event.data.size > 0) {
                    chunks.push(event.data);
                }
            };


        mediaRecorder.onstop =
            () => {

                const mimeType =
                    mediaRecorder.mimeType ||
                    "audio/webm";

                recordedAudioBlob =
                    new Blob(
                        chunks,
                        { type: mimeType }
                    );


                audioPreview.src =
                    URL.createObjectURL(
                        recordedAudioBlob
                    );

                audioPreview.hidden = false;

                recordingStatus.textContent =
                    "Recording ready for detection.";

                startRecordingButton.disabled = false;
                stopRecordingButton.disabled = true;


                if (audioStream) {

                    audioStream
                        .getTracks()
                        .forEach(
                            track => track.stop()
                        );

                    audioStream = null;
                }
            };


        mediaRecorder.start();

        recordingStatus.textContent =
            "Recording in progress...";

        startRecordingButton.disabled = true;
        stopRecordingButton.disabled = false;


    } catch (error) {

        console.error(error);

        recordingStatus.textContent =
            `Microphone error: ${error.message}`;

        stopAudioStream();
    }
}


function stopAudioRecording() {

    if (
        mediaRecorder &&
        mediaRecorder.state !== "inactive"
    ) {

        mediaRecorder.stop();

    } else {

        stopAudioStream();
    }
}


function stopAudioStream() {

    if (audioStream) {

        audioStream
            .getTracks()
            .forEach(
                track => track.stop()
            );

        audioStream = null;
    }

    startRecordingButton.disabled = false;
    stopRecordingButton.disabled = true;
}


startRecordingButton.addEventListener(
    "click",
    startAudioRecording
);

stopRecordingButton.addEventListener(
    "click",
    stopAudioRecording
);


/* ============================================================
   AUDIO FILE
============================================================ */

audioFile.addEventListener(
    "change",
    () => {

        if (audioFile.files.length === 0) {

            audioFileName.textContent =
                "No file selected";

            return;
        }


        const file =
            audioFile.files[0];

        audioFileName.textContent =
            file.name;


        recordedAudioBlob = null;

        audioPreview.hidden = true;
    }
);


/* ============================================================
   WEBCAM
============================================================ */

async function startCamera() {

    try {

        if (!navigator.mediaDevices ||
            !navigator.mediaDevices.getUserMedia) {

            throw new Error(
                "This browser does not support webcam access."
            );
        }


        webcamStream =
            await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: "user",
                    width: {
                        ideal: 1280
                    },
                    height: {
                        ideal: 720
                    }
                },
                audio: false
            });


        webcamPreview.srcObject =
            webcamStream;


        startCameraButton.disabled = true;
        captureImageButton.disabled = false;
        stopCameraButton.disabled = false;


    } catch (error) {

        console.error(error);

        alert(
            `Camera error: ${error.message}`
        );
    }
}


function stopCamera() {

    if (webcamStream) {

        webcamStream
            .getTracks()
            .forEach(
                track => track.stop()
            );

        webcamStream = null;
    }


    webcamPreview.srcObject = null;

    startCameraButton.disabled = false;
    captureImageButton.disabled = true;
    stopCameraButton.disabled = true;
}


async function captureImage() {

    if (!webcamStream) {
        return;
    }


    const videoWidth =
        webcamPreview.videoWidth;

    const videoHeight =
        webcamPreview.videoHeight;


    if (!videoWidth || !videoHeight) {

        alert(
            "The camera image is not ready yet."
        );

        return;
    }


    captureCanvas.width =
        videoWidth;

    captureCanvas.height =
        videoHeight;


    const context =
        captureCanvas.getContext("2d");


    context.drawImage(
        webcamPreview,
        0,
        0,
        videoWidth,
        videoHeight
    );


    captureCanvas.toBlob(
        blob => {

            if (!blob) {

                alert(
                    "Unable to capture image."
                );

                return;
            }


            capturedImageBlob =
                blob;


            imagePreview.src =
                URL.createObjectURL(
                    blob
                );

            imagePreview.hidden = false;

        },
        "image/jpeg",
        0.92
    );
}


startCameraButton.addEventListener(
    "click",
    startCamera
);

captureImageButton.addEventListener(
    "click",
    captureImage
);

stopCameraButton.addEventListener(
    "click",
    stopCamera
);


/* ============================================================
   IMAGE FILE
============================================================ */

imageFile.addEventListener(
    "change",
    () => {

        if (imageFile.files.length === 0) {

            imageFileName.textContent =
                "No file selected";

            return;
        }


        const file =
            imageFile.files[0];

        imageFileName.textContent =
            file.name;


        capturedImageBlob = null;

        imagePreview.src =
            URL.createObjectURL(file);

        imagePreview.hidden = false;
    }
);


/* ============================================================
   BUILD FORM DATA
============================================================ */

function buildFormData() {

    const formData =
        new FormData();


    if (
        currentMode === "audio" ||
        currentMode === "multimodal"
    ) {

        let audioToSend = null;


        if (recordedAudioBlob) {

            audioToSend =
                recordedAudioBlob;

        } else if (
            audioFile.files.length > 0
        ) {

            audioToSend =
                audioFile.files[0];
        }


        if (!audioToSend) {

            throw new Error(
                "Please provide an audio recording or upload an audio file."
            );
        }


        formData.append(
            "audio",
            audioToSend,
            audioToSend.name ||
            "recorded_audio.webm"
        );
    }


    if (
        currentMode === "visual" ||
        currentMode === "multimodal"
    ) {

        let imageToSend = null;


        if (capturedImageBlob) {

            imageToSend =
                capturedImageBlob;

        } else if (
            imageFile.files.length > 0
        ) {

            imageToSend =
                imageFile.files[0];
        }


        if (!imageToSend) {

            throw new Error(
                "Please capture an image or upload an image file."
            );
        }


        formData.append(
            "image",
            imageToSend,
            imageToSend.name ||
            "captured_image.jpg"
        );
    }


    return formData;
}


/* ============================================================
   DETECTION
============================================================ */

async function runDetection() {

    let formData;


    try {

        formData =
            buildFormData();

    } catch (error) {

        showError(
            error.message
        );

        return;
    }


    setLoading(true);


    try {

        const response =
            await fetch(
                DETECT_ENDPOINT,
                {
                    method: "POST",
                    body: formData
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                data.message ||
                `API request failed (${response.status})`
            );
        }


        displayResult(data);


    } catch (error) {

        console.error(error);

        showError(
            `Detection failed: ${error.message}`
        );

    } finally {

        setLoading(false);
    }
}


detectButton.addEventListener(
    "click",
    runDetection
);


/* ============================================================
   DISPLAY RESULT
============================================================ */

function displayResult(data) {

    resultSection.classList.remove(
        "hidden"
    );


    const verdict =
        String(
            data.verdict ||
            "UNKNOWN"
        ).toUpperCase();


    const confidence =
        Number(
            data.confidence || 0
        );


    verdictDisplay.textContent =
        verdict;


    verdictDisplay.className =
        "verdict";


    if (verdict === "SPOOF") {

        verdictDisplay.classList.add(
            "spoof"
        );

    } else if (
        verdict === "REAL"
    ) {

        verdictDisplay.classList.add(
            "real"
        );

    } else {

        verdictDisplay.classList.add(
            "error"
        );
    }


    const confidencePercent =
        Math.max(
            0,
            Math.min(
                100,
                confidence * 100
            )
        );


    confidenceDisplay.textContent =
        `${confidencePercent.toFixed(2)}%`;


    confidenceBar.style.width =
        `${confidencePercent}%`;


    modalityDisplay.textContent =
        formatModality(
            data.modality
        );


    if (
        typeof data.latency_ms ===
        "number"
    ) {

        latencyDisplay.textContent =
            `${data.latency_ms.toFixed(2)} ms`;

    } else {

        latencyDisplay.textContent =
            "Not available";
    }


    jsonResult.textContent =
        JSON.stringify(
            data,
            null,
            2
        );


    resultSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


/* ============================================================
   ERROR
============================================================ */

function showError(message) {

    resultSection.classList.remove(
        "hidden"
    );


    verdictDisplay.textContent =
        "ERROR";

    verdictDisplay.className =
        "verdict error";


    confidenceDisplay.textContent =
        "â€”";

    confidenceBar.style.width =
        "0%";


    modalityDisplay.textContent =
        "â€”";

    latencyDisplay.textContent =
        "â€”";


    jsonResult.textContent =
        message;


    resultSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


/* ============================================================
   LOADING
============================================================ */

function setLoading(isLoading) {

    detectButton.disabled =
        isLoading;


    loadingIndicator.classList.toggle(
        "hidden",
        !isLoading
    );


    if (isLoading) {

        detectButton.textContent =
            "â³ Processing...";

    } else {

        detectButton.textContent =
            "ðŸ” Run Spoof Detection";
    }
}


/* ============================================================
   RESULT RESET
============================================================ */

function hideResult() {

    resultSection.classList.add(
        "hidden"
    );
}


function resetTest() {

    stopAudioStream();

    if (
        mediaRecorder &&
        mediaRecorder.state !== "inactive"
    ) {
        mediaRecorder.stop();
    }


    stopCamera();


    recordedAudioBlob = null;
    capturedImageBlob = null;


    audioFile.value = "";
    imageFile.value = "";


    audioFileName.textContent =
        "No file selected";

    imageFileName.textContent =
        "No file selected";


    audioPreview.src = "";
    audioPreview.hidden = true;


    imagePreview.src = "";
    imagePreview.hidden = true;


    recordingStatus.textContent =
        "Ready to record";


    hideResult();

    setMode("audio");
}


resetButton.addEventListener(
    "click",
    resetTest
);


/* ============================================================
   API HEALTH CHECK
============================================================ */

async function checkAPIHealth() {

    try {

        const response =
            await fetch(
                HEALTH_ENDPOINT,
                {
                    method: "GET"
                }
            );


        if (!response.ok) {
            throw new Error(
                `HTTP ${response.status}`
            );
        }


        const data =
            await response.json();


        apiStatusDot.classList.remove(
            "offline"
        );

        apiStatusDot.classList.add(
            "online"
        );


        apiStatusText.textContent =
            `API Online â€” ${data.status || "healthy"}`;


    } catch (error) {

        console.error(
            "API health check failed:",
            error
        );


        apiStatusDot.classList.remove(
            "online"
        );

        apiStatusDot.classList.add(
            "offline"
        );


        apiStatusText.textContent =
            "API Offline";
    }
}


/* ============================================================
   UTILITY
============================================================ */

function formatModality(
    modality
) {

    if (!modality) {
        return "Not available";
    }


    switch (modality) {

        case "audio":
            return "Audio";

        case "visual":
            return "Visual";

        case "audio+visual":
            return "Audio + Visual";

        default:
            return modality;
    }
}


/* ============================================================
   INITIALIZATION
============================================================ */

setMode("audio");

checkAPIHealth();

setInterval(
    checkAPIHealth,
    30000
);
