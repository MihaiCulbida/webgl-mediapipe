import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const FINGER_TIP = { thumb: 4, index: 8 };
const PORTAL_COLOR = "#00ff66";

const PINCH_RATIO_THRESHOLD = 0.40;
const CHARGE_DURATION_MS = 1600;
const RESET_HOLD_MS = 2000;

const FINGER_JOINTS = {
    index:  { tip: 8,  pip: 6  },
    middle: { tip: 12, pip: 10 },
    ring:   { tip: 16, pip: 14 },
    pinky:  { tip: 20, pip: 18 }
};

const video = document.getElementById('camera');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

let handLandmarker = null;

let mode = 'normal';

let chargeActive = false;
let chargeStartTime = 0;
let chargeProgress = 0;
let chargeReady = false;

let resetHoldStart = 0;

async function init() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        },
        runningMode: "VIDEO",
        numHands: 2
    });

    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    await new Promise(resolve => { video.onloadedmetadata = resolve; });
    video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    requestAnimationFrame(loop);
}

function loop() {
    const now = performance.now();
    const results = handLandmarker.detectForVideo(video, now);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    handleReset(results, now);
    renderMode(results, now);

    requestAnimationFrame(loop);
}

function renderMode(results, now) {
    if (mode === 'fullscreen') {
        drawFullscreenBackground();
    }

    const landmarks = results.landmarks || [];

    if (landmarks.length === 2) {
        const [handA, handB] = landmarks;
        const pinchA = isPinch(handA);
        const pinchB = isPinch(handB);

        const quad = buildQuad(results);
        if (quad) {
            if (mode === 'fullscreen') drawInvertedPortal(quad);
            else drawPortal(quad);
            drawFingerMarkers(quad);
        }

        if (pinchA !== pinchB) {

            if (!chargeActive) {
                chargeActive = true;
                chargeStartTime = now;
            }
            chargeProgress = clamp((now - chargeStartTime) / CHARGE_DURATION_MS, 0, 1);
            if (chargeProgress >= 1) chargeReady = true;

            const pinchHand = pinchA ? handA : handB;
            drawChargeBar(pinchHand, chargeProgress);

        } else if (!pinchA && !pinchB) {

            if (chargeReady) {

                mode = (mode === 'fullscreen') ? 'normal' : 'fullscreen';
                resetCharge();
            } else {
                resetCharge();
            }
        } else {

            resetCharge();
        }
    } else {
        resetCharge();
    }
}

function drawFullscreenBackground() {
    ctx.filter = "grayscale(1) contrast(1.15)";
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.filter = "none";
}

function handleReset(results, now) {
    const landmarks = results.landmarks || [];

    if (landmarks.length === 1 && isHandOpen(landmarks[0])) {
        if (resetHoldStart === 0) resetHoldStart = now;

        if (now - resetHoldStart >= RESET_HOLD_MS) {
            mode = 'normal';
            resetCharge();
            resetHoldStart = 0;
        }
    } else {
        resetHoldStart = 0;
    }
}

function resetCharge() {
    chargeActive = false;
    chargeProgress = 0;
    chargeReady = false;
}

function dist(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
}

function clamp(v, min, max) {
    return Math.min(max, Math.max(min, v));
}

function isPinch(landmarks) {
    const scale = dist(landmarks[0], landmarks[9]);
    if (scale === 0) return false;
    const pinchDist = dist(landmarks[FINGER_TIP.thumb], landmarks[FINGER_TIP.index]);
    return (pinchDist / scale) < PINCH_RATIO_THRESHOLD;
}

function isHandOpen(landmarks) {
    const wrist = landmarks[0];
    let extended = 0;
    for (const key in FINGER_JOINTS) {
        const { tip, pip } = FINGER_JOINTS[key];
        if (dist(wrist, landmarks[tip]) > dist(wrist, landmarks[pip]) * 1.1) {
            extended++;
        }
    }
    return extended >= 3;
}

function buildQuad(results) {
    const landmarks = results.landmarks;
    if (!landmarks || landmarks.length < 2) return null;

    const hands = { Left: null, Right: null };
    landmarks.forEach((lm, i) => {
        const label = results.handednesses?.[i]?.[0]?.categoryName || (i === 0 ? "Left" : "Right");
        hands[label] = lm;
    });
    if (!hands.Left || !hands.Right) return null;

    const toPoint = (lm, idx) => ({
        x: lm[idx].x * canvas.width,
        y: lm[idx].y * canvas.height
    });

    return [
        toPoint(hands.Left, FINGER_TIP.thumb),
        toPoint(hands.Left, FINGER_TIP.index),
        toPoint(hands.Right, FINGER_TIP.index),
        toPoint(hands.Right, FINGER_TIP.thumb)
    ];
}

function drawPortal(points) {
    ctx.save();
    tracePolygon(points);

    ctx.save();
    ctx.clip();
    ctx.filter = "grayscale(1) contrast(1.15)";
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.filter = "none";
    ctx.restore();

    ctx.lineWidth = 3;
    ctx.strokeStyle = PORTAL_COLOR;
    ctx.stroke();
    ctx.restore();
}

function drawInvertedPortal(points) {
    ctx.save();
    tracePolygon(points);

    ctx.save();
    ctx.clip();
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.restore();

    ctx.lineWidth = 3;
    ctx.strokeStyle = PORTAL_COLOR;
    ctx.stroke();
    ctx.restore();
}

function tracePolygon(points) {
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.closePath();
}

function drawFingerMarkers(points) {
    points.forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
        ctx.fillStyle = "#ffffff";
        ctx.fill();
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = PORTAL_COLOR;
        ctx.stroke();
    });
}

function drawChargeBar(pinchHand, progress) {
    const p = midpoint(
        toCanvasPoint(pinchHand[FINGER_TIP.thumb]),
        toCanvasPoint(pinchHand[FINGER_TIP.index])
    );

    const width = 90, height = 14;
    const x = p.x - width / 2;
    const y = p.y - 55 - height / 2;

    ctx.fillStyle = "rgba(0,255,102,0.15)";
    ctx.fillRect(x, y, width, height);

    const filled = width * progress;
    ctx.fillStyle = PORTAL_COLOR;
    ctx.fillRect(x + width - filled, y, filled, height);

    ctx.lineWidth = 2;
    ctx.strokeStyle = PORTAL_COLOR;
    ctx.strokeRect(x, y, width, height);

    ctx.beginPath();
    ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = PORTAL_COLOR;
    ctx.stroke();
}

function toCanvasPoint(landmark) {
    return { x: landmark.x * canvas.width, y: landmark.y * canvas.height };
}

function midpoint(a, b) {
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

init();