import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const video = document.getElementById('camera');
const ink = document.getElementById('ink');
const cursorC = document.getElementById('cursor');
const inkCtx = ink.getContext('2d');
const cursorCtx = cursorC.getContext('2d');
const clearBtn = document.getElementById('clearBtn');

const CONTROL_HAND = 'Left';
const DRAW_HAND = 'Right';

const LINE_WIDTH = 2.5;
const ERASE_RADIUS = 5;
const DRAW_CURSOR_RADIUS = 2;
const POS_SMOOTHING = 0.5;
const EXTEND_RATIO = 1.15;
const PINCH_RATIO = 0.15;
const V_HOLD_MS = 2000;
const CONFIRM_COLOR = '#4a9eff';

let handLandmarker = null;
let drawingEnabled = false;
let mode = 'draw';
let vHoldStart = null;
let vTriggered = false;
let vConfirm = null;
let lastPoint = null;
let smoothPt = null;

clearBtn.addEventListener('click', () => {
    inkCtx.clearRect(0, 0, ink.width, ink.height);
});

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

    const w = video.videoWidth, h = video.videoHeight;
    ink.width = w; ink.height = h;
    cursorC.width = w; cursorC.height = h;

    inkCtx.lineCap = 'round';
    inkCtx.lineJoin = 'round';
    inkCtx.strokeStyle = '#ffffff';
    inkCtx.lineWidth = LINE_WIDTH;

    requestAnimationFrame(loop);
}

function loop() {
    const now = performance.now();
    const results = handLandmarker.detectForVideo(video, now);

    cursorCtx.clearRect(0, 0, cursorC.width, cursorC.height);

    let controlLm = null;
    let drawLm = null;

    if (results.landmarks && results.handedness) {
        for (let i = 0; i < results.landmarks.length; i++) {
            const label = results.handedness[i][0].categoryName;
            if (label === CONTROL_HAND) controlLm = results.landmarks[i];
            if (label === DRAW_HAND) drawLm = results.landmarks[i];
        }
    }

    drawingEnabled = controlLm ? isPinching(controlLm) : false;

    if (drawLm) {
        handleDrawHand(drawLm, now);
    } else {
        lastPoint = null;
        smoothPt = null;
        vHoldStart = null;
        vTriggered = false;
        vConfirm = null;
    }

    drawVConfirm();

    requestAnimationFrame(loop);
}

function dist(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
}

function clamp(v, min, max) {
    return Math.min(max, Math.max(min, v));
}

function midpoint(a, b) {
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function handScale(lm) {
    return dist(lm[0], lm[9]);
}

function fingerExtended(lm, tipIdx, pipIdx, wristIdx = 0) {
    const dTip = dist(lm[tipIdx], lm[wristIdx]);
    const dPip = dist(lm[pipIdx], lm[wristIdx]);
    return dTip > dPip * EXTEND_RATIO;
}

function isPinching(lm) {
    const scale = handScale(lm);
    return dist(lm[4], lm[8]) < scale * PINCH_RATIO;
}

function isVShape(lm) {
    const index = fingerExtended(lm, 8, 6);
    const middle = fingerExtended(lm, 12, 10);
    const ring = fingerExtended(lm, 16, 14);
    const pinky = fingerExtended(lm, 20, 18);
    return index && middle && !ring && !pinky;
}

function toCanvasPoint(lm, idx) {
    return { x: lm[idx].x * ink.width, y: lm[idx].y * ink.height };
}

function handleDrawHand(lm, now) {
    const vState = isVShape(lm);
    if (vState) {
        if (vHoldStart === null) vHoldStart = now;
        const progress = clamp((now - vHoldStart) / V_HOLD_MS, 0, 1);
        const point = midpoint(toCanvasPoint(lm, 8), toCanvasPoint(lm, 12));
        vConfirm = { point, progress };
        if (!vTriggered && progress >= 1) {
            mode = mode === 'draw' ? 'erase' : 'draw';
            vTriggered = true;
        }
    } else {
        vHoldStart = null;
        vTriggered = false;
        vConfirm = null;
    }

    const rawPoint = toCanvasPoint(lm, 8);
    if (!smoothPt) smoothPt = rawPoint;
    smoothPt = {
        x: smoothPt.x + (rawPoint.x - smoothPt.x) * (1 - POS_SMOOTHING),
        y: smoothPt.y + (rawPoint.y - smoothPt.y) * (1 - POS_SMOOTHING)
    };
    const point = smoothPt;

    if (!drawingEnabled) {
        lastPoint = null;
        drawCursor(point, mode === 'erase' ? '#ff5555' : '#ffffff', mode === 'erase' ? ERASE_RADIUS : DRAW_CURSOR_RADIUS, mode === 'erase');
        return;
    }

    if (mode === 'draw') {
        inkCtx.beginPath();
        if (lastPoint) {
            inkCtx.moveTo(lastPoint.x, lastPoint.y);
            inkCtx.lineTo(point.x, point.y);
            inkCtx.stroke();
        } else {
            inkCtx.arc(point.x, point.y, LINE_WIDTH / 2, 0, Math.PI * 2);
            inkCtx.fillStyle = '#ffffff';
            inkCtx.fill();
        }
        lastPoint = point;
        drawCursor(point, '#ffffff', DRAW_CURSOR_RADIUS, false);
    } else {
        inkCtx.save();
        inkCtx.globalCompositeOperation = 'destination-out';
        inkCtx.beginPath();
        inkCtx.arc(point.x, point.y, ERASE_RADIUS, 0, Math.PI * 2);
        inkCtx.fill();
        inkCtx.restore();
        lastPoint = null;
        drawCursor(point, '#ff5555', ERASE_RADIUS, true);
    }
}

function drawCursor(point, color, radius, ringOnly) {
    cursorCtx.beginPath();
    cursorCtx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    if (ringOnly) {
        cursorCtx.strokeStyle = color;
        cursorCtx.lineWidth = 1.5;
        cursorCtx.stroke();
    } else {
        cursorCtx.fillStyle = color;
        cursorCtx.fill();
    }
}

function drawVConfirm() {
    if (!vConfirm) return;

    const { point, progress } = vConfirm;
    const width = 90, height = 14;
    const x = point.x - width / 2;
    const y = point.y - 55 - height / 2;

    cursorCtx.fillStyle = 'rgba(74,217,255,0.15)';
    cursorCtx.fillRect(x, y, width, height);

    const filled = width * progress;
    cursorCtx.fillStyle = CONFIRM_COLOR;
    cursorCtx.fillRect(x + width - filled, y, filled, height);

    cursorCtx.lineWidth = 2;
    cursorCtx.strokeStyle = CONFIRM_COLOR;
    cursorCtx.strokeRect(x, y, width, height);
}

async function main() {
    try {
        await init();
    } catch (err) {
        alert('Eroare: ' + err.message);
    }
}

main();