import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const video = document.getElementById('camera');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

let handLandmarker;

const SPHERE_RINGS = 10;
const SPHERE_SEGMENTS = 16;

function generateSphere() {
  const vertices = [];
  const edges = [];

  for (let i = 0; i <= SPHERE_RINGS; i++) {
    const theta = (i / SPHERE_RINGS) * Math.PI;
    for (let j = 0; j < SPHERE_SEGMENTS; j++) {
      const phi = (j / SPHERE_SEGMENTS) * Math.PI * 2;
      const x = Math.sin(theta) * Math.cos(phi);
      const y = Math.cos(theta);
      const z = Math.sin(theta) * Math.sin(phi);
      vertices.push([x, y, z]);
    }
  }

  for (let i = 0; i <= SPHERE_RINGS; i++) {
    for (let j = 0; j < SPHERE_SEGMENTS; j++) {
      const current = i * SPHERE_SEGMENTS + j;
      const next = i * SPHERE_SEGMENTS + ((j + 1) % SPHERE_SEGMENTS);
      edges.push([current, next]);
      if (i < SPHERE_RINGS) {
        const below = (i + 1) * SPHERE_SEGMENTS + j;
        edges.push([current, below]);
      }
    }
  }

  return { vertices, edges };
}

const SPHERE = generateSphere();

function generateTesseract() {
  const outerScale = 1;
  const innerScale = 0.25;

  const baseVertices = [
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
  ];

  const cubeEdges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
  ];

  const vertices = [
    ...baseVertices.map(v => v.map(c => c * outerScale)),
    ...baseVertices.map(v => v.map(c => c * innerScale))
  ];

  const edges = [
    ...cubeEdges,
    ...cubeEdges.map(([a, b]) => [a + 8, b + 8]),
    ...baseVertices.map((_, i) => [i, i + 8])
  ];

  return { vertices, edges };
}

const TESSERACT = generateTesseract();

const SHAPES = {
  sphere: { geometry: SPHERE, color: '#00ffff', shadow: 'rgba(0, 255, 255, 0.8)' },
  tesseract: { geometry: TESSERACT, color: '#00ffff', shadow: 'rgba(0, 255, 255, 0.8)' }
};

// Ciclul de forme la palma deschisă: sferă -> cub -> ascuns -> sferă -> ...
const OBJECT_CYCLE = ['sphere', 'tesseract', null];

const GESTURE_HOLD_MS = 1000;
const GRAB_RADIUS_MULT = 2.2;
const ROTATION_SMOOTHING = 0.25;

let cycleIndex = -1;

const objectState = {
  active: false,
  type: 'sphere',
  x: 0,
  y: 0,
  size: 90,
  grabbedBy: null,
  grabOffsetX: 0,
  grabOffsetY: 0,
  rotationY: 0.6
};

const gestureState = {};

async function initModels() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 2
  });
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;
  return new Promise(resolve => {
    video.onloadedmetadata = () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      resolve();
    };
  });
}

function distNorm(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function isPinching(landmarks) {
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  const dx = (thumbTip.x - indexTip.x) * canvas.width;
  const dy = (thumbTip.y - indexTip.y) * canvas.height;
  const dist = Math.hypot(dx, dy);
  return dist < 35;
}

function isHandOpen(landmarks) {
  const wrist = landmarks[0];
  const handSize = distNorm(wrist, landmarks[9]) || 0.001;

  const indexExtended = distNorm(landmarks[8], wrist) > distNorm(landmarks[5], wrist) + handSize * 0.15;
  const middleExtended = distNorm(landmarks[12], wrist) > distNorm(landmarks[9], wrist) + handSize * 0.15;
  const ringExtended = distNorm(landmarks[16], wrist) > distNorm(landmarks[13], wrist) + handSize * 0.15;
  const pinkyExtended = distNorm(landmarks[20], wrist) > distNorm(landmarks[17], wrist) + handSize * 0.15;

  const extendedCount = [indexExtended, middleExtended, ringExtended, pinkyExtended].filter(Boolean).length;
  return extendedCount >= 3;
}

function computeHandRotation(landmarks) {
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  const indexMcp = landmarks[5];
  const pinkyMcp = landmarks[17];

  const acrossX = pinkyMcp.x - indexMcp.x;
  const acrossZ = (pinkyMcp.z || 0) - (indexMcp.z || 0);
  const rotationY = Math.atan2(acrossZ, acrossX);

  const upY = middleMcp.y - wrist.y;
  const upZ = (middleMcp.z || 0) - (wrist.z || 0);
  const rotationX = Math.atan2(upZ, upY);

  return { rotationY: -rotationY };
}

function getHandLabel(handednesses, i) {
  return handednesses[i]?.[0]?.categoryName || `Hand${i}`;
}

function getGestureState(label) {
  if (!gestureState[label]) {
    gestureState[label] = { palmOpen: false, palmOpenStart: 0, triggered: false, pinch: false };
  }
  return gestureState[label];
}

function rotatePoint([x, y, z], ry) {
  const cosY = Math.cos(ry);
  const sinY = Math.sin(ry);
  const x1 = x * cosY - z * sinY;
  const z1 = x * sinY + z * cosY;

  return [x1, y, z1];
}

function project(vertex, cx, cy, size, ry) {
  const [x, y, z] = rotatePoint(vertex, ry);
  const perspective = 4;
  const scale = perspective / (perspective + z);
  return {
    x: cx + x * size * scale,
    y: cy + y * size * scale
  };
}

function drawObject(cx, cy, size, ry, grabbed, type) {
  const shape = SHAPES[type] || SHAPES.sphere;
  const projected = shape.geometry.vertices.map(v => project(v, cx, cy, size, ry));

  const color = grabbed ? '#ffcc00' : shape.color;
  ctx.strokeStyle = color;
  ctx.shadowColor = grabbed ? 'rgba(255, 204, 0, 0.8)' : shape.shadow;
  ctx.lineWidth = 1.2;
  ctx.shadowBlur = 8;

  for (const [a, b] of shape.geometry.edges) {
    ctx.beginPath();
    ctx.moveTo(projected[a].x, projected[a].y);
    ctx.lineTo(projected[b].x, projected[b].y);
    ctx.stroke();
  }

  ctx.shadowBlur = 0;
}

function advanceObjectCycle(landmarks) {
  cycleIndex = (cycleIndex + 1) % OBJECT_CYCLE.length;
  const next = OBJECT_CYCLE[cycleIndex];

  if (next === null) {
    objectState.active = false;
    objectState.grabbedBy = null;
    return;
  }

  if (!objectState.active) {
    objectState.x = ((landmarks[8].x + landmarks[12].x) / 2) * canvas.width;
    objectState.y = ((landmarks[8].y + landmarks[12].y) / 2) * canvas.height;
    objectState.size = 90;
  }

  objectState.type = next;
  objectState.active = true;
}

function updateHandGesture(landmarks, label, timestamp) {
  const state = getGestureState(label);

  const palmOpen = isHandOpen(landmarks);
  if (palmOpen) {
    if (!state.palmOpen) {
      state.palmOpen = true;
      state.palmOpenStart = timestamp;
      state.triggered = false;
    } else if (!state.triggered && timestamp - state.palmOpenStart >= GESTURE_HOLD_MS) {
      advanceObjectCycle(landmarks);
      state.triggered = true;
    }
  } else {
    state.palmOpen = false;
    state.triggered = false;
  }

  const pinching = isPinching(landmarks);
  if (objectState.active) {
    const cx = landmarks[9].x * canvas.width;
    const cy = landmarks[9].y * canvas.height;

    if (pinching && !state.pinch && objectState.grabbedBy === null) {
      const d = Math.hypot(cx - objectState.x, cy - objectState.y);
      if (d < objectState.size * GRAB_RADIUS_MULT) {
        objectState.grabbedBy = label;
        objectState.grabOffsetX = objectState.x - cx;
        objectState.grabOffsetY = objectState.y - cy;
      }
    }

    if (pinching && objectState.grabbedBy === label) {
      objectState.x = cx + objectState.grabOffsetX;
      objectState.y = cy + objectState.grabOffsetY;

      const targetRotation = computeHandRotation(landmarks);
      objectState.rotationY += (targetRotation.rotationY - objectState.rotationY) * ROTATION_SMOOTHING;
    }

    if (!pinching && objectState.grabbedBy === label) {
      objectState.grabbedBy = null;
    }
  }

  state.pinch = pinching;
}

function detectLoop() {
  const timestamp = performance.now();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const drawingUtils = new DrawingUtils(ctx);
  const handResult = handLandmarker.detectForVideo(video, timestamp);
  const hands = handResult.landmarks || [];
  const handednesses = handResult.handednesses || [];

  for (let i = 0; i < hands.length; i++) {
    const landmarks = hands[i];
    drawingUtils.drawConnectors(
      landmarks,
      HandLandmarker.HAND_CONNECTIONS,
      { color: "#00FF00", lineWidth: 3 }
    );
    drawingUtils.drawLandmarks(landmarks, { color: "#FF0000", radius: 4 });

    const label = getHandLabel(handednesses, i);
    updateHandGesture(landmarks, label, timestamp);
  }

  if (objectState.active) {
    drawObject(objectState.x, objectState.y, objectState.size, objectState.rotationY, objectState.grabbedBy !== null, objectState.type);
  }

  requestAnimationFrame(detectLoop);
}

async function main() {
  try {
    await startCamera();
    await initModels();
    detectLoop();
  } catch (err) {
    alert('Eroare: ' + err.message);
  }
}

main();