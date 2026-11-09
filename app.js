// Mianxiang — Traditional Chinese face reading, computed from facial landmarks.
// All processing happens locally in the browser via MediaPipe Face Landmarker.

import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ---------------------------------------------------------------------------
// Landmark indices (MediaPipe Face Mesh, 478 points)
// ---------------------------------------------------------------------------
const L = {
  foreheadTop: 10,      // top of forehead mesh (approximate hairline)
  glabella: 9,          // between the eyebrows (yintang 印堂)
  nasion: 168,          // root of the nose
  noseTip: 4,
  subnasale: 2,         // base of nose / top of philtrum
  upperLipTop: 0,
  upperLipBottom: 13,
  lowerLipTop: 14,
  lowerLipBottom: 17,
  chinTip: 152,
  mouthL: 61, mouthR: 291,
  alarL: 129, alarR: 358,          // nose wings
  faceL: 234, faceR: 454,          // widest face points (near temples/ears)
  jawL: 172, jawR: 397,            // jaw width
  eyeLOuter: 33, eyeLInner: 133,
  eyeRInner: 362, eyeROuter: 263,
  eyeLTop: 159, eyeLBottom: 145,
  eyeRTop: 386, eyeRBottom: 374,
  browLUpper: [70, 63, 105, 66, 107],
  browLLower: [46, 53, 52, 65, 55],
  browRUpper: [300, 293, 334, 296, 336],
  browRLower: [276, 283, 282, 295, 285],
  browLInner: 107, browLOuter: 70,
  browRInner: 336, browROuter: 300,
};

// Population statistics calibrated from real phone photographs (which carry
// different lens geometry than paintings or studio portraits). Readings are
// z-scores against these, i.e. "how does this face differ from a typical photo".
const STATS = {
  aspect:          { mean: 1.32,  sd: 0.07 },
  courtUpper:      { mean: 0.223, sd: 0.034 },
  courtMiddle:     { mean: 0.388, sd: 0.016 },
  courtLower:      { mean: 0.389, sd: 0.045 },
  eyeSpacing:      { mean: 1.30,  sd: 0.12 },
  eyeSize:         { mean: 0.208, sd: 0.009 },
  eyeOpenness:     { mean: 0.35,  sd: 0.06 },
  browThickness:   { mean: 0.032, sd: 0.002 },
  browAngle:       { mean: -9.5,  sd: 7.0 },
  browEyeGap:      { mean: 0.082, sd: 0.008 },
  noseLength:      { mean: 0.284, sd: 0.022 },
  noseWidth:       { mean: 0.328, sd: 0.02 },
  mouthWidth:      { mean: 0.44,  sd: 0.04 },
  lipFullness:     { mean: 0.096, sd: 0.014 },
  mouthCornerLift: { mean: 0.07,  sd: 0.03 },
  philtrum:        { mean: 0.084, sd: 0.01 },
  jawRatio:        { mean: 0.80,  sd: 0.032 },
  foreheadJaw:     { mean: 1.06,  sd: 0.065 },
  chinHeight:      { mean: 0.167, sd: 0.019 },
  mouthTiltAsym:   { mean: 0,     sd: 0.045 },
  eyeOpenAsym:     { mean: 0,     sd: 0.10 },
  browGapAsym:     { mean: 0,     sd: 0.14 },
};

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const uploadSection = $("upload-section");
const loadingSection = $("loading-section");
const loadingText = $("loading-text");
const errorSection = $("error-section");
const errorText = $("error-text");
const resultsSection = $("results-section");
const dropZone = $("drop-zone");
const fileInput = $("file-input");
const canvas = $("face-canvas");
const ctx = canvas.getContext("2d");

let landmarker = null;
let lastRender = null; // { img, pts, metrics, reading }

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------
async function getLandmarker() {
  if (landmarker) return landmarker;
  loadingText.textContent = "Loading the face model… (first time takes a few seconds)";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    },
    runningMode: "IMAGE",
    numFaces: 1,
  });
  return landmarker;
}

// ---------------------------------------------------------------------------
// Upload wiring
// ---------------------------------------------------------------------------
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") fileInput.click();
});
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  const file = e.dataTransfer.files?.[0];
  if (file) handleFile(file);
});
fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) handleFile(file);
  fileInput.value = "";
});
$("new-photo").addEventListener("click", resetToUpload);
$("error-retry").addEventListener("click", resetToUpload);
$("toggle-courts").addEventListener("change", redrawOverlay);
$("toggle-points").addEventListener("change", redrawOverlay);
$("share-card").addEventListener("click", shareCard);
$("copy-text").addEventListener("click", copyReadingText);

function show(section) {
  for (const s of [uploadSection, loadingSection, errorSection, resultsSection]) {
    s.hidden = s !== section;
  }
}

function resetToUpload() {
  show(uploadSection);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function fail(msg) {
  errorText.textContent = msg;
  show(errorSection);
}

async function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    fail("That file doesn't look like an image. Please choose a photo.");
    return;
  }
  show(loadingSection);
  try {
    const img = await loadImage(file);
    loadingText.textContent = "Reading the face…";
    const lm = await getLandmarker();
    const result = lm.detect(img);
    if (!result.faceLandmarks?.length) {
      fail("No face found in that photo. Try a clearer, front-facing picture with good lighting.");
      return;
    }
    const pts = result.faceLandmarks[0].map((p) => ({
      x: p.x * img.naturalWidth,
      y: p.y * img.naturalHeight,
    }));
    const metrics = computeMetrics(pts);
    window.__metrics = metrics; // debug/calibration hook
    const reading = interpret(metrics);
    lastRender = { img, pts, metrics, reading };
    renderCanvas();
    renderReading(reading);
    show(resultsSection);
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (err) {
    console.error(err);
    fail("Something went wrong while analyzing the photo. Check your connection (the face model loads from the web) and try again.");
  }
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("bad image")); };
    img.src = url;
  });
}

// ---------------------------------------------------------------------------
// Geometry → metrics
// ---------------------------------------------------------------------------
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

// The face mesh ends mid-forehead, below the true hairline; extrapolate upward.
const HAIRLINE_EXTEND = 0.45;
const hairlineY = (p) => p[L.foreheadTop].y - HAIRLINE_EXTEND * (p[L.glabella].y - p[L.foreheadTop].y);

// Rotate points so the eye line is horizontal — head tilt must not read as asymmetry.
function derotate(p) {
  const a = p[L.eyeLOuter], b = p[L.eyeROuter];
  const roll = Math.atan2(b.y - a.y, b.x - a.x);
  const cx = (a.x + b.x) / 2, cy = (a.y + b.y) / 2;
  const cos = Math.cos(-roll), sin = Math.sin(-roll);
  return p.map((pt) => ({
    x: cx + (pt.x - cx) * cos - (pt.y - cy) * sin,
    y: cy + (pt.x - cx) * sin + (pt.y - cy) * cos,
  }));
}

function computeMetrics(raw) {
  const p = derotate(raw);
  const faceWidth = dist(p[L.faceL], p[L.faceR]);
  const faceHeight = p[L.chinTip].y - hairlineY(p);

  // Three courts (三停) — vertical thirds of the face.
  const upper = p[L.glabella].y - hairlineY(p);
  const middle = p[L.subnasale].y - p[L.glabella].y;
  const lower = p[L.chinTip].y - p[L.subnasale].y;
  const courtTotal = upper + middle + lower;

  // Eyes
  const eyeLWidth = dist(p[L.eyeLOuter], p[L.eyeLInner]);
  const eyeRWidth = dist(p[L.eyeRInner], p[L.eyeROuter]);
  const eyeWidth = (eyeLWidth + eyeRWidth) / 2;
  const interocular = dist(p[L.eyeLInner], p[L.eyeRInner]);
  const eyeLOpen = dist(p[L.eyeLTop], p[L.eyeLBottom]);
  const eyeROpen = dist(p[L.eyeRTop], p[L.eyeRBottom]);
  const eyeOpenness = (eyeLOpen + eyeROpen) / 2 / eyeWidth;

  // Brows
  const browThickness =
    (avgPairDist(p, L.browLUpper, L.browLLower) + avgPairDist(p, L.browRUpper, L.browRLower)) / 2;
  const browAngleL = Math.atan2(p[L.browLInner].y - p[L.browLOuter].y, Math.abs(p[L.browLOuter].x - p[L.browLInner].x));
  const browAngleR = Math.atan2(p[L.browRInner].y - p[L.browROuter].y, Math.abs(p[L.browROuter].x - p[L.browRInner].x));
  const browAngle = ((browAngleL + browAngleR) / 2) * (180 / Math.PI);
  const browGapL = verticalGap(p, L.browLLower, [L.eyeLTop]);
  const browGapR = verticalGap(p, L.browRLower, [L.eyeRTop]);
  const browEyeGap = (browGapL + browGapR) / 2;

  // Nose
  const noseLength = dist(p[L.nasion], p[L.subnasale]);
  const noseWidth = dist(p[L.alarL], p[L.alarR]);

  // Mouth & lips
  const mouthWidth = dist(p[L.mouthL], p[L.mouthR]);
  const upperLip = dist(p[L.upperLipTop], p[L.upperLipBottom]);
  const lowerLip = dist(p[L.lowerLipTop], p[L.lowerLipBottom]);
  const lipFullness = (upperLip + lowerLip) / faceHeight;
  const lipCenterY = (p[L.upperLipBottom].y + p[L.lowerLipTop].y) / 2;
  const mouthCornerLift = (lipCenterY - (p[L.mouthL].y + p[L.mouthR].y) / 2) / mouthWidth;

  // Symmetry (on derotated points; signed values keep which side is which)
  const mouthTiltAsym = (p[L.mouthL].y - p[L.mouthR].y) / mouthWidth;
  const eyeOpenAsym = (eyeLOpen - eyeROpen) / ((eyeLOpen + eyeROpen) / 2);
  const browGapAsym = (browGapL - browGapR) / ((browGapL + browGapR) / 2);

  // Philtrum (人中)
  const philtrum = dist(p[L.subnasale], p[L.upperLipTop]);

  // Chin & jaw
  const jawWidth = dist(p[L.jawL], p[L.jawR]);
  const chinHeight = dist(p[L.lowerLipBottom], p[L.chinTip]);
  const foreheadWidth = dist(p[70], p[300]); // outer brow ends ~ forehead width

  return {
    p: raw,
    faceWidth, faceHeight,
    aspect: faceHeight / faceWidth,
    courts: {
      upper: upper / courtTotal,
      middle: middle / courtTotal,
      lower: lower / courtTotal,
    },
    eyeWidth,
    eyeSpacing: interocular / eyeWidth,
    eyeSize: eyeWidth / faceWidth,
    eyeOpenness,
    browThickness: browThickness / faceHeight,
    browAngle,
    browEyeGap: browEyeGap / faceHeight,
    noseLength: noseLength / faceHeight,
    noseWidth: noseWidth / faceWidth,
    mouthWidth: mouthWidth / faceWidth,
    lipFullness,
    mouthCornerLift,
    mouthTiltAsym,
    eyeOpenAsym,
    browGapAsym,
    philtrum: philtrum / faceHeight,
    jawRatio: jawWidth / faceWidth,
    foreheadJaw: foreheadWidth / jawWidth,
    chinHeight: chinHeight / faceHeight,
  };
}

function avgPairDist(p, idxA, idxB) {
  let sum = 0;
  for (let i = 0; i < idxA.length; i++) sum += dist(p[idxA[i]], p[idxB[i]]);
  return sum / idxA.length;
}

function verticalGap(p, browIdxList, eyeIdxList) {
  const browY = browIdxList.reduce((s, i) => s + p[i].y, 0) / browIdxList.length;
  const eyeY = eyeIdxList.reduce((s, i) => s + p[i].y, 0) / eyeIdxList.length;
  return eyeY - browY;
}

// ---------------------------------------------------------------------------
// Scoring helpers
// ---------------------------------------------------------------------------
const z = (value, key) => (value - STATS[key].mean) / STATS[key].sd;

// 3-way band from a z-score: 0 = low, 1 = typical, 2 = high
const band3 = (zv, lo = -0.5, hi = 0.5) => (zv < lo ? 0 : zv > hi ? 2 : 1);

// Deterministic PRNG seeded from the face's own geometry: the same photo reads
// the same every time, but different faces draw different phrasings.
function makeRng(m) {
  let h = 2166136261;
  const s = [m.aspect, m.eyeSpacing, m.noseWidth, m.mouthWidth, m.jawRatio, m.browEyeGap, m.lipFullness]
    .map((v) => Math.round(v * 1000)).join(",");
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return () => {
    h = Math.imul(h ^ (h >>> 15), 2246822519);
    h = Math.imul(h ^ (h >>> 13), 3266489917);
    return ((h ^= h >>> 16) >>> 0) / 4294967296;
  };
}

const pickVariant = (rng, arr) => arr[Math.floor(rng() * arr.length) % arr.length];

// Courts normalized against population means so a typical face reads ~33/33/33.
function calibratedCourts(m) {
  const u = m.courts.upper / STATS.courtUpper.mean;
  const mi = m.courts.middle / STATS.courtMiddle.mean;
  const lo = m.courts.lower / STATS.courtLower.mean;
  const t = u + mi + lo;
  return { upper: u / t, middle: mi / t, lower: lo / t };
}

// ---------------------------------------------------------------------------
// Interpretation
// ---------------------------------------------------------------------------
function interpret(m) {
  const rng = makeRng(m);
  const element = classifyElement(m);
  const courts = interpretCourts(m, rng);
  const features = interpretFeatures(m, rng);
  const palaces = interpretPalaces(m, rng);
  const symmetry = interpretSymmetry(m, rng);
  const traditions = interpretTraditions(m, rng);
  const summary = buildSummary(element, courts, m, rng);
  return { element, courts, features, palaces, symmetry, traditions, summary };
}

// ----- Five-element face shape (五行面相) -----
const ELEMENTS = {
  wood: {
    hanzi: "木", name: "Wood", pinyin: "mù", glyph: "木",
    sub: "The long face, grown upward like a tree",
    text: "A Wood face runs long and keeps its lines. The old manuals give Wood people the plans: you grow in one direction, slowly, and you don't like being pruned. People trust you with the long jobs. Your season is spring and your virtue is benevolence, 仁. One warning comes with it: a tree that never bends collects wind damage.",
  },
  fire: {
    hanzi: "火", name: "Fire", pinyin: "huǒ", glyph: "火",
    sub: "Broad at the brow, tapering to the chin, like a flame",
    text: "A Fire face is widest at the top and comes to a point, the way a flame does. The manuals say Fire people light rooms and empty them, depending on the day. Ideas come fast, words faster. Your season is summer and your virtue is propriety, 礼. The warning: a fire that never banks its coals eats its own fuel.",
  },
  earth: {
    hanzi: "土", name: "Earth", pinyin: "tǔ", glyph: "土",
    sub: "Square and substantial, built close to the ground",
    text: "An Earth face is broad, square, and planted. The manuals make Earth people the keepers: of families, of businesses, of other people's secrets. You carry weight without mentioning it. Your season is late summer and your virtue is trustworthiness, 信. The warning is simple: soil feeds everyone and forgets to feed itself.",
  },
  metal: {
    hanzi: "金", name: "Metal", pinyin: "jīn", glyph: "金",
    sub: "Clean angles and fine structure, like a cast bell",
    text: "A Metal face has edges. The manuals read good definition as refined qi: high standards, a straight answer, a promise that doesn't rust. You'd rather be correct than comfortable. Your season is autumn and your virtue is righteousness, 义. The warning: the best blades flex. Pure hardness is how things crack.",
  },
  water: {
    hanzi: "水", name: "Water", pinyin: "shuǐ", glyph: "水",
    sub: "Round and fluid, with soft, full curves",
    text: "A Water face is round where others are ruled. The manuals call Water people the ones you cannot corner: they read the room, find the gap, and arrive anyway. Money is said to pool in such faces the way rivers pool downstream. Your season is winter and your virtue is wisdom, 智. The warning: water with no riverbed is just weather.",
  },
};

const ELEMENT_BLEND = {
  wood: "Wood, with a current of",
  fire: "Fire, banked with",
  earth: "Earth, shot through with",
  metal: "Metal, tempered by",
  water: "Water, deepened by",
};

function classifyElement(m) {
  const za = z(m.aspect, "aspect");
  const zj = z(m.jawRatio, "jawRatio");
  const zfj = z(m.foreheadJaw, "foreheadJaw");
  const znw = z(m.noseWidth, "noseWidth");
  const zlip = z(m.lipFullness, "lipFullness");
  const zch = z(m.chinHeight, "chinHeight");
  const scores = {
    wood:  0.8 * za - 0.3 * zj + 0.2 * zch,
    fire:  0.8 * zfj - 0.5 * zj - 0.3 * zch,
    earth: 0.7 * zj + 0.4 * znw - 0.4 * za,
    metal: 0.25 + 0.5 * zj - 0.5 * Math.abs(za) - 0.4 * Math.abs(zfj),
    water: -0.6 * za + 0.45 * zlip - 0.25 * zj,
  };
  const sorted = Object.entries(scores).sort((x, y) => y[1] - x[1]);
  const primary = ELEMENTS[sorted[0][0]];
  const secondary = sorted[0][1] - sorted[1][1] < 0.25 ? ELEMENTS[sorted[1][0]] : null;
  return { ...primary, key: sorted[0][0], secondary, blend: ELEMENT_BLEND[sorted[0][0]] };
}

// ----- Three courts (三停) -----
function interpretCourts(m, rng) {
  const cal = calibratedCourts(m);
  const courtDefs = [
    { key: "upper", name: "Upper Court", hanzi: "上停", span: "Youth (15–30)" },
    { key: "middle", name: "Middle Court", hanzi: "中停", span: "Middle years (31–50)" },
    { key: "lower", name: "Lower Court", hanzi: "下停", span: "Later years (51+)" },
  ];
  const texts = {
    upper: {
      0: ["The forehead takes a smaller share of this face. Old readers would say school was not where your luck lived; the twenties are for learning the trade, and the paying decades come later.",
          "A short upper court. The tradition is blunt about it: a slow start, and nothing wrong with that. What you build by hand at twenty-five still works at sixty."],
      1: ["The upper court takes its fair third. The reading is a youth without drama: help arrived when needed, lessons landed on time.",
          "A standard upper court. Nothing in the forehead argues with the rest of the face, which the manuals count as quiet luck in the early years."],
      2: ["A tall forehead, and the manuals never shut up about tall foreheads: early wit, early favor, elders who took an interest. The first act carries you further than most people's.",
          "The upper court leads this face. That's the scholar's share: ideas before experience, recognition before thirty, and a lifelong habit of being three steps ahead on paper."],
    },
    middle: {
      0: ["A short middle court. The tradition reads the thirties and forties as a bridge you cross rather than a hill you plant a flag on. The reward sits on the far side.",
          "The middle court keeps a low profile. Old readers called this the patient build: mid-life effort that pays out late, like a field sown in autumn."],
      1: ["The middle court holds its third. Career and character pull in the same direction through the middle years, which is rarer than it sounds.",
          "A steady middle court. No cliff, no rocket. The manuals like this: it means the engine and the brakes are the same size."],
      2: ["A long middle court, the self-made share. The manuals put your strongest decades in the thirties and forties, when effort converts to standing at the best exchange rate you'll get.",
          "The middle court dominates. Tradition calls this the prime-of-life face: appetite, stamina, and authority peaking together in the middle decades."],
    },
    lower: {
      0: ["A light lower court. The reading is a late life that stays in motion: smaller house, more tickets. The manuals advise planting a few roots on purpose, since they won't grow by accident.",
          "The chin and jaw take a small share. Old readers saw late years spent light on your feet, long on novelty, short on sitting still. Store some firewood anyway."],
      1: ["The lower court holds its ground. Later years keep what the earlier ones built, at the pace it was built. Comfort without a moat.",
          "A fair lower court. The reading: a last act in proportion, with friendships that are older than most furniture."],
      2: ["A strong lower court, the 地阁, the ground pavilion. This is the tradition's favorite omen: a late harvest, a full table, and a word that carries further at seventy than it did at thirty.",
          "The jaw and chin anchor this face. The manuals read that as late-life authority: loyalty accumulated like interest, and a household that gathers instead of scattering."],
    },
  };
  const courts = courtDefs.map((c) => {
    const v = cal[c.key];
    const zi = (v - 1 / 3) / 0.025;
    const b = band3(zi, -0.8, 0.8);
    return { ...c, value: v, band: b, text: pickVariant(rng, texts[c.key][b]) };
  });
  const spread = Math.max(...courts.map((c) => c.value)) - Math.min(...courts.map((c) => c.value));
  return { courts, balanced: spread < 0.035 };
}

// ----- Five features (五官) -----
function interpretFeatures(m, rng) {
  const features = [];
  const F = (hanzi, name, trait, text) => features.push({ hanzi, name, trait, text });

  // Eyebrows 眉
  {
    const tb = band3(z(m.browThickness, "browThickness"));
    const main = [
      { label: "Fine brows",
        v: ["Light brows. The manuals read them as a light touch: tact, taste, and a preference for winning arguments nobody realized were happening.",
            "Fine brows, drawn with a dry pen. The reading is discernment: you'd rather persuade than push, and you notice what people wear before you notice what they say."] },
      { label: "Even brows",
        v: ["Brows of ordinary weight, which the manuals actually prize: 'neat as if painted.' The reading is a long fuse and a longer memory for kindness.",
            "Nothing excessive in the brows. Old readers took that as emotional bookkeeping done properly: anger spent rarely, loyalty spent steadily."] },
      { label: "Heavy brows",
        v: ["Heavy brows mean strong qi in the old books: quick decisions, full commitment, a temper that arrives late and means it. You defend your people.",
            "Thick brows. The manuals pair them with appetite and courage, and warn everyone else about getting between you and whatever you've decided on."] },
    ][tb];
    const ab = band3(z(m.browAngle, "browAngle"));
    const angleTxt = [
      " They curve down at the ends, the so-called benevolent brow. Strangers tell you things in checkout lines.",
      "",
      " They run straight and slightly lifted, toward the sword brow, 剑眉. The manuals add ambition and a taste for competition.",
    ][ab];
    const gb = band3(z(m.browEyeGap, "browEyeGap"));
    const gapTxt = [
      " They sit low over the eyes: thought and action with almost no daylight between them.",
      "",
      " They sit high, leaving a wide field above the eye. The reading is patience, and decisions that marinate.",
    ][gb];
    F("眉", "Eyebrows · Longevity Official", main.label, pickVariant(rng, main.v) + angleTxt + gapTxt);
  }

  // Eyes 目
  {
    const sb = band3(z(m.eyeSize, "eyeSize"));
    const main = [
      { label: "Compact eyes",
        v: ["Smaller eyes. The old readers liked them for judgment: attention spent like money, and nothing given away free at the table.",
            "Compact, focused eyes. The reading is the appraiser's: you see the flaw, the price, and the exit, in that order."] },
      { label: "Proportioned eyes",
        v: ["Eyes in proportion to the face. The reading is a level lens: the world seen at actual size, neither inflated nor discounted.",
            "Middle-sized eyes, which the manuals score as the honest middle: perceptive without being gullible, private without being locked."] },
      { label: "Large eyes",
        v: ["Large eyes carry what the classics call 神, spirit, the single thing a reader looks for first. The reading is feeling shown, feeling returned, and rooms that notice when you arrive.",
            "Big, open eyes. The manuals read them as a life conducted in daylight: trust given early, which mostly comes back, and attention that is hard to fake."] },
    ][sb];
    const spb = band3(z(m.eyeSpacing, "eyeSpacing"));
    const spTxt = [
      " Set close together, they narrow to a point: a specialist's gaze that holds one thread until it gives up and unknots.",
      " Their spacing lands near the old ideal of one eye-width apart.",
      " Set wide apart, they take in the whole field before judging any corner of it. Tolerance, with range.",
    ][spb];
    const ob = band3(z(m.eyeOpenness, "eyeOpenness"));
    const opTxt = [
      " Held slightly narrowed, the strategist's squint. Nobody reads your cards off your face.",
      "",
      " And held wide open, which makes hiding anything mostly theoretical.",
    ][ob];
    F("目", "Eyes · Inspector Official", main.label, pickVariant(rng, main.v) + spTxt + opTxt);
  }

  // Nose 鼻
  {
    const wb = band3(z(m.noseWidth, "noseWidth"));
    const main = [
      { label: "Narrow nose",
        v: ["A slender nose. In the wealth seat, the manuals read this as money handled with tweezers: fewer streams, better chosen, beautifully spent.",
            "A fine, narrow nose. The reading is standards: you'd rather have one good thing than three adequate ones, and your accounts reflect it."] },
      { label: "Proportioned nose",
        v: ["The nose, which the tradition treats as the money seat, is in proportion here. The reading is income that behaves: it arrives by effort and stays out of drama.",
            "A balanced nose. Old readers scored this as the merchant's steady ledger, earnings and character keeping pace with each other."] },
      { label: "Broad nose",
        v: ["A broad nose with full wings, the famous 金甲, the money bags. This is the most flattered feature in the whole tradition: earning power, plus the generosity that keeps money moving.",
            "The nose is the wealth seat, and this one is built like a vault with the door open: strong earnings, big appetite, and a reflex for picking up the check."] },
    ][wb];
    const lb = band3(z(m.noseLength, "noseLength"));
    const lenTxt = [
      " It runs short, which adds speed: fortune favoring the quick decision over the five-year plan.",
      "",
      " It runs long, which adds patience: money treated as a conversation measured in decades.",
    ][lb];
    F("鼻", "Nose · Judge Official", main.label, pickVariant(rng, main.v) + lenTxt);
  }

  // Mouth 口
  {
    const wb = band3(z(m.mouthWidth, "mouthWidth"));
    const main = [
      { label: "Compact mouth",
        v: ["A smaller mouth. The manuals read it as a vault: words weighed before spent, secrets that die of old age, and a short list of people who matter enormously.",
            "A composed mouth. The reading is discretion, with affection concentrated rather than broadcast. People discover slowly that you'd do anything for them."] },
      { label: "Proportioned mouth",
        v: ["A mouth in proportion. The reading is the dependable speaker: words that match deeds closely enough that people quit double-checking.",
            "The mouth holds the classical middle. Speech and silence each get used at the right moment, and promises hold their shape."] },
      { label: "Wide mouth",
        v: ["A wide mouth. The classics say such a mouth 'can swallow a fist,' their way of saying appetite: for people, for food, for the whole table's attention. Fortune walks in through doors that conversation opened.",
            "A generous mouth, the host's feature. You gather people the way other faces gather dust, and the manuals count every gathering as stored luck."] },
    ][wb];
    const fb = band3(z(m.lipFullness, "lipFullness"));
    const lipTxt = [
      " The lips are cut thin, which reads as precision: affection delivered as acts, not announcements.",
      "",
      " The lips are full, which deepens the reading toward warmth and open-handedness.",
    ][fb];
    const cb = band3(z(m.mouthCornerLift, "mouthCornerLift"));
    const cornerTxt = [
      " At rest, the corners settle downward: a critic's mouth. Standards first, applause afterward, if earned.",
      "",
      " At rest, the corners ride upward, the upturned lotus. Help arrives before you ask for it.",
    ][cb];
    F("口", "Mouth · Exit & Entry Official", main.label, pickVariant(rng, main.v) + lipTxt + cornerTxt);
  }

  // Forehead 额
  {
    const ub = band3(z(m.courts.upper, "courtUpper"));
    const main = [
      { label: "Low forehead",
        v: ["A lower forehead. The reading is intelligence with calluses: knowledge proven in the world, not the library, and respect collected in results.",
            "An economical forehead. Old readers gave such faces the working theory of everything: if it can't be used, it can wait."] },
      { label: "Even forehead",
        v: ["The forehead keeps fair proportion. The manuals call a smooth, even brow the gift of mentors: ideas arriving with help attached.",
            "A balanced forehead. The reading is a fair treaty between imagination and execution, honored by both sides."] },
      { label: "High forehead",
        v: ["A high forehead, 天庭, heaven's court at full height. The manuals read altitude here as a head start: abstraction comes easy, and recognition tends to arrive early.",
            "A tall forehead. The scholar's share: conclusions that show up years before the evidence, and elders who noticed you first."] },
    ][ub];
    const fjb = band3(z(m.foreheadJaw, "foreheadJaw"));
    const fjTxt = [
      " It sits narrower than the jaw below, strength gathered low: instinct leads, intellect follows with the paperwork.",
      "",
      " It spreads wider than the jaw below, which tips the face visionary: the idea arrives first and drags the body along.",
    ][fjb];
    F("额", "Forehead · Heaven's Court", main.label, pickVariant(rng, main.v) + fjTxt);
  }

  // Chin & jaw 颏
  {
    const jb = band3(z(m.jawRatio, "jawRatio"));
    const main = [
      { label: "Tapered jaw",
        v: ["The jaw tapers. The manuals read it as agility: a life steered by wit and course-correction. Their advice is to collect steady allies, who become the ground you stand on.",
            "A narrow jaw, the diplomat's taper. You sidestep what blunter faces bulldoze, and you're out the side door before the argument notices."] },
      { label: "Even jaw",
        v: ["A jaw in proportion. The reading is stamina with a release valve: projects get finished, grudges get retired.",
            "The jaw holds the middle. Endurance and flexibility in roughly equal parts, which old readers scored as a finisher's face."] },
      { label: "Broad jaw",
        v: ["A broad, set jaw. In the manuals this is willpower's home address: slow to promise, incapable of abandoning, and heavier in authority every decade.",
            "A strong jaw squares the lower face. The reading is bedrock: opinions with foundations, loyalty with tenure."] },
    ][jb];
    const cb = band3(z(m.chinHeight, "chinHeight"));
    const chinTxt = [
      " The chin itself is short: a sprinter's finish, late life kept quick and portable.",
      "",
      " The chin runs long, which the classics read as longevity with territory: a last act that owns land.",
    ][cb];
    F("颏", "Chin & Jaw · Ground Pavilion", main.label, pickVariant(rng, main.v) + chinTxt);
  }

  return features;
}

// ----- Twelve palaces (十二宫) -----
function interpretPalaces(m, rng) {
  const P = (name, hanzi, zv, texts) => {
    const b = band3(zv);
    const t = texts[b];
    return { name, hanzi, text: Array.isArray(t) ? pickVariant(rng, t) : t };
  };
  return [
    P("Life Palace", "命宫", z(m.browEyeGap, "browEyeGap"), [
      "The yintang between the brows sits narrow. The reading is a path forced open by will rather than granted by weather. It opens; you just carry the crowbar.",
      "The yintang holds normal width. A road with room to pass, the manuals say, so long as you keep walking it.",
      "The yintang is wide and clear, the single luckiest patch of real estate in the old system. Doors open slightly before you knock.",
    ]),
    P("Career Palace", "官禄宫", z(m.courts.upper, "courtUpper"), [
      "A compact career palace. The ladder matters less to you than the workbench; reputation grows sideways, through peers, and holds.",
      "A standard career palace. Promotion on schedule, the manuals say: no rocket, no ceiling.",
      "A tall career palace on a high forehead. The reading is visible work: achievement that superiors can't help noticing, whether or not you point at it.",
    ]),
    P("Wealth Palace", "财帛宫", z(m.noseWidth, "noseWidth"), [
      "A slim wealth palace. Money moves through taste and timing here: lighter stacking, better spending.",
      "A balanced wealth palace. Sufficiency, honestly earned, is the manuals' verdict, and they consider it underrated.",
      "A full wealth palace, the best-funded nose the tradition knows. Earning power plus the nerve to use it.",
    ]),
    P("Marriage Palace", "夫妻宫", z(m.eyeSpacing, "eyeSpacing"), [
      "A close-set marriage palace. Devotion with standards: fewer bonds, gripped harder.",
      "A balanced marriage palace. Closeness that doesn't crowd; independence that doesn't drift.",
      "A spacious marriage palace. The reading is a household run on trust and latitude, with a partner found through openness.",
    ]),
    P("Children Palace", "子女宫", z(m.eyeOpenness, "eyeOpenness"), [
      "A quiet children palace. Care shows up here as provision and steadiness, groceries before speeches.",
      "A balanced children palace. Warmth given in proportion to need, which children notice more than volume.",
      "A bright, full children palace. Vitality passed down with both hands, and the young gravitate to it.",
    ]),
    P("Health Palace", "疾厄宫", z(m.noseLength, "noseLength"), [
      "A short nose bridge counsels pacing: the engine runs hot and deserves scheduled rest.",
      "A steady nose bridge. A constitution that answers faithfully to how it's treated.",
      "A long, unbroken nose bridge. The old reading is durability: a frame that takes weather.",
    ]),
    P("Travel Palace", "迁移宫", z(m.foreheadJaw, "foreheadJaw"), [
      "A grounded travel palace. Your luck concentrates where your name is known; roots beat routes.",
      "A neutral travel palace. Fortune travels with you and doesn't check the map.",
      "Broad temples mark the travel palace. Luck improves with distance: journeys, moves, and far-off allies all favored.",
    ]),
    P("Fortune & Virtue", "福德宫", z(m.mouthCornerLift, "mouthCornerLift"), [
      "A gravely set mouth marks this palace. Happiness treated as a discipline, kept like a garden rather than won like a raffle.",
      "A level mouth marks this palace: contentment that doesn't depend on applause.",
      "An upturned mouth marks this palace. The old books call it a blessed set: luck follows a face that already looks lucky.",
    ]),
  ];
}

// ----- Symmetry (阴阳 balance) -----
// In the tradition, the left side of the face is yang (阳) and the right is
// yin (阴); a tilt toward one side is read as that principle leading.
function interpretSymmetry(m, rng) {
  const items = [];
  const za = z(m.mouthTiltAsym, "mouthTiltAsym");
  const ze = z(m.eyeOpenAsym, "eyeOpenAsym");
  const zb = z(m.browGapAsym, "browGapAsym");

  const mouthB = band3(Math.abs(za), 0, 0.9); // 0 unused; 1 = even, 2 = tilted
  items.push({
    name: "Mouth", hanzi: "口",
    text: mouthB === 2
      ? pickVariant(rng, [
          "The mouth rests slightly tilted, one corner leading. Old readers liked this: it marks the ironist, someone whose smile has a footnote. You say the thing everyone was thinking, a beat before they admit it.",
          "One corner of the mouth sits higher than the other. The manuals read a tilted mouth as wit with an edge: your compliments are audited, and your jokes land because they're half true.",
        ])
      : pickVariant(rng, [
          "The mouth sits level. The reading is a steady witness: what you say on Monday still stands on Friday.",
          "Both corners hold the same line. The manuals read an even mouth as even dealing, the same face shown to the powerful and the unimportant.",
        ]),
  });

  const eyeB = band3(Math.abs(ze), 0, 0.9);
  items.push({
    name: "Eyes", hanzi: "目",
    text: eyeB === 2
      ? pickVariant(rng, [
          "One eye opens a shade wider than the other, what the tradition calls yin-yang eyes, 阴阳眼. The reading is a double audit: one eye takes the room at face value while the other checks the receipts.",
          "The eyes open unevenly, the 阴阳眼 of the old books. Such faces are supposed to be hard to fool twice: half of you believes people, and the other half takes notes.",
        ])
      : pickVariant(rng, [
          "The eyes open evenly. The reading is a settled gaze: judgment that doesn't wobble with the lighting.",
          "Both eyes hold the same aperture. Old readers scored even eyes as even temper, the same verdict in the morning as at night.",
        ]),
  });

  const browB = band3(Math.abs(zb), 0, 0.9);
  items.push({
    name: "Brows", hanzi: "眉",
    text: browB === 2
      ? pickVariant(rng, [
          "One brow rides higher than the other, the asker's face. A permanent half-question lives there, and the manuals read it as a nose for the unsaid.",
          "The brows sit at different heights. The tradition reads the raised side as the skeptic on duty: belief is available, but it has to apply in writing.",
        ])
      : pickVariant(rng, [
          "The brows sit level with each other. The reading is an even keel between impulse and doubt.",
          "Matched brows. The manuals read them as a balanced court: enthusiasm and skepticism each get a vote.",
        ]),
  });

  const tilted = [mouthB, eyeB, browB].filter((b) => b === 2).length;
  const verdict =
    tilted === 0
      ? "Altogether an unusually even face. The tradition reads strong symmetry as yin and yang in truce: a temperament that others calibrate their instruments against."
      : tilted === 1
      ? "One feature breaks the symmetry, which the old readers preferred to perfection: a face with one signature tilt is a face people remember."
      : "More than one feature sits off-axis, and the manuals consider that the interesting kind of face: yin and yang still negotiating, which reads as range, humor, and weather.";

  return { items, verdict };
}

// ----- Other traditions (他山之镜) -----
function interpretTraditions(m, rng) {
  const cards = [];

  // Vedic Samudrika Shastra
  {
    const fh = band3(z(m.courts.upper, "courtUpper"));
    const lip = band3(z(m.lipFullness, "lipFullness"));
    const eye = band3(z(m.eyeSize, "eyeSize"));
    const fhTxt = [
      "a modest forehead, which it assigns to the doers, people whose knowledge lives in their hands",
      "an even forehead, the mark of a balanced grihastha, a householder who keeps accounts and promises",
      "a broad forehead, which the texts flag for intellect and early fortune",
    ][fh];
    const lipTxt = [
      "Thin lips are listed under discipline: speech rationed, passions governed.",
      "Moderate lips are listed under even speech: neither flattery nor famine.",
      "Full lips are listed under rasa, the capacity for enjoyment, and read as a warm and persuasive tongue.",
    ][lip];
    const eyeTxt = [
      "Narrow eyes fall under shrewdness in these texts, the trader's squint that misses nothing on the scale.",
      "Proportioned eyes fall under clarity, a witness the texts consider reliable.",
      "Large eyes fall under sensitivity, quick to feel and quick to show it.",
    ][eye];
    cards.push({
      name: "Samudrika Shastra", script: "सामुद्रिक",
      origin: "Vedic India",
      text: `The Sanskrit body-reading tradition would note ${fhTxt}. ${lipTxt} ${eyeTxt}`,
    });
  }

  // Greek four temperaments
  {
    const za = z(m.aspect, "aspect");
    const zj = z(m.jawRatio, "jawRatio");
    const zlip = z(m.lipFullness, "lipFullness");
    const zcl = z(m.mouthCornerLift, "mouthCornerLift");
    const zba = z(m.browAngle, "browAngle");
    const scores = {
      choleric: 0.5 * zj + 0.3 * zba + 0.2 * za,
      sanguine: 0.5 * zcl + 0.3 * zlip + 0.2 * z(m.mouthWidth, "mouthWidth"),
      phlegmatic: -0.5 * za - 0.3 * Math.abs(zj) + 0.2 * zlip,
      melancholic: 0.5 * za - 0.3 * zlip - 0.2 * zcl,
    };
    const key = Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0];
    const texts = {
      choleric: "the choleric: fire and yellow bile, the commander's mix. Galen's heirs would prescribe you a project immediately, before you commandeer someone else's.",
      sanguine: "the sanguine: air and blood, the lucky temperament. The physiognomists read an upturned, full-featured face as appetite for company and a pulse that recovers fast from everything, including its own promises.",
      phlegmatic: "the phlegmatic: water and calm. The old physicians meant it kindly: the person who doesn't capsize, and whose advice sounds better the worse things get.",
      melancholic: "the melancholic: earth and black bile, the scholar's humor. Aristotle's school claimed all remarkable men had a measure of it; the long face carries long thoughts.",
    };
    cards.push({
      name: "Four Temperaments", script: "ΦΥΣΙΣ",
      origin: "Classical Greece",
      text: `Greek physiognomy sorted faces into four humors. The geometry here leans ${texts[key]}`,
    });
  }

  // Japanese ninsō
  {
    const nl = band3(z(m.noseLength, "noseLength"));
    const ph = band3(z(m.philtrum, "philtrum"));
    const nlTxt = [
      "a short nose, which it reads as adaptability, the gift of starting over without ceremony",
      "a measured nose, which it reads as a self in proportion, neither servant nor tyrant to its plans",
      "a long nose, which it reads as self-possession, a spine that runs the length of the face",
    ][nl];
    const phTxt = [
      "A short philtrum is read for quickness: a life that keeps its youth in the top drawer.",
      "An even philtrum is read for steady ki: the channel between heaven and earth kept clear.",
      "A long, deep philtrum is the classic longevity groove, and ninsō readers tap it approvingly.",
    ][ph];
    cards.push({
      name: "Ninsō", script: "人相",
      origin: "Edo Japan",
      text: `Japanese ninsō reads the nose as the self. This face shows ${nlTxt}. ${phTxt}`,
    });
  }

  return cards;
}

// ----- Standouts & summary -----
const STANDOUTS = [
  { key: "eyeSpacing", hi: "eyes set unusually wide, a surveyor's gaze", lo: "eyes set unusually close, a jeweler's gaze" },
  { key: "browEyeGap", hi: "brows set high above the eyes, patience built into the architecture", lo: "brows set low over the eyes, no gap between thought and act" },
  { key: "noseWidth", hi: "a wealth seat built broad, the 金甲 money bags", lo: "a finely drawn nose, the connoisseur's" },
  { key: "mouthWidth", hi: "a mouth wide enough to 'swallow a fist of fortune'", lo: "a vault of a mouth, where secrets retire" },
  { key: "lipFullness", hi: "notably full lips, warmth you can see across a room", lo: "finely cut lips, precision in the smallest feature" },
  { key: "jawRatio", hi: "a jaw like a foundation stone", lo: "a jaw tapered like a diplomat's answer" },
  { key: "browAngle", hi: "lifted sword brows, 剑眉, competition in the blood", lo: "deeply curved brows, the benevolent kind strangers trust" },
  { key: "chinHeight", hi: "a long chin, the classic longevity ground", lo: "a quick, compact chin, late life kept light" },
  { key: "eyeOpenness", hi: "eyes held strikingly open, candor on display", lo: "a strategist's narrowed gaze" },
  { key: "philtrum", hi: "a long, deep philtrum, the longevity groove", lo: "a short philtrum, youth kept in the top drawer" },
  { key: "mouthCornerLift", hi: "mouth corners that rise at rest, the upturned lotus", lo: "a gravely set mouth, a critic's rest position" },
  { key: "aspect", hi: "a notably long face, Wood's upward reach", lo: "a notably round face, Water's gathered fullness" },
];

function buildSummary(element, courtInfo, m, rng) {
  const parts = [];
  const art = element.name === "Earth" ? "an" : "a";
  if (element.secondary) {
    parts.push(`This is ${art} ${element.name} (${element.hanzi}) face: ${element.blend} ${element.secondary.name} (${element.secondary.hanzi}) underneath.`);
  } else {
    parts.push(`This is ${art} ${element.name} (${element.hanzi}) face, and not a borderline case.`);
  }

  const courts = courtInfo.courts;
  const dominant = [...courts].sort((a, b) => b.value - a.value)[0];
  if (courtInfo.balanced) {
    parts.push("The three courts sit in rare balance, which the manuals score highest of all: each age of life hands its winnings to the next.");
  } else {
    const phase = dominant.key === "upper" ? "first act" : dominant.key === "middle" ? "middle act" : "late act";
    parts.push(`Of the three courts, the ${dominant.name.toLowerCase()} (${dominant.hanzi}) takes the largest share, so the tradition puts your strongest ground in the ${phase} of life.`);
  }

  const ranked = STANDOUTS
    .map((s) => ({ ...s, zv: z(m[s.key], s.key) }))
    .filter((s) => Math.abs(s.zv) > 0.8)
    .sort((a, b) => Math.abs(b.zv) - Math.abs(a.zv))
    .slice(0, 2);
  if (ranked.length === 2) {
    parts.push(`Two things a street reader would mention before you sat down: ${ranked[0].zv > 0 ? ranked[0].hi : ranked[0].lo}, and ${ranked[1].zv > 0 ? ranked[1].hi : ranked[1].lo}.`);
  } else if (ranked.length === 1) {
    parts.push(`The one thing a street reader would mention before you sat down: ${ranked[0].zv > 0 ? ranked[0].hi : ranked[0].lo}.`);
  } else {
    parts.push("No single feature shouts. The old readers charged extra for faces like this: everything in agreement, nothing leaking.");
  }

  parts.push(pickVariant(rng, [
    "And the standard disclaimer, straight from the masters: 相由心生, the face follows the heart. A kind one renegotiates every clause above.",
    "The old texts close every reading the same way: 相由心生, the face is shaped by the heart. Character amends whatever the features propose.",
  ]));
  return parts.join(" ");
}

// ---------------------------------------------------------------------------
// Canvas rendering
// ---------------------------------------------------------------------------
function renderCanvas() {
  if (!lastRender) return;
  const { img } = lastRender;
  const maxW = 900;
  const scale = Math.min(1, maxW / img.naturalWidth);
  canvas.width = Math.round(img.naturalWidth * scale);
  canvas.height = Math.round(img.naturalHeight * scale);
  redrawOverlay();
}

function redrawOverlay() {
  if (!lastRender) return;
  const { img, pts } = lastRender;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  drawOverlayOn(ctx, pts, img, canvas.width, canvas.height,
    $("toggle-courts").checked, $("toggle-points").checked);
}

function drawOverlayOn(c, pts, img, w, h, showCourts, showPoints) {
  const sx = w / img.naturalWidth;
  const sy = h / img.naturalHeight;
  const P = (i) => ({ x: pts[i].x * sx, y: pts[i].y * sy });

  if (showCourts) {
    const left = P(L.faceL).x - 14;
    const right = P(L.faceR).x + 14;
    const ys = [hairlineY(pts) * sy, P(L.glabella).y, P(L.subnasale).y, P(L.chinTip).y];
    c.save();
    c.strokeStyle = "rgba(224, 194, 104, 0.85)";
    c.fillStyle = "rgba(224, 194, 104, 0.95)";
    c.lineWidth = Math.max(1.2, w * 0.0015);
    c.setLineDash([6, 5]);
    c.font = `${Math.max(11, w * 0.022)}px "Noto Serif SC", serif`;
    for (const y of ys) {
      c.beginPath();
      c.moveTo(left, y);
      c.lineTo(right, y);
      c.stroke();
    }
    c.setLineDash([]);
    const labels = ["上停", "中停", "下停"];
    for (let i = 0; i < 3; i++) {
      const yMid = (ys[i] + ys[i + 1]) / 2;
      c.save();
      c.shadowColor = "rgba(0,0,0,0.8)";
      c.shadowBlur = 4;
      c.fillText(labels[i], Math.min(right + 6, w - 34), yMid + 5);
      c.restore();
    }
    c.restore();
  }

  if (showPoints) {
    const keyPoints = [
      L.foreheadTop, L.glabella, L.subnasale, L.chinTip,
      L.eyeLOuter, L.eyeLInner, L.eyeRInner, L.eyeROuter,
      L.alarL, L.alarR, L.mouthL, L.mouthR, L.noseTip,
      L.browLInner, L.browLOuter, L.browRInner, L.browROuter,
      L.jawL, L.jawR,
    ];
    c.save();
    for (const i of keyPoints) {
      const pt = P(i);
      c.beginPath();
      c.arc(pt.x, pt.y, Math.max(2, w * 0.004), 0, Math.PI * 2);
      c.fillStyle = "rgba(224, 90, 64, 0.9)";
      c.fill();
      c.strokeStyle = "rgba(0,0,0,0.5)";
      c.lineWidth = 0.8;
      c.stroke();
    }
    c.restore();
  }
}

// ---------------------------------------------------------------------------
// Sharing
// ---------------------------------------------------------------------------
function buildShareCanvas() {
  const { img, pts, reading } = lastRender;
  const W = 1080, H = 1500;
  const card = document.createElement("canvas");
  card.width = W; card.height = H;
  const c = card.getContext("2d");

  c.fillStyle = "#16130e";
  c.fillRect(0, 0, W, H);
  const grad = c.createRadialGradient(W / 2, -100, 50, W / 2, -100, 900);
  grad.addColorStop(0, "rgba(201,162,39,0.14)");
  grad.addColorStop(1, "rgba(201,162,39,0)");
  c.fillStyle = grad;
  c.fillRect(0, 0, W, H);

  c.fillStyle = "#c3402b";
  roundRect(c, 60, 56, 86, 86, 14); c.fill();
  c.fillStyle = "#f3e9d6";
  c.font = '900 34px "Noto Serif SC", serif';
  c.textAlign = "center";
  c.fillText("面", 103, 95); c.fillText("相", 103, 130);
  c.textAlign = "left";
  c.fillStyle = "#e0c268";
  c.font = '900 52px "Noto Serif SC", serif';
  c.fillText("Mianxiang", 170, 112);
  c.fillStyle = "#b3a88f";
  c.font = '400 22px "Inter", sans-serif';
  c.fillText("Traditional Chinese Face Reading", 172, 146);

  const pad = 60, photoY = 190, photoH = 620, photoW = W - pad * 2;
  const minX = Math.max(0, Math.min(...pts.map(p => p.x)) - img.naturalWidth * 0.08);
  const maxX = Math.min(img.naturalWidth, Math.max(...pts.map(p => p.x)) + img.naturalWidth * 0.08);
  const minY = Math.max(0, hairlineY(pts) - img.naturalHeight * 0.06);
  const maxY = Math.min(img.naturalHeight, pts[L.chinTip].y + img.naturalHeight * 0.06);
  const crop = { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
  const fit = Math.min(photoW / crop.w, photoH / crop.h);
  const dw = crop.w * fit, dh = crop.h * fit;
  const dx = (W - dw) / 2, dy = photoY + (photoH - dh) / 2;
  c.save();
  roundRect(c, dx, dy, dw, dh, 16); c.clip();
  c.drawImage(img, crop.x, crop.y, crop.w, crop.h, dx, dy, dw, dh);
  c.translate(dx - crop.x * fit, dy - crop.y * fit);
  const fakeImg = { naturalWidth: img.naturalWidth, naturalHeight: img.naturalHeight };
  drawOverlayOn(c, pts, fakeImg, img.naturalWidth * fit, img.naturalHeight * fit, true, false);
  c.restore();
  c.strokeStyle = "rgba(201,162,39,0.5)";
  c.lineWidth = 2;
  roundRect(c, dx, dy, dw, dh, 16); c.stroke();

  let y = photoY + photoH + 86;
  c.fillStyle = "#e0c268";
  c.font = '900 110px "Noto Serif SC", serif';
  c.fillText(reading.element.hanzi, pad, y + 10);
  c.fillStyle = "#ede4d3";
  c.font = '900 54px "Noto Serif SC", serif';
  c.fillText(`${reading.element.name} Face`, pad + 140, y - 28);
  c.fillStyle = "#b3a88f";
  c.font = 'italic 24px "Inter", sans-serif';
  c.fillText(reading.element.sub, pad + 140, y + 10);

  y += 80;
  c.font = '600 26px "Inter", sans-serif';
  const labels = { upper: "Upper 上停", middle: "Middle 中停", lower: "Lower 下停" };
  for (const ct of reading.courts.courts) {
    c.fillStyle = "#ede4d3";
    c.fillText(labels[ct.key], pad, y);
    c.fillStyle = "#2a241a";
    roundRect(c, pad + 230, y - 20, 560, 24, 12); c.fill();
    const wv = Math.min(1, ct.value * 2.2) * 560;
    const bar = c.createLinearGradient(pad + 230, 0, pad + 230 + wv, 0);
    bar.addColorStop(0, "#c9a227"); bar.addColorStop(1, "#e0c268");
    c.fillStyle = bar;
    roundRect(c, pad + 230, y - 20, Math.max(24, wv), 24, 12); c.fill();
    c.fillStyle = "#e0c268";
    c.fillText(`${Math.round(ct.value * 100)}%`, pad + 820, y);
    y += 56;
  }

  y += 16;
  c.font = '500 23px "Inter", sans-serif';
  let cx = pad;
  for (const f of reading.features.slice(0, 6)) {
    const t = `${f.hanzi} ${f.trait}`;
    const tw = c.measureText(t).width + 44;
    if (cx + tw > W - pad) { cx = pad; y += 58; }
    c.fillStyle = "rgba(201,162,39,0.1)";
    roundRect(c, cx, y - 32, tw, 46, 23); c.fill();
    c.strokeStyle = "rgba(201,162,39,0.35)";
    c.lineWidth = 1.5;
    roundRect(c, cx, y - 32, tw, 46, 23); c.stroke();
    c.fillStyle = "#e0c268";
    c.fillText(t, cx + 22, y);
    cx += tw + 14;
  }

  c.fillStyle = "#8a8070";
  c.font = '400 19px "Inter", sans-serif';
  c.fillText("A folk tradition read for fun — not science, not destiny. 相由心生", pad, H - 46);

  return card;
}

function roundRect(c, x, y, w, h, r) {
  c.beginPath();
  c.moveTo(x + r, y);
  c.arcTo(x + w, y, x + w, y + h, r);
  c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r);
  c.arcTo(x, y, x + w, y, r);
  c.closePath();
}

async function shareCard() {
  if (!lastRender) return;
  const card = buildShareCanvas();
  const blob = await new Promise((res) => card.toBlob(res, "image/png"));
  const file = new File([blob], "mianxiang-reading.png", { type: "image/png" });
  if (navigator.canShare?.({ files: [file] })) {
    try {
      await navigator.share({ files: [file], title: "My Mianxiang face reading" });
      return;
    } catch (e) {
      if (e.name === "AbortError") return;
    }
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "mianxiang-reading.png";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  toast("Share card saved as an image.");
}

function readingAsText() {
  const r = lastRender.reading;
  const lines = [];
  lines.push("MY MIANXIANG FACE READING 面相");
  lines.push("");
  lines.push(`Five-Element type: ${r.element.name} ${r.element.hanzi}${r.element.secondary ? ` (with ${r.element.secondary.name} ${r.element.secondary.hanzi})` : ""}`);
  lines.push(r.element.sub);
  lines.push("");
  lines.push("The Three Courts 三停:");
  for (const ct of r.courts.courts) lines.push(`  ${ct.name} ${ct.hanzi} (${ct.span}): ${Math.round(ct.value * 100)}%`);
  lines.push("");
  lines.push(`The Reading: ${r.summary}`);
  lines.push("");
  lines.push("The Five Features 五官:");
  for (const f of r.features) lines.push(`  ${f.hanzi} ${f.name} — ${f.trait}: ${f.text}`);
  lines.push("");
  lines.push("Balance 阴阳:");
  for (const s of r.symmetry.items) lines.push(`  ${s.hanzi} ${s.name}: ${s.text}`);
  lines.push(`  ${r.symmetry.verdict}`);
  lines.push("");
  lines.push("Palaces 十二宫:");
  for (const p of r.palaces) lines.push(`  ${p.name} ${p.hanzi}: ${p.text}`);
  lines.push("");
  lines.push("Other Mirrors:");
  for (const t of r.traditions) lines.push(`  ${t.name} (${t.origin}): ${t.text}`);
  lines.push("");
  lines.push("(Folk traditions read for fun — not science, not destiny.)");
  return lines.join("\n");
}

async function copyReadingText() {
  if (!lastRender) return;
  try {
    await navigator.clipboard.writeText(readingAsText());
    toast("Full reading copied to clipboard.");
  } catch {
    toast("Couldn't access the clipboard — your browser may be blocking it.");
  }
}

let toastTimer = null;
function toast(msg) {
  let el = $("toast");
  if (!el) {
    el = document.createElement("div");
    el.id = "toast";
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 2600);
}

// ---------------------------------------------------------------------------
// Results rendering
// ---------------------------------------------------------------------------
function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function renderReading(r) {
  const blendLine = r.element.secondary
    ? `<p class="element-blend">with a secondary current of <strong>${r.element.secondary.name} ${r.element.secondary.hanzi}</strong></p>`
    : "";
  $("element-card").innerHTML = `
    <div class="element-glyph">${r.element.glyph}</div>
    <p class="element-kicker">Five-Element Face Type · 五行</p>
    <h2 class="element-title"><span class="hanzi">${r.element.hanzi}</span>${r.element.name} <small style="font-weight:400;font-size:15px;color:var(--ink-faint)">${r.element.pinyin}</small></h2>
    <p class="element-sub">${esc(r.element.sub)}</p>
    ${blendLine}
    <p class="element-body">${esc(r.element.text)}</p>
  `;

  const rows = r.courts.courts
    .map(
      (ct) => `
    <div class="court-row">
      <div class="court-label">
        <span><strong>${esc(ct.name)}</strong> ${ct.hanzi} · <em style="color:var(--ink-faint)">${esc(ct.span)}</em></span>
        <span class="pct">${Math.round(ct.value * 100)}%</span>
      </div>
      <div class="court-bar"><div class="court-fill" style="width:0%" data-w="${Math.min(100, ct.value * 220)}"></div></div>
      <p class="court-reading">${esc(ct.text)}</p>
    </div>`
    )
    .join("");
  $("courts-card").innerHTML = `
    <h2 class="courts-title"><span class="hanzi">三停</span>The Three Courts</h2>
    <p class="courts-note">Mianxiang divides the face into three vertical regions, each governing a season of life. ${r.courts.balanced ? "Yours are unusually balanced — considered highly auspicious." : "Proportions are measured from your photo, relative to a typical face."}</p>
    ${rows}
  `;
  requestAnimationFrame(() =>
    document.querySelectorAll(".court-fill").forEach((el) => {
      el.style.width = el.dataset.w + "%";
    })
  );

  $("summary-card").innerHTML = `
    <h3><span class="hanzi">总论</span>The Reading</h3>
    <p>${esc(r.summary)}</p>
  `;

  $("features-grid").innerHTML = r.features
    .map(
      (f) => `
    <div class="feature-card">
      <div class="feature-head">
        <span class="feature-hanzi">${f.hanzi}</span>
        <span class="feature-name">${esc(f.name)}</span>
      </div>
      <span class="feature-trait">${esc(f.trait)}</span>
      <p class="feature-body">${esc(f.text)}</p>
    </div>`
    )
    .join("");

  $("symmetry-grid").innerHTML = r.symmetry.items
    .map(
      (s) => `
    <div class="feature-card">
      <div class="feature-head">
        <span class="feature-hanzi">${s.hanzi}</span>
        <span class="feature-name">${esc(s.name)}</span>
      </div>
      <p class="feature-body">${esc(s.text)}</p>
    </div>`
    )
    .join("") +
    `<div class="feature-card symmetry-verdict"><p class="feature-body">${esc(r.symmetry.verdict)}</p></div>`;

  $("palaces-grid").innerHTML = r.palaces
    .map(
      (p) => `
    <div class="palace-card">
      <div class="palace-head">
        <span class="palace-name">${esc(p.name)}</span>
        <span class="palace-hanzi">${p.hanzi}</span>
      </div>
      <p class="palace-body">${esc(p.text)}</p>
    </div>`
    )
    .join("");

  $("traditions-grid").innerHTML = r.traditions
    .map(
      (t) => `
    <div class="tradition-card">
      <div class="tradition-head">
        <span class="tradition-script">${esc(t.script)}</span>
        <div>
          <div class="tradition-name">${esc(t.name)}</div>
          <div class="tradition-origin">${esc(t.origin)}</div>
        </div>
      </div>
      <p class="feature-body">${esc(t.text)}</p>
    </div>`
    )
    .join("");
}

// Warm the model in the background so first analysis is fast.
getLandmarker().catch(() => { /* will retry on first upload */ });
show(uploadSection);
