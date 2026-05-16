/**
 * SoundSentinel — Inference Web Worker
 * ======================================
 * Runs entirely off the main thread to keep the UI responsive.
 *
 * Responsibilities
 * ----------------
 * 1. Load the ONNX model from /model.onnx on startup
 * 2. On each message (raw PCM Float32Array from main thread):
 *    a. Extract a 326-dimensional feature vector (matching Python training)
 *    b. Run ONNX Runtime Web inference
 *    c. Post class probabilities back to the main thread
 *
 * Feature Vector Layout (must match training/train.py exactly)
 * -------------------------------------------------------------
 * Index   0 –  39  : MFCC means          (40 coefficients)
 * Index  40 –  79  : MFCC stds           (40 coefficients)
 * Index  80 – 119  : MFCC mins           (40 coefficients)
 * Index 120 – 159  : MFCC maxs           (40 coefficients)
 * Index 160 – 199  : Mel-spec means      (40 bands)
 * Index 200 – 239  : Mel-spec stds       (40 bands)
 * Index 240 – 279  : Mel-spec mins       (40 bands)
 * Index 280 – 319  : Mel-spec maxs       (40 bands)
 * Index 320        : Spectral centroid mean
 * Index 321        : Spectral centroid std
 * Index 322        : Spectral rolloff mean
 * Index 323        : Spectral rolloff std
 * Index 324        : Zero-crossing rate mean
 * Index 325        : Zero-crossing rate std
 *
 * Total: 326 features
 *
 * Message Protocol
 * ----------------
 * Incoming (from main thread):
 *   { type: "infer"; pcm: Float32Array; sampleRate: number }
 *   { type: "init" }  — sent on load to trigger model pre-warming
 *
 * Outgoing (to main thread):
 *   { type: "result"; probabilities: Float32Array; inferenceTimeMs: number }
 *   { type: "ready" }  — sent when model is loaded
 *   { type: "error"; message: string }
 */

import * as ort from "onnxruntime-web";

// ---------------------------------------------------------------------------
// Constants matching Python training configuration
// ---------------------------------------------------------------------------

const SAMPLE_RATE = 22050;  // Hz — same as train.py
const N_MFCC = 40;           // MFCC coefficients
const N_MELS = 40;           // Mel filterbank bands
const N_FFT = 2048;          // FFT window size
const HOP_LENGTH = 512;      // Frame hop in samples
const N_CLASSES = 50;

// Pre-computed Mel filterbank (Hz → Mel → linear Hz mapping)
// We'll build this lazily on first inference call
let melFilterbank: Float32Array[] | null = null;

// ONNX session — loaded once at startup
let session: ort.InferenceSession | null = null;
let modelLoaded = false;

// ---------------------------------------------------------------------------
// Mel-frequency helpers
// ---------------------------------------------------------------------------

/** Convert Hz to Mel scale */
function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

/** Convert Mel to Hz */
function melToHz(mel: number): number {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

/**
 * Build a Mel filterbank matrix.
 *
 * Returns an array of N_MELS filters, each of length (N_FFT/2 + 1).
 * Each filter is a triangular window centred on one Mel-scale frequency.
 */
function buildMelFilterbank(sampleRate: number): Float32Array[] {
  const nFft = N_FFT;
  const nMels = N_MELS;
  const fMin = 0;
  const fMax = sampleRate / 2;

  const melMin = hzToMel(fMin);
  const melMax = hzToMel(fMax);

  // nMels + 2 equally-spaced points on the Mel scale
  const melPoints = new Float32Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) {
    melPoints[i] = melMin + (i / (nMels + 1)) * (melMax - melMin);
  }

  // Convert back to Hz, then to FFT bin indices
  const hzPoints = melPoints.map(melToHz);
  const binPoints = hzPoints.map((hz) =>
    Math.floor((nFft + 1) * hz / sampleRate)
  );

  const freqBins = nFft / 2 + 1;
  const filters: Float32Array[] = [];

  for (let m = 1; m <= nMels; m++) {
    const filter = new Float32Array(freqBins);
    const lower = binPoints[m - 1];
    const center = binPoints[m];
    const upper = binPoints[m + 1];

    for (let k = lower; k < center; k++) {
      filter[k] = (k - lower) / (center - lower + 1e-10);
    }
    for (let k = center; k < upper; k++) {
      filter[k] = (upper - k) / (upper - center + 1e-10);
    }

    filters.push(filter);
  }

  return filters;
}

// ---------------------------------------------------------------------------
// FFT (Cooley–Tukey radix-2 DIT)
// ---------------------------------------------------------------------------

/**
 * In-place radix-2 Cooley–Tukey FFT.
 * Input: interleaved [re0, im0, re1, im1, ...] Float64Array of length 2*N
 * N must be a power of two.
 */
function fftInPlace(data: Float64Array): void {
  const n = data.length >> 1; // number of complex samples
  // Bit-reversal permutation
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      // Swap real and imaginary parts
      [data[2 * i], data[2 * j]] = [data[2 * j], data[2 * i]];
      [data[2 * i + 1], data[2 * j + 1]] = [data[2 * j + 1], data[2 * i + 1]];
    }
  }
  // Butterfly operations
  for (let len = 2; len <= n; len <<= 1) {
    const wRe = Math.cos((2 * Math.PI) / len);
    const wIm = -Math.sin((2 * Math.PI) / len);
    for (let i = 0; i < n; i += len) {
      let uRe = 1.0;
      let uIm = 0.0;
      for (let k = 0; k < len >> 1; k++) {
        const evenRe = data[2 * (i + k)];
        const evenIm = data[2 * (i + k) + 1];
        const tRe = uRe * data[2 * (i + k + (len >> 1))] - uIm * data[2 * (i + k + (len >> 1)) + 1];
        const tIm = uRe * data[2 * (i + k + (len >> 1)) + 1] + uIm * data[2 * (i + k + (len >> 1))];
        data[2 * (i + k)] = evenRe + tRe;
        data[2 * (i + k) + 1] = evenIm + tIm;
        data[2 * (i + k + (len >> 1))] = evenRe - tRe;
        data[2 * (i + k + (len >> 1)) + 1] = evenIm - tIm;
        const newURe = uRe * wRe - uIm * wIm;
        uIm = uRe * wIm + uIm * wRe;
        uRe = newURe;
      }
    }
  }
}

/**
 * Compute the power spectrum of a real-valued frame.
 * Returns an array of length N_FFT/2 + 1.
 */
function powerSpectrum(frame: Float32Array): Float32Array {
  const n = N_FFT;
  // Zero-pad frame to N_FFT if needed
  const complex = new Float64Array(2 * n); // interleaved re, im
  for (let i = 0; i < Math.min(frame.length, n); i++) {
    complex[2 * i] = frame[i];
  }
  fftInPlace(complex);
  const nBins = n / 2 + 1;
  const power = new Float32Array(nBins);
  for (let k = 0; k < nBins; k++) {
    const re = complex[2 * k];
    const im = complex[2 * k + 1];
    power[k] = (re * re + im * im) / (n * n);
  }
  return power;
}

// ---------------------------------------------------------------------------
// Hann window
// ---------------------------------------------------------------------------

function hannWindow(length: number): Float32Array {
  const w = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (length - 1)));
  }
  return w;
}

// ---------------------------------------------------------------------------
// DCT-II (for MFCC)
// ---------------------------------------------------------------------------

/**
 * Type-II DCT used by librosa's MFCC.
 * x: input array of length N
 * Returns output of length nCoeffs
 */
function dct2(x: Float32Array, nCoeffs: number): Float32Array {
  const n = x.length;
  const out = new Float32Array(nCoeffs);
  for (let k = 0; k < nCoeffs; k++) {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += x[i] * Math.cos((Math.PI * k * (2 * i + 1)) / (2 * n));
    }
    out[k] = sum * (k === 0 ? Math.sqrt(1 / n) : Math.sqrt(2 / n));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Core feature extraction
// ---------------------------------------------------------------------------

/**
 * Resample audio from one sample rate to another using linear interpolation.
 * For production you'd want a higher-quality resampler, but this is sufficient
 * for the feature accuracy needed at inference time.
 */
function resample(pcm: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate === toRate) return pcm;
  const ratio = fromRate / toRate;
  const outputLength = Math.floor(pcm.length / ratio);
  const out = new Float32Array(outputLength);
  for (let i = 0; i < outputLength; i++) {
    const pos = i * ratio;
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, pcm.length - 1);
    const t = pos - lo;
    out[i] = pcm[lo] * (1 - t) + pcm[hi] * t;
  }
  return out;
}

/**
 * Extract the full 326-dimensional feature vector from a raw PCM buffer.
 *
 * This mirrors the Python feature extraction in training/train.py:
 *   - MFCC × 4 statistics = 160 features
 *   - Mel-spectrogram × 4 statistics = 160 features
 *   - Spectral centroid × 2 = 2 features
 *   - Spectral rolloff × 2 = 2 features
 *   - Zero-crossing rate × 2 = 2 features
 *   Total: 326
 */
function extractFeatures(pcm: Float32Array, inputSampleRate: number): Float32Array {
  // 1. Resample to 22050 Hz to match Python training
  const audio = resample(pcm, inputSampleRate, SAMPLE_RATE);

  // Build Mel filterbank once
  if (!melFilterbank) {
    melFilterbank = buildMelFilterbank(SAMPLE_RATE);
  }

  const hann = hannWindow(N_FFT);
  const nBins = N_FFT / 2 + 1;

  // 2. Frame the signal
  const frames: Float32Array[] = [];
  for (let start = 0; start + N_FFT <= audio.length; start += HOP_LENGTH) {
    const frame = new Float32Array(N_FFT);
    for (let i = 0; i < N_FFT; i++) {
      frame[i] = audio[start + i] * hann[i];
    }
    frames.push(frame);
  }

  if (frames.length === 0) {
    // Audio too short — return zeros
    return new Float32Array(326);
  }

  // 3. Compute power spectra for all frames
  const powerSpectra: Float32Array[] = frames.map(powerSpectrum);

  // 4. Apply Mel filterbank to get log-Mel spectrogram [N_MELS × nFrames]
  const melSpec: Float32Array[] = melFilterbank!.map((filter) => {
    const bandEnergies = new Float32Array(frames.length);
    for (let f = 0; f < frames.length; f++) {
      let energy = 0;
      for (let k = 0; k < nBins; k++) {
        energy += filter[k] * powerSpectra[f][k];
      }
      // Log compression (matching librosa's power_to_db)
      bandEnergies[f] = 10 * Math.log10(Math.max(energy, 1e-10));
    }
    return bandEnergies;
  });

  // 5. Apply DCT to log-Mel spectrogram to get MFCCs [N_MFCC × nFrames]
  const mfccFrames: Float32Array[] = frames.map((_, f) => {
    const logMelFrame = new Float32Array(N_MELS);
    for (let m = 0; m < N_MELS; m++) {
      logMelFrame[m] = melSpec[m][f];
    }
    return dct2(logMelFrame, N_MFCC);
  });

  // 6. Compute per-coefficient statistics for MFCCs [N_MFCC × 4]
  const mfccMeans = new Float32Array(N_MFCC);
  const mfccStds = new Float32Array(N_MFCC);
  const mfccMins = new Float32Array(N_MFCC).fill(Infinity);
  const mfccMaxs = new Float32Array(N_MFCC).fill(-Infinity);

  for (let c = 0; c < N_MFCC; c++) {
    let sum = 0;
    for (let f = 0; f < mfccFrames.length; f++) {
      const v = mfccFrames[f][c];
      sum += v;
      if (v < mfccMins[c]) mfccMins[c] = v;
      if (v > mfccMaxs[c]) mfccMaxs[c] = v;
    }
    mfccMeans[c] = sum / mfccFrames.length;
    let varSum = 0;
    for (let f = 0; f < mfccFrames.length; f++) {
      const diff = mfccFrames[f][c] - mfccMeans[c];
      varSum += diff * diff;
    }
    mfccStds[c] = Math.sqrt(varSum / mfccFrames.length);
  }

  // 7. Compute per-band statistics for Mel-spectrogram [N_MELS × 4]
  const melMeans = new Float32Array(N_MELS);
  const melStds = new Float32Array(N_MELS);
  const melMins = new Float32Array(N_MELS).fill(Infinity);
  const melMaxs = new Float32Array(N_MELS).fill(-Infinity);

  for (let m = 0; m < N_MELS; m++) {
    let sum = 0;
    for (let f = 0; f < frames.length; f++) {
      const v = melSpec[m][f];
      sum += v;
      if (v < melMins[m]) melMins[m] = v;
      if (v > melMaxs[m]) melMaxs[m] = v;
    }
    melMeans[m] = sum / frames.length;
    let varSum = 0;
    for (let f = 0; f < frames.length; f++) {
      const diff = melSpec[m][f] - melMeans[m];
      varSum += diff * diff;
    }
    melStds[m] = Math.sqrt(varSum / frames.length);
  }

  // 8. Spectral centroid [mean, std]
  const freqBinCenters = new Float32Array(nBins);
  for (let k = 0; k < nBins; k++) {
    freqBinCenters[k] = (k * SAMPLE_RATE) / N_FFT;
  }

  const centroidsPerFrame = new Float32Array(frames.length);
  for (let f = 0; f < frames.length; f++) {
    let numerator = 0;
    let denominator = 0;
    for (let k = 0; k < nBins; k++) {
      numerator += freqBinCenters[k] * powerSpectra[f][k];
      denominator += powerSpectra[f][k];
    }
    centroidsPerFrame[f] = denominator > 1e-10 ? numerator / denominator : 0;
  }
  const centroidMean = centroidsPerFrame.reduce((a, b) => a + b, 0) / frames.length;
  let centroidVarSum = 0;
  for (const c of centroidsPerFrame) centroidVarSum += (c - centroidMean) ** 2;
  const centroidStd = Math.sqrt(centroidVarSum / frames.length);

  // 9. Spectral rolloff [mean, std]
  const rolloffsPerFrame = new Float32Array(frames.length);
  for (let f = 0; f < frames.length; f++) {
    let totalEnergy = 0;
    for (let k = 0; k < nBins; k++) totalEnergy += powerSpectra[f][k];
    const threshold = 0.85 * totalEnergy;
    let cumEnergy = 0;
    let rolloffBin = nBins - 1;
    for (let k = 0; k < nBins; k++) {
      cumEnergy += powerSpectra[f][k];
      if (cumEnergy >= threshold) {
        rolloffBin = k;
        break;
      }
    }
    rolloffsPerFrame[f] = (rolloffBin * SAMPLE_RATE) / N_FFT;
  }
  const rolloffMean = rolloffsPerFrame.reduce((a, b) => a + b, 0) / frames.length;
  let rolloffVarSum = 0;
  for (const r of rolloffsPerFrame) rolloffVarSum += (r - rolloffMean) ** 2;
  const rolloffStd = Math.sqrt(rolloffVarSum / frames.length);

  // 10. Zero-crossing rate [mean, std]
  const zcrPerFrame = new Float32Array(frames.length);
  for (let f = 0; f < frames.length; f++) {
    const frame = frames[f];
    let crossings = 0;
    for (let i = 1; i < frame.length; i++) {
      if ((frame[i] >= 0) !== (frame[i - 1] >= 0)) crossings++;
    }
    zcrPerFrame[f] = crossings / (frame.length - 1);
  }
  const zcrMean = zcrPerFrame.reduce((a, b) => a + b, 0) / frames.length;
  let zcrVarSum = 0;
  for (const z of zcrPerFrame) zcrVarSum += (z - zcrMean) ** 2;
  const zcrStd = Math.sqrt(zcrVarSum / frames.length);

  // 11. Assemble final feature vector (326 features)
  const feature = new Float32Array(326);
  let idx = 0;

  // MFCC stats (160)
  for (let c = 0; c < N_MFCC; c++) feature[idx++] = mfccMeans[c];
  for (let c = 0; c < N_MFCC; c++) feature[idx++] = mfccStds[c];
  for (let c = 0; c < N_MFCC; c++) feature[idx++] = mfccMins[c];
  for (let c = 0; c < N_MFCC; c++) feature[idx++] = mfccMaxs[c];

  // Mel-spec stats (160)
  for (let m = 0; m < N_MELS; m++) feature[idx++] = melMeans[m];
  for (let m = 0; m < N_MELS; m++) feature[idx++] = melStds[m];
  for (let m = 0; m < N_MELS; m++) feature[idx++] = melMins[m];
  for (let m = 0; m < N_MELS; m++) feature[idx++] = melMaxs[m];

  // Spectral centroid (2)
  feature[idx++] = centroidMean;
  feature[idx++] = centroidStd;

  // Spectral rolloff (2)
  feature[idx++] = rolloffMean;
  feature[idx++] = rolloffStd;

  // Zero-crossing rate (2)
  feature[idx++] = zcrMean;
  feature[idx++] = zcrStd;

  return feature;
}

// ---------------------------------------------------------------------------
// ONNX model loading
// ---------------------------------------------------------------------------

async function loadModel(): Promise<void> {
  try {
    // Configure ONNX Runtime Web to use WASM backend
    ort.env.wasm.wasmPaths = "/";

    session = await ort.InferenceSession.create("/model.onnx", {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });

    modelLoaded = true;
    self.postMessage({ type: "ready" });
    console.log("[Worker] ONNX model loaded successfully");
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("[Worker] Failed to load ONNX model:", message);
    self.postMessage({
      type: "error",
      message: `Failed to load model: ${message}`,
    });
  }
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

async function runInference(
  pcm: Float32Array,
  sampleRate: number
): Promise<void> {
  if (!session || !modelLoaded) {
    self.postMessage({ type: "error", message: "Model not loaded yet" });
    return;
  }

  const t0 = performance.now();

  try {
    // Extract features matching Python training pipeline
    const features = extractFeatures(pcm, sampleRate);

    // Create ONNX tensor (shape: [1, 326])
    const inputTensor = new ort.Tensor("float32", features, [1, features.length]);

    // Run inference — input name must match what skl2onnx exported
    const feeds: Record<string, ort.Tensor> = {};
    const inputName = session.inputNames[0];
    feeds[inputName] = inputTensor;

    const results = await session.run(feeds);

    // Extract class probabilities
    // skl2onnx with zipmap=False outputs a float tensor named "probabilities"
    const probKey = session.outputNames.find((n) => n.includes("prob")) ?? session.outputNames[1] ?? session.outputNames[0];
    const probTensor = results[probKey];

    let probabilities: Float32Array;
    if (probTensor && probTensor.data instanceof Float32Array) {
      probabilities = probTensor.data;
    } else if (probTensor) {
      // Convert to Float32Array if needed
      probabilities = new Float32Array(probTensor.data as ArrayLike<number>);
    } else {
      // Fallback: uniform distribution
      probabilities = new Float32Array(N_CLASSES).fill(1 / N_CLASSES);
    }

    const inferenceTimeMs = performance.now() - t0;

    self.postMessage(
      {
        type: "result",
        probabilities,
        inferenceTimeMs,
      },
      // Transfer ownership of the buffer to avoid copying
      [probabilities.buffer]
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("[Worker] Inference error:", message);
    self.postMessage({ type: "error", message });
  }
}

// ---------------------------------------------------------------------------
// Message handler
// ---------------------------------------------------------------------------

self.onmessage = async (event: MessageEvent) => {
  const { type } = event.data;

  if (type === "init") {
    await loadModel();
  } else if (type === "infer") {
    const { pcm, sampleRate } = event.data as {
      pcm: Float32Array;
      sampleRate: number;
    };
    await runInference(pcm, sampleRate);
  }
};

// Auto-initialize on load
loadModel();

export {};
