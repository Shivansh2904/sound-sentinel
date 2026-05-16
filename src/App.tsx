import { useCallback, useEffect, useRef, useState } from "react";
import { WaveformCanvas } from "./components/WaveformCanvas";
import {
  ESC50_LABELS,
  getCategoryForIndex,
  getColorForIndex,
} from "./constants/labels";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Prediction {
  label: string;
  probability: number;
  classIndex: number;
  category: string;
  color: string;
}

interface WorkerMessage {
  type: "ready" | "result" | "error";
  probabilities?: Float32Array;
  inferenceTimeMs?: number;
  message?: string;
}

// ---------------------------------------------------------------------------
// Audio configuration
// ---------------------------------------------------------------------------

const FFT_SIZE = 4096;           // AnalyserNode FFT size (for waveform vis)
const CAPTURE_INTERVAL_MS = 1000; // Run inference every N ms
const BUFFER_SIZE = 22050 * 2;   // ~2 seconds of audio at 22050 Hz

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

function topK(probabilities: Float32Array, k: number): Prediction[] {
  const indexed = Array.from(probabilities).map((prob, i) => ({ prob, i }));
  indexed.sort((a, b) => b.prob - a.prob);
  return indexed.slice(0, k).map(({ prob, i }) => ({
    label: ESC50_LABELS[i] ?? "Unknown",
    probability: prob,
    classIndex: i,
    category: getCategoryForIndex(i),
    color: getColorForIndex(i),
  }));
}

// ---------------------------------------------------------------------------
// Main App component
// ---------------------------------------------------------------------------

export default function App() {
  // --- State ---
  const [isRecording, setIsRecording] = useState(false);
  const [isModelReady, setIsModelReady] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [inferenceTimeMs, setInferenceTimeMs] = useState<number | null>(null);
  const [timeDomainData, setTimeDomainData] = useState<Float32Array | null>(null);
  const [statusMessage, setStatusMessage] = useState("Loading model...");

  // --- Refs (don't cause re-renders) ---
  const workerRef = useRef<Worker | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const pcmRingBufferRef = useRef<Float32Array>(new Float32Array(BUFFER_SIZE));
  const ringBufferWriteIdxRef = useRef<number>(0);
  const captureIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const animFrameRef = useRef<number>(0);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);

  // --- Worker setup ---
  useEffect(() => {
    const worker = new Worker(
      new URL("./worker/inference.worker.ts", import.meta.url),
      { type: "module" }
    );

    worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
      const { type } = event.data;
      if (type === "ready") {
        setIsModelReady(true);
        setStatusMessage("Model ready — click Start to classify sounds");
      } else if (type === "result") {
        const { probabilities, inferenceTimeMs: ms } = event.data;
        if (probabilities) {
          setPredictions(topK(probabilities, 3));
          setInferenceTimeMs(ms ?? null);
        }
      } else if (type === "error") {
        const msg = event.data.message ?? "Unknown worker error";
        console.error("[App] Worker error:", msg);
        if (msg.includes("model") || msg.includes("onnx")) {
          setModelError(msg);
          setStatusMessage("Model not found — run training/train.py first");
        }
      }
    };

    worker.onerror = (e) => {
      console.error("[App] Worker crashed:", e);
      setModelError("Worker crashed: " + e.message);
    };

    workerRef.current = worker;

    return () => {
      worker.terminate();
    };
  }, []);

  // --- Waveform animation loop ---
  const startWaveformLoop = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) return;

    const bufferLength = analyser.fftSize;
    const dataArray = new Float32Array(bufferLength);

    const tick = () => {
      analyser.getFloatTimeDomainData(dataArray);
      setTimeDomainData(new Float32Array(dataArray)); // copy to avoid mutation
      animFrameRef.current = requestAnimationFrame(tick);
    };

    animFrameRef.current = requestAnimationFrame(tick);
  }, []);

  const stopWaveformLoop = useCallback(() => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = 0;
    }
    setTimeDomainData(null);
  }, []);

  // --- Start recording ---
  const startRecording = useCallback(async () => {
    if (!isModelReady) return;

    try {
      setStatusMessage("Requesting microphone access...");

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          sampleRate: 44100,
        },
      });

      streamRef.current = stream;

      const audioContext = new AudioContext({ sampleRate: 44100 });
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);

      // AnalyserNode for waveform visualization
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = FFT_SIZE;
      analyser.smoothingTimeConstant = 0.6;
      analyserRef.current = analyser;
      source.connect(analyser);

      // ScriptProcessorNode to capture raw PCM samples into ring buffer
      // Note: ScriptProcessorNode is deprecated but remains the most compatible
      // way to get raw PCM in all browsers. AudioWorklet is preferred for new code.
      const bufferSize = 4096;
      const scriptProcessor = audioContext.createScriptProcessor(bufferSize, 1, 1);
      scriptProcessorRef.current = scriptProcessor;

      scriptProcessor.onaudioprocess = (event) => {
        const inputData = event.inputBuffer.getChannelData(0);
        const writeIdx = ringBufferWriteIdxRef.current;
        const buffer = pcmRingBufferRef.current;

        // Write samples into ring buffer, wrapping around
        for (let i = 0; i < inputData.length; i++) {
          buffer[(writeIdx + i) % BUFFER_SIZE] = inputData[i];
        }
        ringBufferWriteIdxRef.current = (writeIdx + inputData.length) % BUFFER_SIZE;
      };

      source.connect(scriptProcessor);
      scriptProcessor.connect(audioContext.destination);

      // Start waveform animation
      startWaveformLoop();

      // Start inference loop
      captureIntervalRef.current = setInterval(() => {
        if (!workerRef.current || !isModelReady) return;

        // Read a contiguous snapshot from the ring buffer
        const snapshot = new Float32Array(BUFFER_SIZE);
        const writeIdx = ringBufferWriteIdxRef.current;
        for (let i = 0; i < BUFFER_SIZE; i++) {
          snapshot[i] = pcmRingBufferRef.current[(writeIdx + i) % BUFFER_SIZE];
        }

        workerRef.current.postMessage(
          {
            type: "infer",
            pcm: snapshot,
            sampleRate: audioContext.sampleRate,
          },
          [snapshot.buffer] // Transfer ownership — avoids copying 88KB each second
        );
      }, CAPTURE_INTERVAL_MS);

      setIsRecording(true);
      setStatusMessage("Recording — listening for environmental sounds...");
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setStatusMessage(`Microphone error: ${message}`);
      console.error("[App] getUserMedia error:", err);
    }
  }, [isModelReady, startWaveformLoop]);

  // --- Stop recording ---
  const stopRecording = useCallback(() => {
    // Stop inference loop
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }

    // Stop waveform animation
    stopWaveformLoop();

    // Disconnect audio nodes
    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect();
      scriptProcessorRef.current = null;
    }
    if (analyserRef.current) {
      analyserRef.current.disconnect();
      analyserRef.current = null;
    }

    // Close audio context
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    // Stop microphone tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    // Reset ring buffer
    pcmRingBufferRef.current = new Float32Array(BUFFER_SIZE);
    ringBufferWriteIdxRef.current = 0;

    setIsRecording(false);
    setStatusMessage("Stopped — click Start to classify sounds");
    setPredictions([]);
    setInferenceTimeMs(null);
  }, [stopWaveformLoop]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (isRecording) stopRecording();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* ------------------------------------------------------------------ */}
      {/* Header                                                              */}
      {/* ------------------------------------------------------------------ */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-sky-500 to-indigo-600 flex items-center justify-center shadow-lg">
              <svg
                className="w-5 h-5 text-white"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"
                />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">SoundSentinel</h1>
              <p className="text-xs text-slate-400">Real-time environmental sound classification</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Model status badge */}
            <div className="flex items-center gap-2 text-xs">
              <div
                className={`w-2 h-2 rounded-full ${
                  modelError
                    ? "bg-red-500"
                    : isModelReady
                    ? "bg-emerald-400"
                    : "bg-amber-400 animate-pulse"
                }`}
              />
              <span className="text-slate-400">
                {modelError ? "Model Error" : isModelReady ? "Model Ready" : "Loading Model..."}
              </span>
            </div>

            <a
              href="https://github.com/shivansh-mishra/sound-sentinel"
              target="_blank"
              rel="noopener noreferrer"
              className="text-slate-400 hover:text-slate-200 transition-colors"
              title="View on GitHub"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
              </svg>
            </a>
          </div>
        </div>
      </header>

      {/* ------------------------------------------------------------------ */}
      {/* Main content                                                        */}
      {/* ------------------------------------------------------------------ */}
      <main className="flex-1 max-w-5xl mx-auto w-full px-6 py-8 flex flex-col gap-6">

        {/* Model error banner */}
        {modelError && (
          <div className="bg-red-950/60 border border-red-700/60 rounded-xl p-4 text-sm text-red-300">
            <p className="font-semibold mb-1">Model not loaded</p>
            <p className="text-red-400/80">
              Run <code className="bg-red-900/50 px-1.5 py-0.5 rounded text-xs font-mono">python training/train.py</code> to
              generate <code className="bg-red-900/50 px-1.5 py-0.5 rounded text-xs font-mono">public/model.onnx</code>, then
              restart the dev server.
            </p>
          </div>
        )}

        {/* Controls row */}
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            {/* Recording status indicator */}
            <div className="flex items-center gap-2">
              <div
                className={`w-3 h-3 rounded-full ${
                  isRecording
                    ? "bg-red-500 animate-pulse shadow-lg shadow-red-500/50"
                    : "bg-slate-600"
                }`}
              />
              <span className="text-sm text-slate-400">{statusMessage}</span>
            </div>
          </div>

          <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={!isModelReady || !!modelError}
            className={`
              flex items-center gap-2.5 px-6 py-3 rounded-xl font-semibold text-sm
              transition-all duration-200 shadow-lg
              disabled:opacity-40 disabled:cursor-not-allowed
              ${
                isRecording
                  ? "bg-red-600 hover:bg-red-500 text-white shadow-red-600/30 hover:shadow-red-500/40"
                  : "bg-sky-600 hover:bg-sky-500 text-white shadow-sky-600/30 hover:shadow-sky-500/40"
              }
            `}
          >
            {isRecording ? (
              <>
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <rect x="6" y="6" width="12" height="12" rx="1" />
                </svg>
                Stop Recording
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <circle cx="12" cy="12" r="6" />
                </svg>
                Start Recording
              </>
            )}
          </button>
        </div>

        {/* Waveform visualization */}
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest">
              Waveform
            </h2>
            {inferenceTimeMs !== null && (
              <span className="text-xs text-slate-500">
                Inference: {inferenceTimeMs.toFixed(1)}ms
              </span>
            )}
          </div>
          <WaveformCanvas
            timeDomainData={timeDomainData}
            isRecording={isRecording}
            height={110}
          />
        </div>

        {/* Predictions panel */}
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-5">
          <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-4">
            Top Predictions
          </h2>

          {predictions.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-10 gap-3 text-slate-600">
              <svg
                className="w-10 h-10 opacity-40"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M19.114 5.636a9 9 0 010 12.728M16.463 8.288a5.25 5.25 0 010 7.424M6.75 8.25l4.72-4.72a.75.75 0 011.28.53v15.88a.75.75 0 01-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.01 9.01 0 012.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75z"
                />
              </svg>
              <p className="text-sm">
                {isModelReady
                  ? "Click Start and make some noise!"
                  : "Loading ONNX model..."}
              </p>
            </div>
          ) : (
            <div className="flex flex-col gap-4">
              {predictions.map((pred, rank) => (
                <PredictionBar key={pred.classIndex} prediction={pred} rank={rank} />
              ))}
            </div>
          )}
        </div>

        {/* Info cards row */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <InfoCard
            icon={
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15m-6.75-6.75h.008v.008h-.008V8.25zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0z" />
              </svg>
            }
            title="50 Sound Classes"
            description="Animals, nature, human sounds, indoor, and urban environments"
          />
          <InfoCard
            icon={
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
              </svg>
            }
            title="~95% Accuracy"
            description="SVM + XGBoost ensemble trained on the ESC-50 benchmark"
          />
          <InfoCard
            icon={
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
              </svg>
            }
            title="100% Private"
            description="Inference runs in-browser via ONNX Runtime Web — audio never leaves your device"
          />
        </div>
      </main>

      {/* ------------------------------------------------------------------ */}
      {/* Footer                                                              */}
      {/* ------------------------------------------------------------------ */}
      <footer className="border-t border-slate-800 bg-slate-900/50 py-4">
        <div className="max-w-5xl mx-auto px-6 flex items-center justify-between text-xs text-slate-500">
          <span>Built by Shivansh Mishra</span>
          <span>ONNX Runtime Web · ESC-50 · scikit-learn · XGBoost</span>
        </div>
      </footer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface PredictionBarProps {
  prediction: Prediction;
  rank: number;
}

function PredictionBar({ prediction, rank }: PredictionBarProps) {
  const pct = (prediction.probability * 100).toFixed(1);
  const widthPct = Math.max(prediction.probability * 100, 2);

  const rankStyles = [
    "text-lg font-bold",
    "text-base font-semibold text-slate-300",
    "text-sm font-medium text-slate-400",
  ];

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <span className="text-xs font-bold text-slate-600 w-4">#{rank + 1}</span>
          <div>
            <span className={rankStyles[rank]}>{prediction.label}</span>
            <span className="ml-2 text-xs text-slate-500">{prediction.category}</span>
          </div>
        </div>
        <span
          className="text-sm font-bold tabular-nums"
          style={{ color: prediction.color }}
        >
          {pct}%
        </span>
      </div>

      {/* Progress bar track */}
      <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500 ease-out"
          style={{
            width: `${widthPct}%`,
            backgroundColor: prediction.color,
            boxShadow: `0 0 8px ${prediction.color}60`,
          }}
        />
      </div>
    </div>
  );
}

interface InfoCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
}

function InfoCard({ icon, title, description }: InfoCardProps) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col gap-2">
      <div className="w-7 h-7 rounded-md bg-sky-500/10 text-sky-400 flex items-center justify-center">
        {icon}
      </div>
      <p className="text-sm font-semibold text-slate-200">{title}</p>
      <p className="text-xs text-slate-500 leading-relaxed">{description}</p>
    </div>
  );
}
