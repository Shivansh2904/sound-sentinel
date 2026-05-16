import { useEffect, useRef } from "react";

interface WaveformCanvasProps {
  /** Raw time-domain audio data from AnalyserNode.getFloatTimeDomainData() */
  timeDomainData: Float32Array | null;
  isRecording: boolean;
  width?: number;
  height?: number;
}

/**
 * WaveformCanvas
 * ==============
 * Renders a real-time oscilloscope-style waveform visualization of the
 * captured microphone audio using the HTML5 Canvas API.
 *
 * When not recording, shows a flat idle line with a subtle glow.
 */
export function WaveformCanvas({
  timeDomainData,
  isRecording,
  width = 800,
  height = 120,
}: WaveformCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.width / dpr;
      const h = canvas.height / dpr;

      // Clear
      ctx.clearRect(0, 0, w, h);

      // Background
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(0, 0, w, h);

      // Grid lines
      ctx.strokeStyle = "rgba(148, 163, 184, 0.07)";
      ctx.lineWidth = 1;
      for (let i = 1; i < 4; i++) {
        ctx.beginPath();
        ctx.moveTo(0, (h / 4) * i);
        ctx.lineTo(w, (h / 4) * i);
        ctx.stroke();
      }
      // Vertical grid
      for (let i = 1; i < 8; i++) {
        ctx.beginPath();
        ctx.moveTo((w / 8) * i, 0);
        ctx.lineTo((w / 8) * i, h);
        ctx.stroke();
      }

      if (!isRecording || !timeDomainData || timeDomainData.length === 0) {
        // Idle: draw a flat line with subtle glow
        const midY = h / 2;
        ctx.shadowBlur = 6;
        ctx.shadowColor = "rgba(148, 163, 184, 0.3)";
        ctx.strokeStyle = "rgba(148, 163, 184, 0.4)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(0, midY);
        ctx.lineTo(w, midY);
        ctx.stroke();
        ctx.shadowBlur = 0;
        return;
      }

      const data = timeDomainData;
      const sliceWidth = w / data.length;

      // Outer glow pass (wider, lower opacity)
      ctx.shadowBlur = 16;
      ctx.shadowColor = "rgba(56, 189, 248, 0.6)";
      ctx.strokeStyle = "rgba(56, 189, 248, 0.25)";
      ctx.lineWidth = 4;
      ctx.beginPath();
      for (let i = 0; i < data.length; i++) {
        const x = i * sliceWidth;
        const y = (1 - (data[i] + 1) / 2) * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Main waveform line
      ctx.shadowBlur = 8;
      ctx.shadowColor = "rgba(56, 189, 248, 0.9)";
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.beginPath();
      for (let i = 0; i < data.length; i++) {
        const x = i * sliceWidth;
        const y = (1 - (data[i] + 1) / 2) * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Centre line
      ctx.strokeStyle = "rgba(56, 189, 248, 0.15)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(w, h / 2);
      ctx.stroke();
    };

    draw();
  }, [timeDomainData, isRecording]);

  // Handle HiDPI / retina displays
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext("2d");
    if (ctx) ctx.scale(dpr, dpr);
  }, [width, height]);

  return (
    <canvas
      ref={canvasRef}
      className="rounded-lg border border-slate-700/50"
      style={{ width: "100%", height: `${height}px` }}
    />
  );
}
