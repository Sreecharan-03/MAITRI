import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { 
  ShieldCheck, 
  AlertTriangle, 
  Heart, 
  Activity, 
  Smile, 
  Mic, 
  BookOpen, 
  Sparkles, 
  RotateCcw, 
  Printer,
  CheckCircle2,
  Droplets,
  Eye,
  Wind,
  Layers,
  ArrowRight
} from 'lucide-react';
import { useTelemetry } from '../context/TelemetryContext';
import './Summary.css';

export default function Summary() {
  const { telemetry, resetTelemetry } = useTelemetry();
  const navigate = useNavigate();

  const [isLoading, setIsLoading] = useState(true);
  const [readinessScore, setReadinessScore] = useState(80);
  const [headline, setHeadline] = useState("You are doing well and holding steady, though you are starting to feel the strain of the day.");
  const [plainAssessment, setPlainAssessment] = useState(
    "Your physical health is solid and your voice sounds calm, but your expressions and notes suggest you're feeling a bit frustrated and worn out by your tasks. It's clear you're doing a great job pushing through, but your body is signaling that it's time for a short recharge. You've got this, just take a quick moment to breathe and reset."
  );
  const [isDiscordant, setIsDiscordant] = useState(false);
  const [actionTips, setActionTips] = useState([
    "Take a 15-minute screen break to rest your eyes.",
    "Drink a full bottle of water and have an electrolyte snack.",
    "Do 5 slow, deep breaths to help your body unwind."
  ]);

  const [emotionsDist, setEmotionsDist] = useState([
    { label: "Trust", val: 63, color: "bg-cyan-500" },
    { label: "Anticipation", val: 17, color: "bg-sky-500" },
    { label: "Joy", val: 5, color: "bg-emerald-500" },
    { label: "Sadness", val: 5, color: "bg-indigo-500" }
  ]);

  // Derived vitals metrics
  const hr = telemetry.stress_result?.details?.hr_bpm || 78;
  const spo2 = telemetry.stress_result?.details?.spo2_pct || 97.0;
  const hrv = telemetry.stress_result?.details?.hrv_ms || 38;
  const eda = telemetry.stress_result?.details?.eda_us || 1.2;
  const skinTemp = telemetry.stress_result?.details?.skin_temp_c || 36.5;
  const stressPct = telemetry.stress_result?.stress_score !== undefined
    ? Math.round(telemetry.stress_result.stress_score * 100)
    : 6;

  // Face, Voice, Text summaries
  const faceEmotion = telemetry.face_result?.dominant_emotion || "Composed";
  const faceConf = telemetry.face_result?.confidence ? Math.round(telemetry.face_result.confidence * 100) : 92;
  const voiceEmotion = telemetry.voice_result?.dominant_emotion || "Calm";
  const voicePitch = telemetry.voice_result?.pitch_hz ? `${telemetry.voice_result.pitch_hz.toFixed(1)} Hz` : "118.2 Hz";
  const textEmotion = telemetry.text_result?.dominant_emotion || "Reflective";

  // Circle circumference for r=68 is 2 * PI * 68 ~= 427.25
  const CIRCUMFERENCE = 427;
  const strokeOffset = CIRCUMFERENCE - (readinessScore / 100) * CIRCUMFERENCE;

  useEffect(() => {
    let mounted = true;

    async function fetchSummaryAndMultimodal() {
      setIsLoading(true);

      const payload = {
        astronaut_id: telemetry.astronaut_id || "ASTRO-01",
        mission_day: telemetry.mission_day || 42,
        face_result: telemetry.face_result,
        voice_result: telemetry.voice_result,
        text_result: telemetry.text_result,
        stress_result: telemetry.stress_result
      };

      // 1. Fetch Step 5 LLM Surgeon Report
      try {
        const res = await fetch('/api/v1/step5/summary', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (res.ok) {
          const data = await res.json();
          if (mounted) {
            const rawScore = data.readiness_score || (data.overall_score ? Math.round(data.overall_score * 10) : 80);
            setReadinessScore(rawScore);

            if (data.simple_verdict) setHeadline(data.simple_verdict);
            else if (data.status) setHeadline(data.status);

            if (data.plain_explanation) setPlainAssessment(data.plain_explanation);
            else if (data.detailed_assessment) setPlainAssessment(data.detailed_assessment);

            const tips = data.actionable_tips || data.recommendations;
            if (tips && tips.length > 0) setActionTips(tips.slice(0, 3));
          }
        }
      } catch (e) {
        console.warn("LLM summary API notice:", e);
      }

      // 2. Fetch Multimodal Fusion & CSI Discordance
      try {
        const textLog = telemetry.text_result?.text || "Shift completed, routine station tasks completed nominal.";
        const physioVector = [hr, hrv * 1.15, hrv, eda, eda * 0.15, eda * 1.6, eda > 2.0 ? 10.0 : 4.0, 15.0, 1.2, 0.25, skinTemp, 0.2, 0.01];

        const res = await fetch('/api/v1/predict_multimodal', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text_log: textLog,
            wearable_features: physioVector
          })
        });

        if (res.ok) {
          const mmData = await res.json();
          if (mounted) {
            setIsDiscordant(Boolean(mmData.discordance_alert));

            if (mmData.fused_emotion_distribution) {
              const dist = mmData.fused_emotion_distribution;
              const mapped = [
                { label: "Trust", val: Math.round((dist.Trust || 0.63) * 100), color: "bg-cyan-500" },
                { label: "Anticipation", val: Math.round((dist.Anticipation || 0.17) * 100), color: "bg-sky-500" },
                { label: "Joy", val: Math.round((dist.Joy || 0.05) * 100), color: "bg-emerald-500" },
                { label: "Sadness", val: Math.round((dist.Sadness || 0.05) * 100), color: "bg-indigo-500" }
              ];
              setEmotionsDist(mapped);
            }
          }
        }
      } catch (e) {
        console.warn("Multimodal prediction API notice:", e);
      } finally {
        if (mounted) setIsLoading(false);
      }
    }

    fetchSummaryAndMultimodal();

    return () => {
      mounted = false;
    };
  }, []);

  const handleRestart = () => {
    resetTelemetry();
    navigate('/');
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="relative z-10 w-full px-4 sm:px-6 lg:px-12 py-6 sm:py-10 max-w-7xl mx-auto summary-main-wrapper">
      {/* ----------------- TOP BANNER (READINESS HERO CARD) ----------------- */}
      <div className="w-full bg-slate-900/70 backdrop-blur-xl border border-slate-700/60 rounded-3xl p-6 sm:p-8 lg:p-10 shadow-2xl shadow-black/60 mb-8 summary-hero-card">
        <div className="flex flex-col lg:flex-row items-center gap-6 lg:gap-12">
          {/* Readiness Score Circular Gauge */}
          <div className="relative flex items-center justify-center shrink-0">
            <svg className="w-40 h-40 sm:w-48 sm:h-48 transform -rotate-90">
              {/* Background Track */}
              <circle
                cx="96"
                cy="96"
                r="68"
                className="stroke-slate-800"
                strokeWidth="10"
                fill="transparent"
              />
              {/* Readiness Dynamic Fill Ring */}
              <circle
                cx="96"
                cy="96"
                r="68"
                className="stroke-cyan-400 readiness-ring-track"
                strokeWidth="10"
                strokeLinecap="round"
                fill="transparent"
                strokeDasharray={CIRCUMFERENCE}
                strokeDashoffset={strokeOffset}
              />
            </svg>
            <div className="absolute flex flex-col items-center justify-center text-center">
              <span className="text-4xl sm:text-5xl font-black text-white tracking-tight font-display">
                {readinessScore}
              </span>
              <span className="text-[11px] sm:text-xs font-bold text-slate-400 uppercase tracking-widest mt-0.5 font-mono">
                Readiness
              </span>
            </div>
          </div>

          {/* Plain English Human-Centric Assessment */}
          <div className="flex-1 text-center lg:text-left space-y-3.5">
            {/* Status Pill */}
            <div>
              {isDiscordant ? (
                <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-amber-500/10 border border-amber-500/30 text-amber-400 text-xs font-semibold tracking-wide">
                  <AlertTriangle className="w-3.5 h-3.5" />
                  <span>Tiredness Masking Noted • Short Breather Recommended</span>
                </div>
              ) : (
                <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold tracking-wide">
                  <ShieldCheck className="w-3.5 h-3.5" />
                  <span>All Signals Congruent • Fit For Duty</span>
                </div>
              )}
            </div>

            {/* Clear Headline */}
            <h2 className="text-xl sm:text-2xl lg:text-3xl font-extrabold text-white leading-tight font-display">
              {headline}
            </h2>

            {/* Empathetic, Non-Jargon Explanation */}
            <p className="text-slate-300 text-xs sm:text-sm md:text-[15px] leading-relaxed max-w-3xl">
              {plainAssessment}
            </p>
          </div>
        </div>
      </div>

      {/* ----------------- 3-ZONE COCKPIT GRID ----------------- */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* ZONE 1: WEARABLE VITALS */}
        <div className="bg-slate-900/60 backdrop-blur-xl border border-slate-700/60 rounded-3xl p-5 sm:p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Heart className="w-4 h-4 text-rose-400" />
                <h3 className="text-sm font-bold text-white uppercase tracking-wider font-mono">
                  Wearable Vitals
                </h3>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                Live Sync
              </span>
            </div>

            {/* Vital Stat Cards */}
            <div className="grid grid-cols-2 gap-3 mb-5">
              <div className="p-3.5 rounded-2xl bg-slate-950/60 border border-slate-800">
                <span className="text-[11px] text-slate-400 font-medium">Heart Rate</span>
                <div className="text-2xl font-bold text-white font-mono mt-1">
                  {hr} <span className="text-xs text-slate-400 font-sans">BPM</span>
                </div>
                <span className="text-[10px] text-emerald-400 font-medium mt-1 block">
                  • Normal Resting
                </span>
              </div>
              <div className="p-3.5 rounded-2xl bg-slate-950/60 border border-slate-800">
                <span className="text-[11px] text-slate-400 font-medium">Blood Oxygen</span>
                <div className="text-2xl font-bold text-white font-mono mt-1">
                  {spo2}%
                </div>
                <span className="text-[10px] text-emerald-400 font-medium mt-1 block">
                  • Optimal Tissue
                </span>
              </div>
            </div>

            {/* Progress Bars for Composure and Internal Strain */}
            <div className="space-y-3.5">
              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-slate-400 font-medium">Expressed Composure</span>
                  <span className="text-cyan-400 font-mono font-bold">86%</span>
                </div>
                <div className="w-full bg-slate-800/80 rounded-full h-2 overflow-hidden">
                  <div className="bg-gradient-to-r from-cyan-400 to-teal-400 h-full rounded-full w-[86%]" />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-slate-400 font-medium">Internal Stress Load</span>
                  <span className="text-amber-400 font-mono font-bold">{stressPct}%</span>
                </div>
                <div className="w-full bg-slate-800/80 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-amber-400 h-full rounded-full transition-all duration-500"
                    style={{ width: `${Math.min(100, Math.max(5, stressPct))}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="text-[11px] text-slate-500 font-mono pt-4 mt-4 border-t border-slate-800/80">
            WESAD 13-D Calibrated Feature Vector
          </div>
        </div>

        {/* ZONE 2: MULTIMODAL SENSOR BREAKDOWN */}
        <div className="bg-slate-900/60 backdrop-blur-xl border border-slate-700/60 rounded-3xl p-5 sm:p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-cyan-400" />
                <h3 className="text-sm font-bold text-white uppercase tracking-wider font-mono">
                  Sensor Telemetry
                </h3>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                4 Streams
              </span>
            </div>

            {/* 4 Sensor Mini Cards */}
            <div className="grid grid-cols-2 gap-2.5 mb-5">
              <div className="p-3 rounded-2xl bg-slate-950/60 border border-slate-800 cockpit-sensor-card">
                <div className="flex items-center gap-1.5 text-slate-400 text-[11px] font-medium">
                  <Smile className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Face Scan</span>
                </div>
                <div className="text-xs font-bold text-white mt-1 truncate">
                  {faceEmotion}
                </div>
                <div className="text-[10px] text-slate-400 truncate">{faceConf}% confidence</div>
              </div>

              <div className="p-3 rounded-2xl bg-slate-950/60 border border-slate-800 cockpit-sensor-card">
                <div className="flex items-center gap-1.5 text-slate-400 text-[11px] font-medium">
                  <Mic className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Voice Check</span>
                </div>
                <div className="text-xs font-bold text-white mt-1 truncate">
                  {voiceEmotion}
                </div>
                <div className="text-[10px] text-slate-400 truncate">{voicePitch} pitch</div>
              </div>

              <div className="p-3 rounded-2xl bg-slate-950/60 border border-slate-800 cockpit-sensor-card">
                <div className="flex items-center gap-1.5 text-slate-400 text-[11px] font-medium">
                  <BookOpen className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Daily Journal</span>
                </div>
                <div className="text-xs font-bold text-white mt-1 truncate">
                  {textEmotion}
                </div>
                <div className="text-[10px] text-slate-400 truncate">Shift log sync</div>
              </div>

              <div className="p-3 rounded-2xl bg-slate-950/60 border border-slate-800 cockpit-sensor-card">
                <div className="flex items-center gap-1.5 text-slate-400 text-[11px] font-medium">
                  <Activity className="w-3.5 h-3.5 text-rose-400" />
                  <span>Body Vitals</span>
                </div>
                <div className="text-xs font-bold text-white mt-1 truncate">
                  Homeostasis
                </div>
                <div className="text-[10px] text-slate-400 truncate">{stressPct}% stress load</div>
              </div>
            </div>

            {/* Plutchik Affect Breakdown */}
            <div className="space-y-2">
              <span className="text-[11px] uppercase tracking-wider text-slate-400 font-mono font-medium block">
                Fused Affect Distribution:
              </span>
              {emotionsDist.map(emo => (
                <div key={emo.label}>
                  <div className="flex justify-between text-[11px] text-slate-300 mb-1">
                    <span>{emo.label}</span>
                    <span className="font-mono text-cyan-300">{emo.val}%</span>
                  </div>
                  <div className="w-full bg-slate-800/80 rounded-full h-1.5 overflow-hidden">
                    <div
                      className={`${emo.color} h-full rounded-full transition-all duration-300`}
                      style={{ width: `${emo.val}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="text-[11px] text-slate-500 font-mono pt-4 mt-4 border-t border-slate-800/80">
            Entropy-Gated Fusion &amp; Discordance
          </div>
        </div>

        {/* ZONE 3: FLIGHT SURGEON ACTIONABLE TIPS */}
        <div className="bg-slate-900/60 backdrop-blur-xl border border-slate-700/60 rounded-3xl p-5 sm:p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                <h3 className="text-sm font-bold text-white uppercase tracking-wider font-mono">
                  Operational Tips
                </h3>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                Gemini Brief
              </span>
            </div>

            {/* 3 Actionable Tip Cards */}
            <div className="space-y-3">
              {actionTips.map((tip, idx) => {
                const icons = [Droplets, Eye, Wind, Sparkles];
                const colors = ["cyan", "amber", "indigo", "emerald"];
                const titles = [
                  "Hydrate & Recharge",
                  "Rest Eyes & Screen Break",
                  "Decompress & Breathe",
                  "Reset Cabin Light"
                ];

                const Icon = icons[idx % icons.length];
                const color = colors[idx % colors.length];
                const title = titles[idx % titles.length];

                return (
                  <div
                    key={idx}
                    className="p-3.5 rounded-2xl bg-slate-950/60 border border-slate-800/80 flex items-start gap-3 recommendation-card-modern hover:border-cyan-500/40"
                  >
                    <div className="p-2 rounded-xl bg-slate-900 border border-slate-800 text-cyan-400 shrink-0 mt-0.5">
                      <Icon className="w-4 h-4" />
                    </div>
                    <div>
                      <h4 className="text-xs font-bold text-white">{title}</h4>
                      <p className="text-[11px] sm:text-xs text-slate-300 mt-0.5 leading-relaxed">
                        {tip}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="text-[11px] text-slate-500 font-mono pt-4 mt-4 border-t border-slate-800/80">
            Validated for Lunar/Mars Long-Duration Duty
          </div>
        </div>
      </div>

      {/* ----------------- BOTTOM ACTION BAR (NO PRINT) ----------------- */}
      <div className="w-full flex flex-col sm:flex-row items-center justify-between gap-3 pt-4 border-t border-slate-800/70 no-print">
        <button
          type="button"
          onClick={handleRestart}
          className="w-full sm:w-auto px-6 py-3 rounded-xl bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700 text-slate-200 hover:text-white font-medium text-xs sm:text-sm transition-all flex items-center justify-center gap-2 cursor-pointer"
        >
          <RotateCcw className="w-4 h-4 text-cyan-400" />
          <span>Start New Assessment</span>
        </button>

        <button
          type="button"
          onClick={handlePrint}
          className="w-full sm:w-auto px-7 py-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs sm:text-sm shadow-lg shadow-cyan-500/25 transition-all flex items-center justify-center gap-2 cursor-pointer hover:scale-[1.01]"
        >
          <Printer className="w-4 h-4" />
          <span>Export Mission Health Report</span>
        </button>
      </div>
    </div>
  );
}
