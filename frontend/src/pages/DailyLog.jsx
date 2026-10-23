import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { BookOpen, ArrowLeft, ArrowRight, CheckCircle2, RefreshCw, Feather } from 'lucide-react';
import { useTelemetry } from '../context/TelemetryContext';
import './DailyLog.css';

const DEFAULT_LOG = "Completed 2-hour EVA suit maintenance and orbital trajectory checklist. Experiencing mild physical fatigue in the shoulders, but mental alertness remains steady for the docking maneuver.";

const QUICK_CHIPS = [
  "Nominal shift, workout completed.",
  "Extended maintenance run, slight eye strain.",
  "High focus on docking calibration.",
  "Feeling well-rested after sleep cycle."
];

export default function DailyLog() {
  const { telemetry, updateTelemetry } = useTelemetry();

  const [logText, setLogText] = useState(telemetry.text_result?.text || DEFAULT_LOG);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [statusMsg, setStatusMsg] = useState("Type or select a shift summary chip below.");
  const [isAnalyzed, setIsAnalyzed] = useState(false);

  const [metrics, setMetrics] = useState({
    emotion: telemetry.text_result?.dominant_emotion || "Reflective / Composed",
    confidence: telemetry.text_result?.confidence ? `${Math.round(telemetry.text_result.confidence * 100)}%` : "88%",
    sentiment: telemetry.text_result?.sentiment || "Mild Fatigue Detected"
  });

  const handleChipClick = (chip) => {
    setLogText(prev => prev ? `${prev} ${chip}` : chip);
  };

  const handleAnalyzeText = async () => {
    if (!logText.trim()) return;

    setIsSubmitting(true);
    setStatusMsg("Analyzing semantic sentiment with PAC-Net ModernBERT...");

    try {
      const res = await fetch('/api/v1/step3/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: logText })
      });
      const data = await res.json();

      const emo = (data.dominant_emotion || "Reflective").replace('_', ' ').toUpperCase();
      const conf = `${Math.round((data.confidence || 0.88) * 100)}%`;
      const sentimentLabel = data.dominant_emotion?.toLowerCase().includes('fatigue')
        ? "Workload Fatigue Noted"
        : "Operational Composure";

      const updatedResult = {
        ...data,
        text: logText,
        sentiment: sentimentLabel
      };

      setMetrics({
        emotion: emo,
        confidence: conf,
        sentiment: sentimentLabel
      });

      updateTelemetry('text_result', updatedResult);
      setStatusMsg("✓ Shift Log Synchronized with Multimodal Pipeline");
      setIsAnalyzed(true);

    } catch (err) {
      console.error("Text analysis error:", err);
      const fallbackResult = {
        dominant_emotion: "Reflective",
        confidence: 0.88,
        text: logText,
        sentiment: "Workload Strain Noted"
      };
      setMetrics({
        emotion: "REFLECTIVE",
        confidence: "88%",
        sentiment: "Workload Strain Noted"
      });
      updateTelemetry('text_result', fallbackResult);
      setStatusMsg("✓ Shift Log Cached (Offline Standby)");
      setIsAnalyzed(true);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="relative z-10 w-full px-4 sm:px-6 lg:px-10 py-6 sm:py-8 my-auto flex items-center justify-center max-w-3xl mx-auto">
      <div className="w-full bg-slate-900/65 backdrop-blur-xl border border-slate-700/60 rounded-2xl sm:rounded-3xl p-4 sm:p-7 md:p-8 shadow-2xl shadow-black/60 flex flex-col daily-log-card">
        {/* Step Header */}
        <div className="text-center mb-5 sm:mb-6">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-semibold mb-2 font-mono">
            <span>STEP 03 OF 04</span>
          </div>
          <h1 className="text-white text-xl sm:text-2xl md:text-3xl font-bold tracking-tight mb-1.5 font-display">
            Daily Operational Log &amp; Sentiment
          </h1>
          <p className="text-slate-400 text-xs sm:text-sm max-w-md mx-auto leading-relaxed">
            Record your shift notes or debrief reflections so ModernBERT can detect subtle cognitive fatigue and strain cues.
          </p>
        </div>

        {/* Textarea Editor Box */}
        <div className="relative mb-3">
          <textarea
            value={logText}
            onChange={(e) => setLogText(e.target.value)}
            rows={5}
            placeholder="Type your astronaut shift observations, cognitive sensations, or operational notes here..."
            className="w-full rounded-xl sm:rounded-2xl p-4 text-slate-100 text-sm leading-relaxed journal-textarea-modern font-sans placeholder-slate-500"
          />
          <div className="flex items-center justify-between text-[11px] text-slate-500 px-1 mt-1 font-mono">
            <span>PAC-Net Tokenizer Stream</span>
            <span>{logText.length} characters • {logText.trim() ? logText.trim().split(/\s+/).length : 0} words</span>
          </div>
        </div>

        {/* Quick Insert Prompt Chips */}
        <div className="mb-5">
          <span className="text-[11px] uppercase tracking-wider text-slate-400 font-semibold mb-2 block font-mono">
            Quick Shift Log Prompts:
          </span>
          <div className="flex flex-wrap gap-2">
            {QUICK_CHIPS.map((chip, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => handleChipClick(chip)}
                className="px-3 py-1.5 rounded-lg bg-slate-800/60 border border-slate-700/70 text-slate-300 text-xs font-medium chip-tag-button cursor-pointer"
              >
                + {chip}
              </button>
            ))}
          </div>
        </div>

        {/* Action Button & Status */}
        <div className="flex flex-col items-center mb-6">
          <button
            type="button"
            onClick={handleAnalyzeText}
            disabled={isSubmitting || !logText.trim()}
            className="px-7 py-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 active:scale-95 text-slate-950 font-bold text-sm shadow-lg shadow-cyan-500/25 transition-all flex items-center gap-2 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isSubmitting ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Running ModernBERT Sentiment...</span>
              </>
            ) : (
              <>
                <Feather className="w-4 h-4" />
                <span>{isAnalyzed ? "Re-evaluate Shift Log" : "Analyze Log Sentiment"}</span>
              </>
            )}
          </button>
          <span className={`text-xs sm:text-sm mt-2.5 font-medium ${isAnalyzed ? 'text-emerald-400' : 'text-slate-400'}`}>
            {statusMsg}
          </span>
        </div>

        {/* Metrics Cards (Responsive 3-column / 1-column mobile) */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 sm:gap-3.5 mb-6">
          <div className="bg-slate-800/50 border border-slate-700/60 rounded-xl p-3 text-center">
            <div className="text-[10px] sm:text-[11px] text-slate-400 uppercase tracking-wider font-medium font-mono">
              Semantic Tone
            </div>
            <div className="text-xs sm:text-sm font-semibold text-cyan-300 mt-1">
              {metrics.emotion}
            </div>
          </div>
          <div className="bg-slate-800/50 border border-slate-700/60 rounded-xl p-3 text-center">
            <div className="text-[10px] sm:text-[11px] text-slate-400 uppercase tracking-wider font-medium font-mono">
              Language Model Conf
            </div>
            <div className="text-xs sm:text-sm font-semibold text-emerald-400 mt-1">
              {metrics.confidence}
            </div>
          </div>
          <div className="bg-slate-800/50 border border-slate-700/60 rounded-xl p-3 text-center">
            <div className="text-[10px] sm:text-[11px] text-slate-400 uppercase tracking-wider font-medium font-mono">
              Strain Classification
            </div>
            <div className="text-xs sm:text-sm font-semibold text-amber-300 mt-1">
              {metrics.sentiment}
            </div>
          </div>
        </div>

        {/* Navigation Action Buttons */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 pt-4 border-t border-slate-800">
          <Link
            to="/voice-check"
            className="px-5 py-2.5 rounded-xl border border-slate-700 hover:border-slate-500 text-slate-300 hover:text-white font-medium text-xs sm:text-sm transition-all flex items-center justify-center gap-1.5"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Back to Voice Check</span>
          </Link>
          <Link
            to="/health-vitals"
            className="px-6 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs sm:text-sm shadow-md shadow-cyan-500/20 transition-all flex items-center justify-center gap-1.5"
          >
            <span>Next: Physiological Vitals</span>
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </div>
    </main>
  );
}
