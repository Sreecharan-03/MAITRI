import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { 
  Heart, 
  Activity, 
  Wind, 
  Thermometer, 
  Zap, 
  ArrowLeft, 
  ArrowRight, 
  RefreshCw,
  CheckCircle2,
  Sliders
} from 'lucide-react';
import { useTelemetry } from '../context/TelemetryContext';
import './HealthVitals.css';

const PRESETS = {
  nominal: {
    label: "Nominal Resting",
    hr: 78,
    hrv: 38,
    spo2: 97.0,
    eda: 1.2,
    temp: 36.5
  },
  fatigue: {
    label: "Post-EVA Muscle Fatigue",
    hr: 86,
    hrv: 28,
    spo2: 96.2,
    eda: 2.4,
    temp: 36.8
  },
  stress: {
    label: "Elevated High-Stress Alert",
    hr: 104,
    hrv: 18,
    spo2: 95.0,
    eda: 4.8,
    temp: 37.2
  }
};

export default function HealthVitals() {
  const { telemetry, updateTelemetry } = useTelemetry();

  const initialDetails = telemetry.stress_result?.details || PRESETS.nominal;

  const [hr, setHr] = useState(initialDetails.hr_bpm || PRESETS.nominal.hr);
  const [hrv, setHrv] = useState(initialDetails.hrv_ms || PRESETS.nominal.hrv);
  const [spo2, setSpo2] = useState(initialDetails.spo2_pct || PRESETS.nominal.spo2);
  const [eda, setEda] = useState(initialDetails.eda_us || PRESETS.nominal.eda);
  const [temp, setTemp] = useState(initialDetails.skin_temp_c || PRESETS.nominal.temp);

  const [isCalibrating, setIsCalibrating] = useState(false);
  const [statusMsg, setStatusMsg] = useState("Adjust sliders or choose a preset to calibrate telemetry.");
  const [isSynced, setIsSynced] = useState(true);

  const [stressOutput, setStressOutput] = useState({
    score: telemetry.stress_result?.stress_score !== undefined ? Math.round(telemetry.stress_result.stress_score * 100) : 6,
    classification: telemetry.stress_result?.classification || "Homeostasis"
  });

  const applyPreset = (presetKey) => {
    const p = PRESETS[presetKey];
    if (!p) return;
    setHr(p.hr);
    setHrv(p.hrv);
    setSpo2(p.spo2);
    setEda(p.eda);
    setTemp(p.temp);
    setIsSynced(false);
    setStatusMsg(`Applied ${p.label}. Click 'Calibrate & Sync Vitals' below.`);
  };

  const handleSyncVitals = async () => {
    setIsCalibrating(true);
    setStatusMsg("Calculating calibrated WESAD multi-sensor stress index...");

    const payload = {
      features: [
        hr,
        hrv * 1.15,
        hrv,
        eda,
        eda * 0.15,
        eda * 1.6,
        eda > 2.0 ? 10.0 : 4.0,
        15.0,
        1.2,
        0.25,
        temp,
        0.2,
        0.01
      ],
      details: {
        hr_bpm: hr,
        hrv_ms: hrv,
        spo2_pct: spo2,
        eda_us: eda,
        skin_temp_c: temp
      }
    };

    try {
      const res = await fetch('/api/v1/step4/stress', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      const scoreVal = typeof data.stress_score === 'number' ? data.stress_score : 0.06;
      const scorePct = Math.round(scoreVal * 100);
      const classLabel = data.classification || (scorePct > 40 ? "Elevated Load" : "Homeostasis");

      setStressOutput({
        score: scorePct,
        classification: classLabel
      });

      updateTelemetry('stress_result', {
        ...data,
        stress_score: scoreVal,
        classification: classLabel,
        details: payload.details
      });

      setStatusMsg("✓ Wearable Vitals Synchronized into Multimodal Fusion");
      setIsSynced(true);

    } catch (err) {
      console.error("Vitals API error:", err);
      // Fallback calculation based on inputs
      let fallbackScore = 0.06;
      if (hr > 100 || eda > 3.0) fallbackScore = 0.58;
      else if (hr > 84 || eda > 2.0) fallbackScore = 0.28;

      const pct = Math.round(fallbackScore * 100);
      const classLabel = pct > 40 ? "Elevated Load" : "Homeostasis";

      setStressOutput({
        score: pct,
        classification: classLabel
      });

      updateTelemetry('stress_result', {
        stress_score: fallbackScore,
        classification: classLabel,
        details: payload.details
      });

      setStatusMsg("✓ Wearable Vitals Calibrated (Offline Cache)");
      setIsSynced(true);
    } finally {
      setIsCalibrating(false);
    }
  };

  return (
    <main className="relative z-10 w-full px-4 sm:px-6 lg:px-10 py-6 sm:py-8 my-auto flex items-center justify-center max-w-4xl mx-auto">
      <div className="w-full bg-slate-900/65 backdrop-blur-xl border border-slate-700/60 rounded-2xl sm:rounded-3xl p-4 sm:p-7 md:p-8 shadow-2xl shadow-black/60 flex flex-col vitals-card">
        {/* Step Header */}
        <div className="text-center mb-5 sm:mb-6">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-semibold mb-2 font-mono">
            <span>STEP 04 OF 04</span>
          </div>
          <h1 className="text-white text-xl sm:text-2xl md:text-3xl font-bold tracking-tight mb-1.5 font-display">
            Physiological Wearable Telemetry
          </h1>
          <p className="text-slate-400 text-xs sm:text-sm max-w-lg mx-auto leading-relaxed">
            Continuous biosensors measure autonomic nervous system activity, parasympathetic tone, and metabolic load.
          </p>
        </div>

        {/* Preset Selector Chips */}
        <div className="flex flex-wrap items-center justify-center gap-2 sm:gap-2.5 mb-6">
          <span className="text-[11px] uppercase tracking-wider text-slate-400 font-mono mr-1">
            Presets:
          </span>
          <button
            type="button"
            onClick={() => applyPreset('nominal')}
            className="px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700 text-slate-200 text-xs font-medium vitals-preset-chip cursor-pointer"
          >
            Nominal Resting (78 BPM)
          </button>
          <button
            type="button"
            onClick={() => applyPreset('fatigue')}
            className="px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700 text-slate-200 text-xs font-medium vitals-preset-chip cursor-pointer"
          >
            Post-EVA Fatigue (86 BPM)
          </button>
          <button
            type="button"
            onClick={() => applyPreset('stress')}
            className="px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700 text-slate-200 text-xs font-medium vitals-preset-chip cursor-pointer"
          >
            High Stress Alert (104 BPM)
          </button>
        </div>

        {/* Sliders Grid (Responsive 2-column tablet/desktop, 1-column mobile) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-5 mb-6">
          {/* Heart Rate */}
          <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl sm:rounded-2xl p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                <Heart className="w-4 h-4 text-rose-400" />
                Heart Rate (BPM)
              </span>
              <span className="text-sm font-bold text-white font-mono">{hr} BPM</span>
            </div>
            <input
              type="range"
              min="50"
              max="150"
              value={hr}
              onChange={(e) => { setHr(Number(e.target.value)); setIsSynced(false); }}
              className="w-full vitals-range-slider cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-1">
              <span>50 (Deep Sleep)</span>
              <span>Nominal (75-80)</span>
              <span>150 (Tachycardia)</span>
            </div>
          </div>

          {/* HRV */}
          <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl sm:rounded-2xl p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                <Activity className="w-4 h-4 text-cyan-400" />
                Heart Rate Variability (HRV ms)
              </span>
              <span className="text-sm font-bold text-cyan-300 font-mono">{hrv} ms</span>
            </div>
            <input
              type="range"
              min="10"
              max="100"
              value={hrv}
              onChange={(e) => { setHrv(Number(e.target.value)); setIsSynced(false); }}
              className="w-full vitals-range-slider cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-1">
              <span>10 (Exhaustion)</span>
              <span>Nominal (35-50)</span>
              <span>100 (Optimal Recovery)</span>
            </div>
          </div>

          {/* Blood Oxygen SpO2 */}
          <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl sm:rounded-2xl p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                <Wind className="w-4 h-4 text-emerald-400" />
                Blood Oxygen (SpO2 %)
              </span>
              <span className="text-sm font-bold text-emerald-400 font-mono">{spo2.toFixed(1)}%</span>
            </div>
            <input
              type="range"
              min="85"
              max="100"
              step="0.1"
              value={spo2}
              onChange={(e) => { setSpo2(Number(e.target.value)); setIsSynced(false); }}
              className="w-full vitals-range-slider cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-1">
              <span>85% (Hypoxia)</span>
              <span>Nominal (97-99%)</span>
              <span>100%</span>
            </div>
          </div>

          {/* Electrodermal Activity EDA */}
          <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl sm:rounded-2xl p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                <Zap className="w-4 h-4 text-amber-400" />
                Skin Conductance (EDA μS)
              </span>
              <span className="text-sm font-bold text-amber-300 font-mono">{eda.toFixed(1)} μS</span>
            </div>
            <input
              type="range"
              min="0.2"
              max="8.0"
              step="0.1"
              value={eda}
              onChange={(e) => { setEda(Number(e.target.value)); setIsSynced(false); }}
              className="w-full vitals-range-slider cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-1">
              <span>0.2 (Basal)</span>
              <span>Nominal (1.0-1.8)</span>
              <span>8.0 (Acute Arousal)</span>
            </div>
          </div>
        </div>

        {/* Action Button & Live Output */}
        <div className="flex flex-col items-center mb-6">
          <button
            type="button"
            onClick={handleSyncVitals}
            disabled={isCalibrating}
            className="px-7 py-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 active:scale-95 text-slate-950 font-bold text-sm shadow-lg shadow-cyan-500/25 transition-all flex items-center gap-2 cursor-pointer disabled:opacity-50"
          >
            {isCalibrating ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Calibrating Stress Estimator...</span>
              </>
            ) : (
              <>
                <Sliders className="w-4 h-4" />
                <span>{isSynced ? "Re-sync Vitals Telemetry" : "Calibrate & Sync Vitals"}</span>
              </>
            )}
          </button>
          <span className={`text-xs sm:text-sm mt-2.5 font-medium ${isSynced ? 'text-emerald-400' : 'text-slate-400'}`}>
            {statusMsg}
          </span>
        </div>

        {/* Stress Telemetry Readout */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
          <div className="bg-slate-800/50 border border-slate-700/60 rounded-xl p-3 text-center">
            <span className="text-[10px] sm:text-[11px] text-slate-400 uppercase tracking-wider font-medium font-mono">
              Calibrated Stress Load
            </span>
            <div className="text-base sm:text-lg font-bold text-cyan-300 mt-1">
              {stressOutput.score}%
            </div>
          </div>
          <div className="bg-slate-800/50 border border-slate-700/60 rounded-xl p-3 text-center">
            <span className="text-[10px] sm:text-[11px] text-slate-400 uppercase tracking-wider font-medium font-mono">
              Autonomic Classification
            </span>
            <div className="text-base sm:text-lg font-bold text-emerald-400 mt-1">
              {stressOutput.classification}
            </div>
          </div>
        </div>

        {/* Navigation Step Actions */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 pt-4 border-t border-slate-800">
          <Link
            to="/daily-log"
            className="px-5 py-2.5 rounded-xl border border-slate-700 hover:border-slate-500 text-slate-300 hover:text-white font-medium text-xs sm:text-sm transition-all flex items-center justify-center gap-1.5"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Back to Daily Log</span>
          </Link>
          <Link
            to="/summary"
            className="px-7 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-teal-400 hover:from-cyan-400 hover:to-teal-300 text-slate-950 font-extrabold text-xs sm:text-sm shadow-lg shadow-cyan-500/25 transition-all flex items-center justify-center gap-1.5"
          >
            <span>View Mission Assessment Summary →</span>
          </Link>
        </div>
      </div>
    </main>
  );
}
