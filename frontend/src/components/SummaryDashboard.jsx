import React from 'react';
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
  Download,
  CheckCircle2,
  Moon,
  Droplets,
  Layers
} from 'lucide-react';

export default function SummaryDashboard({ reportData, onRestart }) {
  // Fallback data when props not fully provided
  const report = {
    readinessScore: reportData?.readiness_score || reportData?.readinessScore || (reportData?.overall_score ? Math.round(reportData.overall_score * 10) : 80),
    statusTitle: reportData?.simple_verdict || reportData?.status || "High focus, but your body is carrying hidden fatigue.",
    statusDesc: reportData?.plain_explanation || reportData?.detailed_assessment || "Physically you are in a great state and your vitals are steady. Your voice and face show composure, but your journal notes indicate that extended tasks are beginning to cause mental wear. A quick 15-minute break will reset your focus.",
    isDiscordant: reportData?.isDiscordant || reportData?.discordance_alert || false,
    csiScore: reportData?.csiScore || 0.142,
    heartRate: reportData?.heartRate || reportData?.heart_rate || 78,
    spO2: reportData?.spO2 || reportData?.spo2 || 97.0,
    expressedComposure: reportData?.expressedComposure || 86,
    internalStrain: reportData?.internalStrain || 6,
    sensors: reportData?.sensors || [
      { name: "Face Scan", state: "Composed", detail: "Mild eye fatigue", icon: Smile, color: "text-emerald-400", border: "border-emerald-500/30" },
      { name: "Voice Check", state: "Calm / Steady", detail: "118.2 Hz pitch", icon: Mic, color: "text-cyan-400", border: "border-cyan-500/30" },
      { name: "Daily Journal", state: "Fatigue Noted", detail: "Workload pressure", icon: BookOpen, color: "text-amber-400", border: "border-amber-500/30" },
      { name: "Body Vitals", state: "Homeostasis", detail: "Low stress (6%)", icon: Activity, color: "text-emerald-400", border: "border-emerald-500/30" },
    ],
    emotions: reportData?.emotions || [
      { label: "Trust", val: 63, color: "bg-cyan-500" },
      { label: "Anticipation", val: 17, color: "bg-sky-500" },
      { label: "Joy", val: 5, color: "bg-emerald-500" },
      { label: "Sadness", val: 5, color: "bg-indigo-500" },
      { label: "Other", val: 10, color: "bg-slate-600" },
    ],
    recommendations: (reportData?.actionable_tips || reportData?.recommendations || [
      "Take a 15-minute screen break to rest your eyes.",
      "Drink a full bottle of water and have an electrolyte snack.",
      "Do 5 slow, deep breaths to help your body unwind."
    ]).map((tip, idx) => {
      const icons = [Droplets, Moon, Sparkles, Activity];
      const colors = ["from-cyan-500/20 to-transparent", "from-indigo-500/20 to-transparent", "from-amber-500/20 to-transparent", "from-emerald-500/20 to-transparent"];
      const borders = ["border-cyan-500/40", "border-indigo-500/40", "border-amber-500/40", "border-emerald-500/40"];
      const iconColors = ["text-cyan-400", "text-indigo-400", "text-amber-400", "text-emerald-400"];
      const titles = ["Hydrate & Recharge", "Rest Eyes & Screen Break", "Decompress & Breathe", "Reset Lighting & Space"];
      return {
        id: idx + 1,
        title: titles[idx % titles.length],
        desc: tip,
        icon: icons[idx % icons.length],
        color: colors[idx % colors.length],
        border: borders[idx % borders.length],
        iconColor: iconColors[idx % iconColors.length]
      };
    })
  };

  return (
    <div className="min-h-screen text-slate-100 p-4 sm:p-6 lg:p-10 font-sans antialiased">
      <div className="max-w-7xl mx-auto space-y-6">

        {/* ----------------- TOP BANNER (HERO STATUS) ----------------- */}
        <section className="relative overflow-hidden rounded-3xl bg-slate-900/60 backdrop-blur-2xl border border-slate-700/50 p-6 lg:p-8 shadow-2xl">
          <div className="absolute top-0 right-0 -mr-20 -mt-20 w-80 h-80 rounded-full bg-cyan-500/10 blur-3xl pointer-events-none" />
          
          <div className="flex flex-col lg:flex-row items-center gap-8">
            {/* Circular Readiness Dial */}
            <div className="relative shrink-0 flex items-center justify-center">
              <svg className="w-40 h-40 transform -rotate-90">
                <circle cx="80" cy="80" r="68" stroke="currentColor" strokeWidth="10" className="text-slate-800" fill="transparent" />
                <circle 
                  cx="80" cy="80" r="68" 
                  stroke="currentColor" 
                  strokeWidth="10" 
                  strokeDasharray={427}
                  strokeDashoffset={427 - (427 * report.readinessScore) / 100}
                  strokeLinecap="round" 
                  className="text-cyan-400 transition-all duration-1000 ease-out" 
                  fill="transparent" 
                />
              </svg>
              <div className="absolute flex flex-col items-center justify-center text-center">
                <span className="text-4xl font-extrabold font-mono tracking-tight text-white">{report.readinessScore}</span>
                <span className="text-[11px] uppercase tracking-widest text-slate-400 font-semibold">Readiness</span>
              </div>
            </div>

            {/* Humanized Doctor's Assessment */}
            <div className="flex-1 text-center lg:text-left space-y-3">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold tracking-wide">
                <CheckCircle2 className="w-3.5 h-3.5" />
                All Signals Congruent • Fit For Duty
              </div>
              <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
                {report.statusTitle}
              </h1>
              <p className="text-sm sm:text-base text-slate-300 leading-relaxed max-w-3xl">
                {report.statusDesc}
              </p>
            </div>
          </div>
        </section>

        {/* ----------------- 3-COLUMN COCKPIT GRID ----------------- */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-12 gap-6">

          {/* CARD 1: Autonomic Vital Signs (Cols: 4) */}
          <div className="lg:col-span-4 rounded-3xl bg-slate-900/60 backdrop-blur-xl border border-slate-700/50 p-6 flex flex-col justify-between space-y-6 shadow-xl">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-cyan-400 font-semibold text-sm tracking-wide">
                <Heart className="w-4 h-4" /> Wearable Vitals
              </div>
              <span className="text-[11px] px-2.5 py-0.5 rounded-full bg-cyan-950/80 border border-cyan-800 text-cyan-300 font-mono">
                Live Sync
              </span>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 rounded-2xl bg-slate-950/40 border border-slate-800">
                <span className="text-xs text-slate-400 font-medium">Heart Rate</span>
                <div className="text-2xl font-bold font-mono text-white mt-1">{report.heartRate} <span className="text-xs text-slate-400 font-sans">BPM</span></div>
                <div className="text-[11px] text-emerald-400 mt-1 flex items-center gap-1">
                  <span>●</span> Normal Resting
                </div>
              </div>
              <div className="p-4 rounded-2xl bg-slate-950/40 border border-slate-800">
                <span className="text-xs text-slate-400 font-medium">Blood Oxygen</span>
                <div className="text-2xl font-bold font-mono text-white mt-1">{report.spO2}%</div>
                <div className="text-[11px] text-emerald-400 mt-1 flex items-center gap-1">
                  <span>●</span> Optimal Tissue
                </div>
              </div>
            </div>

            {/* Expressed vs Internal Strain Gauge */}
            <div className="space-y-3 pt-2 border-t border-slate-800">
              <div className="flex justify-between text-xs font-medium">
                <span className="text-slate-400">Expressed Composure</span>
                <span className="text-cyan-300 font-mono">{report.expressedComposure}%</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2">
                <div className="bg-gradient-to-r from-cyan-500 to-emerald-400 h-2 rounded-full" style={{ width: `${report.expressedComposure}%` }} />
              </div>

              <div className="flex justify-between text-xs font-medium">
                <span className="text-slate-400">Internal Stress Load</span>
                <span className="text-amber-400 font-mono">{report.internalStrain}%</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2">
                <div className="bg-amber-400 h-2 rounded-full" style={{ width: `${report.internalStrain}%` }} />
              </div>
            </div>
          </div>

          {/* CARD 2: Emotional Affect Breakdown (Cols: 4) */}
          <div className="lg:col-span-4 rounded-3xl bg-slate-900/60 backdrop-blur-xl border border-slate-700/50 p-6 flex flex-col justify-between space-y-4 shadow-xl">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-indigo-400 font-semibold text-sm tracking-wide">
                <Layers className="w-4 h-4" /> Mood & Affect Balance
              </div>
              <span className="text-[11px] text-slate-400 font-mono">Plutchik Space</span>
            </div>

            <div className="space-y-3">
              {report.emotions.map((item) => (
                <div key={item.label} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-300">{item.label}</span>
                    <span className="text-slate-400 font-mono">{item.val}%</span>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-1.5">
                    <div className={`${item.color} h-1.5 rounded-full transition-all duration-500`} style={{ width: `${item.val}%` }} />
                  </div>
                </div>
              ))}
            </div>

            <div className="p-3 rounded-2xl bg-slate-950/40 border border-slate-800 text-[11px] text-slate-400 leading-relaxed">
              💡 <span className="text-slate-200 font-medium">Insight:</span> High Trust and Anticipation dominate your state, pointing to high mission focus.
            </div>
          </div>

          {/* CARD 3: 4-Channel Sensor Snapshot (Cols: 4) */}
          <div className="lg:col-span-4 rounded-3xl bg-slate-900/60 backdrop-blur-xl border border-slate-700/50 p-6 flex flex-col justify-between space-y-4 shadow-xl">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400">Sensor Verification</span>
              <span className="text-[11px] text-emerald-400 flex items-center gap-1">● 4/4 Connected</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-1 gap-2.5">
              {report.sensors.map((s) => {
                const Icon = s.icon;
                return (
                  <div key={s.name} className={`flex items-center gap-3 p-3 rounded-2xl bg-slate-950/40 border ${s.border}`}>
                    <div className="p-2 rounded-xl bg-slate-900 shrink-0">
                      <Icon className={`w-4 h-4 ${s.color}`} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-slate-200">{s.name}</span>
                        <span className={`text-[11px] font-medium ${s.color}`}>{s.state}</span>
                      </div>
                      <p className="text-[11px] text-slate-400 truncate">{s.detail}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

        </div>

        {/* ----------------- ACTIONABLE RECOVERY PROTOCOLS ----------------- */}
        <section className="space-y-4">
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-cyan-400" /> Recommended For Your Current Shift
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {report.recommendations.map((rec) => {
              const Icon = rec.icon;
              return (
                <div 
                  key={rec.id} 
                  className={`relative p-5 rounded-3xl bg-gradient-to-b ${rec.color} bg-slate-900/60 backdrop-blur-xl border ${rec.border} flex flex-col justify-between space-y-3 shadow-lg hover:scale-[1.02] transition-transform`}
                >
                  <div className="flex items-start justify-between">
                    <div className="p-2.5 rounded-2xl bg-slate-950/60 border border-slate-800">
                      <Icon className={`w-5 h-5 ${rec.iconColor}`} />
                    </div>
                    <span className="text-xs font-mono font-bold text-slate-500">0{rec.id}</span>
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-white">{rec.title}</h3>
                    <p className="text-xs text-slate-300 mt-1 leading-relaxed">{rec.desc}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* ----------------- FOOTER ACTIONS ----------------- */}
        <div className="flex flex-col sm:flex-row items-center justify-between pt-6 border-t border-slate-800 gap-4">
          <button 
            onClick={onRestart}
            className="w-full sm:w-auto flex items-center justify-center gap-2 px-6 py-3 rounded-2xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 border border-slate-700 transition shadow-md cursor-pointer"
          >
            <RotateCcw className="w-4 h-4" /> Start New Check-in
          </button>

          <button 
            className="w-full sm:w-auto flex items-center justify-center gap-2 px-8 py-3 rounded-2xl bg-gradient-to-r from-cyan-500 to-sky-500 hover:from-cyan-400 hover:to-sky-400 text-slate-950 font-bold text-xs shadow-lg shadow-cyan-500/20 transition hover:shadow-cyan-500/40 cursor-pointer"
          >
            <Download className="w-4 h-4 text-slate-950" /> Download Mission Summary (PDF)
          </button>
        </div>

      </div>
    </div>
  );
}
