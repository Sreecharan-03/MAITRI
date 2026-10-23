import React from 'react';
import { Link } from 'react-router-dom';
import { 
  ScanFace, 
  Mic2, 
  BookOpenCheck, 
  ActivitySquare, 
  ShieldAlert, 
  ArrowRight, 
  CheckCircle, 
  Cpu
} from 'lucide-react';
import './LandingPage.css';

export default function LandingPage() {
  const steps = [
    {
      step: "01",
      title: "Facial Micro-Expressions",
      tech: "SwinV2 & Facial Strain Index",
      desc: "Live webcam capture evaluates outward composure and muscular tension indicators.",
      icon: ScanFace,
      color: "text-cyan-400",
      border: "hover:border-cyan-500/50"
    },
    {
      step: "02",
      title: "Vocal Prosody",
      tech: "Wav2Vec2 Acoustic Model",
      desc: "15-second speech sample measures fundamental frequency, pitch shift, and acoustic stability.",
      icon: Mic2,
      color: "text-teal-400",
      border: "hover:border-teal-500/50"
    },
    {
      step: "03",
      title: "Crew Operational Log",
      tech: "PAC-Net & ModernBERT",
      desc: "Daily debrief entry detects workload fatigue, sentiment shifts, and cognitive burden.",
      icon: BookOpenCheck,
      color: "text-indigo-400",
      border: "hover:border-indigo-500/50"
    },
    {
      step: "04",
      title: "Physiological Vitals",
      tech: "WESAD Multi-Sensor Model",
      desc: "Continuous heart rate, HRV, electrodermal activity, and skin temperature telemetry.",
      icon: ActivitySquare,
      color: "text-emerald-400",
      border: "hover:border-emerald-500/50"
    }
  ];

  return (
    <div className="relative z-10 w-full flex flex-col justify-between min-h-[calc(100vh-5rem)] px-4 sm:px-6 lg:px-12 py-6 sm:py-10 max-w-7xl mx-auto landing-hero-section">
      {/* Hero Content */}
      <div className="text-center max-w-3xl mx-auto my-auto pt-4 sm:pt-8 pb-8 sm:pb-12">
        {/* Status Pill Badge */}
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-slate-900/80 border border-cyan-500/30 text-cyan-300 text-xs sm:text-[13px] font-medium tracking-wide mb-6 hero-glow-badge backdrop-blur-md">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          <span>Autonomous Mission Health Telemetry</span>
          <span className="text-slate-500">|</span>
          <span className="text-slate-400 font-mono text-[11px]">NASA / ISRO Spec</span>
        </div>

        {/* Main Heading */}
        <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-[54px] font-extrabold tracking-tight hero-title-gradient leading-[1.15] mb-5 sm:mb-6 font-display">
          Crew Health &amp; Stress Monitoring for Long-Duration Spaceflight
        </h1>

        {/* Sub-headline */}
        <p className="text-slate-300 text-sm sm:text-base md:text-lg max-w-2xl mx-auto leading-relaxed mb-8 sm:mb-10 font-sans">
          Multimodal psychological telemetry fusing facial micro-expressions, vocal tone, and wearable biometric sensors into real-time health intelligence for deep space missions.
        </p>

        {/* Call to Action Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-3.5 sm:gap-4 max-w-md mx-auto">
          <Link
            to="/face-scan"
            className="w-full sm:w-auto px-7 py-3.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-sm sm:text-base shadow-lg shadow-cyan-500/25 transition-all duration-200 flex items-center justify-center gap-2 cta-button-glow hover:scale-[1.02] active:scale-[0.98]"
          >
            <span>Begin Readiness Assessment</span>
            <ArrowRight className="w-4 h-4" />
          </Link>
          <Link
            to="/summary"
            className="w-full sm:w-auto px-6 py-3.5 rounded-xl bg-slate-900/70 hover:bg-slate-800/80 border border-slate-700/80 text-slate-200 hover:text-white font-medium text-sm sm:text-base transition-all duration-200 flex items-center justify-center gap-2 backdrop-blur-md"
          >
            <span>View Latest Report</span>
          </Link>
        </div>
      </div>

      {/* 4 Multi-modal Steps Grid (Responsive 1-col mobile, 2-col tablet, 4-col desktop) */}
      <div className="w-full pt-6 border-t border-slate-800/60 mt-auto">
        <div className="flex items-center justify-between mb-4">
          <span className="text-xs uppercase tracking-wider font-semibold text-slate-400 flex items-center gap-2 font-mono">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            Multimodal Telemetry Pipeline
          </span>
          <span className="text-xs text-slate-500 font-mono hidden sm:inline">Entropy-Gated Fusion</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
          {steps.map((st) => {
            const Icon = st.icon;
            return (
              <div
                key={st.step}
                className={`p-4 sm:p-5 rounded-2xl bg-slate-900/50 backdrop-blur-md border border-slate-800/80 mission-feature-card ${st.border} flex flex-col justify-between`}
              >
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-mono font-bold text-slate-500">STAGE {st.step}</span>
                    <Icon className={`w-5 h-5 ${st.color}`} />
                  </div>
                  <h3 className="text-white font-semibold text-sm sm:text-[15px] mb-1">{st.title}</h3>
                  <div className="text-[11px] font-mono text-cyan-400/90 mb-2">{st.tech}</div>
                  <p className="text-slate-400 text-xs leading-relaxed">{st.desc}</p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
