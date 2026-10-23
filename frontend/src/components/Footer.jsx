import React from 'react';
import './Footer.css';

export default function Footer() {
  return (
    <footer className="relative z-10 w-full py-4 sm:py-5 footer-border-glass mt-auto no-print">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-12 flex flex-col sm:flex-row items-center justify-between gap-2 text-[11px] sm:text-xs text-slate-400 font-mono">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
          <span>MAITRI // Long-Duration Mission Telemetry System</span>
        </div>
        <div className="text-slate-500">
          Flight Surgeon Model &amp; Multimodal Entropy Core
        </div>
      </div>
    </footer>
  );
}
