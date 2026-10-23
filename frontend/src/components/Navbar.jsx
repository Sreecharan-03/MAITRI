import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { UserCheck } from 'lucide-react';
import './Navbar.css';

export default function Navbar() {
  const location = useLocation();
  const path = location.pathname;

  const getStepInfo = () => {
    switch (path) {
      case '/face-scan':
        return { step: '1 of 4', label: 'Face Scan' };
      case '/voice-check':
        return { step: '2 of 4', label: 'Voice Check' };
      case '/daily-log':
        return { step: '3 of 4', label: 'Daily Log' };
      case '/health-vitals':
        return { step: '4 of 4', label: 'Physio Vitals' };
      case '/summary':
        return { step: 'Complete', label: 'Assessment Summary' };
      default:
        return null;
    }
  };

  const stepInfo = getStepInfo();

  return (
    <header className="relative z-20 w-full px-4 sm:px-8 lg:px-12 h-16 sm:h-20 navbar-glass flex items-center transition-all duration-300">
      <div className="w-full flex items-center justify-between max-w-7xl mx-auto">
        {/* Left: Official MAITRI Brand Logo */}
        <Link to="/" className="flex items-center gap-2 sm:gap-3 group cursor-pointer" aria-label="MAITRI Home">
          <img
            src="/assets/maitri_logo.png"
            alt="MAITRI Logo"
            className="h-8 sm:h-10 md:h-11 w-auto object-contain logo-glow"
          />
        </Link>

        {/* Center: Clean indicator pill if on a step */}
        {stepInfo && (
          <div className="hidden sm:flex items-center gap-2 px-3.5 sm:px-4 py-1.5 rounded-full bg-slate-800/80 border border-slate-700/60 shadow-sm">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
            <span className="text-xs font-medium text-slate-200">
              {stepInfo.step !== 'Complete' ? `Step ${stepInfo.step}: ` : ''}
              <span className="text-cyan-300">{stepInfo.label}</span>
            </span>
          </div>
        )}

        {/* Right: Clean pill badge */}
        <div className="flex items-center gap-3">
          <div className="px-3 sm:px-3.5 py-1.5 rounded-full bg-slate-800/80 border border-slate-700/60 text-[11px] sm:text-xs font-medium text-slate-300 shadow-sm flex items-center gap-1.5">
            <UserCheck className="w-3.5 h-3.5 text-cyan-400" />
            <span className="hidden xs:inline">Astronaut Check-In</span>
            <span className="xs:hidden">ASTRO-01</span>
          </div>
        </div>
      </div>
    </header>
  );
}
