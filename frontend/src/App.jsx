import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { TelemetryProvider } from './context/TelemetryContext';
import BackgroundAtmosphere from './components/BackgroundAtmosphere';
import Navbar from './components/Navbar';
import Footer from './components/Footer';

// Pages with their dedicated CSS files
import LandingPage from './pages/LandingPage';
import FaceScan from './pages/FaceScan';
import VoiceCheck from './pages/VoiceCheck';
import DailyLog from './pages/DailyLog';
import HealthVitals from './pages/HealthVitals';
import Summary from './pages/Summary';

export default function App() {
  return (
    <TelemetryProvider>
      <BrowserRouter>
        <div className="min-h-screen bg-[#02040A] text-slate-100 flex flex-col justify-between selection:bg-cyan-500/20 selection:text-cyan-300 relative overflow-x-hidden font-sans">
          {/* Universal Space Background */}
          <BackgroundAtmosphere />

          {/* Navigation Bar */}
          <Navbar />

          {/* Routed Views */}
          <div className="flex-1 flex flex-col justify-center">
            <Routes>
              <Route path="/" element={<LandingPage />} />
              <Route path="/index.html" element={<LandingPage />} />
              <Route path="/face-scan" element={<FaceScan />} />
              <Route path="/face-scan.html" element={<FaceScan />} />
              <Route path="/voice-check" element={<VoiceCheck />} />
              <Route path="/voice-check.html" element={<VoiceCheck />} />
              <Route path="/daily-log" element={<DailyLog />} />
              <Route path="/daily-log.html" element={<DailyLog />} />
              <Route path="/health-vitals" element={<HealthVitals />} />
              <Route path="/health-vitals.html" element={<HealthVitals />} />
              <Route path="/summary" element={<Summary />} />
              <Route path="/summary.html" element={<Summary />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </div>

          {/* Standard Footer */}
          <Footer />
        </div>
      </BrowserRouter>
    </TelemetryProvider>
  );
}
