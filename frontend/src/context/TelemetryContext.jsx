import React, { createContext, useContext, useState, useEffect } from 'react';

const STORAGE_KEY = 'maitri_telemetry_state';

const initialDefaultState = {
  astronaut_id: "ASTRO-01",
  mission_day: 42,
  face_result: null,
  voice_result: null,
  text_result: null,
  stress_result: {
    stress_score: 0.06,
    classification: "Homeostasis",
    details: {
      hr_bpm: 78,
      hrv_ms: 38,
      spo2_pct: 97.0,
      eda_us: 1.2,
      skin_temp_c: 36.5
    }
  },
  summary_result: null
};

const TelemetryContext = createContext(null);

export function TelemetryProvider({ children }) {
  const [telemetry, setTelemetry] = useState(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        return { ...initialDefaultState, ...JSON.parse(stored) };
      }
    } catch (e) {
      console.warn("Could not load telemetry from localStorage", e);
    }
    return initialDefaultState;
  });

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(telemetry));
    } catch (e) {
      console.warn("Could not save telemetry to localStorage", e);
    }
  }, [telemetry]);

  const updateTelemetry = (key, data) => {
    setTelemetry(prev => ({
      ...prev,
      [key]: data
    }));
  };

  const resetTelemetry = () => {
    setTelemetry(initialDefaultState);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch (e) {}
  };

  return (
    <TelemetryContext.Provider value={{ telemetry, updateTelemetry, resetTelemetry }}>
      {children}
    </TelemetryContext.Provider>
  );
}

export function useTelemetry() {
  const context = useContext(TelemetryContext);
  if (!context) {
    throw new Error("useTelemetry must be used within a TelemetryProvider");
  }
  return context;
}
