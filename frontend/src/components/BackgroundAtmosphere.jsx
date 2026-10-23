import React from 'react';

export default function BackgroundAtmosphere() {
  return (
    <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
      {/* Real High-Resolution Space Wallpaper */}
      <div
        className="absolute inset-0 bg-cover bg-bottom bg-no-repeat opacity-100 scale-[1.02]"
        style={{ backgroundImage: "url('/assets/space_wallpaper.jpg')" }}
      />
      {/* Sleek Deep Space Dark Overlays */}
      <div className="absolute inset-0 bg-gradient-to-t from-[#02040A]/85 via-[#02040A]/35 to-[#02040A]/40" />
      <div className="absolute inset-0 bg-gradient-to-b from-[#02040A]/60 via-transparent to-transparent opacity-60" />
      
      {/* Cyan / Indigo Radial Nebula Glow */}
      <div className="absolute top-[28%] left-1/2 -translate-x-1/2 -translate-y-1/2 w-[90vw] max-w-[950px] h-[340px] sm:h-[580px] bg-[radial-gradient(ellipse_at_center,rgba(6,182,212,0.18)_0%,rgba(99,102,241,0.08)_50%,transparent_75%)] blur-3xl" />
      <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-cyan-500/25 to-transparent" />
    </div>
  );
}
