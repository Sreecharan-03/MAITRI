import { resolve } from 'path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        faceScan: resolve(__dirname, 'face-scan.html'),
        voiceCheck: resolve(__dirname, 'voice-check.html'),
        dailyLog: resolve(__dirname, 'daily-log.html'),
        healthVitals: resolve(__dirname, 'health-vitals.html'),
        summary: resolve(__dirname, 'summary.html')
      }
    }
  },
  server: {
    port: 3000,
    host: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      }
    }
  }
});
