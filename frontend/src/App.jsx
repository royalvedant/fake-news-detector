import React, { useState, useCallback } from 'react';
import Particles from "react-tsparticles";
import { loadSlim } from "tsparticles-slim";
import { motion, AnimatePresence } from "framer-motion";
import { Sun, Moon, Search, AlertTriangle, CheckCircle } from "lucide-react";
import './App.css';

function App() {
  const [url, setUrl] = useState('');
  const [result, setResult] = useState('');
  const [headline, setHeadline] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [warning, setWarning] = useState('');
  const [isDark, setIsDark] = useState(true);
  const [loading, setLoading] = useState(false);

  const particlesInit = useCallback(async (engine) => {
    await loadSlim(engine);
  }, []);

  const handleDetect = async () => {
    if (!url) return;
    setLoading(true);
    setResult('');
    try {
      const response = await fetch('http://localhost:5000/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url })
      });
      const data = await response.json();
      if (data.error) {
        setResult("Error processing URL");
      } else {
        setResult(data.label);
        setHeadline(data.headline);
        setConfidence(data.confidence);
        setWarning(data.warning || '');
      }
    } catch (error) {
      setResult("Error connecting to server");
    }
    setLoading(false);
  };

  const particlesOptions = {
    fullScreen: { enable: false },
    background: { color: "transparent" },
    fpsLimit: 120,
    particles: {
      number: { value: 30, density: { enable: true, area: 800 } },
      color: { value: isDark ? "#ffffff" : "#000000" },
      shape: { type: "square" },
      opacity: {
        value: 0.3,
        random: true,
        anim: { enable: true, speed: 1, opacity_min: 0.1, sync: false }
      },
      size: {
        value: { min: 2, max: 10 },
        random: true
      },
      move: {
        enable: true,
        speed: 1.5,
        direction: "none",
        random: true,
        straight: false,
        outModes: { default: "out" }
      }
    },
    detectRetina: true
  };

  return (
    <div className={isDark ? "app-container dark" : "app-container light"}>
      <Particles
        id="tsparticles"
        init={particlesInit}
        options={particlesOptions}
      />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card"
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div />
          <button className="toggle-btn" onClick={() => setIsDark(!isDark)}>
            {isDark ? <Sun size={14} /> : <Moon size={14} />} {isDark ? 'Light' : 'Dark'} Mode
          </button>
        </div>
        <h1>FAKE NEWS DETECTOR</h1>
        <p style={{ color: 'var(--text-muted)', marginBottom: '30px', fontSize: '0.9rem' }}>
          ENTER WEBSITE URL:
        </p>

        <div className="input-group">
          <div style={{ position: 'relative' }}>
            <Search
              size={18}
              style={{ position: 'absolute', left: '15px', top: '18px', color: 'var(--text-muted)' }}
            />
            <input
              type="text"
              placeholder="Paste article URL here..."
              style={{ paddingLeft: '45px' }}
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleDetect()}
            />
          </div>
          <button
            className="detect-btn"
            onClick={handleDetect}
            disabled={loading}
          >
            {loading ? "Analyzing..." : "Detect"}
          </button>
        </div>

        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="result-box"
            >
              <div className="result-label">AI VERDICT</div>
              <div className={`result-verdict verdict-${result}`}>
                {result === "REAL" ? (
                  <><CheckCircle size={24} style={{ verticalAlign: 'middle', marginRight: '10px' }} /> REAL</>
                ) : (
                  <><AlertTriangle size={24} style={{ verticalAlign: 'middle', marginRight: '10px' }} /> LIKELY FAKE</>
                )}
              </div>
              
              {headline && (
                <div className="result-headline">
                  <div className="result-label" style={{ marginBottom: '8px' }}>ARTICLE HEADLINE</div>
                  <div className="headline-text">
                    "{headline}"
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}

export default App;
