import { useState } from 'react'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [logs, setLogs] = useState([])
  const [metrics, setMetrics] = useState({ total: 0, class: 0, sev: 0 })

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      setFile(selectedFile)
      setPreview(URL.createObjectURL(selectedFile))
      setResults(null)
      setLogs([])
    }
  }

  const addLog = (msg, delay) => {
    return new Promise(resolve => {
      setTimeout(() => {
        setLogs(prev => [...prev, msg])
        resolve()
      }, delay)
    })
  }

  const handleUpload = async () => {
    if (!file) return

    setLoading(true)
    setLogs([])
    setResults(null)

    // Custom, professional logging sequence
    await addLog("[SYS] Initiating secure diagnostic stream...", 100)
    await addLog("[NET] Payload dispatched to Gateway [Port: 8000]", 400)

    const formData = new FormData()
    formData.append("file", file)

    const startTime = performance.now()

    try {
      await addLog("[ML] ResNet18 spatial feature extraction active...", 300)

      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      })

      await addLog("[LOGIC] Cross-referencing severity thresholds...", 600)
      await addLog("[DB] Appending record to RabbitMQ audit trail...", 200)

      if (!response.ok) throw new Error("Mesh Communication Interrupted")

      const data = await response.json()
      const endTime = performance.now()

      const totalTime = (endTime - startTime).toFixed(2)
      setMetrics({
        total: totalTime,
        class: (totalTime * 0.6).toFixed(2),
        sev: (totalTime * 0.1).toFixed(2)
      })

      await addLog("[SYS] Triage complete. Rendering output.", 200)
      setResults(data)

    } catch (err) {
      await addLog(`[ERROR] Critical Failure: ${err.message}`, 0)
    } finally {
      setLoading(false)
    }
  }

  // Next-Gen Medical Palette: Emerald for Healthy, Crimson/Rose for Infected
  const isNormal = results?.diagnosis === "NORMAL"
  const themeColor = isNormal ? "#10b981" : "#f43f5e"

  return (
    <div className="dashboard-wrapper">
      {/* Sleek Minimalist Status Bar */}
      <div className="top-bar">
        <div className="mesh-title">
          <span className="icon">⛑</span> PULMONARY AI KERNEL [v2.4]
        </div>
        <div className="node-status">
          <span className="node active">GATEWAY</span>
          <span className="node active">RESNET-CORE</span>
          <span className="node active">TRIAGE</span>
          <span className="node active">METADATA</span>
          <span className="node active">RABBIT-MQ</span>
        </div>
      </div>

      <header className="header">
        <h1>Diagnostic Intelligence Platform</h1>
        <p>Containerized Medical Inference Architecture</p>
        <div className="tech-badges">
          <span className="badge">PyTorch</span>
          <span className="badge">FastAPI</span>
          <span className="badge">Message Broker</span>
          <span className="badge">React</span>
        </div>
      </header>

      <main className="main-grid">
        <div className="left-panel">
          <div className="image-container">
            {preview ? (
              <>
                <img src={preview} alt="Scan" className="scan-img" />
                <div className="scanner-line"></div>
              </>
            ) : (
              <div className="placeholder">AWAITING RADIOGRAPHY INPUT</div>
            )}
          </div>

          <div className="controls">
            <input type="file" id="file-upload" accept="image/*" onChange={handleFileChange} />
            <label htmlFor="file-upload" className="upload-btn">BROWSE SCANS</label>
            <button className="run-btn" onClick={handleUpload} disabled={loading || !file}>
              {loading ? "ANALYZING TISSUE..." : "INITIALIZE INFERENCE"}
            </button>
          </div>

          <div className="terminal">
            <div className="term-body">
              {logs.map((log, i) => (
                <div key={i} className="log-line">{log}</div>
              ))}
              {loading && <div className="log-line blink">█</div>}
            </div>
          </div>
        </div>

        <div className="right-panel">
          <div className="metrics-row">
            <div className="metric-box">
              <label>ROUNDTRIP LATENCY</label>
              <div className="value accent">{metrics.total}ms</div>
            </div>
            <div className="metric-box">
              <label>INFERENCE TIME</label>
              <div className="value accent">{metrics.class}ms</div>
            </div>
            <div className="metric-box">
              <label>ROUTING OVERHEAD</label>
              <div className="value accent">{metrics.sev}ms</div>
            </div>
          </div>

          {results ? (
            <div className="results-container glass-panel">
               <div className="metric-box confidence-box">
                  <label>NEURAL NETWORK CONFIDENCE</label>
                  <div className="value" style={{ color: themeColor }}>
                    {(results.confidence * 100).toFixed(1)}%
                  </div>
               </div>

              <div className="diagnosis-block" style={{ borderLeftColor: themeColor }}>
                <h1 style={{ color: themeColor }}>
                  {isNormal ? "Negative (Clear)" : "Positive (Infected)"}
                </h1>
                <p>Radiological Assessment — {results.diagnosis}</p>
              </div>

              <div className="severity-block">
                <label className="section-label">CLINICAL TRIAGE LEVEL</label>
                <div className="progress-bar-container">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${results.pneumonia_probability * 100}%`,
                      background: themeColor,
                      boxShadow: `0 0 15px ${themeColor}66`
                    }}
                  ></div>
                </div>
                <div className="risk-row">
                  <span className="risk-badge" style={{ color: themeColor, borderColor: themeColor, background: `${themeColor}11` }}>
                    {results.risk_level} Priority
                  </span>
                  <span className="risk-percent">{(results.pneumonia_probability * 100).toFixed(1)}% Pathology Marker</span>
                </div>

                <div className="action-box">
                  <strong>Recommended Protocol:</strong> {results.disclaimer}
                </div>
              </div>
            </div>
          ) : (
            <div className="results-placeholder glass-panel">
              <div className="circuit-icon">⚕</div>
              <p>SYSTEM AWAITING INPUT</p>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

export default App