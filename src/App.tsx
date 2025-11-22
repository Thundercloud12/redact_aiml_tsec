import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom'
import './App.css'
import { Home } from './components/Home'
import { HowItWorks } from './components/HowItWorks'
import { Features } from './components/Features'
import { About } from './components/About'
import { agriGuardAPI, formatFileSize, getFileType } from './services/api'
import type { ImageValidationResult, PlantDiseaseResult } from './services/api'

interface AnalysisState {
  isAnalyzing: boolean;
  validation?: ImageValidationResult;
  diseaseAnalysis?: PlantDiseaseResult;
  error?: string;
}

function NavBar() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [analysis, setAnalysis] = useState<AnalysisState>({ isAnalyzing: false })
  const location = useLocation()

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return;

    setSelectedFile(file)
    setAnalysis({ isAnalyzing: true })

    try {
      // First validate the image
      const validation = await agriGuardAPI.validateImage(file)
      
      if (validation.not_corrupted) {
        // If validation passes, analyze for plant disease
        const diseaseAnalysis = await agriGuardAPI.analyzePlantDisease(file)
        setAnalysis({
          isAnalyzing: false,
          validation,
          diseaseAnalysis
        })
      } else {
        // If validation fails, show validation errors
        setAnalysis({
          isAnalyzing: false,
          validation,
          error: validation.errors.join(', ')
        })
      }
    } catch (error) {
      setAnalysis({
        isAnalyzing: false,
        error: error instanceof Error ? error.message : 'Analysis failed'
      })
    }
  }

  const triggerFileUpload = () => {
    const fileInput = document.getElementById('file-upload') as HTMLInputElement
    fileInput?.click()
  }

  const clearAnalysis = () => {
    setSelectedFile(null)
    setAnalysis({ isAnalyzing: false })
  }

  const isActive = (path: string) => {
    return location.pathname === path ? 'nav-link active' : 'nav-link'
  }

  return (
    <>
      <nav className="navbar">
        <div className="nav-container">
          <Link to="/" className="nav-logo">
            <span className="logo-icon">🍃</span>
            <span className="logo-text">AgriGuard</span>
          </Link>
          <div className="nav-menu">
            <Link to="/" className={isActive('/')}>Home</Link>
            <Link to="/how-it-works" className={isActive('/how-it-works')}>How It Works</Link>
            <Link to="/features" className={isActive('/features')}>Features</Link>
            <Link to="/about" className={isActive('/about')}>About</Link>
            <button className="nav-upload-btn" onClick={triggerFileUpload}>
              📤 Upload Image
            </button>
          </div>
        </div>
      </nav>

      {/* Hidden File Input */}
      <input
        type="file"
        id="file-upload"
        accept="image/*"
        onChange={handleFileUpload}
        style={{ display: 'none' }}
      />

      {/* Analysis Results */}
      {selectedFile && (
        <div className="analysis-overlay">
          <div className="analysis-modal">
            <div className="analysis-header">
              <h3>🌱 AgriGuard Analysis Results</h3>
              <button className="close-btn" onClick={clearAnalysis}>✕</button>
            </div>
            
            <div className="file-info">
              <div className="file-details">
                <span className="file-name">📁 {selectedFile.name}</span>
                <span className="file-meta">{getFileType(selectedFile)} • {formatFileSize(selectedFile.size)}</span>
              </div>
            </div>

            {analysis.isAnalyzing && (
              <div className="analyzing-state">
                <div className="spinner"></div>
                <p>🔬 Analyzing plant health...</p>
                <div className="analysis-steps">
                  <div className="step">✓ Image uploaded</div>
                  <div className="step active">🔍 Validating image integrity</div>
                  <div className="step">🤖 Running AI disease detection</div>
                  <div className="step">📊 Generating recommendations</div>
                </div>
              </div>
            )}

            {analysis.error && (
              <div className="error-state">
                <div className="error-icon">❌</div>
                <h4>Analysis Failed</h4>
                <p>{analysis.error}</p>
                <button className="retry-btn" onClick={triggerFileUpload}>Try Another Image</button>
              </div>
            )}

            {analysis.validation && !analysis.isAnalyzing && !analysis.error && (
              <div className="results-container">
                {/* Image Validation Results */}
                <div className="validation-section">
                  <h4>🔒 Image Validation</h4>
                  <div className="validation-grid">
                    <div className={`validation-item ${analysis.validation.not_corrupted ? 'success' : 'error'}`}>
                      <span className="icon">{analysis.validation.not_corrupted ? '✅' : '❌'}</span>
                      <span>Image Integrity</span>
                    </div>
                    <div className={`validation-item ${analysis.validation.format_valid ? 'success' : 'error'}`}>
                      <span className="icon">{analysis.validation.format_valid ? '✅' : '❌'}</span>
                      <span>Format Valid</span>
                    </div>
                    <div className="validation-item">
                      <span className="icon">🔐</span>
                      <span>SHA-256: {analysis.validation.hash ? analysis.validation.hash.substring(0, 16) + '...' : 'N/A'}</span>
                    </div>
                  </div>
                  {analysis.validation.warnings.length > 0 && (
                    <div className="warnings">
                      <h5>⚠️ Warnings:</h5>
                      {analysis.validation.warnings.map((warning, i) => (
                        <p key={i} className="warning">{warning}</p>
                      ))}
                    </div>
                  )}
                </div>

                {/* Disease Analysis Results */}
                {analysis.diseaseAnalysis && (
                  <div className="disease-section">
                    <h4>🌿 Plant Disease Analysis</h4>
                    <div className="disease-result">
                      {analysis.diseaseAnalysis.disease_detected ? (
                        <>
                          <div className="disease-detected">
                            <div className="disease-header">
                              <span className="status-icon">🚨</span>
                              <div>
                                <h5>Disease Detected: {analysis.diseaseAnalysis.disease_type}</h5>
                                <p>Confidence: {(analysis.diseaseAnalysis.confidence_score * 100).toFixed(1)}%</p>
                              </div>
                              <div className={`severity-badge ${analysis.diseaseAnalysis.severity}`}>
                                {analysis.diseaseAnalysis.severity.toUpperCase()}
                              </div>
                            </div>
                          </div>
                          
                          <div className="treatments">
                            <h5>💊 Recommended Treatments:</h5>
                            <ul>
                              {analysis.diseaseAnalysis.treatment_recommendations.map((treatment, i) => (
                                <li key={i}>{treatment}</li>
                              ))}
                            </ul>
                          </div>
                        </>
                      ) : (
                        <div className="healthy-plant">
                          <span className="status-icon">✅</span>
                          <div>
                            <h5>Healthy Plant Detected</h5>
                            <p>No signs of disease found. Continue regular plant care.</p>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  )
}

function App() {
  return (
    <Router>
      <div className="app">
        <NavBar />
        
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/how-it-works" element={<HowItWorks />} />
            <Route path="/features" element={<Features />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>

        {/* Footer - appears on all pages */}
        <footer className="footer">
          <div className="container">
            <div className="footer-content">
              <div className="team-info">
                <h3>Team AgriGuard</h3>
                <p>Built with ❤️ for agricultural innovation</p>
              </div>
              <div className="footer-links">
                <a href="https://github.com" className="footer-link">
                  📂 GitHub Repository
                </a>
                <div className="hackathon-badge">
                  🏆 Hackathon 2025
                </div>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </Router>
  )
}

export default App
