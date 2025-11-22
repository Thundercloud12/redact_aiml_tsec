import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom'
import './App.css'
import { Home } from './components/Home'
import { HowItWorks } from './components/HowItWorks'
import { Features } from './components/Features'
import { About } from './components/About'

function NavBar() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const location = useLocation()

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      // Simulate analysis
      setIsAnalyzing(true)
      setTimeout(() => {
        setIsAnalyzing(false)
      }, 3000)
    }
  }

  const triggerFileUpload = () => {
    const fileInput = document.getElementById('file-upload') as HTMLInputElement
    fileInput?.click()
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

      {/* Upload Status */}
      {selectedFile && (
        <div className="upload-status">
          <div className="upload-info">
            <span className="file-name">📁 {selectedFile.name}</span>
            {isAnalyzing ? (
              <span className="analyzing">🔍 Analyzing...</span>
            ) : (
              <span className="complete">✅ Ready for analysis</span>
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
