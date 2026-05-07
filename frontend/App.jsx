import { useState } from 'react'
import PlayerSearch from './PlayerSearch'
import ResultCard   from './ResultCard'
import SeasonCard   from './SeasonCard'
import ModelCharts  from './ModelCharts'

export default function App() {
  const [selectedPlayer, setSelectedPlayer] = useState(null)
  const [week,           setWeek]           = useState('')
  const [mode,           setMode]           = useState('game')
  const [result,         setResult]         = useState(null)
  const [seasonResult,   setSeasonResult]   = useState(null)
  const [loading,        setLoading]        = useState(false)
  const [error,          setError]          = useState(null)

  function handleModeChange(newMode) {
    setMode(newMode)
    setResult(null)
    setSeasonResult(null)
    setError(null)
  }

  async function handlePredict() {
    if (!selectedPlayer) return
    setLoading(true)
    setError(null)
    setResult(null)
    setSeasonResult(null)

    try {
      if (mode === 'season') {
        const res = await fetch(`/predict/season?player_name=${encodeURIComponent(selectedPlayer)}`)
        if (!res.ok) {
          const err = await res.json()
          throw new Error(err.detail || 'Season projection failed')
        }
        setSeasonResult(await res.json())
      } else {
        const body = { player_name: selectedPlayer }
        if (week) body.week = parseInt(week)

        const res = await fetch('/predict', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify(body),
        })
        if (!res.ok) {
          const err = await res.json()
          throw new Error(err.detail || 'Prediction failed')
        }
        setResult(await res.json())
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const labelStyle = {
    display:       'block',
    marginBottom:  8,
    fontFamily:    "'Barlow Condensed', sans-serif",
    fontWeight:    700,
    fontSize:      '0.75rem',
    letterSpacing: '0.2em',
    color:         'var(--text-muted)',
    textTransform: 'uppercase',
  }

  const canPredict = selectedPlayer && !loading

  const panelStyle = {
    border:        '1px solid var(--border)',
    borderRadius:  12,
    padding:       24,
    display:       'flex',
    flexDirection: 'column',
    gap:           20,
  }

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>

      {/* Header */}
      <header style={{ padding: '40px 24px 24px' }}>
        <div style={{ maxWidth: 1140, margin: '0 auto' }}>
          <p style={{
            fontFamily:    "'Barlow Condensed', sans-serif",
            fontWeight:    700,
            fontSize:      '0.8rem',
            letterSpacing: '0.25em',
            color:         'var(--gold)',
            textTransform: 'uppercase',
            margin:        '0 0 6px 0',
          }}>
            CPSC 483 · Machine Learning
          </p>
          <h1 style={{
            fontFamily:    "'Barlow Condensed', sans-serif",
            fontWeight:    900,
            fontSize:      'clamp(3rem, 12vw, 5rem)',
            lineHeight:    0.88,
            letterSpacing: '0.01em',
            textTransform: 'uppercase',
            color:         'var(--text)',
            margin:        0,
          }}>
            Fantasy<br />
            <span style={{ color: 'var(--gold)' }}>Cast</span>
          </h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', letterSpacing: '0.04em', margin: '10px 0 0 0' }}>
            PPR · QB · RB · WR · TE
          </p>
        </div>
      </header>

      {/* Gold divider */}
      <div style={{
        height:     1,
        background: 'linear-gradient(90deg, transparent, var(--gold) 20%, var(--gold) 80%, transparent)',
        opacity:    0.2,
        margin:     '0 24px',
      }} />

      {/* Two-column layout */}
      <main style={{ flex: 1, padding: '32px 24px 40px' }}>
        <div style={{
          maxWidth:            1140,
          margin:              '0 auto',
          display:             'grid',
          gridTemplateColumns: '1fr 1fr',
          gap:                 28,
          alignItems:          'stretch',
        }}>

          {/* Left panel — form + results */}
          <div style={panelStyle}>

            {/* Mode toggle */}
            <div>
              <label style={labelStyle}>Mode</label>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, height: 50 }}>
                {[['GAME', 'game'], ['SEASON', 'season']].map(([label, val]) => {
                  const active = mode === val
                  return (
                    <button
                      key={val}
                      onClick={() => handleModeChange(val)}
                      style={{
                        borderRadius:  8,
                        fontFamily:    "'Barlow Condensed', sans-serif",
                        fontWeight:    800,
                        fontSize:      '0.9rem',
                        letterSpacing: '0.15em',
                        cursor:        'pointer',
                        transition:    'all 0.15s',
                        background:    active ? 'var(--gold)' : 'var(--surface)',
                        border:        active ? '2px solid var(--gold)' : '2px solid var(--border)',
                        color:         active ? '#0d1117' : 'var(--text-muted)',
                      }}
                    >
                      {label}
                    </button>
                  )
                })}
              </div>
              {mode === 'season' && (
                <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem', margin: '8px 0 0 0', letterSpacing: '0.04em' }}>
                  Projects all 17 games using 2024 stats · compares vs 2025 actuals
                </p>
              )}
              {mode === 'game' && (
                <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem', margin: '8px 0 0 0', letterSpacing: '0.04em' }}>
                  Enter a 2025 week (1–18) to see prediction vs actual result
                </p>
              )}
            </div>

            {/* Player search */}
            <div>
              <label style={labelStyle}>Player</label>
              <PlayerSearch onSelect={setSelectedPlayer} />
            </div>

            {/* Week — game mode only */}
            {mode === 'game' && (
              <div>
                <label style={labelStyle}>
                  Week <span style={{ opacity: 0.4, fontWeight: 400 }}>(optional)</span>
                </label>
                <input
                  type="number"
                  min="1"
                  max="18"
                  value={week}
                  onChange={e => setWeek(e.target.value)}
                  placeholder="1 – 18"
                  style={{
                    width:         '100%',
                    padding:       '12px 16px',
                    borderRadius:  8,
                    outline:       'none',
                    background:    'var(--surface)',
                    border:        '2px solid var(--border)',
                    color:         'var(--text)',
                    fontFamily:    "'Barlow Condensed', sans-serif",
                    fontWeight:    600,
                    fontSize:      '1.1rem',
                    transition:    'border-color 0.15s',
                    boxSizing:     'border-box',
                  }}
                  onFocus={e => (e.target.style.borderColor = 'var(--gold)')}
                  onBlur={e  => (e.target.style.borderColor = 'var(--border)')}
                />
              </div>
            )}

            {/* Action button */}
            <button
              onClick={handlePredict}
              disabled={!canPredict}
              style={{
                width:         '100%',
                padding:       '16px',
                borderRadius:  8,
                border:        'none',
                fontFamily:    "'Barlow Condensed', sans-serif",
                fontWeight:    900,
                fontSize:      '1.2rem',
                letterSpacing: '0.18em',
                textTransform: 'uppercase',
                cursor:        canPredict ? 'pointer' : 'not-allowed',
                background:    canPredict
                  ? 'linear-gradient(135deg, var(--gold) 0%, var(--gold-dim) 100%)'
                  : 'var(--surface)',
                color:         canPredict ? '#0d1117' : 'rgba(230,237,243,0.2)',
                boxShadow:     canPredict ? '0 4px 20px rgba(245,197,66,0.25)' : 'none',
                transition:    'all 0.15s',
              }}
              onMouseEnter={e => { if (canPredict) e.currentTarget.style.transform = 'translateY(-2px)' }}
              onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)' }}
            >
              {loading ? (
                <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10 }}>
                  <span style={{
                    display:      'inline-block',
                    width:        18,
                    height:       18,
                    border:       '2px solid rgba(13,17,23,0.4)',
                    borderTop:    '2px solid #0d1117',
                    borderRadius: '50%',
                    animation:    'spin 0.7s linear infinite',
                  }} />
                  {mode === 'season' ? 'Projecting...' : 'Predicting...'}
                </span>
              ) : (
                mode === 'season' ? 'Project Season' : 'Predict Points'
              )}
            </button>

            {/* Error */}
            {error && (
              <div
                className="animate-fade-up"
                style={{
                  padding:      '14px 16px',
                  borderRadius: 8,
                  background:   'rgba(212,63,63,0.1)',
                  border:       '1px solid rgba(212,63,63,0.3)',
                  color:        '#e88',
                  fontSize:     '0.9rem',
                }}
              >
                {error}
              </div>
            )}

            {/* Results */}
            {result       && mode === 'game'   && <ResultCard key={result.player_name + result.predicted_ppr_points} result={result} />}
            {seasonResult && mode === 'season' && <SeasonCard key={seasonResult.player_name} result={seasonResult} />}

          </div>

          {/* Right panel — model charts */}
          <div style={{ ...panelStyle, gap: 0 }}>
            <ModelCharts />
          </div>

        </div>
      </main>

      {/* Footer */}
      <footer style={{ padding: '20px 24px', textAlign: 'center', color: 'rgba(230,237,243,0.18)', fontSize: '0.75rem', letterSpacing: '0.08em' }}>
        nfl_data_py · scikit-learn · FastAPI
      </footer>

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}
