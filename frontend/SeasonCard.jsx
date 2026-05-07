const POS_STYLES = {
  QB: { bg: 'rgba(59,130,246,0.15)',  color: '#60a5fa' },
  RB: { bg: 'rgba(34,197,94,0.15)',   color: '#4ade80' },
  WR: { bg: 'rgba(249,115,22,0.15)',  color: '#fb923c' },
  TE: { bg: 'rgba(168,85,247,0.15)',  color: '#c084fc' },
}

function errorColor(diff, mae) {
  const abs = Math.abs(diff)
  if (abs <= mae)     return '#4ade80'
  if (abs <= mae * 2) return '#fbbf24'
  return '#f87171'
}

function AccuracyBadge({ weeks, mae }) {
  const played = weeks.filter(w => w.actual_ppr != null)
  if (played.length === 0) return null
  const withinMae = played.filter(w => Math.abs(w.actual_ppr - w.predicted_ppr) <= mae).length
  const pct = Math.round((withinMae / played.length) * 100)
  return (
    <span style={{
      fontSize:      '0.7rem',
      fontFamily:    "'Barlow Condensed', sans-serif",
      fontWeight:    700,
      letterSpacing: '0.1em',
      color:         pct >= 50 ? '#4ade80' : '#fbbf24',
      background:    pct >= 50 ? 'rgba(74,222,128,0.1)' : 'rgba(251,191,36,0.1)',
      border:        `1px solid ${pct >= 50 ? 'rgba(74,222,128,0.3)' : 'rgba(251,191,36,0.3)'}`,
      borderRadius:  4,
      padding:       '2px 8px',
    }}>
      {pct}% within MAE
    </span>
  )
}

export default function SeasonCard({ result }) {
  const pos        = POS_STYLES[result.position] ?? { bg: 'rgba(148,163,184,0.15)', color: '#94a3b8' }
  const hasActuals = result.season_total_actual != null
  const totalDiff  = hasActuals ? result.season_total_actual - result.season_total_predicted : null

  const statBox = (label, value, color = 'var(--text)') => (
    <div style={{ textAlign: 'center', background: 'var(--bg)', borderRadius: 8, padding: '12px 8px' }}>
      <p style={{
        fontFamily: "'Barlow Condensed', sans-serif",
        fontWeight: 900,
        fontSize:   'clamp(1.6rem, 5vw, 2.2rem)',
        lineHeight: 1,
        color,
        margin:     0,
      }}>
        {value}
      </p>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.62rem', letterSpacing: '0.18em', textTransform: 'uppercase', margin: '5px 0 0 0' }}>
        {label}
      </p>
    </div>
  )

  return (
    <div
      className="animate-fade-up"
      style={{
        background:   'var(--surface)',
        border:       '1px solid var(--border)',
        borderRadius: 12,
        padding:      '24px',
        marginTop:    8,
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 20 }}>
        <div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.7rem', letterSpacing: '0.2em', textTransform: 'uppercase', margin: '0 0 4px 0' }}>
            2025 Season Projection
          </p>
          <h2 style={{ fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 700, fontSize: '1.4rem', color: 'var(--text)', margin: 0, lineHeight: 1.1 }}>
            {result.player_name}
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', margin: '4px 0 0 0', letterSpacing: '0.06em' }}>
            {result.team} · 2025 Regular Season
          </p>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 6 }}>
          <span style={{
            background:    pos.bg,
            color:         pos.color,
            fontFamily:    "'Barlow Condensed', sans-serif",
            fontWeight:    800,
            fontSize:      '0.85rem',
            letterSpacing: '0.12em',
            padding:       '4px 12px',
            borderRadius:  4,
            border:        `1px solid ${pos.color}44`,
          }}>
            {result.position}
          </span>
          <AccuracyBadge weeks={result.weeks} mae={result.model_mae} />
        </div>
      </div>

      {/* Season totals */}
      <div style={{
        display:             'grid',
        gridTemplateColumns: hasActuals ? '1fr 1fr 1fr' : '1fr 1fr',
        gap:                 10,
        marginBottom:        20,
      }}>
        {statBox('Projected Total', result.season_total_predicted.toFixed(1), 'var(--gold)')}
        {hasActuals && statBox('Actual Total', result.season_total_actual.toFixed(1), '#e6edf3')}
        {hasActuals
          ? statBox('Season Error', `${totalDiff >= 0 ? '+' : ''}${totalDiff.toFixed(1)}`, errorColor(totalDiff, result.model_mae * Math.sqrt(result.weeks.length)))
          : statBox('Avg / Game', (result.season_total_predicted / result.weeks.length).toFixed(1), 'var(--text-muted)')
        }
      </div>

      {/* Week-by-week table */}
      <div style={{ overflowY: 'auto', maxHeight: 340, borderRadius: 8, border: '1px solid var(--border)' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
          <thead>
            <tr style={{
              background:  'var(--bg)',
              position:    'sticky',
              top:         0,
              zIndex:      1,
            }}>
              {['Wk', 'Opponent', 'Projected', ...(hasActuals ? ['Actual', 'Error'] : [])].map(h => (
                <th key={h} style={{
                  padding:       '8px 10px',
                  textAlign:     h === 'Wk' ? 'center' : h === 'Projected' || h === 'Actual' || h === 'Error' ? 'right' : 'left',
                  color:         'var(--text-muted)',
                  fontFamily:    "'Barlow Condensed', sans-serif",
                  fontWeight:    700,
                  fontSize:      '0.65rem',
                  letterSpacing: '0.15em',
                  textTransform: 'uppercase',
                  borderBottom:  '1px solid var(--border)',
                  whiteSpace:    'nowrap',
                }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {result.weeks.map((w, i) => {
              if (w.is_bye) {
                return (
                  <tr key={w.week} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                    <td style={{ padding: '7px 10px', textAlign: 'center', color: 'var(--text-muted)', fontWeight: 600 }}>
                      {w.week}
                    </td>
                    <td colSpan={hasActuals ? 4 : 2} style={{
                      padding:       '7px 10px',
                      fontFamily:    "'Barlow Condensed', sans-serif",
                      fontWeight:    700,
                      fontSize:      '0.75rem',
                      letterSpacing: '0.15em',
                      color:         'var(--text)',
                      textTransform: 'uppercase',
                    }}>
                      BYE WEEK
                    </td>
                  </tr>
                )
              }

              const diff = w.actual_ppr != null ? w.actual_ppr - w.predicted_ppr : null
              return (
                <tr
                  key={w.week}
                  style={{
                    background:   i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.02)',
                    borderBottom: '1px solid rgba(255,255,255,0.04)',
                  }}
                >
                  <td style={{ padding: '7px 10px', textAlign: 'center', color: 'var(--text-muted)', fontWeight: 600 }}>
                    {w.week}
                  </td>
                  <td style={{ padding: '7px 10px', color: 'var(--text)', fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 600, letterSpacing: '0.04em' }}>
                    {w.is_home ? '' : <span style={{ color: 'var(--text-muted)', marginRight: 2 }}>@</span>}
                    {w.opponent}
                  </td>
                  <td style={{ padding: '7px 10px', textAlign: 'right', color: 'var(--gold)', fontWeight: 700 }}>
                    {w.predicted_ppr != null ? w.predicted_ppr.toFixed(1) : '—'}
                  </td>
                  {hasActuals && (
                    <>
                      <td style={{ padding: '7px 10px', textAlign: 'right', fontWeight: w.actual_ppr != null ? 700 : 400 }}>
                        {w.dnp
                          ? <span style={{ color: 'var(--text-muted)', fontSize: '0.72rem', fontStyle: 'italic', letterSpacing: '0.08em' }}>DNP</span>
                          : w.actual_ppr != null
                            ? <span style={{ color: '#e6edf3' }}>{w.actual_ppr.toFixed(1)}</span>
                            : <span style={{ color: 'var(--text-muted)' }}>—</span>
                        }
                      </td>
                      <td style={{ padding: '7px 10px', textAlign: 'right', color: diff != null ? errorColor(diff, result.model_mae) : 'var(--text-muted)', fontWeight: 700 }}>
                        {diff != null ? `${diff >= 0 ? '+' : ''}${diff.toFixed(1)}` : '—'}
                      </td>
                    </>
                  )}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Model info */}
      <div style={{ borderTop: '1px solid var(--border)', paddingTop: 12, marginTop: 16, textAlign: 'center' }}>
        <p style={{ color: 'rgba(230,237,243,0.22)', fontSize: '0.68rem', letterSpacing: '0.14em', textTransform: 'uppercase', margin: 0 }}>
          {result.model_used} · ±{result.model_mae.toFixed(1)} pts/game typical error
        </p>
      </div>
    </div>
  )
}
