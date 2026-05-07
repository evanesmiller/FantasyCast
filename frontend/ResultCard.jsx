const POS_STYLES = {
  QB: { bg: 'rgba(59,130,246,0.15)',  color: '#60a5fa' },
  RB: { bg: 'rgba(34,197,94,0.15)',   color: '#4ade80' },
  WR: { bg: 'rgba(249,115,22,0.15)',  color: '#fb923c' },
  TE: { bg: 'rgba(168,85,247,0.15)',  color: '#c084fc' },
}

function errorColor(diff, mae) {
  const abs = Math.abs(diff)
  if (abs <= mae)       return '#4ade80'  // within MAE → green
  if (abs <= mae * 2)   return '#fbbf24'  // within 2× MAE → yellow
  return '#f87171'                         // beyond 2× MAE → red
}

export default function ResultCard({ result }) {
  const pos  = POS_STYLES[result.position] ?? { bg: 'rgba(148,163,184,0.15)', color: '#94a3b8' }
  const hasActual = result.actual_ppr_points != null
  const diff      = hasActual ? result.actual_ppr_points - result.predicted_ppr_points : null

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
      {/* Player name + position badge */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 20 }}>
        <div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.7rem', letterSpacing: '0.2em', textTransform: 'uppercase', margin: '0 0 4px 0' }}>
            Player
          </p>
          <h2 style={{ fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 700, fontSize: '1.4rem', color: 'var(--text)', margin: 0, lineHeight: 1.1 }}>
            {result.player_name}
          </h2>
          {result.is_bye ? (
            <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', margin: '4px 0 0 0', letterSpacing: '0.06em' }}>
              BYE WEEK
            </p>
          ) : result.opponent ? (
            <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', margin: '4px 0 0 0', letterSpacing: '0.06em' }}>
              {result.is_home ? 'vs' : '@'} <span style={{ color: 'var(--text)', fontWeight: 600 }}>{result.opponent}</span>
            </p>
          ) : null}
        </div>
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
          flexShrink:    0,
        }}>
          {result.position}
        </span>
      </div>

      {/* Points display */}
      {hasActual ? (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, padding: '16px 0' }}>
          {[
            { label: 'Predicted', value: result.predicted_ppr_points.toFixed(1), color: 'var(--gold)' },
            { label: 'Actual',    value: result.actual_ppr_points.toFixed(1),    color: '#e6edf3' },
            { label: 'Error',     value: `${diff >= 0 ? '+' : ''}${diff.toFixed(1)}`, color: errorColor(diff, result.model_mae) },
          ].map(({ label, value, color }) => (
            <div key={label} style={{ textAlign: 'center' }}>
              <p style={{
                fontFamily: "'Barlow Condensed', sans-serif",
                fontWeight: 900,
                fontSize:   'clamp(1.8rem, 7vw, 2.8rem)',
                lineHeight: 1,
                color,
                margin:     0,
              }}>
                {value}
              </p>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.65rem', letterSpacing: '0.18em', textTransform: 'uppercase', margin: '6px 0 0 0' }}>
                {label}
              </p>
            </div>
          ))}
        </div>
      ) : (
        <div style={{ textAlign: 'center', padding: '16px 0' }}>
          <p style={{
            fontFamily: "'Barlow Condensed', sans-serif",
            fontWeight: 900,
            fontSize:   'clamp(3.5rem, 14vw, 5.5rem)',
            lineHeight: 1,
            color:      'var(--gold)',
            margin:     0,
          }}>
            {result.predicted_ppr_points.toFixed(1)}
          </p>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.72rem', letterSpacing: '0.2em', textTransform: 'uppercase', margin: '8px 0 0 0' }}>
            Predicted PPR Points
          </p>
          <p style={{ color: 'rgba(230,237,243,0.28)', fontSize: '0.8rem', margin: '4px 0 0 0' }}>
            ±{result.model_mae.toFixed(1)} pts typical error
          </p>
        </div>
      )}

      {/* Bye week notice */}
      {result.is_bye && (
        <div style={{
          margin:       '0 0 12px 0',
          padding:      '8px 14px',
          borderRadius: 6,
          background:   'rgba(212,63,63,0.1)',
          border:       '1px solid rgba(212,63,63,0.3)',
          textAlign:    'center',
        }}>
          <p style={{ color: '#f87171', fontSize: '0.75rem', letterSpacing: '0.12em', textTransform: 'uppercase', margin: 0, fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 700 }}>
            Bye Week — No Game Scheduled
          </p>
        </div>
      )}

      {/* DNP notice */}
      {result.dnp && (
        <div style={{
          margin:       '0 0 12px 0',
          padding:      '8px 14px',
          borderRadius: 6,
          background:   'rgba(251,191,36,0.07)',
          border:       '1px solid rgba(251,191,36,0.2)',
          textAlign:    'center',
        }}>
          <p style={{ color: '#fbbf24', fontSize: '0.75rem', letterSpacing: '0.12em', textTransform: 'uppercase', margin: 0, fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 700 }}>
            Player did not play this game
          </p>
        </div>
      )}

      {/* Accuracy label when actual shown */}
      {hasActual && !result.dnp && (
        <p style={{ textAlign: 'center', color: 'rgba(230,237,243,0.35)', fontSize: '0.72rem', letterSpacing: '0.12em', margin: '0 0 12px 0' }}>
          2025 ACTUAL · ±{result.model_mae.toFixed(1)} pts typical error
        </p>
      )}

      {/* Divider + model name */}
      <div style={{ borderTop: '1px solid var(--border)', paddingTop: 12, marginTop: hasActual ? 0 : 12, textAlign: 'center' }}>
        <p style={{ color: 'rgba(230,237,243,0.22)', fontSize: '0.68rem', letterSpacing: '0.14em', textTransform: 'uppercase', margin: 0 }}>
          {result.model_used}
        </p>
      </div>
    </div>
  )
}
