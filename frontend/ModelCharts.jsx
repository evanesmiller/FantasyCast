import { useState } from 'react'

const CHARTS = [
  { file: 'position_accuracy.png', label: 'Per-Position Accuracy' },
  { file: 'model_comparison.png',  label: 'Model Comparison' },
  { file: 'regression_lines.png',  label: 'Predicted vs Actual — Regression Lines' },
]

function ChartImage({ file, label }) {
  const [missing, setMissing] = useState(false)

  if (missing) {
    return (
      <div style={{
        padding:      '24px',
        borderRadius: 8,
        background:   'var(--bg)',
        border:       '1px dashed var(--border)',
        textAlign:    'center',
        color:        'var(--text-muted)',
        fontSize:     '0.85rem',
      }}>
        Chart not found — run <code style={{ color: 'var(--gold)', fontSize: '0.8rem' }}>train_models.py</code> first.
      </div>
    )
  }

  return (
    <img
      src={`/charts/${file}`}
      alt={label}
      onError={() => setMissing(true)}
      style={{
        width:        '100%',
        borderRadius: 8,
        border:       '1px solid var(--border)',
        display:      'block',
      }}
    />
  )
}

export default function ModelCharts() {
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'space-between', gap: 24 }}>
      {CHARTS.map(({ file, label }) => (
        <div key={file}>
          <p style={{
            fontFamily:    "'Barlow Condensed', sans-serif",
            fontWeight:    700,
            fontSize:      '0.72rem',
            letterSpacing: '0.18em',
            textTransform: 'uppercase',
            color:         'var(--text-muted)',
            margin:        '0 0 10px 0',
          }}>
            {label}
          </p>
          <ChartImage file={file} label={label} />
        </div>
      ))}
    </div>
  )
}
