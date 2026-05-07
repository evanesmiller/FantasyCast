import { useState, useEffect, useRef } from 'react'

export default function PlayerSearch({ onSelect }) {
  const [query,   setQuery]   = useState('')
  const [results, setResults] = useState([])
  const [open,    setOpen]    = useState(false)
  const containerRef  = useRef(null)
  const timerRef      = useRef(null)
  const abortRef      = useRef(null)
  const justSelected  = useRef(false)

  // Close dropdown when clicking outside
  useEffect(() => {
    function onMouseDown(e) {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', onMouseDown)
    return () => document.removeEventListener('mousedown', onMouseDown)
  }, [])

  useEffect(() => {
    clearTimeout(timerRef.current)

    if (justSelected.current) {
      justSelected.current = false
      return
    }

    if (query.length < 2) {
      abortRef.current?.abort()
      setResults([])
      setOpen(false)
      return
    }

    // Short debounce — feels live without hammering the server on every keystroke
    timerRef.current = setTimeout(async () => {
      abortRef.current?.abort()
      abortRef.current = new AbortController()

      try {
        const res  = await fetch(`/players?q=${encodeURIComponent(query)}`, { signal: abortRef.current.signal })
        const data = await res.json()
        const list = data.results ?? []
        setResults(list)
        setOpen(list.length > 0)
      } catch (e) {
        if (e.name !== 'AbortError') {
          setResults([])
          setOpen(false)
        }
      }
    }, 100)

    return () => clearTimeout(timerRef.current)
  }, [query])

  function select(name) {
    justSelected.current = true
    setQuery(name)
    setResults([])
    setOpen(false)
    onSelect(name)
  }

  function handleChange(e) {
    setQuery(e.target.value)
    onSelect(null)
  }

  const inputStyle = {
    background:    'var(--surface)',
    border:        '2px solid var(--border)',
    color:         'var(--text)',
    fontFamily:    "'Barlow Condensed', sans-serif",
    fontWeight:    600,
    fontSize:      '1.1rem',
    letterSpacing: '0.02em',
    width:         '100%',
    padding:       '12px 16px',
    borderRadius:  '8px',
    outline:       'none',
    transition:    'border-color 0.15s',
    boxSizing:     'border-box',
  }

  return (
    <div ref={containerRef} style={{ position: 'relative' }}>
      <input
        type="text"
        value={query}
        onChange={handleChange}
        placeholder="e.g. Justin Jefferson"
        autoComplete="off"
        style={inputStyle}
        onFocus={e => {
          e.target.style.borderColor = 'var(--gold)'
          if (results.length > 0) setOpen(true)
        }}
        onBlur={e => (e.target.style.borderColor = 'var(--border)')}
      />

      {open && (
        <ul style={{
          position:     'absolute',
          left:         0,
          right:        0,
          top:          '100%',
          marginTop:    4,
          zIndex:       50,
          borderRadius: 8,
          overflow:     'hidden',
          background:   'var(--surface-2)',
          border:       '1px solid var(--border)',
          listStyle:    'none',
          padding:      0,
          margin:       '4px 0 0 0',
        }}>
          {results.map((name, i) => (
            <li
              key={name}
              onMouseDown={() => select(name)}
              style={{
                padding:      '11px 16px',
                cursor:       'pointer',
                fontFamily:   "'Barlow Condensed', sans-serif",
                fontWeight:   600,
                fontSize:     '1rem',
                color:        'var(--text)',
                borderBottom: i < results.length - 1 ? '1px solid var(--border)' : 'none',
                transition:   'background 0.1s',
              }}
              onMouseEnter={e => (e.currentTarget.style.background = 'rgba(245,197,66,0.08)')}
              onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
            >
              {name}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
