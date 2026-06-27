import { useState, useEffect, useRef } from 'react'
import {
  X, Maximize2, Pin, PinOff, AlertTriangle,
  WifiOff, Loader2, RefreshCw,
} from 'lucide-react'
import { useAppStore } from '../store/appStore'

// ── types ──────────────────────────────────────────────────────────────────────

interface TileState {
  jobId: string
  label: string
  alertCount: number
  streamRevision: number
  imgStatus: 'loading' | 'live' | 'error'
  prevAlertCount: number
}

// ── mock animated feed ─────────────────────────────────────────────────────────

function MockFeed({ label }: { label: string }) {
  const [box, setBox]   = useState({ x: 40, y: 30, dx: 0.6, dy: 0.4 })
  const [box2, setBox2] = useState({ x: 65, y: 50, dx: -0.4, dy: 0.7 })

  useEffect(() => {
    const bounce = (v: { x: number; y: number; dx: number; dy: number }, mx: number, my: number) => {
      let { x, y, dx, dy } = v
      x += dx; y += dy
      if (x < 3 || x > mx) dx = -dx
      if (y < 3 || y > my) dy = -dy
      return { x: Math.max(3, Math.min(mx, x)), y: Math.max(3, Math.min(my, y)), dx, dy }
    }
    const t = setInterval(() => {
      setBox((p)  => bounce(p,  74, 65))
      setBox2((p) => bounce(p,  74, 65))
    }, 40)
    return () => clearInterval(t)
  }, [])

  const boxes = [
    { b: box,  id: 'ID-312', conf: '0.87', color: '#22d3ee' },
    { b: box2, id: 'ID-091', conf: '0.73', color: '#f59e0b' },
  ]

  return (
    <div className="relative w-full h-full bg-gray-900/80 overflow-hidden">
      <div className="absolute inset-0 grid place-items-center pointer-events-none">
        <span className="text-[10px] font-mono text-gray-700 tracking-widest">MOCK · {label}</span>
      </div>
      {boxes.map(({ b, id, conf, color }) => (
        <div
          key={id}
          className="absolute"
          style={{ left: `${b.x}%`, top: `${b.y}%`, width: '16%', height: '24%', border: `1.5px solid ${color}` }}
        >
          <div className="absolute -top-4 left-0 text-[8px] font-bold px-1 whitespace-nowrap text-black" style={{ background: color }}>
            {id} · {conf}
          </div>
        </div>
      ))}
      <div className="absolute top-2 right-2 flex items-center gap-1 bg-black/70 border border-emerald-500/40 text-emerald-400 text-[9px] font-mono px-2 py-0.5 rounded">
        <span className="w-1 h-1 rounded-full bg-emerald-400 animate-pulse" />LIVE
      </div>
    </div>
  )
}

// ── single tile ────────────────────────────────────────────────────────────────

function Tile({
  state, backendBase, isMock,
  isPinned, compact,
  onTogglePin, onReconnect, onFullscreenReq, onStatusChange,
}: {
  state: TileState
  backendBase: string
  isMock: boolean
  isPinned: boolean
  compact: boolean
  onTogglePin: () => void
  onReconnect: () => void
  onFullscreenReq: (el: HTMLDivElement | null) => void
  onStatusChange: (jobId: string, s: TileState['imgStatus']) => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const streamSrc = `${backendBase}/api/v1/stream/job/${state.jobId}`
  const hasAlert = state.alertCount > 0
  const newAlert = state.alertCount > state.prevAlertCount

  return (
    <div
      ref={ref}
      className={`relative bg-black overflow-hidden group h-full ${
        isPinned ? 'ring-2 ring-inset ring-cyan-500/50' : ''
      } ${newAlert ? 'ring-2 ring-inset ring-amber-500/70' : ''}`}
    >
      {/* stream / mock layer */}
      {isMock ? (
        <div className="absolute inset-0">
          <MockFeed label={state.label} />
        </div>
      ) : (
        <>
          <img
            key={state.streamRevision}
            src={streamSrc}
            onLoad={() => onStatusChange(state.jobId, 'live')}
            onError={() => onStatusChange(state.jobId, 'error')}
            className="absolute inset-0 w-full h-full object-contain"
            alt=""
          />
          {state.imgStatus === 'loading' && (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-black/90">
              <Loader2 size={compact ? 16 : 22} className="animate-spin text-cyan-500" />
              <p className="text-[9px] font-mono text-gray-600">Waiting for annotated frames…</p>
            </div>
          )}
          {state.imgStatus === 'error' && (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-black">
              <WifiOff size={compact ? 14 : 20} className="text-gray-600" />
              <p className="text-[9px] font-mono text-gray-600">Stream unavailable</p>
              <button
                onClick={onReconnect}
                className="flex items-center gap-1.5 text-[9px] font-mono px-2.5 py-1.5 bg-gray-800 border border-gray-700 rounded text-gray-400 hover:text-gray-200 transition-colors"
              >
                <RefreshCw size={9} /> Reconnect
              </button>
            </div>
          )}
        </>
      )}

      {/* alert badge */}
      {hasAlert && (
        <div className="absolute top-2 left-2 z-20 flex items-center gap-1 bg-amber-500 text-black text-[8px] font-bold px-1.5 py-0.5 rounded shadow">
          <AlertTriangle size={8} />
          {state.alertCount} ALERT{state.alertCount !== 1 ? 'S' : ''}
        </div>
      )}

      {/* live badge */}
      {(isMock || state.imgStatus === 'live') && (
        <div className="absolute top-2 right-2 z-20 flex items-center gap-1 bg-black/70 border border-emerald-500/40 text-emerald-400 text-[9px] font-mono px-2 py-0.5 rounded">
          <span className="w-1 h-1 rounded-full bg-emerald-400 animate-pulse" />LIVE
        </div>
      )}

      {/* hover controls */}
      <div className="absolute bottom-0 left-0 right-0 z-30 opacity-0 group-hover:opacity-100 transition-opacity duration-200 bg-gradient-to-t from-black/95 via-black/50 to-transparent pt-8 pb-2.5 px-3">
        <div className="flex items-end justify-between gap-2">
          <div className="min-w-0">
            <p className={`font-semibold text-gray-100 truncate leading-tight ${compact ? 'text-[10px]' : 'text-xs'}`}>
              {state.label}
            </p>
            <p className="text-[8px] font-mono text-gray-600 mt-0.5">{state.jobId.slice(0, 12)}…</p>
          </div>
          <div className="flex items-center gap-1.5 flex-shrink-0">
            <button
              onClick={onTogglePin}
              title={isPinned ? 'Unpin' : 'Pin to main view'}
              className="p-1.5 bg-gray-900/80 border border-gray-700 rounded text-gray-400 hover:text-cyan-400 hover:border-cyan-700 transition-colors"
            >
              {isPinned ? <PinOff size={11} /> : <Pin size={11} />}
            </button>
            <button
              onClick={() => onFullscreenReq(ref.current)}
              title="Fullscreen (Esc to exit)"
              className="p-1.5 bg-gray-900/80 border border-gray-700 rounded text-gray-400 hover:text-cyan-400 hover:border-cyan-700 transition-colors"
            >
              <Maximize2 size={11} />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── root component ─────────────────────────────────────────────────────────────

export default function LiveFeed() {
  const { backendUrl, accessMode } = useAppStore()
  const isMock = accessMode === 'MOCK'

  const params = new URLSearchParams(window.location.search)
  const rawJobs   = params.get('jobs') ?? params.get('job') ?? ''
  const rawLabels = params.get('labels') ?? ''
  // Popup windows don't share Zustand state; Operations passes the base URL explicitly
  const passedBase = params.get('base') ? decodeURIComponent(params.get('base')!) : null
  const backendBase = passedBase ?? backendUrl.replace(/\/api\/v1\/?$/, '')

  const jobIds = rawJobs.split(',').filter(Boolean)
  const labelList = rawLabels.split(',').map((l) => decodeURIComponent(l))

  const [tiles, setTiles] = useState<TileState[]>(() =>
    jobIds.map((id, i) => ({
      jobId: id,
      label: labelList[i] ?? `Camera ${i + 1}`,
      alertCount: 0,
      prevAlertCount: 0,
      streamRevision: 0,
      imgStatus: 'loading' as const,
    }))
  )

  const [pinnedId, setPinnedId] = useState<string | null>(null)

  // ── alert count polling (every 10s) ────────────────────────────────────────
  useEffect(() => {
    if (isMock || jobIds.length === 0) return
    const poll = async () => {
      for (const id of jobIds) {
        try {
          const res = await fetch(`${backendBase}/api/v1/videos/processing-status/${id}`, {
            headers: { 'User-Agent': 'EchoFace-Dashboard/1.0', 'ngrok-skip-browser-warning': '1' },
            signal: AbortSignal.timeout ? AbortSignal.timeout(5000) : undefined,
          })
          if (!res.ok) continue
          const data = (await res.json()) as { alerts_created?: number }
          setTiles((prev) =>
            prev.map((t) =>
              t.jobId === id
                ? { ...t, prevAlertCount: t.alertCount, alertCount: data.alerts_created ?? 0 }
                : t
            )
          )
        } catch { /* non-fatal */ }
      }
    }
    poll()
    const timer = setInterval(poll, 10_000)
    return () => clearInterval(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backendBase, isMock, jobIds.length])

  // ── helpers ────────────────────────────────────────────────────────────────
  const reconnect = (jobId: string) =>
    setTiles((prev) =>
      prev.map((t) => t.jobId === jobId ? { ...t, streamRevision: t.streamRevision + 1, imgStatus: 'loading' } : t)
    )

  const updateStatus = (jobId: string, status: TileState['imgStatus']) =>
    setTiles((prev) =>
      prev.map((t) => t.jobId === jobId ? { ...t, imgStatus: status } : t)
    )

  const requestFullscreen = (el: HTMLDivElement | null) => {
    if (el?.requestFullscreen) void el.requestFullscreen()
  }

  const togglePin = (jobId: string) =>
    setPinnedId((prev) => (prev === jobId ? null : jobId))

  // ── derived ────────────────────────────────────────────────────────────────
  const totalAlerts = tiles.reduce((s, t) => s + t.alertCount, 0)
  const count = tiles.length
  const pinnedTile = tiles.find((t) => t.jobId === pinnedId)
  const otherTiles  = tiles.filter((t) => t.jobId !== pinnedId)

  const gridCols =
    count <= 1 ? 'grid-cols-1'
    : count === 2 ? 'grid-cols-2'
    : count <= 4 ? 'grid-cols-2'
    : count <= 6 ? 'grid-cols-3'
    : 'grid-cols-4'

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-gray-300 font-mono select-none overflow-hidden">

      {/* ── header ──────────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-gray-800 flex-shrink-0 z-40">
        <div className="flex items-center gap-3">
          <span className="text-sm font-semibold text-cyan-400 tracking-widest">ECHOFACE</span>
          <span className="text-gray-700">·</span>
          <span className="text-sm text-gray-400">Live Surveillance</span>
          {count > 0 && (
            <span className="text-[10px] font-mono text-gray-600 bg-gray-900 border border-gray-800 px-2 py-0.5 rounded">
              {count} feed{count !== 1 ? 's' : ''}
            </span>
          )}
          {pinnedId && (
            <span className="text-[10px] font-mono text-cyan-600 bg-cyan-500/8 border border-cyan-500/20 px-2 py-0.5 rounded">
              PINNED · {tiles.find((t) => t.jobId === pinnedId)?.label}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {totalAlerts > 0 && (
            <div className="flex items-center gap-1.5 text-[10px] font-mono text-amber-400 bg-amber-400/10 border border-amber-400/30 px-2.5 py-1 rounded">
              <AlertTriangle size={10} />
              {totalAlerts} alert{totalAlerts !== 1 ? 's' : ''}
            </div>
          )}
          {isMock && (
            <span className="text-[10px] font-mono text-violet-400 bg-violet-400/10 border border-violet-400/20 px-2 py-0.5 rounded">
              MOCK
            </span>
          )}
          <button onClick={() => window.close()} className="p-1 text-gray-600 hover:text-gray-300 transition-colors" title="Close">
            <X size={15} />
          </button>
        </div>
      </div>

      {/* ── feed area ───────────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-hidden">
        {count === 0 ? (
          <div className="h-full flex flex-col items-center justify-center gap-4 text-gray-700">
            <Loader2 size={28} className="animate-spin text-cyan-800" />
            <p className="text-sm font-mono">No active feeds</p>
            <p className="text-[11px] font-mono text-gray-800">Start tracking from the Operations page</p>
          </div>

        ) : pinnedId && pinnedTile ? (
          /* Pinned layout: large left + mini stack right */
          <div className="flex h-full">
            <div className="flex-1 min-w-0 min-h-0">
              <Tile
                state={pinnedTile}
                backendBase={backendBase}
                isMock={isMock}
                isPinned
                compact={false}
                onTogglePin={() => togglePin(pinnedTile.jobId)}
                onReconnect={() => reconnect(pinnedTile.jobId)}
                onFullscreenReq={requestFullscreen}
                onStatusChange={updateStatus}
              />
            </div>
            {otherTiles.length > 0 && (
              <div className="w-52 flex-shrink-0 flex flex-col border-l border-gray-900 bg-gray-950">
                {otherTiles.map((tile) => (
                  <div key={tile.jobId} style={{ height: `${100 / otherTiles.length}%` }} className="border-b border-gray-900 last:border-0">
                    <Tile
                      state={tile}
                      backendBase={backendBase}
                      isMock={isMock}
                      isPinned={false}
                      compact
                      onTogglePin={() => togglePin(tile.jobId)}
                      onReconnect={() => reconnect(tile.jobId)}
                      onFullscreenReq={requestFullscreen}
                      onStatusChange={updateStatus}
                    />
                  </div>
                ))}
              </div>
            )}
          </div>

        ) : (
          /* Normal grid */
          <div className={`grid ${gridCols} h-full gap-px bg-gray-900`}>
            {tiles.map((tile) => (
              <div key={tile.jobId} className="bg-black min-h-0">
                <Tile
                  state={tile}
                  backendBase={backendBase}
                  isMock={isMock}
                  isPinned={false}
                  compact={count > 4}
                  onTogglePin={() => togglePin(tile.jobId)}
                  onReconnect={() => reconnect(tile.jobId)}
                  onFullscreenReq={requestFullscreen}
                  onStatusChange={updateStatus}
                />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
