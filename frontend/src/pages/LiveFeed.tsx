import { useEffect, useState } from 'react'
import { X, Loader2, AlertCircle } from 'lucide-react'
import { useAppStore } from '../store/appStore'

export default function LiveFeed() {
  const { backendUrl } = useAppStore()
  const jobId = new URLSearchParams(window.location.search).get('job')
  const backendBase = backendUrl.replace(/\/api\/v1\/?$/, '')

  const [src, setSrc] = useState<string | null>(null)
  const [tick, setTick] = useState(0)
  const [imgError, setImgError] = useState(false)
  const [imgLoaded, setImgLoaded] = useState(false)

  // Refresh the frame every 2 seconds by updating the cache-busting timestamp
  useEffect(() => {
    if (!jobId) return
    setSrc(`${backendBase}/api/v1/videos/preview-image/${jobId}?t=${Date.now()}`)
    setImgError(false)
    const t = setInterval(() => setTick((n) => n + 1), 2000)
    return () => clearInterval(t)
  }, [jobId, backendBase])

  // On each tick, bump the URL so the browser fetches a fresh frame
  useEffect(() => {
    if (!jobId || tick === 0) return
    setSrc(`${backendBase}/api/v1/videos/preview-image/${jobId}?t=${Date.now()}`)
    setImgError(false)
  }, [tick, jobId, backendBase])

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-gray-300 font-mono select-none">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3.5 border-b border-gray-800 flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="text-sm font-semibold text-cyan-400 tracking-wide">ECHOFACE</div>
          <span className="text-gray-700">·</span>
          <div className="text-sm text-gray-400">Live Tracking Feed</div>
          {jobId && (
            <span className="text-[10px] text-gray-700 font-mono">{jobId.slice(0, 8)}…</span>
          )}
        </div>
        <button
          onClick={() => window.close()}
          className="text-gray-600 hover:text-gray-300 transition-colors"
          title="Close window"
        >
          <X size={16} />
        </button>
      </div>

      {/* Body */}
      <div className="flex-1 flex items-center justify-center overflow-hidden bg-black relative">
        {!jobId && (
          <div className="flex flex-col items-center gap-3 text-gray-600">
            <AlertCircle size={24} />
            <p className="text-sm">No job ID — open this page from Operations</p>
          </div>
        )}

        {jobId && !imgLoaded && !imgError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-gray-600 z-10">
            <Loader2 size={22} className="animate-spin text-cyan-500" />
            <p className="text-xs">Waiting for first annotated frame…</p>
          </div>
        )}

        {jobId && imgError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-gray-600 z-10">
            <AlertCircle size={22} className="text-amber-500" />
            <p className="text-xs">Frame not ready yet — retrying every 2s</p>
          </div>
        )}

        {src && (
          <img
            key={src}
            src={src}
            onLoad={() => { setImgLoaded(true); setImgError(false) }}
            onError={() => setImgError(true)}
            style={{
              width: '100%', height: '100%', objectFit: 'contain',
              // Keep the last good frame visible while the next one loads
              opacity: imgError ? 0 : 1,
            }}
            alt=""
          />
        )}

        {/* Pulse indicator in corner when live */}
        {imgLoaded && !imgError && (
          <div className="absolute top-3 right-3 flex items-center gap-1.5 bg-black/60 px-2 py-1 rounded text-[10px] font-mono text-cyan-400">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
            LIVE
          </div>
        )}
      </div>
    </div>
  )
}
