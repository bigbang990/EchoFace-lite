import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { X, Radio, Loader2 } from 'lucide-react'
import { useAppStore } from '../store/appStore'

interface JobStatus {
  status: string
  avg_fps: number
  total_faces_detected: number
  alerts_created: number
  processed_frames: number
  total_frames: number
}

export default function LiveFeed() {
  const [params] = useSearchParams()
  const jobId = params.get('job')
  const { backendUrl } = useAppStore()
  const backendBase = backendUrl.replace(/\/api\/v1\/?$/, '')

  const [tick, setTick] = useState(0)
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [imgOk, setImgOk] = useState(false)

  // Cache-bust the preview image every 2 s while job is active
  useEffect(() => {
    if (!jobId) return
    const t = setInterval(() => setTick((v) => v + 1), 2000)
    return () => clearInterval(t)
  }, [jobId])

  // Poll job status every 2 s
  useEffect(() => {
    if (!jobId) return
    let stopped = false
    const poll = async () => {
      try {
        const r = await fetch(`${backendBase}/api/v1/videos/processing-status/${jobId}`)
        if (!r.ok || stopped) return
        const s: JobStatus = await r.json()
        setJobStatus(s)
      } catch {
        // ignore — backend may not be reachable yet
      }
    }
    poll()
    const t = setInterval(poll, 2000)
    return () => {
      stopped = true
      clearInterval(t)
    }
  }, [jobId, backendBase])

  const previewUrl = jobId
    ? `${backendBase}/api/v1/videos/preview-image/${jobId}?t=${tick}`
    : null

  const isProcessing = jobStatus?.status === 'processing'
  const isDone = jobStatus?.status === 'completed' || jobStatus?.status === 'failed'

  const statusLabel = !jobId
    ? 'No active session'
    : !jobStatus
    ? 'Connecting…'
    : isProcessing
    ? 'Processing'
    : jobStatus.status === 'completed'
    ? 'Complete'
    : jobStatus.status === 'failed'
    ? 'Failed'
    : jobStatus.status

  const dotClass = !jobId || !jobStatus
    ? 'bg-gray-700'
    : isProcessing
    ? 'bg-cyan-400 animate-pulse'
    : jobStatus.status === 'completed'
    ? 'bg-emerald-500'
    : 'bg-red-500'

  const labelClass = isProcessing
    ? 'text-cyan-400'
    : isDone && jobStatus?.status === 'completed'
    ? 'text-emerald-400'
    : isDone
    ? 'text-red-400'
    : 'text-gray-700'

  const pct =
    jobStatus && jobStatus.total_frames > 0
      ? Math.round((jobStatus.processed_frames / jobStatus.total_frames) * 100)
      : null

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-gray-300 font-mono select-none">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3.5 border-b border-gray-800 flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="text-sm font-semibold text-cyan-400 tracking-wide">ECHOFACE</div>
          <span className="text-gray-700">·</span>
          <div className="text-sm text-gray-400">Live Tracking Feed</div>
          {jobId && (
            <span className="text-[10px] font-mono text-gray-600 border border-gray-800 px-2 py-0.5 rounded">
              {jobId.slice(0, 8)}…
            </span>
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
      <div className="flex-1 flex items-center justify-center overflow-hidden bg-black">
        {!jobId ? (
          <div className="text-center space-y-4">
            <div className="w-16 h-16 rounded-full bg-gray-900 border border-gray-800 flex items-center justify-center mx-auto">
              <Radio size={28} className="text-gray-700" />
            </div>
            <div className="text-gray-500 text-sm">No active feed</div>
            <div className="text-gray-700 text-xs leading-relaxed">
              Start a tracking session from Operations<br />to begin live monitoring
            </div>
          </div>
        ) : !imgOk ? (
          <div className="text-center space-y-3">
            <Loader2 size={24} className="text-gray-700 animate-spin mx-auto" />
            <div className="text-gray-600 text-xs">Waiting for first annotated frame…</div>
          </div>
        ) : null}

        {/* Image is always mounted when jobId exists — hidden until first load */}
        {previewUrl && (
          <img
            key={tick}
            src={previewUrl}
            onLoad={() => setImgOk(true)}
            onError={() => setImgOk(false)}
            className={`max-w-full max-h-full object-contain ${imgOk ? 'block' : 'hidden'}`}
            alt="Annotated live feed"
          />
        )}
      </div>

      {/* Footer status bar */}
      <div className="px-6 py-3 border-t border-gray-800 flex items-center gap-6 flex-shrink-0">
        <div className="flex items-center gap-1.5 text-[10px]">
          <span className={`w-1.5 h-1.5 rounded-full ${dotClass}`} />
          <span className={labelClass}>{statusLabel}</span>
        </div>
        <div className="text-[10px] text-gray-700">
          FPS: {jobStatus?.avg_fps != null ? jobStatus.avg_fps.toFixed(1) : '--'}
        </div>
        <div className="text-[10px] text-gray-700">
          Faces: {jobStatus?.total_faces_detected ?? '--'}
        </div>
        <div className="text-[10px] text-gray-700">
          Alerts: {jobStatus?.alerts_created ?? '--'}
        </div>
        {pct != null && (
          <div className="ml-auto text-[10px] text-gray-700">{pct}% complete</div>
        )}
      </div>
    </div>
  )
}
