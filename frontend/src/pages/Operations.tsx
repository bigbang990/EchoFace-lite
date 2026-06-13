import { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Upload, Video, Play, Square, CheckCircle2, AlertTriangle,
  RotateCcw, Loader2, Film,
} from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { useVideoJob } from '../api/hooks'

// ── mock mode state (component-local only, no persistence needed) ─────────────

interface MockStats {
  framesAnalyzed: number
  tracksDetected: number
  matchesFound: number
  elapsedSeconds: number
}

function useMockJob() {
  const [state, setState] = useState<'idle' | 'running' | 'done'>('idle')
  const [stats, setStats] = useState<MockStats>({
    framesAnalyzed: 0, tracksDetected: 0, matchesFound: 0, elapsedSeconds: 0,
  })
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const start = () => {
    setState('running')
    setStats({ framesAnalyzed: 0, tracksDetected: 0, matchesFound: 0, elapsedSeconds: 0 })
    intervalRef.current = setInterval(() => {
      setStats((prev) => {
        const elapsed = prev.elapsedSeconds + 1
        const shouldMatch = prev.matchesFound === 0 && elapsed > 8 && Math.random() > 0.85
        return {
          framesAnalyzed: prev.framesAnalyzed + Math.floor(Math.random() * 6 + 20),
          tracksDetected: Math.min(prev.tracksDetected + (Math.random() > 0.7 ? 1 : 0), 12),
          matchesFound: shouldMatch ? 1 : prev.matchesFound,
          elapsedSeconds: elapsed,
        }
      })
    }, 1000)
  }

  const stop = () => {
    setState('done')
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null }
  }

  useEffect(() => () => { if (intervalRef.current) clearInterval(intervalRef.current) }, [])

  return { state, stats, start, stop }
}

// ── main component ─────────────────────────────────────────────────────────────

export default function Operations() {
  const { accessMode, backendUrl, activeJobId, setActiveJobId } = useAppStore()
  const fileRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)

  const isMock = accessMode === 'MOCK'

  // Real job polling — only active when not mock and job exists
  const { progress, previewUrl, error: jobError } = useVideoJob(
    !isMock ? activeJobId : null
  )

  // Mock mode fallback
  const mock = useMockJob()

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped && dropped.type.startsWith('video/')) setFile(dropped)
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) setFile(e.target.files[0])
  }

  const startTracking = async () => {
    if (isMock) {
      mock.start()
      return
    }
    if (!file) return
    setUploading(true)
    setUploadError(null)
    try {
      const form = new FormData()
      form.append('video', file)
      const res = await fetch(`${backendUrl}/videos/upload-and-process`, {
        method: 'POST',
        body: form,
      })
      if (!res.ok) {
        const msg = await res.text().catch(() => res.statusText)
        throw new Error(`Upload failed (${res.status}): ${msg}`)
      }
      const data = await res.json()
      setActiveJobId(String(data.job_id))
    } catch (e) {
      setUploadError((e as Error).message)
    } finally {
      setUploading(false)
    }
  }

  const clearJob = () => {
    setActiveJobId(null)
    setFile(null)
    setUploadError(null)
  }

  const fmtDuration = (s: number) =>
    `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`

  // ── derived display state ──────────────────────────────────────────────────

  const realJobRunning = !isMock && activeJobId !== null
  const realStatus = progress?.status ?? 'queued'
  const realDone = realJobRunning && (realStatus === 'completed' || realStatus === 'failed')
  const realRunning = realJobRunning && !realDone

  const showUploadUI = isMock ? mock.state === 'idle' : !activeJobId
  const trackState: 'idle' | 'running' | 'done' =
    isMock ? mock.state :
    realRunning ? 'running' :
    realDone ? 'done' : 'idle'

  const pct = progress && progress.total_frames > 0
    ? Math.round((progress.processed_frames / progress.total_frames) * 100)
    : null

  return (
    <div className="p-8 max-w-3xl">
      <div className="mb-7 flex items-start justify-between">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">Operations</h1>
          <p className="text-xs font-mono text-gray-600 mt-1">
            Feed video into the tracking pipeline — file upload · live CCTV streams in a future release
          </p>
        </div>
        {(realJobRunning || realDone) && (
          <button
            onClick={clearJob}
            className="flex items-center gap-1.5 px-3 py-1.5 border border-gray-700 rounded text-xs font-mono text-gray-500 hover:text-gray-300 hover:border-gray-600 transition-colors"
          >
            <RotateCcw size={11} /> New job
          </button>
        )}
      </div>

      {/* ── Upload / File Panel ─────────────────────────────────────────────── */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 mb-5">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-8 h-8 bg-gray-800 rounded flex items-center justify-center">
            <Video size={15} className="text-cyan-400" />
          </div>
          <div>
            <div className="text-sm font-semibold text-gray-200">Camera Source — File Upload</div>
            <div className="text-[10px] font-mono text-gray-600">
              CAM-UPLOAD-001 · {isMock ? 'mode: mock' : `job: ${activeJobId ?? 'none'}`}
            </div>
          </div>
          <div className="ml-auto">
            <span className={`text-[10px] font-mono px-2 py-1 rounded border ${
              trackState === 'running'
                ? 'text-cyan-400 border-cyan-500/40 bg-cyan-500/10'
                : trackState === 'done'
                ? 'text-emerald-400 border-emerald-500/30 bg-emerald-500/8'
                : 'text-gray-600 border-gray-700 bg-gray-800/50'
            }`}>
              {trackState === 'running' ? '● PROCESSING' : trackState === 'done' ? '✓ COMPLETE' : '○ STANDBY'}
            </span>
          </div>
        </div>

        <input ref={fileRef} type="file" accept="video/*" className="hidden" onChange={handleFileChange} />

        {showUploadUI && (
          <>
            {!file ? (
              <div
                onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() => fileRef.current?.click()}
                className={`border-2 border-dashed rounded-lg p-10 flex flex-col items-center gap-3 cursor-pointer transition-all duration-200 ${
                  dragOver ? 'border-cyan-500/60 bg-cyan-500/5' : 'border-gray-700 hover:border-gray-600 hover:bg-gray-800/30'
                }`}
              >
                <Upload size={24} className={dragOver ? 'text-cyan-400' : 'text-gray-600'} />
                <div className="text-sm text-gray-500 text-center">
                  <div>Drop a video file here or <span className="text-cyan-500">click to browse</span></div>
                  <div className="text-[11px] font-mono text-gray-700 mt-1">MP4 · AVI · MOV · MKV</div>
                </div>
              </div>
            ) : (
              <div className="border border-gray-700 bg-gray-800/40 rounded-lg p-4">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-gray-900 rounded border border-gray-700 flex items-center justify-center flex-shrink-0">
                    <Video size={18} className="text-gray-500" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-semibold text-gray-200 truncate">{file.name}</div>
                    <div className="text-[10px] font-mono text-gray-600 mt-0.5">
                      {(file.size / 1024 / 1024).toFixed(1)} MB · {file.type || 'video'}
                    </div>
                  </div>
                  {!uploading && (
                    <button onClick={() => setFile(null)} className="text-gray-600 hover:text-gray-400 transition-colors text-lg leading-none">×</button>
                  )}
                </div>
              </div>
            )}

            {uploadError && (
              <div className="mt-3 border border-red-500/30 bg-red-500/8 rounded px-3 py-2 text-xs font-mono text-red-400">
                {uploadError}
              </div>
            )}

            {file && (
              <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mt-4">
                <button
                  onClick={startTracking}
                  disabled={uploading}
                  className="w-full flex items-center justify-center gap-2 py-3 bg-cyan-500/15 border border-cyan-500/40 text-cyan-400 rounded text-sm font-semibold tracking-wide hover:bg-cyan-500/25 transition-colors disabled:opacity-50 disabled:pointer-events-none"
                >
                  {uploading ? (
                    <><Loader2 size={15} className="animate-spin" /> Uploading…</>
                  ) : (
                    <><Play size={15} /> Activate Tracking Pipeline</>
                  )}
                </button>
              </motion.div>
            )}
          </>
        )}

        {/* Resuming indicator when job already exists */}
        {realJobRunning && !file && (
          <div className="border border-gray-700 bg-gray-800/30 rounded-lg p-4 flex items-center gap-3">
            <Video size={16} className="text-gray-500 flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <div className="text-sm text-gray-400 font-mono">Job <span className="text-cyan-400">{activeJobId?.slice(0, 8)}…</span></div>
              <div className="text-[10px] font-mono text-gray-600 mt-0.5">Processing on server — polling status</div>
            </div>
            {realRunning && <Loader2 size={14} className="text-cyan-400 animate-spin flex-shrink-0" />}
          </div>
        )}
      </div>

      {/* ── Real job stats panel ────────────────────────────────────────────── */}
      <AnimatePresence>
        {realJobRunning && progress && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-gray-900 border border-gray-800 rounded-lg p-6 mb-5"
          >
            <div className="flex items-center justify-between mb-5">
              <h2 className="text-[10px] font-mono text-gray-600 tracking-widest">PIPELINE STATUS</h2>
              {realRunning && (
                <div className="flex items-center gap-2 text-[10px] font-mono text-cyan-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                  LIVE · 2s
                </div>
              )}
              {realStatus === 'failed' && (
                <span className="text-[10px] font-mono text-red-400">FAILED</span>
              )}
            </div>

            {/* Progress bar */}
            {pct !== null && (
              <div className="mb-5">
                <div className="flex justify-between text-[10px] font-mono text-gray-600 mb-1.5">
                  <span>{progress.processed_frames.toLocaleString()} / {progress.total_frames.toLocaleString()} frames</span>
                  <span>{pct}%</span>
                </div>
                <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-cyan-500 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${pct}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
              </div>
            )}

            <div className="grid grid-cols-4 gap-4 mb-5">
              {[
                { label: 'FRAMES',   value: progress.processed_frames.toLocaleString(), color: 'text-gray-200' },
                { label: 'ALERTS',   value: progress.alerts_created,                    color: progress.alerts_created > 0 ? 'text-amber-400' : 'text-gray-500' },
                { label: 'AVG FPS',  value: progress.avg_fps > 0 ? progress.avg_fps.toFixed(1) : '—',   color: 'text-cyan-400' },
                { label: 'ELAPSED',  value: fmtDuration(Math.round(progress.processing_duration_seconds)), color: 'text-gray-400' },
              ].map((m) => (
                <div key={m.label} className="bg-gray-950 border border-gray-800 rounded p-3">
                  <div className="text-[9px] font-mono text-gray-600 tracking-widest mb-1">{m.label}</div>
                  <div className={`text-xl font-mono font-semibold ${m.color}`}>{m.value}</div>
                </div>
              ))}
            </div>

            {progress.alerts_created > 0 && (
              <motion.div
                initial={{ opacity: 0, scale: 0.97 }}
                animate={{ opacity: 1, scale: 1 }}
                className="border border-amber-500/30 bg-amber-500/8 rounded-lg p-4 flex items-start gap-3 mb-4"
              >
                <AlertTriangle size={16} className="text-amber-400 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="text-sm font-semibold text-amber-300">
                    {progress.alerts_created} Potential Match{progress.alerts_created !== 1 ? 'es' : ''} Detected
                  </div>
                  <div className="text-xs text-gray-500 mt-0.5 font-mono">
                    Open a Case Workspace to review and confirm sightings
                  </div>
                </div>
              </motion.div>
            )}

            {realStatus === 'completed' && progress.alerts_created === 0 && (
              <div className="flex items-center gap-2 text-sm text-gray-500 mb-4">
                <CheckCircle2 size={14} className="text-emerald-400" />
                Processing complete — no matches found in this source
              </div>
            )}

            {progress.error_message && (
              <div className="text-xs font-mono text-red-400 mb-4">{progress.error_message}</div>
            )}

            {jobError && (
              <div className="text-xs font-mono text-red-400">{jobError}</div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Preview video panel ─────────────────────────────────────────────── */}
      <AnimatePresence>
        {previewUrl && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden mb-5"
          >
            <div className="flex items-center gap-2 px-5 py-3 border-b border-gray-800">
              <Film size={13} className="text-cyan-400" />
              <h2 className="text-[10px] font-mono text-gray-400 tracking-widest">ANNOTATED PREVIEW</h2>
              <span className="ml-auto text-[10px] font-mono text-gray-700">bbox overlay · processed frames</span>
            </div>
            <div className="bg-black">
              <video
                src={previewUrl}
                controls
                autoPlay={false}
                className="w-full max-h-[360px] object-contain"
                onError={() => console.warn('Preview video failed to load:', previewUrl)}
              />
            </div>
            <div className="px-5 py-2.5 text-[10px] font-mono text-gray-700">
              {previewUrl.split('/').pop()}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Mock mode stats panel ───────────────────────────────────────────── */}
      <AnimatePresence>
        {isMock && (mock.state === 'running' || mock.state === 'done') && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-gray-900 border border-gray-800 rounded-lg p-6"
          >
            <div className="flex items-center justify-between mb-5">
              <h2 className="text-[10px] font-mono text-gray-600 tracking-widest">PIPELINE STATUS</h2>
              {mock.state === 'running' && (
                <div className="flex items-center gap-2 text-[10px] font-mono text-violet-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" />
                  MOCK · LIVE
                </div>
              )}
            </div>

            <div className="grid grid-cols-4 gap-4 mb-6">
              {[
                { label: 'FRAMES',  value: mock.stats.framesAnalyzed.toLocaleString(), color: 'text-gray-200' },
                { label: 'TRACKS',  value: mock.stats.tracksDetected,                  color: 'text-cyan-400' },
                { label: 'MATCHES', value: mock.stats.matchesFound,                    color: mock.stats.matchesFound > 0 ? 'text-amber-400' : 'text-gray-500' },
                { label: 'ELAPSED', value: fmtDuration(mock.stats.elapsedSeconds),     color: 'text-gray-400' },
              ].map((m) => (
                <div key={m.label} className="bg-gray-950 border border-gray-800 rounded p-3">
                  <div className="text-[9px] font-mono text-gray-600 tracking-widest mb-1">{m.label}</div>
                  <div className={`text-xl font-mono font-semibold ${m.color}`}>{m.value}</div>
                </div>
              ))}
            </div>

            {mock.stats.matchesFound > 0 && (
              <motion.div
                initial={{ opacity: 0, scale: 0.97 }}
                animate={{ opacity: 1, scale: 1 }}
                className="border border-amber-500/30 bg-amber-500/8 rounded-lg p-4 flex items-start gap-3"
              >
                <AlertTriangle size={16} className="text-amber-400 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="text-sm font-semibold text-amber-300">Potential Match Detected</div>
                  <div className="text-xs text-gray-500 mt-0.5 font-mono">
                    Frame {Math.floor(mock.stats.framesAnalyzed * 0.6)} · Confidence 91% · Open the Case Workspace to review
                  </div>
                </div>
              </motion.div>
            )}

            {mock.state === 'done' && mock.stats.matchesFound === 0 && (
              <div className="flex items-center gap-2 text-sm text-gray-500">
                <CheckCircle2 size={14} className="text-emerald-400" />
                Processing complete — no definitive matches found in this source
              </div>
            )}

            {mock.state === 'running' && (
              <div className="mt-5">
                <button
                  onClick={mock.stop}
                  className="w-full flex items-center justify-center gap-2 py-2.5 bg-red-500/10 border border-red-500/30 text-red-400 rounded text-sm font-semibold hover:bg-red-500/15 transition-colors"
                >
                  <Square size={13} /> Stop Tracking
                </button>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
