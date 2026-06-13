import { useCallback, useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Upload, Video, Play, Square, CheckCircle2, AlertTriangle,
  RotateCcw, Loader2, Film, Download, Activity,
  Gauge, Cpu, ShieldCheck, TrendingDown, Zap,
} from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { useVideoJob } from '../api/hooks'
import { createApiClient } from '../api/client'

// ── types ──────────────────────────────────────────────────────────────────────

type RawMetrics = Record<string, unknown>

interface JobMetricsSnapshot {
  capturedAt: string
  jobId: string
  // from observability endpoint
  obs_fps: number
  detector_latency_ms: number
  stable_matches: number
  identity_switch_rate: number
  validator_rejection_rate: number
  gpu_status: string
  hardware_backend_type: number
  // from job progress
  total_frames: number
  avg_fps: number
  alerts_created: number
  total_faces_detected: number
  total_faces_rejected: number
  blur_rejections: number
  processing_seconds: number
}

// ── mock job ───────────────────────────────────────────────────────────────────

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

// ── helpers ────────────────────────────────────────────────────────────────────

const fmt = (s: number) =>
  `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.round(s) % 60).padStart(2, '0')}`

function MetricRow({
  icon: Icon, label, value, sub, accent,
}: {
  icon: React.ElementType
  label: string
  value: string
  sub?: string
  accent?: string
}) {
  return (
    <div className="flex items-center gap-3 py-2.5 border-b border-gray-800/60 last:border-0">
      <div className="w-7 h-7 rounded bg-gray-800 flex items-center justify-center flex-shrink-0">
        <Icon size={13} className={accent ?? 'text-gray-500'} />
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-[10px] font-mono text-gray-600 leading-none mb-0.5">{label}</div>
        <div className={`text-[13px] font-mono font-semibold ${accent ?? 'text-gray-300'}`}>{value}</div>
      </div>
      {sub && <div className="text-[10px] font-mono text-gray-600 flex-shrink-0">{sub}</div>}
    </div>
  )
}

// ── main component ─────────────────────────────────────────────────────────────

export default function Operations() {
  const { accessMode, backendUrl, activeJobId, setActiveJobId } = useAppStore()
  const fileRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [snapshot, setSnapshot] = useState<JobMetricsSnapshot | null>(null)
  const capturedRef = useRef(false)

  const isMock = accessMode === 'MOCK'
  const isAdmin = accessMode === 'ADMIN'

  const { progress, previewUrl, error: jobError } = useVideoJob(!isMock ? activeJobId : null)
  const mock = useMockJob()

  // ── capture metrics snapshot on job completion (ADMIN only) ──────────────────
  const captureMetrics = useCallback(async () => {
    if (!isAdmin || !progress || capturedRef.current) return
    capturedRef.current = true
    // capture before the async gap so TypeScript knows it's non-null
    const p = progress
    try {
      const raw = await createApiClient(backendUrl).get<RawMetrics>('/observability/metrics')
      setSnapshot({
        capturedAt: new Date().toISOString(),
        jobId: activeJobId ?? '',
        obs_fps: Number(raw.fps ?? raw.average_processing_fps ?? 0),
        detector_latency_ms: Number(raw.detector_latency_ms ?? raw.detector_runtime_ms ?? 0),
        stable_matches: Number(raw.stable_matches ?? 0),
        identity_switch_rate: Number(raw.identity_switch_rate ?? 0),
        validator_rejection_rate: Number(raw.validator_rejection_rate ?? 0),
        gpu_status: String(raw.gpu_status ?? 'UNKNOWN'),
        hardware_backend_type: Number(raw.hardware_backend_type ?? 0),
        total_frames: p.total_frames,
        avg_fps: Number(p.avg_fps ?? 0),
        alerts_created: p.alerts_created,
        total_faces_detected: Number(p.total_faces_detected ?? 0),
        total_faces_rejected: Number(p.total_faces_rejected ?? 0),
        blur_rejections: Number(p.blur_rejections ?? 0),
        processing_seconds: Number(p.processing_duration_seconds ?? 0),
      })
    } catch { /* non-fatal */ }
  }, [isAdmin, progress, backendUrl, activeJobId])

  useEffect(() => {
    if (progress?.status === 'completed') void captureMetrics()
  }, [progress?.status, captureMetrics])

  // ── job actions ────────────────────────────────────────────────────────────
  const clearJob = () => {
    setActiveJobId(null)
    setFile(null)
    setUploadError(null)
    setSnapshot(null)
    capturedRef.current = false
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false)
    const f = e.dataTransfer.files[0]
    if (f?.type.startsWith('video/')) setFile(f)
  }

  const startTracking = async () => {
    if (isMock) { mock.start(); return }
    if (!file) return
    setSnapshot(null)
    capturedRef.current = false
    setUploading(true)
    setUploadError(null)
    try {
      const form = new FormData()
      form.append('video', file)
      const res = await fetch(`${backendUrl}/videos/upload-and-process`, { method: 'POST', body: form })
      if (!res.ok) throw new Error(`Upload failed (${res.status}): ${await res.text().catch(() => res.statusText)}`)
      const data = await res.json()
      setActiveJobId(String(data.job_id))
    } catch (e) {
      setUploadError((e as Error).message)
    } finally {
      setUploading(false)
    }
  }

  const downloadMetrics = () => {
    if (!snapshot) return
    const json = JSON.stringify({ snapshot, progress }, null, 2)
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([json], { type: 'application/json' }))
    a.download = `echoface_metrics_${snapshot.jobId.slice(0, 8)}.json`
    a.click()
  }

  // ── derived state ──────────────────────────────────────────────────────────
  const realJobActive = !isMock && activeJobId !== null
  const realStatus = progress?.status ?? 'queued'
  const realDone = realJobActive && (realStatus === 'completed' || realStatus === 'failed')
  const realRunning = realJobActive && !realDone
  const showUpload = isMock ? mock.state === 'idle' : !activeJobId
  const trackState: 'idle' | 'running' | 'done' =
    isMock ? mock.state : realRunning ? 'running' : realDone ? 'done' : 'idle'
  const pct = progress && progress.total_frames > 0
    ? Math.round((progress.processed_frames / progress.total_frames) * 100)
    : null

  return (
    <div className="flex h-full overflow-hidden">

      {/* ── left column ───────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto p-8 min-w-0">
        {/* header */}
        <div className="mb-7 flex items-start justify-between">
          <div>
            <h1 className="text-xl font-semibold text-gray-100">Operations</h1>
            <p className="text-xs font-mono text-gray-600 mt-1">
              Feed video into the tracking pipeline — file upload · live CCTV streams in a future release
            </p>
          </div>
          {(realJobActive || realDone || mock.state !== 'idle') && (
            <button
              onClick={clearJob}
              className="flex items-center gap-1.5 px-3 py-1.5 border border-gray-700 rounded text-xs font-mono text-gray-500 hover:text-gray-300 hover:border-gray-600 transition-colors flex-shrink-0"
            >
              <RotateCcw size={11} /> New job
            </button>
          )}
        </div>

        {/* ── upload / file selection ──────────────────────────────────────── */}
        <AnimatePresence mode="wait">
          {showUpload && (
            <motion.div
              key="upload"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="bg-gray-900 border border-gray-800 rounded-xl p-6 mb-5"
            >
              <div className="flex items-center gap-3 mb-5">
                <div className="w-8 h-8 bg-gray-800 rounded-lg flex items-center justify-center">
                  <Video size={15} className="text-cyan-400" />
                </div>
                <div>
                  <div className="text-sm font-semibold text-gray-200">Camera Source — File Upload</div>
                  <div className="text-[10px] font-mono text-gray-600">CAM-UPLOAD-001 · {isMock ? 'mode: mock' : 'standby'}</div>
                </div>
                <span className="ml-auto text-[10px] font-mono px-2 py-1 rounded border text-gray-600 border-gray-700 bg-gray-800/50">
                  ○ STANDBY
                </span>
              </div>

              <input ref={fileRef} type="file" accept="video/*" className="hidden" onChange={(e) => { if (e.target.files?.[0]) setFile(e.target.files[0]) }} />

              {!file ? (
                <div
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={handleDrop}
                  onClick={() => fileRef.current?.click()}
                  className={`border-2 border-dashed rounded-xl p-10 flex flex-col items-center gap-3 cursor-pointer transition-all duration-200 ${dragOver ? 'border-cyan-500/60 bg-cyan-500/5' : 'border-gray-700 hover:border-gray-600 hover:bg-gray-800/30'}`}
                >
                  <Upload size={22} className={dragOver ? 'text-cyan-400' : 'text-gray-600'} />
                  <div className="text-sm text-gray-500 text-center">
                    Drop a video file here or <span className="text-cyan-500">click to browse</span>
                    <div className="text-[11px] font-mono text-gray-700 mt-1">MP4 · AVI · MOV · MKV</div>
                  </div>
                </div>
              ) : (
                <div className="border border-gray-700 bg-gray-800/40 rounded-xl p-4 flex items-center gap-4">
                  <div className="w-11 h-11 bg-gray-900 rounded-lg border border-gray-700 flex items-center justify-center flex-shrink-0">
                    <Video size={17} className="text-gray-500" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-semibold text-gray-200 truncate">{file.name}</div>
                    <div className="text-[10px] font-mono text-gray-600 mt-0.5">
                      {(file.size / 1024 / 1024).toFixed(1)} MB · {file.type || 'video'}
                    </div>
                  </div>
                  {!uploading && (
                    <button onClick={() => setFile(null)} className="text-gray-600 hover:text-gray-400 transition-colors text-xl leading-none">×</button>
                  )}
                </div>
              )}

              {uploadError && (
                <div className="mt-3 border border-red-500/30 bg-red-500/8 rounded-lg px-3 py-2 text-xs font-mono text-red-400">
                  {uploadError}
                </div>
              )}

              {(file || isMock) && (
                <motion.button
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  onClick={startTracking}
                  disabled={uploading}
                  className="mt-4 w-full flex items-center justify-center gap-2 py-3 bg-cyan-500/15 border border-cyan-500/40 text-cyan-400 rounded-xl text-sm font-semibold tracking-wide hover:bg-cyan-500/25 transition-colors disabled:opacity-50 disabled:pointer-events-none"
                >
                  {uploading ? <><Loader2 size={15} className="animate-spin" /> Uploading…</> : <><Play size={15} /> Activate Tracking Pipeline</>}
                </motion.button>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── job running — compact status + progress ──────────────────────── */}
        <AnimatePresence>
          {realJobActive && progress && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-gray-900 border border-gray-800 rounded-xl p-5 mb-5"
            >
              {/* status row */}
              <div className="flex items-center gap-3 mb-4">
                <div className="w-8 h-8 bg-gray-800 rounded-lg flex items-center justify-center">
                  {realRunning
                    ? <Loader2 size={14} className="text-cyan-400 animate-spin" />
                    : realStatus === 'failed'
                    ? <span className="text-red-400 text-xs">✕</span>
                    : <CheckCircle2 size={14} className="text-emerald-400" />}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-[11px] font-mono text-gray-400">
                    Job <span className="text-cyan-400">{activeJobId?.slice(0, 12)}…</span>
                  </div>
                  <div className="text-[10px] font-mono text-gray-700 mt-0.5">
                    {realRunning ? 'Processing on server — polling every 2s' : realStatus === 'failed' ? 'Job failed' : 'Processing complete'}
                  </div>
                </div>
                <span className={`text-[10px] font-mono px-2 py-1 rounded border flex-shrink-0 ${
                  realRunning ? 'text-cyan-400 border-cyan-500/40 bg-cyan-500/10'
                  : realStatus === 'failed' ? 'text-red-400 border-red-500/30 bg-red-500/8'
                  : 'text-emerald-400 border-emerald-500/30 bg-emerald-500/8'
                }`}>
                  {realRunning ? '● PROCESSING' : realStatus === 'failed' ? '✕ FAILED' : '✓ COMPLETE'}
                </span>
              </div>

              {/* progress bar */}
              {pct !== null && (
                <div className="mb-4">
                  <div className="flex justify-between text-[10px] font-mono text-gray-600 mb-1.5">
                    <span>{progress.processed_frames.toLocaleString()} / {progress.total_frames.toLocaleString()} frames</span>
                    <span>{pct}%</span>
                  </div>
                  <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                    <motion.div
                      className={`h-full rounded-full ${realStatus === 'failed' ? 'bg-red-500' : 'bg-cyan-500'}`}
                      initial={{ width: 0 }}
                      animate={{ width: `${pct}%` }}
                      transition={{ duration: 0.4 }}
                    />
                  </div>
                </div>
              )}

              {/* stat tiles */}
              <div className="grid grid-cols-4 gap-3">
                {[
                  { label: 'FRAMES',  value: progress.processed_frames.toLocaleString(), accent: 'text-gray-200' },
                  { label: 'ALERTS',  value: String(progress.alerts_created), accent: progress.alerts_created > 0 ? 'text-amber-400' : 'text-gray-500' },
                  { label: 'AVG FPS', value: progress.avg_fps > 0 ? progress.avg_fps.toFixed(1) : '—', accent: 'text-cyan-400' },
                  { label: 'ELAPSED', value: fmt(Math.round(progress.processing_duration_seconds ?? 0)), accent: 'text-gray-400' },
                ].map((m) => (
                  <div key={m.label} className="bg-gray-950 border border-gray-800 rounded-lg p-3">
                    <div className="text-[9px] font-mono text-gray-600 tracking-widest mb-1">{m.label}</div>
                    <div className={`text-lg font-mono font-semibold ${m.accent}`}>{m.value}</div>
                  </div>
                ))}
              </div>

              {/* alerts banner */}
              {progress.alerts_created > 0 && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-4 border border-amber-500/30 bg-amber-500/8 rounded-lg p-3 flex items-center gap-3">
                  <AlertTriangle size={14} className="text-amber-400 flex-shrink-0" />
                  <div className="text-sm font-medium text-amber-300">
                    {progress.alerts_created} potential match{progress.alerts_created !== 1 ? 'es' : ''} detected —{' '}
                    <span className="text-amber-400/70 font-normal text-xs">open a Case Workspace to review</span>
                  </div>
                </motion.div>
              )}

              {realStatus === 'completed' && progress.alerts_created === 0 && (
                <div className="mt-3 flex items-center gap-2 text-xs font-mono text-gray-500">
                  <CheckCircle2 size={13} className="text-emerald-400" /> Processing complete — no matches found in this source
                </div>
              )}

              {(progress.error_message || jobError) && (
                <div className="mt-3 text-xs font-mono text-red-400">{progress.error_message ?? jobError}</div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── annotated preview ─────────────────────────────────────────────── */}
        <AnimatePresence>
          {previewUrl && (
            <motion.div
              key="preview"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden mb-5"
            >
              <div className="flex items-center gap-2 px-5 py-3 border-b border-gray-800">
                <Film size={13} className="text-cyan-400" />
                <h2 className="text-[10px] font-mono text-gray-400 tracking-widest">ANNOTATED PREVIEW</h2>
                <span className="ml-auto text-[10px] font-mono text-gray-700">bbox overlay · processed frames</span>
              </div>
              <div className="bg-black">
                <video
                  key={previewUrl}
                  src={previewUrl}
                  controls
                  className="w-full max-h-[400px] object-contain"
                  onError={(e) => console.warn('Preview failed:', previewUrl, e)}
                />
              </div>
              <div className="px-5 py-2 text-[10px] font-mono text-gray-700 truncate">
                {previewUrl.split('/').pop()}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── mock mode panel ───────────────────────────────────────────────── */}
        <AnimatePresence>
          {isMock && mock.state !== 'idle' && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="bg-gray-900 border border-gray-800 rounded-xl p-6">
              <div className="flex items-center justify-between mb-5">
                <h2 className="text-[10px] font-mono text-gray-600 tracking-widest">PIPELINE STATUS</h2>
                {mock.state === 'running' && (
                  <div className="flex items-center gap-2 text-[10px] font-mono text-violet-400">
                    <span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" /> MOCK · LIVE
                  </div>
                )}
              </div>
              <div className="grid grid-cols-4 gap-3 mb-5">
                {[
                  { label: 'FRAMES',  value: mock.stats.framesAnalyzed.toLocaleString(), accent: 'text-gray-200' },
                  { label: 'TRACKS',  value: String(mock.stats.tracksDetected),          accent: 'text-cyan-400' },
                  { label: 'MATCHES', value: String(mock.stats.matchesFound),            accent: mock.stats.matchesFound > 0 ? 'text-amber-400' : 'text-gray-500' },
                  { label: 'ELAPSED', value: fmt(mock.stats.elapsedSeconds),             accent: 'text-gray-400' },
                ].map((m) => (
                  <div key={m.label} className="bg-gray-950 border border-gray-800 rounded-lg p-3">
                    <div className="text-[9px] font-mono text-gray-600 tracking-widest mb-1">{m.label}</div>
                    <div className={`text-lg font-mono font-semibold ${m.accent}`}>{m.value}</div>
                  </div>
                ))}
              </div>
              {mock.stats.matchesFound > 0 && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="border border-amber-500/30 bg-amber-500/8 rounded-lg p-3 flex items-center gap-3 mb-4">
                  <AlertTriangle size={14} className="text-amber-400 flex-shrink-0" />
                  <div className="text-sm font-medium text-amber-300">
                    Potential match detected — <span className="text-xs font-normal text-amber-400/70">Frame {Math.floor(mock.stats.framesAnalyzed * 0.6)} · Confidence 91%</span>
                  </div>
                </motion.div>
              )}
              {mock.state === 'done' && mock.stats.matchesFound === 0 && (
                <div className="flex items-center gap-2 text-xs font-mono text-gray-500">
                  <CheckCircle2 size={13} className="text-emerald-400" /> Complete — no definitive matches found
                </div>
              )}
              {mock.state === 'running' && (
                <button onClick={mock.stop} className="mt-4 w-full flex items-center justify-center gap-2 py-2.5 bg-red-500/10 border border-red-500/30 text-red-400 rounded-lg text-sm font-semibold hover:bg-red-500/15 transition-colors">
                  <Square size={12} /> Stop Tracking
                </button>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ── right panel — ADMIN only ─────────────────────────────────────────── */}
      {isAdmin && (
        <aside className="w-72 flex-shrink-0 border-l border-gray-800 bg-gray-950 overflow-y-auto">
          <div className="p-5">
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2">
                <Activity size={12} className="text-cyan-400" />
                <h3 className="text-[10px] font-mono text-gray-400 tracking-widest">JOB METRICS</h3>
              </div>
              {snapshot && (
                <button
                  onClick={downloadMetrics}
                  title="Download metrics JSON"
                  className="flex items-center gap-1 px-2 py-1 border border-gray-700 rounded text-[10px] font-mono text-gray-500 hover:text-gray-300 hover:border-gray-600 transition-colors"
                >
                  <Download size={10} /> JSON
                </button>
              )}
            </div>
            <div className="text-[9px] font-mono text-gray-700 mb-5">
              {snapshot
                ? `Captured ${new Date(snapshot.capturedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`
                : 'Snapshot taken on job completion'}
            </div>

            <AnimatePresence mode="wait">
              {!snapshot ? (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex flex-col items-center justify-center py-12 text-center"
                >
                  <div className="w-10 h-10 rounded-full bg-gray-900 border border-gray-800 flex items-center justify-center mb-3">
                    <Activity size={16} className="text-gray-700" />
                  </div>
                  <div className="text-[11px] font-mono text-gray-600">No data yet</div>
                  <div className="text-[10px] font-mono text-gray-700 mt-1">Run a job to capture metrics</div>
                  {realRunning && (
                    <div className="mt-4 flex items-center gap-1.5 text-[10px] font-mono text-cyan-600">
                      <Loader2 size={10} className="animate-spin" /> Processing…
                    </div>
                  )}
                </motion.div>
              ) : (
                <motion.div key="data" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}>

                  {/* ── processing stats ─────────────────────────────────── */}
                  <div className="text-[9px] font-mono text-gray-600 tracking-widest mb-2">PROCESSING</div>
                  <MetricRow
                    icon={Gauge}
                    label="Avg processing FPS"
                    value={snapshot.avg_fps.toFixed(1)}
                    accent={snapshot.avg_fps > 20 ? 'text-emerald-400' : snapshot.avg_fps > 5 ? 'text-cyan-400' : 'text-amber-400'}
                  />
                  <MetricRow
                    icon={Film}
                    label="Total frames"
                    value={snapshot.total_frames.toLocaleString()}
                  />
                  <MetricRow
                    icon={Zap}
                    label="Elapsed time"
                    value={fmt(Math.round(snapshot.processing_seconds))}
                  />
                  <MetricRow
                    icon={AlertTriangle}
                    label="Alerts created"
                    value={String(snapshot.alerts_created)}
                    accent={snapshot.alerts_created > 0 ? 'text-amber-400' : 'text-gray-500'}
                  />

                  {/* ── detector ──────────────────────────────────────────── */}
                  <div className="text-[9px] font-mono text-gray-600 tracking-widest mt-5 mb-2">DETECTOR</div>
                  <MetricRow
                    icon={Cpu}
                    label="Detector latency"
                    value={`${snapshot.detector_latency_ms.toFixed(1)} ms`}
                    accent={snapshot.detector_latency_ms < 20 ? 'text-emerald-400' : snapshot.detector_latency_ms < 50 ? 'text-cyan-400' : 'text-amber-400'}
                  />
                  <MetricRow
                    icon={ShieldCheck}
                    label="Faces detected"
                    value={String(snapshot.total_faces_detected)}
                  />
                  <MetricRow
                    icon={TrendingDown}
                    label="Rejection rate"
                    value={`${(snapshot.validator_rejection_rate * 100).toFixed(1)}%`}
                    sub={`${snapshot.total_faces_rejected} rejected`}
                    accent={snapshot.validator_rejection_rate < 0.2 ? 'text-emerald-400' : 'text-amber-400'}
                  />
                  <MetricRow
                    icon={TrendingDown}
                    label="Blur rejections"
                    value={String(snapshot.blur_rejections)}
                    accent="text-gray-400"
                  />

                  {/* ── identity model ───────────────────────────────────── */}
                  <div className="text-[9px] font-mono text-gray-600 tracking-widest mt-5 mb-2">IDENTITY MODEL</div>
                  <MetricRow
                    icon={Activity}
                    label="Stable matches"
                    value={String(snapshot.stable_matches)}
                    accent="text-cyan-400"
                  />
                  <MetricRow
                    icon={TrendingDown}
                    label="Identity switch rate"
                    value={snapshot.identity_switch_rate.toFixed(4)}
                    accent={snapshot.identity_switch_rate === 0 ? 'text-emerald-400' : 'text-amber-400'}
                  />

                  {/* ── hardware ──────────────────────────────────────────── */}
                  <div className="text-[9px] font-mono text-gray-600 tracking-widest mt-5 mb-2">HARDWARE</div>
                  <MetricRow
                    icon={Cpu}
                    label="Backend"
                    value={snapshot.hardware_backend_type === 1 ? 'GPU' : 'CPU'}
                    accent={snapshot.hardware_backend_type === 1 ? 'text-emerald-400' : 'text-cyan-400'}
                  />
                  <MetricRow
                    icon={ShieldCheck}
                    label="GPU status"
                    value={snapshot.gpu_status}
                    accent={snapshot.gpu_status === 'OK' ? 'text-emerald-400' : 'text-amber-400'}
                  />
                  <MetricRow
                    icon={Gauge}
                    label="Obs. FPS"
                    value={snapshot.obs_fps.toFixed(1)}
                    sub="rolling avg"
                    accent="text-gray-400"
                  />

                  <div className="mt-5 pt-4 border-t border-gray-800">
                    <div className="text-[9px] font-mono text-gray-700 mb-2">Job ID</div>
                    <div className="text-[10px] font-mono text-gray-600 break-all">{snapshot.jobId}</div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </aside>
      )}
    </div>
  )
}
