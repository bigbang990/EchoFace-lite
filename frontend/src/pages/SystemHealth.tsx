import { Cpu, Activity, Layers, Timer, AlertTriangle, CheckCircle2, BarChart2, Download, RefreshCw } from 'lucide-react'
import {
  LineChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { useAppStore } from '../store/appStore'
import { useSystemMetrics } from '../api/hooks'
import type { SystemMetrics } from '../types'

function downloadBlob(content: string, filename: string, mime: string) {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

function exportJSON(m: SystemMetrics, fpsHistory: { t: string; v: number }[]) {
  const payload = {
    exported_at: new Date().toISOString(),
    metrics: m,
    fps_history: fpsHistory,
  }
  downloadBlob(
    JSON.stringify(payload, null, 2),
    `echoface_metrics_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`,
    'application/json'
  )
}

function exportCSV(m: SystemMetrics) {
  const rows: string[] = ['metric,value']
  for (const [k, v] of Object.entries(m)) {
    rows.push(`${k},${v}`)
  }
  rows.push(`exported_at,${new Date().toISOString()}`)
  downloadBlob(
    rows.join('\n'),
    `echoface_metrics_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`,
    'text/csv'
  )
}

export default function SystemHealth() {
  const { accessMode } = useAppStore()
  const { data: m, loading, error, fpsHistory } = useSystemMetrics()

  const isLive = accessMode === 'ADMIN'

  const uptimeH = Math.floor(m.uptime_seconds / 3600)
  const uptimeM = Math.floor((m.uptime_seconds % 3600) / 60)

  return (
    <div className="p-8 max-w-5xl">
      <div className="mb-7 flex items-start justify-between">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">System Health</h1>
          <p className="text-xs font-mono text-gray-600 mt-1">
            Pipeline telemetry — Admin view
          </p>
        </div>
        <div className="flex items-center gap-3">
          {isLive && !error && (
            <div className="flex items-center gap-1.5 text-[10px] font-mono text-cyan-400">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
              LIVE · 3s
            </div>
          )}
          {error && (
            <div className="text-[10px] font-mono text-red-400 flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
              {error}
            </div>
          )}
          <button
            onClick={() => exportCSV(m)}
            className="flex items-center gap-1.5 px-3 py-1.5 border border-gray-700 rounded text-xs font-mono text-gray-400 hover:text-gray-200 hover:border-gray-600 transition-colors"
          >
            <Download size={11} /> CSV
          </button>
          <button
            onClick={() => exportJSON(m, fpsHistory)}
            className="flex items-center gap-1.5 px-3 py-1.5 border border-gray-700 rounded text-xs font-mono text-gray-400 hover:text-gray-200 hover:border-gray-600 transition-colors"
          >
            <Download size={11} /> JSON
          </button>
        </div>
      </div>

      {loading && (
        <div className="text-sm font-mono text-gray-600 mb-6 flex items-center gap-2">
          <RefreshCw size={13} className="animate-spin" /> Fetching metrics…
        </div>
      )}

      <div className="grid grid-cols-3 gap-4 mb-5">
        <StatusBlock label="GPU Backend" value={m.gpu_status}             sub={`hardware_backend_type=${m.hardware_backend_type}`} ok={m.gpu_status === 'OK'} Icon={Cpu} />
        <StatusBlock label="CPU Load"    value={`${m.cpu_percent.toFixed(1)}%`} sub="system process"  ok={m.cpu_percent < 70}  Icon={Activity} />
        <StatusBlock label="Memory"      value={`${(m.memory_mb / 1024).toFixed(1)} GB`} sub={`${m.memory_mb} MB`} ok={m.memory_mb < 6000} Icon={Layers} />
      </div>

      <div className="grid grid-cols-4 gap-3 mb-6">
        {[
          { label: 'AVG FPS',          value: m.fps.toFixed(1),              Icon: BarChart2, color: 'text-cyan-400',  detail: 'frames / sec' },
          { label: 'DETECTOR LATENCY', value: `${m.detector_latency_ms}ms`,  Icon: Timer,     color: 'text-cyan-400',  detail: 'det_size=(320,320)' },
          { label: 'ACTIVE TRACKS',    value: m.active_tracks,               Icon: Layers,    color: 'text-cyan-400',  detail: 'identities in-frame' },
          { label: 'QUEUE DEPTH',      value: m.queue_depth,                 Icon: Activity,  color: m.queue_depth > 10 ? 'text-amber-400' : 'text-cyan-400', detail: 'frames pending' },
        ].map((tile) => (
          <div key={tile.label} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <tile.Icon size={13} className="text-gray-600" />
              <div className="text-[9px] font-mono text-gray-600 tracking-widest">{tile.label}</div>
            </div>
            <div className={`text-2xl font-mono font-bold ${tile.color}`}>{tile.value}</div>
            <div className="text-[10px] font-mono text-gray-700 mt-1">{tile.detail}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-4 mb-5">
        {[
          {
            label: 'IDENTITY SWITCH RATE',
            value: m.identity_switch_rate.toFixed(3),
            good: m.identity_switch_rate < 0.01,
            goodText: 'Excellent — no swaps',
            badText: 'Review tracker config',
          },
          {
            label: 'CONFIRMATION RATE',
            value: `${(m.confirmation_rate * 100).toFixed(1)}%`,
            good: m.confirmation_rate >= 0.9,
            goodText: 'High precision',
            badText: 'Below target threshold',
          },
          {
            label: 'VALIDATOR REJECTION',
            value: `${(m.validator_rejection_rate * 100).toFixed(1)}%`,
            good: m.validator_rejection_rate <= 0.15,
            goodText: 'Within bounds',
            badText: 'High — check thresholds',
          },
        ].map((row) => (
          <div key={row.label} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <div className="text-[9px] font-mono text-gray-600 tracking-widest mb-2">{row.label}</div>
            <div className="text-xl font-mono font-bold text-gray-100 mb-1">{row.value}</div>
            <div className={`flex items-center gap-1.5 text-[10px] font-mono ${row.good ? 'text-emerald-500' : 'text-amber-500'}`}>
              {row.good ? <CheckCircle2 size={10} /> : <AlertTriangle size={10} />}
              {row.good ? row.goodText : row.badText}
            </div>
          </div>
        ))}
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-lg p-5 mb-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-[10px] font-mono text-gray-600 tracking-widest">
            FPS HISTORY {isLive ? '— LIVE' : '— 24H SAMPLE'}
          </h2>
          <span className="text-[10px] font-mono text-gray-700">avg {m.fps.toFixed(1)} fps</span>
        </div>
        <div className="h-40">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={fpsHistory} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
              <XAxis dataKey="t" tick={{ fill: '#4b5563', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }} axisLine={false} tickLine={false} interval={3} />
              <YAxis domain={[40, 100]} tick={{ fill: '#4b5563', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }} axisLine={false} tickLine={false} width={28} />
              <Tooltip
                contentStyle={{ background: '#111827', border: '1px solid #1f2937', borderRadius: 6, fontSize: 11, fontFamily: 'JetBrains Mono, monospace', color: '#9ca3af' }}
                formatter={(v: number) => [`${v} fps`, 'FPS']}
              />
              <Line type="monotone" dataKey="v" stroke="#22d3ee" strokeWidth={1.5} dot={false} activeDot={{ r: 3, fill: '#22d3ee' }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-lg p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-[10px] font-mono text-gray-600 tracking-widest">PLATFORM CONFIG</h2>
          <span className="text-[10px] font-mono text-gray-700">uptime {uptimeH}h {uptimeM}m</span>
        </div>
        <div className="grid grid-cols-2 gap-x-10 gap-y-2 font-mono text-[11px]">
          {[
            ['backend',            m.hardware_backend_type === 1 ? 'GPU' : 'CPU'],
            ['providers',          m.hardware_backend_type === 1 ? 'CUDAExecutionProvider' : 'CPUExecutionProvider'],
            ['det_size',           m.hardware_backend_type === 1 ? '(640, 640)' : '(320, 320)'],
            ['det_interval',       m.hardware_backend_type === 1 ? '3' : '6'],
            ['conf_threshold',     m.hardware_backend_type === 1 ? '0.45' : '0.35'],
            ['validator_cutoff',   m.hardware_backend_type === 1 ? '0.55' : '0.40'],
            ['detector_budget_ms', m.hardware_backend_type === 1 ? '150' : '5000'],
            ['max_track_survival', m.hardware_backend_type === 1 ? '3000 ms' : '6000 ms'],
            ['stable_matches',     String(m.stable_matches)],
            ['uptime',             `${uptimeH}h ${uptimeM}m`],
          ].map(([k, v]) => (
            <div key={k} className="flex justify-between py-1 border-b border-gray-800/60 last:border-0">
              <span className="text-gray-600">{k}</span>
              <span className="text-gray-300">{v}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function StatusBlock({ label, value, sub, ok, Icon }: {
  label: string; value: string; sub: string; ok: boolean; Icon: React.ElementType
}) {
  return (
    <div className={`bg-gray-900 border rounded-lg p-5 flex items-center gap-4 ${ok ? 'border-emerald-500/20' : 'border-red-500/20'}`}>
      <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${ok ? 'bg-emerald-500/10' : 'bg-red-500/10'}`}>
        <Icon size={18} className={ok ? 'text-emerald-400' : 'text-red-400'} />
      </div>
      <div>
        <div className="text-[10px] font-mono text-gray-600 tracking-widest mb-0.5">{label}</div>
        <div className={`text-xl font-mono font-bold ${ok ? 'text-emerald-400' : 'text-red-400'}`}>{value}</div>
        <div className="text-[10px] font-mono text-gray-700 mt-0.5">{sub}</div>
      </div>
    </div>
  )
}
