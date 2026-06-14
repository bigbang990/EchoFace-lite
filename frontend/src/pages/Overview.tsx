import { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Activity, AlertTriangle, CheckCircle2, TrendingUp,
  Zap, Timer, Layers, Cpu, ArrowRight,
} from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { useIncidents, useSystemMetrics, useCameraHealthSummary, deriveActivityFeed } from '../api/hooks'
import { mockActivityFeed } from '../mock/data'
import StatusIndicator from '../components/StatusIndicator'

export default function Overview() {
  const navigate = useNavigate()
  const { accessMode } = useAppStore()
  const { data: incidents, loading: incLoading } = useIncidents()
  const { data: m, error: metricsError } = useSystemMetrics()
  const { data: camHealth } = useCameraHealthSummary()

  const isAdmin = accessMode === 'ADMIN'

  const trackingCount = incidents.filter((i) => i.status === 'TRACKING').length
  const openCount     = incidents.filter((i) => i.status === 'OPEN').length
  const resolvedCount = incidents.filter((i) => i.status === 'RESOLVED' || i.status === 'CLOSED').length
  const pendingAlerts = incidents.reduce((s, i) => s + i.pending_alert_count, 0)

  const activityFeed = useMemo(
    () => (accessMode === 'MOCK' ? mockActivityFeed : deriveActivityFeed(incidents)),
    [incidents, accessMode]
  )

  const uptimeH = Math.floor(m.uptime_seconds / 3600)
  const uptimeM = Math.floor((m.uptime_seconds % 3600) / 60)

  // Resolve hardware label — type 0 = CPU (not an error)
  const hwLabel  = m.hardware_backend_type === 1 ? 'GPU' : 'CPU'
  const hwStatus = m.hardware_backend_type === 1
    ? (m.gpu_status === 'OK' ? 'ACTIVE' : m.gpu_status)
    : 'ACTIVE'
  const hwColor  = m.hardware_backend_type === 1 && m.gpu_status !== 'OK'
    ? 'text-amber-400'
    : 'text-emerald-400'

  return (
    <div className="p-8 max-w-6xl">
      {/* ── header ─────────────────────────────────────────────────────────── */}
      <div className="mb-7 flex items-start justify-between">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">Operations Overview</h1>
          <p className="text-xs font-mono text-gray-600 mt-1">
            {new Date().toLocaleDateString('en-GB', {
              weekday: 'long', year: 'numeric', month: 'long', day: 'numeric',
            })}
            &ensp;·&ensp;
            {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </p>
        </div>
        {isAdmin && !metricsError && (
          <div className="flex items-center gap-1.5 text-[10px] font-mono text-cyan-400">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
            LIVE · 30s refresh
          </div>
        )}
        {metricsError && (
          <div className="text-[10px] font-mono text-red-400 flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
            metrics offline
          </div>
        )}
      </div>

      {/* ── ADMIN live telemetry tiles ──────────────────────────────────────── */}
      {isAdmin && (
        <div className="grid grid-cols-4 gap-3 mb-6">
          <TelemetryTile
            label="AVG FPS"
            value={m.fps > 0 ? m.fps.toFixed(1) : '—'}
            sub={m.fps > 0 ? 'frames / sec' : 'no active job'}
            Icon={Zap}
            color="text-cyan-400"
          />
          <TelemetryTile
            label="DETECTOR"
            value={m.detector_latency_ms > 0 ? `${m.detector_latency_ms.toFixed(0)}ms` : '—'}
            sub={m.detector_latency_ms > 0 ? 'avg latency' : 'no active job'}
            Icon={Timer}
            color="text-cyan-400"
          />
          <TelemetryTile
            label="ACTIVE TRACKS"
            value={String(m.active_tracks)}
            sub="identities in-frame"
            Icon={Layers}
            color="text-cyan-400"
          />
          <TelemetryTile
            label={`${hwLabel} BACKEND`}
            value={hwStatus}
            sub={`type ${m.hardware_backend_type} · ${m.fps > 0 ? 'processing' : 'idle'}`}
            Icon={Cpu}
            color={hwColor}
          />
        </div>
      )}

      {/* ── summary stat cards (clickable) ─────────────────────────────────── */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <StatCard
          label="TRACKING CASES"
          value={incLoading ? '—' : trackingCount}
          Icon={Activity}
          color="text-cyan-400"
          border="border-cyan-500/20"
          bg="bg-cyan-500/5"
          onClick={() => navigate('/cases?filter=tracking')}
        />
        <StatCard
          label="PENDING ALERTS"
          value={incLoading ? '—' : pendingAlerts}
          Icon={AlertTriangle}
          color="text-amber-400"
          border={pendingAlerts > 0 ? 'border-amber-500/40' : 'border-amber-500/20'}
          bg={pendingAlerts > 0 ? 'bg-amber-500/10' : 'bg-amber-500/5'}
          onClick={() => navigate('/cases?filter=alerts')}
          pulse={pendingAlerts > 0}
        />
        <StatCard
          label="CASES RESOLVED"
          value={incLoading ? '—' : resolvedCount}
          Icon={CheckCircle2}
          color="text-emerald-400"
          border="border-emerald-500/20"
          bg="bg-emerald-500/5"
          onClick={() => navigate('/cases?filter=resolved')}
        />
      </div>

      {/* ── bottom row ─────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-5 gap-4">
        {/* System status */}
        <div className="col-span-2 bg-gray-900 border border-gray-800 rounded-lg p-5">
          <h2 className="text-[10px] font-mono text-gray-600 tracking-widest mb-4">SYSTEM STATUS</h2>
          <div className="space-y-3">
            <StatusIndicator
              status={metricsError ? 'offline' : 'online'}
              label="Engine API"
              detail={metricsError ? 'unreachable' : `${hwLabel} · ${hwStatus.toLowerCase()}`}
            />
            <StatusIndicator status="online"  label="Embedding Engine" detail="ArcFace / buffalo_l" />
            {/* Camera Sources — live health summary, 30s poll */}
            <div className="flex items-start gap-2.5 py-0.5">
              <span className={`w-1.5 h-1.5 rounded-full mt-[5px] flex-shrink-0 ${
                camHealth.online > 0 ? 'bg-emerald-400' : camHealth.total > 0 ? 'bg-red-400' : 'bg-gray-700'
              }`} />
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-300">Camera Sources</span>
                  <button
                    onClick={() => navigate('/administration')}
                    className="text-[10px] font-mono text-cyan-600 hover:text-cyan-400 transition-colors"
                  >
                    Manage →
                  </button>
                </div>
                <div className="text-[10px] font-mono text-gray-600 mt-0.5">
                  {camHealth.total === 0
                    ? 'none registered'
                    : `● ${camHealth.total} registered  ● ${camHealth.online} online  ◌ ${camHealth.offline} offline`}
                </div>
              </div>
            </div>
            <StatusIndicator
              status={metricsError ? 'offline' : 'online'}
              label="INC API"
              detail="incidents · persons · sightings"
            />
          </div>
          <div className="mt-5 pt-4 border-t border-gray-800">
            <div className="text-[10px] font-mono text-gray-600 tracking-widest mb-2">SESSION</div>
            <div className="grid grid-cols-2 gap-3 text-[11px] font-mono">
              <div>
                <div className="text-gray-600">UPTIME</div>
                <div className="text-gray-300 mt-0.5">
                  {m.uptime_seconds > 0 ? `${uptimeH}h ${uptimeM}m` : '—'}
                </div>
              </div>
              <div>
                <div className="text-gray-600">OPEN CASES</div>
                <div className="text-gray-300 mt-0.5">{openCount + trackingCount}</div>
              </div>
            </div>
          </div>
        </div>

        {/* Recent activity */}
        <div className="col-span-3 bg-gray-900 border border-gray-800 rounded-lg p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-[10px] font-mono text-gray-600 tracking-widest">RECENT ACTIVITY</h2>
            <button
              onClick={() => navigate('/cases')}
              className="flex items-center gap-1 text-[10px] font-mono text-gray-600 hover:text-cyan-400 transition-colors"
            >
              All cases <ArrowRight size={10} />
            </button>
          </div>
          <div className="space-y-0.5 overflow-y-auto max-h-72">
            {activityFeed.map((evt) => {
              const accent =
                evt.type === 'SIGHTING_DETECTED' ? 'text-amber-400' :
                evt.type === 'ALERT_VERIFIED' || evt.type === 'CASE_CLOSED' ? 'text-emerald-400' :
                evt.type === 'CASE_CREATED' ? 'text-cyan-400' : 'text-gray-600'
              const clickable = evt.incident_ref != null
              return (
                <div
                  key={evt.id}
                  onClick={clickable ? () => navigate('/cases') : undefined}
                  className={`flex items-start gap-3 py-2.5 border-b border-gray-800/60 last:border-0 ${clickable ? 'cursor-pointer hover:bg-gray-800/30 -mx-2 px-2 rounded transition-colors' : ''}`}
                >
                  <div className={`text-[10px] font-mono flex-shrink-0 mt-0.5 w-12 ${accent}`}>
                    {new Date(evt.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-300 leading-snug">{evt.message}</p>
                    {evt.incident_ref && (
                      <span className="text-[10px] font-mono text-gray-600">{evt.incident_ref}</span>
                    )}
                  </div>
                  {evt.type === 'SIGHTING_DETECTED' && (
                    <AlertTriangle size={11} className="text-amber-400 flex-shrink-0 mt-1" />
                  )}
                </div>
              )
            })}
            {activityFeed.length === 0 && (
              <div className="text-sm text-gray-600 text-center py-10">No activity yet</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── sub-components ─────────────────────────────────────────────────────────────

function TelemetryTile({ label, value, sub, Icon, color }: {
  label: string; value: string; sub: string; Icon: React.ElementType; color: string
}) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 flex items-start gap-3">
      <Icon size={14} className={`${color} mt-0.5 flex-shrink-0`} />
      <div className="min-w-0">
        <div className="text-[9px] font-mono text-gray-600 tracking-widest mb-1">{label}</div>
        <div className={`text-xl font-mono font-semibold truncate ${color}`}>{value}</div>
        <div className="text-[10px] font-mono text-gray-700 mt-0.5 truncate">{sub}</div>
      </div>
    </div>
  )
}

interface StatCardProps {
  label: string
  value: number | string
  Icon: React.ElementType
  color: string
  border: string
  bg: string
  onClick?: () => void
  pulse?: boolean
}
function StatCard({ label, value, Icon, color, border, bg, onClick, pulse }: StatCardProps) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left ${bg} border ${border} rounded-lg p-5 flex items-center gap-4 transition-all duration-150 ${onClick ? 'hover:brightness-110 active:scale-[0.98] cursor-pointer' : 'cursor-default'} group`}
    >
      <div className="w-10 h-10 bg-gray-900/60 rounded-lg flex items-center justify-center flex-shrink-0">
        <Icon size={18} className={color} />
      </div>
      <div className="flex-1 min-w-0">
        <div className={`text-3xl font-mono font-bold ${color}`}>{value}</div>
        <div className="text-[10px] font-mono text-gray-600 tracking-widest mt-0.5">{label}</div>
      </div>
      {onClick && (
        <ArrowRight size={14} className="text-gray-700 group-hover:text-gray-400 transition-colors flex-shrink-0" />
      )}
      {pulse && (
        <span className={`w-2 h-2 rounded-full ${color.replace('text-', 'bg-')} animate-pulse flex-shrink-0`} />
      )}
    </button>
  )
}
