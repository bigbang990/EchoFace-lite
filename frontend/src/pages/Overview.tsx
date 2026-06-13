import { useMemo } from 'react'
import { Activity, AlertTriangle, CheckCircle2, TrendingUp, Zap, Timer, Layers, Cpu } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { useIncidents, useSystemMetrics, useCameras, deriveActivityFeed } from '../api/hooks'
import { mockActivityFeed } from '../mock/data'
import StatusIndicator from '../components/StatusIndicator'

export default function Overview() {
  const { accessMode } = useAppStore()
  const { data: incidents, loading: incLoading } = useIncidents()
  const { data: m, error: metricsError } = useSystemMetrics()
  const { data: cameras } = useCameras()

  const isAdmin = accessMode === 'ADMIN'

  const trackingCount = incidents.filter((i) => i.status === 'TRACKING').length
  const openCount     = incidents.filter((i) => i.status === 'OPEN').length
  const resolvedCount = incidents.filter((i) => i.status === 'RESOLVED').length
  const pendingAlerts = incidents.reduce((s, i) => s + i.alert_count, 0)
  const activeCams    = cameras.filter((c) => c.status === 'ACTIVE').length

  const activityFeed = useMemo(
    () => (accessMode === 'MOCK' ? mockActivityFeed : deriveActivityFeed(incidents)),
    [incidents, accessMode]
  )

  const uptimeH = Math.floor(m.uptime_seconds / 3600)
  const uptimeM = Math.floor((m.uptime_seconds % 3600) / 60)

  return (
    <div className="p-8 max-w-6xl">
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
            LIVE · 3s refresh
          </div>
        )}
        {metricsError && (
          <div className="text-[10px] font-mono text-red-400 flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
            {metricsError}
          </div>
        )}
      </div>

      {isAdmin && (
        <div className="grid grid-cols-4 gap-3 mb-6">
          {[
            { label: 'AVG FPS',       value: m.fps.toFixed(1),            sub: 'frames / sec',           Icon: Zap,      color: 'text-cyan-400' },
            { label: 'DETECTOR',      value: `${m.detector_latency_ms}ms`,sub: 'avg latency',            Icon: Timer,    color: 'text-cyan-400' },
            { label: 'ACTIVE TRACKS', value: m.active_tracks,             sub: 'identities in-frame',    Icon: Layers,   color: 'text-cyan-400' },
            {
              label: 'GPU BACKEND',
              value: m.gpu_status,
              sub: `hardware type ${m.hardware_backend_type}`,
              Icon: Cpu,
              color: m.gpu_status === 'OK' ? 'text-emerald-400' : 'text-red-400',
            },
          ].map((tile) => (
            <div key={tile.label} className="bg-gray-900 border border-gray-800 rounded-lg p-4 flex items-start gap-3">
              <tile.Icon size={15} className={`${tile.color} mt-0.5 flex-shrink-0`} />
              <div>
                <div className="text-[9px] font-mono text-gray-600 tracking-widest mb-1">{tile.label}</div>
                <div className={`text-xl font-mono font-semibold ${tile.color}`}>{tile.value}</div>
                <div className="text-[10px] font-mono text-gray-700 mt-0.5">{tile.sub}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="grid grid-cols-3 gap-4 mb-6">
        <StatCard label="TRACKING CASES" value={incLoading ? '—' : trackingCount} Icon={Activity}    color="text-cyan-400"    border="border-cyan-500/20"    bg="bg-cyan-500/5" />
        <StatCard label="PENDING ALERTS" value={incLoading ? '—' : pendingAlerts} Icon={AlertTriangle} color="text-amber-400"   border="border-amber-500/20"   bg="bg-amber-500/5" />
        <StatCard label="CASES RESOLVED" value={incLoading ? '—' : resolvedCount} Icon={CheckCircle2}  color="text-emerald-400" border="border-emerald-500/20" bg="bg-emerald-500/5" />
      </div>

      <div className="grid grid-cols-5 gap-4">
        <div className="col-span-2 bg-gray-900 border border-gray-800 rounded-lg p-5">
          <h2 className="text-[10px] font-mono text-gray-600 tracking-widest mb-4">SYSTEM STATUS</h2>
          <div className="space-y-3">
            <StatusIndicator status="online"  label="Tracking Engine"  detail="SORT v2.1 — cpu path" />
            <StatusIndicator status="online"  label="Embedding Engine" detail="ArcFace / buffalo_l" />
            <StatusIndicator
              status={activeCams > 0 ? 'online' : 'offline'}
              label="Camera Sources"
              detail={`${activeCams} / ${cameras.length} active`}
            />
            <StatusIndicator status={metricsError ? 'offline' : 'online'} label="API Gateway" detail="FastAPI 0.104 / ngrok" />
          </div>
          <div className="mt-5 pt-4 border-t border-gray-800">
            <div className="text-[10px] font-mono text-gray-600 tracking-widest mb-2">SESSION</div>
            <div className="grid grid-cols-2 gap-3 text-[11px] font-mono">
              <div>
                <div className="text-gray-600">UPTIME</div>
                <div className="text-gray-300 mt-0.5">{uptimeH}h {uptimeM}m</div>
              </div>
              <div>
                <div className="text-gray-600">OPEN CASES</div>
                <div className="text-gray-300 mt-0.5">{openCount + trackingCount}</div>
              </div>
            </div>
          </div>
        </div>

        <div className="col-span-3 bg-gray-900 border border-gray-800 rounded-lg p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-[10px] font-mono text-gray-600 tracking-widest">RECENT ACTIVITY</h2>
            <TrendingUp size={12} className="text-gray-700" />
          </div>
          <div className="space-y-0.5 overflow-y-auto max-h-80">
            {activityFeed.map((evt) => {
              const accent =
                evt.type === 'SIGHTING_DETECTED' ? 'text-amber-400' :
                evt.type === 'ALERT_VERIFIED' || evt.type === 'CASE_CLOSED' ? 'text-emerald-400' :
                evt.type === 'CASE_CREATED' ? 'text-cyan-400' : 'text-gray-600'
              return (
                <div key={evt.id} className="flex items-start gap-3 py-2.5 border-b border-gray-800/60 last:border-0">
                  <div className={`text-[10px] font-mono flex-shrink-0 mt-0.5 w-12 ${accent}`}>
                    {new Date(evt.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-300 leading-snug">{evt.message}</p>
                    {evt.incident_ref && (
                      <span className="text-[10px] font-mono text-gray-600">{evt.incident_ref}</span>
                    )}
                  </div>
                </div>
              )
            })}
            {activityFeed.length === 0 && (
              <div className="text-sm text-gray-600 text-center py-8">No activity yet</div>
            )}
          </div>
        </div>
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
}
function StatCard({ label, value, Icon, color, border, bg }: StatCardProps) {
  return (
    <div className={`${bg} border ${border} rounded-lg p-5 flex items-center gap-4`}>
      <div className="w-10 h-10 bg-gray-900/60 rounded-lg flex items-center justify-center flex-shrink-0">
        <Icon size={18} className={color} />
      </div>
      <div>
        <div className={`text-3xl font-mono font-bold ${color}`}>{value}</div>
        <div className="text-[10px] font-mono text-gray-600 tracking-widest mt-0.5">{label}</div>
      </div>
    </div>
  )
}
