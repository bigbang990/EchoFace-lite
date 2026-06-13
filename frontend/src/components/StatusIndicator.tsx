type StatusLevel = 'online' | 'offline' | 'degraded' | 'unknown'

interface Props {
  status: StatusLevel
  label: string
  detail?: string
}

const cfg: Record<StatusLevel, { dot: string; text: string; badge: string }> = {
  online:   { dot: 'bg-emerald-400', text: 'text-emerald-400', badge: 'ONLINE' },
  offline:  { dot: 'bg-red-400',     text: 'text-red-400',     badge: 'OFFLINE' },
  degraded: { dot: 'bg-amber-400',   text: 'text-amber-400',   badge: 'DEGRADED' },
  unknown:  { dot: 'bg-gray-600',    text: 'text-gray-600',    badge: 'UNKNOWN' },
}

export default function StatusIndicator({ status, label, detail }: Props) {
  const c = cfg[status]
  return (
    <div className="flex items-center gap-3 py-1">
      <div className={`w-2 h-2 rounded-full flex-shrink-0 ${c.dot} ${status === 'online' ? 'animate-pulse-dot' : ''}`} />
      <div className="flex-1 min-w-0">
        <div className="text-sm text-gray-300 leading-tight">{label}</div>
        {detail && <div className="text-[10px] font-mono text-gray-600 mt-0.5">{detail}</div>}
      </div>
      <div className={`text-[9px] font-mono tracking-wider flex-shrink-0 ${c.text}`}>
        {c.badge}
      </div>
    </div>
  )
}
