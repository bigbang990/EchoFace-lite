import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  FolderOpen,
  UserPlus,
  Radio,
  AlertTriangle,
  CheckCircle2,
  MessageSquare,
  XCircle,
  PauseCircle,
  Cpu,
} from 'lucide-react'
import type { TimelineEntry, Sighting } from '../types'

const typeCfg = {
  CASE_CREATED:        { Icon: FolderOpen,     dot: 'bg-gray-700',         icon: 'text-gray-400' },
  PERSON_ENROLLED:     { Icon: UserPlus,       dot: 'bg-blue-900/60',      icon: 'text-blue-400' },
  EMBEDDINGS_GENERATED:{ Icon: Cpu,            dot: 'bg-violet-900/50',    icon: 'text-violet-400' },
  TRACKING_STARTED:    { Icon: Radio,          dot: 'bg-cyan-900/50',      icon: 'text-cyan-400' },
  SIGHTING_DETECTED:   { Icon: AlertTriangle,  dot: 'bg-amber-900/50',     icon: 'text-amber-400' },
  ALERT_VERIFIED:      { Icon: CheckCircle2,   dot: 'bg-emerald-900/50',   icon: 'text-emerald-400' },
  COMMENT_ADDED:       { Icon: MessageSquare,  dot: 'bg-violet-900/40',    icon: 'text-violet-400' },
  CASE_CLOSED:         { Icon: XCircle,        dot: 'bg-gray-800',         icon: 'text-gray-600' },
  TRACKING_PAUSED:     { Icon: PauseCircle,    dot: 'bg-amber-900/30',     icon: 'text-amber-500' },
} as const

interface AlertCardProps {
  sighting: Sighting
  onConfirm: (id: string) => void
  onReject: (id: string) => void
}

function AlertCard({ sighting, onConfirm, onReject }: AlertCardProps) {
  const pct = Math.round(sighting.confidence * 100)
  const barColor = pct >= 85 ? 'bg-emerald-500' : pct >= 65 ? 'bg-amber-500' : 'bg-red-500'
  const pctColor = pct >= 85 ? 'text-emerald-400' : pct >= 65 ? 'text-amber-400' : 'text-red-400'

  return (
    <div className="mt-3 border border-amber-500/25 bg-amber-500/5 rounded-lg p-4">
      <div className="flex gap-4">
        <div className="w-[72px] h-[72px] flex-shrink-0 bg-gray-900 border border-gray-700 rounded flex flex-col items-center justify-center gap-1">
          <div className="text-[8px] font-mono text-gray-600">FRAME</div>
          <div className="text-[10px] font-mono text-gray-500">
            #{String(sighting.frame_index).padStart(5, '0')}
          </div>
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-2.5">
            <span className="text-[10px] font-mono text-gray-500 uppercase">Match</span>
            <span className="text-sm font-semibold text-gray-100">{sighting.person_name}</span>
          </div>

          <div className="mb-2.5">
            <div className="flex justify-between text-[10px] font-mono mb-1">
              <span className="text-gray-600">CONFIDENCE</span>
              <span className={pctColor}>{pct}%</span>
            </div>
            <div className="h-1 bg-gray-800 rounded-full overflow-hidden">
              <div className={`h-full ${barColor} rounded-full`} style={{ width: `${pct}%` }} />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-x-4 text-[10px] font-mono">
            <div>
              <div className="text-gray-600">SOURCE</div>
              <div className="text-gray-400 truncate">{sighting.source_name}</div>
            </div>
            <div>
              <div className="text-gray-600">DETECTED</div>
              <div className="text-gray-400">
                {new Date(sighting.timestamp).toLocaleTimeString([], {
                  hour: '2-digit',
                  minute: '2-digit',
                  second: '2-digit',
                })}
              </div>
            </div>
          </div>
        </div>
      </div>

      {sighting.status === 'PENDING' && (
        <div className="flex gap-2 mt-3 pt-3 border-t border-amber-500/15">
          <button
            onClick={() => onConfirm(sighting.id)}
            className="flex-1 py-2 text-[11px] font-mono font-semibold tracking-wider bg-emerald-500/15 border border-emerald-500/35 text-emerald-400 rounded hover:bg-emerald-500/25 transition-colors"
          >
            CONFIRM MATCH
          </button>
          <button
            onClick={() => onReject(sighting.id)}
            className="flex-1 py-2 text-[11px] font-mono font-semibold tracking-wider bg-red-500/10 border border-red-500/25 text-red-400 rounded hover:bg-red-500/20 transition-colors"
          >
            REJECT
          </button>
        </div>
      )}

      {sighting.status === 'CONFIRMED' && (
        <div className="mt-3 pt-3 border-t border-emerald-500/20 flex items-center justify-center gap-2 text-[11px] font-mono text-emerald-400 tracking-wider">
          <CheckCircle2 size={12} />
          MATCH CONFIRMED
        </div>
      )}

      {sighting.status === 'REJECTED' && (
        <div className="mt-3 pt-3 border-t border-red-500/20 flex items-center justify-center gap-2 text-[11px] font-mono text-red-400 tracking-wider">
          <XCircle size={12} />
          REJECTED
        </div>
      )}
    </div>
  )
}

interface Props {
  entries: TimelineEntry[]
  onConfirmSighting?: (id: string) => void
  onRejectSighting?: (id: string) => void
}

export default function Timeline({ entries, onConfirmSighting, onRejectSighting }: Props) {
  const [expanded, setExpanded] = useState<Set<string>>(
    new Set(
      entries
        .filter((e) => e.type === 'SIGHTING_DETECTED' && e.sighting?.status === 'PENDING')
        .map((e) => e.id)
    )
  )

  const toggle = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })

  return (
    <div>
      {entries.map((entry, i) => {
        const c = typeCfg[entry.type] ?? typeCfg.CASE_CREATED
        const isLast = i === entries.length - 1
        const isExpandable = entry.type === 'SIGHTING_DETECTED' && !!entry.sighting
        const isExpanded = expanded.has(entry.id)

        return (
          <motion.div
            key={entry.id}
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.04 }}
            className="relative flex gap-4"
          >
            <div className="flex flex-col items-center flex-shrink-0 w-8">
              <div
                className={`w-8 h-8 rounded-full ${c.dot} flex items-center justify-center z-10`}
              >
                <c.Icon size={13} className={c.icon} />
              </div>
              {!isLast && <div className="w-px flex-1 bg-gray-800/70 mt-1" style={{ minHeight: 20 }} />}
            </div>

            <div className="flex-1 pb-5 min-w-0">
              <button
                className={`w-full text-left ${isExpandable ? 'cursor-pointer' : 'cursor-default'}`}
                onClick={() => isExpandable && toggle(entry.id)}
              >
                <div className="flex items-start justify-between gap-3">
                  <p className="text-sm text-gray-300 leading-snug">{entry.message}</p>
                  <span className="text-[10px] font-mono text-gray-600 whitespace-nowrap flex-shrink-0 mt-0.5">
                    {new Date(entry.timestamp).toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </span>
                </div>
                {isExpandable && (
                  <span className="text-[10px] font-mono text-amber-500/60 mt-1 block">
                    {isExpanded ? '▲ collapse details' : '▼ view alert details'}
                  </span>
                )}
              </button>

              <AnimatePresence>
                {isExpandable && isExpanded && entry.sighting && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    style={{ overflow: 'hidden' }}
                  >
                    <AlertCard
                      sighting={entry.sighting}
                      onConfirm={onConfirmSighting ?? (() => {})}
                      onReject={onRejectSighting ?? (() => {})}
                    />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        )
      })}
    </div>
  )
}
