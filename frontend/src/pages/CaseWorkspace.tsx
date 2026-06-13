import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  ArrowLeft, MapPin, Calendar, User, MessageSquare,
  PauseCircle, CheckCircle2, Send, Loader2, RefreshCw,
} from 'lucide-react'
import Timeline from '../components/Timeline'
import { useIncidentDetail } from '../api/hooks'
import { useAppStore } from '../store/appStore'
import type { IncidentStatus, TimelineEntry } from '../types'

const statusCfg: Record<IncidentStatus, { label: string; color: string; dot: string }> = {
  OPEN:     { label: 'OPEN',     color: 'text-amber-400',   dot: 'bg-amber-400' },
  TRACKING: { label: 'TRACKING', color: 'text-cyan-400',    dot: 'bg-cyan-400 animate-pulse-dot' },
  RESOLVED: { label: 'RESOLVED', color: 'text-emerald-400', dot: 'bg-emerald-400' },
  CLOSED:   { label: 'CLOSED',   color: 'text-gray-500',    dot: 'bg-gray-500' },
}

function Initials({ name }: { name: string }) {
  const parts = name.trim().split(' ')
  const initials =
    parts.length >= 2 ? `${parts[0][0]}${parts[parts.length - 1][0]}` : name.slice(0, 2)
  return (
    <div className="w-24 h-24 rounded-full bg-gray-800 border border-gray-700 flex items-center justify-center font-mono font-semibold text-2xl text-gray-400">
      {initials.toUpperCase()}
    </div>
  )
}

function PersonAvatar({ name, photoUrl }: { name: string; photoUrl: string | null }) {
  const [imgError, setImgError] = useState(false)
  if (photoUrl && !imgError) {
    return (
      <img
        src={photoUrl}
        alt={name}
        onError={() => setImgError(true)}
        className="w-24 h-24 rounded-full object-cover border border-gray-700"
      />
    )
  }
  return <Initials name={name || '?'} />
}

export default function CaseWorkspace() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const { accessMode, incUrl } = useAppStore()
  const { incident, persons, timeline: baseTimeline, loading, error, refetch } = useIncidentDetail(id)

  const [extraEntries, setExtraEntries] = useState<TimelineEntry[]>([])
  const [localStatus, setLocalStatus] = useState<IncidentStatus | null>(null)
  const [comment, setComment] = useState('')

  const backendBase = incUrl.replace(/\/api\/v1\/?$/, '')

  const entries = [
    ...baseTimeline.map((e) => {
      const override = extraEntries.find((ex) => ex.sighting?.id && ex.sighting.id === e.sighting?.id)
      return override ?? e
    }),
    ...extraEntries.filter((e) => !baseTimeline.some((b) => b.id === e.id)),
  ]

  const currentStatus = localStatus ?? incident?.status ?? 'OPEN'
  const cfg = statusCfg[currentStatus]
  const person = persons[0]
  const photoUrl = person?.source_image_path
    ? `${backendBase}/${person.source_image_path}`
    : null

  const addComment = () => {
    if (!comment.trim()) return
    setExtraEntries((prev) => [
      ...prev,
      {
        id: `comment-${Date.now()}`,
        type: 'COMMENT_ADDED' as const,
        timestamp: new Date().toISOString(),
        message: `Operator note: "${comment.trim()}"`,
      },
    ])
    setComment('')
  }

  const confirmSighting = (sightingId: string) => {
    setExtraEntries((prev) => {
      const fromBase = baseTimeline.find((e) => e.sighting?.id === sightingId)
      const verifiedEntry: TimelineEntry = {
        id: `confirm-${sightingId}-${Date.now()}`,
        type: 'ALERT_VERIFIED',
        timestamp: new Date().toISOString(),
        message: 'Alert confirmed — identity match verified by operator',
      }
      // If an override for this sighting already exists in extraEntries, update it
      const existingOverride = prev.find((e) => e.id === fromBase?.id)
      if (existingOverride) {
        return [
          ...prev.map((e) =>
            e.id === fromBase!.id ? { ...e, sighting: { ...e.sighting!, status: 'CONFIRMED' as const } } : e
          ),
          verifiedEntry,
        ]
      }
      // Otherwise create an override from the base entry
      if (fromBase) {
        return [
          ...prev,
          { ...fromBase, sighting: { ...fromBase.sighting!, status: 'CONFIRMED' as const } },
          verifiedEntry,
        ]
      }
      return [...prev, verifiedEntry]
    })
  }

  const rejectSighting = (sightingId: string) => {
    setExtraEntries((prev) => {
      const fromBase = baseTimeline.find((e) => e.sighting?.id === sightingId)
      if (!fromBase) return prev
      const already = prev.find((e) => e.id === fromBase.id)
      if (already) {
        return prev.map((e) =>
          e.id === fromBase.id ? { ...e, sighting: { ...e.sighting!, status: 'REJECTED' as const } } : e
        )
      }
      return [...prev, { ...fromBase, sighting: { ...fromBase.sighting!, status: 'REJECTED' as const } }]
    })
  }

  const patchStatus = async (backendStatus: 'open' | 'active' | 'closed') => {
    if (accessMode !== 'MOCK' && incident) {
      await fetch(`${incUrl}/incidents/${incident.id}/status`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: backendStatus }),
      }).catch(console.error)
    }
  }

  const pauseTracking = () => {
    setLocalStatus('OPEN')
    void patchStatus('open')
    setExtraEntries((prev) => [
      ...prev,
      {
        id: `pause-${Date.now()}`,
        type: 'TRACKING_PAUSED' as const,
        timestamp: new Date().toISOString(),
        message: 'Tracking paused — pipeline suspended, embeddings still registered',
      },
    ])
  }

  const closeCase = () => {
    setLocalStatus('CLOSED')
    void patchStatus('closed')
    setExtraEntries((prev) => [
      ...prev,
      {
        id: `close-${Date.now()}`,
        type: 'CASE_CLOSED' as const,
        timestamp: new Date().toISOString(),
        message: `Case ${incident?.ref ?? ''} closed — embedding search disabled`,
      },
    ])
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-gray-600">
        <Loader2 size={20} className="animate-spin mr-3" />
        <span className="font-mono text-sm">Loading case…</span>
      </div>
    )
  }

  if (error || !incident) {
    return (
      <div className="p-8 text-center">
        <div className="text-sm text-red-400 font-mono mb-4">{error ?? 'Case not found'}</div>
        <button onClick={() => navigate('/cases')} className="text-cyan-400 text-sm hover:underline">
          ← Back to Cases
        </button>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-6 py-4 border-b border-gray-800 flex items-center gap-4 bg-gray-950 flex-shrink-0">
        <button onClick={() => navigate('/cases')} className="text-gray-600 hover:text-gray-300 transition-colors">
          <ArrowLeft size={16} />
        </button>
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <span className="text-xs font-mono text-gray-600">{incident.ref}</span>
          <h1 className="text-sm font-semibold text-gray-200 truncate">{incident.title}</h1>
          <div className={`inline-flex items-center gap-1.5 text-[10px] font-mono px-2 py-0.5 rounded border border-gray-700 flex-shrink-0 ${cfg.color}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} />
            {cfg.label}
          </div>
        </div>
        <button onClick={refetch} className="text-gray-600 hover:text-gray-300 transition-colors flex-shrink-0" title="Refresh">
          <RefreshCw size={14} />
        </button>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Left: Profile */}
        <aside className="w-64 flex-shrink-0 border-r border-gray-800 bg-gray-950 overflow-y-auto p-5">
          <div className="flex flex-col items-center text-center mb-5">
            <PersonAvatar name={person?.name ?? '??'} photoUrl={accessMode === 'MOCK' ? null : photoUrl} />
            <h2 className="text-base font-semibold text-gray-100 mt-3">{person?.name ?? 'Unknown'}</h2>
            <div className="text-xs font-mono text-gray-600 mt-0.5">
              {person?.age ? `${person.age} yrs` : ''}{person?.age && person?.gender ? ' · ' : ''}{person?.gender ?? ''}
            </div>
          </div>
          <div className="space-y-3 text-[12px]">
            {person?.description && (
              <div>
                <div className="text-[10px] font-mono text-gray-600 tracking-widest mb-1">DESCRIPTION</div>
                <p className="text-gray-400 leading-relaxed">{person.description}</p>
              </div>
            )}
            <div className="border-t border-gray-800 pt-3">
              <div className="text-[10px] font-mono text-gray-600 tracking-widest mb-2">LAST SEEN</div>
              <div className="flex items-start gap-2 text-gray-400 mb-1.5">
                <MapPin size={11} className="text-gray-600 mt-0.5 flex-shrink-0" />
                <span>{incident.last_seen_location}</span>
              </div>
              <div className="flex items-center gap-2 text-gray-500 font-mono">
                <Calendar size={11} className="text-gray-600 flex-shrink-0" />
                <span>
                  {new Date(incident.last_seen_at).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })}
                  {' '}{new Date(incident.last_seen_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>
            </div>
            <div className="border-t border-gray-800 pt-3">
              <div className="text-[10px] font-mono text-gray-600 tracking-widest mb-2">CASE INFO</div>
              <div className="space-y-1.5 font-mono text-[10px] text-gray-500">
                {[
                  ['Opened', new Date(incident.created_at).toLocaleDateString('en-GB')],
                  ['Ref', incident.ref],
                  ['Enrolled', `${incident.person_count} person`],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between">
                    <span className="text-gray-600">{k}</span>
                    <span>{v}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </aside>

        {/* Center: Timeline */}
        <div className="flex-1 overflow-y-auto p-6 min-w-0">
          <div className="flex items-center justify-between mb-5">
            <h3 className="text-[10px] font-mono text-gray-600 tracking-widest">CASE TIMELINE</h3>
            <span className="text-[10px] font-mono text-gray-700">{entries.length} events</span>
          </div>
          <Timeline entries={entries} onConfirmSighting={confirmSighting} onRejectSighting={rejectSighting} />
        </div>

        {/* Right: Actions */}
        <aside className="w-64 flex-shrink-0 border-l border-gray-800 bg-gray-950 overflow-y-auto p-5">
          <h3 className="text-[10px] font-mono text-gray-600 tracking-widest mb-5">ACTIONS</h3>
          <div className="space-y-4">
            <div>
              <div className="text-[10px] font-mono text-gray-600 tracking-wider mb-2 flex items-center gap-1.5">
                <MessageSquare size={9} /> ADD COMMENT
              </div>
              <textarea
                value={comment}
                onChange={(e) => setComment(e.target.value)}
                rows={3}
                className="w-full bg-gray-900 border border-gray-700 focus:border-cyan-600/50 rounded px-3 py-2 text-xs text-gray-300 outline-none resize-none transition-colors placeholder-gray-700"
                placeholder="Operator note…"
                onKeyDown={(e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) addComment() }}
              />
              <button
                onClick={addComment}
                disabled={!comment.trim()}
                className="mt-2 w-full flex items-center justify-center gap-2 py-2 bg-gray-800 border border-gray-700 text-gray-400 rounded text-xs hover:bg-gray-700 hover:text-gray-200 transition-colors disabled:opacity-30 disabled:pointer-events-none"
              >
                <Send size={11} /> Add Note
              </button>
            </div>

            <div className="border-t border-gray-800 pt-4">
              <div className="text-[10px] font-mono text-gray-600 tracking-wider mb-2 flex items-center gap-1.5">
                <User size={9} /> CASE ACTIONS
              </div>
              <div className="space-y-2">
                {currentStatus === 'TRACKING' && (
                  <ActionBtn icon={PauseCircle} label="Pause Tracking" onClick={pauseTracking} variant="warn" />
                )}
                {(currentStatus === 'OPEN' || currentStatus === 'TRACKING') && (
                  <ActionBtn icon={CheckCircle2} label="Resolve & Close" onClick={closeCase} variant="success" />
                )}
                {(currentStatus === 'RESOLVED' || currentStatus === 'CLOSED') && (
                  <div className="text-center text-[11px] font-mono text-gray-600 py-3">Case is resolved</div>
                )}
              </div>
            </div>

            <div className="border-t border-gray-800 pt-4">
              <div className="text-[10px] font-mono text-gray-600 tracking-wider mb-3">QUICK STATS</div>
              <div className="space-y-2 font-mono text-[11px]">
                {[
                  ['Timeline events', entries.length],
                  ['Alerts generated', entries.filter((e) => e.type === 'SIGHTING_DETECTED').length],
                  ['Alerts confirmed', entries.filter((e) => e.type === 'ALERT_VERIFIED').length],
                ].map(([label, value]) => (
                  <div key={String(label)} className="flex justify-between text-gray-600">
                    <span>{label}</span><span className="text-gray-400">{value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  )
}

function ActionBtn({ icon: Icon, label, onClick, variant }: {
  icon: React.ElementType; label: string; onClick: () => void; variant: 'warn' | 'success'
}) {
  const cls = {
    warn:    'border-amber-500/30 bg-amber-500/8 text-amber-400 hover:bg-amber-500/15',
    success: 'border-emerald-500/30 bg-emerald-500/8 text-emerald-400 hover:bg-emerald-500/15',
  }[variant]
  return (
    <motion.button
      whileTap={{ scale: 0.97 }}
      onClick={onClick}
      className={`w-full flex items-center gap-2 px-3 py-2.5 border rounded text-xs font-medium transition-colors ${cls}`}
    >
      <Icon size={13} />{label}
    </motion.button>
  )
}
