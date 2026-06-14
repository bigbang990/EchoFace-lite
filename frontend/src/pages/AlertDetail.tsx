import { useState, useMemo, useEffect } from 'react'

const CONFIDENCE_FLOOR = 0.65  // mirrors ALERT_MIN_CONFIDENCE_FLOOR on the backend
import { useParams, useNavigate, Link } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  ArrowLeft, Camera, Clock, Film, Crosshair, CheckCircle2,
  XCircle, AlertTriangle, MapPin, User, Send, ChevronRight,
  ExternalLink, RefreshCw, Loader2,
} from 'lucide-react'
import { useIncidentDetail, useAlert } from '../api/hooks'
import { useAppStore } from '../store/appStore'
import ImageZoomModal from '../components/ImageZoomModal'
import type { Sighting } from '../types'

const normPath = (p: string) => p.replace(/^[/\\]/, '').replace(/\\/g, '/')

function buildUrl(path: string, base: string) {
  if (path.startsWith('http')) return path
  return `${base}/${normPath(path)}`
}

function confidenceColor(pct: number) {
  if (pct >= 85) return { bar: 'bg-emerald-500', text: 'text-emerald-400', border: 'border-emerald-500/30', bg: 'bg-emerald-500/10' }
  if (pct >= 65) return { bar: 'bg-amber-500',   text: 'text-amber-400',   border: 'border-amber-500/30',   bg: 'bg-amber-500/10'   }
  return              { bar: 'bg-red-500',        text: 'text-red-400',     border: 'border-red-500/30',     bg: 'bg-red-500/10'     }
}

function statusBadge(status: Sighting['status']) {
  switch (status) {
    case 'CONFIRMED': return { label: 'CONFIRMED', cls: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10', Icon: CheckCircle2 }
    case 'REJECTED':  return { label: 'REJECTED',  cls: 'text-red-400 border-red-500/30 bg-red-500/10',             Icon: XCircle }
    default:          return { label: 'PENDING',   cls: 'text-amber-400 border-amber-500/30 bg-amber-500/10',       Icon: AlertTriangle }
  }
}

function fmt(ts: string) {
  const d = new Date(ts)
  return {
    date: d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }),
    time: d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
  }
}

// ── EnrolledPhoto ─────────────────────────────────────────────────────────────

function EnrolledPhoto({ src, label, onClick }: { src: string; label: string; onClick: () => void }) {
  const [err, setErr] = useState(false)
  return (
    <button
      onClick={onClick}
      className="group relative w-full aspect-square bg-gray-900 border border-gray-700 rounded overflow-hidden hover:border-cyan-600/50 transition-colors"
    >
      {err ? (
        <div className="w-full h-full flex items-center justify-center text-[9px] font-mono text-gray-700">NO IMAGE</div>
      ) : (
        <img src={src} alt={label} onError={() => setErr(true)} className="w-full h-full object-cover group-hover:opacity-80 transition-opacity" />
      )}
      <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-[9px] font-mono text-gray-300 px-1 py-0.5 truncate opacity-0 group-hover:opacity-100 transition-opacity">
        {label}
      </div>
    </button>
  )
}

// ── HistoryRow ────────────────────────────────────────────────────────────────

function HistoryRow({
  s,
  isActive,
  incidentId,
}: {
  s: Sighting
  isActive: boolean
  incidentId: string
}) {
  const pct = Math.round(s.confidence * 100)
  const c = confidenceColor(pct)
  const badge = statusBadge(s.status)
  const { time } = fmt(s.timestamp)
  return (
    <Link
      to={`/cases/${incidentId}/alerts/${s.id}`}
      className={`flex items-center gap-3 px-3 py-2.5 rounded border transition-colors ${
        isActive
          ? 'border-cyan-600/40 bg-cyan-600/8'
          : 'border-transparent hover:border-gray-700 hover:bg-gray-900/60'
      }`}
    >
      <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
        s.status === 'CONFIRMED' ? 'bg-emerald-500' :
        s.status === 'REJECTED'  ? 'bg-red-500' : 'bg-amber-500 animate-pulse'
      }`} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className={`text-[11px] font-mono font-semibold ${c.text}`}>{pct}%</span>
          <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded border ${badge.cls}`}>{badge.label}</span>
        </div>
        <div className="text-[10px] font-mono text-gray-600 mt-0.5 truncate">
          {time} · {s.source_name}
        </div>
      </div>
      {isActive && <ChevronRight size={11} className="text-cyan-500 flex-shrink-0" />}
    </Link>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function AlertDetail() {
  const { id: incidentId, sightingId } = useParams<{ id: string; sightingId: string }>()
  const navigate = useNavigate()
  const { accessMode, incUrl } = useAppStore()
  const { incident, persons, sightings, loading, error, refetch } = useIncidentDetail(incidentId)
  const { alert: alertData, refetch: refetchAlert } = useAlert(sightingId)
  const backendBase = incUrl.replace(/\/api\/v1\/?$/, '')

  const [note, setNote] = useState('')
  const [notes, setNotes] = useState<string[]>([])
  const [localStatus, setLocalStatus] = useState<Sighting['status'] | null>(null)

  // Seed notes from persisted operator_notes on first load
  useEffect(() => {
    if (!alertData?.operator_notes) return
    const lines = alertData.operator_notes
      .split('\n')
      .filter(l => l.trim().length > 0)
    setNotes(lines)
  }, [alertData?.operator_notes])
  const [saving, setSaving] = useState(false)
  const [zoomImages, setZoomImages] = useState<Array<{ src: string; alt: string }> | null>(null)
  const [zoomIdx, setZoomIdx] = useState(0)

  const sighting = useMemo(
    () => sightings.find(s => String(s.id) === String(sightingId)) ?? null,
    [sightings, sightingId]
  )

  // All sightings for the same person, newest first
  const personHistory = useMemo(() => {
    if (!sighting) return []
    return sightings
      .filter(s => s.person_id === sighting.person_id)
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
  }, [sightings, sighting])

  // Reference photos for the matched person
  const personRecord = useMemo(
    () => persons.find(p => String(p.id) === String(sighting?.person_id)),
    [persons, sighting]
  )

  const refPhotos = useMemo(() => {
    if (!personRecord) return []
    const imgs: Array<{ src: string; alt: string }> = []
    if (personRecord.source_image_path)
      imgs.push({ src: buildUrl(personRecord.source_image_path, backendBase), alt: `${personRecord.name} — reference` })
    for (const ep of personRecord.extra_photo_paths ?? [])
      imgs.push({ src: buildUrl(ep, backendBase), alt: `${personRecord.name} — extra` })
    return imgs
  }, [personRecord, backendBase])

  const effectiveStatus = localStatus ?? sighting?.status ?? 'PENDING'
  const caseClosed = incident?.status === 'CLOSED' || alertData?.incident_status === 'closed'
  const badge = statusBadge(effectiveStatus)
  const snapUrl = sighting?.snapshot_path ? buildUrl(sighting.snapshot_path, backendBase) : null
  const pct = sighting ? Math.round(sighting.confidence * 100) : 0
  const c = confidenceColor(pct)
  const ts = sighting ? fmt(sighting.timestamp) : null

  const doConfirm = async () => {
    if (!sighting || !incident) return
    setLocalStatus('CONFIRMED')
    setSaving(true)
    if (accessMode !== 'MOCK') {
      await fetch(`${incUrl}/incidents/${incident.id}/sightings/${sighting.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: 'confirmed' }),
      }).catch(console.error)
      await refetch()
    }
    setSaving(false)
  }

  const doReject = async () => {
    if (!sighting || !incident) return
    setLocalStatus('REJECTED')
    setSaving(true)
    if (accessMode !== 'MOCK') {
      await fetch(`${incUrl}/incidents/${incident.id}/sightings/${sighting.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: 'rejected' }),
      }).catch(console.error)
      await refetch()
    }
    setSaving(false)
  }

  const addNote = async () => {
    if (!note.trim() || !sightingId) return
    const text = note.trim()
    setNote('')
    if (accessMode !== 'MOCK') {
      await fetch(`${incUrl}/alerts/${sightingId}/notes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': '1' },
        body: JSON.stringify({ note: text }),
      }).catch(console.error)
      await refetchAlert()
    } else {
      // MOCK: optimistic local append with fake timestamp
      const stamp = new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC'
      setNotes(prev => [...prev, `[${stamp}] ${text}`])
    }
  }

  const openZoom = (images: Array<{ src: string; alt: string }>, idx: number) => {
    setZoomImages(images)
    setZoomIdx(idx)
  }

  // ── Loading / error states ────────────────────────────────────────────────

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-gray-600">
        <Loader2 size={18} className="animate-spin mr-3" />
        <span className="font-mono text-sm">Loading alert…</span>
      </div>
    )
  }

  if (error || !incident) {
    return (
      <div className="p-8 text-center">
        <div className="text-sm text-red-400 font-mono mb-4">{error ?? 'Case not found'}</div>
        <button onClick={() => navigate(`/cases/${incidentId}`)} className="text-cyan-400 text-sm hover:underline">
          ← Back to Case
        </button>
      </div>
    )
  }

  if (!sighting) {
    return (
      <div className="p-8 text-center">
        <div className="text-sm text-red-400 font-mono mb-4">Alert not found in this case.</div>
        <button onClick={() => navigate(`/cases/${incidentId}`)} className="text-cyan-400 text-sm hover:underline">
          ← Back to Case
        </button>
      </div>
    )
  }

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-full bg-gray-950">
      {/* Header */}
      <div className="px-6 py-3.5 border-b border-gray-800 flex items-center gap-3 flex-shrink-0">
        <button
          onClick={() => navigate(`/cases/${incidentId}`)}
          className="text-gray-600 hover:text-gray-300 transition-colors flex-shrink-0"
        >
          <ArrowLeft size={16} />
        </button>

        <div className="flex items-center gap-2 text-[11px] font-mono text-gray-600 flex-shrink-0">
          <Link to={`/cases/${incidentId}`} className="hover:text-gray-300 transition-colors">
            {incident.ref}
          </Link>
          <span>/</span>
          <span className="text-gray-500">Alert #{String(sightingId).slice(-6)}</span>
        </div>

        <div className={`inline-flex items-center gap-1.5 text-[10px] font-mono px-2 py-0.5 rounded border ${badge.cls} flex-shrink-0`}>
          <badge.Icon size={10} />
          {badge.label}
        </div>

        <div className="flex-1" />

        {caseClosed && (
          <div className="flex items-center gap-1.5 text-[10px] font-mono text-gray-500 border border-gray-700 px-2 py-1 rounded flex-shrink-0">
            CASE CLOSED — READ ONLY
          </div>
        )}

        {!caseClosed && effectiveStatus === 'PENDING' && (
          <div className="flex items-center gap-2 flex-shrink-0">
            <button
              onClick={doConfirm}
              disabled={saving}
              className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-mono font-semibold bg-emerald-500/15 border border-emerald-500/35 text-emerald-400 rounded hover:bg-emerald-500/25 transition-colors disabled:opacity-40"
            >
              <CheckCircle2 size={11} /> CONFIRM MATCH
            </button>
            <button
              onClick={doReject}
              disabled={saving}
              className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-mono font-semibold bg-red-500/10 border border-red-500/25 text-red-400 rounded hover:bg-red-500/20 transition-colors disabled:opacity-40"
            >
              <XCircle size={11} /> REJECT
            </button>
          </div>
        )}

        {effectiveStatus !== 'PENDING' && (
          <div className={`flex items-center gap-1.5 text-[11px] font-mono ${badge.cls} px-2 py-1 rounded border`}>
            <badge.Icon size={11} />
            {effectiveStatus === 'CONFIRMED' ? 'Match verified by operator' : 'Marked as false positive'}
          </div>
        )}

        <button onClick={refetch} className="text-gray-600 hover:text-gray-300 transition-colors flex-shrink-0" title="Refresh">
          <RefreshCw size={13} />
        </button>
      </div>

      {/* Body */}
      <div className="flex-1 flex overflow-hidden">

        {/* ── Left: Evidence panel ────────────────────────────────────────── */}
        <aside className="w-56 flex-shrink-0 border-r border-gray-800 overflow-y-auto p-4 space-y-5">

          {/* Detected face crop */}
          <div>
            <div className="text-[9px] font-mono text-gray-600 tracking-widest mb-2">DETECTED FACE</div>
            <div className="w-full aspect-square bg-gray-900 border border-gray-700 rounded overflow-hidden">
              {snapUrl ? (
                <motion.img
                  src={snapUrl}
                  alt="Detected face"
                  className="w-full h-full object-cover cursor-zoom-in hover:opacity-90 transition-opacity"
                  whileHover={{ scale: 1.03 }}
                  onClick={() => {
                    const allSnaps = personHistory
                      .filter(s => s.snapshot_path)
                      .map(s => ({ src: buildUrl(s.snapshot_path!, backendBase), alt: `Frame #${s.frame_index}` }))
                    const activeIdx = allSnaps.findIndex(s => s.src === snapUrl)
                    openZoom(allSnaps, activeIdx >= 0 ? activeIdx : 0)
                  }}
                />
              ) : (
                <div className="w-full h-full flex flex-col items-center justify-center gap-1">
                  <AlertTriangle size={20} className="text-gray-700" />
                  <span className="text-[9px] font-mono text-gray-700">NO SNAPSHOT</span>
                </div>
              )}
            </div>
            <div className="text-[9px] font-mono text-gray-700 text-center mt-1">
              Frame #{String(sighting.frame_index).padStart(5, '0')}
            </div>
          </div>

          {/* All snapshots of this person (mini filmstrip) */}
          {personHistory.filter(s => s.snapshot_path).length > 1 && (
            <div>
              <div className="text-[9px] font-mono text-gray-600 tracking-widest mb-2">
                ALL SNAPSHOTS ({personHistory.filter(s => s.snapshot_path).length})
              </div>
              <div className="grid grid-cols-3 gap-1">
                {personHistory
                  .filter(s => s.snapshot_path)
                  .slice(0, 9)
                  .map((s, i) => {
                    const url = buildUrl(s.snapshot_path!, backendBase)
                    const isActive = String(s.id) === String(sightingId)
                    const allSnaps = personHistory
                      .filter(x => x.snapshot_path)
                      .map(x => ({ src: buildUrl(x.snapshot_path!, backendBase), alt: `Frame #${x.frame_index}` }))
                    return (
                      <button
                        key={s.id}
                        onClick={() => openZoom(allSnaps, i)}
                        className={`aspect-square bg-gray-900 rounded overflow-hidden border transition-colors ${
                          isActive ? 'border-cyan-500/60' : 'border-gray-800 hover:border-gray-600'
                        }`}
                      >
                        <img src={url} alt={`snap ${i}`} className="w-full h-full object-cover" />
                      </button>
                    )
                  })}
              </div>
            </div>
          )}

          {/* Reference (enrolled) photos */}
          {refPhotos.length > 0 && (
            <div>
              <div className="text-[9px] font-mono text-gray-600 tracking-widest mb-2">
                REFERENCE PHOTOS ({refPhotos.length})
              </div>
              <div className="grid grid-cols-2 gap-1.5">
                {refPhotos.map((img, i) => (
                  <EnrolledPhoto
                    key={img.src}
                    src={img.src}
                    label={img.alt}
                    onClick={() => openZoom(refPhotos, i)}
                  />
                ))}
              </div>
            </div>
          )}
        </aside>

        {/* ── Center: Sighting record ─────────────────────────────────────── */}
        <div className="flex-1 overflow-y-auto p-6 min-w-0">
          <div className="max-w-xl">

            {/* Person identity block */}
            <div className="flex items-start gap-4 mb-6">
              <div className={`w-12 h-12 rounded-full border-2 ${c.border} flex items-center justify-center flex-shrink-0`}>
                <User size={20} className={c.text} />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-gray-100">{sighting.person_name}</h2>
                <div className="text-[11px] font-mono text-gray-600 mt-0.5">
                  Person ID #{sighting.person_id} · {incident.title}
                </div>
              </div>
            </div>

            {/* Confidence */}
            <div className={`rounded-lg border ${c.border} ${c.bg} p-4 mb-4`}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-mono text-gray-500 tracking-widest">MATCH CONFIDENCE</span>
                <span className={`text-2xl font-mono font-bold ${c.text}`}>{pct}%</span>
              </div>
              <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                <motion.div
                  className={`h-full ${c.bar} rounded-full`}
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{ duration: 0.6, ease: 'easeOut' }}
                />
              </div>
              <div className="text-[10px] font-mono text-gray-600 mt-2">
                {pct >= 85 ? 'High confidence — strong identity signal' :
                 pct >= 65 ? 'Medium confidence — review recommended' :
                             'Low confidence — likely false positive'}
              </div>
            </div>

            {/* Detail grid */}
            <div className="grid grid-cols-2 gap-3 mb-4">
              {[
                {
                  Icon: Camera,
                  label: 'CAMERA / SOURCE',
                  value: sighting.source_name || `CAM-${sighting.camera_id}` || '—',
                  sub: sighting.camera_id ? `ID: ${sighting.camera_id}` : undefined,
                },
                {
                  Icon: MapPin,
                  label: 'LOCATION',
                  value: incident.last_seen_location || 'Unknown location',
                  sub: undefined,
                },
                {
                  Icon: Clock,
                  label: 'DETECTED AT',
                  value: ts?.time ?? '—',
                  sub: ts?.date,
                },
                {
                  Icon: Film,
                  label: 'FRAME INDEX',
                  value: `#${String(sighting.frame_index).padStart(5, '0')}`,
                  sub: undefined,
                },
                {
                  Icon: Crosshair,
                  label: 'ALERT ID',
                  value: `#${String(sighting.id).slice(-8).toUpperCase()}`,
                  sub: undefined,
                },
                {
                  Icon: ExternalLink,
                  label: 'CASE REF',
                  value: incident.ref,
                  sub: incident.title,
                },
              ].map(({ Icon, label, value, sub }) => (
                <div key={label} className="bg-gray-900/50 border border-gray-800 rounded-lg p-3">
                  <div className="flex items-center gap-1.5 text-[9px] font-mono text-gray-600 tracking-widest mb-1.5">
                    <Icon size={9} /> {label}
                  </div>
                  <div className="text-sm font-mono text-gray-200 truncate">{value}</div>
                  {sub && <div className="text-[10px] font-mono text-gray-600 truncate mt-0.5">{sub}</div>}
                </div>
              ))}
            </div>

            {/* Operator notes */}
            <div className="border border-gray-800 rounded-lg p-4">
              <div className="text-[10px] font-mono text-gray-600 tracking-widest mb-3 flex items-center gap-1.5">
                <Send size={9} /> OPERATOR NOTES
              </div>

              {notes.length > 0 && (
                <div className="space-y-2 mb-3">
                  {notes.map((n, i) => (
                    <div key={i} className="text-[11px] font-mono text-gray-400 bg-gray-900 rounded px-3 py-2 border border-gray-800">
                      <span className="text-gray-600 mr-2">{i + 1}.</span>{n}
                    </div>
                  ))}
                </div>
              )}

              {caseClosed ? (
                <div className="text-[10px] font-mono text-gray-600 py-2">
                  Case closed — notes locked
                </div>
              ) : (
                <div className="flex gap-2">
                  <input
                    value={note}
                    onChange={e => setNote(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && addNote()}
                    placeholder="Add forensic note…"
                    className="flex-1 bg-gray-900 border border-gray-700 focus:border-cyan-600/50 rounded px-3 py-2 text-xs text-gray-300 outline-none placeholder-gray-700"
                  />
                  <button
                    onClick={addNote}
                    disabled={!note.trim()}
                    className="px-3 py-2 bg-gray-800 border border-gray-700 text-gray-400 rounded text-xs hover:bg-gray-700 hover:text-gray-200 transition-colors disabled:opacity-30 disabled:pointer-events-none flex-shrink-0"
                  >
                    <Send size={12} />
                  </button>
                </div>
              )}
              <div className="text-[9px] font-mono text-gray-700 mt-1.5">
                Notes are appended with a UTC timestamp and saved to the case record.
              </div>
            </div>
          </div>
        </div>

        {/* ── Right: Detection history ────────────────────────────────────── */}
        <aside className="w-56 flex-shrink-0 border-l border-gray-800 overflow-y-auto p-4">
          <div className="text-[9px] font-mono text-gray-600 tracking-widest mb-3 flex items-center justify-between">
            <span>DETECTION HISTORY</span>
            <span className="text-gray-700">{personHistory.length}</span>
          </div>

          {/* Summary counts — sourced from GET /alerts/:id when available */}
          <div className="grid grid-cols-3 gap-1 mb-4">
            {[
              {
                label: 'TOTAL',
                value: alertData?.sighting_count ?? personHistory.length,
                cls: 'text-gray-400',
              },
              {
                label: 'VALID',
                value: alertData
                  ? alertData.sightings.filter(s => (s.confidence ?? 0) >= CONFIDENCE_FLOOR).length
                  : personHistory.filter(s => s.status === 'CONFIRMED').length,
                cls: 'text-emerald-400',
              },
              {
                label: 'LOW',
                value: alertData
                  ? alertData.sightings.filter(s => (s.confidence ?? 0) < CONFIDENCE_FLOOR).length
                  : personHistory.filter(s => s.status === 'REJECTED').length,
                cls: 'text-amber-400',
              },
            ].map(({ label, value, cls }) => (
              <div key={label} className="bg-gray-900 border border-gray-800 rounded p-2 text-center">
                <div className={`text-sm font-mono font-bold ${cls}`}>{value}</div>
                <div className="text-[8px] font-mono text-gray-700">{label}</div>
              </div>
            ))}
          </div>

          <div className="space-y-1">
            <AnimatePresence>
              {personHistory.map((s, i) => (
                <motion.div
                  key={s.id}
                  initial={{ opacity: 0, x: 12 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.03 }}
                >
                  <HistoryRow
                    s={s}
                    isActive={String(s.id) === String(sightingId)}
                    incidentId={incidentId!}
                  />
                </motion.div>
              ))}
            </AnimatePresence>

            {personHistory.length === 0 && (
              <div className="text-[10px] font-mono text-gray-700 text-center py-6">
                No detection history
              </div>
            )}
          </div>

          {/* Future video clip placeholder */}
          <div className="mt-5 border border-dashed border-gray-800 rounded-lg p-3 text-center">
            <Film size={16} className="text-gray-800 mx-auto mb-1" />
            <div className="text-[9px] font-mono text-gray-700">VIDEO CLIP</div>
            <div className="text-[9px] font-mono text-gray-800 mt-0.5">Trimmed footage coming</div>
          </div>
        </aside>
      </div>

      {/* Zoom modal */}
      <ImageZoomModal
        images={zoomImages ?? []}
        startIndex={zoomIdx}
        isOpen={!!zoomImages}
        onClose={() => setZoomImages(null)}
      />
    </div>
  )
}
