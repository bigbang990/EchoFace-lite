import { useState, useEffect, useCallback, useRef } from 'react'
import {
  Plus, Trash2, ChevronDown, ChevronRight, X, Loader2,
} from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { createApiClient } from '../api/client'

// ── types ──────────────────────────────────────────────────────────────────────

interface Site {
  id: string
  name: string
  description?: string
}

interface Zone {
  id: string
  name: string
  site_id: string
  camera_count?: number
  online_count?: number
}

interface CameraRow {
  id: string
  label: string
  zone_name?: string
  source_type: string
  status: string
  trust_level: string
  direction?: string
}

interface HealthSummary {
  total: number
  online: number
  offline: number
  reconnecting: number
  unknown: number
}

// ── mock data ──────────────────────────────────────────────────────────────────

const MOCK_SITES: Site[] = [
  { id: '1', name: 'HQ Building', description: 'Headquarters' },
  { id: '2', name: 'Warehouse A', description: '' },
]

const MOCK_ZONES: Record<string, Zone[]> = {
  '1': [
    { id: '1', name: 'Main Entrance', site_id: '1', camera_count: 2, online_count: 2 },
    { id: '2', name: 'Lobby', site_id: '1', camera_count: 1, online_count: 1 },
  ],
  '2': [
    { id: '3', name: 'Loading Bay', site_id: '2', camera_count: 1, online_count: 0 },
  ],
}

const MOCK_CAMERAS: CameraRow[] = [
  { id: '1', label: 'CAM-001', zone_name: 'Main Entrance', source_type: 'rtsp', status: 'online',  trust_level: 'high',   direction: 'North' },
  { id: '2', label: 'CAM-002', zone_name: 'Lobby',         source_type: 'rtsp', status: 'online',  trust_level: 'medium', direction: 'South' },
  { id: '3', label: 'CAM-003', zone_name: 'Loading Bay',   source_type: 'rtsp', status: 'offline', trust_level: 'low',    direction: 'East'  },
]

const MOCK_HEALTH: HealthSummary = { total: 3, online: 2, offline: 1, reconnecting: 0, unknown: 0 }

// ── StatusDot ─────────────────────────────────────────────────────────────────

function StatusDot({ status }: { status: string }) {
  const map: Record<string, { label: string; cls: string }> = {
    online:       { label: '● Online',       cls: 'text-emerald-400' },
    offline:      { label: '◌ Offline',      cls: 'text-red-400'     },
    reconnecting: { label: '↺ Reconnecting', cls: 'text-amber-400'   },
  }
  const cfg = map[status] ?? { label: '? Unknown', cls: 'text-gray-500' }
  return <span className={`text-[10px] font-mono ${cfg.cls}`}>{cfg.label}</span>
}

// ── RegisterCameraModal ───────────────────────────────────────────────────────

function RegisterCameraModal({
  sites,
  backendUrl,
  onClose,
  onSuccess,
}: {
  sites: Site[]
  backendUrl: string
  onClose: () => void
  onSuccess: () => void
}) {
  const [name, setName] = useState('')
  const [sourceType, setSourceType] = useState('rtsp')
  const [streamUrl, setStreamUrl] = useState('')
  const [siteId, setSiteId] = useState('')
  const [zoneId, setZoneId] = useState('')
  const [zones, setZones] = useState<Zone[]>([])
  const [direction, setDirection] = useState('North')
  const [trustLevel, setTrustLevel] = useState('medium')
  const [overlapGroup, setOverlapGroup] = useState('')
  const [retentionDays, setRetentionDays] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const client = createApiClient(backendUrl)

  useEffect(() => {
    if (!siteId) { setZones([]); setZoneId(''); return }
    client
      .get<Array<Record<string, unknown>>>(`/sites/${siteId}/zones`)
      .then((raw) =>
        setZones(raw.map((z) => ({ id: String(z.id), name: String(z.name ?? ''), site_id: siteId })))
      )
      .catch(() => {})
  }, [siteId, backendUrl])

  const register = async () => {
    if (!name.trim() || !sourceType) { setError('Name and source type are required'); return }
    setSaving(true); setError(null)
    try {
      await client.post('/cameras', {
        label: name.trim(),
        source_type: sourceType,
        stream_url: streamUrl || undefined,
        zone_id: zoneId ? Number(zoneId) : undefined,
        direction,
        trust_level: trustLevel,
        overlap_group: overlapGroup || undefined,
        retention_days: retentionDays ? Number(retentionDays) : undefined,
      })
      onSuccess()
      onClose()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-gray-900 border border-gray-700 rounded-xl w-[460px] max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-800">
          <div className="text-sm font-semibold text-gray-200">Register Camera</div>
          <button onClick={onClose} className="text-gray-600 hover:text-gray-300 transition-colors">
            <X size={14} />
          </button>
        </div>

        <div className="p-5 space-y-4">
          <Field label="NAME *">
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Main Entrance"
              className="field-input"
            />
          </Field>

          <Field label="SOURCE TYPE *">
            <select value={sourceType} onChange={(e) => setSourceType(e.target.value)} className="field-input">
              <option value="file">File</option>
              <option value="rtsp">RTSP</option>
              <option value="android">Android</option>
              <option value="nvr">NVR</option>
              <option value="dvr">DVR</option>
            </select>
          </Field>

          {(sourceType === 'rtsp' || sourceType === 'android' || sourceType === 'nvr') && (
            <Field label="STREAM URL">
              <input
                value={streamUrl}
                onChange={(e) => setStreamUrl(e.target.value)}
                placeholder="rtsp://192.168.1.100:554/stream"
                className="field-input font-mono text-xs"
              />
            </Field>
          )}

          <Field label="SITE">
            <select
              value={siteId}
              onChange={(e) => { setSiteId(e.target.value); setZoneId('') }}
              className="field-input"
            >
              <option value="">Select site…</option>
              {sites.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
            </select>
          </Field>

          <Field label="ZONE">
            <select
              value={zoneId}
              onChange={(e) => setZoneId(e.target.value)}
              disabled={!siteId || zones.length === 0}
              className="field-input disabled:opacity-40"
            >
              <option value="">Select zone…</option>
              {zones.map((z) => <option key={z.id} value={z.id}>{z.name}</option>)}
            </select>
          </Field>

          <div className="grid grid-cols-2 gap-4">
            <Field label="DIRECTION">
              <select value={direction} onChange={(e) => setDirection(e.target.value)} className="field-input">
                {['North','South','East','West','North-East','North-West','South-East','South-West'].map((d) => (
                  <option key={d} value={d}>{d}</option>
                ))}
              </select>
            </Field>

            <Field label="TRUST LEVEL">
              <select value={trustLevel} onChange={(e) => setTrustLevel(e.target.value)} className="field-input">
                <option value="high">High</option>
                <option value="medium">Medium (default)</option>
                <option value="low">Low</option>
              </select>
            </Field>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <Field label="OVERLAP GROUP">
              <input
                value={overlapGroup}
                onChange={(e) => setOverlapGroup(e.target.value)}
                placeholder="entrance-a (optional)"
                className="field-input"
              />
            </Field>

            <Field label="RETENTION DAYS">
              <input
                type="number"
                value={retentionDays}
                onChange={(e) => setRetentionDays(e.target.value)}
                placeholder="30 (optional)"
                className="field-input"
              />
            </Field>
          </div>

          {error && (
            <div className="border border-red-500/30 bg-red-500/8 rounded px-3 py-2 text-xs font-mono text-red-400">
              {error}
            </div>
          )}
        </div>

        <div className="px-5 py-4 border-t border-gray-800 flex justify-end gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 text-xs font-mono text-gray-500 hover:text-gray-300 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={register}
            disabled={saving}
            className="flex items-center gap-1.5 px-4 py-2 text-xs font-mono bg-cyan-500/15 border border-cyan-500/40 text-cyan-400 rounded hover:bg-cyan-500/25 transition-colors disabled:opacity-40"
          >
            {saving ? <><Loader2 size={11} className="animate-spin" /> Registering…</> : 'Register'}
          </button>
        </div>
      </div>
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-[9px] font-mono text-gray-600 tracking-widest mb-1.5">{label}</label>
      {children}
    </div>
  )
}

// ── Administration ─────────────────────────────────────────────────────────────

export default function Administration() {
  const { backendUrl, accessMode } = useAppStore()
  const isMock = accessMode === 'MOCK'

  const [sites, setSites] = useState<Site[]>([])
  const [zonesBySite, setZonesBySite] = useState<Record<string, Zone[]>>({})
  const [cameras, setCameras] = useState<CameraRow[]>([])
  const [health, setHealth] = useState<HealthSummary>({ total: 0, online: 0, offline: 0, reconnecting: 0, unknown: 0 })
  const [expandedSites, setExpandedSites] = useState<Set<string>>(new Set(['1']))
  const [showRegisterModal, setShowRegisterModal] = useState(false)
  const loadCancelledRef = useRef(false)

  const [addingSite, setAddingSite] = useState(false)
  const [newSiteName, setNewSiteName] = useState('')
  const [addingZoneForSite, setAddingZoneForSite] = useState<string | null>(null)
  const [newZoneName, setNewZoneName] = useState('')

  const loadSites = useCallback(async () => {
    if (isMock) { setSites(MOCK_SITES); setZonesBySite(MOCK_ZONES); return }
    const client = createApiClient(backendUrl)
    try {
      const raw = await client.get<Array<Record<string, unknown>>>('/sites')
      if (loadCancelledRef.current) return
      const list: Site[] = raw.map((s) => ({
        id: String(s.id),
        name: String(s.name ?? ''),
        description: s.description ? String(s.description) : undefined,
      }))
      setSites(list)
      const zonesMap: Record<string, Zone[]> = {}
      await Promise.all(
        list.map(async (site) => {
          try {
            const rawZ = await client.get<Array<Record<string, unknown>>>(`/sites/${site.id}/zones`)
            zonesMap[site.id] = rawZ.map((z) => ({
              id: String(z.id),
              name: String(z.name ?? ''),
              site_id: site.id,
              camera_count: z.camera_count != null ? Number(z.camera_count) : undefined,
              online_count:  z.online_count  != null ? Number(z.online_count)  : undefined,
            }))
          } catch { zonesMap[site.id] = [] }
        })
      )
      if (!loadCancelledRef.current) setZonesBySite(zonesMap)
    } catch { /* ignore */ }
  }, [backendUrl, isMock])

  const loadCameras = useCallback(async () => {
    if (isMock) { setCameras(MOCK_CAMERAS); setHealth(MOCK_HEALTH); return }
    const client = createApiClient(backendUrl)
    const [camsResult, healthResult] = await Promise.allSettled([
      client.get<Array<Record<string, unknown>>>('/cameras'),
      client.get<Record<string, unknown>>('/cameras/health-summary'),
    ])
    if (loadCancelledRef.current) return

    if (camsResult.status === 'fulfilled') {
      const rawCams = camsResult.value
      console.log('[Administration] cameras loaded:', rawCams.length, rawCams)
      setCameras(rawCams.map((c) => ({
        id: String(c.id),
        label: String(c.label ?? c.name ?? ''),
        zone_name: c.zone_name ? String(c.zone_name) : undefined,
        source_type: String(c.source_type ?? 'unknown'),
        status: String(c.status ?? 'unknown'),
        trust_level: String(c.trust_level ?? 'medium'),
        direction: c.direction ? String(c.direction) : undefined,
      })))
    } else {
      console.warn('[Administration] cameras fetch failed:', camsResult.reason)
    }

    if (healthResult.status === 'fulfilled') {
      const rawHealth = healthResult.value
      setHealth({
        total:        Number(rawHealth.total        ?? 0),
        online:       Number(rawHealth.online        ?? 0),
        offline:      Number(rawHealth.offline       ?? 0),
        reconnecting: Number(rawHealth.reconnecting  ?? 0),
        unknown:      Number(rawHealth.unknown       ?? 0),
      })
    }
  }, [backendUrl, isMock])

  useEffect(() => {
    loadCancelledRef.current = false
    loadSites()
    loadCameras()
    return () => { loadCancelledRef.current = true }
  }, [loadSites, loadCameras])

  // ── Site actions ────────────────────────────────────────────────────────────

  const addSite = async () => {
    if (!newSiteName.trim()) return
    if (isMock) {
      setSites((prev) => [...prev, { id: String(Date.now()), name: newSiteName.trim() }])
      setNewSiteName(''); setAddingSite(false); return
    }
    try {
      const raw = await createApiClient(backendUrl).post<Record<string, unknown>>('/sites', {
        name: newSiteName.trim(), description: '',
      })
      setSites((prev) => [...prev, { id: String(raw.id), name: String(raw.name ?? newSiteName.trim()) }])
      setNewSiteName(''); setAddingSite(false)
    } catch { /* ignore */ }
  }

  const deleteSite = async (id: string) => {
    if (isMock) { setSites((prev) => prev.filter((s) => s.id !== id)); return }
    try {
      await createApiClient(backendUrl).del(`/sites/${id}`)
      setSites((prev) => prev.filter((s) => s.id !== id))
      setZonesBySite((prev) => { const n = { ...prev }; delete n[id]; return n })
    } catch { /* ignore */ }
  }

  // ── Zone actions ────────────────────────────────────────────────────────────

  const addZone = async (siteId: string) => {
    if (!newZoneName.trim()) return
    if (isMock) {
      setZonesBySite((prev) => ({
        ...prev,
        [siteId]: [...(prev[siteId] ?? []), { id: String(Date.now()), name: newZoneName.trim(), site_id: siteId }],
      }))
      setNewZoneName(''); setAddingZoneForSite(null); return
    }
    try {
      const raw = await createApiClient(backendUrl).post<Record<string, unknown>>('/zones', {
        name: newZoneName.trim(), site_id: Number(siteId),
      })
      setZonesBySite((prev) => ({
        ...prev,
        [siteId]: [...(prev[siteId] ?? []), { id: String(raw.id), name: String(raw.name ?? newZoneName.trim()), site_id: siteId }],
      }))
      setNewZoneName(''); setAddingZoneForSite(null)
    } catch { /* ignore */ }
  }

  const deleteZone = async (siteId: string, zoneId: string) => {
    if (isMock) {
      setZonesBySite((prev) => ({ ...prev, [siteId]: (prev[siteId] ?? []).filter((z) => z.id !== zoneId) }))
      return
    }
    try {
      await createApiClient(backendUrl).del(`/zones/${zoneId}`)
      setZonesBySite((prev) => ({ ...prev, [siteId]: (prev[siteId] ?? []).filter((z) => z.id !== zoneId) }))
    } catch { /* ignore */ }
  }

  // ── Camera actions ──────────────────────────────────────────────────────────

  const deleteCamera = async (id: string) => {
    if (isMock) { setCameras((prev) => prev.filter((c) => c.id !== id)); return }
    try {
      await createApiClient(backendUrl).del(`/cameras/${id}`)
      await loadCameras()
    } catch { /* ignore */ }
  }

  const toggleSite = (id: string) =>
    setExpandedSites((prev) => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="p-8 max-w-6xl">
      <div className="mb-7">
        <h1 className="text-xl font-semibold text-gray-100">Administration</h1>
        <p className="text-xs font-mono text-gray-600 mt-1">Manage sites, zones, and camera sources</p>
      </div>

      <div className="grid grid-cols-2 gap-5">

        {/* ── Panel 1: Sites & Zones ──────────────────────────────────────── */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-[10px] font-mono text-gray-600 tracking-widest">SITES & ZONES</h2>
            <button
              onClick={() => setAddingSite(true)}
              className="flex items-center gap-1.5 px-2.5 py-1.5 text-[10px] font-mono bg-gray-800 border border-gray-700 text-gray-400 rounded hover:text-gray-200 hover:border-gray-600 transition-colors"
            >
              <Plus size={10} /> Add Site
            </button>
          </div>

          {addingSite && (
            <div className="mb-4 flex gap-2">
              <input
                autoFocus
                value={newSiteName}
                onChange={(e) => setNewSiteName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') addSite()
                  if (e.key === 'Escape') { setAddingSite(false); setNewSiteName('') }
                }}
                placeholder="Site name…"
                className="flex-1 bg-gray-950 border border-gray-700 focus:border-cyan-600/50 rounded px-3 py-1.5 text-xs text-gray-200 outline-none"
              />
              <button onClick={addSite} className="px-3 py-1.5 text-[10px] font-mono bg-cyan-500/15 border border-cyan-500/40 text-cyan-400 rounded hover:bg-cyan-500/25">Save</button>
              <button onClick={() => { setAddingSite(false); setNewSiteName('') }} className="px-2 py-1.5 text-[10px] text-gray-600 hover:text-gray-300">✕</button>
            </div>
          )}

          {sites.length === 0 ? (
            <div className="text-center py-10 text-[11px] font-mono text-gray-600">
              No sites registered. Add a site to begin.
            </div>
          ) : (
            <div className="space-y-2">
              {sites.map((site) => (
                <div key={site.id} className="border border-gray-800 rounded-lg overflow-hidden">
                  {/* Site row */}
                  <div className="flex items-center gap-2 px-3 py-2.5 bg-gray-800/30">
                    <button onClick={() => toggleSite(site.id)} className="text-gray-500 hover:text-gray-300 flex-shrink-0">
                      {expandedSites.has(site.id) ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                    </button>
                    <span className="flex-1 text-[13px] font-medium text-gray-200">{site.name}</span>
                    <button
                      onClick={() => {
                        setAddingZoneForSite(site.id)
                        setExpandedSites((prev) => new Set([...prev, site.id]))
                      }}
                      className="flex items-center gap-1 text-[10px] font-mono text-gray-600 hover:text-cyan-400 transition-colors"
                    >
                      <Plus size={9} /> Zone
                    </button>
                    <button
                      onClick={() => deleteSite(site.id)}
                      className="text-gray-700 hover:text-red-400 transition-colors ml-1"
                    >
                      <Trash2 size={11} />
                    </button>
                  </div>

                  {/* Zones */}
                  {expandedSites.has(site.id) && (
                    <div className="px-3 pb-2 pt-1 space-y-0.5">
                      {addingZoneForSite === site.id && (
                        <div className="flex gap-2 mt-1 mb-2">
                          <input
                            autoFocus
                            value={newZoneName}
                            onChange={(e) => setNewZoneName(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') addZone(site.id)
                              if (e.key === 'Escape') { setAddingZoneForSite(null); setNewZoneName('') }
                            }}
                            placeholder="Zone name…"
                            className="flex-1 bg-gray-950 border border-gray-700 focus:border-cyan-600/50 rounded px-2 py-1 text-xs text-gray-200 outline-none"
                          />
                          <button onClick={() => addZone(site.id)} className="px-2 py-1 text-[10px] font-mono bg-cyan-500/15 border border-cyan-500/40 text-cyan-400 rounded">Save</button>
                          <button onClick={() => { setAddingZoneForSite(null); setNewZoneName('') }} className="px-1.5 py-1 text-[10px] text-gray-600 hover:text-gray-300">✕</button>
                        </div>
                      )}

                      {(zonesBySite[site.id] ?? []).length === 0 && addingZoneForSite !== site.id && (
                        <div className="text-[10px] font-mono text-gray-700 py-2 pl-2">No zones — add one above</div>
                      )}

                      {(zonesBySite[site.id] ?? []).map((zone) => {
                        const camCount   = zone.camera_count  ?? cameras.filter((c) => c.zone_name === zone.name).length
                        const onlineCount = zone.online_count ?? cameras.filter((c) => c.zone_name === zone.name && c.status === 'online').length
                        return (
                          <div
                            key={zone.id}
                            className="flex items-center gap-2 py-2 pl-5 text-xs text-gray-400 border-b border-gray-800/40 last:border-0"
                          >
                            <span className="flex-1">{zone.name}</span>
                            <span className="text-[10px] font-mono text-gray-600">{camCount} cameras</span>
                            {onlineCount > 0 && (
                              <span className="text-[10px] font-mono text-emerald-500">● {onlineCount}</span>
                            )}
                            <button
                              onClick={() => deleteZone(site.id, zone.id)}
                              className="text-gray-800 hover:text-red-400 transition-colors"
                            >
                              <Trash2 size={10} />
                            </button>
                          </div>
                        )
                      })}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ── Panel 2: Camera Sources ─────────────────────────────────────── */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-[10px] font-mono text-gray-600 tracking-widest">CAMERA SOURCES</h2>
            <button
              onClick={() => setShowRegisterModal(true)}
              className="flex items-center gap-1.5 px-2.5 py-1.5 text-[10px] font-mono bg-gray-800 border border-gray-700 text-gray-400 rounded hover:text-gray-200 hover:border-gray-600 transition-colors"
            >
              <Plus size={10} /> Register Camera
            </button>
          </div>

          {health.total > 0 && (
            <div className="flex items-center gap-4 mb-4 text-[10px] font-mono pb-3 border-b border-gray-800">
              <span className="text-emerald-400">● {health.online} Online</span>
              <span className="text-red-400">◌ {health.offline} Offline</span>
              {health.reconnecting > 0 && (
                <span className="text-amber-400">↺ {health.reconnecting} Reconnecting</span>
              )}
              <span className="text-gray-700 ml-auto">{health.total} registered</span>
            </div>
          )}

          {cameras.length === 0 ? (
            <div className="text-center py-10 text-[11px] font-mono text-gray-600">
              No cameras registered.
            </div>
          ) : (
            <div className="space-y-0">
              <div className="grid grid-cols-[1fr_60px_100px_32px] gap-3 px-2 py-1.5 text-[9px] font-mono text-gray-700 tracking-widest border-b border-gray-800">
                <span>NAME / ZONE</span>
                <span>TYPE</span>
                <span>STATUS</span>
                <span />
              </div>
              {cameras.map((cam) => (
                <div
                  key={cam.id}
                  className="grid grid-cols-[1fr_60px_100px_32px] gap-3 items-center px-2 py-2.5 border-b border-gray-800/40 last:border-0"
                >
                  <div className="min-w-0">
                    <div className="text-[12px] font-medium text-gray-200 truncate">{cam.label}</div>
                    {cam.zone_name && (
                      <div className="text-[10px] font-mono text-gray-600 truncate">{cam.zone_name}</div>
                    )}
                  </div>
                  <span className="text-[10px] font-mono text-gray-500 uppercase">{cam.source_type}</span>
                  <StatusDot status={cam.status} />
                  <button
                    onClick={() => deleteCamera(cam.id)}
                    className="text-gray-700 hover:text-red-400 transition-colors"
                  >
                    <Trash2 size={11} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {showRegisterModal && (
        <RegisterCameraModal
          sites={sites}
          backendUrl={backendUrl}
          onClose={() => setShowRegisterModal(false)}
          onSuccess={() => { void loadCameras(); void loadSites() }}
        />
      )}
    </div>
  )
}
