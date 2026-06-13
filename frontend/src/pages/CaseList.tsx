import { useMemo } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { AlertTriangle, ArrowRight, PlusCircle, Loader2 } from 'lucide-react'
import { useIncidents } from '../api/hooks'
import type { IncidentStatus } from '../types'

const statusCfg: Record<IncidentStatus, { label: string; color: string; border: string; dot: string; card: string }> = {
  OPEN:     { label: 'OPEN',     color: 'text-amber-400',   border: 'border-amber-500/30',   dot: 'bg-amber-400',              card: 'border-gray-800 hover:border-gray-700' },
  TRACKING: { label: 'TRACKING', color: 'text-cyan-400',    border: 'border-cyan-500/30',    dot: 'bg-cyan-400 animate-pulse-dot', card: 'border-gray-800 hover:border-cyan-700/40' },
  RESOLVED: { label: 'RESOLVED', color: 'text-emerald-400', border: 'border-emerald-500/30', dot: 'bg-emerald-400',            card: 'border-gray-800 hover:border-gray-700' },
  CLOSED:   { label: 'CLOSED',   color: 'text-gray-600',    border: 'border-gray-700',       dot: 'bg-gray-600',               card: 'border-gray-800 hover:border-gray-700' },
}

type FilterKey = 'all' | 'open' | 'tracking' | 'alerts' | 'resolved'

const FILTERS: { key: FilterKey; label: string }[] = [
  { key: 'all',      label: 'All' },
  { key: 'tracking', label: 'Tracking' },
  { key: 'open',     label: 'Open' },
  { key: 'alerts',   label: 'Alerts' },
  { key: 'resolved', label: 'Resolved' },
]

export default function CaseList() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const { data: incidents, loading, error } = useIncidents()

  const activeFilter = (searchParams.get('filter') as FilterKey | null) ?? 'all'

  const setFilter = (key: FilterKey) => {
    if (key === 'all') setSearchParams({})
    else setSearchParams({ filter: key })
  }

  const filtered = useMemo(() => {
    switch (activeFilter) {
      case 'tracking': return incidents.filter((i) => i.status === 'TRACKING')
      case 'open':     return incidents.filter((i) => i.status === 'OPEN')
      case 'alerts':   return incidents.filter((i) => i.alert_count > 0)
      case 'resolved': return incidents.filter((i) => i.status === 'RESOLVED' || i.status === 'CLOSED')
      default:         return incidents
    }
  }, [incidents, activeFilter])

  const alertCount  = incidents.filter((i) => i.alert_count > 0).length
  const activeCount = incidents.filter((i) => i.status !== 'CLOSED').length

  return (
    <div className="p-8 max-w-4xl">
      {/* header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">Cases</h1>
          <p className="text-xs font-mono text-gray-600 mt-1">
            {loading
              ? 'Loading…'
              : `${incidents.length} total · ${activeCount} active`}
          </p>
        </div>
        <button
          onClick={() => navigate('/cases/new')}
          className="flex items-center gap-2 px-4 py-2 bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 rounded-lg text-sm font-medium hover:bg-cyan-500/20 transition-colors"
        >
          <PlusCircle size={14} />
          New Case
        </button>
      </div>

      {/* filter bar */}
      <div className="flex items-center gap-1.5 mb-5 bg-gray-900 border border-gray-800 rounded-lg p-1">
        {FILTERS.map((f) => {
          const isActive = activeFilter === f.key
          const hasAlert = f.key === 'alerts' && alertCount > 0
          return (
            <button
              key={f.key}
              onClick={() => setFilter(f.key)}
              className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded text-xs font-mono transition-all duration-150 flex-1 justify-center ${
                isActive
                  ? 'bg-gray-800 text-gray-100 shadow-sm'
                  : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              {f.label}
              {hasAlert && (
                <span className={`w-1.5 h-1.5 rounded-full bg-amber-400 ${isActive ? '' : 'animate-pulse'}`} />
              )}
              {f.key !== 'all' && !hasAlert && (
                <span className="text-[9px] text-gray-700">
                  {f.key === 'tracking' ? incidents.filter((i) => i.status === 'TRACKING').length
                   : f.key === 'open'     ? incidents.filter((i) => i.status === 'OPEN').length
                   : f.key === 'alerts'   ? alertCount
                   : incidents.filter((i) => i.status === 'RESOLVED' || i.status === 'CLOSED').length}
                </span>
              )}
              {hasAlert && (
                <span className="text-[9px] text-amber-400">{alertCount}</span>
              )}
            </button>
          )
        })}
      </div>

      {/* loading — only on first fetch */}
      {loading && incidents.length === 0 && (
        <div className="flex items-center justify-center py-20 text-gray-600">
          <Loader2 size={18} className="animate-spin mr-3" />
          <span className="text-sm font-mono">Loading cases…</span>
        </div>
      )}

      {error && (
        <div className="border border-red-500/30 bg-red-500/8 rounded-lg p-4 text-sm text-red-400 font-mono">
          Failed to load: {error}
        </div>
      )}

      {!error && (
        <AnimatePresence mode="wait">
          <motion.div
            key={activeFilter}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="space-y-2.5"
          >
            {filtered.length === 0 && !loading && (
              <div className="text-center text-sm text-gray-600 py-14">
                {activeFilter === 'all'
                  ? <>No cases yet — <button className="text-cyan-400 hover:underline" onClick={() => navigate('/cases/new')}>create one</button></>
                  : `No ${activeFilter} cases`}
              </div>
            )}

            {filtered.map((inc, i) => {
              const cfg = statusCfg[inc.status]
              const hasAlerts = inc.alert_count > 0

              return (
                <motion.div
                  key={inc.id}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.04 }}
                >
                  <button
                    onClick={() => navigate(`/cases/${inc.id}`)}
                    className={`w-full text-left bg-gray-900 border rounded-lg p-5 transition-all duration-150 group ${
                      hasAlerts
                        ? 'border-amber-500/30 hover:border-amber-500/50'
                        : cfg.card
                    }`}
                  >
                    <div className="flex items-start gap-4">
                      {/* alert stripe */}
                      {hasAlerts && (
                        <div className="w-0.5 self-stretch bg-amber-500/60 rounded-full flex-shrink-0" />
                      )}

                      <div className="flex-1 min-w-0">
                        {/* badges row */}
                        <div className="flex items-center gap-2 mb-2 flex-wrap">
                          <span className="text-[10px] font-mono text-gray-600">{inc.ref || `INC-${String(inc.id).padStart(3, '0')}`}</span>
                          <div className={`inline-flex items-center gap-1.5 text-[10px] font-mono px-2 py-0.5 rounded border ${cfg.color} ${cfg.border}`}>
                            <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} />
                            {cfg.label}
                          </div>
                          {hasAlerts && (
                            <div className="inline-flex items-center gap-1 text-[10px] font-mono text-amber-400 bg-amber-500/12 border border-amber-500/30 px-2 py-0.5 rounded">
                              <AlertTriangle size={9} />
                              {inc.alert_count} alert{inc.alert_count !== 1 ? 's' : ''}
                            </div>
                          )}
                          {inc.person_count > 0 && (
                            <span className="text-[10px] font-mono text-gray-600">
                              {inc.person_count} person{inc.person_count !== 1 ? 's' : ''}
                            </span>
                          )}
                        </div>

                        {/* title */}
                        <h3 className="text-sm font-semibold text-gray-100 group-hover:text-white mb-1 truncate">
                          {inc.title}
                        </h3>

                        {/* description */}
                        {inc.description && (
                          <p className="text-xs text-gray-500 line-clamp-1 leading-relaxed mb-2">
                            {inc.description}
                          </p>
                        )}

                        {/* meta row */}
                        <div className="flex items-center gap-3 text-[10px] font-mono text-gray-700">
                          <span>Opened {new Date(inc.created_at).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })}</span>
                          {inc.updated_at !== inc.created_at && (
                            <>
                              <span>·</span>
                              <span>Updated {new Date(inc.updated_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                            </>
                          )}
                        </div>
                      </div>

                      <ArrowRight size={15} className="text-gray-700 group-hover:text-gray-400 transition-colors flex-shrink-0 mt-0.5" />
                    </div>
                  </button>
                </motion.div>
              )
            })}
          </motion.div>
        </AnimatePresence>
      )}
    </div>
  )
}
