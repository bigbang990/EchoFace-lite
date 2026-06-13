import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { AlertTriangle, ArrowRight, PlusCircle, Loader2, RefreshCw } from 'lucide-react'
import { useIncidents } from '../api/hooks'
import type { IncidentStatus } from '../types'

const statusCfg: Record<IncidentStatus, { label: string; color: string; border: string; dot: string }> = {
  OPEN:     { label: 'OPEN',     color: 'text-amber-400',   border: 'border-amber-500/30',   dot: 'bg-amber-400' },
  TRACKING: { label: 'TRACKING', color: 'text-cyan-400',    border: 'border-cyan-500/30',    dot: 'bg-cyan-400 animate-pulse-dot' },
  RESOLVED: { label: 'RESOLVED', color: 'text-emerald-400', border: 'border-emerald-500/30', dot: 'bg-emerald-400' },
  CLOSED:   { label: 'CLOSED',   color: 'text-gray-600',    border: 'border-gray-700',       dot: 'bg-gray-600' },
}

export default function CaseList() {
  const navigate = useNavigate()
  const { data: incidents, loading, error, refetch } = useIncidents()

  return (
    <div className="p-8 max-w-4xl">
      <div className="flex items-center justify-between mb-7">
        <div>
          <h1 className="text-xl font-semibold text-gray-100">Cases</h1>
          <p className="text-xs font-mono text-gray-600 mt-1">
            {loading ? 'Loading…' : `${incidents.length} total · ${incidents.filter((i) => i.status !== 'CLOSED').length} active`}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={refetch}
            className="p-2 text-gray-600 hover:text-gray-300 transition-colors"
            title="Refresh"
          >
            <RefreshCw size={14} />
          </button>
          <button
            onClick={() => navigate('/cases/new')}
            className="flex items-center gap-2 px-4 py-2 bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 rounded text-sm font-medium hover:bg-cyan-500/20 transition-colors"
          >
            <PlusCircle size={14} />
            New Case
          </button>
        </div>
      </div>

      {loading && (
        <div className="flex items-center justify-center py-20 text-gray-600">
          <Loader2 size={20} className="animate-spin mr-3" />
          <span className="text-sm font-mono">Loading cases…</span>
        </div>
      )}

      {error && (
        <div className="border border-red-500/30 bg-red-500/8 rounded-lg p-5 text-sm text-red-400 font-mono">
          Failed to load cases: {error}
        </div>
      )}

      {!loading && !error && (
        <div className="space-y-3">
          {incidents.length === 0 && (
            <div className="text-center text-sm text-gray-600 py-16">
              No cases yet —{' '}
              <button
                className="text-cyan-400 hover:underline"
                onClick={() => navigate('/cases/new')}
              >
                create one
              </button>
            </div>
          )}
          {incidents.map((inc, i) => {
            const cfg = statusCfg[inc.status]
            return (
              <motion.div
                key={inc.id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
              >
                <button
                  onClick={() => navigate(`/cases/${inc.id}`)}
                  className="w-full text-left bg-gray-900 border border-gray-800 hover:border-gray-700 rounded-lg p-5 transition-all duration-150 group"
                >
                  <div className="flex items-start gap-4">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-3 mb-2 flex-wrap">
                        <span className="text-xs font-mono text-gray-600">{inc.ref}</span>
                        <div className={`inline-flex items-center gap-1.5 text-[10px] font-mono px-2 py-0.5 rounded border ${cfg.color} ${cfg.border}`}>
                          <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} />
                          {cfg.label}
                        </div>
                        {inc.alert_count > 0 && (
                          <div className="inline-flex items-center gap-1 text-[10px] font-mono text-amber-400 bg-amber-500/10 border border-amber-500/25 px-2 py-0.5 rounded">
                            <AlertTriangle size={9} />
                            {inc.alert_count} alert{inc.alert_count !== 1 ? 's' : ''}
                          </div>
                        )}
                      </div>
                      <h3 className="text-base font-semibold text-gray-100 group-hover:text-white mb-1">
                        {inc.title}
                      </h3>
                      <p className="text-sm text-gray-500 line-clamp-2 leading-relaxed">
                        {inc.description}
                      </p>
                      <div className="flex items-center gap-4 mt-3 text-[11px] font-mono text-gray-600 flex-wrap">
                        <span>{inc.last_seen_location}</span>
                        <span>·</span>
                        <span>
                          {new Date(inc.last_seen_at).toLocaleDateString('en-GB', {
                            day: 'numeric', month: 'short', year: 'numeric',
                          })}
                        </span>
                        <span>·</span>
                        <span>{inc.person_count} person{inc.person_count !== 1 ? 's' : ''} enrolled</span>
                      </div>
                    </div>
                    <div className="flex-shrink-0 mt-1 text-gray-700 group-hover:text-gray-400 transition-colors">
                      <ArrowRight size={16} />
                    </div>
                  </div>
                </button>
              </motion.div>
            )
          })}
        </div>
      )}
    </div>
  )
}
