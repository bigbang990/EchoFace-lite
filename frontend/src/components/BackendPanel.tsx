import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  X, Loader2, Cpu, Globe, Server, RefreshCw,
  CheckCircle2, XCircle, ChevronRight, Link2,
} from 'lucide-react'
import { useAppStore, BACKENDS, INC_DEFAULT_URL, type BackendEntry } from '../store/appStore'
import { useHealthCheck } from '../api/hooks'

interface Props {
  open: boolean
  onClose: () => void
}

// ── single backend card ────────────────────────────────────────────────────────

function BackendCard({
  entry,
  isActive,
  onSelect,
}: {
  entry: BackendEntry
  isActive: boolean
  onSelect: () => void
}) {
  const { status, recheck } = useHealthCheck(entry.url)

  const statusDot =
    status === 'online'   ? 'bg-emerald-400' :
    status === 'offline'  ? 'bg-red-400' :
    'bg-gray-600 animate-pulse'

  const statusLabel =
    status === 'online'  ? 'ONLINE' :
    status === 'offline' ? 'OFFLINE' :
    'CHECKING'

  const statusColor =
    status === 'online'  ? 'text-emerald-400' :
    status === 'offline' ? 'text-red-400' :
    'text-gray-600'

  return (
    <button
      onClick={onSelect}
      className={`w-full text-left rounded-xl p-4 border transition-all duration-150 ${
        isActive
          ? 'border-cyan-500/50 bg-cyan-500/8 shadow-[0_0_0_1px_rgba(6,182,212,0.15)]'
          : 'border-gray-800 bg-gray-900/60 hover:border-gray-700 hover:bg-gray-800/50'
      }`}
    >
      <div className="flex items-center gap-3">
        {/* icon */}
        <div className={`w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 ${
          entry.type === 'local' ? 'bg-cyan-500/12 border border-cyan-500/20' : 'bg-amber-500/12 border border-amber-500/20'
        }`}>
          {entry.type === 'local'
            ? <Cpu size={15} className="text-cyan-400" />
            : <Globe size={15} className="text-amber-400" />}
        </div>

        {/* name + url */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-gray-100">{entry.name}</span>
            {isActive && (
              <span className="text-[9px] font-mono text-cyan-400 bg-cyan-500/10 border border-cyan-500/25 px-1.5 py-0.5 rounded">
                ACTIVE
              </span>
            )}
          </div>
          <div className="text-[10px] font-mono text-gray-600 mt-0.5 truncate">{entry.url}</div>
        </div>

        {/* status */}
        <div className="flex items-center gap-1.5 flex-shrink-0">
          <span className={`w-1.5 h-1.5 rounded-full ${statusDot}`} />
          <span className={`text-[10px] font-mono ${statusColor}`}>{statusLabel}</span>
          <button
            onClick={(e) => { e.stopPropagation(); recheck() }}
            className="ml-0.5 p-0.5 text-gray-700 hover:text-gray-400 transition-colors rounded"
            title="Recheck"
          >
            <RefreshCw size={10} />
          </button>
        </div>
      </div>

      {/* right arrow hint on hover */}
      {!isActive && (
        <div className="mt-2 flex items-center gap-1 text-[9px] font-mono text-gray-700">
          <ChevronRight size={10} /> click to connect
        </div>
      )}
    </button>
  )
}

// ── custom URL section ─────────────────────────────────────────────────────────

function CustomUrlForm({ onConnect }: { onConnect: (url: string) => void }) {
  const [open, setOpen] = useState(false)
  const [url, setUrl] = useState('')

  const connect = () => {
    const trimmed = url.trim()
    if (trimmed) { onConnect(trimmed); setUrl(''); setOpen(false) }
  }

  return (
    <div>
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 text-[11px] font-mono text-gray-600 hover:text-cyan-400 transition-colors"
      >
        <Server size={11} />
        {open ? '✕ cancel' : '+ custom URL'}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="mt-3 space-y-2">
              <input
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://xyz.ngrok-free.app/api/v1"
                className="w-full bg-gray-950 border border-gray-700 focus:border-cyan-600/50 rounded-lg px-3 py-2 text-xs font-mono text-gray-200 outline-none placeholder-gray-700 transition-colors"
                onKeyDown={(e) => { if (e.key === 'Enter') connect(); if (e.key === 'Escape') setOpen(false) }}
                autoFocus
              />
              <button
                onClick={connect}
                disabled={!url.trim()}
                className="w-full py-2 bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 rounded-lg text-[11px] font-mono tracking-wider hover:bg-cyan-500/20 transition-colors disabled:opacity-30 disabled:pointer-events-none"
              >
                CONNECT
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ── INC URL override (advanced / hidden by default) ───────────────────────────

function IncUrlOverride() {
  const { incUrl, backendUrl, setIncUrl } = useAppStore()
  const [open, setOpen] = useState(false)
  const [draft, setDraft] = useState(incUrl)
  const { status } = useHealthCheck(incUrl)

  const isSynced = incUrl === backendUrl
  const statusColor = status === 'online' ? 'text-emerald-400' : status === 'offline' ? 'text-red-400' : 'text-gray-600'
  const statusIcon  = status === 'online' ? <CheckCircle2 size={10} className="text-emerald-400" /> : status === 'offline' ? <XCircle size={10} className="text-red-400" /> : <Loader2 size={10} className="text-gray-500 animate-spin" />

  const save = () => {
    const t = draft.trim()
    if (t) { setIncUrl(t); setOpen(false) }
  }

  return (
    <div className="border-t border-gray-800/70 pt-4">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 text-[10px] font-mono text-gray-700 hover:text-gray-500 transition-colors"
      >
        <Link2 size={10} />
        INC API endpoint
        {!isSynced && <span className="text-amber-400 ml-1">· overridden</span>}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="mt-3 space-y-2">
              <div className="flex items-center gap-2 bg-gray-950 border border-gray-800 rounded-lg px-3 py-2">
                {statusIcon}
                <span className={`text-[10px] font-mono flex-1 truncate ${statusColor}`}>{incUrl}</span>
              </div>
              <input
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                placeholder={backendUrl}
                className="w-full bg-gray-950 border border-gray-700 focus:border-cyan-600/50 rounded-lg px-3 py-2 text-[11px] font-mono text-gray-200 outline-none placeholder-gray-700 transition-colors"
                onKeyDown={(e) => { if (e.key === 'Enter') save(); if (e.key === 'Escape') setOpen(false) }}
              />
              <div className="flex gap-2">
                <button onClick={save} disabled={!draft.trim()} className="flex-1 py-1.5 bg-gray-800 border border-gray-700 text-gray-300 rounded text-[10px] font-mono hover:bg-gray-700 transition-colors disabled:opacity-30">
                  Save
                </button>
                {!isSynced && (
                  <button onClick={() => { setIncUrl(backendUrl); setDraft(backendUrl); setOpen(false) }} className="px-3 py-1.5 border border-gray-700 text-gray-500 rounded text-[10px] hover:bg-gray-800 hover:text-gray-300 transition-colors">
                    Reset
                  </button>
                )}
              </div>
              <p className="text-[9px] font-mono text-gray-700 leading-relaxed">
                Single-server: leave blank to use the same URL as the engine backend.
                Override only needed for Phase B split-server setup.
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ── main panel ─────────────────────────────────────────────────────────────────

export default function BackendPanel({ open, onClose }: Props) {
  const { backendName, backendUrl, incUrl, setBackend } = useAppStore()

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.6 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black z-40"
            onClick={onClose}
          />

          <motion.div
            initial={{ x: -320, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -320, opacity: 0 }}
            transition={{ type: 'spring', damping: 30, stiffness: 240 }}
            className="fixed left-56 top-0 bottom-0 w-80 bg-gray-950 border-r border-gray-800 z-50 flex flex-col"
          >
            {/* header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-gray-800">
              <div>
                <h2 className="text-sm font-semibold text-gray-100">API Connection</h2>
                <p className="text-[10px] font-mono text-gray-600 mt-0.5">
                  Connect to the EchoFace backend
                </p>
              </div>
              <button
                onClick={onClose}
                className="w-7 h-7 flex items-center justify-center rounded-lg text-gray-600 hover:text-gray-300 hover:bg-gray-800 transition-colors"
              >
                <X size={14} />
              </button>
            </div>

            {/* body */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {/* preset backends */}
              <div>
                <div className="text-[9px] font-mono text-gray-600 tracking-widest mb-2.5 px-0.5">PRESETS</div>
                <div className="space-y-2">
                  {BACKENDS.map((entry) => (
                    <BackendCard
                      key={entry.name}
                      entry={entry}
                      isActive={backendUrl === entry.url}
                      onSelect={() => { setBackend(entry.name, entry.url); onClose() }}
                    />
                  ))}
                </div>
              </div>

              {/* custom URL */}
              <div className="border-t border-gray-800/70 pt-3">
                <CustomUrlForm
                  onConnect={(url) => { setBackend('Custom', url); onClose() }}
                />
              </div>

              {/* INC override (advanced) */}
              <IncUrlOverride />
            </div>

            {/* footer — active connection */}
            <div className="px-5 py-4 border-t border-gray-800 bg-gray-900/50">
              <div className="text-[9px] font-mono text-gray-600 tracking-widest mb-2">ACTIVE CONNECTION</div>
              <div className="flex items-center gap-2 mb-1">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                <span className="text-[11px] font-semibold text-gray-300">{backendName}</span>
              </div>
              <div className="text-[10px] font-mono text-gray-600 truncate">{backendUrl}</div>
              {incUrl !== backendUrl && (
                <div className="text-[9px] font-mono text-amber-400/70 truncate mt-1">
                  INC: {incUrl}
                </div>
              )}
              {incUrl === backendUrl && (
                <div className="text-[9px] font-mono text-gray-700 mt-1">
                  Engine + INC routes on same server
                </div>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
