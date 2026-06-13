import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, CheckCircle2, XCircle, Loader2, Cpu, Globe, Server, RefreshCw } from 'lucide-react'
import { useAppStore, BACKENDS, type BackendEntry } from '../store/appStore'
import { useHealthCheck } from '../api/hooks'

interface Props {
  open: boolean
  onClose: () => void
}

function BackendRow({
  entry,
  isActive,
  onSelect,
}: {
  entry: BackendEntry
  isActive: boolean
  onSelect: () => void
}) {
  const { status, recheck } = useHealthCheck(entry.url)

  return (
    <button
      onClick={onSelect}
      className={`w-full text-left border rounded-lg p-4 transition-all duration-150 ${
        isActive
          ? 'border-cyan-500/40 bg-cyan-500/8'
          : 'border-gray-700 bg-gray-800/30 hover:border-gray-600 hover:bg-gray-800/50'
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3">
          <div
            className={`w-8 h-8 rounded flex items-center justify-center flex-shrink-0 ${
              entry.type === 'local' ? 'bg-cyan-500/15' : 'bg-amber-500/15'
            }`}
          >
            {entry.type === 'local' ? (
              <Cpu size={14} className="text-cyan-400" />
            ) : (
              <Globe size={14} className="text-amber-400" />
            )}
          </div>
          <div className="min-w-0">
            <div className="text-sm font-semibold text-gray-200">{entry.name}</div>
            <div className="text-[10px] font-mono text-gray-600 mt-0.5 truncate">{entry.url}</div>
          </div>
        </div>

        <div className="flex items-center gap-1.5 flex-shrink-0 mt-0.5">
          {status === 'checking' && <Loader2 size={11} className="text-gray-500 animate-spin" />}
          {status === 'online' && <CheckCircle2 size={11} className="text-emerald-400" />}
          {status === 'offline' && <XCircle size={11} className="text-red-400" />}
          <span
            className={`text-[9px] font-mono tracking-wider ${
              status === 'online'
                ? 'text-emerald-400'
                : status === 'offline'
                ? 'text-red-400'
                : 'text-gray-600'
            }`}
          >
            {status.toUpperCase()}
          </span>
          <button
            onClick={(e) => {
              e.stopPropagation()
              recheck()
            }}
            className="ml-1 text-gray-700 hover:text-gray-400 transition-colors"
          >
            <RefreshCw size={10} />
          </button>
        </div>
      </div>

      {isActive && (
        <div className="mt-2 flex items-center gap-1.5 text-[10px] font-mono text-cyan-400">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse-dot" />
          ACTIVE
        </div>
      )}
    </button>
  )
}

export default function BackendPanel({ open, onClose }: Props) {
  const { backendName, backendUrl, setBackend } = useAppStore()
  const [customUrl, setCustomUrl] = useState('')
  const [showCustom, setShowCustom] = useState(false)

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 z-40"
            onClick={onClose}
          />

          <motion.div
            initial={{ x: -300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -300, opacity: 0 }}
            transition={{ type: 'spring', damping: 28, stiffness: 220 }}
            className="fixed left-56 top-0 bottom-0 w-80 bg-gray-900 border-r border-gray-700 z-50 flex flex-col shadow-2xl"
          >
            <div className="flex items-center justify-between px-5 py-4 border-b border-gray-800">
              <div>
                <h2 className="text-sm font-semibold text-gray-100">Backend Registry</h2>
                <div className="text-[10px] font-mono text-gray-600 mt-0.5">
                  Switch API target · mirrors backend_registry.py
                </div>
              </div>
              <button
                onClick={onClose}
                className="text-gray-500 hover:text-gray-300 transition-colors"
              >
                <X size={16} />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {BACKENDS.map((entry) => (
                <BackendRow
                  key={entry.name}
                  entry={entry}
                  isActive={backendUrl === entry.url}
                  onSelect={() => {
                    setBackend(entry.name, entry.url)
                    onClose()
                  }}
                />
              ))}

              <div className="pt-3 border-t border-gray-800">
                <button
                  onClick={() => setShowCustom((v) => !v)}
                  className="text-[11px] font-mono text-gray-600 hover:text-cyan-400 transition-colors flex items-center gap-2"
                >
                  <Server size={11} />
                  {showCustom ? '− cancel custom URL' : '+ custom backend URL'}
                </button>

                {showCustom && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    className="mt-3 space-y-2 overflow-hidden"
                  >
                    <input
                      value={customUrl}
                      onChange={(e) => setCustomUrl(e.target.value)}
                      placeholder="http://host:8000/api/v1"
                      className="w-full bg-gray-950 border border-gray-700 focus:border-cyan-600/50 rounded px-3 py-2 text-xs font-mono text-gray-200 outline-none placeholder-gray-700 transition-colors"
                    />
                    <button
                      onClick={() => {
                        const url = customUrl.trim()
                        if (url) {
                          setBackend('Custom', url)
                          setCustomUrl('')
                          setShowCustom(false)
                          onClose()
                        }
                      }}
                      disabled={!customUrl.trim()}
                      className="w-full py-2 bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 rounded text-[11px] font-mono tracking-wider hover:bg-cyan-500/20 transition-colors disabled:opacity-30 disabled:pointer-events-none"
                    >
                      CONNECT
                    </button>
                  </motion.div>
                )}
              </div>
            </div>

            <div className="px-5 py-4 border-t border-gray-800 space-y-1">
              <div className="text-[10px] font-mono text-gray-600">
                ACTIVE:{' '}
                <span className="text-gray-300">{backendName}</span>
              </div>
              <div className="text-[9px] font-mono text-gray-700 truncate">{backendUrl}</div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
