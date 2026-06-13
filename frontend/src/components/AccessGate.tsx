import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import type { AccessMode } from '../types'

interface Props {
  onAccess: (mode: AccessMode) => void
}

const VALID_CODES: Record<string, AccessMode> = {
  MOCK:  'MOCK',
  DEMO:  'DEMO',
  ADMIN: 'ADMIN',
}

const MODE_LABELS: Record<AccessMode, string> = {
  MOCK:  'MOCK DATA MODE',
  DEMO:  'DEMO MODE',
  ADMIN: 'ADMIN MODE',
}

export default function AccessGate({ onAccess }: Props) {
  const [code, setCode] = useState('')
  const [status, setStatus] = useState<'idle' | 'error' | 'success'>('idle')
  const [grantedMode, setGrantedMode] = useState<AccessMode | null>(null)
  const [shakeKey, setShakeKey] = useState(0)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const upper = code.trim().toUpperCase()
    const mode = VALID_CODES[upper]
    if (mode) {
      setGrantedMode(mode)
      setStatus('success')
      setTimeout(() => onAccess(mode), 800)
    } else {
      setShakeKey((k) => k + 1)
      setStatus('error')
      setCode('')
      setTimeout(() => setStatus('idle'), 2200)
    }
  }

  return (
    <div
      className="min-h-screen bg-gray-950 flex flex-col items-center justify-center"
      style={{
        backgroundImage:
          'linear-gradient(rgba(31,41,55,0.18) 1px, transparent 1px), linear-gradient(90deg, rgba(31,41,55,0.18) 1px, transparent 1px)',
        backgroundSize: '40px 40px',
      }}
    >
      <motion.div
        key={shakeKey}
        animate={status === 'error' ? { x: [0, -12, 12, -8, 8, -4, 4, 0] } : {}}
        transition={{ duration: 0.5, ease: 'easeInOut' }}
        className="w-full max-w-[340px] px-4"
      >
        <div className="text-center mb-10">
          <div className="flex items-center justify-center gap-3 mb-3">
            <img
              src="/favicon.svg"
              alt="EchoFace"
              className="w-10 h-10 object-contain"
              style={{ filter: 'brightness(0) invert(1)' }}
            />
            <span className="text-4xl font-mono font-semibold text-cyan-400 tracking-tight">
              ECHOFACE
            </span>
          </div>
          <div className="text-[11px] font-mono text-gray-600 tracking-[0.3em]">
            IDENTITY INTELLIGENCE PLATFORM
          </div>
        </div>

        <div className="border border-gray-800 bg-gray-900/90 backdrop-blur-sm rounded-lg p-8">
          <div className="text-[10px] font-mono text-gray-500 tracking-[0.25em] text-center mb-5">
            ENTER ACCESS CODE
          </div>

          <form onSubmit={handleSubmit} autoComplete="off">
            <input
              type="text"
              value={code}
              onChange={(e) => setCode(e.target.value.toUpperCase())}
              className={`w-full bg-gray-950 border rounded px-4 py-3 font-mono text-2xl text-center tracking-[0.4em] outline-none transition-all duration-200 placeholder-gray-800 text-gray-100 ${
                status === 'error'
                  ? 'border-red-500/60'
                  : status === 'success'
                  ? 'border-emerald-500/60'
                  : 'border-gray-700 focus:border-cyan-500/70'
              }`}
              placeholder="· · · · · ·"
              maxLength={16}
              autoFocus
              spellCheck={false}
            />

            <button
              type="submit"
              className="mt-4 w-full border border-cyan-500/40 hover:bg-cyan-500/15 active:bg-cyan-500/25 text-cyan-400 font-mono text-sm tracking-[0.2em] py-3 rounded transition-all duration-150"
            >
              AUTHENTICATE
            </button>
          </form>

          <div className="h-8 mt-4 flex items-center justify-center">
            <AnimatePresence mode="wait">
              {status === 'error' && (
                <motion.div
                  key="denied"
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="font-mono text-xs text-red-400 tracking-[0.25em] flex items-center gap-2"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
                  ACCESS DENIED
                </motion.div>
              )}
              {status === 'success' && grantedMode && (
                <motion.div
                  key="auth"
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="font-mono text-xs text-emerald-400 tracking-[0.25em] flex items-center gap-2"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  {MODE_LABELS[grantedMode]}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        <div className="mt-6 text-center font-mono text-[10px] text-gray-800">
          EchoFace Lite v1.0.1 &nbsp;·&nbsp; Restricted Access
        </div>
      </motion.div>
    </div>
  )
}
