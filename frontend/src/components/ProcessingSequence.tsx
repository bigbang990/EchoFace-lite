import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Check, Loader2 } from 'lucide-react'

export interface ProcessingStep {
  label: string
  detail?: string
  durationMs?: number
}

interface Props {
  steps: ProcessingStep[]
  onComplete?: () => void
}

export default function ProcessingSequence({ steps, onComplete }: Props) {
  const [current, setCurrent] = useState(0)
  const [done, setDone] = useState<Set<number>>(new Set())

  useEffect(() => {
    if (current >= steps.length) {
      onComplete?.()
      return
    }
    const ms = steps[current]?.durationMs ?? 1500
    const t = setTimeout(() => {
      setDone((prev) => new Set([...prev, current]))
      setCurrent((c) => c + 1)
    }, ms)
    return () => clearTimeout(t)
  }, [current, steps, onComplete])

  const allDone = current >= steps.length

  return (
    <div className="space-y-4">
      {steps.map((step, i) => {
        const isComplete = done.has(i)
        const isCurrent = i === current
        const isPending = i > current

        return (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: isPending ? 0.35 : 1, y: 0 }}
            transition={{ delay: i * 0.08 }}
            className="flex items-start gap-4"
          >
            <div className="flex-shrink-0 w-8 h-8 flex items-center justify-center">
              <AnimatePresence mode="wait">
                {isComplete && (
                  <motion.div
                    key="check"
                    initial={{ scale: 0, rotate: -90 }}
                    animate={{ scale: 1, rotate: 0 }}
                    transition={{ type: 'spring', stiffness: 400, damping: 20 }}
                    className="w-8 h-8 rounded-full bg-emerald-500/15 border border-emerald-500/40 flex items-center justify-center"
                  >
                    <Check size={13} className="text-emerald-400" strokeWidth={2.5} />
                  </motion.div>
                )}
                {isCurrent && (
                  <motion.div
                    key="spin"
                    initial={{ scale: 0.6, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="w-8 h-8 rounded-full bg-cyan-500/15 border border-cyan-500/40 flex items-center justify-center"
                  >
                    <Loader2 size={13} className="text-cyan-400 animate-spin" />
                  </motion.div>
                )}
                {isPending && (
                  <motion.div
                    key="pending"
                    className="w-8 h-8 rounded-full bg-gray-800 border border-gray-700 flex items-center justify-center"
                  >
                    <div className="w-2 h-2 rounded-full bg-gray-700" />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            <div className="flex-1 pt-1.5 min-w-0">
              <div className={`text-sm font-medium leading-tight ${
                isComplete ? 'text-emerald-400' : isCurrent ? 'text-cyan-400' : 'text-gray-600'
              }`}>
                {step.label}
              </div>
              {step.detail && (
                <div className="text-[11px] font-mono text-gray-600 mt-0.5">{step.detail}</div>
              )}
              {isCurrent && (
                <motion.div
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: 1 }}
                  transition={{ duration: (step.durationMs ?? 1500) / 1000, ease: 'linear' }}
                  className="h-px bg-cyan-500/40 mt-2 origin-left"
                />
              )}
            </div>

            <div className="pt-1.5 flex-shrink-0">
              {isComplete && (
                <span className="text-[10px] font-mono text-emerald-600">DONE</span>
              )}
              {isCurrent && (
                <span className="text-[10px] font-mono text-cyan-600 animate-pulse">PROCESSING</span>
              )}
            </div>
          </motion.div>
        )
      })}

      {allDone && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="pt-4 border-t border-gray-800"
        >
          <div className="flex items-center gap-2 text-emerald-400">
            <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse-dot" />
            <span className="text-sm font-semibold">All systems operational — tracking active</span>
          </div>
        </motion.div>
      )}
    </div>
  )
}
