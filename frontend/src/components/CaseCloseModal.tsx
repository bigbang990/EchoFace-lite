import { useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, AlertTriangle, Paperclip, Loader2 } from 'lucide-react'
import { useAppStore } from '../store/appStore'

const CLOSE_REASONS = [
  { value: 'person_found_safe',     label: 'Person Found — Safe' },
  { value: 'person_found_deceased', label: 'Person Found — Deceased' },
  { value: 'case_withdrawn',        label: 'Case Withdrawn' },
  { value: 'duplicate_case',        label: 'Duplicate Case' },
  { value: 'other',                 label: 'Other' },
] as const

interface Props {
  incidentId: number
  incidentRef: string
  onCancel: () => void
  onClosed: () => void
}

export default function CaseCloseModal({ incidentId, incidentRef, onCancel, onClosed }: Props) {
  const { accessMode, incUrl } = useAppStore()
  const [reason, setReason]     = useState('')
  const [summary, setSummary]   = useState('')
  const [closedBy, setClosedBy] = useState('')
  const [files, setFiles]       = useState<File[]>([])
  const [submitting, setSubmitting] = useState(false)
  const [error, setError]       = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const valid = reason !== '' && summary.trim().length > 0

  const handleSubmit = async () => {
    if (!valid) return
    setSubmitting(true)
    setError(null)
    try {
      if (accessMode === 'MOCK') {
        await new Promise(r => setTimeout(r, 600))
        onClosed()
        return
      }

      // incUrl already contains the /api/v1 prefix (same pattern as patchStatus in CaseWorkspace)
      const base = incUrl.trim().replace(/\/$/, '')

      // Upload evidence files first (if any)
      if (files.length > 0) {
        const fd = new FormData()
        files.forEach(f => fd.append('files', f))
        const evRes = await fetch(`${base}/incidents/${incidentId}/evidence`, {
          method: 'POST',
          body: fd,
        })
        if (!evRes.ok) {
          const msg = await evRes.text().catch(() => evRes.statusText)
          throw new Error(`Evidence upload failed: ${msg}`)
        }
      }

      // Submit closure
      const res = await fetch(`${base}/incidents/${incidentId}/close`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reason,
          summary: summary.trim(),
          closed_by: closedBy.trim() || null,
        }),
      })
      if (!res.ok) {
        const msg = await res.text().catch(() => res.statusText)
        throw new Error(msg)
      }
      onClosed()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={(e) => { if (e.target === e.currentTarget) onCancel() }}
      >
        <motion.div
          className="bg-gray-900 border border-gray-700 rounded-lg w-full max-w-lg mx-4 shadow-2xl"
          initial={{ opacity: 0, scale: 0.95, y: 12 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 12 }}
          transition={{ duration: 0.18 }}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800">
            <div className="flex items-center gap-3">
              <AlertTriangle size={16} className="text-amber-400" />
              <span className="font-mono text-sm font-semibold text-gray-100">
                Close Case — {incidentRef}
              </span>
            </div>
            <button onClick={onCancel} className="text-gray-500 hover:text-gray-300 transition-colors">
              <X size={16} />
            </button>
          </div>

          {/* Body */}
          <div className="px-6 py-5 space-y-4">
            {/* Reason */}
            <div>
              <label className="block text-[10px] font-mono text-gray-500 tracking-wider mb-1.5">
                CLOSING REASON <span className="text-red-500">*</span>
              </label>
              <select
                value={reason}
                onChange={e => setReason(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded text-sm font-mono text-gray-200 px-3 py-2 focus:outline-none focus:border-gray-500"
              >
                <option value="">Select a reason…</option>
                {CLOSE_REASONS.map(r => (
                  <option key={r.value} value={r.value}>{r.label}</option>
                ))}
              </select>
            </div>

            {/* Summary */}
            <div>
              <label className="block text-[10px] font-mono text-gray-500 tracking-wider mb-1.5">
                CLOSING SUMMARY <span className="text-red-500">*</span>
              </label>
              <textarea
                value={summary}
                onChange={e => setSummary(e.target.value)}
                rows={4}
                placeholder="Describe the outcome and any relevant findings…"
                className="w-full bg-gray-800 border border-gray-700 rounded text-sm font-mono text-gray-200 px-3 py-2 resize-none focus:outline-none focus:border-gray-500 placeholder-gray-600"
              />
              <div className="text-right text-[10px] font-mono text-gray-600 mt-0.5">
                {summary.length} / 4000
              </div>
            </div>

            {/* Closed by */}
            <div>
              <label className="block text-[10px] font-mono text-gray-500 tracking-wider mb-1.5">
                CLOSED BY (OPTIONAL)
              </label>
              <input
                type="text"
                value={closedBy}
                onChange={e => setClosedBy(e.target.value)}
                placeholder="Operator name or badge number"
                maxLength={128}
                className="w-full bg-gray-800 border border-gray-700 rounded text-sm font-mono text-gray-200 px-3 py-2 focus:outline-none focus:border-gray-500 placeholder-gray-600"
              />
            </div>

            {/* Evidence */}
            <div>
              <label className="block text-[10px] font-mono text-gray-500 tracking-wider mb-1.5">
                EVIDENCE ATTACHMENTS (OPTIONAL)
              </label>
              <input
                ref={fileRef}
                type="file"
                multiple
                className="hidden"
                onChange={e => setFiles(Array.from(e.target.files ?? []))}
              />
              <button
                type="button"
                onClick={() => fileRef.current?.click()}
                className="flex items-center gap-2 text-[11px] font-mono text-gray-400 hover:text-gray-200 border border-gray-700 hover:border-gray-500 rounded px-3 py-2 transition-colors w-full"
              >
                <Paperclip size={12} />
                {files.length === 0
                  ? 'Attach files…'
                  : `${files.length} file${files.length > 1 ? 's' : ''} selected`}
              </button>
              {files.length > 0 && (
                <ul className="mt-1.5 space-y-0.5">
                  {files.map((f, i) => (
                    <li key={i} className="text-[10px] font-mono text-gray-500 truncate pl-1">
                      {f.name}
                    </li>
                  ))}
                </ul>
              )}
            </div>

            {error && (
              <div className="text-[11px] font-mono text-red-400 bg-red-400/10 border border-red-400/20 rounded px-3 py-2">
                {error}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-gray-800">
            <button
              onClick={onCancel}
              disabled={submitting}
              className="px-4 py-2 text-xs font-mono text-gray-400 hover:text-gray-200 transition-colors disabled:opacity-40"
            >
              CANCEL
            </button>
            <button
              onClick={handleSubmit}
              disabled={!valid || submitting}
              className="flex items-center gap-2 px-4 py-2 text-xs font-mono bg-red-900/40 hover:bg-red-900/60 border border-red-700/50 text-red-300 rounded transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {submitting ? <Loader2 size={12} className="animate-spin" /> : null}
              CONFIRM CLOSE
            </button>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}
