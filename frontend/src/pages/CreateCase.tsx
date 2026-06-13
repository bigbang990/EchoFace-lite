import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, X, ArrowRight, ArrowLeft, ExternalLink, AlertTriangle, CheckCircle2, XCircle, Loader2, AlertCircle } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { nextIncidentRef } from '../mock/data'

type Step = 'person' | 'last-seen' | 'photos' | 'processing' | 'success'

interface FormData {
  name: string
  age: string
  gender: string
  description: string
  location: string
  lastSeenDate: string
  lastSeenTime: string
  notes: string
  photos: File[]
}

interface ProcStep {
  label: string
  status: 'idle' | 'running' | 'ok' | 'warn' | 'fail'
  detail: string
}

interface PhotoWarning {
  rejected: number
  accepted: number
  reasons: string[]
}

const PROC_STEPS: ProcStep[] = [
  { label: 'Analyzing reference photos',   status: 'idle', detail: '' },
  { label: 'Creating face embeddings',      status: 'idle', detail: '' },
  { label: 'Creating case record',          status: 'idle', detail: '' },
  { label: 'Activating tracking profile',   status: 'idle', detail: '' },
]

const STEPS: Step[] = ['person', 'last-seen', 'photos', 'processing', 'success']
const GENDERS = ['Female', 'Male', 'Non-binary / Other', 'Prefer not to say']

export default function CreateCase() {
  const navigate = useNavigate()
  const { accessMode, incUrl } = useAppStore()
  const fileRef = useRef<HTMLInputElement>(null)
  const [step, setStep] = useState<Step>('person')
  const [caseRef] = useState(() => nextIncidentRef())
  const [form, setForm] = useState<FormData>({
    name: '',
    age: '',
    gender: 'Female',
    description: '',
    location: '',
    lastSeenDate: new Date().toISOString().slice(0, 10),
    lastSeenTime: '21:30',
    notes: '',
    photos: [],
  })

  const [procSteps, setProcSteps] = useState<ProcStep[]>(PROC_STEPS.map((s) => ({ ...s })))
  const [photoWarning, setPhotoWarning] = useState<PhotoWarning | null>(null)
  // Refs survive async continuations without stale-closure issues
  const personIdRef = useRef<string>('')
  const incidentIdRef = useRef<string>('')

  const currentIdx = STEPS.indexOf(step)

  const update = (field: keyof FormData, value: string | File[]) =>
    setForm((f) => ({ ...f, [field]: value }))

  const next = () => setStep(STEPS[currentIdx + 1])
  const back = () => setStep(STEPS[currentIdx - 1])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) update('photos', [...form.photos, ...Array.from(e.target.files)])
  }

  const removePhoto = (i: number) =>
    update('photos', form.photos.filter((_, idx) => idx !== i))

  const updateStep = (idx: number, status: ProcStep['status'], detail: string) =>
    setProcSteps((prev) => prev.map((s, i) => (i === idx ? { ...s, status, detail } : s)))

  // MOCK: fake delays, then success screen
  const runMockProcessing = async () => {
    const details = [
      'Face detected · 1 embedding generated',
      `${form.photos.length} photo${form.photos.length !== 1 ? 's' : ''} · embeddings generated`,
      'Case record written to database',
      `${form.name || 'Unknown'} enrolled · tracking active`,
    ]
    const delays = [1200, 2200, 900, 1100]
    for (let i = 0; i < 4; i++) {
      updateStep(i, 'running', '')
      await new Promise((r) => setTimeout(r, delays[i]))
      updateStep(i, 'ok', details[i])
    }
    setTimeout(() => setStep('success'), 600)
  }

  // REAL step 1: enroll person with first photo
  const runProcessing = async () => {
    personIdRef.current = ''
    incidentIdRef.current = ''
    setPhotoWarning(null)
    setProcSteps(PROC_STEPS.map((s) => ({ ...s })))

    // Step 0: first photo → create person + embedding
    updateStep(0, 'running', '')
    const pForm = new FormData()
    pForm.append('display_name', form.name)
    pForm.append(
      'notes',
      `Age: ${form.age || 'unknown'}, Gender: ${form.gender}. ${form.description}`
    )
    if (form.photos.length > 0) pForm.append('image', form.photos[0])

    let personId: string
    try {
      const pRes = await fetch(`${incUrl}/persons`, { method: 'POST', body: pForm })
      if (!pRes.ok) {
        const errText = await pRes.text().catch(() => pRes.statusText)
        updateStep(0, 'fail', `Enrollment failed: ${errText}`)
        return
      }
      const pData = await pRes.json()
      // Backend returns PersonEnrollOut = { person: PersonOut, deduplicated: bool }
      personId = String(pData.person?.id ?? pData.id ?? '')
      if (!personId) {
        updateStep(0, 'fail', 'No person ID in response — check backend logs')
        return
      }
      personIdRef.current = personId
      const dedup = pData.deduplicated ? ' · matched existing profile' : ''
      updateStep(0, 'ok', `Face detected · embedding generated${dedup}`)
    } catch (e) {
      updateStep(0, 'fail', (e as Error).message)
      return
    }

    // Step 1: additional photos (one by one for per-photo feedback)
    let totalAccepted = 1
    let totalRejected = 0
    const rejectionReasons: string[] = []

    if (form.photos.length > 1) {
      updateStep(
        1,
        'running',
        `Processing ${form.photos.length - 1} additional photo${form.photos.length > 2 ? 's' : ''}…`
      )
      for (let i = 1; i < form.photos.length; i++) {
        try {
          const ef = new FormData()
          ef.append('images', form.photos[i])
          const res = await fetch(`${incUrl}/persons/${personId}/photos`, {
            method: 'POST',
            body: ef,
          })
          if (res.ok) {
            const d = await res.json()
            // PersonEnrollMultiOut = { person, photos_accepted, photos_rejected, rejection_reasons }
            totalAccepted += Number(d.photos_accepted ?? 1)
            const rej = Number(d.photos_rejected ?? 0)
            totalRejected += rej
            if (rej > 0 && Array.isArray(d.rejection_reasons)) {
              rejectionReasons.push(...(d.rejection_reasons as string[]))
            }
          }
        } catch { /* continue — non-fatal */ }
      }

      if (totalRejected > 0) {
        const reason = rejectionReasons[0] ?? 'no clear face detected'
        updateStep(
          1,
          'warn',
          `${totalAccepted} accepted · ${totalRejected} rejected (${reason})`
        )
        setPhotoWarning({ rejected: totalRejected, accepted: totalAccepted, reasons: rejectionReasons })
        return // pause — wait for agent confirmation below
      } else {
        updateStep(1, 'ok', `${totalAccepted} photos · ${totalAccepted} embeddings generated`)
      }
    } else {
      updateStep(1, 'ok', 'Primary photo processed')
    }

    await continueCreatingCase()
  }

  // REAL steps 2–3: create incident + link person
  const continueCreatingCase = async () => {
    const personId = personIdRef.current

    // Step 2: create incident
    updateStep(2, 'running', '')
    let incidentId: string
    try {
      const incRes = await fetch(`${incUrl}/incidents`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: `Missing: ${form.name}`,
          description: form.notes
            ? `${form.notes}. Last seen: ${form.location}`
            : `Last seen: ${form.location} on ${form.lastSeenDate} at ${form.lastSeenTime}`,
          operator_id: 'operator',
        }),
      })
      if (!incRes.ok) {
        const errText = await incRes.text().catch(() => incRes.statusText)
        updateStep(2, 'fail', `Failed: ${errText}`)
        return
      }
      const incData = await incRes.json()
      incidentId = String(incData.id ?? '')
      incidentIdRef.current = incidentId
      const ref = incData.ref ? ` · ${incData.ref}` : ''
      updateStep(2, 'ok', `Case created${ref}`)
    } catch (e) {
      updateStep(2, 'fail', (e as Error).message)
      return
    }

    // Step 3: link person → incident
    updateStep(3, 'running', '')
    try {
      const linkRes = await fetch(
        `${incUrl}/incidents/${incidentId}/persons/${personId}`,
        { method: 'POST' }
      )
      if (!linkRes.ok) {
        updateStep(3, 'fail', `Link failed (${linkRes.status}) — person was enrolled but not linked`)
        return
      }
      updateStep(3, 'ok', `${form.name} enrolled · tracking profile active`)
      setTimeout(() => navigate(`/cases/${incidentId}`), 1400)
    } catch (e) {
      updateStep(3, 'fail', (e as Error).message)
    }
  }

  const handleSubmit = () => {
    setStep('processing')
    setProcSteps(PROC_STEPS.map((s) => ({ ...s })))
    setPhotoWarning(null)
    if (accessMode === 'MOCK') {
      runMockProcessing()
    } else {
      runProcessing()
    }
  }

  return (
    <div className="p-8 max-w-2xl">
      <div className="mb-7">
        <h1 className="text-xl font-semibold text-gray-100">Create New Case</h1>
        <p className="text-xs font-mono text-gray-600 mt-1">
          Open a missing person investigation and register them with the tracking pipeline
        </p>
      </div>

      <StepProgress current={currentIdx} total={3} />

      <div className="mt-8 bg-gray-900 border border-gray-800 rounded-lg p-7">
        <AnimatePresence mode="wait">
          {step === 'person' && (
            <StepPanel key="person" title="Person Details" subtitle="Step 1 of 3">
              <Field label="Full name">
                <input
                  type="text"
                  value={form.name}
                  onChange={(e) => update('name', e.target.value)}
                  className={inputCls}
                  placeholder="e.g. Sarah Chen"
                />
              </Field>
              <div className="grid grid-cols-2 gap-4">
                <Field label="Age">
                  <input
                    type="number"
                    value={form.age}
                    onChange={(e) => update('age', e.target.value)}
                    className={inputCls}
                    placeholder="e.g. 24"
                    min={1}
                    max={120}
                  />
                </Field>
                <Field label="Gender">
                  <select
                    value={form.gender}
                    onChange={(e) => update('gender', e.target.value)}
                    className={inputCls}
                  >
                    {GENDERS.map((g) => <option key={g} value={g}>{g}</option>)}
                  </select>
                </Field>
              </div>
              <Field label="Physical description">
                <textarea
                  value={form.description}
                  onChange={(e) => update('description', e.target.value)}
                  className={`${inputCls} resize-none`}
                  rows={3}
                  placeholder="Clothing, distinguishing features, hair colour, etc."
                />
              </Field>
              <div className="flex justify-end">
                <NextBtn disabled={!form.name} onClick={next} />
              </div>
            </StepPanel>
          )}

          {step === 'last-seen' && (
            <StepPanel key="last-seen" title="Last Known Location" subtitle="Step 2 of 3">
              <Field label="Location">
                <input
                  type="text"
                  value={form.location}
                  onChange={(e) => update('location', e.target.value)}
                  className={inputCls}
                  placeholder="e.g. Whitechapel Market, London E1"
                />
              </Field>
              <div className="grid grid-cols-2 gap-4">
                <Field label="Date">
                  <input
                    type="date"
                    value={form.lastSeenDate}
                    onChange={(e) => update('lastSeenDate', e.target.value)}
                    className={inputCls}
                  />
                </Field>
                <Field label="Time">
                  <input
                    type="time"
                    value={form.lastSeenTime}
                    onChange={(e) => update('lastSeenTime', e.target.value)}
                    className={inputCls}
                  />
                </Field>
              </div>
              <Field label="Additional notes">
                <textarea
                  value={form.notes}
                  onChange={(e) => update('notes', e.target.value)}
                  className={`${inputCls} resize-none`}
                  rows={3}
                  placeholder="Context, circumstances, who reported, etc."
                />
              </Field>
              <div className="flex justify-between">
                <BackBtn onClick={back} />
                <NextBtn disabled={!form.location} onClick={next} />
              </div>
            </StepPanel>
          )}

          {step === 'photos' && (
            <StepPanel key="photos" title="Reference Photos" subtitle="Step 3 of 3">
              <p className="text-sm text-gray-500 mb-5">
                Upload one or more clear photos. Multiple angles improve match accuracy.
              </p>
              <input
                ref={fileRef}
                type="file"
                multiple
                accept="image/*"
                className="hidden"
                onChange={handleFileChange}
              />
              <button
                onClick={() => fileRef.current?.click()}
                className="w-full border-2 border-dashed border-gray-700 hover:border-cyan-600/50 rounded-lg p-8 flex flex-col items-center gap-3 transition-colors text-gray-600 hover:text-gray-400"
              >
                <Upload size={24} />
                <span className="text-sm">Click to upload photos</span>
                <span className="text-[11px] font-mono text-gray-700">PNG / JPG / WEBP — multiple files OK</span>
              </button>
              {form.photos.length > 0 && (
                <div className="mt-4 space-y-2">
                  {form.photos.map((file, i) => (
                    <div key={i} className="flex items-center gap-3 bg-gray-800/60 border border-gray-700 rounded px-4 py-2.5">
                      <div className="w-8 h-8 bg-gray-700 rounded flex items-center justify-center flex-shrink-0 overflow-hidden">
                        <img
                          src={URL.createObjectURL(file)}
                          alt=""
                          className="w-full h-full object-cover"
                          onLoad={(e) => URL.revokeObjectURL((e.target as HTMLImageElement).src)}
                        />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-sm text-gray-300 truncate">{file.name}</div>
                        <div className="text-[10px] font-mono text-gray-600">{(file.size / 1024).toFixed(0)} KB</div>
                      </div>
                      <button onClick={() => removePhoto(i)} className="text-gray-600 hover:text-gray-400 transition-colors">
                        <X size={14} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
              <div className="flex justify-between mt-6">
                <BackBtn onClick={back} />
                <button
                  onClick={handleSubmit}
                  disabled={form.photos.length === 0}
                  className="flex items-center gap-2 px-5 py-2.5 bg-cyan-500/15 border border-cyan-500/40 text-cyan-400 rounded text-sm font-medium hover:bg-cyan-500/25 transition-colors disabled:opacity-30 disabled:pointer-events-none"
                >
                  Create Case & Activate
                  <ArrowRight size={14} />
                </button>
              </div>
            </StepPanel>
          )}

          {step === 'processing' && (
            <StepPanel key="processing" title="Creating Case" subtitle={caseRef}>
              <div className="mb-5 text-sm text-gray-500">
                Enrolling{' '}
                <span className="font-semibold text-gray-300">{form.name || 'subject'}</span>
                {' '}— {form.photos.length} photo{form.photos.length !== 1 ? 's' : ''} submitted
              </div>

              <div className="space-y-3">
                {procSteps.map((s, i) => (
                  <div
                    key={i}
                    className={`flex items-start gap-3 rounded-lg px-4 py-3 border transition-colors ${
                      s.status === 'idle'    ? 'border-gray-800 bg-transparent' :
                      s.status === 'running' ? 'border-cyan-500/25 bg-cyan-500/5' :
                      s.status === 'ok'      ? 'border-emerald-500/25 bg-emerald-500/5' :
                      s.status === 'warn'    ? 'border-amber-500/25 bg-amber-500/5' :
                                               'border-red-500/25 bg-red-500/5'
                    }`}
                  >
                    <div className="flex-shrink-0 mt-0.5">
                      {s.status === 'idle'    && <div className="w-4 h-4 rounded-full border border-gray-700" />}
                      {s.status === 'running' && <Loader2 size={16} className="text-cyan-400 animate-spin" />}
                      {s.status === 'ok'      && <CheckCircle2 size={16} className="text-emerald-400" />}
                      {s.status === 'warn'    && <AlertCircle size={16} className="text-amber-400" />}
                      {s.status === 'fail'    && <XCircle size={16} className="text-red-400" />}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className={`text-[13px] font-medium ${
                        s.status === 'idle'    ? 'text-gray-600' :
                        s.status === 'running' ? 'text-cyan-300' :
                        s.status === 'ok'      ? 'text-emerald-300' :
                        s.status === 'warn'    ? 'text-amber-300' :
                                                  'text-red-300'
                      }`}>{s.label}</div>
                      {s.detail && (
                        <div className="text-[11px] font-mono text-gray-500 mt-0.5 break-words">{s.detail}</div>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              {/* Photo quality warning — agent confirmation required */}
              {photoWarning && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-5 bg-amber-500/8 border border-amber-500/30 rounded-lg p-4"
                >
                  <div className="flex items-start gap-3">
                    <AlertTriangle size={15} className="text-amber-400 flex-shrink-0 mt-0.5" />
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-semibold text-amber-300 mb-1">
                        Photo Quality Warning — Agent Confirmation Required
                      </div>
                      <p className="text-[11px] text-amber-400/80 mb-1">
                        {photoWarning.rejected} of {photoWarning.accepted + photoWarning.rejected}{' '}
                        photo{photoWarning.rejected !== 1 ? 's' : ''} could not be processed.
                        {photoWarning.reasons[0] ? ` Reason: ${photoWarning.reasons[0]}` : ''}
                      </p>
                      <p className="text-[10px] text-gray-500 mb-3">
                        These photos may not contain a detectable face or are too low quality.
                        They will not be included in the tracking profile.
                        Confirm to proceed with the {photoWarning.accepted} accepted photo{photoWarning.accepted !== 1 ? 's' : ''}.
                      </p>
                      <div className="flex gap-2">
                        <button
                          onClick={() => { setPhotoWarning(null); void continueCreatingCase() }}
                          className="px-3 py-1.5 bg-amber-500/15 border border-amber-500/30 text-amber-300 rounded text-[11px] font-medium hover:bg-amber-500/25 transition-colors"
                        >
                          Proceed with {photoWarning.accepted} photo{photoWarning.accepted !== 1 ? 's' : ''}
                        </button>
                        <button
                          onClick={() => {
                            setPhotoWarning(null)
                            setProcSteps(PROC_STEPS.map((s) => ({ ...s })))
                            setStep('photos')
                          }}
                          className="px-3 py-1.5 border border-gray-700 text-gray-500 rounded text-[11px] hover:bg-gray-800 hover:text-gray-300 transition-colors"
                        >
                          Re-upload Photos
                        </button>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Any step failed — offer to retry or go back */}
              {!photoWarning && procSteps.some((s) => s.status === 'fail') && (
                <div className="mt-5 flex justify-center gap-3">
                  <button
                    onClick={() => { setProcSteps(PROC_STEPS.map((s) => ({ ...s }))); handleSubmit() }}
                    className="px-4 py-2 bg-cyan-500/15 border border-cyan-500/30 text-cyan-400 rounded text-xs hover:bg-cyan-500/25 transition-colors"
                  >
                    Retry
                  </button>
                  <button
                    onClick={() => setStep('photos')}
                    className="px-4 py-2 border border-gray-700 text-gray-500 rounded text-xs hover:bg-gray-800 hover:text-gray-300 transition-colors"
                  >
                    Back to Photos
                  </button>
                </div>
              )}
            </StepPanel>
          )}

          {step === 'success' && (
            <StepPanel key="success" title="Case Created" subtitle={caseRef}>
              <div className="text-center py-4">
                <motion.div
                  initial={{ scale: 0.6, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ type: 'spring', stiffness: 300 }}
                  className="w-16 h-16 bg-emerald-500/15 border border-emerald-500/40 rounded-full flex items-center justify-center mx-auto mb-5"
                >
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-emerald-400">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                </motion.div>
                <h3 className="text-lg font-semibold text-gray-100 mb-1">{caseRef} is live</h3>
                <p className="text-sm text-gray-500 mb-6">
                  Tracking pipeline is armed. Upload a video source in Operations to begin scanning.
                </p>
                <div className="inline-flex items-center gap-2 px-5 py-2.5 bg-gray-800 border border-gray-700 rounded text-xs font-mono text-gray-400 mb-6">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse-dot" />
                  {form.name} · {form.photos.length} embeddings · TRACKING
                </div>
                <div className="flex justify-center gap-3">
                  <button
                    onClick={() => navigate('/cases')}
                    className="flex items-center gap-2 px-5 py-2.5 bg-cyan-500/15 border border-cyan-500/40 text-cyan-400 rounded text-sm font-medium hover:bg-cyan-500/25 transition-colors"
                  >
                    View All Cases
                    <ExternalLink size={13} />
                  </button>
                  <button
                    onClick={() => navigate('/operations')}
                    className="px-5 py-2.5 border border-gray-700 text-gray-400 rounded text-sm hover:bg-gray-800 transition-colors"
                  >
                    Go to Operations
                  </button>
                </div>
              </div>
            </StepPanel>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

function StepPanel({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 12 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -12 }}
      transition={{ duration: 0.2 }}
    >
      <div className="mb-6">
        <h2 className="text-base font-semibold text-gray-100">{title}</h2>
        <div className="text-[11px] font-mono text-gray-600 mt-0.5">{subtitle}</div>
      </div>
      <div className="space-y-4">{children}</div>
    </motion.div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-xs font-mono text-gray-500 tracking-wider mb-1.5">
        {label.toUpperCase()}
      </label>
      {children}
    </div>
  )
}

function StepProgress({ current, total }: { current: number; total: number }) {
  return (
    <div className="flex items-center gap-2">
      {Array.from({ length: total }).map((_, i) => (
        <div
          key={i}
          className={`h-1 flex-1 rounded-full transition-all duration-300 ${
            i < current ? 'bg-cyan-500' : i === current ? 'bg-cyan-500/50' : 'bg-gray-800'
          }`}
        />
      ))}
    </div>
  )
}

function NextBtn({ onClick, disabled }: { onClick: () => void; disabled?: boolean }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="flex items-center gap-2 px-5 py-2.5 bg-cyan-500/15 border border-cyan-500/40 text-cyan-400 rounded text-sm font-medium hover:bg-cyan-500/25 transition-colors disabled:opacity-30 disabled:pointer-events-none"
    >
      Continue <ArrowRight size={14} />
    </button>
  )
}

function BackBtn({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-2 px-4 py-2.5 border border-gray-700 text-gray-500 rounded text-sm hover:bg-gray-800 hover:text-gray-300 transition-colors"
    >
      <ArrowLeft size={14} /> Back
    </button>
  )
}

const inputCls =
  'w-full bg-gray-950 border border-gray-700 focus:border-cyan-600/60 rounded px-3 py-2.5 text-sm text-gray-100 outline-none transition-colors placeholder-gray-700'
