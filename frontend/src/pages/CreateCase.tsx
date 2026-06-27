import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, X, ArrowRight, ArrowLeft, ExternalLink, AlertTriangle, CheckCircle2, XCircle, Loader2, AlertCircle, RefreshCw, Check } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { nextIncidentRef } from '../mock/data'

type Step = 'person' | 'last-seen' | 'photos' | 'validating' | 'face-picker' | 'processing' | 'success'

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

interface EnrollmentConflict {
  conflict: true
  person_id: number
  person_name: string
  incident_id: number
  incident_ref: string
  incident_title: string
  incident_status: string
  incident_opened_at: string
  similarity: number
}

interface DetectedFace {
  face_id: string
  bbox: number[]
  det_score: number
  pose_bucket: string
  quality_score: number | null
  thumbnail_b64: string
  is_recommended: boolean
}

interface PhotoDetectionResult {
  photo_index: number
  filename: string
  image_hash: string
  status: 'ok' | 'no_face' | 'invalid'
  reason: string | null
  photo_thumbnail_b64: string | null
  faces: DetectedFace[]
}

interface BatchDetectionOut {
  photos: PhotoDetectionResult[]
  total_photos: number
  photos_with_faces: number
  photos_with_no_face: number
}

const PROC_STEPS: ProcStep[] = [
  { label: 'Re-validating selected faces',  status: 'idle', detail: '' },
  { label: 'Creating face embeddings',      status: 'idle', detail: '' },
  { label: 'Creating case record',          status: 'idle', detail: '' },
  { label: 'Activating tracking profile',   status: 'idle', detail: '' },
]

const STEPS: Step[] = ['person', 'last-seen', 'photos', 'validating', 'face-picker', 'processing', 'success']
const GENDERS = ['Female', 'Male', 'Non-binary / Other', 'Prefer not to say']

const POSE_LABELS: Record<string, string> = {
  frontal: 'Frontal',
  left_profile: 'Left profile',
  right_profile: 'Right profile',
  partial: 'Partial',
  unknown: 'Unknown',
}

const POSE_COLORS: Record<string, string> = {
  frontal: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
  left_profile: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
  right_profile: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
  partial: 'bg-orange-500/20 text-orange-300 border-orange-500/30',
  unknown: 'bg-gray-500/20 text-gray-300 border-gray-500/30',
}

export default function CreateCase() {
  const navigate = useNavigate()
  const { accessMode, incUrl } = useAppStore()
  const fileRef = useRef<HTMLInputElement>(null)
  const [step, setStep] = useState<Step>('person')
  const [caseRef] = useState(() => nextIncidentRef())
  const [form, setForm] = useState<FormData>({
    name: '', age: '', gender: 'Female', description: '',
    location: '', lastSeenDate: new Date().toISOString().slice(0, 10),
    lastSeenTime: '21:30', notes: '', photos: [],
  })

  const [procSteps, setProcSteps] = useState<ProcStep[]>(PROC_STEPS.map((s) => ({ ...s })))
  const [enrollConflict, setEnrollConflict] = useState<EnrollmentConflict | null>(null)
  const [conflictConfirmText, setConflictConfirmText] = useState('')

  const [detectionResults, setDetectionResults] = useState<BatchDetectionOut | null>(null)
  const [detectionLoading, setDetectionLoading] = useState(false)
  const [detectionError, setDetectionError] = useState<string | null>(null)
  // Map of photo_index → selected face_id
  const [faceSelections, setFaceSelections] = useState<Record<number, string>>({})
  // Outliers reported by the backend on confirm-enroll
  const [outlierIndices, setOutlierIndices] = useState<number[]>([])
  const [outlierApproved, setOutlierApproved] = useState(false)

  const personIdRef = useRef<string>('')
  const incidentIdRef = useRef<string>('')
  const isSubmittingRef = useRef(false)

  // For UX the step progress bar should only count user-visible steps
  const visibleStepIndex = (s: Step): number => {
    if (s === 'person') return 0
    if (s === 'last-seen') return 1
    if (s === 'photos') return 2
    if (s === 'validating' || s === 'face-picker') return 3
    return 4
  }

  const update = (field: keyof FormData, value: string | File[]) =>
    setForm((f) => ({ ...f, [field]: value }))

  const next = () => {
    const idx = STEPS.indexOf(step)
    setStep(STEPS[idx + 1])
  }
  const back = () => {
    const idx = STEPS.indexOf(step)
    setStep(STEPS[idx - 1])
  }

  // Dedupe by name + size + lastModified — same file selected twice no longer creates duplicates
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return
    const newFiles = Array.from(e.target.files)
    const existingKeys = new Set(form.photos.map(f => `${f.name}|${f.size}|${f.lastModified}`))
    const filtered = newFiles.filter(f => !existingKeys.has(`${f.name}|${f.size}|${f.lastModified}`))
    if (filtered.length > 0) {
      update('photos', [...form.photos, ...filtered])
    }
    // Reset input so re-selecting the same file would still trigger a change event next time
    e.target.value = ''
  }

  const removePhoto = (i: number) =>
    update('photos', form.photos.filter((_, idx) => idx !== i))

  const updateStep = (idx: number, status: ProcStep['status'], detail: string) =>
    setProcSteps((prev) => prev.map((s, i) => (i === idx ? { ...s, status, detail } : s)))

  // ── Stage 1: detect faces in all uploaded photos ────────────────────────────
  const runDetection = async () => {
    if (form.photos.length === 0) return
    setDetectionLoading(true)
    setDetectionError(null)
    setDetectionResults(null)
    setFaceSelections({})
    setOutlierIndices([])
    setOutlierApproved(false)
    setStep('validating')

    if (accessMode === 'MOCK') {
      await new Promise(r => setTimeout(r, 900))
      const mock: BatchDetectionOut = {
        photos: form.photos.map((f, i) => ({
          photo_index: i, filename: f.name, image_hash: `mock-${i}`,
          status: 'ok', reason: null, photo_thumbnail_b64: null,
          faces: [{
            face_id: `face-${i}-0`, bbox: [0, 0, 100, 100],
            det_score: 0.95, pose_bucket: 'frontal', quality_score: 0.85,
            thumbnail_b64: '', is_recommended: true,
          }],
        })),
        total_photos: form.photos.length,
        photos_with_faces: form.photos.length,
        photos_with_no_face: 0,
      }
      setDetectionResults(mock)
      // auto-select the recommended face per photo
      const sel: Record<number, string> = {}
      mock.photos.forEach(p => {
        const rec = p.faces.find(f => f.is_recommended)
        if (rec) sel[p.photo_index] = rec.face_id
      })
      setFaceSelections(sel)
      setDetectionLoading(false)
      setStep('face-picker')
      return
    }

    try {
      const fd = new FormData()
      for (const photo of form.photos) fd.append('images', photo)
      const res = await fetch(`${incUrl}/persons/detect-faces`, { method: 'POST', body: fd })
      if (!res.ok) {
        const txt = await res.text().catch(() => res.statusText)
        setDetectionError(`Face detection failed: ${txt}`)
        setDetectionLoading(false)
        return
      }
      const data = await res.json() as BatchDetectionOut
      setDetectionResults(data)
      // Auto-select the recommended face on photos with only one face,
      // or the recommended-flagged face when multiple
      const sel: Record<number, string> = {}
      data.photos.forEach(p => {
        if (p.status !== 'ok' || p.faces.length === 0) return
        if (p.faces.length === 1) {
          sel[p.photo_index] = p.faces[0].face_id
        } else {
          const rec = p.faces.find(f => f.is_recommended)
          if (rec) sel[p.photo_index] = rec.face_id
        }
      })
      setFaceSelections(sel)
      setStep('face-picker')
    } catch (e) {
      setDetectionError((e as Error).message)
    }
    setDetectionLoading(false)
  }

  const selectFace = (photoIdx: number, faceId: string) => {
    setFaceSelections(prev => ({ ...prev, [photoIdx]: faceId }))
    setOutlierIndices([])  // re-validate on next submit
  }

  const photosWithSelection = (): number[] => {
    return Object.keys(faceSelections).map(Number).filter(idx => faceSelections[idx])
  }

  const canSubmitSelections = (): boolean => {
    if (!detectionResults) return false
    // At least one photo must have a face selected
    return photosWithSelection().length > 0
  }

  // ── Stage 2: confirm selections + create person ─────────────────────────────
  const runEnrollment = async (forceOutlier = false, forceConflict = false) => {
    if (isSubmittingRef.current) return
    if (!detectionResults) return
    isSubmittingRef.current = true
    personIdRef.current = ''
    incidentIdRef.current = ''
    setEnrollConflict(null)
    setConflictConfirmText('')
    setProcSteps(PROC_STEPS.map((s) => ({ ...s })))
    setStep('processing')

    // Build selections payload
    const selectedPhotoIndices = photosWithSelection()
    const selections = selectedPhotoIndices.map(idx => {
      const photo = detectionResults.photos.find(p => p.photo_index === idx)!
      return {
        photo_index: idx,
        image_hash: photo.image_hash,
        face_id: faceSelections[idx],
      }
    })

    if (accessMode === 'MOCK') {
      // Mock the four steps
      const details = [
        `${selections.length} face${selections.length !== 1 ? 's' : ''} re-validated`,
        `${selections.length} embedding${selections.length !== 1 ? 's' : ''} generated`,
        'Case record written to database',
        `${form.name || 'Unknown'} enrolled · tracking active`,
      ]
      const delays = [800, 1400, 700, 900]
      for (let i = 0; i < 4; i++) {
        updateStep(i, 'running', '')
        await new Promise((r) => setTimeout(r, delays[i]))
        updateStep(i, 'ok', details[i])
      }
      setTimeout(() => setStep('success'), 600)
      isSubmittingRef.current = false
      return
    }

    // Stage 1: re-validate (backend re-detects + checks outliers)
    updateStep(0, 'running', '')
    const fd = new FormData()
    fd.append('display_name', form.name)
    fd.append('notes', `Age: ${form.age || 'unknown'}, Gender: ${form.gender}. ${form.description}`)
    fd.append('selections_json', JSON.stringify(selections))
    if (forceOutlier) fd.append('skip_outlier_check', 'true')
    if (forceConflict) fd.append('force_create', 'true')
    // Re-upload the selected photos' file bytes
    for (const idx of selectedPhotoIndices) {
      fd.append('images', form.photos[idx])
    }

    let personId: string
    try {
      const res = await fetch(`${incUrl}/persons/confirm-enroll`, { method: 'POST', body: fd })

      if (res.status === 409) {
        const errData = await res.json().catch(() => null)
        if (errData?.detail?.outliers_detected) {
          // Outlier — surface back to operator
          updateStep(0, 'warn', `${errData.detail.outlier_photo_indices.length} photo(s) flagged as potential identity mismatch`)
          setOutlierIndices(errData.detail.outlier_photo_indices)
          setStep('face-picker')
          isSubmittingRef.current = false
          return
        }
        if (errData?.detail?.conflict) {
          setEnrollConflict(errData.detail as EnrollmentConflict)
          updateStep(0, 'fail', 'Identity conflict — person may already be in an active case')
          isSubmittingRef.current = false
          return
        }
      }

      if (!res.ok) {
        const errText = await res.text().catch(() => res.statusText)
        updateStep(0, 'fail', `Enrollment failed: ${errText}`)
        isSubmittingRef.current = false
        return
      }

      const pData = await res.json()
      personId = String(pData.person?.id ?? '')
      if (!personId) {
        updateStep(0, 'fail', 'No person ID in response')
        isSubmittingRef.current = false
        return
      }
      personIdRef.current = personId
      updateStep(0, 'ok', `${selections.length} face${selections.length !== 1 ? 's' : ''} re-validated`)
      updateStep(1, 'ok', `${selections.length} embedding${selections.length !== 1 ? 's' : ''} generated`)
    } catch (e) {
      updateStep(0, 'fail', (e as Error).message)
      isSubmittingRef.current = false
      return
    }

    await continueCreatingCase()
  }

  const continueCreatingCase = async () => {
    if (!personIdRef.current) return
    const personId = personIdRef.current

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
        isSubmittingRef.current = false
        return
      }
      const incData = await incRes.json()
      incidentId = String(incData.id ?? '')
      incidentIdRef.current = incidentId
      const ref = incData.ref ? ` · ${incData.ref}` : ''
      updateStep(2, 'ok', `Case created${ref}`)
    } catch (e) {
      updateStep(2, 'fail', (e as Error).message)
      isSubmittingRef.current = false
      return
    }

    updateStep(3, 'running', '')
    try {
      const linkRes = await fetch(`${incUrl}/incidents/${incidentId}/persons/${personId}`, { method: 'POST' })
      if (!linkRes.ok) {
        updateStep(3, 'fail', `Link failed (${linkRes.status})`)
        isSubmittingRef.current = false
        return
      }
      updateStep(3, 'ok', `${form.name} enrolled · tracking profile active`)
      isSubmittingRef.current = false
      setTimeout(() => navigate(`/cases/${incidentId}`), 1400)
    } catch (e) {
      updateStep(3, 'fail', (e as Error).message)
      isSubmittingRef.current = false
    }
  }

  return (
    <div className="p-8 max-w-3xl">
      <div className="mb-7">
        <h1 className="text-xl font-semibold text-gray-100">Create New Case</h1>
        <p className="text-xs font-mono text-gray-600 mt-1">
          Open a missing person investigation and register them with the tracking pipeline
        </p>
      </div>

      <StepProgress current={visibleStepIndex(step)} total={5} />

      <div className="mt-8 bg-gray-900 border border-gray-800 rounded-lg p-7">
        <AnimatePresence mode="wait">
          {step === 'person' && (
            <StepPanel key="person" title="Person Details" subtitle="Step 1 of 5">
              <Field label="Full name">
                <input type="text" value={form.name} onChange={(e) => update('name', e.target.value)} className={inputCls} placeholder="e.g. Sarah Chen" />
              </Field>
              <div className="grid grid-cols-2 gap-4">
                <Field label="Age">
                  <input type="number" value={form.age} onChange={(e) => update('age', e.target.value)} className={inputCls} placeholder="e.g. 24" min={1} max={120} />
                </Field>
                <Field label="Gender">
                  <select value={form.gender} onChange={(e) => update('gender', e.target.value)} className={inputCls}>
                    {GENDERS.map((g) => <option key={g} value={g}>{g}</option>)}
                  </select>
                </Field>
              </div>
              <Field label="Physical description">
                <textarea value={form.description} onChange={(e) => update('description', e.target.value)} className={`${inputCls} resize-none`} rows={3} placeholder="Clothing, distinguishing features, hair colour, etc." />
              </Field>
              <div className="flex justify-end">
                <NextBtn disabled={!form.name} onClick={next} />
              </div>
            </StepPanel>
          )}

          {step === 'last-seen' && (
            <StepPanel key="last-seen" title="Last Known Location" subtitle="Step 2 of 5">
              <Field label="Location">
                <input type="text" value={form.location} onChange={(e) => update('location', e.target.value)} className={inputCls} placeholder="e.g. Whitechapel Market, London E1" />
              </Field>
              <div className="grid grid-cols-2 gap-4">
                <Field label="Date">
                  <input type="date" value={form.lastSeenDate} onChange={(e) => update('lastSeenDate', e.target.value)} className={inputCls} />
                </Field>
                <Field label="Time">
                  <input type="time" value={form.lastSeenTime} onChange={(e) => update('lastSeenTime', e.target.value)} className={inputCls} />
                </Field>
              </div>
              <Field label="Additional notes">
                <textarea value={form.notes} onChange={(e) => update('notes', e.target.value)} className={`${inputCls} resize-none`} rows={3} placeholder="Context, circumstances, who reported, etc." />
              </Field>
              <div className="flex justify-between">
                <BackBtn onClick={back} />
                <NextBtn disabled={!form.location} onClick={next} />
              </div>
            </StepPanel>
          )}

          {step === 'photos' && (
            <StepPanel key="photos" title="Reference Photos" subtitle="Step 3 of 5">
              <p className="text-sm text-gray-500 mb-5">
                Upload 1–8 photos. Group photos and profile shots are fine — you'll select the subject's face in the next step.
              </p>
              <input ref={fileRef} type="file" multiple accept="image/*" className="hidden" onChange={handleFileChange} />
              <button
                onClick={() => fileRef.current?.click()}
                className="w-full border-2 border-dashed border-gray-700 hover:border-cyan-600/50 rounded-lg p-8 flex flex-col items-center gap-3 transition-colors text-gray-600 hover:text-gray-400"
              >
                <Upload size={24} />
                <span className="text-sm">Click to upload photos</span>
                <span className="text-[11px] font-mono text-gray-700">PNG / JPG / WEBP — up to 8 photos · duplicates are automatically removed</span>
              </button>
              {form.photos.length > 0 && (
                <div className="mt-4 space-y-2">
                  {form.photos.map((file, i) => (
                    <div key={`${file.name}-${file.size}-${file.lastModified}`} className="flex items-center gap-3 bg-gray-800/60 border border-gray-700 rounded px-4 py-2.5">
                      <div className="w-8 h-8 bg-gray-700 rounded flex items-center justify-center flex-shrink-0 overflow-hidden">
                        <img src={URL.createObjectURL(file)} alt="" className="w-full h-full object-cover" onLoad={(e) => URL.revokeObjectURL((e.target as HTMLImageElement).src)} />
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
                  onClick={() => void runDetection()}
                  disabled={form.photos.length === 0}
                  className="flex items-center gap-2 px-5 py-2.5 bg-cyan-500/15 border border-cyan-500/40 text-cyan-400 rounded text-sm font-medium hover:bg-cyan-500/25 transition-colors disabled:opacity-30 disabled:pointer-events-none"
                >
                  Detect Faces <ArrowRight size={14} />
                </button>
              </div>
            </StepPanel>
          )}

          {step === 'validating' && (
            <StepPanel key="validating" title="Detecting Faces" subtitle="Step 4 of 5">
              <div className="flex flex-col items-center py-10 gap-4">
                <Loader2 size={28} className="text-cyan-400 animate-spin" />
                <p className="text-sm text-gray-500">Analyzing {form.photos.length} photo{form.photos.length !== 1 ? 's' : ''}…</p>
                <p className="text-[11px] font-mono text-gray-700">Detecting all faces in each photo</p>
              </div>
              {detectionError && (
                <div className="bg-red-500/8 border border-red-500/30 rounded-lg p-4">
                  <p className="text-xs text-red-400">{detectionError}</p>
                  <button onClick={() => void runDetection()} className="mt-2 flex items-center gap-1.5 text-[11px] text-gray-500 hover:text-gray-300">
                    <RefreshCw size={11} /> Retry
                  </button>
                  <button onClick={() => setStep('photos')} className="ml-3 text-[11px] text-gray-500 hover:text-gray-300">Back to photos</button>
                </div>
              )}
            </StepPanel>
          )}

          {step === 'face-picker' && detectionResults && (
            <StepPanel key="face-picker" title="Select the Subject" subtitle="Step 4 of 5">
              <p className="text-sm text-gray-500 mb-5">
                For each photo, click the face that belongs to <span className="text-gray-300 font-medium">{form.name}</span>.
                Photos with only one detected face are pre-selected.
              </p>

              {outlierIndices.length > 0 && !outlierApproved && (
                <div className="mb-5 bg-amber-500/8 border border-amber-500/30 rounded-lg p-4">
                  <div className="flex items-start gap-3">
                    <AlertTriangle size={15} className="text-amber-400 flex-shrink-0 mt-0.5" />
                    <div className="flex-1">
                      <div className="text-xs font-semibold text-amber-300 mb-1">Identity Mismatch Detected</div>
                      <p className="text-[11px] text-amber-400/80 mb-3">
                        Photo{outlierIndices.length !== 1 ? 's' : ''} #{outlierIndices.map(i => i + 1).join(', #')}{' '}
                        appear{outlierIndices.length === 1 ? 's' : ''} to be a different person than the others.
                        Either re-pick the face on the flagged photo{outlierIndices.length !== 1 ? 's' : ''} or approve to proceed anyway.
                      </p>
                      <div className="flex gap-2">
                        <button
                          onClick={() => setOutlierApproved(true)}
                          className="px-3 py-1.5 bg-amber-500/15 border border-amber-500/30 text-amber-300 rounded text-[11px] font-medium hover:bg-amber-500/25 transition-colors"
                        >
                          Approve anyway
                        </button>
                        <button
                          onClick={() => {
                            // Remove the outlier photos from selections
                            const next = { ...faceSelections }
                            outlierIndices.forEach(i => delete next[i])
                            setFaceSelections(next)
                            setOutlierIndices([])
                          }}
                          className="px-3 py-1.5 border border-gray-700 text-gray-500 rounded text-[11px] hover:bg-gray-800 hover:text-gray-300 transition-colors"
                        >
                          Drop flagged photos
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div className="space-y-4">
                {detectionResults.photos.map((photo) => {
                  const isOutlier = outlierIndices.includes(photo.photo_index)
                  const selected = faceSelections[photo.photo_index]

                  if (photo.status !== 'ok') {
                    return (
                      <div key={photo.photo_index} className="flex items-center gap-3 rounded-lg px-4 py-3 border border-red-500/25 bg-red-500/5">
                        <XCircle size={16} className="text-red-400 flex-shrink-0" />
                        <div className="flex-1 min-w-0">
                          <div className="text-[13px] font-medium text-red-300 truncate">{photo.filename}</div>
                          <div className="text-[10px] font-mono text-gray-500 truncate">{photo.reason ?? 'no face detected'}</div>
                        </div>
                      </div>
                    )
                  }

                  return (
                    <div
                      key={photo.photo_index}
                      className={`rounded-lg border p-4 transition-colors ${
                        isOutlier ? 'border-amber-500/40 bg-amber-500/5' :
                        selected ? 'border-emerald-500/30 bg-emerald-500/5' :
                        'border-gray-800 bg-gray-900/50'
                      }`}
                    >
                      <div className="flex items-start gap-4">
                        {/* Original photo thumbnail */}
                        <div className="w-20 h-20 rounded bg-gray-800 flex-shrink-0 overflow-hidden">
                          {photo.photo_thumbnail_b64 ? (
                            <img src={`data:image/jpeg;base64,${photo.photo_thumbnail_b64}`} alt="" className="w-full h-full object-cover" />
                          ) : null}
                        </div>

                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between gap-2 mb-2">
                            <div className="min-w-0">
                              <div className="text-[12px] font-medium text-gray-300 truncate">{photo.filename}</div>
                              <div className="text-[10px] font-mono text-gray-600">
                                {photo.faces.length} face{photo.faces.length !== 1 ? 's' : ''} detected
                                {photo.faces.length === 1 && ' · auto-selected'}
                              </div>
                            </div>
                            {selected && !isOutlier && (
                              <CheckCircle2 size={14} className="text-emerald-400 flex-shrink-0" />
                            )}
                            {isOutlier && (
                              <AlertTriangle size={14} className="text-amber-400 flex-shrink-0" />
                            )}
                          </div>

                          {/* Face candidates — clickable */}
                          <div className="flex flex-wrap gap-2 mt-2">
                            {photo.faces.map((face) => {
                              const isSelected = selected === face.face_id
                              return (
                                <button
                                  key={face.face_id}
                                  onClick={() => selectFace(photo.photo_index, face.face_id)}
                                  className={`relative flex flex-col items-center gap-1 p-1 rounded border transition-all ${
                                    isSelected ? 'border-cyan-500/60 bg-cyan-500/10 ring-2 ring-cyan-500/30' :
                                    'border-gray-700 hover:border-gray-600 bg-gray-800/40'
                                  }`}
                                  style={{ width: '80px' }}
                                >
                                  <div className="w-full aspect-square rounded bg-gray-900 overflow-hidden">
                                    {face.thumbnail_b64 ? (
                                      <img
                                        src={`data:image/jpeg;base64,${face.thumbnail_b64}`}
                                        alt=""
                                        className="w-full h-full object-cover"
                                      />
                                    ) : null}
                                  </div>
                                  <div className={`text-[8px] font-mono px-1 py-0.5 rounded border w-full text-center truncate ${
                                    POSE_COLORS[face.pose_bucket] ?? POSE_COLORS.unknown
                                  }`}>
                                    {POSE_LABELS[face.pose_bucket] ?? face.pose_bucket}
                                  </div>
                                  {isSelected && (
                                    <div className="absolute top-1 right-1 w-4 h-4 rounded-full bg-cyan-500 flex items-center justify-center">
                                      <Check size={10} className="text-gray-900" strokeWidth={3} />
                                    </div>
                                  )}
                                </button>
                              )
                            })}
                          </div>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>

              {/* Summary */}
              <div className="pt-3 mt-4 border-t border-gray-800">
                <p className="text-[11px] text-gray-500 font-mono">
                  {photosWithSelection().length} of {detectionResults.total_photos} photo{detectionResults.total_photos !== 1 ? 's' : ''} have a face selected
                </p>
              </div>

              <div className="flex justify-between pt-2">
                <button onClick={() => { setDetectionResults(null); setFaceSelections({}); setStep('photos') }} className="flex items-center gap-2 px-4 py-2.5 border border-gray-700 text-gray-500 rounded text-sm hover:bg-gray-800 hover:text-gray-300 transition-colors">
                  <ArrowLeft size={14} /> Back
                </button>
                <button
                  onClick={() => void runEnrollment(outlierApproved)}
                  disabled={!canSubmitSelections()}
                  className="flex items-center gap-2 px-5 py-2.5 bg-cyan-500/15 border border-cyan-500/40 text-cyan-400 rounded text-sm font-medium hover:bg-cyan-500/25 transition-colors disabled:opacity-30 disabled:pointer-events-none"
                >
                  Create Case & Enroll <ArrowRight size={14} />
                </button>
              </div>
            </StepPanel>
          )}

          {step === 'processing' && (
            <StepPanel key="processing" title="Creating Case" subtitle={caseRef}>
              <div className="mb-5 text-sm text-gray-500">
                Enrolling <span className="font-semibold text-gray-300">{form.name || 'subject'}</span> — {photosWithSelection().length} face{photosWithSelection().length !== 1 ? 's' : ''} selected
              </div>
              <div className="space-y-3">
                {procSteps.map((s, i) => (
                  <div key={i} className={`flex items-start gap-3 rounded-lg px-4 py-3 border transition-colors ${
                    s.status === 'idle' ? 'border-gray-800 bg-transparent' :
                    s.status === 'running' ? 'border-cyan-500/25 bg-cyan-500/5' :
                    s.status === 'ok' ? 'border-emerald-500/25 bg-emerald-500/5' :
                    s.status === 'warn' ? 'border-amber-500/25 bg-amber-500/5' :
                    'border-red-500/25 bg-red-500/5'
                  }`}>
                    <div className="flex-shrink-0 mt-0.5">
                      {s.status === 'idle' && <div className="w-4 h-4 rounded-full border border-gray-700" />}
                      {s.status === 'running' && <Loader2 size={16} className="text-cyan-400 animate-spin" />}
                      {s.status === 'ok' && <CheckCircle2 size={16} className="text-emerald-400" />}
                      {s.status === 'warn' && <AlertCircle size={16} className="text-amber-400" />}
                      {s.status === 'fail' && <XCircle size={16} className="text-red-400" />}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className={`text-[13px] font-medium ${
                        s.status === 'idle' ? 'text-gray-600' :
                        s.status === 'running' ? 'text-cyan-300' :
                        s.status === 'ok' ? 'text-emerald-300' :
                        s.status === 'warn' ? 'text-amber-300' : 'text-red-300'
                      }`}>{s.label}</div>
                      {s.detail && <div className="text-[11px] font-mono text-gray-500 mt-0.5 break-words">{s.detail}</div>}
                    </div>
                  </div>
                ))}
              </div>

              {enrollConflict && (
                <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mt-5 bg-red-500/8 border border-red-500/30 rounded-lg p-4">
                  <div className="flex items-start gap-3">
                    <AlertTriangle size={15} className="text-red-400 flex-shrink-0 mt-0.5" />
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-semibold text-red-300 mb-1">Active Case Conflict</div>
                      <p className="text-[11px] text-gray-300 mb-0.5">
                        <span className="text-white font-medium">{enrollConflict.person_name}</span> is already enrolled in <span className="text-white font-medium">{enrollConflict.incident_ref}</span> with a {Math.round(enrollConflict.similarity * 100)}% identity match.
                      </p>
                      <p className="text-[10px] text-gray-500 mb-3">
                        {enrollConflict.incident_title} · {enrollConflict.incident_status.toUpperCase()}
                      </p>
                      <div className="flex gap-2 mb-4">
                        <button onClick={() => navigate(`/cases/${enrollConflict.incident_id}`)} className="px-3 py-1.5 bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 rounded text-[11px] font-medium hover:bg-cyan-500/20 transition-colors">View Case</button>
                      </div>
                      <div className="border-t border-red-500/20 pt-3">
                        <p className="text-[10px] text-gray-600 mb-2">To create a separate case anyway, type <span className="text-gray-400 font-mono">CREATE DUPLICATE</span> below:</p>
                        <div className="flex gap-2">
                          <input value={conflictConfirmText} onChange={e => setConflictConfirmText(e.target.value)} placeholder="CREATE DUPLICATE" className="flex-1 bg-gray-900 border border-gray-700 rounded px-3 py-1.5 text-[11px] font-mono text-gray-200 outline-none focus:border-red-500/50 placeholder-gray-700" />
                          <button disabled={conflictConfirmText !== 'CREATE DUPLICATE'} onClick={() => { setEnrollConflict(null); setConflictConfirmText(''); void runEnrollment(outlierApproved, true) }} className="px-3 py-1.5 bg-red-900/30 border border-red-700/40 text-red-400 rounded text-[11px] font-medium hover:bg-red-900/50 transition-colors disabled:opacity-30 disabled:cursor-not-allowed">Create Anyway</button>
                        </div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}

              {procSteps.some((s) => s.status === 'fail') && !enrollConflict && (
                <div className="mt-5 flex justify-center gap-3">
                  <button onClick={() => { setProcSteps(PROC_STEPS.map((s) => ({ ...s }))); isSubmittingRef.current = false; void runEnrollment(outlierApproved) }} className="px-4 py-2 bg-cyan-500/15 border border-cyan-500/30 text-cyan-400 rounded text-xs hover:bg-cyan-500/25 transition-colors">Retry</button>
                  <button onClick={() => setStep('face-picker')} className="px-4 py-2 border border-gray-700 text-gray-500 rounded text-xs hover:bg-gray-800 hover:text-gray-300 transition-colors">Back to face selection</button>
                </div>
              )}
            </StepPanel>
          )}

          {step === 'success' && (
            <StepPanel key="success" title="Case Created" subtitle={caseRef}>
              <div className="text-center py-4">
                <motion.div initial={{ scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: 'spring', stiffness: 300 }} className="w-16 h-16 bg-emerald-500/15 border border-emerald-500/40 rounded-full flex items-center justify-center mx-auto mb-5">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-emerald-400">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                </motion.div>
                <h3 className="text-lg font-semibold text-gray-100 mb-1">{caseRef} is live</h3>
                <p className="text-sm text-gray-500 mb-6">Tracking pipeline is armed. Upload a video source in Operations to begin scanning.</p>
                <div className="inline-flex items-center gap-2 px-5 py-2.5 bg-gray-800 border border-gray-700 rounded text-xs font-mono text-gray-400 mb-6">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse-dot" />
                  {form.name} · {photosWithSelection().length} embeddings · TRACKING
                </div>
                <div className="flex justify-center gap-3">
                  <button onClick={() => navigate('/cases')} className="flex items-center gap-2 px-5 py-2.5 bg-cyan-500/15 border border-cyan-500/40 text-cyan-400 rounded text-sm font-medium hover:bg-cyan-500/25 transition-colors">
                    View All Cases <ExternalLink size={13} />
                  </button>
                  <button onClick={() => navigate('/operations')} className="px-5 py-2.5 border border-gray-700 text-gray-400 rounded text-sm hover:bg-gray-800 transition-colors">Go to Operations</button>
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
    <motion.div initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -12 }} transition={{ duration: 0.2 }}>
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
      <label className="block text-xs font-mono text-gray-500 tracking-wider mb-1.5">{label.toUpperCase()}</label>
      {children}
    </div>
  )
}

function StepProgress({ current, total }: { current: number; total: number }) {
  return (
    <div className="flex items-center gap-2">
      {Array.from({ length: total }).map((_, i) => (
        <div key={i} className={`h-1 flex-1 rounded-full transition-all duration-300 ${i < current ? 'bg-cyan-500' : i === current ? 'bg-cyan-500/50' : 'bg-gray-800'}`} />
      ))}
    </div>
  )
}

function NextBtn({ onClick, disabled }: { onClick: () => void; disabled?: boolean }) {
  return (
    <button onClick={onClick} disabled={disabled} className="flex items-center gap-2 px-5 py-2.5 bg-cyan-500/15 border border-cyan-500/40 text-cyan-400 rounded text-sm font-medium hover:bg-cyan-500/25 transition-colors disabled:opacity-30 disabled:pointer-events-none">
      Continue <ArrowRight size={14} />
    </button>
  )
}

function BackBtn({ onClick }: { onClick: () => void }) {
  return (
    <button onClick={onClick} className="flex items-center gap-2 px-4 py-2.5 border border-gray-700 text-gray-500 rounded text-sm hover:bg-gray-800 hover:text-gray-300 transition-colors">
      <ArrowLeft size={14} /> Back
    </button>
  )
}

const inputCls = 'w-full bg-gray-950 border border-gray-700 focus:border-cyan-600/60 rounded px-3 py-2.5 text-sm text-gray-100 outline-none transition-colors placeholder-gray-700'
