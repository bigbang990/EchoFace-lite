import { create } from 'zustand'
import type { AccessMode } from '../types'

export interface BackendEntry {
  name: string
  url: string
  type: 'local' | 'remote' | 'custom'
}

export const BACKENDS: BackendEntry[] = [
  { name: 'Local CPU', url: 'http://127.0.0.1:8000/api/v1', type: 'local' },
  { name: 'Colab GPU', url: 'https://4864-34-178-169-176.ngrok-free.app/api/v1', type: 'remote' },
]

export const INC_DEFAULT_URL = 'https://f525-34-178-169-176.ngrok-free.app/api/v1'

interface AppState {
  accessMode: AccessMode | null
  activeCaseId: string | null
  activeJobId: string | null
  backendName: string
  backendUrl: string
  incUrl: string
  setAccessMode: (mode: AccessMode) => void
  setActiveCaseId: (id: string) => void
  setActiveJobId: (id: string | null) => void
  setBackend: (name: string, url: string) => void
  setIncUrl: (url: string) => void
  logout: () => void
}

export const useAppStore = create<AppState>((set) => ({
  accessMode: null,
  activeCaseId: null,
  activeJobId: null,
  backendName: BACKENDS[0].name,
  backendUrl: BACKENDS[0].url,
  incUrl: INC_DEFAULT_URL,
  setAccessMode: (mode) => set({ accessMode: mode }),
  setActiveCaseId: (id) => set({ activeCaseId: id }),
  setActiveJobId: (id) => set({ activeJobId: id }),
  setBackend: (name, url) => set({ backendName: name, backendUrl: url }),
  setIncUrl: (url) => set({ incUrl: url }),
  logout: () => set({ accessMode: null, activeCaseId: null, activeJobId: null }),
}))
