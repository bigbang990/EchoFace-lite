import { Routes, Route, Navigate } from 'react-router-dom'
import { useAppStore } from './store/appStore'
import AccessGate from './components/AccessGate'
import Layout from './components/Layout'
import Overview from './pages/Overview'
import CreateCase from './pages/CreateCase'
import Operations from './pages/Operations'
import CaseList from './pages/CaseList'
import CaseWorkspace from './pages/CaseWorkspace'
import SystemHealth from './pages/SystemHealth'

export default function App() {
  const { accessMode, setAccessMode } = useAppStore()

  if (!accessMode) {
    return <AccessGate onAccess={setAccessMode} />
  }

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/cases" element={<CaseList />} />
        <Route path="/cases/new" element={<CreateCase />} />
        <Route path="/cases/:id" element={<CaseWorkspace />} />
        <Route path="/operations" element={<Operations />} />
        <Route
          path="/system-health"
          element={
            accessMode === 'ADMIN' ? <SystemHealth /> : <Navigate to="/" replace />
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  )
}
