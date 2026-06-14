import { Routes, Route, Navigate } from 'react-router-dom'
import { useAppStore } from './store/appStore'
import AccessGate from './components/AccessGate'
import Layout from './components/Layout'
import Overview from './pages/Overview'
import CreateCase from './pages/CreateCase'
import Operations from './pages/Operations'
import CaseList from './pages/CaseList'
import CaseWorkspace from './pages/CaseWorkspace'
import AlertDetail from './pages/AlertDetail'
import SystemHealth from './pages/SystemHealth'
import Administration from './pages/Administration'
import LiveFeed from './pages/LiveFeed'

export default function App() {
  const { accessMode, setAccessMode } = useAppStore()

  return (
    <Routes>
      {/* Standalone window — no auth, no layout shell */}
      <Route path="/live-feed" element={<LiveFeed />} />

      {/* All other routes go through auth + layout */}
      <Route
        path="*"
        element={
          !accessMode ? (
            <AccessGate onAccess={setAccessMode} />
          ) : (
            <Layout>
              <Routes>
                <Route path="/" element={<Overview />} />
                <Route path="/cases" element={<CaseList />} />
                <Route path="/cases/new" element={<CreateCase />} />
                <Route path="/cases/:id" element={<CaseWorkspace />} />
                <Route path="/cases/:id/alerts/:sightingId" element={<AlertDetail />} />
                <Route path="/operations" element={<Operations />} />
                <Route
                  path="/system-health"
                  element={
                    accessMode === 'ADMIN' ? <SystemHealth /> : <Navigate to="/" replace />
                  }
                />
                <Route
                  path="/administration"
                  element={
                    accessMode === 'ADMIN' || accessMode === 'MOCK'
                      ? <Administration />
                      : <Navigate to="/" replace />
                  }
                />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </Layout>
          )
        }
      />
    </Routes>
  )
}
