import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  PlusCircle,
  Video,
  FolderOpen,
  Cpu,
  ShieldOff,
  LogOut,
  ServerCog,
} from 'lucide-react'
import { useAppStore } from '../store/appStore'
import BackendPanel from './BackendPanel'

const demoNavItems = [
  { to: '/',          icon: LayoutDashboard, label: 'Overview',     end: true  },
  { to: '/cases/new', icon: PlusCircle,      label: 'Create Case',  end: false },
  { to: '/operations',icon: Video,           label: 'Operations',   end: false },
  { to: '/cases',     icon: FolderOpen,      label: 'Cases',        end: true  },
]

const adminNavItems = [
  { to: '/system-health', icon: Cpu, label: 'System Health', end: false },
]

interface Props {
  children: React.ReactNode
}

export default function Layout({ children }: Props) {
  const { accessMode, backendName, logout } = useAppStore()
  const [backendOpen, setBackendOpen] = useState(false)

  const isMock  = accessMode === 'MOCK'
  const isAdmin = accessMode === 'ADMIN'

  const modeCfg = {
    MOCK:  { label: 'MOCK',  color: 'text-violet-400', dot: 'bg-violet-400' },
    DEMO:  { label: 'DEMO',  color: 'text-cyan-400',   dot: 'bg-cyan-400'   },
    ADMIN: { label: 'ADMIN', color: 'text-amber-400',  dot: 'bg-amber-400'  },
  }[accessMode ?? 'DEMO']

  return (
    <div className="flex h-screen bg-gray-950 overflow-hidden">
      <aside className="w-56 flex-shrink-0 bg-gray-950 border-r border-gray-800/80 flex flex-col">
        <div className="px-5 py-5 border-b border-gray-800/80">
          <div className="flex items-center gap-2.5">
            <img
              src="/favicon.svg"
              alt="EchoFace"
              className="w-6 h-6 object-contain flex-shrink-0"
              style={{ filter: 'brightness(0) invert(1)' }}
            />
            <div>
              <div className="text-sm font-mono font-semibold text-cyan-400 tracking-wide leading-none">
                ECHOFACE
              </div>
              <div className="text-[9px] font-mono text-gray-700 tracking-widest mt-0.5">
                LITE v1.0.1
              </div>
            </div>
          </div>
        </div>

        <nav className="flex-1 overflow-y-auto py-3">
          <NavSection label="NAVIGATION" items={demoNavItems} />

          {(isAdmin || isMock) && (
            <NavSection label="ADMIN" items={adminNavItems} />
          )}

          {(isAdmin || isMock) && (
            <div className="px-3 mt-1">
              <div className="flex items-center gap-3 px-3 py-2 rounded text-[13px] text-gray-700 select-none cursor-not-allowed">
                <ShieldOff size={14} className="flex-shrink-0" />
                <div className="leading-tight">
                  <div>Administration</div>
                  <div className="text-[9px] text-gray-800 mt-0.5">Available in v2.0</div>
                </div>
              </div>
            </div>
          )}
        </nav>

        <div className="px-4 py-4 border-t border-gray-800/80 space-y-3">
          {!isMock && (
            <button
              onClick={() => setBackendOpen(true)}
              className="w-full flex items-center gap-2 px-3 py-2 rounded border border-gray-800 hover:border-gray-700 text-gray-500 hover:text-gray-300 transition-all duration-150 text-xs"
            >
              <ServerCog size={13} />
              <div className="flex-1 text-left leading-tight">
                <div className="text-[10px] font-mono text-gray-600">BACKEND</div>
                <div className="text-[11px] truncate">{backendName}</div>
              </div>
            </button>
          )}

          <div className="flex items-center justify-between">
            <div>
              <div className={`text-[9px] font-mono font-semibold tracking-[0.2em] ${modeCfg.color}`}>
                {modeCfg.label} MODE
              </div>
              <button
                onClick={logout}
                className="mt-1 flex items-center gap-1 text-[10px] font-mono text-gray-600 hover:text-gray-400 transition-colors"
              >
                <LogOut size={9} />
                exit session
              </button>
            </div>
            <div className={`w-2 h-2 rounded-full animate-pulse-dot ${modeCfg.dot}`} />
          </div>
        </div>
      </aside>

      <main className="flex-1 overflow-y-auto min-w-0">{children}</main>

      <BackendPanel open={backendOpen} onClose={() => setBackendOpen(false)} />
    </div>
  )
}

interface NavSectionProps {
  label: string
  items: { to: string; icon: React.ElementType; label: string; end: boolean }[]
}

function NavSection({ label, items }: NavSectionProps) {
  return (
    <div className="px-3 mb-1">
      <div className="text-[9px] font-mono text-gray-700 tracking-[0.2em] px-3 py-2">{label}</div>
      {items.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          end={item.end}
          className={({ isActive }) =>
            `flex items-center gap-3 px-3 py-2.5 rounded text-[13px] font-medium transition-all duration-150 ${
              isActive
                ? 'bg-cyan-500/10 text-cyan-400 border-l-2 border-cyan-400 pl-[10px]'
                : 'text-gray-500 hover:text-gray-200 hover:bg-gray-800/40'
            }`
          }
        >
          <item.icon size={14} />
          {item.label}
        </NavLink>
      ))}
    </div>
  )
}
