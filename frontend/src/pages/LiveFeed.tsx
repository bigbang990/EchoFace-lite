import { X, Radio } from 'lucide-react'

export default function LiveFeed() {
  return (
    <div className="flex flex-col h-screen bg-gray-950 text-gray-300 font-mono select-none">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3.5 border-b border-gray-800 flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="text-sm font-semibold text-cyan-400 tracking-wide">ECHOFACE</div>
          <span className="text-gray-700">·</span>
          <div className="text-sm text-gray-400">Live Tracking Feed</div>
        </div>
        <button
          onClick={() => window.close()}
          className="text-gray-600 hover:text-gray-300 transition-colors"
          title="Close window"
        >
          <X size={16} />
        </button>
      </div>

      {/* Body */}
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="w-16 h-16 rounded-full bg-gray-900 border border-gray-800 flex items-center justify-center mx-auto">
            <Radio size={28} className="text-gray-700" />
          </div>
          <div className="text-gray-500 text-sm">No active feed</div>
          <div className="text-gray-700 text-xs leading-relaxed">
            Start a tracking session from Operations<br />to begin live monitoring
          </div>
        </div>
      </div>

      {/* Footer status bar */}
      <div className="px-6 py-3 border-t border-gray-800 flex items-center gap-6 flex-shrink-0">
        <div className="flex items-center gap-1.5 text-[10px]">
          <span className="w-1.5 h-1.5 rounded-full bg-gray-700" />
          <span className="text-gray-700">No active session</span>
        </div>
        <div className="text-[10px] text-gray-700">FPS: --</div>
        <div className="text-[10px] text-gray-700">Faces: --</div>
      </div>
    </div>
  )
}
