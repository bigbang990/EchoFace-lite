import { X } from 'lucide-react'

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

      {/* Body — MJPEG stream, browser decodes multipart/x-mixed-replace natively */}
      <div className="flex-1 flex items-center justify-center overflow-hidden bg-black">
        <img
          src="/api/v1/stream/live/1"
          style={{ width: '100%', height: '100%', objectFit: 'contain' }}
          alt="Live feed"
        />
      </div>
    </div>
  )
}
