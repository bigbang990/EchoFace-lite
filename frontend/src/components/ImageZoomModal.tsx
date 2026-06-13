import { useCallback, useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, ChevronLeft, ChevronRight } from 'lucide-react'

interface ImageItem {
  src: string
  alt: string
}

interface ImageZoomModalProps {
  images: ImageItem[]
  startIndex?: number
  isOpen: boolean
  onClose: () => void
}

export default function ImageZoomModal({
  images,
  startIndex = 0,
  isOpen,
  onClose,
}: ImageZoomModalProps) {
  const [index, setIndex] = useState(startIndex)
  const [zoomScale, setZoomScale] = useState(1)
  const [translate, setTranslate] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const dragStart = useRef({ x: 0, y: 0, tx: 0, ty: 0 })
  const didDrag = useRef(false)
  const overlayRef = useRef<HTMLDivElement>(null)

  const resetView = useCallback(() => {
    setZoomScale(1)
    setTranslate({ x: 0, y: 0 })
  }, [])

  // Sync index + reset view when modal opens
  useEffect(() => {
    if (isOpen) {
      setIndex(startIndex)
      resetView()
    }
  }, [isOpen, startIndex, resetView])

  // Reset view on image change
  useEffect(() => {
    resetView()
  }, [index, resetView])

  const goNext = useCallback(() => {
    setIndex(i => (i + 1) % images.length)
  }, [images.length])

  const goPrev = useCallback(() => {
    setIndex(i => (i - 1 + images.length) % images.length)
  }, [images.length])

  // Keyboard: Escape / ArrowLeft / ArrowRight
  useEffect(() => {
    if (!isOpen) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
      else if (e.key === 'ArrowRight') goNext()
      else if (e.key === 'ArrowLeft') goPrev()
    }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [isOpen, onClose, goNext, goPrev])

  // Body scroll lock
  useEffect(() => {
    if (isOpen) document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = '' }
  }, [isOpen])

  // Non-passive wheel for zoom (required to call preventDefault)
  useEffect(() => {
    const el = overlayRef.current
    if (!el || !isOpen) return
    const handler = (e: WheelEvent) => {
      e.preventDefault()
      setZoomScale(prev => {
        const next = Math.min(4, Math.max(1, prev - e.deltaY * 0.005))
        if (next <= 1) setTranslate({ x: 0, y: 0 })
        return next
      })
    }
    el.addEventListener('wheel', handler, { passive: false })
    return () => el.removeEventListener('wheel', handler)
  }, [isOpen])

  const handleMouseDown = (e: React.MouseEvent) => {
    didDrag.current = false
    if (zoomScale <= 1) return
    setIsDragging(true)
    dragStart.current = { x: e.clientX, y: e.clientY, tx: translate.x, ty: translate.y }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return
    const dx = e.clientX - dragStart.current.x
    const dy = e.clientY - dragStart.current.y
    if (Math.abs(dx) > 2 || Math.abs(dy) > 2) didDrag.current = true
    setTranslate({ x: dragStart.current.tx + dx, y: dragStart.current.ty + dy })
  }

  const handleMouseUp = () => setIsDragging(false)

  // Only close on backdrop click, not after a drag
  const handleOverlayClick = () => {
    if (!didDrag.current) onClose()
  }

  const current = images[index] ?? images[0]
  const hasMultiple = images.length > 1

  if (!current) return null

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          ref={overlayRef}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={handleOverlayClick}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          className="fixed inset-0 z-50 bg-black/85 flex items-center justify-center select-none"
          style={{ cursor: zoomScale > 1 ? (isDragging ? 'grabbing' : 'grab') : 'default' }}
        >
          {/* Close */}
          <button
            onClick={(e) => { e.stopPropagation(); onClose() }}
            className="absolute top-4 right-4 z-10 text-gray-400 hover:text-white transition-colors p-1"
          >
            <X size={24} />
          </button>

          {/* Counter */}
          {hasMultiple && (
            <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10 text-xs font-mono text-gray-400 bg-black/60 px-3 py-1 rounded pointer-events-none">
              {index + 1} / {images.length}
            </div>
          )}

          {/* Prev */}
          {hasMultiple && (
            <button
              onClick={(e) => { e.stopPropagation(); goPrev() }}
              className="absolute left-4 z-10 text-gray-400 hover:text-white transition-colors p-2 bg-black/40 hover:bg-black/60 rounded-full"
            >
              <ChevronLeft size={24} />
            </button>
          )}

          {/* Image with enter animation per-image */}
          <AnimatePresence mode="wait">
            <motion.div
              key={current.src}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              style={{
                transform: `translate(${translate.x}px, ${translate.y}px) scale(${zoomScale})`,
                transformOrigin: 'center',
              }}
            >
              <img
                src={current.src}
                alt={current.alt}
                draggable={false}
                className="max-w-[90vw] max-h-[90vh] object-contain rounded"
                style={{ pointerEvents: 'none', display: 'block' }}
              />
            </motion.div>
          </AnimatePresence>

          {/* Next */}
          {hasMultiple && (
            <button
              onClick={(e) => { e.stopPropagation(); goNext() }}
              className="absolute right-4 z-10 text-gray-400 hover:text-white transition-colors p-2 bg-black/40 hover:bg-black/60 rounded-full"
            >
              <ChevronRight size={24} />
            </button>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  )
}
