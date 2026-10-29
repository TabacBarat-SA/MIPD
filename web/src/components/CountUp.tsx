import { animate } from 'motion'
import { useEffect, useRef } from 'react'

/** Anima un numero desde su valor anterior. Con movimiento reducido salta directamente al valor. */
export default function CountUp({ value, format, duration = 1.2 }: { value: number; format: (n: number) => string; duration?: number }) {
  const el = useRef<HTMLSpanElement>(null)
  const prev = useRef(0)
  useEffect(() => {
    const node = el.current
    if (!node) return
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
      node.textContent = format(value)
      prev.current = value
      return
    }
    const controls = animate(prev.current, value, {
      duration,
      ease: [0.16, 1, 0.3, 1],
      onUpdate: v => { node.textContent = format(v) },
    })
    prev.current = value
    return () => controls.stop()
  }, [value, format, duration])
  return <span ref={el} className="num">{format(value)}</span>
}
