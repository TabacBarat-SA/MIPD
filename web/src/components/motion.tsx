import { motion, useMotionValue, useReducedMotion, useSpring, useTransform, type HTMLMotionProps } from 'motion/react'
import { useRef, type PointerEvent, type ReactNode } from 'react'
import { EASE, SPRING_SOFT } from '../lib/motion'

/** Aparece al entrar en pantalla: fundido, subida y desenfoque que se aclara. */
export function Reveal({ children, delay = 0, y = 28, className, ...rest }: { children: ReactNode; delay?: number; y?: number; className?: string } & Omit<HTMLMotionProps<'div'>, 'children'>) {
  const reduced = useReducedMotion()
  if (reduced) return <div className={className}>{children}</div>
  return (
    <motion.div className={className} initial={{ opacity: 0, y }} whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }} transition={{ duration: 0.9, delay, ease: EASE }} {...rest}>
      {children}
    </motion.div>
  )
}

/** Titular que sube palabra a palabra desde una mascara. */
export function TextReveal({ text, className, delay = 0, as: Tag = 'h1' }: { text: string; className?: string; delay?: number; as?: 'h1' | 'h2' }) {
  const reduced = useReducedMotion()
  const words = text.split(' ')
  const M = Tag === 'h1' ? motion.h1 : motion.h2
  return (
    <M className={className} aria-label={text}>
      {words.map((w, i) => (
        <span key={i} className="word-mask" aria-hidden>
          <motion.span className="inline-block" initial={reduced ? false : { y: '110%', rotate: 4 }} animate={{ y: 0, rotate: 0 }}
            transition={{ duration: 1, delay: delay + i * 0.07, ease: EASE }}>
            {w}{i < words.length - 1 ? ' ' : ''}
          </motion.span>
        </span>
      ))}
    </M>
  )
}

/** Boton o enlace que se acerca suavemente al cursor. */
export function Magnetic({ children, strength = 0.25 }: { children: ReactNode; strength?: number }) {
  const reduced = useReducedMotion()
  const x = useMotionValue(0), y = useMotionValue(0)
  const sx = useSpring(x, SPRING_SOFT), sy = useSpring(y, SPRING_SOFT)
  const ref = useRef<HTMLDivElement>(null)
  if (reduced) return <div className="inline-block">{children}</div>
  const move = (e: PointerEvent) => {
    const r = ref.current!.getBoundingClientRect()
    x.set((e.clientX - r.left - r.width / 2) * strength); y.set((e.clientY - r.top - r.height / 2) * strength)
  }
  return (
    <motion.div ref={ref} className="inline-block" style={{ x: sx, y: sy }} onPointerMove={move} onPointerLeave={() => { x.set(0); y.set(0) }}>
      {children}
    </motion.div>
  )
}

/** Inclinacion 3D sutil segun la posicion del cursor. */
export function Tilt({ children, className = '', max = 4 }: { children: ReactNode; className?: string; max?: number }) {
  const reduced = useReducedMotion()
  const px = useMotionValue(0.5), py = useMotionValue(0.5)
  const rx = useSpring(useTransform(py, [0, 1], [max, -max]), SPRING_SOFT)
  const ry = useSpring(useTransform(px, [0, 1], [-max, max]), SPRING_SOFT)
  if (reduced) return <div className={className}>{children}</div>
  const move = (e: PointerEvent<HTMLElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    px.set((e.clientX - r.left) / r.width); py.set((e.clientY - r.top) / r.height)
  }
  return (
    <motion.div className={className} style={{ rotateX: rx, rotateY: ry, transformPerspective: 1100 }} onPointerMove={move} onPointerLeave={() => { px.set(0.5); py.set(0.5) }}>
      {children}
    </motion.div>
  )
}
