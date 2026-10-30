/** Curvas y resortes compartidos: una sola sensacion de movimiento en toda la web. */
export const EASE = [0.16, 1, 0.3, 1] as const
export const SPRING_SNAPPY = { type: 'spring', stiffness: 500, damping: 38 } as const
export const SPRING_SOFT = { type: 'spring', stiffness: 170, damping: 24, mass: 0.8 } as const
export const prefersReducedMotion = () => typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches
