/**
 * @file geometry.ts
 * @description Fundamental geometric primitives used throughout the lane-studio model.
 */

/** 2D point in map coordinate space (meters). */
export interface Point2D {
  readonly x: number;
  readonly y: number;
}

/** 2D pose: position + orientation. */
export interface Pose2D {
  readonly x: number;
  readonly y: number;
  /** Heading angle in radians, measured counter-clockwise from the +X axis. */
  readonly heading: number;
}

/**
 * Affine transformation matrix for 2D coordinate conversion.
 *
 * Represents the transformation:
 *   x' = m00*x + m01*y + tx
 *   y' = m10*x + m11*y + ty
 */
export interface AffineTransform2D {
  readonly m00: number;
  readonly m01: number;
  readonly m10: number;
  readonly m11: number;
  readonly tx: number;
  readonly ty: number;
}
