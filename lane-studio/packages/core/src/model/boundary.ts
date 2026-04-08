/**
 * @file boundary.ts
 * @description Boundary (lane marking line) model.
 *
 * A Boundary is the fundamental drawable entity in the editor.
 * It represents a single lane marking line (e.g., a white solid line).
 * Lanes reference boundaries by ID; adjacent lanes share the same boundary.
 */

import type { Point2D } from "./geometry";

/** Unique identifier for a boundary. */
export type BoundaryId = string & { readonly __brand: unique symbol };

/**
 * A single point on a boundary line, carrying both position and
 * user-defined attributes (line type, width, etc.).
 *
 * The `attributes` record is schema-driven: its keys and value types
 * are defined by the user's configuration (see `SchemaDefinition`).
 * The editor UI dynamically generates controls based on the schema.
 */
export interface BoundaryPoint {
  readonly position: Point2D;
  /**
   * User-defined attributes for this point.
   * Keys correspond to attribute names in the boundary_point schema.
   * Values are `string | number | boolean` depending on the attribute type.
   *
   * @example
   * {
   *   line_type: "solid",
   *   width_m: 0.15,
   *   reflectance: 0.8,
   * }
   */
  readonly attributes: Readonly<Record<string, string | number | boolean>>;
}

/**
 * Connection point between boundaries.
 * When a boundary's start or end connects to another boundary,
 * a Joint records this relationship.
 */
export interface BoundaryJoint {
  /** The boundary this joint connects to. */
  readonly boundaryId: BoundaryId;
  /**
   * Which end of the target boundary is connected.
   * - "start": this joint connects to the target's first point.
   * - "end":   this joint connects to the target's last point.
   */
  readonly end: "start" | "end";
}

/**
 * A boundary line: an ordered sequence of points forming a polyline.
 *
 * Boundaries are the **shared** entities between adjacent lanes.
 * For example, in a 3-lane road there are 4 boundaries (outer-left,
 * lane-divider-1, lane-divider-2, outer-right), and adjacent lanes
 * reference the same divider boundary.
 */
export interface Boundary {
  readonly id: BoundaryId;
  readonly points: readonly BoundaryPoint[];

  /**
   * Optional joint at the start of this boundary.
   * Non-null when this boundary begins where another boundary ends
   * (e.g., after a merge or at a road junction).
   */
  readonly startJoint: BoundaryJoint | null;

  /**
   * Optional joint at the end of this boundary.
   * Non-null when this boundary ends where another boundary begins
   * (e.g., before a split or at a road junction).
   */
  readonly endJoint: BoundaryJoint | null;
}
