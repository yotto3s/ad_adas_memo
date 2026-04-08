/**
 * @file lane.ts
 * @description Lane model.
 *
 * A Lane represents a drivable corridor bounded by two Boundaries.
 * Lanes do not own boundary point data; they reference shared Boundaries by ID.
 */

import type { BoundaryId } from "./boundary";

/** Unique identifier for a lane. */
export type LaneId = string & { readonly __brand: unique symbol };

/**
 * A single lane (driving corridor).
 *
 * The lane's geometry is fully determined by its left and right boundaries.
 * Additional lane-level attributes (speed limit, surface type, etc.) are
 * stored in the schema-driven `attributes` record.
 */
export interface Lane {
  readonly id: LaneId;

  /** ID of the left boundary (in the lane's travel direction). */
  readonly leftBoundaryId: BoundaryId;

  /** ID of the right boundary (in the lane's travel direction). */
  readonly rightBoundaryId: BoundaryId;

  /**
   * User-defined lane-level attributes.
   * Keys correspond to attribute names in the lane schema.
   *
   * @example
   * {
   *   speed_limit: 60,
   *   road_surface: "asphalt",
   * }
   */
  readonly attributes: Readonly<Record<string, string | number | boolean>>;
}
