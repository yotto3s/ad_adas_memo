/**
 * @file map.ts
 * @description Top-level map model aggregating all static entities.
 */

import type { Boundary, BoundaryId } from "./boundary";
import type { Lane, LaneId } from "./lane";
import type { TopologyMap } from "./topology";

/**
 * The complete map: all boundaries, lanes, and their topology.
 *
 * This is the single source of truth for the static road structure.
 * The editor maintains a mutable version of this; export serializes
 * a snapshot.
 */
export interface LaneMap {
  /** All boundaries keyed by ID for O(1) lookup. */
  readonly boundaries: Readonly<Record<BoundaryId, Boundary>>;

  /** All lanes keyed by ID. */
  readonly lanes: Readonly<Record<LaneId, Lane>>;

  /** Connectivity relationships between lanes. */
  readonly topology: TopologyMap;
}
