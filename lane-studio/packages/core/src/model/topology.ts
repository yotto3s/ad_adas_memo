/**
 * @file topology.ts
 * @description Topology model for lane connectivity.
 *
 * Topology is stored as a separate structure keyed by LaneId,
 * rather than embedded inside each Lane, to keep the Lane type lean
 * and to allow topology to be independently edited/exported.
 *
 * The set of relation types (predecessor, successor, left_adjacent, ...)
 * is user-configurable via the schema. The default configuration provides
 * the four standard relations, but users can add domain-specific ones
 * (e.g., "merges_into", "branches_from").
 */

import type { LaneId } from "./lane";

/**
 * Topology entry for a single lane.
 *
 * Each key is a relation name (defined in the schema's `topology_relations`),
 * and the value is an array of connected LaneIds.
 *
 * @example
 * {
 *   predecessor: ["lane_000"],
 *   successor: ["lane_002", "lane_003"],   // branch
 *   left_adjacent: ["lane_010"],
 *   right_adjacent: [],
 * }
 */
export type LaneTopologyEntry = Readonly<Record<string, readonly LaneId[]>>;

/**
 * Complete topology map: LaneId → relations.
 *
 * @example
 * {
 *   "lane_001": {
 *     predecessor: [],
 *     successor: ["lane_002"],
 *     left_adjacent: [],
 *     right_adjacent: ["lane_010"],
 *   },
 *   "lane_010": {
 *     predecessor: [],
 *     successor: ["lane_011"],
 *     left_adjacent: ["lane_001"],
 *     right_adjacent: [],
 *   },
 * }
 */
export type TopologyMap = Readonly<Record<LaneId, LaneTopologyEntry>>;
