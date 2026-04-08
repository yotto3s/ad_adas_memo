/**
 * @file scenario.ts
 * @description Scenario: the top-level project model combining map and trajectory.
 *
 * A Scenario is what gets saved/loaded as a project file,
 * and what gets fed into the export pipeline.
 */

import type { LaneMap } from "./map";
import type { Trajectory } from "./trajectory";

/**
 * Metadata about the scenario.
 */
export interface ScenarioMeta {
  /** Human-readable name. */
  readonly name: string;

  /** Optional description. */
  readonly description: string;

  /** Schema config file path or inline schema identifier used for this scenario. */
  readonly schemaId: string;

  /** Creation timestamp (ISO 8601). */
  readonly createdAt: string;

  /** Last modification timestamp (ISO 8601). */
  readonly updatedAt: string;
}

/**
 * A complete scenario: static map + vehicle trajectory + metadata.
 */
export interface Scenario {
  readonly meta: ScenarioMeta;

  /** The static road structure. */
  readonly map: LaneMap;

  /** The vehicle trajectory (keyframes). */
  readonly trajectory: Trajectory;
}
