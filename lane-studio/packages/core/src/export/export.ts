/**
 * @file export.ts
 * @description Exporter interface and configuration types.
 *
 * The export pipeline:
 * 1. Take a Scenario (map + trajectory)
 * 2. Interpolate trajectory keyframes into dense frames
 * 3. For each frame, transform map entities into vehicle-local coordinates
 * 4. Optionally filter by sensor range
 * 5. Serialize to the target format
 *
 * The default exporter preserves the internal data structure as-is
 * (shared boundary references, topology, schema-driven attributes).
 * Custom exporters can reshape the data (expand shared refs, convert
 * to Lanelet2/CommonRoad, etc.).
 */

import type { LaneMap } from "../model/map";
import type { Scenario } from "../model/scenario";
import type { VehicleFrame } from "../model/trajectory";
import type { TopologyMap } from "../model/topology";
import type { BoundaryId } from "../model/boundary";
import type { LaneId } from "../model/lane";
import type { Point2D } from "../model/geometry";

// ---------------------------------------------------------------------------
// Export configuration
// ---------------------------------------------------------------------------

export interface ExportConfig {
  /** Output format for the default exporter. */
  readonly format: "json" | "yaml";

  /**
   * Coordinate system convention for vehicle-local coordinates.
   * This is stored as metadata in the output; the actual transform
   * is handled by the transform module.
   *
   * Common conventions:
   * - "front_x_left_y"  : +X forward, +Y left  (ISO 8855)
   * - "front_y_right_x" : +Y forward, +X right  (image coordinates)
   *
   * Users can define custom strings; the transform module must support them.
   */
  readonly coordinateSystem: string;

  /** Interpolation frame rate in Hz. */
  readonly frameRateHz: number;

  /**
   * Maximum distance (meters) from the vehicle to include entities.
   * Boundaries / lanes beyond this range are excluded from frame output.
   * Set to `Infinity` to include everything.
   */
  readonly sensorRangeM: number;

  /**
   * Whether to include topology references to lanes that are outside
   * the sensor range in a given frame.
   *
   * - true:  keep all topology references (consumer must handle missing lanes)
   * - false: remove references to lanes not present in the frame
   */
  readonly includeOutOfRangeTopology: boolean;
}

// ---------------------------------------------------------------------------
// Export output types (frame-level)
// ---------------------------------------------------------------------------

/** A boundary with points transformed to vehicle-local coordinates. */
export interface ExportedBoundary {
  readonly id: BoundaryId;
  readonly points: readonly {
    readonly position: Point2D;
    readonly attributes: Readonly<Record<string, string | number | boolean>>;
  }[];
}

/** A lane as it appears in a single exported frame. */
export interface ExportedLane {
  readonly id: LaneId;
  readonly leftBoundaryId: BoundaryId;
  readonly rightBoundaryId: BoundaryId;
  readonly attributes: Readonly<Record<string, string | number | boolean>>;
}

/** A single exported frame: vehicle state + visible map entities. */
export interface ExportedFrame {
  readonly timestamp: number;
  readonly ego: {
    /** Pose in map coordinates (for reference / debugging). */
    readonly mapPose: { readonly x: number; readonly y: number; readonly heading: number };
    readonly speed: number;
  };
  readonly boundaries: readonly ExportedBoundary[];
  readonly lanes: readonly ExportedLane[];
  readonly topology: TopologyMap;
}

/** Complete export output. */
export interface ExportResult {
  /** Static map data (same structure as internal model). */
  readonly map: LaneMap;
  /** Per-frame vehicle-local data. */
  readonly frames: readonly ExportedFrame[];
  /** The config used for this export. */
  readonly config: ExportConfig;
  /** Schema ID used. */
  readonly schemaId: string;
}

// ---------------------------------------------------------------------------
// Exporter interface
// ---------------------------------------------------------------------------

/**
 * An exporter converts a Scenario into a serialized output string.
 *
 * The default exporter serializes `ExportResult` directly to JSON/YAML.
 * Custom exporters can reshape the data arbitrarily.
 */
export interface Exporter {
  /** Unique identifier for this exporter. */
  readonly id: string;
  /** Human-readable name shown in the export dialog. */
  readonly name: string;
  /** Brief description. */
  readonly description: string;

  /**
   * Export the scenario.
   *
   * @param scenario - The full scenario (map + trajectory).
   * @param frames   - Pre-interpolated vehicle frames.
   * @param config   - Export configuration.
   * @returns Serialized output string (JSON, YAML, XML, etc.).
   */
  export(
    scenario: Scenario,
    frames: readonly VehicleFrame[],
    config: ExportConfig,
  ): string;
}
