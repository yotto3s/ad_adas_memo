/**
 * @file index.ts
 * @description Public API for @lane-studio/core
 */

// Geometry primitives
export type { Point2D, Pose2D, AffineTransform2D } from "./model/geometry";

// Boundary
export type {
  BoundaryId,
  BoundaryPoint,
  BoundaryJoint,
  Boundary,
} from "./model/boundary";

// Lane
export type { LaneId, Lane } from "./model/lane";

// Topology
export type { LaneTopologyEntry, TopologyMap } from "./model/topology";

// Trajectory
export type {
  KeyframeId,
  VehicleKeyframe,
  VehicleFrame,
  Trajectory,
} from "./model/trajectory";

// Map & Scenario
export type { LaneMap } from "./model/map";
export type { ScenarioMeta, Scenario } from "./model/scenario";

// Schema
export type {
  EnumAttributeDef,
  NumberAttributeDef,
  BooleanAttributeDef,
  AttributeDef,
  SchemaDefinition,
} from "./schema/schema";
export { DEFAULT_SCHEMA } from "./schema/default-schema";

// Export
export type {
  ExportConfig,
  ExportedBoundary,
  ExportedLane,
  ExportedFrame,
  ExportResult,
  Exporter,
} from "./export/export";
