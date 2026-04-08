/**
 * @file schema.ts
 * @description Schema definition for user-configurable attributes.
 *
 * The schema drives:
 * - Which attributes appear on boundary points, lanes, and keyframes
 * - The property panel UI (dynamically generated from the schema)
 * - Validation of attribute values
 * - Visual mapping in the editor (e.g., line_type → color/dash pattern)
 */

// ---------------------------------------------------------------------------
// Attribute type definitions
// ---------------------------------------------------------------------------

interface BaseAttributeDef {
  /** Display label in the property panel. */
  readonly label: string;
  /** Optional tooltip / help text. */
  readonly description?: string;
}

/**
 * Enumeration attribute: the user picks from a fixed set of string values.
 *
 * @example
 * {
 *   type: "enum",
 *   label: "Line Type",
 *   values: ["solid", "dashed", "double_solid"],
 *   default: "solid",
 *   display: {
 *     solid:        { color: "#FFFFFF", dash: null },
 *     dashed:       { color: "#FFFFFF", dash: [8, 6] },
 *     double_solid: { color: "#FFAA00", dash: null },
 *   },
 * }
 */
export interface EnumAttributeDef extends BaseAttributeDef {
  readonly type: "enum";
  readonly values: readonly string[];
  readonly default: string;
  /**
   * Optional per-value display hints for the editor canvas.
   * Keys must be a subset of `values`.
   */
  readonly display?: Readonly<
    Record<
      string,
      {
        /** Stroke color (CSS). */
        readonly color?: string;
        /** Dash array (SVG/Canvas style), or null for solid. */
        readonly dash?: readonly number[] | null;
        /** Stroke width override in pixels. */
        readonly strokeWidth?: number;
      }
    >
  >;
}

/** Numeric attribute with optional range constraints. */
export interface NumberAttributeDef extends BaseAttributeDef {
  readonly type: "number";
  readonly default: number;
  readonly min?: number;
  readonly max?: number;
  /** Step size for the UI slider / spinner. */
  readonly step?: number;
  /** Unit label shown next to the input (e.g., "m", "km/h"). */
  readonly unit?: string;
}

/** Boolean (toggle) attribute. */
export interface BooleanAttributeDef extends BaseAttributeDef {
  readonly type: "boolean";
  readonly default: boolean;
}

/** Union of all attribute definition types. */
export type AttributeDef =
  | EnumAttributeDef
  | NumberAttributeDef
  | BooleanAttributeDef;

// ---------------------------------------------------------------------------
// Schema definition
// ---------------------------------------------------------------------------

/**
 * Complete schema configuration.
 *
 * Loaded from a YAML/JSON config file or provided programmatically.
 * The editor reads this at startup and dynamically builds the UI.
 */
export interface SchemaDefinition {
  /** Schema identifier (referenced by `ScenarioMeta.schemaId`). */
  readonly id: string;

  /** Human-readable name. */
  readonly name: string;

  /**
   * Attributes attached to each BoundaryPoint.
   * Keys become the attribute names in `BoundaryPoint.attributes`.
   */
  readonly boundaryPointAttributes: Readonly<Record<string, AttributeDef>>;

  /**
   * Attributes attached to each Lane.
   * Keys become the attribute names in `Lane.attributes`.
   */
  readonly laneAttributes: Readonly<Record<string, AttributeDef>>;

  /**
   * Attributes attached to each VehicleKeyframe.
   * Keys become the attribute names in `VehicleKeyframe.attributes`.
   */
  readonly keyframeAttributes: Readonly<Record<string, AttributeDef>>;

  /**
   * Available topology relation types.
   * Each string becomes a key in `LaneTopologyEntry`.
   *
   * @default ["predecessor", "successor", "left_adjacent", "right_adjacent"]
   */
  readonly topologyRelations: readonly string[];
}
