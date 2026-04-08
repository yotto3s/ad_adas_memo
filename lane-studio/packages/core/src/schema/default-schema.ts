/**
 * @file default-schema.ts
 * @description Built-in default schema definition.
 *
 * Provides a reasonable starting point for road lane editing.
 * Users can override this with their own schema config file.
 */

import type { SchemaDefinition } from "./schema";

export const DEFAULT_SCHEMA: SchemaDefinition = {
  id: "default",
  name: "Default Road Schema",

  boundaryPointAttributes: {
    line_type: {
      type: "enum",
      label: "Line Type",
      values: [
        "solid",
        "dashed",
        "double_solid",
        "double_dashed",
        "solid_dashed", // solid on left, dashed on right
        "dashed_solid", // dashed on left, solid on right
        "none", // no visible marking (e.g., road edge)
      ],
      default: "solid",
      display: {
        solid: { color: "#FFFFFF", dash: null },
        dashed: { color: "#FFFFFF", dash: [8, 6] },
        double_solid: { color: "#FFD700", dash: null, strokeWidth: 3 },
        double_dashed: { color: "#FFD700", dash: [8, 6], strokeWidth: 3 },
        solid_dashed: { color: "#FFD700", dash: null },
        dashed_solid: { color: "#FFD700", dash: [8, 6] },
        none: { color: "#666666", dash: [2, 4] },
      },
    },
    width_m: {
      type: "number",
      label: "Line Width",
      default: 0.15,
      min: 0.01,
      max: 1.0,
      step: 0.01,
      unit: "m",
    },
    color: {
      type: "enum",
      label: "Line Color",
      values: ["white", "yellow", "blue", "orange"],
      default: "white",
    },
  },

  laneAttributes: {
    speed_limit: {
      type: "number",
      label: "Speed Limit",
      default: 60,
      min: 0,
      max: 300,
      step: 10,
      unit: "km/h",
    },
    road_surface: {
      type: "enum",
      label: "Road Surface",
      values: ["asphalt", "concrete", "gravel", "dirt"],
      default: "asphalt",
    },
  },

  keyframeAttributes: {
    // Empty by default; users can add acceleration, steering, etc.
  },

  topologyRelations: [
    "predecessor",
    "successor",
    "left_adjacent",
    "right_adjacent",
  ],
} as const satisfies SchemaDefinition;
