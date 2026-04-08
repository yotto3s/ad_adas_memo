# Lane Studio — Design Document

> A browser-based lane map editor for generating road perception test data.
> Intended audience: Claude Code (or any developer) implementing this project.

---

## 1. Project Overview

### 1.1 What it does

Lane Studio is a web app where users:

1. **Draw road lanes** on an infinite 2D canvas (top-down view, paint-tool feel)
2. **Attach metadata** to lane boundaries (line type, width, color — all user-configurable via schema)
3. **Define lane topology** (predecessor/successor, left/right adjacent)
4. **Place vehicle keyframes** (position, heading, speed, timestamp) on the map
5. **Export** — interpolate the trajectory, transform all visible lanes into vehicle-local coordinates per frame, and output JSON/YAML

The tool is designed to be **generic and open-source**. No domain-specific assumptions are baked in; the attribute schema and export format are fully customizable.

### 1.2 Key design principles

- **Boundary as first-class entity**: Adjacent lanes share boundary references (not copies)
- **Schema-driven attributes**: All metadata (line type, width, etc.) is defined in a config file; the UI is dynamically generated from it
- **Export preserves internal structure**: The default exporter outputs shared boundary references as-is; reshaping (expanding, converting to Lanelet2, etc.) is the job of custom exporters
- **tldraw as the canvas foundation**: We get pan, zoom, selection, undo/redo, serialization for free

---

## 2. Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| Canvas engine | **tldraw SDK** (`tldraw` npm) | Infinite canvas, custom shapes/tools, undo/redo, serialization, React-native |
| UI framework | **React 18+** | tldraw requirement |
| Styling | **Tailwind CSS** | Utility-first, fast iteration |
| State management | **tldraw store** + **Zustand** (sidebar/panel state only) | tldraw manages canvas state; Zustand for UI state (selected panel tab, export dialog, etc.) |
| Build tool | **Vite** | Fast dev server, good TS support |
| Language | **TypeScript (strict)** | |
| Package manager | **pnpm** with workspaces | Monorepo for `core` + `editor` packages |

### 2.1 Package structure

```
lane-studio/
├── packages/
│   ├── core/                    # Framework-independent logic
│   │   └── src/
│   │       ├── model/           # Type definitions (already written — see Section 3)
│   │       ├── schema/          # Schema types + default schema
│   │       ├── interpolation/   # Trajectory interpolation (Catmull-Rom)
│   │       ├── transform/       # Map coord → vehicle-local coord
│   │       └── export/          # Exporter interface + default JSON/YAML exporter
│   │
│   └── editor/                  # React + tldraw app
│       └── src/
│           ├── shapes/          # Custom tldraw shape definitions
│           ├── tools/           # Custom tldraw tool definitions
│           ├── panels/          # Sidebar panels (React components)
│           ├── store/           # Zustand store for UI state
│           ├── export-ui/       # Export dialog component
│           ├── App.tsx          # Root component
│           └── main.tsx         # Entry point
│
├── configs/
│   └── default-schema.yaml     # Default attribute config (YAML version)
│
├── pnpm-workspace.yaml
├── tsconfig.json
└── vite.config.ts
```

---

## 3. Core Data Model (already implemented)

The `core/model/` types are already written. Summary of the key types:

### 3.1 Entity hierarchy

```
Scenario
├── ScenarioMeta          { name, description, schemaId, createdAt, updatedAt }
├── LaneMap
│   ├── boundaries        Record<BoundaryId, Boundary>
│   │   └── Boundary      { id, points: BoundaryPoint[], startJoint, endJoint }
│   │       └── BoundaryPoint { position: Point2D, attributes: Record<string, ...> }
│   ├── lanes             Record<LaneId, Lane>
│   │   └── Lane          { id, leftBoundaryId, rightBoundaryId, attributes }
│   └── topology          Record<LaneId, LaneTopologyEntry>
│       └── LaneTopologyEntry  Record<relationName, LaneId[]>
└── Trajectory
    └── keyframes         VehicleKeyframe[]
        └── VehicleKeyframe { id, pose: Pose2D, speed, timestamp, attributes }
```

### 3.2 Key invariants

- `Lane.leftBoundaryId` and `Lane.rightBoundaryId` always point to entries in `LaneMap.boundaries`
- Adjacent lanes MUST reference the same `BoundaryId` for their shared divider
- `BoundaryPoint.attributes` keys match the schema's `boundaryPointAttributes`
- `Lane.attributes` keys match the schema's `laneAttributes`
- `topology` keys are a subset of the schema's `topologyRelations`

### 3.3 Schema system

`SchemaDefinition` defines what attributes exist and their types (enum/number/boolean). The editor reads this and dynamically generates:
- Property panel controls (dropdowns for enum, sliders/inputs for number, toggles for boolean)
- Canvas visual mapping (enum values → colors/dash patterns via the `display` field)
- Default values for newly created entities

---

## 4. tldraw Integration Architecture

### 4.1 Custom shapes

We define three custom tldraw shapes. Each shape type has a `ShapeUtil` class that defines how it renders, how it's selected, and how it serializes.

#### 4.1.1 `BoundaryShape`

The **primary drawable** shape. Represents a single lane marking line.

```typescript
interface BoundaryShapeProps {
  // Core model data
  boundaryId: BoundaryId;
  points: BoundaryPoint[];       // positions in tldraw world coords
  startJoint: BoundaryJoint | null;
  endJoint: BoundaryJoint | null;
}
```

**Rendering** (`BoundaryShapeUtil`):
- Draws a polyline through `points`
- Stroke color and dash pattern derived from the first point's `line_type` attribute + schema's `display` mapping
- When segments have different `line_type` values, render each segment with its own style
- Control points (small circles) shown when selected, draggable for editing
- Snap to other boundary endpoints when dragging to form joints

**Interactions**:
- Click to select → shows control points + opens property panel
- Drag control points to reshape
- Drag endpoint near another boundary's endpoint → snap + create joint

#### 4.1.2 `LaneShape`

A **non-drawable** logical shape. It exists in the tldraw store to represent a lane, but its geometry is fully derived from its two boundary references.

```typescript
interface LaneShapeProps {
  laneId: LaneId;
  leftBoundaryId: BoundaryId;    // Reference to a BoundaryShape
  rightBoundaryId: BoundaryId;   // Reference to a BoundaryShape
  attributes: Record<string, string | number | boolean>;
}
```

**Rendering** (`LaneShapeUtil`):
- Draws a semi-transparent fill polygon between the left and right boundary polylines
- Color tinted by lane index or attribute (e.g., surface type)
- When selected, highlighted with a brighter fill
- Lane ID label drawn at the center

**Interactions**:
- Click on lane fill area to select → opens lane property panel
- Not directly drawable; created as a side effect of the `DrawRoadTool` or `AddAdjacentLaneTool`

#### 4.1.3 `KeyframeShape`

Represents a vehicle keyframe on the canvas.

```typescript
interface KeyframeShapeProps {
  keyframeId: KeyframeId;
  pose: Pose2D;                  // x, y, heading
  speed: number;
  timestamp: number;
  attributes: Record<string, string | number | boolean>;
}
```

**Rendering** (`KeyframeShapeUtil`):
- Triangle/arrow icon indicating position and heading direction
- Size proportional to speed (visual hint)
- Label showing timestamp
- When multiple keyframes exist, draw interpolated trajectory curve between them (Catmull-Rom preview)

**Interactions**:
- Click to select → opens keyframe property panel
- Drag to reposition
- Rotate handle to change heading

### 4.2 Custom tools

#### 4.2.1 `DrawRoadTool`

The main drawing tool. Creates a multi-lane road from a centerline sketch.

**Behavior**:
1. User clicks points on the canvas to define a centerline polyline
2. A "ghost" preview shows the road outline based on current lane count + width settings
3. Double-click (or press Enter) to commit
4. On commit:
   a. Resample the centerline at regular intervals (e.g., 5m)
   b. Generate N+1 boundary polylines (offset from center) where N = lane count
   c. Create N `BoundaryShape`s and N `LaneShape`s in the tldraw store
   d. Auto-set left/right adjacency topology
   e. Select the first lane

**Tool settings** (shown in a toolbar popover when tool is active):
- Lane count (default: 2)
- Lane width in meters (default: 3.5)
- Default line type for edge boundaries (default: "solid")
- Default line type for divider boundaries (default: "dashed")

#### 4.2.2 `AddAdjacentLaneTool`

Adds a lane to the left or right of an existing lane.

**Behavior**:
1. User selects an existing lane
2. Activates tool, chooses "Add Left" or "Add Right"
3. The existing outer boundary is reused as the new lane's inner boundary (shared reference)
4. A new boundary is generated at the specified offset
5. Topology automatically updated

#### 4.2.3 `KeyframeTool`

Places vehicle keyframes on the canvas.

**Behavior**:
1. Click to place a keyframe at that position
2. Heading defaults to the direction toward the next click (or 0 for the first)
3. Speed and timestamp editable in the property panel
4. Timestamp auto-increments based on distance and previous keyframe's speed

### 4.3 Syncing tldraw store ↔ core model

tldraw has its own reactive store (`Editor.store`). We need a **bidirectional sync** layer:

```
tldraw store (shapes)  ←→  core model (Scenario)
```

**Approach**: The tldraw store is the **source of truth** during editing. We derive the core `Scenario` model from the tldraw store on demand (for export, save).

```typescript
// editor/src/store/sync.ts

/** Extract a Scenario from the current tldraw editor state. */
function extractScenario(editor: Editor, schema: SchemaDefinition): Scenario {
  const boundaryShapes = editor.getCurrentPageShapes()
    .filter(s => s.type === 'boundary') as BoundaryShape[];
  const laneShapes = editor.getCurrentPageShapes()
    .filter(s => s.type === 'lane') as LaneShape[];
  const kfShapes = editor.getCurrentPageShapes()
    .filter(s => s.type === 'keyframe') as KeyframeShape[];
  
  // Convert to core model types...
  return { meta, map: { boundaries, lanes, topology }, trajectory: { keyframes } };
}

/** Load a Scenario into the tldraw editor. */
function loadScenario(editor: Editor, scenario: Scenario): void {
  editor.store.clear();
  // Create BoundaryShapes, LaneShapes, KeyframeShapes from scenario data
}
```

**Coordinate system**: tldraw uses screen-like coordinates (Y increases downward). We'll use this directly and apply the map→meters conversion factor only during export. A scale indicator on the canvas shows the current meters-per-pixel ratio.

---

## 5. UI Layout

```
┌─────────────────────────────────────────────────────────┐
│  Toolbar                                                │
│  [Draw Road] [Add Lane] [Keyframe] [Select] │ [Export]  │
├────────────────────────────────────────┬────────────────┤
│                                        │  Side Panel    │
│                                        │                │
│           tldraw Canvas                │  ┌───────────┐ │
│           (infinite, pan/zoom)         │  │ Properties │ │
│                                        │  │            │ │
│                                        │  │ (dynamic   │ │
│                                        │  │  from      │ │
│                                        │  │  schema)   │ │
│                                        │  │            │ │
│                                        │  ├───────────┤ │
│                                        │  │ Topology   │ │
│                                        │  │            │ │
│                                        │  │ pred: [dd] │ │
│                                        │  │ succ: [dd] │ │
│                                        │  │ left: [dd] │ │
│                                        │  │ right:[dd] │ │
│                                        │  ├───────────┤ │
│                                        │  │ Keyframes  │ │
│                                        │  │ (list)     │ │
│                                        │  └───────────┘ │
├────────────────────────────────────────┴────────────────┤
│  Status bar: cursor position (m) │ scale │ entity count │
└─────────────────────────────────────────────────────────┘
```

### 5.1 Toolbar

Top bar with tool buttons. The active tool is highlighted. When `DrawRoadTool` is active, a sub-bar shows lane count / width settings.

### 5.2 Side Panel

Context-sensitive panel on the right. Content depends on what's selected:

**Nothing selected** → instructions / map overview
**BoundaryShape selected** → boundary point attributes (schema-driven form). If boundary has segments with varying attributes, show a point list with per-point editing.
**LaneShape selected** → lane attributes + topology editor (dropdown per relation, multi-select for lane IDs)
**KeyframeShape selected** → pose (x, y, heading), speed, timestamp fields

The property panel is **generated dynamically** from the `SchemaDefinition`:
- `EnumAttributeDef` → `<select>` dropdown
- `NumberAttributeDef` → `<input type="number">` with min/max/step, or a slider
- `BooleanAttributeDef` → toggle switch

### 5.3 Export Dialog

Modal dialog triggered by the Export button:

- Format: JSON / YAML toggle
- Coordinate system: dropdown (from a predefined list + custom string)
- Frame rate: number input (Hz)
- Sensor range: number input (meters), or "unlimited"
- Include out-of-range topology: checkbox
- Preview: read-only text area showing a sample of the first frame
- [Export Map Only] [Export Frames Only] [Export All] buttons
- Output downloads as a file

---

## 6. Core Algorithms

### 6.1 Road drawing → boundary generation

```
Input: centerline points (screen coords), laneCount, laneWidth (m)

1. Resample centerline at even intervals (default: 5m × pixelsPerMeter)
2. For i in 0..laneCount:
     offset = (-laneCount/2 + i) × laneWidth × pixelsPerMeter
     boundaryPoints = offsetPolyline(centerline, offset)
3. Create BoundaryShape for each boundary polyline
4. Create LaneShape for each adjacent pair of boundaries
5. Set left/right adjacency topology automatically
```

**offsetPolyline algorithm**:
For each point, compute the average normal of the incoming and outgoing segments, then offset along that normal. Handle sharp corners by clamping the offset or inserting extra points (miter limit).

### 6.2 Trajectory interpolation

```
Input: keyframes sorted by timestamp, target frameRate (Hz)

1. Build a Catmull-Rom spline through keyframe positions
2. Interpolate heading with circular lerp (handle 0/2π wraparound)
3. Interpolate speed linearly
4. Sample at 1/frameRate intervals from first to last keyframe timestamp
5. Output: VehicleFrame[] with pose + speed + timestamp
```

### 6.3 Coordinate transform (map → vehicle-local)

```
For each VehicleFrame at time t:
  ego = frame.pose  (x, y, heading)
  
  For each boundary point p in the map:
    // Translate to ego-relative
    dx = p.x - ego.x
    dy = p.y - ego.y
    // Rotate by -heading
    local_x =  dx * cos(-heading) + dy * sin(-heading)  -- wrong
    local_x =  dx * cos(heading) + dy * sin(heading)
    local_y = -dx * sin(heading) + dy * cos(heading)
    
    // Apply coordinate convention
    // For "front_x_left_y" (ISO 8855): output (local_x, local_y)
    // For other conventions: apply additional rotation/flip
```

### 6.4 Sensor range clipping

```
For each boundary in the map:
  Filter points to those within sensorRangeM of the ego position
  If a boundary has 0 points in range, exclude it entirely
  
For each lane:
  If both left and right boundary are excluded, exclude the lane
  
For topology:
  If includeOutOfRangeTopology is false:
    Remove references to lanes not present in this frame
```

---

## 7. File Save/Load

### 7.1 Project file format

The project file (`.lane-studio.json`) saves the entire tldraw document state. This preserves all shape positions, zoom level, undo history, etc.

```typescript
interface ProjectFile {
  version: 1;
  schemaId: string;
  meta: ScenarioMeta;
  tldrawSnapshot: TLStoreSnapshot;  // tldraw's built-in serialization
}
```

### 7.2 Schema config file

A separate YAML file that defines the attribute schema. Referenced by `schemaId` in the project file. The app ships with `default-schema.yaml` and users can load custom schemas.

---

## 8. Implementation Plan (Phased)

### Phase 1: Minimal canvas + road drawing

**Goal**: Draw a centerline → see lanes appear with boundaries

- [ ] Initialize Vite + React + tldraw project
- [ ] Implement `BoundaryShapeUtil` (render polyline with style)
- [ ] Implement `LaneShapeUtil` (render fill between two boundaries)
- [ ] Implement `DrawRoadTool` (click points → commit → generate shapes)
- [ ] Basic toolbar (Draw Road / Select tool toggle)
- [ ] Grid background with scale indicator

**Deliverable**: User can draw roads and see multi-lane results on the canvas.

### Phase 2: Selection + property editing

- [ ] Click to select lane or boundary
- [ ] Side panel with schema-driven property editor
- [ ] Boundary attribute editing (line type, width per point)
- [ ] Lane attribute editing (speed limit, surface)
- [ ] Visual feedback: boundary style changes when attributes change

### Phase 3: Topology

- [ ] Topology panel with dropdown selectors per relation
- [ ] Visual overlay showing topology connections (arrows between lane centers)
- [ ] Auto-topology for `AddAdjacentLaneTool`

### Phase 4: Vehicle keyframes

- [ ] `KeyframeShapeUtil` (arrow icon with heading)
- [ ] `KeyframeTool` (click to place)
- [ ] Keyframe property panel (pose, speed, timestamp)
- [ ] Trajectory preview curve (Catmull-Rom) drawn on canvas
- [ ] Playback scrubber: slider that moves a vehicle icon along the trajectory

### Phase 5: Export

- [ ] Implement `core/interpolation` (Catmull-Rom)
- [ ] Implement `core/transform` (coordinate conversion)
- [ ] Implement default exporter (JSON/YAML)
- [ ] Export dialog UI
- [ ] Sensor range clipping

### Phase 6: Polish + extras

- [ ] Save/load project files
- [ ] Custom schema loading (drag-drop YAML)
- [ ] `AddAdjacentLaneTool` implementation
- [ ] Boundary joint snapping (for merge/split roads)
- [ ] Keyboard shortcuts
- [ ] Undo/redo (tldraw built-in, verify it works with custom shapes)

---

## 9. tldraw-Specific Implementation Notes

### 9.1 Defining a custom shape

```typescript
// shapes/BoundaryShapeUtil.ts
import { ShapeUtil, TLBaseShape, Geometry2d } from 'tldraw';

type BoundaryShape = TLBaseShape<'boundary', BoundaryShapeProps>;

class BoundaryShapeUtil extends ShapeUtil<BoundaryShape> {
  static override type = 'boundary' as const;

  // Define the default props for new shapes
  getDefaultProps(): BoundaryShapeProps { ... }

  // Define the hit-test geometry (for selection, snapping)
  getGeometry(shape: BoundaryShape): Geometry2d { ... }

  // React component that renders the shape
  component(shape: BoundaryShape) { ... }

  // SVG component for export/screenshot
  indicator(shape: BoundaryShape) { ... }
}
```

### 9.2 Defining a custom tool

```typescript
// tools/DrawRoadTool.ts
import { StateNode } from 'tldraw';

class DrawRoadTool extends StateNode {
  static override id = 'draw-road';

  // Child states for the tool's state machine
  static override children = () => [DrawRoadIdle, DrawRoadDrawing];
  static override initial = 'idle';
}

class DrawRoadIdle extends StateNode {
  static override id = 'idle';
  
  override onPointerDown() {
    // Transition to drawing state
    this.parent.transition('drawing');
  }
}

class DrawRoadDrawing extends StateNode {
  static override id = 'drawing';
  
  // Track points as user clicks
  override onPointerDown() { ... }
  
  // Commit on double-click or Enter
  override onDoubleClick() { ... }
  override onKeyDown(info) {
    if (info.key === 'Enter') this.commit();
    if (info.key === 'Escape') this.cancel();
  }
}
```

### 9.3 Registering shapes and tools

```typescript
// App.tsx
import { Tldraw } from 'tldraw';

const customShapeUtils = [BoundaryShapeUtil, LaneShapeUtil, KeyframeShapeUtil];
const customTools = [DrawRoadTool, AddAdjacentLaneTool, KeyframeTool];

function App() {
  return (
    <Tldraw
      shapeUtils={customShapeUtils}
      tools={customTools}
      components={{
        Toolbar: CustomToolbar,       // Our toolbar with custom buttons
        // Can also override other UI components
      }}
      onMount={(editor) => {
        // Initialize editor state, load schema, etc.
      }}
    />
  );
}
```

### 9.4 Accessing the editor

```typescript
// Inside any React component within tldraw
import { useEditor } from 'tldraw';

function PropertyPanel() {
  const editor = useEditor();
  
  // Get selected shapes
  const selectedShapes = editor.getSelectedShapes();
  
  // Update a shape's props
  editor.updateShape({
    id: shape.id,
    type: 'boundary',
    props: { ...newProps },
  });
  
  // Create a shape programmatically (used by DrawRoadTool)
  editor.createShape({
    type: 'boundary',
    x: 0, y: 0,
    props: { ... },
  });
}
```

---

## 10. Coordinate System Details

### 10.1 Canvas coordinates

tldraw uses a coordinate system where:
- **X** increases to the right
- **Y** increases downward
- Units are pixels (but we treat them as `pixels = meters × PIXELS_PER_METER`)

We display a scale bar on the canvas so the user knows the current mapping.

### 10.2 `PIXELS_PER_METER` constant

Default: `10` (1 meter = 10 pixels). This means a 3.5m lane width = 35px, which gives reasonable visual fidelity at default zoom.

### 10.3 Export coordinate transform

When exporting frames, positions are divided by `PIXELS_PER_METER` to convert to meters, then rotated into the vehicle-local frame.

---

## 11. Default Schema (YAML)

```yaml
id: default
name: Default Road Schema

boundary_point_attributes:
  line_type:
    type: enum
    label: Line Type
    values: [solid, dashed, double_solid, double_dashed, solid_dashed, dashed_solid, none]
    default: solid
    display:
      solid:         { color: "#FFFFFF", dash: null }
      dashed:        { color: "#FFFFFF", dash: [8, 6] }
      double_solid:  { color: "#FFD700", dash: null, stroke_width: 3 }
      double_dashed: { color: "#FFD700", dash: [8, 6], stroke_width: 3 }
      none:          { color: "#666666", dash: [2, 4] }
  width_m:
    type: number
    label: Line Width
    default: 0.15
    min: 0.01
    max: 1.0
    step: 0.01
    unit: m
  color:
    type: enum
    label: Line Color
    values: [white, yellow, blue, orange]
    default: white

lane_attributes:
  speed_limit:
    type: number
    label: Speed Limit
    default: 60
    min: 0
    max: 300
    step: 10
    unit: km/h
  road_surface:
    type: enum
    label: Road Surface
    values: [asphalt, concrete, gravel, dirt]
    default: asphalt

keyframe_attributes: {}

topology_relations:
  - predecessor
  - successor
  - left_adjacent
  - right_adjacent
```

---

## 12. Export Output Examples

### 12.1 Map-only export (JSON)

```json
{
  "schema_id": "default",
  "config": {
    "format": "json",
    "coordinate_system": "front_x_left_y",
    "pixels_per_meter": 10
  },
  "map": {
    "boundaries": {
      "bnd_001": {
        "id": "bnd_001",
        "points": [
          { "position": { "x": 0.0, "y": 0.0 }, "attributes": { "line_type": "solid", "width_m": 0.15, "color": "white" } },
          { "position": { "x": 5.0, "y": 0.0 }, "attributes": { "line_type": "solid", "width_m": 0.15, "color": "white" } }
        ],
        "start_joint": null,
        "end_joint": null
      },
      "bnd_002": {
        "id": "bnd_002",
        "points": [
          { "position": { "x": 0.0, "y": -3.5 }, "attributes": { "line_type": "dashed", "width_m": 0.15, "color": "white" } },
          { "position": { "x": 5.0, "y": -3.5 }, "attributes": { "line_type": "dashed", "width_m": 0.15, "color": "white" } }
        ],
        "start_joint": null,
        "end_joint": null
      }
    },
    "lanes": {
      "lane_001": {
        "id": "lane_001",
        "left_boundary_id": "bnd_001",
        "right_boundary_id": "bnd_002",
        "attributes": { "speed_limit": 60, "road_surface": "asphalt" }
      }
    },
    "topology": {
      "lane_001": {
        "predecessor": [],
        "successor": [],
        "left_adjacent": [],
        "right_adjacent": ["lane_002"]
      }
    }
  }
}
```

### 12.2 Frame export (JSON, single frame)

```json
{
  "timestamp": 1.0,
  "ego": {
    "map_pose": { "x": 25.0, "y": -1.75, "heading": 0.0 },
    "speed": 15.0
  },
  "boundaries": {
    "bnd_001": {
      "id": "bnd_001",
      "points": [
        { "position": { "x": -20.0, "y": 1.75 }, "attributes": { "line_type": "solid", "width_m": 0.15, "color": "white" } },
        { "position": { "x": -15.0, "y": 1.75 }, "attributes": { "line_type": "solid", "width_m": 0.15, "color": "white" } }
      ]
    }
  },
  "lanes": {
    "lane_001": {
      "id": "lane_001",
      "left_boundary_id": "bnd_001",
      "right_boundary_id": "bnd_002",
      "attributes": { "speed_limit": 60, "road_surface": "asphalt" }
    }
  },
  "topology": {
    "lane_001": {
      "predecessor": [],
      "successor": [],
      "left_adjacent": [],
      "right_adjacent": ["lane_002"]
    }
  }
}
```

---

## 13. Dependencies

```json
{
  "dependencies": {
    "tldraw": "^3.x",
    "react": "^18.x",
    "react-dom": "^18.x",
    "zustand": "^5.x",
    "js-yaml": "^4.x"
  },
  "devDependencies": {
    "typescript": "^5.x",
    "vite": "^6.x",
    "@vitejs/plugin-react": "^4.x",
    "tailwindcss": "^4.x",
    "vitest": "^3.x"
  }
}
```

---

## 14. Testing Strategy

- **Unit tests** (`vitest`): `core/interpolation`, `core/transform`, `core/export` — pure functions, easy to test
- **Integration tests**: Create shapes programmatically via tldraw editor API → verify `extractScenario()` produces correct model
- **Manual QA**: Focus on the drawing UX, since canvas interaction is hard to unit-test

---

## 15. Future Considerations (out of scope for v1)

- **Multi-trajectory**: Multiple vehicles with different trajectories
- **3D elevation**: Height data per boundary point
- **Import**: Load from OpenDRIVE / Lanelet2 / CommonRoad
- **Collaborative editing**: tldraw supports multiplayer via Cloudflare Workers
- **Plugin system**: Load custom exporters dynamically (ESM import)
- **Background image**: Load aerial/satellite imagery as a canvas backdrop for tracing
