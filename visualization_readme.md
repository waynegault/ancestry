# Code Graph Visualization

A standalone HTML-based interactive visualization tool for code graphs using D3.js.

## Features

- **Interactive Graph Visualization**: Nodes and edges representing code entities and their relationships
- **Multiple Layout Options**: Force-directed, circular, and hierarchical layouts
- **Node Type Color Coding**: Different colors for files, functions, classes, methods, modules, and workflows
- **Edge Type Color Coding**: Different colors for different relationship types (calls, defines, uses, etc.)
- **Interactive Controls**:
  - Search/filter nodes by name or type
  - Click nodes to highlight connections
  - Drag nodes to reposition them
  - Zoom and pan the graph
  - Toggle node labels on/off
  - Export graph as PNG image
- **Real-time Statistics**: Shows node and edge counts by type
- **Responsive Design**: Adapts to different screen sizes

## Usage

1. **Open the Visualization**:
   - Start a local web server: `python -m http.server 8000`
   - Open your browser to: `http://localhost:8000/visualize_code_graph.html`

2. **Load Your Data**:
   - The visualization automatically loads `docs/code_graph.json` if it exists
   - Or use the "Load JSON" button to load any code graph JSON file

3. **Interact with the Graph**:
   - **Click nodes** to highlight their connections
   - **Drag nodes** to reposition them
   - **Use mouse wheel** to zoom in/out
   - **Click and drag background** to pan the view
  - **Use dropdown filters** to select specific quality levels (stable, monitor, needs-review, dead-code) and node types (file, function, class, method, module, workflow, config, asset)
  - **Use text search box** to filter nodes by any characteristic (name, type, path, summary, mechanism, quality flags, concerns, opportunities, tests, notes)
  - **Combine filters** - dropdown and text filters work together for precise filtering
   - **Change layout** using the dropdown menu

4. **Export Results**:
   - Click "Export PNG" to save the current view as an image

## Data Format

The visualization expects JSON data with the following structure:

```json
{
  "nodes": [
    {
      "id": "unique_identifier",
      "type": "file|function|class|method|module|workflow",
      "name": "display_name",
      "path": "optional_file_path",
      "summary": "optional_description"
    }
  ],
  "edges": [
    {
      "source": "source_node_id",
      "target": "target_node_id",
      "type": "relationship_type",
      "description": "optional_edge_description"
    }
  ]
}
```

## Node Types and Colors

- **File** (Red): Source code files
- **Function** (Teal): Standalone functions
- **Class** (Blue): Class definitions
- **Method** (Green): Class methods
- **Module** (Yellow): Python modules
- **Workflow** (Pink): Workflow/coordination functions
- **Config** (Orange): Configuration data structures and loaders
- **Asset** (Purple): Non-code artifacts (templates, prompts, SQL files)

## Enhanced Tooltip Information

The visualization displays comprehensive node information including:
- **Basic Info**: Name, type, connections, file path
- **Documentation**: Summary and mechanism descriptions
- **Quality Metrics**: Quality flags (stable, monitor, needs-review, dead-code)
- **Concerns & Opportunities**: Lists of reservations and improvement ideas
- **Testing**: Test coverage information
- **Notes**: Free-form observations and TODO references

## Edge Types and Colors

- **defines**: File defines a function/class
- **calls**: Function calls another function
- **depends-on**: Dependency relationship
- **uses**: Uses relationship
- **mutates**: Mutates/modifies relationship
- **decorated-with**: Decorator relationship
- **instantiates**: Instantiation relationship
- **wraps**: Wrapper relationship
- **returns**: Return relationship

## Browser Compatibility

Works in all modern browsers that support:
- ES6+ JavaScript
- SVG
- D3.js v7

## Dependencies

- D3.js v7 (loaded from CDN)
- No other external dependencies

## Customization

You can easily customize:
- Colors for node types in the `colorScale` object
- Node sizes in the `getNodeSize` method
- Edge colors in the `getLinkColor` method
- Layout algorithms in the layout methods

## Performance

Optimized for graphs with:
- Up to 1000 nodes (larger graphs may require performance tuning)
- Real-time interaction and smooth animations
- Efficient force simulation with collision detection
