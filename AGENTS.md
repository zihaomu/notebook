# Workspace instructions

## Article diagrams

- Treat `diagrams/specs/*.json` as the source of truth for generated diagrams.
- Use the local Excalidraw renderer after changing a diagram specification.
- Do not manually edit files under `diagrams/generated/`.
- Never overwrite or delete files under `diagrams/final/`; they contain manual edits.
- Keep stable semantic node IDs when changing labels or relationships.
- Use SVG, not screenshots, in Markdown articles.
- Keep all generation local; do not create Excalidraw share links or upload scenes.
- Verify labels do not overflow, nodes do not overlap, and reading order is clear.

### Workflow

1. Read the article and identify the relationship that needs clarification. Skip the diagram when prose or a short list is clearer.
2. Choose the smallest fitting structure: argument map, causal chain, comparison, timeline, hierarchy, or process.
3. Create or update `diagrams/specs/<semantic-name>.json`; use stable semantic node IDs and keep one concise claim or fact per node.
4. Generate the editable scene and SVG locally:

   ```bash
   /Users/zmu/.codex/skills/local-excalidraw-article-diagrams/scripts/render.sh \
     /absolute/path/to/diagrams/specs/<semantic-name>.json
   ```

5. Confirm the renderer reports `network: local-only`, both generated files exist, JSON parses, and the SVG has a nonzero `viewBox`.
6. For every new theme or materially changed layout, render the SVG locally and visually inspect clipping, label overflow, edge-label collisions, node overlap, and reading order.
7. Insert the generated SVG into the Markdown article with a relative path. Keep the generated `.excalidraw` file beside it for later editing.
8. Regenerate only under `diagrams/generated/` while logic is changing. Create a copy under `diagrams/final/` only when the user explicitly requests a new manual-polish copy and no file of the same name exists.
9. On completion, report the changed spec, generated `.excalidraw` and SVG paths, article link update, validation performed, and whether anything under `diagrams/final/` was touched.
