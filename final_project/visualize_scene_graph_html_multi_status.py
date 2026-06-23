#!/usr/bin/env python3
"""
visualize_scene_graph_html_multi_status.py

A multi-event version of visualize_scene_graph_html.py.

Use this when you have one merged graph.json but multiple event.json files.
It shows:
  - all nodes
  - all functional relations from event.json files
  - all status transitions from event.json files
  - selected reasoning for each event

The 3D viewer is intentionally kept simple and uses the same scene .ply plus
graph nodes/edges. The right-side status table is rendered statically, so it
works even if Three.js fails.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import html
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def esc(x: Any) -> str:
    return html.escape(str(x if x is not None else ""), quote=True)


def read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        print(f"[WARN] Missing JSON: {path}")
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_xyz(x: Any) -> Optional[List[float]]:
    if not isinstance(x, (list, tuple)) or len(x) < 3:
        return None
    out = []
    for v in x[:3]:
        try:
            fv = float(v)
        except Exception:
            return None
        if not math.isfinite(fv):
            return None
        out.append(fv)
    return out


def stable_color(label: str) -> str:
    label = str(label or "unknown").strip().lower()
    digest = hashlib.md5(label.encode("utf-8")).digest()
    r, g, b = digest[0], digest[1], digest[2]
    r = int(75 + 0.68 * r)
    g = int(75 + 0.68 * g)
    b = int(75 + 0.68 * b)
    return f"#{r:02x}{g:02x}{b:02x}"


def relpath_for_html(target: Path, html_path: Path) -> str:
    target = Path(target).resolve()
    html_dir = Path(html_path).resolve().parent
    return os.path.relpath(target, html_dir).replace(os.sep, "/")


def compact_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    raw_nodes = graph.get("nodes", {}) or {}
    raw_edges = graph.get("edges", {}) or {}

    nodes: Dict[str, Dict[str, Any]] = {}
    for node_id, node in raw_nodes.items():
        node_id = str(node_id)
        label = str(node.get("label", "unknown"))
        nodes[node_id] = {
            "node_id": node_id,
            "label": label,
            "node_type": str(node.get("node_type", "unknown")),
            "centroid": safe_xyz(node.get("centroid")),
            "n_points": int(node.get("n_points", 0) or 0),
            "state": str(node.get("state", "unknown")),
            "description": str(node.get("description", "")),
            "confidence": float(node.get("confidence", 0.0) or 0.0),
            "color": stable_color(label),
        }

    edges: Dict[str, Dict[str, Any]] = {}
    for edge_id, edge in raw_edges.items():
        edge_id = str(edge_id)
        src_id = str(edge.get("src_id", ""))
        dst_id = str(edge.get("dst_id", ""))
        edges[edge_id] = {
            "edge_id": edge_id,
            "src_id": src_id,
            "dst_id": dst_id,
            "src_label": nodes.get(src_id, {}).get("label", src_id),
            "dst_label": nodes.get(dst_id, {}).get("label", dst_id),
            "edge_type": str(edge.get("edge_type", "functional")),
            "relationship": str(edge.get("relationship", "unknown")),
            "confidence": float(edge.get("confidence", 0.0) or 0.0),
            "cot_reasoning": str(edge.get("cot_reasoning", "")),
        }

    return {"nodes": nodes, "edges": edges}


def compact_event(event: Dict[str, Any], source_path: str) -> Dict[str, Any]:
    keep = [
        "clip_id", "frame_start", "frame_end",
        "interaction_frame_start", "interaction_frame_end",
        "status_frame_start", "status_frame_end",
        "status_appended_inactive_clip_id",
        "status_appended_inactive_frame_start",
        "status_appended_inactive_frame_end",
        "element", "tracked_object", "object", "parent_object",
        "functional_relationship", "functional_confidence", "functional_reasoning",
        "object_state_before", "object_state_after", "object_state",
        "status_changed", "status_change_confidence", "status_change_reasoning",
        "status_video_path", "object_state_active_video_path",
        "detection_reasoning", "motion_coupling_reasoning",
    ]
    out = {k: event[k] for k in keep if k in event}
    out["source_event_json"] = source_path
    return out


def event_clip_text(e: Dict[str, Any]) -> str:
    clip_id = e.get("clip_id", "?")
    fs = e.get("frame_start", "?")
    fe = e.get("frame_end", "?")
    sfs = e.get("status_frame_start")
    sfe = e.get("status_frame_end")
    if sfs is not None and sfe is not None and (sfs != fs or sfe != fe):
        return f"{clip_id}: interaction {fs}-{fe}, status {sfs}-{sfe}"
    return f"{clip_id}: {fs}-{fe}"


def render_summary(events: Sequence[Dict[str, Any]]) -> str:
    changed = sum(1 for e in events if e.get("status_changed") is True)
    unchanged = sum(1 for e in events if e.get("status_changed") is False)
    unknown = len(events) - changed - unchanged
    return f"""
    <div class="kv">
      <div class="k">Events</div><div class="v">{len(events)}</div>
      <div class="k">Changed</div><div class="v"><span class="pill good">{changed}</span></div>
      <div class="k">Unchanged</div><div class="v"><span class="pill warn">{unchanged}</span></div>
      <div class="k">Unknown</div><div class="v"><span class="pill">{unknown}</span></div>
    </div>
    """


def render_nodes(graph: Dict[str, Any]) -> str:
    nodes = list((graph.get("nodes") or {}).values())
    if not nodes:
        return '<div class="small">No nodes found in graph.json.</div>'
    rows = []
    for n in nodes:
        payload = esc(json.dumps(n, ensure_ascii=False))
        rows.append(f"""
        <div class="nodeItem" onclick='selectNode(JSON.parse(this.dataset.payload))' data-payload="{payload}">
          <div class="swatch" style="background:{esc(n.get('color'))}"></div>
          <div class="nodeMain">
            <div class="nodeLabel">{esc(n.get('label'))} <span class="small">({esc(n.get('node_type'))})</span></div>
            <div class="nodeMeta">state={esc(n.get('state'))}, points={esc(n.get('n_points'))}</div>
          </div>
        </div>
        """)
    return "\n".join(rows)


def render_events(events: Sequence[Dict[str, Any]]) -> str:
    if not events:
        return '<div class="small">No event.json loaded.</div>'
    rows = []
    for e in events:
        before = e.get("object_state_before", "unknown")
        after = e.get("object_state_after") or e.get("object_state") or "unknown"
        changed = e.get("status_changed")
        changed_html = '<span class="pill good">changed</span>' if changed is True else '<span class="pill warn">unchanged</span>' if changed is False else '<span class="pill">unknown</span>'
        payload = esc(json.dumps(e, ensure_ascii=False))
        rows.append(f"""
        <div class="eventItem" onclick='selectEvent(JSON.parse(this.dataset.payload))' data-payload="{payload}">
          <div class="eventTitle">{esc(event_clip_text(e))} {changed_html}</div>
          <div class="eventMeta">{esc(e.get("element", "-"))} --{esc(e.get("functional_relationship", "unknown"))}--> {esc(e.get("object") or e.get("tracked_object") or "-")}</div>
          <div class="eventMeta">status: <b>{esc(before)}</b> → <b>{esc(after)}</b></div>
        </div>
        """)
    return "\n".join(rows)


def render_status_table(events: Sequence[Dict[str, Any]]) -> str:
    if not events:
        return '<tr><td colspan="6">No event.json loaded.</td></tr>'
    rows = []
    for e in events:
        payload = esc(json.dumps(e, ensure_ascii=False))
        obj = e.get("object") or e.get("tracked_object") or "-"
        before = e.get("object_state_before", "unknown")
        after = e.get("object_state_after") or e.get("object_state") or "unknown"
        changed = e.get("status_changed")
        changed_html = '<span class="pill good">true</span>' if changed is True else '<span class="pill warn">false</span>' if changed is False else "-"
        rows.append(f"""
        <tr onclick='selectEvent(JSON.parse(this.dataset.payload))' data-payload="{payload}">
          <td>{esc(event_clip_text(e))}</td>
          <td>{esc(e.get("element", "-"))}</td>
          <td>{esc(obj)}</td>
          <td>{esc(before)}</td>
          <td><span class="pill good">{esc(after)}</span></td>
          <td>{changed_html}</td>
        </tr>
        """)
    return "\n".join(rows)


def make_html(scene_ply_rel: str, graph: Dict[str, Any], events: List[Dict[str, Any]], title: str, point_size: float, node_radius: float) -> str:
    graph_json = json.dumps(graph, ensure_ascii=False)
    scene_ply_json = json.dumps(scene_ply_rel)
    summary_html = render_summary(events)
    nodes_html = render_nodes(graph)
    events_html = render_events(events)
    status_rows = render_status_table(events)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{esc(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{ --blue:#0b376d; --blue2:#0f5f9e; --panel:#fff; --panel2:#f5f7fb; --border:#d9e1ec; --text:#172033; --muted:#667085; --good:#149954; --warn:#c47f00; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:Arial, Helvetica, sans-serif; color:var(--text); background:#eef2f7; overflow:hidden; }}
  #topbar {{ height:48px; background:var(--blue); color:white; display:flex; align-items:center; padding:0 18px; gap:14px; }}
  #topbar .title {{ font-weight:700; font-size:17px; }}
  #topbar .subtitle {{ color:#bad4ef; font-size:13px; }}
  #main {{ display:grid; grid-template-columns:1fr 560px; height:calc(100vh - 48px); min-height:0; }}
  #viewerWrap {{ position:relative; background:#111827; min-width:0; min-height:0; }}
  #viewer {{ width:100%; height:100%; display:block; }}
  #overlayHelp {{ position:absolute; left:16px; top:16px; background:rgba(0,0,0,.62); color:white; padding:10px 12px; border-radius:8px; font-size:13px; line-height:1.35; max-width:520px; }}
  #threeError {{ position:absolute; left:16px; bottom:16px; background:rgba(130,30,30,.88); color:white; padding:10px 12px; border-radius:8px; font-size:13px; max-width:620px; display:none; white-space:pre-wrap; }}
  #side {{ background:var(--panel); border-left:1px solid var(--border); overflow-y:auto; padding:14px; }}
  .card {{ background:white; border:1px solid var(--border); border-radius:10px; margin-bottom:12px; overflow:hidden; }}
  .card h2 {{ margin:0; font-size:15px; background:var(--panel2); border-bottom:1px solid var(--border); padding:10px 12px; }}
  .card .body {{ padding:10px 12px; }}
  .kv {{ display:grid; grid-template-columns:105px 1fr; gap:6px 10px; font-size:13px; line-height:1.35; }}
  .k,.muted {{ color:var(--muted); }}
  .v {{ font-weight:600; }}
  .pill {{ display:inline-block; padding:2px 7px; border-radius:999px; font-size:12px; font-weight:700; background:#e8f1ff; color:#0b4e8a; }}
  .pill.good {{ background:#e8f7ef; color:var(--good); }}
  .pill.warn {{ background:#fff2d7; color:var(--warn); }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  th {{ text-align:left; padding:8px; color:white; background:var(--blue2); font-weight:700; }}
  td {{ border-bottom:1px solid var(--border); padding:8px; vertical-align:top; }}
  tr:hover td {{ background:#f2f7ff; cursor:pointer; }}
  .nodeItem,.eventItem {{ border-bottom:1px solid var(--border); padding:8px 0; cursor:pointer; }}
  .nodeItem {{ display:flex; gap:8px; align-items:flex-start; }}
  .nodeItem:hover,.eventItem:hover {{ background:#f8fbff; }}
  .eventTitle {{ font-weight:700; font-size:13px; margin-bottom:3px; }}
  .eventMeta,.nodeMeta,.small {{ color:var(--muted); font-size:12px; margin-top:2px; }}
  .swatch {{ width:14px; height:14px; border-radius:3px; border:1px solid rgba(0,0,0,.2); margin-top:2px; flex:0 0 auto; }}
  .nodeMain {{ flex:1; min-width:0; }}
  .nodeLabel {{ font-weight:700; font-size:13px; }}
  .reasoning {{ font-size:13px; line-height:1.45; color:#263449; white-space:pre-wrap; }}
  #selectedTitle {{ font-weight:700; margin-bottom:6px; }}
  button {{ background:#0f5f9e; color:white; border:0; padding:7px 10px; border-radius:6px; cursor:pointer; font-weight:700; }}
  .btnRow {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; }}
</style>
</head>
<body>
<div id="topbar"><div class="title">Functional 3D Scene Graph Viewer</div><div class="subtitle">{esc(title)} · all status transitions</div></div>
<div id="main">
  <div id="viewerWrap">
    <canvas id="viewer"></canvas>
    <div id="overlayHelp"><b>Mouse controls</b><br>Left drag: rotate · Right drag: pan · Scroll: zoom<br>This viewer shows all loaded status transitions.</div>
    <div id="threeError"></div>
  </div>
  <aside id="side">
    <div class="card"><h2>Summary</h2><div class="body">{summary_html}<div class="btnRow"><button onclick="resetCamera()">Reset camera</button><button onclick="toggleGraph()">Show/hide graph</button></div></div></div>
    <div class="card"><h2>All events</h2><div class="body">{events_html}</div></div>
    <div class="card"><h2>All status transitions</h2><div class="body"><table><thead><tr><th>Clip</th><th>Element</th><th>Object</th><th>Before</th><th>After</th><th>Changed</th></tr></thead><tbody>{status_rows}</tbody></table></div></div>
    <div class="card"><h2>Nodes</h2><div class="body">{nodes_html}</div></div>
    <div class="card"><h2>Selected description</h2><div class="body"><div id="selectedTitle">All status viewer</div><div class="reasoning" id="selectedText">Click any event/status row to see functional reasoning and status reasoning.</div></div></div>
  </aside>
</div>
<script>
window.SCENE_PLY_DATA = {scene_ply_json};
window.GRAPH_DATA = {graph_json};
window.POINT_SIZE = {float(point_size)};
window.NODE_RADIUS = {float(node_radius)};
function selectNode(n) {{
  document.getElementById("selectedTitle").textContent = `${{n.label}} (${{n.node_type}})`;
  document.getElementById("selectedText").textContent = `State: ${{n.state || "unknown"}}\\nConfidence: ${{n.confidence ?? "-"}}\\nPoints: ${{n.n_points ?? 0}}\\n\\n${{n.description || "No description saved."}}`;
  if (window.focusNode3D) window.focusNode3D(n.node_id);
}}
function selectEvent(e) {{
  const before = e.object_state_before || "unknown";
  const after = e.object_state_after || e.object_state || "unknown";
  const clipText = e.status_frame_start !== undefined ? `${{e.clip_id}}: interaction ${{e.frame_start}}-${{e.frame_end}}, status ${{e.status_frame_start}}-${{e.status_frame_end}}` : `${{e.clip_id}}: ${{e.frame_start}}-${{e.frame_end}}`;
  document.getElementById("selectedTitle").textContent = `${{e.element || "-"}} --${{e.functional_relationship || "unknown"}}--> ${{e.object || e.tracked_object || "-"}}`;
  document.getElementById("selectedText").textContent =
    `Clip: ${{clipText}}\\n` +
    `Status transition: ${{before}} → ${{after}}\\n` +
    `Status changed: ${{e.status_changed ?? "-"}}\\n` +
    `Status confidence: ${{e.status_change_confidence ?? "-"}}\\n\\n` +
    `Functional reasoning:\\n${{e.functional_reasoning || "No functional reasoning saved."}}\\n\\n` +
    `Status reasoning:\\n${{e.status_change_reasoning || "No status reasoning saved."}}\\n\\n` +
    `Source event.json:\\n${{e.source_event_json || "-"}}`;
}}
window.resetCamera = function() {{ if (window.resetCamera3D) window.resetCamera3D(); }};
window.toggleGraph = function() {{ if (window.toggleGraph3D) window.toggleGraph3D(); }};
</script>
<script type="importmap">{{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}}}</script>
<script type="module">
import * as THREE from "three";
import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";
import {{ PLYLoader }} from "three/addons/loaders/PLYLoader.js";
const SCENE_PLY = window.SCENE_PLY_DATA;
const GRAPH = window.GRAPH_DATA || {{}};
const POINT_SIZE = window.POINT_SIZE || 0.008;
const NODE_RADIUS = window.NODE_RADIUS || 0.035;
const errorBox = document.getElementById("threeError");
function showError(msg) {{ console.error(msg); errorBox.style.display = "block"; errorBox.textContent = msg; }}
const canvas = document.getElementById("viewer");
const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
renderer.setPixelRatio(window.devicePixelRatio || 1);
renderer.setClearColor(0x111827, 1);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111827);
const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000);
camera.position.set(0, -2.2, 1.4);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 0.8));
scene.add(new THREE.AxesHelper(0.25));
const graphGroup = new THREE.Group();
scene.add(graphGroup);
let scenePointCloud = null;
let graphVisible = true;
const nodeMeshes = new Map();
const edgeMeshes = new Map();
function safeVec3(arr) {{
  if (!Array.isArray(arr) || arr.length < 3) return null;
  const x = Number(arr[0]), y = Number(arr[1]), z = Number(arr[2]);
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null;
  return new THREE.Vector3(x, y, z);
}}
function makeTextSprite(text, color = "#ffffff") {{
  const c = document.createElement("canvas");
  const ctx = c.getContext("2d");
  ctx.font = "bold 34px Arial";
  const metrics = ctx.measureText(text);
  c.width = Math.ceil(metrics.width + 28);
  c.height = 56;
  ctx.font = "bold 34px Arial";
  ctx.fillStyle = "rgba(0,0,0,0.70)";
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.strokeRect(1.5, 1.5, c.width - 3, c.height - 3);
  ctx.fillStyle = "#ffffff";
  ctx.fillText(text, 14, 39);
  const texture = new THREE.CanvasTexture(c);
  const material = new THREE.SpriteMaterial({{ map: texture, transparent: true }});
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(c.width / 520, c.height / 520, 1);
  return sprite;
}}
function buildGraph3D() {{
  const nodes = GRAPH.nodes || {{}};
  const edges = GRAPH.edges || {{}};
  for (const [id, node] of Object.entries(nodes)) {{
    const p = safeVec3(node.centroid);
    if (!p) continue;
    const radius = node.node_type === "element" ? NODE_RADIUS * 0.85 : NODE_RADIUS;
    const sphere = new THREE.Mesh(new THREE.SphereGeometry(radius, 24, 16), new THREE.MeshBasicMaterial({{ color: node.color || "#ff9933" }}));
    sphere.position.copy(p);
    graphGroup.add(sphere);
    nodeMeshes.set(id, sphere);
    const label = makeTextSprite(node.label || id, node.color);
    label.position.copy(p.clone().add(new THREE.Vector3(0, 0, NODE_RADIUS * 2.8)));
    graphGroup.add(label);
  }}
  for (const [id, edge] of Object.entries(edges)) {{
    const src = nodes[edge.src_id];
    const dst = nodes[edge.dst_id];
    const ps = safeVec3(src && src.centroid);
    const pd = safeVec3(dst && dst.centroid);
    if (!ps || !pd) continue;
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([ps, pd]), new THREE.LineBasicMaterial({{ color: 0xff4d4d }}));
    graphGroup.add(line);
    edgeMeshes.set(id, line);
    const mid = ps.clone().add(pd).multiplyScalar(0.5);
    const label = makeTextSprite(edge.relationship || "relation", "#ff4d4d");
    label.position.copy(mid.clone().add(new THREE.Vector3(0, 0, NODE_RADIUS * 2.2)));
    graphGroup.add(label);
  }}
}}
function focusPoint(p) {{
  if (!p) return;
  controls.target.copy(p);
  camera.position.copy(p.clone().add(new THREE.Vector3(0.35, -0.75, 0.35)));
  camera.updateProjectionMatrix();
  controls.update();
}}
window.focusNode3D = function(nodeId) {{
  for (const m of nodeMeshes.values()) m.scale.set(1, 1, 1);
  const mesh = nodeMeshes.get(nodeId);
  if (mesh) {{ mesh.scale.set(1.8, 1.8, 1.8); focusPoint(mesh.position); }}
}};
window.resetCamera3D = function() {{
  if (scenePointCloud && scenePointCloud.geometry.boundingSphere) {{
    const c = scenePointCloud.geometry.boundingSphere.center;
    const r = scenePointCloud.geometry.boundingSphere.radius || 1.0;
    controls.target.copy(c);
    camera.position.set(c.x, c.y - r * 2.2, c.z + r * 0.9);
    camera.near = Math.max(0.001, r / 1000);
    camera.far = Math.max(1000, r * 20);
    camera.updateProjectionMatrix();
    controls.update();
  }}
}};
window.toggleGraph3D = function() {{ graphVisible = !graphVisible; graphGroup.visible = graphVisible; }};
function loadPointCloud() {{
  const loader = new PLYLoader();
  loader.load(SCENE_PLY, (geometry) => {{
    geometry.computeBoundingSphere();
    const hasColor = geometry.getAttribute("color") !== undefined;
    const mat = new THREE.PointsMaterial({{ size: POINT_SIZE, vertexColors: hasColor, color: hasColor ? 0xffffff : 0xd0d7e2, sizeAttenuation: true }});
    scenePointCloud = new THREE.Points(geometry, mat);
    scene.add(scenePointCloud);
    window.resetCamera3D();
  }}, undefined, () => {{ showError("Could not load PLY file: " + SCENE_PLY + "\\n\\nThe right-side functional graph is still valid."); }});
}}
function resize() {{
  const wrap = document.getElementById("viewerWrap");
  const w = wrap.clientWidth;
  const h = wrap.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(1, h);
  camera.updateProjectionMatrix();
}}
window.addEventListener("resize", resize);
function animate() {{ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }}
try {{ buildGraph3D(); loadPointCloud(); resize(); animate(); }} catch (e) {{ showError("Three.js viewer failed:\\n" + String(e)); }}
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-ply", type=Path, required=True)
    parser.add_argument("--graph-json", type=Path, required=True)
    parser.add_argument("--event-jsons", type=Path, nargs="*", default=[])
    parser.add_argument("--event-glob", type=str, default="")
    parser.add_argument("--out-html", type=Path, required=True)
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--point-size", type=float, default=0.008)
    parser.add_argument("--node-radius", type=float, default=0.035)
    args = parser.parse_args()

    if not args.scene_ply.exists():
        raise FileNotFoundError(f"scene ply not found: {args.scene_ply}")
    if not args.graph_json.exists():
        raise FileNotFoundError(f"graph json not found: {args.graph_json}")

    event_paths: List[Path] = list(args.event_jsons or [])
    if args.event_glob:
        event_paths.extend(Path(p) for p in sorted(glob.glob(args.event_glob)))

    seen = set()
    deduped = []
    for p in event_paths:
        rp = str(Path(p).resolve())
        if rp not in seen:
            seen.add(rp)
            deduped.append(Path(p))
    event_paths = deduped

    graph = compact_graph(read_json(args.graph_json))
    events = []
    for p in event_paths:
        event_raw = read_json(p)
        if event_raw:
            events.append(compact_event(event_raw, source_path=str(p)))

    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    title = args.title or f"{args.scene_ply.parent.name} / {args.scene_ply.name}"
    html_text = make_html(
        scene_ply_rel=relpath_for_html(args.scene_ply, args.out_html),
        graph=graph,
        events=events,
        title=title,
        point_size=args.point_size,
        node_radius=args.node_radius,
    )
    args.out_html.write_text(html_text, encoding="utf-8")

    print(f"[OK] Wrote viewer: {args.out_html}")
    print(f"[OK] Loaded events: {len(events)}")
    for e in events:
        print(
            f"  - {e.get('element')} --{e.get('functional_relationship')}--> "
            f"{e.get('object') or e.get('tracked_object')}: "
            f"{e.get('object_state_before')} -> {e.get('object_state_after') or e.get('object_state')}"
        )


if __name__ == "__main__":
    main()
