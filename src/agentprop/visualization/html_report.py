"""Self-contained interactive HTML view for workflow graphs and control traces.

The output is a single HTML file with no external assets or network calls, so
it can be opened locally, attached to CI artifacts, or shared as-is.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentprop.core import AgentGraph

_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>__TITLE__ — AgentProp</title>
<style>
:root { --bg:#0e1116; --panel:#161b22; --ink:#e6edf3; --dim:#8b949e; --accent:#12c95b; }
body { margin:0; font:14px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;
       background:var(--bg); color:var(--ink); }
header { padding:16px 24px; border-bottom:1px solid #30363d; }
header h1 { margin:0; font-size:18px; }
header .sub { color:var(--dim); font-size:12px; }
main { display:flex; gap:0; }
#graph { flex:2; min-height:560px; }
#side { flex:1; max-width:420px; padding:16px 24px; border-left:1px solid #30363d; }
.card { background:var(--panel); border:1px solid #30363d; border-radius:8px;
        padding:10px 14px; margin-bottom:12px; }
.card h2 { margin:0 0 6px; font-size:13px; color:var(--dim); text-transform:uppercase; }
.badge { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11px;
         margin:1px 3px 1px 0; border:1px solid #30363d; }
.badge.verifier { border-color:var(--accent); color:var(--accent); }
.badge.seed { border-color:#58a6ff; color:#58a6ff; }
.badge.bottleneck { border-color:#f0883e; color:#f0883e; }
.trace-row { border-left:3px solid #30363d; padding:2px 10px; margin:4px 0; font-size:12px; }
.trace-row.decision { border-left-color:var(--accent); }
svg text { fill:var(--ink); font-size:11px; pointer-events:none; }
circle { cursor:grab; }
.legend { color:var(--dim); font-size:12px; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="sub">AgentProp workflow view · __NODES__ nodes · __EDGES__ edges</div>
</header>
<main>
  <svg id="graph"></svg>
  <div id="side">
    <div class="card"><h2>Legend</h2>
      <div class="legend">
        <span class="badge verifier">verifier</span>
        <span class="badge seed">seed</span>
        <span class="badge bottleneck">bottleneck</span>
        — node size scales with token cost; edge opacity with weight.
      </div>
    </div>
    <div class="card" id="analysis"></div>
    <div class="card" id="detail"><h2>Node detail</h2><div>Click a node.</div></div>
    <div class="card" id="trace"></div>
  </div>
</main>
<script>
const DATA = __DATA__;
const svg = document.getElementById("graph");
const W = svg.clientWidth || 800, H = Math.max(svg.clientHeight, 560);
svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
const nodes = DATA.nodes.map((n, i) => ({...n,
  x: W/2 + 180*Math.cos(2*Math.PI*i/DATA.nodes.length),
  y: H/2 + 180*Math.sin(2*Math.PI*i/DATA.nodes.length), vx:0, vy:0}));
const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
const edges = DATA.edges.filter(e => byId[e.source] && byId[e.target]);
const verifiers = new Set(DATA.analysis.verifier_placement || []);
const seeds = new Set(DATA.analysis.recommended_seeds || []);
const bottlenecks = new Set((DATA.analysis.bottlenecks || []).map(b => b.node));

function tick(iters) {
  for (let it = 0; it < iters; it++) {
    for (const a of nodes) for (const b of nodes) {
      if (a === b) continue;
      let dx = a.x-b.x, dy = a.y-b.y, d2 = Math.max(dx*dx+dy*dy, 64);
      const f = 2600/d2; a.vx += f*dx/Math.sqrt(d2); a.vy += f*dy/Math.sqrt(d2);
    }
    for (const e of edges) {
      const s = byId[e.source], t = byId[e.target];
      let dx = t.x-s.x, dy = t.y-s.y, d = Math.max(Math.hypot(dx,dy),1);
      const f = 0.01*(d-130);
      s.vx += f*dx/d; s.vy += f*dy/d; t.vx -= f*dx/d; t.vy -= f*dy/d;
    }
    for (const n of nodes) {
      n.vx += 0.005*(W/2-n.x); n.vy += 0.005*(H/2-n.y);
      n.x = Math.min(Math.max(n.x + n.vx*0.5, 30), W-30);
      n.y = Math.min(Math.max(n.y + n.vy*0.5, 30), H-30);
      n.vx *= 0.6; n.vy *= 0.6;
    }
  }
}
tick(300);

const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}
const defs = el("defs", {});
const marker = el("marker", {id:"arrow", viewBox:"0 0 10 10", refX:"20", refY:"5",
  markerWidth:"6", markerHeight:"6", orient:"auto-start-reverse"});
marker.appendChild(el("path", {d:"M 0 0 L 10 5 L 0 10 z", fill:"#8b949e"}));
defs.appendChild(marker); svg.appendChild(defs);
const maxW = Math.max(...edges.map(e => e.weight || 1), 1);
for (const e of edges) {
  const s = byId[e.source], t = byId[e.target];
  svg.appendChild(el("line", {x1:s.x, y1:s.y, x2:t.x, y2:t.y, stroke:"#8b949e",
    "stroke-opacity": (0.25 + 0.6*(e.weight||1)/maxW).toFixed(2),
    "stroke-width":"1.4", "marker-end":"url(#arrow)"}));
}
const maxCost = Math.max(...nodes.map(n => n.token_cost || 0), 1);
function nodeColor(n) {
  if (verifiers.has(n.id)) return "#12c95b";
  if (seeds.has(n.id)) return "#58a6ff";
  if (bottlenecks.has(n.id)) return "#f0883e";
  return "#c9d1d9";
}
for (const n of nodes) {
  const r = 9 + 9*(n.token_cost||0)/maxCost;
  const c = el("circle", {cx:n.x, cy:n.y, r:r, fill:"#161b22",
    stroke:nodeColor(n), "stroke-width":"2.5"});
  c.addEventListener("click", () => showDetail(n));
  svg.appendChild(c);
  svg.appendChild(el("text", {x:n.x, y:n.y - r - 5, "text-anchor":"middle"}))
     .textContent = n.id;
}
function showDetail(n) {
  const rows = [["type", n.type], ["role", n.role], ["token cost", n.token_cost],
    ["latency", n.latency], ["reliability", n.reliability], ["error rate", n.error_rate]]
    .filter(([,v]) => v !== null && v !== undefined && v !== "")
    .map(([k,v]) => `<div><span style="color:var(--dim)">${k}</span>: ${v}</div>`).join("");
  const tags = [verifiers.has(n.id) ? '<span class="badge verifier">verifier</span>' : "",
    seeds.has(n.id) ? '<span class="badge seed">seed</span>' : "",
    bottlenecks.has(n.id) ? '<span class="badge bottleneck">bottleneck</span>' : ""].join("");
  document.getElementById("detail").innerHTML =
    `<h2>Node detail</h2><strong>${n.id}</strong> ${tags}${rows}`;
}
const a = DATA.analysis;
const pct = v => (100*v).toFixed(1) + "%";
document.getElementById("analysis").innerHTML = `<h2>Analysis</h2>
  <div>Resolving coverage: <strong>${pct(a.resolving_coverage ?? 0)}</strong>
  (dropout: ${pct(a.fault_tolerant_coverage ?? 0)})</div>
  <div>Constrained routing savings: <strong>${pct(a.constrained_savings ?? 0)}</strong></div>
  <div>Verifiers: ${(a.verifier_placement||[])
    .map(v=>`<span class="badge verifier">${v}</span>`).join("") || "—"}</div>
  <div>Seeds: ${(a.recommended_seeds||[])
    .map(v=>`<span class="badge seed">${v}</span>`).join("") || "—"}</div>`;
const traceDiv = document.getElementById("trace");
if (DATA.trace.length) {
  traceDiv.innerHTML = "<h2>Decision trace</h2>" + DATA.trace.map(r => {
    const kind = r.row_type || r.type || "event";
    const p = r.payload || r;
    const decision = p.decision || p.action || "";
    const cls = decision ? "trace-row decision" : "trace-row";
    const extras = ["node_id","reason","tokens","cost"].filter(k => p[k] !== undefined)
      .map(k => `${k}=${p[k]}`).join(" · ");
    return `<div class="${cls}"><strong>${kind}</strong> ${decision}` +
      ` <span style="color:var(--dim)">${extras}</span></div>`;
  }).join("");
} else {
  traceDiv.innerHTML = "<h2>Decision trace</h2><div class=\\"legend\\">No trace loaded. " +
    "Pass --trace path/to/trace.jsonl to overlay control decisions.</div>";
}
</script>
</body>
</html>
"""


def render_workflow_view(
    graph: AgentGraph,
    *,
    title: str = "workflow",
    analysis: dict[str, Any] | None = None,
    trace_rows: list[dict[str, Any]] | None = None,
) -> str:
    """Render a single self-contained interactive HTML page for ``graph``."""

    data = {
        "nodes": [node.to_dict() for node in graph.nodes()],
        "edges": [edge.to_dict() for edge in graph.edges()],
        "analysis": analysis or {},
        "trace": trace_rows or [],
    }
    payload = json.dumps(data).replace("</", "<\\/")
    return (
        _PAGE_TEMPLATE.replace("__TITLE__", title)
        .replace("__NODES__", str(graph.node_count))
        .replace("__EDGES__", str(graph.edge_count))
        .replace("__DATA__", payload)
    )


def load_trace_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL trace rows, skipping lines that fail to parse."""

    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def write_workflow_view(
    graph: AgentGraph,
    out_path: str | Path,
    *,
    title: str = "workflow",
    analysis: dict[str, Any] | None = None,
    trace_rows: list[dict[str, Any]] | None = None,
) -> Path:
    """Write the interactive view to ``out_path`` and return the path."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html = render_workflow_view(graph, title=title, analysis=analysis, trace_rows=trace_rows)
    path.write_text(html)
    return path
