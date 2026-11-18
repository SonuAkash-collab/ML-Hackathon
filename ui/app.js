// PathSnap evaluator UI (Laptop 2).
// Talks to the FastAPI backend on Laptop 1. The UI only ever sends a trip ID;
// it never sees reference routes, only the accuracy the API computes.

const $ = (id) => document.getElementById(id);
const COLORS = { gps: "#1f6fd1", route: "#d62728", matched: "#7a1b8c", roads: "#9a9a9a" };

let apiBase = "";
let layers = {};
let layerControl = null;

// ---------------------------------------------------------------- map
const map = L.map("map", { preferCanvas: true }).setView([49.87, 8.65], 12);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

const legend = L.control({ position: "bottomleft" });
legend.onAdd = () => {
  const div = L.DomUtil.create("div", "legend");
  div.innerHTML =
    `<i class="dot" style="background:${COLORS.gps}"></i>GPS observations<br>` +
    `<i style="background:${COLORS.route}"></i>Predicted route (HMM)<br>` +
    `<i class="dot" style="background:${COLORS.matched}"></i>Matched points<br>` +
    `<i style="background:${COLORS.roads}"></i>Candidate roads`;
  return div;
};
legend.addTo(map);

// ---------------------------------------------------------------- helpers
function savedApi() {
  const fromQuery = new URLSearchParams(location.search).get("api");
  if (fromQuery) return fromQuery;
  try { return localStorage.getItem("pathsnap_api") || "http://localhost:8000"; }
  catch (e) { return "http://localhost:8000"; }
}

function remember(url) {
  try { localStorage.setItem("pathsnap_api", url); } catch (e) { /* storage not available */ }
}

function logLine(text, isError = false) {
  const li = document.createElement("li");
  const t = new Date().toLocaleTimeString();
  li.textContent = `[${t}] ${text}`;
  if (isError) li.className = "err";
  $("log").prepend(li);
}

function pct(x) { return x == null ? "–" : (100 * x).toFixed(2) + "%"; }
function num(x, d = 3) { return x == null ? "–" : Number(x).toFixed(d); }

function setStatus(text, cls) {
  const s = $("apiStatus");
  s.textContent = text;
  s.className = "status " + cls;
}

async function api(path, options = {}) {
  const res = await fetch(apiBase + path, {
    headers: { "Content-Type": "application/json" }, ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* not json */ }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json();
}

// ---------------------------------------------------------------- actions
async function connect() {
  apiBase = $("apiUrl").value.trim().replace(/\/+$/, "");
  remember(apiBase);
  setStatus("connecting…", "idle");
  try {
    const h = await api("/health");
    setStatus(`connected · ${h.model}`, "ok");
    logLine(`GET /health → ok (model ${h.model}, ${h.n_trips} trips)`);
    const { trips } = await api("/trips");
    logLine(`GET /trips → ${trips.length} trips`);
    renderTrips(trips);
  } catch (e) {
    setStatus("unreachable", "err");
    logLine(`cannot reach ${apiBase}: ${e.message}`, true);
    $("tripList").innerHTML = `<p class="muted">Could not reach the API. Check the address, that both laptops are on the same network and that port 8000 is allowed through the firewall.</p>`;
  }
}

function renderTrips(trips) {
  const box = $("tripList");
  box.innerHTML = "";
  if (!trips.length) {
    box.innerHTML = `<p class="muted">The API has no trips in its evaluation folder.</p>`;
    return;
  }
  trips.forEach((t) => {
    const b = document.createElement("button");
    b.className = "trip";
    b.dataset.trip = t.trip_id;
    b.innerHTML = `<b>${t.trip_id}</b><span>${t.n_observations.toLocaleString()} GPS points</span>`;
    b.onclick = () => runTrip(t.trip_id);
    box.appendChild(b);
  });
}

async function runTrip(tripId) {
  document.querySelectorAll(".trip").forEach((b) => {
    b.classList.toggle("active", b.dataset.trip === tripId);
    b.disabled = true;
  });
  $("busy").classList.remove("hidden");
  logLine(`POST /predict {trip_id: ${tripId}}`);
  const t0 = performance.now();
  try {
    const r = await api("/predict", { method: "POST", body: JSON.stringify({ trip_id: tripId }) });
    const ms = performance.now() - t0;
    logLine(`← ${tripId}: ${r.n_observations} obs, HMM acc ${pct(r.metrics && r.metrics.accuracy_hmm)} (${(ms / 1000).toFixed(1)} s)`);
    draw(r);
    showMetrics(r, ms);
  } catch (e) {
    logLine(`predict ${tripId} failed: ${e.message}`, true);
  } finally {
    $("busy").classList.add("hidden");
    document.querySelectorAll(".trip").forEach((b) => (b.disabled = false));
  }
}

function draw(r) {
  Object.values(layers).forEach((l) => map.removeLayer(l));
  if (layerControl) map.removeControl(layerControl);

  const roads = L.layerGroup(
    r.candidate_roads.map((seg) => L.polyline(seg, { color: COLORS.roads, weight: 2, opacity: 0.55 }))
  );
  const gps = L.layerGroup(
    r.gps.map((p, i) => L.circleMarker([p[0], p[1]], {
      radius: 2.5, color: COLORS.gps, weight: 0, fillOpacity: 0.55,
    }).bindTooltip(`GPS #${i} · t=${p[2]}s`))
  );
  const route = L.polyline(r.predicted_route, { color: COLORS.route, weight: 4, opacity: 0.9 });
  const matched = L.layerGroup(
    r.matched_points.map((p) => L.circleMarker(p, { radius: 2, color: COLORS.matched, weight: 0, fillOpacity: 0.9 }))
  );
  const start = L.circleMarker(r.gps[0], { radius: 7, color: "#fff", weight: 2, fillColor: "#1a9e3a", fillOpacity: 1 }).bindTooltip("start");
  const end = L.circleMarker(r.gps[r.gps.length - 1], { radius: 7, color: "#fff", weight: 2, fillColor: "#000", fillOpacity: 1 }).bindTooltip("end");

  layers = { roads, gps, route, matched, start, end };
  roads.addTo(map); gps.addTo(map); route.addTo(map); start.addTo(map); end.addTo(map);
  layerControl = L.control.layers(null, {
    "Candidate roads": roads, "GPS observations": gps, "Predicted route (HMM)": route,
    "Matched points": matched,
  }, { collapsed: false }).addTo(map);
  if (r.predicted_route.length) map.fitBounds(route.getBounds(), { padding: [30, 30] });
  else if (r.gps.length) map.fitBounds(L.latLngBounds(r.gps.map((p) => [p[0], p[1]])), { padding: [30, 30] });
}

function showMetrics(r, ms) {
  $("resultCard").classList.remove("hidden");
  $("resTrip").textContent = r.trip_id;
  const m = r.metrics || {};
  $("mAccHmm").textContent = pct(m.accuracy_hmm);
  $("mAccClf").textContent = pct(m.accuracy_classifier);
  const rows = [
    ["GPS observations", r.n_observations.toLocaleString()],
    ["Observations with a prediction", r.n_predicted.toLocaleString()],
    ["Scored observations", m.n_labelled != null ? m.n_labelled.toLocaleString() : "–"],
    ["Route continuity (HMM)", num(m.continuity_hmm)],
    ["Route continuity (classifier)", num(m.continuity_classifier)],
    ["Route precision", num(m.route_precision_hmm)],
    ["Route recall", num(m.route_recall_hmm)],
    ["Route F1", num(m.route_f1_hmm)],
    ["Candidate miss rate", pct(m.candidate_miss_rate)],
    ["Viterbi segments", r.decoder ? r.decoder.n_segments : "–"],
    ["Server pipeline time", num(r.timings_s.total_s, 2) + " s"],
    ["Round trip time", (ms / 1000).toFixed(2) + " s"],
    ["Model", `${r.model.name} (${r.model.variant})`],
  ];
  $("metricRows").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  if (!r.metrics) {
    $("metricRows").innerHTML += `<tr><td colspan="2" class="muted">No reference route on the server for this trip, so accuracy was not computed.</td></tr>`;
  }
}

// ---------------------------------------------------------------- init
$("apiUrl").value = savedApi();
$("connectBtn").onclick = connect;
$("apiUrl").addEventListener("keydown", (e) => { if (e.key === "Enter") connect(); });
connect();
