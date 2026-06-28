import "./styles.css";

const project = {
  "slug": "ai-3d-studio",
  "title": "AI 3D Studio — A Unified Multi-Backend Image-to-3D Generation and Blender Integration Framework",
  "summary": "A production-oriented, open-source Python framework that unifies multiple image-to-3D models (TripoSR, SF3D, TRELLIS, Hunyuan3D, and more) behind one interface, then chains background removal, mesh cleanup, UV unwrapping, format conversion, headless Blender turntable rendering and ComfyUI video workflows into a complete 11-stage pipeline from raw image to a video-ready conditioning pack.",
  "category": "ai-data",
  "year": 2026,
  "status": "prototype",
  "technologies": [
    "Python",
    "PyTorch",
    "Pydantic v2",
    "trimesh",
    "rembg",
    "xatlas",
    "Blender (headless/bpy)",
    "ComfyUI",
    "FastAPI",
    "pytest"
  ],
  "githubUrl": "https://github.com/Justin21523/ai-3d-studio",
  "readmeUrl": "https://github.com/Justin21523/ai-3d-studio#readme",
  "problem": "Off-the-shelf image-to-3D models (TripoSR, SF3D, TRELLIS, Hunyuan3D, InstantMesh, CRM, and others) each come with different environment requirements, model-loading conventions, and output formats. Using them in a real project means rewriting per-backend boilerplate for setup, inference calls, format conversion, and downstream integration, with no consistent interface and no easy path to batch or reproducible asset generation.",
  "solution": "A backend registry built around an abstract base class, BaseBackend: every 3D model implements just four methods (generate, check_availability, estimate_requirements, export_metadata), so adding a backend means adding one file. The entire data flow is typed with Pydantic v2 and persisted as YAML, and all filesystem paths are externalized to configs/paths.yaml. Preprocessing (rembg background removal, quality checks), postprocessing (trimesh mesh cleanup, xatlas UV unwrapping, GLB/OBJ/FBX/PLY conversion), headless Blender turntable plus depth/normal/mask render passes, video conditioning pack export, and a pure-HTTP ComfyUI client are all wired together by an 11-stage GenerationPipeline that writes a manifest after each stage for reproducibility and resume support.",
  "architecture": "This case study is generated from the portfolio catalog pipeline using README, Git metadata, package/build configuration, and media signals. The final architecture narrative still needs source-level review. Current detected technology signals include: Python, PyTorch, Pydantic v2, trimesh, rembg, xatlas, Blender (headless/bpy), ComfyUI, FastAPI, pytest.",
  "setupGuide": "This project does not expose a verified runnable web command yet. Review the README/source tree and add exact install, run, test, and build commands before interview use.\nNo verified build command was detected. Treat the current portfolio page as a case-study placeholder until build steps are reviewed.",
  "features": [
    "Detected technical signals: Python, PyTorch, Pydantic v2, trimesh, rembg, xatlas, Blender (headless/bpy), ComfyUI, FastAPI, pytest,README evidence exists and can support a fuller reviewed case study,A public GitHub repository is linked for source traceability",
    "A backend registry built around an abstract base class, BaseBackend: every 3D model implements just four methods (generate, check_availability, estimate_requirements, export_metadata), so adding a backend means adding one file",
    "The entire data flow is typed with Pydantic v2 and persisted as YAML, and all filesystem paths are externalized to configs/paths",
    "Preprocessing (rembg background removal, quality checks), postprocessing (trimesh mesh cleanup, xatlas UV unwrapping, GLB/OBJ/FBX/PLY conversion), headless Blender turntable plus depth/normal/mask render passes, video conditioning pack export, and a pure-HTTP ComfyUI client are all wired together by an 11-stage GenerationPipeline that writes a manifest after each stage for reproducibility and resume support"
  ],
  "metrics": [
    {
      "label": "Demo Modules",
      "value": "4"
    },
    {
      "label": "Tech Stack",
      "value": "10"
    },
    {
      "label": "Mode",
      "value": "Fixture"
    },
    {
      "label": "Status",
      "value": "prototype"
    }
  ],
  "records": [
    {
      "id": "flow-01",
      "name": "Detected technical signals: Python, PyTorch, Pydantic v2, trimesh, rembg, xatlas, Blender (headless/bpy), ComfyUI, FastAPI, pytest,README evidence exists and can support a fuller reviewed case study,A public GitHub repository is linked for source traceability",
      "status": "Ready",
      "owner": "Frontend"
    },
    {
      "id": "flow-02",
      "name": "A backend registry built around an abstract base class, BaseBackend: every 3D model implements just four methods (generate, check_availability, estimate_requirements, export_metadata), so adding a backend means adding one file",
      "status": "Review",
      "owner": "Data"
    },
    {
      "id": "flow-03",
      "name": "The entire data flow is typed with Pydantic v2 and persisted as YAML, and all filesystem paths are externalized to configs/paths",
      "status": "Queued",
      "owner": "Automation"
    },
    {
      "id": "flow-04",
      "name": "Preprocessing (rembg background removal, quality checks), postprocessing (trimesh mesh cleanup, xatlas UV unwrapping, GLB/OBJ/FBX/PLY conversion), headless Blender turntable plus depth/normal/mask render passes, video conditioning pack export, and a pure-HTTP ComfyUI client are all wired together by an 11-stage GenerationPipeline that writes a manifest after each stage for reproducibility and resume support",
      "status": "Ready",
      "owner": "Product"
    }
  ]
};

const state = {
  tab: "overview",
  query: "",
  selected: project.records[0]?.id ?? "",
};

function matches(record) {
  const q = state.query.trim().toLowerCase();
  if (!q) return true;
  return [record.name, record.status, record.owner].join(" ").toLowerCase().includes(q);
}

function renderMetrics() {
  return project.metrics.map((metric) => `
    <div class="metric">
      <span>${metric.label}</span>
      <strong>${metric.value}</strong>
    </div>
  `).join("");
}

function renderTabs() {
  return ["overview", "workflow", "data", "architecture"].map((tab) => `
    <button class="tab ${state.tab === tab ? "active" : ""}" data-tab="${tab}">${tab}</button>
  `).join("");
}

function renderOverview() {
  return `
    <section class="panel hero-panel">
      <div>
        <p class="eyebrow">${project.category} · ${project.year}</p>
        <h1>${project.title}</h1>
        <p class="lead">${project.summary}</p>
      </div>
      <div class="metrics">${renderMetrics()}</div>
    </section>
    <section class="panel split">
      <div>
        <h2>Problem</h2>
        <p>${project.problem}</p>
      </div>
      <div>
        <h2>Solution</h2>
        <p>${project.solution}</p>
      </div>
    </section>
  `;
}

function renderWorkflow() {
  return `
    <section class="panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">Demo workflow</p>
          <h2>Interactive Review Flow</h2>
        </div>
        <button id="runDemo" class="primary">Run demo pass</button>
      </div>
      <div class="timeline">
        ${project.features.map((feature, index) => `
          <article class="step">
            <span>${String(index + 1).padStart(2, "0")}</span>
            <p>${feature}</p>
          </article>
        `).join("")}
      </div>
      <output id="demoOutput" class="output">Ready to run the guided demo.</output>
    </section>
  `;
}

function renderData() {
  const rows = project.records.filter(matches);
  return `
    <section class="panel">
      <div class="section-head">
        <div>
          <p class="eyebrow">Fixture data</p>
          <h2>Sample Records</h2>
        </div>
        <input id="search" value="${state.query}" placeholder="Filter records" />
      </div>
      <div class="table">
        ${rows.map((record) => `
          <button class="row ${state.selected === record.id ? "selected" : ""}" data-record="${record.id}">
            <span>${record.id}</span>
            <strong>${record.name}</strong>
            <em>${record.owner}</em>
            <b>${record.status}</b>
          </button>
        `).join("") || `<p class="empty">No records match this filter.</p>`}
      </div>
    </section>
  `;
}

function renderArchitecture() {
  return `
    <section class="panel split">
      <div>
        <p class="eyebrow">Architecture</p>
        <h2>How the demo is organized</h2>
        <p>${project.architecture}</p>
        <pre>demo-app/
  src/main.js
  src/styles.css
  index.html
  package.json</pre>
      </div>
      <div>
        <p class="eyebrow">Run guide</p>
        <h2>Local commands</h2>
        <pre>${project.setupGuide}</pre>
        <div class="chips">${project.technologies.slice(0, 12).map((tech) => `<span>${tech}</span>`).join("")}</div>
      </div>
    </section>
  `;
}

function render() {
  const views = {
    overview: renderOverview,
    workflow: renderWorkflow,
    data: renderData,
    architecture: renderArchitecture,
  };
  document.querySelector("#app").innerHTML = `
    <header class="topbar">
      <a href="${project.githubUrl}" class="brand">${project.title}</a>
      <nav>${renderTabs()}</nav>
      <a class="repo" href="${project.readmeUrl}">README</a>
    </header>
    <main>${views[state.tab]()}</main>
  `;

  document.querySelectorAll("[data-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      state.tab = button.dataset.tab;
      render();
    });
  });
  document.querySelector("#search")?.addEventListener("input", (event) => {
    state.query = event.target.value;
    render();
    document.querySelector("#search")?.focus();
  });
  document.querySelectorAll("[data-record]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selected = button.dataset.record;
      render();
    });
  });
  document.querySelector("#runDemo")?.addEventListener("click", () => {
    const output = document.querySelector("#demoOutput");
    if (output) output.textContent = `${project.title}: ${project.records.length} fixture records processed and ${project.features.length} workflow checks completed.`;
  });
}

render();
