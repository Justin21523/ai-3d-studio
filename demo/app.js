const artifacts = [
  ["source.png", "Input object image", "linear-gradient(90deg,#26d0a3,#2e6ee6)"],
  ["demo_asset.glb", "Generated textured mesh", "linear-gradient(90deg,#2e6ee6,#5138a8)"],
  ["demo_asset.obj", "Format conversion output", "linear-gradient(90deg,#d44b3e,#f0bd4c)"],
  ["depth/0001.png", "Depth conditioning pass", "linear-gradient(90deg,#111827,#64748b)"],
  ["normal/0001.png", "Normal conditioning pass", "linear-gradient(90deg,#1f9d75,#c9891d)"],
  ["pipeline_manifest.yaml", "Stage-level provenance", "linear-gradient(90deg,#2563eb,#1f9d75)"],
];

const scenarios = {
  product: {
    label: "Product Turntable",
    preview: "demo_asset.glb",
    command: "ai3d run-pipeline --backend demo --input samples/product.png --output-dir outputs/product --no-blender --no-video-pack",
  },
  character: {
    label: "Character Bust",
    preview: "character_demo.glb",
    command: "ai3d run-backend --backend demo --input samples/character.png --output-dir outputs/character --output-type glb --output-type obj",
  },
  prop: {
    label: "Game Prop Concept",
    preview: "prop_demo.glb",
    command: "ai3d run-pipeline --backend demo --input samples/prop.png --output-dir outputs/prop --no-blender --no-video-pack",
  },
};

const artifactGrid = document.querySelector("#artifact-grid");
const runButton = document.querySelector("#run-button");
const scenarioSelect = document.querySelector("#scenario-select");
const backendSelect = document.querySelector("#backend-select");
const scenarioLabel = document.querySelector("#scenario-label");
const cliCommand = document.querySelector("#cli-command");
const previewLabel = document.querySelector("#preview-label");
const runtimeStatus = document.querySelector("#runtime-status");
const smokeStatus = document.querySelector("#smoke-status");
const artifactCount = document.querySelector("#artifact-count");
const runId = document.querySelector("#run-id");

function renderArtifacts() {
  artifactGrid.innerHTML = artifacts
    .map(
      ([name, label, color]) => `
        <article class="artifact-card">
          <div>
            <div class="swatch" style="background:${color}"></div>
            <strong>${name}</strong>
          </div>
          <span>${label}</span>
        </article>
      `
    )
    .join("");
}

function updateScenario() {
  const scenario = scenarios[scenarioSelect.value];
  scenarioLabel.textContent = scenario.label;
  previewLabel.textContent = scenario.preview;
  cliCommand.textContent = scenario.command;
}

function setStageState(index, state) {
  const stage = document.querySelectorAll(".stage")[index];
  stage.classList.remove("complete", "active", "queued");
  stage.classList.add(state);
}

function runDemo() {
  runButton.disabled = true;
  runButton.textContent = "Running pipeline...";
  runtimeStatus.textContent = "Pipeline running";
  smokeStatus.textContent = "Running";
  artifactCount.textContent = "0 ready";
  runId.textContent = `demo-${Math.floor(Date.now() / 1000).toString().slice(-6)}`;

  document.querySelectorAll(".stage").forEach((stage, index) => {
    stage.classList.remove("complete", "active", "queued");
    stage.classList.add(index === 0 ? "active" : "queued");
  });

  let step = 0;
  const timer = setInterval(() => {
    setStageState(step, "complete");
    step += 1;
    if (step < 6) {
      setStageState(step, "active");
      artifactCount.textContent = `${Math.max(1, step)} ready`;
      return;
    }
    clearInterval(timer);
    artifactCount.textContent = "6 ready";
    runtimeStatus.textContent = "Mock-safe run complete";
    smokeStatus.textContent = "Passed";
    runButton.disabled = false;
    runButton.textContent = "Run mock-safe pipeline";
  }, 360);
}

scenarioSelect.addEventListener("change", updateScenario);
backendSelect.addEventListener("change", () => {
  const mode = backendSelect.value === "demo" ? "Mock-safe mode ready" : "Real backend requires local services";
  runtimeStatus.textContent = mode;
});
runButton.addEventListener("click", runDemo);

renderArtifacts();
updateScenario();
