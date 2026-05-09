const PARTICLE_COUNT = 300;
const SPREAD = 600;
const ARENA_SLUG = 'vg-art';
const CELL = 128;

// Picking state — original (unsorted) arrays, populated in buildScene
let arenaBlocks     = [];
let texIndicesArray = [];
let positionsArray  = null;
let sizesArray      = null;
let pointsMesh      = null;

// --- Scene setup ---

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 5000);
camera.position.set(0, 0, 900);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(0xF4F4F4);
document.body.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.rotateSpeed = 0.5;
controls.zoomSpeed = 0.8;
controls.panSpeed = 0.6;
controls.minDistance = 100;
controls.maxDistance = 2000;

window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// --- Are.na fetch ---

async function fetchArenaBlocks() {
  const pages = await Promise.all([1, 2, 3, 4].map(page =>
    fetch(`https://api.are.na/v2/channels/${ARENA_SLUG}/contents?per=100&page=${page}`)
      .then(r => r.json())
      .catch(() => ({ contents: [] }))
  ));
  const all = pages.flatMap(d => d.contents || []).filter(b => b.class === 'Image' && b.image);
  return all.sort(() => Math.random() - 0.5);
}

// --- Image loading ---

function loadImage(url) {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload  = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

function setProgress(pct) {
  document.getElementById('loading-bar').style.width = pct + '%';
}

// --- Atlas builder ---

function buildAtlas(images) {
  const count = images.length;
  const cols  = Math.ceil(Math.sqrt(count));
  const rows  = Math.ceil(count / cols);

  const atlasCanvas = document.createElement('canvas');
  atlasCanvas.width  = CELL * cols;
  atlasCanvas.height = CELL * rows;
  const ctx = atlasCanvas.getContext('2d');

  images.forEach((img, i) => {
    if (!img) return;
    const col = i % cols;
    const row = Math.floor(i / cols);
    const scale = Math.max(CELL / img.width, CELL / img.height);
    const sw = CELL / scale, sh = CELL / scale;
    const sx = (img.width - sw) / 2, sy = (img.height - sh) / 2;
    ctx.drawImage(img, sx, sy, sw, sh, col * CELL, row * CELL, CELL, CELL);
  });

  const tex = new THREE.CanvasTexture(atlasCanvas);
  tex.needsUpdate = true;
  return { tex, cols, rows };
}

// --- Depth sorting ---
// Reorders the GPU draw buffers back-to-front each frame so the painter's
// algorithm draws far particles first and near ones on top.

const _meshInvMatrix  = new THREE.Matrix4();
const _camLocal       = new THREE.Vector3();

function makeSorter(posAttr, sizeAttr, texAttr, origTexFloat) {
  const depths    = new Float32Array(PARTICLE_COUNT);
  const sortOrder = Array.from({ length: PARTICLE_COUNT }, (_, i) => i);
  const tmpPos    = new Float32Array(PARTICLE_COUNT * 3);
  const tmpSize   = new Float32Array(PARTICLE_COUNT);
  const tmpTex    = new Float32Array(PARTICLE_COUNT);

  return function sort() {
    pointsMesh.updateMatrixWorld();
    _meshInvMatrix.copy(pointsMesh.matrixWorld).invert();
    _camLocal.copy(camera.position).applyMatrix4(_meshInvMatrix);

    const cx = _camLocal.x, cy = _camLocal.y, cz = _camLocal.z;

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const dx = positionsArray[i * 3]     - cx;
      const dy = positionsArray[i * 3 + 1] - cy;
      const dz = positionsArray[i * 3 + 2] - cz;
      depths[i]    = dx * dx + dy * dy + dz * dz;
      sortOrder[i] = i;
    }

    // Far → near so nearest particles are drawn last (on top)
    sortOrder.sort((a, b) => depths[b] - depths[a]);

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const j = sortOrder[i];
      tmpPos[i * 3]     = positionsArray[j * 3];
      tmpPos[i * 3 + 1] = positionsArray[j * 3 + 1];
      tmpPos[i * 3 + 2] = positionsArray[j * 3 + 2];
      tmpSize[i] = sizesArray[j];
      tmpTex[i]  = origTexFloat[j];
    }

    posAttr.array.set(tmpPos);  posAttr.needsUpdate  = true;
    sizeAttr.array.set(tmpSize); sizeAttr.needsUpdate = true;
    texAttr.array.set(tmpTex);   texAttr.needsUpdate  = true;
  };
}

// --- Particle picking ---
// Uses the original (unsorted) positionsArray so picking is unaffected by draw order.

const _localPos = new THREE.Vector3();
const _worldPos = new THREE.Vector3();
const _viewPos  = new THREE.Vector3();

function findParticleAt(event) {
  if (!pointsMesh) return -1;

  const rect = renderer.domElement.getBoundingClientRect();
  const mx = event.clientX - rect.left;
  const my = event.clientY - rect.top;
  const w  = rect.width;
  const h  = rect.height;
  const dpr = renderer.getPixelRatio();

  let bestIdx  = -1;
  let bestDist = Infinity;

  pointsMesh.updateMatrixWorld();

  for (let i = 0; i < PARTICLE_COUNT; i++) {
    _localPos.set(positionsArray[i * 3], positionsArray[i * 3 + 1], positionsArray[i * 3 + 2]);
    _worldPos.copy(_localPos).applyMatrix4(pointsMesh.matrixWorld);

    const proj = _worldPos.clone().project(camera);
    if (proj.z > 1) continue;

    const sx = (proj.x + 1) / 2 * w;
    const sy = (-proj.y + 1) / 2 * h;

    _viewPos.copy(_worldPos).applyMatrix4(camera.matrixWorldInverse);
    const depth = -_viewPos.z;
    if (depth <= 0) continue;

    const cssRadius = (sizesArray[i] * 400.0 / depth) / 2 / dpr;

    const dx = mx - sx, dy = my - sy;
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (dist <= cssRadius && dist < bestDist) {
      bestDist = dist;
      bestIdx  = i;
    }
  }

  return bestIdx;
}

// --- Lightbox ---

const lightboxEl       = document.getElementById('lightbox');
const lightboxImg      = document.getElementById('lightbox-img');
const lightboxClose    = document.getElementById('lightbox-close');
const lightboxBackdrop = document.getElementById('lightbox-backdrop');

function openLightbox(block) {
  lightboxImg.src = block.image.large?.url || block.image.original?.url;
  lightboxEl.classList.add('open');
}

function closeLightbox() {
  lightboxEl.classList.remove('open');
  lightboxImg.src = '';
}

lightboxClose.addEventListener('click', closeLightbox);
lightboxBackdrop.addEventListener('click', closeLightbox);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeLightbox(); });

// --- Interaction handlers ---

let hoverTick = 0;
renderer.domElement.addEventListener('mousemove', (e) => {
  const now = Date.now();
  if (now - hoverTick < 40) return;
  hoverTick = now;
  renderer.domElement.style.cursor = findParticleAt(e) !== -1 ? 'pointer' : 'default';
});

renderer.domElement.addEventListener('click', (e) => {
  const idx = findParticleAt(e);
  if (idx === -1) return;
  const block = arenaBlocks[texIndicesArray[idx]];
  if (block) openLightbox(block);
});

// --- Scene builder ---

function buildScene(vertexShader, fragmentShader, atlasTex, atlasCols, atlasRows, blocks) {
  arenaBlocks = blocks;

  // Source arrays — never reordered, used for picking and as sort source
  positionsArray  = new Float32Array(PARTICLE_COUNT * 3);
  sizesArray      = new Float32Array(PARTICLE_COUNT);
  texIndicesArray = new Array(PARTICLE_COUNT);

  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const theta = Math.random() * Math.PI * 2;
    const phi   = Math.acos(2 * Math.random() - 1);
    const r     = Math.cbrt(Math.random()) * SPREAD;

    positionsArray[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
    positionsArray[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    positionsArray[i * 3 + 2] = r * Math.cos(phi);

    sizesArray[i]      = Math.random() * 60 + 30;
    texIndicesArray[i] = Math.floor(Math.random() * blocks.length);
  }

  const origTexFloat = new Float32Array(texIndicesArray);

  // GPU buffers — dynamic, rewritten each frame by the sorter
  const posAttr  = new THREE.BufferAttribute(new Float32Array(PARTICLE_COUNT * 3), 3);
  const sizeAttr = new THREE.BufferAttribute(new Float32Array(PARTICLE_COUNT), 1);
  const texAttr  = new THREE.BufferAttribute(new Float32Array(PARTICLE_COUNT), 1);
  posAttr.setUsage(THREE.DynamicDrawUsage);
  sizeAttr.setUsage(THREE.DynamicDrawUsage);
  texAttr.setUsage(THREE.DynamicDrawUsage);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position',  posAttr);
  geometry.setAttribute('aSize',     sizeAttr);
  geometry.setAttribute('aTexIndex', texAttr);

  const material = new THREE.ShaderMaterial({
    uniforms: {
      uAtlas:     { value: atlasTex },
      uAtlasCols: { value: atlasCols },
      uAtlasRows: { value: atlasRows },
    },
    vertexShader,
    fragmentShader,
    transparent: true,
    depthWrite: false,
  });

  pointsMesh = new THREE.Points(geometry, material);
  scene.add(pointsMesh);

  const sortParticles = makeSorter(posAttr, sizeAttr, texAttr, origTexFloat);
  sortParticles(); // initial sort before first frame

  const loadingEl = document.getElementById('loading');
  gsap.to(loadingEl, {
    opacity: 0,
    duration: 0.8,
    delay: 0.3,
    onComplete: () => { loadingEl.style.display = 'none'; }
  });

  let userActive = false;
  let idleTimer;
  renderer.domElement.addEventListener('pointerdown', () => {
    userActive = true;
    clearTimeout(idleTimer);
  });
  renderer.domElement.addEventListener('pointerup', () => {
    idleTimer = setTimeout(() => { userActive = false; }, 3000);
  });

  let t = 0;
  function animate() {
    requestAnimationFrame(animate);
    t += 0.0005;
    if (!userActive) {
      pointsMesh.rotation.y += 0.0008;
      pointsMesh.rotation.x = Math.sin(t) * 0.08;
    }
    controls.update();
    sortParticles(); // re-sort every frame to maintain correct draw order
    renderer.render(scene, camera);
  }
  animate();
}

// --- Boot ---

async function boot() {
  const [vertexShader, fragmentShader, shuffled] = await Promise.all([
    fetch('shaders/vertex.glsl').then(r => r.text()),
    fetch('shaders/fragment.glsl').then(r => r.text()),
    fetchArenaBlocks(),
  ]);

  let done = 0;
  const images = await Promise.all(
    shuffled.map(async block => {
      const url = block.image.large?.url || block.image.original?.url;
      const img = await loadImage(url);
      setProgress(++done / shuffled.length * 100);
      return img;
    })
  );

  const validBlocks = [];
  const validImages = [];
  shuffled.forEach((block, i) => {
    if (images[i]) { validBlocks.push(block); validImages.push(images[i]); }
  });

  const { tex, cols, rows } = buildAtlas(validImages);
  buildScene(vertexShader, fragmentShader, tex, cols, rows, validBlocks);
}

boot().catch(err => {
  console.error('Boot failed:', err);
  document.getElementById('loading-text').textContent = 'Failed to load';
});
