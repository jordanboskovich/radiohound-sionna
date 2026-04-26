import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver
import numpy as np
import os

# ── Constants ──────────────────────────────────────────────────────────────────
SCENE_PATH = "simple_scene/NOTRE_DAME_3_563481.0550282092_4616582.2273852/simple_OSM_scene.xml"
FREQUENCY     = 2.4e9
TX_POWER_DBM  = 20
THRESHOLD_DBM = -90
MAX_DEPTH     = 4
NUM_SAMPLES   = int(1e6)
RESOLUTION    = 1

# Grid bounds (meters, centered at scene origin)
X_MIN, X_MAX = -105, 105
Y_MIN, Y_MAX = -164, 164

# Sensor grid
N_SENSORS_X = 32
N_SENSORS_Y = 64  # 32x64 = 2048

# Output directory — change for each job
OUTPUT_DIR = f"dataset_v3/dataset_1m"

# ── Setup ──────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Working directory: {os.getcwd()}")
print(f"Output directory: {OUTPUT_DIR}")

# Receiver grid
x = np.arange(X_MIN, X_MAX, RESOLUTION)
y = np.arange(Y_MIN, Y_MAX, RESOLUTION)
X, Y = np.meshgrid(x, y)
grid_points = np.stack([X.flatten(), Y.flatten(), np.zeros(X.size)], axis=1)
print(f"Receiver grid: {len(x)} x {len(y)} = {len(grid_points)} points at {RESOLUTION}m resolution")

# Sensor positions
sensor_x = np.linspace(X_MIN * 0.8, X_MAX * 0.8, N_SENSORS_X)
sensor_y = np.linspace(Y_MIN * 0.8, Y_MAX * 0.8, N_SENSORS_Y)
SX, SY = np.meshgrid(sensor_x, sensor_y)
sensor_positions = [[float(sx), float(sy), 10.0] for sx, sy in zip(SX.flatten(), SY.flatten())]
print(f"Total sensor positions: {len(sensor_positions)}")

# ── Resume logic ───────────────────────────────────────────────────────────────
completed_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npz')]
completed = len(completed_files)
print(f"Already completed: {completed}/{len(sensor_positions)}")

# ── Antenna ────────────────────────────────────────────────────────────────────
array = PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V"
)

# ── Load scene ─────────────────────────────────────────────────────────────────
print("Loading scene...")
scene = load_scene(SCENE_PATH)
scene.frequency = FREQUENCY
scene.synthetic_array = False
scene.tx_array = array
scene.rx_array = array

# Add all receivers once
print("Adding receivers...")
for i, pos in enumerate(grid_points):
    rx = Receiver(name=f"rx_{i}", position=pos.tolist(), orientation=[0, 0, 0])
    rx.array = array
    scene.add(rx)
print(f"Added {len(grid_points)} receivers")

# ── Main loop ──────────────────────────────────────────────────────────────────
paths_solver = PathSolver()

for idx, sensor_pos in enumerate(sensor_positions):
    save_path = os.path.join(OUTPUT_DIR, f"sample_{idx:04d}.npz")

    # Skip if already done
    if os.path.exists(save_path):
        print(f"[{idx+1}/{len(sensor_positions)}] Skipping (already done)")
        continue

    print(f"[{idx+1}/{len(sensor_positions)}] Sensor at {sensor_pos}...")

    # Remove previous transmitter
    if "sensor" in scene.transmitters:
        scene.remove("sensor")

    # Place sensor
    tx = Transmitter(name="sensor", position=sensor_pos, orientation=[0, 0, 0])
    tx.array = array
    scene.add(tx)

    # Run simulation
    paths = paths_solver(scene, max_depth=MAX_DEPTH, samples_per_src=NUM_SAMPLES)

    # Compute RSS
    h_real = paths.a[0]
    h_imag = paths.a[1]
    path_power = h_real**2 + h_imag**2
    total_power = np.sum(path_power, axis=(3, 4))
    rss_linear = total_power[:, 0, 0]
    rss_dbm = 10 * np.log10(rss_linear + 1e-20) + TX_POWER_DBM
    rss_2d = rss_dbm.reshape(len(y), len(x))

    # Binary coverage map
    coverage = (rss_2d > THRESHOLD_DBM).astype(np.int8)

    n_covered = coverage.sum()
    print(f"    Coverage: {n_covered}/{len(grid_points)} cells ({100*n_covered/len(grid_points):.1f}%)")

    # Save as NPZ — one file per sample
    np.savez(save_path,
             sensor_pos=np.array(sensor_pos),
             rss_map=rss_2d,
             coverage_map=coverage)

print(f"\nDone. {len(os.listdir(OUTPUT_DIR))} samples saved to '{OUTPUT_DIR}/'")
