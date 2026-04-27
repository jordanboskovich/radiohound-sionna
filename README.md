# RadioHound Sionna Dataset Generation

**Author:** Jordan Boskovich  
**Lab:** RadioHound / Wireless Institute, University of Notre Dame  
**Semester:** Spring 2026

---

## Project Overview

The RadioHound project aims to deploy RF sensors around the Notre Dame campus to detect nearby emitters and map frequency usage across different bands.

This repository documents the first phase of a sensor placement optimization project. Given a limited number of RadioHound sensors, the goal is to find placement configurations that maximize spatial coverage — i.e., the area from which an emitter can be detected. This dataset is a stepping stone toward two things:

1. **Training a neural network** to predict optimal sensor placements given a coverage objective
2. **Validation against real-world data** by comparing simulated coverage maps against measurements collected by physically deploying RadioHound sensors on the Notre Dame Debartolo quad

The dataset consists of pairs of the form `(sensor_position, coverage_map)`, where the coverage map is a 2D binary grid indicating which areas of the scene a sensor at that position can detect an emitter from. Coverage maps are generated using **Sionna**, NVIDIA's differentiable ray-tracing library for RF simulation, over a 3D scene of the Notre Dame campus built from **OpenStreetMap** data.

---

## Repository Structure

```
sionna_osm_scene/
├── OSM_to_Sionna.ipynb         # Scene generation notebook
├── ND-Simulation.ipynb         # Single-sensor simulation and visualization
├── Dataset-Gen.ipynb           # Dataset generation (Jupyter version)
├── Dataset-Visualization.ipynb # Dataset visualization notebook
├── Location-Visualization.ipynb# Location/scene visualization notebook
├── dataset_v2/
│   ├── dataset_gen_3m.py       # Dataset generation script, 3m resolution
│   ├── dataset_gen_1m.py       # Dataset generation script, 1m resolution
│   ├── submit_3m.sh            # CRC job submission script, 3m
│   ├── submit_1m.sh            # CRC job submission script, 1m
│   └── Visualization.ipynb     # Jupyter Notebook to visualize sample coverage maps
├── dataset_v3/                 # Same structure as v2, roads removed from scene
├── simple_scene/
│   ├── NOTRE_DAME_2_.../       # Scene with roads (used for dataset_v2)
│   └── NOTRE_DAME_3_.../       # Scene without roads (used for dataset_v3)
└── environment.yml             # Conda environment
```

---

## Environment Setup

### Prerequisites

- Access to Notre Dame's CRC cluster
- Conda installed

### Install

```bash
conda env create -f environment.yml
conda activate sionna_env
```

The environment includes Sionna 1.2.1, TensorFlow 2.20.0, osmnx, pyvista, open3d, pyproj, and shapely. You must also set the following environment variables (already included in the job submission scripts):

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export DRJIT_LIBLLVM_PATH=$CONDA_PREFIX/lib/libLLVM-21.so
```

---

## Step 1: Scene Generation

The 3D scene is generated from OpenStreetMap data using the `OSM_to_Sionna.ipynb` notebook. The output is a Mitsuba-format XML scene file containing building and ground meshes, which Sionna loads for ray tracing.

### Bounding Box

The scene covers the following area of Notre Dame's campus (God Quad / DeBartolo area):

```python
west, east = -86.23842196322462, -86.23573245290231
south, north = 41.69701511660061, 41.69996986433866
```

This corresponds to approximately 210m × 328m. All scene coordinates are in meters, centered at the scene's centroid.

### Coordinate Reference System

All projections use **EPSG:26916** (UTM Zone 16N), which is correct for Notre Dame's location. The original notebook this was adapted from used EPSG:26915 (UTM Zone 15N, correct for the Philippines), which caused significant geometric distortion and was corrected.

### Building Heights

OSM building data rarely includes height information. The following manual overrides are applied:

```python
level_overrides = {
    "Notre Dame Stadium": 10,
    "Duncan Student Center": 6,
    "DeBartolo Hall": 3,
    "Cushing Hall of Engineering": 4,
    "Fitzpatrick Hall of Engineering": 6,
    "Nieuwland Science Hall": 4,
    "Riley Hall of Art and Design": 3,
    "Decio Faculty Hall": 3,
    "Malloy Hall": 3,
    "O'Shaughnessy Hall": 4,
    "Snite Museum": 2,
    "Stepan Chemistry Hall": 5,
}
```

Buildings without a name or override default to 14m (~4 stories).

### Roads

The original scene included road meshes generated from OSM road network data. These were extruded 0.25m above ground level. This caused a significant artifact: receiver points placed at z=0 fell below the road surface, causing Sionna's ray tracer to treat the road mesh as an obstacle. This produced unrealistic dead zones along every walkway in the scene.

**Fix:** Road meshes are excluded from the scene XML entirely. The road mesh files are still generated but not added to the scene. This is controlled in `OSM_to_Sionna.ipynb` — the `ET.SubElement` calls that add roads to the XML are commented out.

The `dataset_v2` dataset was generated **with** roads (affected by this artifact). The `dataset_v3` dataset was generated **without** roads and is the preferred dataset.

### Key API Changes from Original Notebook

The original notebook was written for an older version of osmnx. The following changes were required:

- `ox.geometries.geometries_from_polygon` → `ox.features_from_polygon`
- `network_type='all_private'` → `network_type='all'`
- All `epsg:26915` references → `epsg:26916`

---

## Step 2: Single Sensor Simulation

Before generating the full dataset, individual sensor simulations can be run using `ND-Simulation.ipynb`. This loads the scene, places a single transmitter, places receivers at every grid point, and computes received signal strength (RSS) using Sionna's `PathSolver`.

### Physical Setup

| Parameter | Value |
|-----------|-------|
| Frequency | 2.4 GHz (common drone control band) |
| Transmit power | 20 dBm |
| Detection threshold | -90 dBm |
| Sensor height | 10m |
| Antenna pattern | Isotropic |
| Polarization | Vertical |
| Max ray depth | 4 bounces |
| Ray samples | 1,000,000 |

### How RSS is Computed

Sionna traces rays from the transmitter and accumulates contributions from all paths that reach each receiver — direct line-of-sight, reflections off buildings, and diffractions around edges. The total received power is the sum of squared magnitudes of the complex channel coefficients across all paths. This is then converted to dBm and compared against the detection threshold to produce a binary coverage map.

The reciprocity of RF propagation means this is equivalent to asking: if a drone (emitter) is at each grid point, can the sensor (receiver) detect it?

### Coordinate System

All simulation coordinates are in meters, centered at the scene origin (the centroid of the bounding box). A sensor at `[0, 0, 10]` is at the center of the scene at 10m height.

---

## Step 3: Dataset Generation

The full dataset is generated by `dataset_gen_3m.py` (3m resolution) and `dataset_gen_1m.py` (1m resolution). Each script loops over 2048 sensor positions arranged in a 32×64 uniform grid across the scene, runs a Sionna simulation for each, and saves the result as an NPZ file.

### Sensor Grid

```python
N_SENSORS_X = 32
N_SENSORS_Y = 64  # 32x64 = 2048 total positions

sensor_x = np.linspace(X_MIN * 0.8, X_MAX * 0.8, N_SENSORS_X)
sensor_y = np.linspace(Y_MIN * 0.8, Y_MAX * 0.8, N_SENSORS_Y)
```

The 0.8 multiplier keeps sensor positions slightly inside the scene boundary. Some positions may fall inside buildings — these are not filtered out and can be excluded in post-processing if needed.

### Dataset Format

Each sample is saved as an individual NPZ file (`sample_0000.npz`, `sample_0001.npz`, etc.):

```python
np.savez(save_path,
         sensor_pos=np.array(sensor_pos),   # shape: (3,) — [x, y, z] in meters
         rss_map=rss_2d,                     # shape: (ny, nx) — RSS in dBm
         coverage_map=coverage)              # shape: (ny, nx) — binary int8
```

Loading a sample:

```python
data = np.load("sample_0000.npz")
sensor_pos = data["sensor_pos"]     # e.g. [-84.0, -131.2, 10.0]
rss_map = data["rss_map"]           # float64, dBm values
coverage_map = data["coverage_map"] # int8, 0 or 1
```

### Grid Dimensions

| Resolution | Grid size | Points per simulation |
|------------|-----------|----------------------|
| 3m | 70 × 110 | 7,700 |
| 1m | 210 × 328 | 68,880 |

### Resume Logic

The script checks for existing NPZ files at the start of each run and skips already-completed positions. This means the job can be interrupted and resubmitted without losing progress:

```python
if os.path.exists(save_path):
    print(f"[{idx+1}/{len(sensor_positions)}] Skipping (already done)")
    continue
```

---

## Step 4: Running on CRC

All simulation jobs are run on Notre Dame's CRC cluster using the GPU queue for TensorFlow/Sionna acceleration.

### Job Submission

```bash
qsub submit_3m.sh   # 3m resolution job
qsub submit_1m.sh   # 1m resolution job
```

### Monitoring

```bash
qstat -u jboskovi                    # check job status
cat ~/sionna_3m.o[job_id]           # view output log
```

### Key Job Script Settings

```bash
#$ -q gpu
#$ -l gpu_card=1
#$ -l h_rt=96:00:00   # 96 hour max wall time on GPU queue
#$ -pe smp 4
```

Jobs are configured to email `jboskovi@nd.edu` on start, finish, or abort (`-m abe`).

---

## Known Issues and Limitations

**Road mesh artifact (fixed in dataset_v3):** Road meshes generated from OSM were extruded 0.25m above ground, causing receiver points at z=0 to fall below the surface. This produced unrealistic coverage shadows along all walkways. Fixed by excluding road meshes from the scene.

**Building positions may be slightly off:** OSM building footprints are not always perfectly accurate. Some buildings may be slightly misaligned relative to their real-world positions.

**Some sensor positions fall inside buildings:** The uniform sensor grid does not account for building locations. Samples where the sensor is inside a building are physically unrealistic but are included in the dataset. These can be filtered in post-processing.

**49-sample legacy dataset:** An earlier JSON-format dataset of 49 samples exists in the repository from early testing. This dataset used a coarser 7×7 sensor grid and is superseded by the NPZ datasets in `dataset_v2` and `dataset_v3`.

**dataset_v2 road artifact:** The `dataset_v2` dataset was generated with road meshes included in the scene. Coverage maps in this dataset show unrealistic dead zones along walkways. Use `dataset_v3` for clean data.

---

## Next Steps

1. **Validate against real data** — deploy RadioHound sensors on the Notre Dame quad and compare measured coverage against simulated coverage maps
2. **Classical baselines** — implement greedy algorithm, simulated annealing, and genetic algorithm approaches to the coverage maximization problem
3. **Neural network** — train a CNN to predict optimal sensor placements from coverage maps
4. **Scale dataset** — generate data from additional campus areas and sensor heights to improve generalization
