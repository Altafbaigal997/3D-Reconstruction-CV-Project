# 3D-Reconstruction-CV-Project

# 3D Reconstruction using Structure from Motion (SfM)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A complete **Structure from Motion (SfM)** pipeline built from scratch in Python for 3D scene reconstruction from 2D images. Implements incremental reconstruction with GPU-accelerated Bundle Adjustment and dense stereo matching.


##  Project Overview

This project implements a full incremental SfM pipeline as part of Computer Vision coursework at LUMS. The system reconstructs 3D point clouds and camera poses from sequential images without using black-box reconstruction libraries.

### Key Features

-  **SIFT Feature Detection** with FLANN-based matching
-  **Intelligent Image Sequencing** using greedy overlap optimization
-  **Essential Matrix Decomposition** for initial camera pair
-  **Incremental PnP-RANSAC** for camera localization
-  **Multi-view Triangulation** with reprojection error filtering
-  **GPU-Accelerated Bundle Adjustment** using PyTorch
-  **Dense Stereo Reconstruction** with CUDA support
-  **Web-based 3D Visualization** using Three.js

##  Pipeline Architecture
```
Input Images
    ↓
┌─────────────────────────────────────────┐
│  1. Image Sequencing                    │
│     - SIFT feature extraction           │
│     - Pairwise matching (FLANN)         │
│     - Greedy overlap optimization       │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  2. Bootstrap (First 2 Images)          │
│     - Essential Matrix estimation       │
│     - Pose recovery (cheirality check)  │
│     - Initial triangulation             │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  3. Incremental Reconstruction          │
│     - PnP-RANSAC camera localization    │
│     - New point triangulation           │
│     - Bundle Adjustment (every N frames)│
│     - Scale normalization               │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  4. Dense Reconstruction (Optional)     │
│     - Stereo rectification              │
│     - Semi-Global Block Matching        │
│     - Disparity-to-3D conversion        │
└─────────────────────────────────────────┘
    ↓
Output: 3D Point Cloud + Camera Poses
```

##  Tech Stack

- **Python 3.8+**
- **OpenCV 4.x** - Feature detection, matching, geometry
- **NumPy** - Numerical operations
- **PyTorch 2.0+** - GPU-accelerated Bundle Adjustment
- **Matplotlib** - Visualization
- **Three.js** - Web-based 3D viewer

##  Installation

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# CUDA (optional, for GPU acceleration)
nvidia-smi
```

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/3D-Reconstruction-SFM.git
cd 3D-Reconstruction-SFM

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```txt
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
torch>=2.0.0
pillow>=10.0.0
scipy>=1.10.0
tqdm>=4.65.0
```

##  Usage

### Basic Reconstruction
```bash
# Run full SfM pipeline
python components/incremental_sfm.py
```

### Configuration

Edit the main script to configure:
```python
# Camera intrinsics (update for your camera)
K = np.array([
    [2184, 0, 1512],
    [0, 2184, 2016],
    [0, 0, 1]
], dtype=np.float32)

# Dataset path
dataset_path = 'data/your_dataset'

# Reconstruction settings
sfm = IncrementalSfM(
    K=K,
    ordered_image_paths=image_paths,
    use_bundle_adjustment=True,   # Enable/disable BA
    use_dense_stereo=False,        # Enable dense reconstruction
    ba_interval=5,                 # Run BA every 5 frames
    visualization_interval=10      # Visualize every 10 frames
)
```

### Image Sequencing (Optional)

If your images are **not** in sequential order:
```python
# In incremental_sfm.py
IMAGES_ALREADY_SEQUENCED = False  # Enable auto-sequencing
```

The sequencer will:
- Extract SIFT features for all images
- Find best initial pair (most matches)
- Order remaining images by visual overlap
- Cache features and match results

### Viewing Results

**Method 1: Python Viewer**
```bash
python visualize_reconstruction.py
```

**Method 2: Web Viewer**
```bash
# Start HTTP server
cd web_viewer
python -m http.server 8000

# Open browser: http://localhost:8000/index.html
```

##  Project Structure
```
3D-Reconstruction-SFM/
├── components/
│   ├── incremental_sfm.py          # Main SfM pipeline
│   ├── image_sequencer.py          # Image ordering
│   ├── essential_matrix.py         # Essential matrix estimation
│   ├── pose_detection.py           # Camera pose recovery
│   ├── triangulation.py            # 3D point triangulation
│   ├── pnp_solver.py               # PnP + Bundle Adjustment
│   ├── dense_stereo.py             # Dense reconstruction
│   └── bundle_adjustment_gpu.py    # GPU-accelerated BA
├── data/
│   └── Dataset_m3/                 # Input images
├── output/
│   └── sfm_output/
│       ├── final_reconstruction.ply
│       ├── camera_poses.json
│       ├── statistics.json
│       └── visualizations/
├── cache_files/
│   ├── feature_cache.pkl           # Cached SIFT features
│   └── match_results.pkl           # Cached match matrix
├── web_viewer/
│   ├── index.html
│   ├── js/
│   └── data/
├── visualize_reconstruction.py
├── requirements.txt
└── README.md
```

##  Technical Details

### 1. Feature Extraction & Matching
```python
# SIFT feature detection
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(image, None)

# FLANN-based matching with Lowe's ratio test
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
```

### 2. Essential Matrix & Pose Recovery
```python
# Essential Matrix estimation
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)

# Decompose to rotation and translation
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
```

### 3. PnP Camera Localization
```python
# Solve PnP with RANSAC
success, rvec, tvec, inliers = cv2.solvePnPRansac(
    objectPoints=points_3d,
    imagePoints=points_2d,
    cameraMatrix=K,
    distCoeffs=None,
    reprojectionError=8.0,
    confidence=0.99
)
```

### 4. Triangulation with Filtering
```python
# Triangulate points
points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
points_3d = (points_4d[:3] / points_4d[3]).T

# Filter by depth
valid_depth = (points_3d[:, 2] > 0.5) & (points_3d[:, 2] < 50)

# Filter by reprojection error
projected = (P @ points_3d_homogeneous.T).T
projected_2d = projected[:, :2] / projected[:, 2:3]
error = np.linalg.norm(pts - projected_2d, axis=1)
valid_reproj = error < 5.0

points_3d = points_3d[valid_depth & valid_reproj]
```

### 5. GPU Bundle Adjustment
```python
# PyTorch-based optimization
rvecs = torch.tensor(rvecs, requires_grad=True, device='cuda')
tvecs = torch.tensor(tvecs, requires_grad=True, device='cuda')
points_3d = torch.tensor(points_3d, requires_grad=True, device='cuda')

optimizer = torch.optim.Adam([rvecs, tvecs, points_3d], lr=0.01)

for iteration in range(max_iterations):
    projected = project_points(points_3d, rvecs, tvecs, K)
    loss = torch.mean((observed_2d - projected) ** 2)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Results

### Reconstruction Statistics

- **Images Processed:** 5/26 (19% success rate)
- **3D Points Generated:** 1,322 sparse points
- **Cameras Reconstructed:** 5 camera poses
- **Reprojection Error:** ~1.5 pixels (high accuracy)
- **Processing Time:** ~5 minutes (26 images, GPU enabled)

### Performance Metrics

| Component | Time (per image) | Notes |
|-----------|------------------|-------|
| Feature Extraction | ~1-2 sec | Cached after first run |
| Feature Matching | <1 sec | FLANN KD-tree |
| PnP Localization | <0.5 sec | RANSAC with 100+ inliers |
| Triangulation | <0.5 sec | Linear DLT method |
| Bundle Adjustment | ~10 sec | Every 5 frames, GPU |
| Dense Stereo | 2-3 sec/pair | GPU-accelerated SGBM |

## Known Limitations

### Scene Requirements

⚠️ **The pipeline struggles with:**
- Repetitive textures (e.g., brick walls, tiled floors)
- Low-texture surfaces (white walls, uniform colors)
- Non-static content (TV screens, moving objects)
- Highly planar scenes (lack of 3D structure)

✅ **Works best with:**
- Textured objects (furniture, outdoor landmarks)
- Good lighting conditions
- Static scenes
- 60-80% overlap between consecutive images
- 50+ images for complete coverage

### Technical Limitations

- **Bundle Adjustment:** Requires proper observation tracking for optimal results
- **Dense Reconstruction:** Memory-intensive for high-resolution images
- **Scale Drift:** Can accumulate over long sequences without loop closure

##  Learning Outcomes

This project demonstrates:

1. **Epipolar Geometry:** Essential Matrix, Fundamental Matrix
2. **Camera Calibration:** Intrinsic and extrinsic parameters
3. **3D Reconstruction:** Triangulation, depth estimation
4. **Optimization:** Non-linear least squares, RANSAC
5. **GPU Programming:** PyTorch for computational geometry
6. **Computer Vision Pipeline:** End-to-end system design

## Future Improvements

- [ ] Implement proper observation tracking for Bundle Adjustment
- [ ] Add loop closure detection for drift correction
- [ ] Support for uncalibrated cameras (auto-calibration)
- [ ] Replace SIFT with learning-based features (SuperPoint, LoFTR)
- [ ] Implement Visual SLAM for video input
- [ ] Add RGB-D sensor support (LiDAR, RealSense)
- [ ] Neural rendering (NeRF, Gaussian Splatting)

## References

- Hartley & Zisserman - *Multiple View Geometry in Computer Vision*
- Snavely et al. - *Photo Tourism: Exploring Photo Collections in 3D*
- Schönberger & Frahm - *Structure-from-Motion Revisited*
- [COLMAP Documentation](https://colmap.github.io/)
- [OpenCV SfM Tutorials](https://docs.opencv.org/master/d9/d0c/group__calib3d.html)

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

**Altaf Hussain & Muhammad Salman Ahmed**
- LUMS MS AI

## Acknowledgments
- OpenCV Community
- PyTorch Team

---
