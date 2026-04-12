# AURORA: Adaptive Uncertainty-Aware Robust 4D Radar Fusion for All-Weather Multi-Modal Localization

## 📌 Overview

Reliable ego-vehicle localization is a fundamental requirement for autonomous driving, yet existing multi-modal systems often degrade under adverse weather and illumination conditions.

This repository contains the implementation of **AURORA**, an end-to-end multi-modal odometry framework designed to achieve robust localization by explicitly modeling environmental conditions and sensor reliability.

The framework integrates **4D radar, camera, and IMU data** within a unified architecture that dynamically adapts fusion strategies based on learned environmental degradation signals. 

---

## 🚀 Key Contributions

* **Environment-aware feature modulation**
  Explicit modeling of weather and illumination severity to dynamically adapt visual feature representations.

* **Policy-regularized soft-gated fusion**
  Continuous, reliability-aware multi-modal fusion that prevents degenerate modality selection.

* **Uncertainty-aware pose estimation**
  Joint regression of 6-DoF pose and heteroscedastic uncertainty for reliability-aware localization.

* **Robust performance in adverse conditions**
  Significant improvements in translational and rotational drift under challenging environments.

---

## 🧠 Method Overview

AURORA follows a multi-modal architecture based on three key principles:

1. **Modality complementarity** (camera, radar, IMU)
2. **Environment-conditioned feature modulation**
3. **Adaptive, policy-driven fusion**

### Architecture

![Architecture](results/figures/aurora_architecture.png)

The model extracts modality-specific features and dynamically modulates visual representations using learned environmental severity indicators. These features are fused through a policy-regularized mechanism and processed by a recurrent decoder for joint pose and uncertainty estimation.

---

## 📊 Results

The proposed framework achieves strong performance on the HeRCULES dataset:

* **ATE (Absolute Trajectory Error):** 2.04 m
* **RPE (translational):** 0.174 m/m
* **RPE (rotational):** 0.044 deg/m

Environment-aware modulation reduces:

* translational drift by **38.9%**
* rotational drift by **42.1%** 

### Trajectory Estimation

<p align="center">
  <img src="results/figures/trajectory_1.png" width="30%">
  <img src="results/figures/trajectory_2.png" width="30%">
  <img src="results/figures/trajectory_3.png" width="30%">
</p>

### Comparison with A2DO

<p align="center">
  <img src="results/figures/a2do_1.png" width="30%">
  <img src="results/figures/a2do_2.png" width="30%">
  <img src="results/figures/a2do_3.png" width="30%">
</p>

---

## 📁 Repository Structure

```text
AURORA/
├── README.md
├── .gitignore
├── src/
├── results/
│   └── figures/
├── configs/
└── scripts/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/alicedidomenico01/AURORA-Adaptive-Uncertainty-Aware-Robust-4D-Radar-Fusion-For-All-Weather-Multi-modal-Localization.git
cd AURORA
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Example training command:

```bash
python imu_seq2seq_lstm_radarstep_twohead_quat_adaloss_rmse_se3patch_inmodelenc_batched.py --mode train --seqs "00,01,02,03,04,05,06,07,08,09,10,11,12,13,15,16,18,20"
```

Example evaluation:

```bash
python imu_seq2seq_lstm_radarstep_twohead_quat_adaloss_rmse_se3patch_inmodelenc_batched.py --mode eval --seqs "00,01,02,03,04,05,06,07,08,09,10,11,12,13,15,16,18,20"
```

---

## 📄 Paper

The full paper describing AURORA will be available in this repository.


---

## 👤 Author

**Alice Di Domenico**

---

## 📌 Notes

This repository focuses on research reproducibility and clean implementation of multi-modal sensor fusion for robust localization under environmental degradation.
