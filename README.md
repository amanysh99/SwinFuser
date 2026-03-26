# SwinFuser

**Camera-LiDAR Fusion for Autonomous Driving via Bidirectional Cross-Modal Attention and Swin Transformer Backbones**

> Submitted to *Neurocomputing* (Elsevier, Q1)

---

## Overview

SwinFuser is an autonomous driving architecture that fuses camera and LiDAR sensor data using a three-stage bidirectional cross-modal attention mechanism built on Swin Transformer backbones. It is evaluated on the **CARLA Leaderboard Longest6** benchmark using imitation learning.

The key contribution is a **bidirectional cross-modal attention fusion** mechanism that enables rich feature exchange between RGB image features and LiDAR point cloud features at multiple stages of the network.

---

## Architecture

- **Backbone:** Swin Transformer (camera) + PointPillar (LiDAR)
- **Fusion:** Three-stage bidirectional cross-modal attention
- **Task heads:** Waypoint prediction, traffic light state, velocity
- **Training:** Imitation learning on CARLA expert demonstrations

---

## Results on CARLA Longest6

| Model        | Driving Score (DS) | Route Completion (RC) | Infraction Score (IS) |
|--------------|-------------------|----------------------|----------------------|
| TransFuser   | 45.05%            | 79.44%               | 0.596                |
| **SwinFuser**| **55.60%**        | **87.21%**           | **0.668**            |

SwinFuser achieves significant improvements over the TransFuser baseline, particularly in collision avoidance and route completion.

---

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.10+
- CARLA 0.9.10
- CUDA 11.1+

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SwinFuser.git
cd SwinFuser

# Install dependencies
pip install -r requirements.txt
```

---

## Training

```bash
python train_swin.py \
  --id swinfuser_experiment \
  --batch_size 8 \
  --epochs 40 \
  --logdir ./logs
```

For multi-GPU training (DDP):

```bash
torchrun --nproc_per_node=NUM_GPUS train_swin.py \
  --id swinfuser_ddp \
  --batch_size 8
```

---

## Evaluation on CARLA

```bash
python submission_agent.py \
  --checkpoint ./logs/swinfuser_experiment/best_model.pth \
  --host localhost \
  --port 2000
```

---

## Project Structure

```
SwinFuser/
├── config.py             # Training configuration
├── data.py               # Dataset loading
├── model.py              # Main SwinFuser model
├── swin_transfuser_ptt.py # Swin Transformer backbone
├── train_swin.py         # Training script
├── submission_agent.py   # CARLA evaluation agent
├── transfuser.py         # TransFuser baseline
├── geometric_fusion.py   # Geometric fusion module
├── late_fusion.py        # Late fusion module
├── latentTF.py           # Latent TransFuser
├── point_pillar.py       # LiDAR PointPillar encoder
└── utils.py              # Utility functions
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{swinfuser2024,
  title={SwinFuser: Bidirectional Cross-Modal Attention for Camera-LiDAR Fusion in Autonomous Driving},
  author={Your Name},
  journal={Neurocomputing},
  year={2024}
}
```

---

## Acknowledgements

This work builds upon [TransFuser](https://github.com/autonomousvision/transfuser)
(Chitta et al., PAMI 2023), licensed under the MIT License.
We thank the Autonomous Vision Group for their open-source contribution.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
