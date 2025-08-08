# Development of a Vehicle Trajectory Prediction Algorithm using Deep Learning Techniques  (Ανάπτυξη Αλγορίθμου Πρόβλεψης Τροχιάς Οχημάτων με Τεχνικές Βαθιάς Μάθησης)


**Real-time vehicle trajectory prediction in the CARLA simulator using classical kinematic equations and deep learning models.**  
This project forms the practical and experimental core of my diploma thesis, aiming to **compare physics-based and deep learning-based methods** for forecasting vehicle motion in a simulated urban environment.

---

## 📌 Introduction & Motivation

Accurate **vehicle trajectory prediction** is a crucial component in **autonomous driving** and **intelligent transportation systems (ITS)**. It enables:

- Collision avoidance and path planning.
- Cooperative maneuvers between autonomous and human-driven vehicles.
- Traffic flow optimization and safety enhancement.

However, predicting motion in dynamic, multi-agent environments presents challenges such as:

- Complex interactions between road users.
- Uncertainty in driver behavior.
- Limited or noisy perception data.

This project addresses these challenges by evaluating **two contrasting approaches** within a controlled simulation environment.

---

## 📖 Project Overview

The project compares:

1. **Deep Learning Approach** – Using the **CRAT-Pred** neural network model, trained on the **Argoverse Motion Forecasting v1.1** dataset, capable of **multi-agent, multi-modal** trajectory prediction.
2. **Classical Kinematics Approach** – Applying **deterministic motion equations** based on velocity, yaw, and elapsed time to estimate future positions.

All experiments were conducted in the **CARLA Simulator**, an open-source platform for autonomous driving research, allowing reproducible and high-fidelity urban traffic scenarios.

---

## ⚙️ System Architecture / Workflow

**Pipeline:**

1. **Data Collection from CARLA** – Track ego and NPC vehicles in real-time.
2. **Preprocessing** – Compute displacements, normalize orientation via rotation matrices.
3. **Prediction**  
   - **CRAT-Pred Model** – LSTM encoder + Graph Neural Network + Multi-Head Attention + Residual Decoder for 60 future points (3s horizon).
   - **Kinematics** – Predict positions using classical motion equations.
4. **Post-Processing & Visualization** – Inverse rotation, CARLA in-simulator rendering, and matplotlib plotting.
5. **Evaluation** – Calculate metrics:
   - ADE (Average Displacement Error)
   - FDE (Final Displacement Error – CRAT-Pred only)
   - MR (Miss Rate)

---

## 🌟 Main Features

- **Real-time CARLA simulation** with ego vehicle (Tesla Model 3) and multiple NPCs.
- **Multi-modal trajectory prediction** using CRAT-Pred.
- **Physics-based prediction** using classical kinematics.
- **Evaluation & Logging**:
  - Per-second metric calculation.
  - CSV logging of predictions vs ground truth.
  - Automatic trajectory plots.
- **Scalable framework** for adding new prediction models.

---

## 🛠 Technical Stack

- **Languages**: Python
- **Frameworks**: PyTorch, PyTorch Lightning
- **Libraries**: CARLA API, NumPy, Pandas, Matplotlib
- **Tools**: CARLA Simulator
- **Dataset**: Argoverse Motion Forecasting v1.1 (for CRAT-Pred training)

---

## 🔍 Methodology

### **1. CRAT-Pred Deep Learning Approach**
- **Architecture**:
  - LSTM Encoder for temporal patterns.
  - Graph Neural Network for agent interactions.
  - Multi-Head Self-Attention for spatial-temporal dependencies.
  - Residual Decoder for generating predictions.
- **Prediction**:
  - Outputs 60 (x, y) coordinates for each mode.
  - Selects most likely mode using **Final Displacement Error**.
- **Strengths**:
  - Captures complex behaviors.
  - Handles multiple agents simultaneously.
- **Limitations**:
  - Requires large training datasets.
  - Sensitive to domain shifts.

### **2. Kinematics-Based Approach**
- **Equations**:
  - \( x_{t+\Delta t} = x_t + v \cdot \cos(\psi) \cdot \Delta t \)
  - \( y_{t+\Delta t} = y_t + v \cdot \sin(\psi) \cdot \Delta t \)
- **Assumptions**:
  - Constant velocity and yaw over the prediction horizon.
- **Strengths**:
  - Fast and computationally lightweight.
- **Limitations**:
  - Cannot adapt to sudden changes in trajectory.

---

## 📊 Evaluation Results

| Method                  | Scenario | ADE ↓ | FDE ↓ | MR ↓  |
|------------------------|----------|-------|-------|-------|
| Kinematics             | Urban 1  | 1.52  | –     | 0.12  |
| CRAT-Pred (Deep Learning) | Urban 1  | 0.89  | 1.21  | 0.05  |
| Kinematics             | Urban 2  | 1.73  | –     | 0.18  |
| CRAT-Pred              | Urban 2  | 1.05  | 1.37  | 0.07  |

*(↓ lower is better)*

### Example Visualization
![Trajectory Plot](results/trajectory_plot.png)

---

## 📌 Conclusion & Future Work

**Findings:**
- CRAT-Pred outperformed kinematics in **non-linear and interactive traffic scenarios**.
- Kinematics remained competitive in simple, straight-path motion.
- Deep learning handled uncertainty and multi-modal futures better.

**Future Directions:**
- Integrating map-based contextual information.
- Testing under adverse weather and sensor noise conditions.
- Extending to pedestrian and cyclist trajectory prediction.

---

## 🙏 Acknowledgements
- [CARLA Simulator](https://carla.org) team for providing the simulation platform.
- CRAT-Pred authors for their publicly available model and code.
- [Argoverse](https://www.argoverse.org) dataset creators for high-quality motion forecasting data.

---

## 📜 License
This project is released under the MIT License.
