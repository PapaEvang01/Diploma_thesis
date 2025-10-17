# **Development of a Vehicle Trajectory Prediction Algorithm using Deep Learning Techniques**  

This repository contains the full implementation of my diploma thesis project on **real-time vehicle trajectory prediction** in the [CARLA](https://carla.org) autonomous driving simulator.  
The project compares two fundamentally different forecasting methods:  
1. **Physics-Based Prediction** using classical kinematic equations.  
2. **Deep Learning-Based Prediction** using the **pretrained CRAT-Pred neural network model**, adapted for real-time use in CARLA.  

---

## **1. Introduction & Motivation**  
Vehicle trajectory prediction is a key component in **autonomous driving** and **intelligent transportation systems**.  
It enables proactive decision-making for path planning, collision avoidance, and cooperative driving.  

**Main challenges:**  
- Uncertainty in driver behavior.  
- Multi-agent interaction complexity.  
- Real-time adaptation to dynamic urban environments.  

---

## **2. Project Overview**  
- **Testing Environment**: CARLA Simulator.  
- **Goal**: Compare the accuracy, robustness, and real-time feasibility of physics-based vs deep learning-based methods.  
- **Data Sources**:  
  - Real-time streaming from CARLA for both methods.  
  - **Argoverse Motion Forecasting v1.1** dataset for CRAT-Pred pretraining.  

---

## **3. System Architecture / Workflow**  

1. **Data Collection**: Vehicle positions and velocities from CARLA.  
2. **Preprocessing**:  
   - CRAT-Pred: Rotation normalization of trajectories.  
   - Kinematics: Direct yaw and velocity extraction.  
3. **Prediction**:  
   - CRAT-Pred: Multi-modal neural network inference.  
   - Kinematics: Classical equations of motion.  
4. **Post-Processing**:  
   - CRAT-Pred: Inverse rotation to CARLA world coordinates.  
5. **Visualization**: Real-time debug markers in CARLA.  
6. **Evaluation**: Metrics (ADE, FDE, MR) computed for each scenario.  

---

## **4. Prediction Approaches**  

### **4.1 CRAT-Pred Deep Learning Approach**  
- **Architecture**:  
  - LSTM Encoder for temporal motion patterns.  
  - Graph Neural Network for agent interaction modeling.  
  - Multi-Head Self-Attention for spatial–temporal dependencies.  
  - Residual Decoder for generating multi-modal predictions.  
- **Prediction**:  
  - **Pretrained CRAT-Pred model** (trained on Argoverse v1.1).  
  - Outputs 60 \((x, y)\) coordinates over 3 seconds for each mode.  
  - Most likely mode selected using **Final Displacement Error (FDE)**.  
- **Integration & Adjustments**:  
  - Adapted preprocessing pipeline for CARLA streaming data.  
  - Implemented rotation normalization and inverse transformations.  
  - Enabled real-time inference and visualization.  
- **Strengths**:  
  - Captures complex, non-linear driving behaviors.  
  - Handles multi-agent interactions effectively.  
- **Limitations**:  
  - Sensitive to domain shifts between datasets and simulation.  
  - Requires GPU acceleration for optimal real-time performance.  

---

### **4.2 Kinematics-Based Approach**  
- **Method**: Predicts future positions assuming constant velocity and heading.  
- **Equations**:  

**Position update:**

\[
x_{t+Dt} = x_t + v * cos(psi) * Dt
\]

\[
y_{t+Dt} = y_t + v * sin(psi) * Dt
\]

**Velocity components:**

\[
v_x = v * cos(psi)
\]

\[
v_y = v * sin(psi)
\]

Where:  
- \(x_t, y_t\) = current position (m)  
- \(v\) = speed magnitude (m/s)  
- \(psi\) = yaw/heading angle (radians)  
- \(Delta t\) = prediction time step (s)  

- **Strengths**:  
  - Extremely fast and lightweight.  
  - No training required.  
- **Limitations**:  
  - Assumes constant motion.  
  - Performs poorly during turns or sudden maneuvers.  

---

## **5. Evaluation Metrics**  
- **Average Displacement Error (ADE)** – Mean distance between predicted and ground truth points.  
- **Final Displacement Error (FDE)** – Distance between final predicted point and ground truth (used for CRAT-Pred).  
- **Miss Rate (MR)** – Fraction of predictions exceeding a distance threshold from ground truth.  

---

## **6. Results Summary**

The evaluation compared two fundamentally different trajectory prediction methods in the CARLA simulator:
1. **Kinematic Method** – a physics-based baseline using motion equations.
2. **CRAT-Pred** – a deep learning model combining LSTM, GNN, and Attention mechanisms.

Each model was tested across three driving scenarios representing typical motion behaviors in autonomous driving:  
**Straight Path**, **Right Turn**, and **Left Turn**.

---

### **Quantitative Results**

| **Scenario** | **Method** | **Duration (s)** | **ADE (per timestep)** | **ADE (per second)** | **MR (per timestep)** | **MR (per second)** |
|---------------|-------------|------------------|-------------------------|----------------------|-----------------------|---------------------|
| Straight Line | Kinematics | 12 | **1.6724 m** | **0.5970 m** | 39.64 % | 0.00 % |
| Straight Line | CRAT-Pred | 18 | **0.01394 m** | **0.01824 m** | 0.00 % | 0.00 % |
| Right Turn | Kinematics | 12 | **1.3447 m** | **0.7142 m** | 43.70 % | 0.00 % |
| Right Turn | CRAT-Pred | 12 | **0.01562 m** | **0.02111 m** | 0.00 % | 0.00 % |
| Left Turn | Kinematics | 18 | **1.2834 m** | **0.6599 m** | 40.00 % | 0.00 % |
| Left Turn | CRAT-Pred | 12 | **0.01731 m** | **0.02380 m** | 0.00 % | 0.00 % |

---

### **Analysis and Discussion**

- The **Kinematic Method** demonstrated stable and computationally efficient performance, particularly in straight-line scenarios.  
  However, it exhibited growing positional drift in curved trajectories due to the lack of higher-order motion modeling and interaction awareness.

- The **CRAT-Pred Deep Learning Model** achieved **exceptionally low prediction errors** across all cases, maintaining near-zero ADE and MR even in challenging curved paths.  
  Its combination of temporal encoding, spatial graph reasoning, and attention-based feature fusion allows it to generalize effectively to dynamic, multi-agent environments.

- The results confirm that **data-driven models significantly outperform deterministic baselines** when environmental complexity increases, though they require greater computational resources and pre-training.

---

### **Conclusions and Future Work**

- The **Kinematic Method** remains an interpretable and fast baseline, ideal for lightweight applications or embedded systems with limited compute.  
- The **CRAT-Pred model** serves as a powerful benchmark for accurate, multi-modal trajectory forecasting in realistic driving simulations.

**Future Directions:**
- Develop **hybrid models** that fuse the simplicity of kinematic motion equations with the adaptability of deep learning corrections.  
- Extend CRAT-Pred evaluation to **real-world datasets** (e.g., Argoverse 2, nuScenes) and multi-agent planning scenarios.  
- Integrate prediction outputs into **control and decision-making loops** within CARLA for closed-loop autonomous driving tests.

---

---

## **7. Technical Stack**  
- **Languages**: Python  
- **Frameworks**: PyTorch, PyTorch Lightning  
- **Libraries**: CARLA API, NumPy, Pandas, Matplotlib  
- **Tools**: CARLA Simulator  
- **Dataset**: Argoverse Motion Forecasting v1.1  

---

## **8. Key Features**  
- Real-time CARLA simulation with ego and NPC vehicles.  
- Multi-modal deep learning trajectory prediction.  
- Physics-based trajectory forecasting.  
- Per-second CSV logging of predictions and metrics.  
- Visual comparison of predicted vs ground truth trajectories.  

---

## **9. Author**  
**Evangelos Papaioannou**  
Diploma Thesis – 2025  
Department of Electrical and Computer Engineering  
Democritus University of Thrace  
