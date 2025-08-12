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
x_{t+Dt} = x_t + v * cos(\psi) * Dt
\]

\[
y_{t+Dt} = y_t + v * sin(\psi) * Dt
\]

**Velocity components:**
\[
v_x = v * cos(\psi)
\]

\[
v_y = v * sin(\psi)
\]

Where:  
- \(x_t, y_t\) = current position (m)  
- \(v\) = speed magnitude (m/s)  
- \(\psi\) = yaw/heading angle (radians)  
- \(\Delta t\) = prediction time step (s)  

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
The models were tested in multiple CARLA driving scenarios (straight roads, curves, multi-agent traffic).  

**Findings:**  
- CRAT-Pred outperformed in urban and interaction-heavy situations.  
- Kinematics excelled in straight, constant-speed motion.  

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
