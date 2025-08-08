# **Ανάπτυξη Αλγορίθμου Πρόβλεψης Τροχιάς Οχημάτων με Τεχνικές Βαθιάς Μάθησης**  
# **Development of a Vehicle Trajectory Prediction Algorithm using Deep Learning Techniques**  

This repository contains the full implementation of a diploma thesis project focused on **real-time vehicle trajectory prediction** in the [CARLA](https://carla.org) autonomous driving simulator.  
The project compares two fundamentally different forecasting methods:  
1. **Physics-Based Prediction** using classical kinematic equations.  
2. **Deep Learning-Based Prediction** using the **pretrained CRAT-Pred neural network model**, adapted for real-time use in CARLA.  

---

## **1. Introduction & Motivation**  
Vehicle trajectory prediction is a critical component in **autonomous driving** and **intelligent transportation systems**.  
It enables proactive decision-making for path planning, collision avoidance, and cooperative driving.  

**Challenges include:**  
- Handling uncertainty in driver behavior.  
- Accounting for multi-agent interactions.  
- Adapting models to dynamic and complex urban environments.  

---

## **2. Project Overview**  
- **Testing Environment**: CARLA simulator.  
- **Goal**: Compare the accuracy, robustness, and real-time feasibility of physics-based vs deep learning-based trajectory prediction methods.  
- **Data Source**:  
  - Real-time streaming from CARLA for both methods.  
  - Argoverse Motion Forecasting v1.1 dataset for CRAT-Pred pretraining.  

---

## **3. System Architecture / Workflow**  

1. **Data Collection**: Vehicle positions and velocities streamed from CARLA.  
2. **Preprocessing**: Rotation normalization for CRAT-Pred; direct velocity/yaw extraction for kinematics.  
3. **Prediction**:  
   - CRAT-Pred: Multi-modal deep learning inference.  
   - Kinematics: Physics equations of motion.  
4. **Post-Processing**: Inverse rotation to CARLA coordinates (for CRAT-Pred).  
5. **Visualization**: Real-time debug markers in CARLA.  
6. **Evaluation**: Metrics (ADE, FDE, MR) computed per scenario.  

---

## **4. Prediction Approaches**  

### **4.1 CRAT-Pred Deep Learning Approach**  
- **Architecture**:  
  - LSTM Encoder for temporal motion patterns.  
  - Graph Neural Network for agent interactions.  
  - Multi-Head Self-Attention for spatial–temporal dependencies.  
  - Residual Decoder for multi-modal output.  
- **Prediction**:  
  - **Pretrained CRAT-Pred model** (Argoverse v1.1).  
  - Outputs 60 \((x, y)\) coordinates over 3 seconds for each mode.  
  - Most likely mode selected using **Final Displacement Error (FDE)**.  
- **Integration & Adjustments**:  
  - Adapted preprocessing for CARLA live data.  
  - Implemented rotation normalization & inverse transformation.  
  - Real-time inference & visualization inside CARLA.  
- **Strengths**:  
  - Captures complex, non-linear driving behaviors.  
  - Handles multiple interacting agents.  
- **Limitations**:  
  - Sensitive to domain shifts between training and simulation.  
  - Requires GPU acceleration for real-time use.  

---

### **4.2 Kinematics-Based Approach**  
- **Method**: Uses classical equations of motion to project future positions based on current speed and heading.  
- **Equations**:  

\[
x_{t+\Delta t} = x_t + v_x \cdot \Delta t
\]  

\[
y_{t+\Delta t} = y_t + v_y \cdot \Delta t
\]  

Where:  
- \(x_t, y_t\) = current position.  
- \(v_x, v_y\) = velocity components in the world frame.  
- \(\Delta t\) = prediction time step.  

- **Strengths**:  
  - Computationally lightweight.  
  - No training data required.  
- **Limitations**:  
  - Assumes constant velocity and direction.  
  - Cannot model turns or interaction dynamics well.  

---

## **5. Evaluation Metrics**  
- **Average Displacement Error (ADE)** – Mean distance between predicted and ground truth points.  
- **Final Displacement Error (FDE)** – Distance between final predicted point and ground truth (CRAT-Pred only).  
- **Miss Rate (MR)** – Fraction of predictions exceeding a distance threshold from ground truth.  

---

## **6. Results Summary**  
Both methods were tested in multiple CARLA traffic scenarios, including straight roads, curves, and multi-agent interactions.  

**Key findings:**  
- CRAT-Pred excelled in complex urban scenes with interacting agents.  
- Kinematics worked well for constant-speed, straight-line motion but degraded in turns and dynamic traffic.  

---

## **7. Technical Stack**  
- **Languages**: Python  
- **Frameworks**: PyTorch, PyTorch Lightning  
- **Libraries**: CARLA API, NumPy, Pandas, Matplotlib  
- **Tools**: CARLA Simulator  
- **Dataset**: Argoverse Motion Forecasting v1.1 (for CRAT-Pred pretraining)  

---

## **8. Key Features**  
- Real-time CARLA simulation with ego and NPC vehicles.  
- Multi-modal trajectory prediction (CRAT-Pred).  
- Physics-based trajectory prediction (Kinematics).  
- Per-second logging & CSV exports of GT vs predicted positions.  
- Real-time and offline visualizations.  

---

## **9. Author**  
**Evangelos Papaioannou**  
Diploma Thesis, 2025  
Department of Electrical and Computer Engineering  
Democritus University of Thrace  
