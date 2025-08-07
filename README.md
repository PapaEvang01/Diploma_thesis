# Real-Time Vehicle Trajectory Prediction in CARLA  
### Diploma Thesis Project by Evangelos Papaioannou

This repository presents the full implementation of a diploma thesis focused on **real-time trajectory prediction of vehicles** in the [CARLA](https://carla.org) autonomous driving simulator. The project investigates and compares two fundamentally different approaches to forecasting future vehicle motion:

- **Physics-Based Prediction** using classical kinematic equations
- **Deep Learning-Based Prediction** using the CRAT-Pred neural network model

---

## 🎓 Project Context

This work was carried out in the context of the diploma thesis titled:

**Ανάπτυξη Αλγορίθμου Πρόβλεψης Τροχιάς Οχημάτων με Τεχνικές Βαθιάς Μάθησης**  
(*Development of a Vehicle Trajectory Prediction Algorithm using Deep Learning Techniques*)

The goal was to design, implement, and evaluate a real-time system that predicts how a vehicle will move in a simulated environment, and to **compare the performance** of traditional kinematics against advanced machine learning models trained on real-world traffic data.

---

## 📁 Repository Overview

The repository is organized into two main subprojects:

### 1. `carla-kinematics-trajectory-prediction/`
Implements a **classical approach** based on velocity, yaw, and time. The method uses CARLA's API to track a Tesla Model 3 in real time, estimate its position 1 second into the future, and compare the predictions against ground truth positions.

- Predictions are visualized inside CARLA.
- Evaluation metrics (ADE and Miss Rate) are computed.
- Results are saved to CSV and plotted for inspection.

### 2. `crat-pred-trajectories-in-carla/`
Integrates the **CRAT-Pred deep learning model** with CARLA for **multi-agent, multi-modal trajectory prediction**. It uses a trained neural network (with LSTM encoders, graph neural networks, and attention mechanisms) to infer complex motion patterns in urban environments.

- Automatically spawns an ego vehicle and NPCs.
- Tracks vehicle displacements and normalizes them.
- Feeds data into CRAT-Pred to predict 60 future positions over 3 seconds.
- Selects the most accurate mode (based on FDE).
- Inversely rotates and visualizes predictions in CARLA.
- Logs per-timestep ADE and Miss Rate in real time.

---

## 🔬 Evaluation

Both systems were evaluated under real-time simulation conditions using:

- **Average Displacement Error (ADE)**  
- **Final Displacement Error (FDE)** *(for CRAT-Pred)*
- **Miss Rate (MR)**

Detailed performance comparison was performed across different traffic scenarios, and the results were exported for analysis and visual presentation.

---

## 🧠 Key Contributions

- Full real-time integration with the CARLA simulator.
- Implementation of trajectory forecasting via classical physics and deep learning.
- CRAT-Pred architecture loading, inference, and post-processing pipeline.
- Per-second logging of predictions vs ground truth.
- Visualization tools for plotting trajectories and evaluating accuracy.
- Clean and modular code organized into reusable components.

---

## 👨‍💻 Author

**Evangelos Papaioannou**  
Diploma Thesis, 2025  
Department of Electrical and Computer Engineering  
Democritus University of Thrace

For questions, feel free to contact me via GitHub or LinkedIn.
