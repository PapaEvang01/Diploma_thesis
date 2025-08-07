Kinematic Trajectory Prediction in CARLA
========================================

This project implements real-time prediction of vehicle motion using classical kinematic equations,
within the CARLA simulator environment. It was developed as part of a diploma thesis comparing
physics-based methods with deep learning-based trajectory forecasting.

Overview
--------
The system connects to a live CARLA simulation, spawns a Tesla Model 3 as the ego vehicle,
and begins trajectory prediction after movement starts. Using the current velocity and yaw angle,
it estimates the vehicle's future position 1 second ahead (at 20Hz resolution). Predictions are 
visualized in CARLA and evaluated in real time.

Predicted vs ground truth positions are logged to CSV, and the overall accuracy is measured 
using the Average Displacement Error (ADE) and Miss Rate (MR). A plot of the full trajectory 
comparison is also generated for visual inspection.

Project Structure
-----------------
carla_kinematics_project/
├── src/
│   ├── carla_kinematics.py     # Main prediction script
│   ├── requirements.txt        # Python dependencies
│   └── README.txt              
├── results/
│   ├── metrics.csv             # Logs of predicted vs actual positions (auto-generated)
│   └── trajectory_plot.png     # Visual plot of GT vs predicted positions (auto-generated)

Running the Simulation
----------------------
1. Ensure the CARLA simulator is running (localhost:2000).
2. Open a terminal and navigate to the `src/` directory.
3. Install required packages:

       pip install -r requirements.txt

4. Launch the script:

       python carla_kinematics.py

Output & Logging
----------------
- Predictions are logged every second (20 steps per second).
- `metrics.csv` contains: Time (s), Timestep, GT_X, GT_Y, Pred_X, Pred_Y, ADE, MissRate
- `trajectory_plot.png` shows the complete path comparison.

Author & Context
----------------
This project is part of the diploma thesis:

**Ανάπτυξη Αλγορίθμου Πρόβλεψης Τροχιάς Οχημάτων με Τεχνικές Βαθιάς Μάθησης**  
(*Development of a Vehicle Trajectory Prediction Algorithm using Deep Learning Techniques*)

**Author**: Evangelos Papaioannou
