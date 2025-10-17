Kinematic Trajectory Prediction in CARLA
========================================

This project implements real-time prediction of vehicle motion using classical kinematic equations,
within the CARLA simulator environment. It was developed as part of a diploma thesis comparing
physics-based methods with deep learning-based trajectory forecasting.

--------------------------------------------------
Overview
--------------------------------------------------
The system connects to a live CARLA simulation, spawns a Tesla Model 3 as the ego vehicle,
and begins trajectory prediction after movement starts. Using the current velocity and yaw angle,
it estimates the vehicle's future position 1 second ahead (at 20Hz resolution). Predictions are 
visualized in CARLA and evaluated in real time.

Predicted vs ground truth positions are logged to CSV, and the overall accuracy is measured 
using the Average Displacement Error (ADE) and Miss Rate (MR). A plot of the full trajectory 
comparison is also generated for visual inspection.

--------------------------------------------------

Project Structure
--------------------------------------------------
carla-kinematics-trajectory-prediction/
├── src/
│   └── carla_kinematics.py                # Main real-time trajectory prediction script
├── results_kinematics/
│   ├── scenario1_leftturn/
│   │   ├── carla_view.png                 # CARLA visualization image
│   │   ├── description_scenario1.txt      # Scenario 1 description and results
│   │   ├── gt_vs_pred.png                 # Plot comparing predicted vs actual positions
│   │   └── trajectories_1.csv             # Detailed trajectory data
│   ├── scenario2_rightturn/
│   │   ├── carla_view.png
│   │   ├── description_scenario2.txt
│   │   ├── gt_vs_pred.png
│   │   └── trajectories.csv
│   ├── scenario3_straightline/
│   │   ├── carla_view.png
│   │   ├── description_scenario3.txt
│   │   ├── gt_vs_pred.png
│   │   └── trajectories.csv
│   └── results_kinematics_description.txt # Global summary of all scenarios
├── requirements.txt                       # Python dependencies
└── README.txt                             # Project documentation

--------------------------------------------------
Simulation Scenarios and Results
--------------------------------------------------
Three simulation scenarios were executed in the CARLA environment to evaluate the kinematic model:

1) Scenario 1 – Left Turn
   - Description: Vehicle follows a left-turn curve at moderate speed.
   - Duration: 18 s
   - minADE (per timestep): 1.2834 m
   - MR (per timestep): 40.00 %
   - minADE (per second): 0.6599 m
   - MR (per second): 0.00 %
   - Files:
       • Image: results_kinematics/scenario1_leftturn/carla_view.png
       • Plot: results_kinematics/scenario1_leftturn/gt_vs_pred.png
       • CSV: results_kinematics/scenario1_leftturn/trajectories.csv
       • Summary: description_scenario1.txt

2) Scenario 2 – Right Turn
   - Description: Vehicle performs a right-turn trajectory under similar conditions.
   - Duration: 12 s
   - minADE (per timestep): 1.3447 m
   - MR (per timestep): 43.70 %
   - minADE (per second): 0.7142 m
   - MR (per second): 0.00 %
   - Files:
       • Image: results_kinematics/scenario2_rightturn/carla_view.png
       • Plot: results_kinematics/scenario2_rightturn/gt_vs_pred.png
       • CSV: results_kinematics/scenario2_rightturn/trajectories.csv
       • Summary: description_scenario2.txt

3) Scenario 3 – Straight Line
   - Description: Vehicle moves along a straight road section with constant velocity.
   - Duration: 12 s
   - minADE (per timestep): 1.6724 m
   - MR (per timestep): 39.64 %
   - minADE (per second): 0.5970 m
   - MR (per second): 0.00 %
   - Files:
       • Image: results_kinematics/scenario3_straightline/carla_view.png
       • Plot: results_kinematics/scenario3_straightline/gt_vs_pred.png
       • CSV: results_kinematics/scenario3_straightline/trajectories.csv
       • Summary: description_scenario3.txt

--------------------------------------------------
Output & Logging
--------------------------------------------------
Each scenario generates the following outputs:
- One CARLA image showing predicted and actual trajectories.
- One plot comparing predicted vs ground-truth positions.
- One CSV file with detailed trajectory data and computed ADE/MR values per timestep and per second.
- One text summary file (.txt) describing the key results and observations for that scenario.

All outputs are automatically stored in the corresponding results_kinematics/scenarioX/ folder.

--------------------------------------------------
Author & Context
--------------------------------------------------
This project is part of the diploma thesis:

"Development of a Vehicle Trajectory Prediction Algorithm using Deep Learning Techniques"
by Evangelos Papaioannou, Democritus University of Thrace (D.U.Th.)

The kinematic prediction method serves as a physics-based baseline, relying solely on fundamental motion equations
without any learning components. It demonstrates reliable short-term predictions for simple, low-dynamic trajectories,
while providing a quantitative benchmark for evaluating advanced deep learning models such as CRAT-Pred.

--------------------------------------------------
License
--------------------------------------------------
This project is distributed for academic and research purposes only.
For further use or publication, please cite this diploma thesis.
