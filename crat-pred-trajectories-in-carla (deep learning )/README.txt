CRAT-Pred Real-Time Trajectory Prediction in CARLA
==================================================

This project integrates the CRAT-Pred deep learning model with the CARLA simulator
to perform real-time multi-modal trajectory prediction for autonomous vehicles.
It tracks vehicles in the simulation, processes their motion, predicts future positions,
and evaluates the predictions with standard metrics.

--------------------------------------------------
Overview
--------------------------------------------------
The project connects to the CARLA simulator in asynchronous mode, spawns a Tesla Model 3 as the ego vehicle along with optional NPCs,
and starts collecting trajectory data after 3 seconds. A trained CRAT-Pred model predicts vehicle positions over a 3-second horizon at 20Hz.
The most accurate prediction mode is selected based on Final Displacement Error (FDE), then transformed back to global coordinates and visualized in CARLA.
Predicted and actual positions are saved to CSV, and performance metrics such as Average Displacement Error (ADE) and Miss Rate (MR) are calculated
and stored in the results/ folder.

--------------------------------------------------
Folder Structure
--------------------------------------------------
crat-pred-trajectories-in-carla/
├── src/
│   ├── crat_pred_movement_predictor.py     # Main real-time trajectory prediction script
│   ├── model_loader.py                     # Loads CRAT-Pred model and checkpoint
│   └── npc_spawner.py                      # Spawns NPC vehicles in CARLA
├── results_dl/
│   ├── scenario1_straight/
│   │   ├── gt_vs_pred_vehicle_24.csv        # Ground truth vs predicted positions
│   │   ├── metrics_per_timestep_vehicle_24.csv # ADE/FDE metrics per timestep
│   │   ├── trajectory_plot.png              # GT vs predicted trajectory plot
│   │   └── RESULTS_scenario1_DL.txt         # Scenario 1 summary and observations
│   ├── scenario2_rightturn/
│   │   ├── gt_vs_pred_vehicle_25.csv
│   │   ├── img2_cratpred.png
│   │   ├── metrics_per_timestep_vehicle_25.csv
│   │   ├── trajectory_plot.png
│   │   └── RESULTS_scenario2_DL.txt
│   ├── scenario3_leftturn/
│   │   ├── gt_vs_pred_vehicle_26.csv
│   │   ├── img3_cratpred.png
│   │   ├── metrics_per_timestep_vehicle_26.csv
│   │   ├── trajectory_plot.png
│   │   └── RESULTS_scenario3_DL.txt
│   └── results_dl_description.txt           # Global description of all DL scenarios
├── requirements.txt                         # Python dependencies
└── README.txt                               # Project documentation

--------------------------------------------------
Simulation Scenarios and Results
--------------------------------------------------
Three simulation scenarios were executed in the CARLA environment to evaluate the CRAT-Pred model:

1) Scenario 1 – Straight
   - Description: Vehicle moves straight with minor lane maneuvers.
   - Duration: 18 s
   - ADE (per timestep): 0.01394 m
   - MR (per timestep): 0.00 %
   - ADE (per second): 0.01824 m
   - MR (per second): 0.00 %
   - Files:
       • GT vs Predicted CSV: results_dl/scenario1_straight/gt_vs_pred_vehicle_24.csv
       • Metrics per Timestep: results_dl/scenario1_straight/metrics_per_timestep_vehicle_24.csv
       • Plot: results_dl/scenario1_straight/trajectory_plot.png
       • Summary: results_dl/scenario1_straight/RESULTS_scenario1_DL.txt

2) Scenario 2 – Right Turn
   - Description: Vehicle moves straight and performs a right turn.
   - Duration: 12 s
   - ADE (per timestep): 0.01562 m
   - MR (per timestep): 0.00 %
   - ADE (per second): 0.02111 m
   - MR (per second): 0.00 %
   - Files:
       • GT vs Predicted CSV: results_dl/scenario2_rightturn/gt_vs_pred_vehicle_25.csv
       • Image: results_dl/scenario2_rightturn/img2_cratpred.png
       • Metrics per Timestep: results_dl/scenario2_rightturn/metrics_per_timestep_vehicle_25.csv
       • Plot: results_dl/scenario2_rightturn/trajectory_plot.png
       • Summary: results_dl/scenario2_rightturn/RESULTS_scenario2_DL.txt

3) Scenario 3 – Left Turn
   - Description: Vehicle moves straight and performs a left turn.
   - Duration: 12 s
   - ADE (per timestep): 0.01731 m
   - MR (per timestep): 0.00 %
   - ADE (per second): 0.02380 m
   - MR (per second): 0.00 %
   - Files:
       • GT vs Predicted CSV: results_dl/scenario3_leftturn/gt_vs_pred_vehicle_26.csv
       • Image: results_dl/scenario3_leftturn/img3_cratpred.png
       • Metrics per Timestep: results_dl/scenario3_leftturn/metrics_per_timestep_vehicle_26.csv
       • Plot: results_dl/scenario3_leftturn/trajectory_plot.png
       • Summary: results_dl/scenario3_leftturn/RESULTS_scenario3_DL.txt

--------------------------------------------------
Output Contents
--------------------------------------------------
Each scenario generates the following outputs:
- One CARLA image showing predicted and actual trajectories.
- One plot comparing predicted vs ground-truth positions.
- One CSV file with detailed trajectory data and computed ADE/FDE values per timestep and per second.
- One text summary file (.txt) describing the key results and observations for that scenario.

All outputs are automatically stored in the corresponding results/crat_pred/scenarioX/ folder.

--------------------------------------------------
Author & Context
--------------------------------------------------
This project was developed as part of a diploma thesis titled:

"Development of a Vehicle Trajectory Prediction Algorithm using Deep Learning Techniques"
by Evangelos Papaioannou, Democritus University of Thrace (D.U.Th.)

The CRAT-Pred model is based on a deep learning architecture that combines:
- LSTM encoders for temporal sequence modeling,
- Graph-based interaction reasoning using Graph Neural Networks (GNNs),
- Multi-Head Self-Attention for capturing spatial-temporal dependencies.

This hybrid architecture enables highly accurate, multi-modal trajectory forecasts
in complex and dynamic urban driving scenarios simulated in CARLA.

--------------------------------------------------
License
--------------------------------------------------
This project is distributed for academic and research purposes only.
For further use or publication, please cite the original CRAT-Pred paper and this diploma thesis.
