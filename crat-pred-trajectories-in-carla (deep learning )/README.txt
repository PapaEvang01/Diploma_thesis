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
cratpred_project/
├── src/
│   ├── crat_pred_movement_predictor.py   # Main real-time prediction script
│   ├── model_loader.py                   # Loads CRAT-Pred from best checkpoint
│   ├── npc_spawner.py                    # Spawns random NPC vehicles in CARLA
│   ├── requirements.txt                  # Python dependencies
│   └── README.txt
├── results/
│   ├── *.csv, *.png, *.csv               # Output metrics, predictions, and plots (created at runtime)

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
       • Image: results/crat_pred/scenario1_straight/carla_view.png
       • Plot: results/crat_pred/scenario1_straight/gt_vs_pred.png
       • CSV: results/crat_pred/scenario1_straight/trajectories.csv
       • Summary: results/crat_pred_scenario1.txt

2) Scenario 2 – Right Turn
   - Description: Vehicle moves straight and performs a right turn.
   - Duration: 12 s
   - ADE (per timestep): 0.01562 m
   - MR (per timestep): 0.00 %
   - ADE (per second): 0.02111 m
   - MR (per second): 0.00 %
   - Files:
       • Image: results/crat_pred/scenario2_rightturn/carla_view.png
       • Plot: results/crat_pred/scenario2_rightturn/gt_vs_pred.png
       • CSV: results/crat_pred/scenario2_rightturn/trajectories.csv
       • Summary: results/crat_pred_scenario2.txt

3) Scenario 3 – Left Turn
   - Description: Vehicle moves straight and performs a left turn.
   - Duration: 12 s
   - ADE (per timestep): 0.01731 m
   - MR (per timestep): 0.00 %
   - ADE (per second): 0.02380 m
   - MR (per second): 0.00 %
   - Files:
       • Image: results/crat_pred/scenario3_leftturn/carla_view.png
       • Plot: results/crat_pred/scenario3_leftturn/gt_vs_pred.png
       • CSV: results/crat_pred/scenario3_leftturn/trajectories.csv
       • Summary: results/crat_pred_scenario3.txt

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
