
CRAT-Pred Real-Time Trajectory Prediction in CARLA
==================================================

This project integrates the CRAT-Pred deep learning model with the CARLA simulator
to perform real-time multi-modal trajectory prediction for autonomous vehicles.
It tracks vehicles in the simulation, processes their motion, predicts future positions,
and evaluates the predictions with standard metrics.

Overview
--------
The project connects to the CARLA simulator in asynchronous mode, spawns a Tesla Model 3 as the ego vehicle along with optional NPCs, 
and starts collecting trajectory data after 3 seconds. A trained CRAT-Pred model predicts vehicle positions over a 3-second horizon at 20Hz.
The most accurate prediction mode is selected based on Final Displacement Error (FDE), then transformed back to global coordinates and visualized in CARLA.
Predicted and actual positions are saved to CSV, and performance metrics such as Average Displacement Error (ADE) and Miss Rate (MR) are calculated 
and stored in the results/ folder.

--------

Folder Structure
----------------
cratpred_project/
├── src/
│   ├── crat_pred_movement_predictor.py   # Main real-time prediction script
│   ├── model_loader.py                   # Loads CRAT-Pred from best checkpoint
│   ├── npc_spawner.py                    # Spawns random NPC vehicles in CARLA
│   ├── requirements.txt                  # Python dependencies
│   └── README.txt                       
├── results/
    ├── *.csv, *.png, *.txt               # Output metrics, predictions, and plots (created at runtime)

Dependencies
------------
Install the required Python packages using:

    pip install -r requirements.txt

Running the Simulation
----------------------
1. Start the CARLA simulator (must be running at localhost:2000).
2. Navigate to the `src/` directory.
3. Run the prediction script:

       python crat_pred_movement_predictor.py

This will:
- Load the CRAT-Pred model from the best checkpoint.
- Spawn the ego vehicle and additional NPCs.
- Begin real-time prediction and visualization.
- Log evaluation results to the `results/` folder.

Author & Context
This project was developed as part of a diploma thesis titled:

Ανάπτυξη Αλγορίθμου Πρόβλεψης Τροχιάς Οχημάτων με Τεχνικές Βαθιάς Μάθησης
(Development of a Vehicle Trajectory Prediction Algorithm using Deep Learning Techniques)

The CRAT-Pred model is based on a deep learning architecture that combines LSTM encoders, graph-based interaction modeling, and attention mechanisms to generate accurate, multi-modal trajectory forecasts in complex urban driving scenarios.

Author: Evangelos Papaioannou
