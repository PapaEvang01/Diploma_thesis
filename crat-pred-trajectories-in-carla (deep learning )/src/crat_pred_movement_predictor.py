"""
Trajectory Prediction in CARLA using CRAT-Pred
------------------------------------------------
This script connects to the CARLA simulator and performs real-time trajectory prediction
for an ego vehicle and optionally spawned NPC vehicles using the CRAT-Pred deep learning model.

Key Features:
- Connects to the CARLA simulator in asynchronous mode.
- Spawns a Tesla Model 3 as the ego vehicle, along with optional random NPC vehicles.
- Tracks vehicle positions over time, starting from t = 3s.
- Calculates displacements and applies rotation normalization based on the last motion vector.
- Feeds normalized displacements into the CRAT-Pred deep learning model to generate future trajectory predictions.
- Selects the most likely prediction mode using Final Displacement Error (FDE).
- Inversely rotates the predicted trajectory back to world coordinates and visualizes in CARLA.
- Computes detailed evaluation metrics:
    • ADE (Average Displacement Error) per timestep
    • Miss Rate (MR) based on a configurable distance threshold
- Logs all ground truth and predicted positions to CSV files for analysis.
- Saves per-timestep metrics in both CSV and TXT summary formats.
- Plots the predicted vs. ground truth trajectory and saves the plot as an image.
- Maintains a top-down spectator camera locked on the ego vehicle for visualization.
- Automatically destroys all spawned vehicles and exits gracefully on simulation stop.

This setup enables end-to-end evaluation and visualization of real-time, deep learning-based
multi-modal trajectory predictions in a realistic simulated driving environment.
"""

import time
import random
import numpy as np
import torch
import carla
import sys
import os
import math
import pandas as pd
import csv
import matplotlib.pyplot as plt

# Ensure Python can find the "model/" directory
sys.path.append("/home/user/PycharmProjects/crat-pred/model")

# Try importing the CRAT-Pred model
from crat_pred import CratPred
from npc_spawner import npc_spawner
from model_loader import ModelLoader

class MovementPredictor():
    def __init__(self):
      self.vehicles = []
      self.vehicle_histories = {}
      self.vehicle_origins = {}
      self.vehicle_positions_log = {}
      self.vehicle_centers_matrix = {}
      self.model = None

    def load_cratpred_model(self):
      """
      Loads the CRAT-Pred model using the best available checkpoint.

      Returns:
          CratPred: The trained CRAT-Pred model in evaluation mode.
      """
      best_checkpoint = self.get_best_checkpoint()
      print(f"Loading best checkpoint: {best_checkpoint}")

      self.model = CratPred.load_from_checkpoint(best_checkpoint, strict=False)  # Load the model without strict version checks
      self.model.eval()  # Set model to evaluation mode
    
    def get_best_checkpoint(self, checkpoint_dir=None):
        """
        Finds the best CRAT-Pred model checkpoint by selecting the one with the lowest fde_val.

        Args:
            checkpoint_dir (str, optional): Path to the directory containing model checkpoint files.
                Defaults to '~/crat-pred/lightning_logs/version_51/checkpoints'.

        Returns:
            str: The path to the best checkpoint file.
        """
        if checkpoint_dir is None:
             checkpoint_dir = os.path.expanduser('~/PycharmProjects/crat-pred/lightning_logs/version_2/checkpoints')

        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]

        def extract_fde(filename):
            try:
                return float(filename.split("fde_val=")[-1].split("-")[0])
            except (ValueError, IndexError):
                return float("inf")

        best_checkpoint = min(checkpoints, key=extract_fde)
        return os.path.join(checkpoint_dir, best_checkpoint)

    def connect_to_carla(self):
        """
        Connects to the CARLA simulator and ensures a valid connection.
        Returns:
            world (carla.World): The CARLA world object.
        """
        try:
            client = carla.Client('localhost', 2000)
            client.set_timeout(5.0)  # Set a timeout for connection
            self.world = client.get_world()
            print("Connected to CARLA.")
        except Exception as e:
            print(f"Error connecting to CARLA: {e}")
            exit()

    def ensure_async_mode(self):
      """
      Ensures that CARLA is running in asynchronous mode.
      Args:
          world (carla.World): The CARLA world object.
      """
      settings = self.world.get_settings()
      if settings.synchronous_mode:
          print("Disabling synchronous mode...")
          settings.synchronous_mode = False
          settings.fixed_delta_seconds = 0.05
          self.world.apply_settings(settings)

    def update_spectator_view(self, vehicle):
      """
      Updates the spectator camera to **follow the Ego Vehicle from directly above** while showing a **wider view of the road**.

      Args:
          world (carla.World): The CARLA world object.
          vehicle (carla.Actor): The Ego Vehicle actor.
      """
      spectator = self.world.get_spectator()
      transform = vehicle.get_transform()
      spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20), carla.Rotation(pitch=-90)))

    def draw_predicted_trajectory(self, prediction_time=5, step_size=0.5):
        """
        Draws the predicted waypoints for each vehicle based on CRAT-Pred predictions.

        - First **3 waypoints** are **green**.
        - Remaining waypoints are **blue**.
        - Displays the **first 10 predicted positions**.
        - Uses CARLA's debug draw utility.

        Args:
            world (carla.World): The CARLA world object.
            vehicles (list): List of CARLA vehicle actors whose trajectories are being visualized.
            prediction_time (int): Number of seconds to predict into the future.
            step_size (float): Time step between predicted waypoints.
        """
        max_waypoints = 100  # Display first 10 waypoints

        for vehicle in self.vehicles:
            vehicle_id = vehicle.id
            # Ensure we have predictions for this vehicle
            if f"predicted_{vehicle_id}" not in self.vehicle_positions_log:
                print(f"DEBUG: No predicted positions found for Vehicle {vehicle_id}")
                continue

            # Retrieve predicted trajectory positions
            predicted_positions = self.vehicle_positions_log[f"predicted_{vehicle_id}"]

            # Ensure there are predicted positions
            if not predicted_positions or len(predicted_positions) < 2:
                print(f"DEBUG: Not enough predicted positions for Vehicle {vehicle_id}")
                continue

            # **Select first 10 waypoints**
            waypoints_to_draw = predicted_positions[:max_waypoints]
            # Draw waypoints
            for idx, (waypoint_x, waypoint_y) in enumerate(waypoints_to_draw):
                # **First 3 waypoints = GREEN, the rest = BLUE**
                color = carla.Color(0, 255, 0) if idx < 3 else carla.Color(0, 0, 255)

                self.world.debug.draw_string(
                    carla.Location(float(waypoint_x), float(waypoint_y), float(vehicle.get_transform().location.z)),
                    "O",  # Waypoint symbol
                    draw_shadow=False,
                    color=color,
                    life_time=-1, 
                    persistent_lines=False  # No connecting lines
                )

            #print(f"[INFO] Visualized {len(waypoints_to_draw)} waypoints for Vehicle {vehicle_id}")

    def store_vehicle_positions(self, timestamp):
        """
        Stores the position of each vehicle at every second, starting from t = 3s.

        The function records each vehicle's (x, y) position in the `vehicle_positions_log`,
        allowing for real vs. predicted trajectory comparisons.

        Args:
            timestamp (int): The current simulation time in seconds.
            vehicles (list): A list of CARLA vehicle actors being tracked.

        Modifies:
            vehicle_positions_log (dict): Updates the log with vehicle positions at each timestamp.

        """
        self.vehicle_positions_log[timestamp] = {}

        for vehicle in self.vehicles:
            transform = vehicle.get_transform()
            self.vehicle_positions_log[timestamp][vehicle.id] = [transform.location.x, transform.location.y]

        print(f"Stored positions at timestamp {timestamp}s: {self.vehicle_positions_log[timestamp]}")  # Debug print

    def initialize_histories(self):
        for vehicle in self.vehicles:
            self.vehicle_histories[vehicle.id] = []
            self.vehicle_origins[vehicle.id] = [vehicle.get_transform().location.x, vehicle.get_transform().location.y]

    def calculate_rotation(self, vehicle_id):
        """
        Computes the rotation matrix that aligns the last movement vector
        of the vehicle's trajectory with the X-axis (standard CRAT-Pred normalization).

        Args:
            vehicle_id (int): The unique ID of the vehicle.

        Returns:
            torch.Tensor: A (1, 2, 2) rotation matrix.
        """

        # If we don't have enough position history for this vehicle (less than 2), return identity
        if vehicle_id not in self.vehicle_centers_matrix or self.vehicle_centers_matrix[vehicle_id].shape[1] < 2:
           # print(f"[DEBUG] Vehicle {vehicle_id}: Not enough history for rotation. Returning identity.")
            return torch.eye(2, dtype=torch.float32).unsqueeze(0)

        # Extract the last two recorded positions of the vehicle
        positions = self.vehicle_centers_matrix[vehicle_id].squeeze(0).numpy()
        prev, curr = positions[-2], positions[-1]

        # Compute the movement vector (difference between last two positions)
        dx, dy = curr[0] - prev[0], curr[1] - prev[1]
        norm = np.linalg.norm([dx, dy])  # Get the magnitude of the movement vector

        # If the vehicle hasn’t moved (zero vector), return identity matrix
        if norm == 0:
           # print(f"[DEBUG] Vehicle {vehicle_id}: No movement detected. Returning identity rotation.")
            return torch.eye(2, dtype=torch.float32).unsqueeze(0)

        # Compute the angle between the movement vector and the positive X-axis (in radians)
        angle = np.arctan2(dy, dx)
        angle_degrees = np.degrees(angle)  # For debugging

        # Construct a 2D rotation matrix to rotate the movement vector to align with the X-axis
        # We rotate by -angle to align the motion vector with the +X axis
        rotation = torch.tensor([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle),  np.cos(-angle)]
        ], dtype=torch.float32).unsqueeze(0)  # Shape: (1, 2, 2)

        return rotation

    def predict_future_cratpred(self, vehicle, timestamp):
        """
        Predicts future trajectories for a vehicle using the CRAT-Pred model with rotation normalization.

        Args:
            vehicle (carla.Actor): The CARLA vehicle instance being tracked.
            model (torch.nn.Module): The trained CRAT-Pred model.
            timestamp (int): The current simulation timestamp in seconds.

        Returns:
            list: A list of predicted (x, y) positions for the next timesteps.
        """
        print(f"\n[INFO] Processing Vehicle {vehicle.type_id} (ID: {vehicle.id}) at {timestamp}s")

        # ---  Get current position of the vehicle ---
        transform = vehicle.get_transform()
        current_x, current_y = transform.location.x, transform.location.y 

        if vehicle.id not in self.vehicle_origins:
            self.vehicle_origins[vehicle.id] = [current_x, current_y]
            print(f"[DEBUG] Stored First Recorded Position for Vehicle {vehicle.id}: (X: {current_x}, Y: {current_y})")

        self.vehicle_histories.setdefault(vehicle.id, []).append([current_x, current_y])
        
        if len(self.vehicle_histories[vehicle.id]) > 21:
            self.vehicle_histories[vehicle.id].pop(0)

        numeric_keys = sorted([t for t in self.vehicle_positions_log.keys() if isinstance(t, int)])
        centers_list = [self.vehicle_positions_log[t][vehicle.id] for t in numeric_keys if vehicle.id in self.vehicle_positions_log[t]]

        if not centers_list:
            centers_list.append([current_x, current_y])

        centers_matrix = torch.tensor(centers_list, dtype=torch.float32).unsqueeze(0)
        self.vehicle_centers_matrix[vehicle.id] = centers_matrix

        centers_matrix_np = centers_matrix.squeeze(0).numpy()

        # ---  Calculate displacements from positions (first difference) ---
        if timestamp == 3:
            print("[DEBUG] Skipping prediction at t=3s to establish initial position.")
            return

        if timestamp == 4:
            displacements = np.array([[centers_matrix_np[-1, 0] - centers_matrix_np[-2, 0],
                                            centers_matrix_np[-1, 1] - centers_matrix_np[-2, 1]]])
        elif centers_matrix_np.shape[0] > 1:
            displacements = np.diff(centers_matrix_np, axis=0)
        else:
            displacements = np.array([[0.0, 0.0]])
            print(f"[DEBUG] Not enough history for displacement at {timestamp}s. Using (0,0) for Vehicle {vehicle.id}.")

        displacements_tensor = torch.tensor(displacements, dtype=torch.float32).unsqueeze(0)
        ones_feature = torch.ones((displacements_tensor.shape[0], displacements_tensor.shape[1], 1), dtype=torch.float32)
        displacements_tensor = torch.cat((displacements_tensor, ones_feature), dim=-1)

        if 3 in self.vehicle_positions_log and vehicle.id in self.vehicle_positions_log[3]:
            origin_x, origin_y = self.vehicle_positions_log[3][vehicle.id]
        else:
            print(f"[ERROR] Missing t=3s position for Vehicle {vehicle.id}. Exiting prediction.")
            return None

        origin = torch.tensor([origin_x, origin_y], dtype=torch.float32).view(1, 2)
        centers = centers_matrix[:, -1, :].view(1, 2)

        # ---  Compute rotation matrix to align trajectory with the X-axis ---
        rotation_matrix = self.calculate_rotation(vehicle.id)
        if isinstance(rotation_matrix, np.ndarray):
            rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32)
        elif rotation_matrix.dim() == 3:
            rotation_matrix = rotation_matrix.squeeze(0)

        # --- Apply rotation to all inputs (displacements, center, origin) ---
        displ_rotated = torch.matmul(displacements_tensor[:, :, :2], rotation_matrix.T)
        centers_rotated = torch.matmul(centers, rotation_matrix.T)
        origin_rotated = torch.matmul(origin, rotation_matrix.T)
        displ_rotated = torch.cat((displ_rotated, ones_feature), dim=-1)

        # --- Build CRAT-Pred batch input ---
        batch = {
            "displ": (displ_rotated,),
            "centers": (centers_rotated,),
            "rotation": rotation_matrix.unsqueeze(0),
            "origin": origin_rotated
        }

       #  DEBUG: Check all normalized tensors before prediction
       # print(f"[DEBUG] displ_rotated shape: {displ_rotated.shape}")
       # print(f"[DEBUG] centers_rotated shape: {centers_rotated.shape}")
       # print(f"[DEBUG] origin_rotated shape: {origin_rotated.shape}")
       # print(f"[DEBUG] displ_rotated (first 3 steps):\n{displ_rotated[0, :3]}")
       # print(f"[DEBUG] centers_rotated:\n{centers_rotated}")

        # --- Run the prediction model ---
        with torch.no_grad():
            #print("[DEBUG] Entering torch.no_grad() context")

            agents_per_sample = [batch["displ"][0].shape[0]]
            #print(f"[DEBUG] agents_per_sample: {agents_per_sample}")

            #print("[DEBUG] Running encoder_lstm...")
            encoder_out = self.model.encoder_lstm(batch["displ"][0], agents_per_sample)
            #print("[DEBUG] encoder_lstm output shape:", encoder_out.shape)

            last_hidden_state = encoder_out  # shape: [batch_size, 128]

            #print("[DEBUG] Running agent_gnn...")
            agent_gnn_out = self.model.agent_gnn(
                last_hidden_state, batch["centers"][0], agents_per_sample
            )
            #print("[DEBUG] agent_gnn output shape:", agent_gnn_out.shape)

            #print("[DEBUG] Running multihead_self_attention...")
            attention_out_batch = self.model.multihead_self_attention(agent_gnn_out, agents_per_sample)
            #print("[DEBUG] attention_out_batch length:", len(attention_out_batch))
            #if len(attention_out_batch) > 0:
                #print("[DEBUG] attention_out_batch[0] shape:", attention_out_batch[0].shape)

            #  Stack attention outputs to get shape: (batch_size, latent_dim)
            attention_out = torch.stack(attention_out_batch)
            #print("[DEBUG] stacked attention_out shape:", attention_out.shape)

            #print("[DEBUG] Running decoder_residual...")
            predictions = self.model.decoder_residual(attention_out, self.model.is_frozen)
            #print("[DEBUG] decoder_residual output shape:", predictions.shape)


        if predictions is None:
            print(f"[ERROR] CRAT-Pred returned None for Vehicle {vehicle.id}. Skipping prediction.")
            return None
        
        predictions = predictions.squeeze(0).cpu().numpy()  # (num_modes, num_steps, 2)
        predictions = predictions.reshape(predictions.shape[0], -1, 2)

        if predictions.ndim == 4 and predictions.shape[0] == 1:
            predictions = predictions[0]  # squeeze batch
        num_modes = predictions.shape[0]

        #print(predictions.ndim)
        if predictions.ndim != 3 or num_modes == 0:
            print(f"[ERROR] Invalid prediction shape: {predictions.shape}. Skipping prediction.")
            return None

        # ---  Evaluate FDE (Final Displacement Error) and pick the best mode ---
        real_last_x, real_last_y = centers.numpy().flatten()
        final_positions = predictions[:, -1, :]  # (num_modes, 2)
        fde_scores = np.linalg.norm(final_positions - [real_last_x, real_last_y], axis=1)
        # print(f"[DEBUG] FDE Scores: {fde_scores}")

        best_mode_index = int(np.argmin(fde_scores))
        best_fde_score = fde_scores[best_mode_index] if fde_scores.ndim == 1 else fde_scores.flatten()[best_mode_index]
        #print(f"[DEBUG] Selected best mode index: {best_mode_index} with FDE = {best_fde_score:.4f}")

        # --- Select best prediction mode (already reshaped to [num_modes, num_steps, 2]) ---
        best_mode_displacements = predictions[best_mode_index]  # shape: (num_steps, 2)

        # --- Convert to tensor (local frame positions) ---
        pred_tensor = torch.tensor(best_mode_displacements, dtype=torch.float32)

        # --- Rotate back to world frame and add back the last known center (i.e., global origin) ---
        pred_world = torch.matmul(pred_tensor, rotation_matrix) + centers  # shape: (num_steps, 2)

        # --- Convert to Python list of (x, y) positions ---
        predicted_positions = pred_world.cpu().numpy().tolist()

        # --- Store and print ---
        self.vehicle_positions_log[f"predicted_{vehicle.id}"] = predicted_positions

        print(f"[INFO] Vehicle {vehicle.id} ({vehicle.type_id}) at {timestamp}s -> Current Position: (X: {current_x:.3f}, Y: {current_y:.3f})")
        print(f"[INFO] Predicted positions for next timesteps: {predicted_positions[:5]}...")

        return predicted_positions

    def save_gt_vs_first_prediction_by_frame(self, vehicle_id, filename="gt_vs_pred_frames.csv"):
        """
        Saves GT and first predicted positions for every timestamp where both exist.
        Uses prediction from t and GT from t+1.
        Each row: [Timestep, GT_X, GT_Y, Pred_X, Pred_Y]
        """
        header = ["Timestep", "GT_X", "GT_Y", "Pred_X", "Pred_Y"]
        rows = [header]

        pred_log = self.vehicle_positions_log.get("pred_log", {}).get(vehicle_id, {})
        timestamps = sorted(pred_log.keys())

        for t in timestamps:
            pred_x, pred_y = pred_log[t]

            gt_time = t + 1
            if gt_time in self.vehicle_positions_log and vehicle_id in self.vehicle_positions_log[gt_time]:
                gt_x, gt_y = self.vehicle_positions_log[gt_time][vehicle_id]
            else:
                continue  # Missing GT

            row = [t, round(gt_x, 4), round(gt_y, 4), round(pred_x, 4), round(pred_y, 4)]
            rows.append(row)

        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        print(f"[INFO] Saved frame-level GT vs prediction CSV to {filename}")

    def compute_metrics_per_timestep(self, csv_path, threshold=2.0, output_csv="metrics_per_timestep.csv"):
        """
        Computes ADE and Miss Rate per timestep and saves them to a CSV.

        Args:
            csv_path (str): Path to the CSV file with columns:
                            Timestep, GT_X, GT_Y, Pred_X, Pred_Y
            threshold (float): Distance threshold (in meters) for Miss Rate
            output_csv (str): Path to the output CSV with per-timestep metrics
        """
        df = pd.read_csv(csv_path)

        results = []
        grouped = df.groupby("Timestep")

        for timestep, group in grouped:
            gt = group[["GT_X", "GT_Y"]].values
            pred = group[["Pred_X", "Pred_Y"]].values
            distances = np.linalg.norm(gt - pred, axis=1)

            ade = np.mean(distances)
            mr = np.mean(distances > threshold)  # Miss rate = % of predictions > threshold

            results.append({"Timestep": timestep, "ADE": ade, "MissRate": mr})

        # Save to CSV
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"[INFO] Saved per-timestep ADE & MissRate to {output_csv}")

    def plot_trajectories(self, csv_path, save_path="trajectory_plot.png"):
        """
        Plots the ground truth and predicted trajectories from a CSV file.

        Args:
            csv_path (str): Path to the CSV file with columns GT_X, GT_Y, Pred_X, Pred_Y.
            save_path (str): Path to save the resulting plot image.

        Saves:
            A PNG image comparing ground truth vs. predicted trajectories.
        """
        df = pd.read_csv(csv_path)

        plt.figure(figsize=(6, 6))
        plt.plot(df["GT_X"], df["GT_Y"], label="Ground Truth", marker='o')
        plt.plot(df["Pred_X"], df["Pred_Y"], label="Predicted", marker='x')
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Predicted vs Ground Truth Trajectory")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"[INFO] Trajectory plot saved as: {save_path}")

if __name__ == '__main__':
    print("[INFO] Connecting to CARLA...")
    predictor = MovementPredictor()
    predictor.connect_to_carla()
    predictor.ensure_async_mode()

    try:
        # Spawn vehicles
        spawner = npc_spawner(predictor.world)
        predictor.vehicles, predictor.vehicle_origins = spawner.spawn_npcs()
        predictor.initialize_histories()

        # Load CRAT-Pred model
        model_loader = ModelLoader()
        predictor.model = model_loader.load_cratpred_model()

        print("[INFO] All vehicles spawned. Waiting 3 seconds for movement before predictions start...\n")
        time.sleep(3)

        for i in range(100000):
            predictor.store_vehicle_positions(i)

            if i >= 3:
                for vehicle in predictor.vehicles:
                    if vehicle.id not in predictor.vehicle_histories:
                        print(f"[WARN] Vehicle {vehicle.id} not found in history. Skipping.")
                        continue

                    predicted = predictor.predict_future_cratpred(vehicle, i)

                    # Store the first predicted point per timestamp
                    if predicted and len(predicted) >= 1:
                        if "pred_log" not in predictor.vehicle_positions_log:
                            predictor.vehicle_positions_log["pred_log"] = {}
                        if vehicle.id not in predictor.vehicle_positions_log["pred_log"]:
                            predictor.vehicle_positions_log["pred_log"][vehicle.id] = {}
                        predictor.vehicle_positions_log["pred_log"][vehicle.id][i] = predicted[0]

                    predictor.draw_predicted_trajectory(prediction_time=5, step_size=0.5)

            predictor.update_spectator_view(predictor.vehicles[-1])

            timestamp = predictor.world.wait_for_tick()
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")

    finally:
        print("[INFO] Cleaning up...")

        # Save per-frame GT vs first prediction (for ego vehicle)
        ego_vehicle = predictor.vehicles[-1]
        csv_filename = f"gt_vs_pred_vehicle_{ego_vehicle.id}.csv"
        predictor.save_gt_vs_first_prediction_by_frame(ego_vehicle.id, filename=csv_filename)

        # Compute per-timestep metrics and save
        metrics_csv = f"metrics_per_timestep_vehicle_{ego_vehicle.id}.csv"
        predictor.compute_metrics_per_timestep(csv_filename, threshold=2.0, output_csv=metrics_csv)

        # Also write summary metrics to .txt
        metrics_summary = f"metrics_summary_vehicle_{ego_vehicle.id}.txt"
        metrics_df = pd.read_csv(metrics_csv)
        mean_ade = metrics_df["ADE"].mean()
        mean_mr = metrics_df["MissRate"].mean()

        with open(metrics_summary, "w") as f:
            f.write(f"Average ADE over {len(metrics_df)} timesteps: {mean_ade:.4f} meters\n")
            f.write(f"Average Miss Rate @2.0m: {mean_mr:.4f}\n")

        # Plot trajectories
        predictor.plot_trajectories(csv_filename)

        # Clean up vehicles
        for v in predictor.vehicles:
            v.destroy()

        print(f"[INFO] Trajectory CSV saved as: {csv_filename}")
        print(f"[INFO] Per-timestep metrics saved as: {metrics_csv}")
        print(f"[INFO] Metrics summary saved as: {metrics_summary}")
        print("[INFO] All vehicles destroyed. Simulation terminated.")
        os._exit(0)
