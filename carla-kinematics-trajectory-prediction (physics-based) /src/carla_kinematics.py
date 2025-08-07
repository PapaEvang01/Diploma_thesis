"""
carla_kinematics.py
-------------------

Real-Time Vehicle Tracking & Trajectory Prediction in CARLA using Kinematic Equations.

This script performs the following tasks:
- Connects to the CARLA simulator in asynchronous mode.
- Spawns an ego vehicle (Tesla Model 3) and optionally NPC vehicles.
- Predicts future positions using constant velocity and yaw-based kinematics.
- Visualizes predictions in real time within the CARLA simulation.
- Evaluates prediction accuracy using ADE (Average Displacement Error) and MissRate.
- Logs metrics per second into a CSV and generates trajectory plots.
- Dynamically follows the ego vehicle with a top-down spectator camera.
- Gracefully cleans up all actors when the simulation ends.

"""

# === Imports ===
import carla
import random
import math
import time
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd

# Constants
INITIAL_DELAY = 3.0
UPDATE_INTERVAL = 0.1  # Main update interval (0.1s)
PREDICTION_HORIZON = 1.0  # Predict 1 second into the future
STEP_INTERVAL = UPDATE_INTERVAL  # Each step = 0.1s
STEPS_PER_SECOND = int(PREDICTION_HORIZON / STEP_INTERVAL)  # 10 steps per prediction


# === CARLA Setup Functions ===

def connect_to_carla():
    """
    Connects to the CARLA simulator and ensures a valid connection.
    Returns:
        world (carla.World): The CARLA world object.
    """
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)  # Set a timeout for connection
        world = client.get_world()
        return world
    except Exception as e:
        print(f"Error connecting to CARLA: {e}")
        exit()

def ensure_async_mode(world):
    """
    Ensures that CARLA is running in asynchronous mode.
    Args:
        world (carla.World): The CARLA world object.
    """
    settings = world.get_settings()
    if settings.synchronous_mode:
        print("Disabling synchronous mode...")
        settings.synchronous_mode = False
        world.apply_settings(settings)

def get_spawn_points(world):
    """
    Retrieves and shuffles available spawn points in the CARLA world.
    Returns:
        list: A list of shuffled spawn points.
    """
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    if not spawn_points:
        print("No available spawn points. Exiting.")
        exit()
    return spawn_points

def spawn_ego_vehicle(world, spawn_points, blueprint_library):
    """
    Spawns the ego vehicle (Tesla Model 3) in the CARLA world.
    Returns:
        carla.Actor: The spawned ego vehicle actor.
    """
    ego_vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    if not spawn_points or not ego_vehicle_bp:
        print("Could not spawn Ego Vehicle. Exiting.")
        exit()
    
    ego_spawn_point = spawn_points.pop()
    ego_vehicle = world.try_spawn_actor(ego_vehicle_bp, ego_spawn_point)

    if ego_vehicle:
        ego_vehicle.set_autopilot(True)
        print(f"Ego Vehicle Spawned at {ego_spawn_point.location}")
        return ego_vehicle
    else:
        print("Failed to spawn Ego Vehicle.")
        exit()

def spawn_npc_vehicles(world, spawn_points, blueprint_library, num_vehicles=20):
    """
    Spawns NPC vehicles at available spawn points.
    Returns:
        list: A list of spawned NPC vehicle actors.
    """
    vehicles_list = []
    vehicle_blueprints = [bp for bp in blueprint_library.filter("vehicle.*") if bp.has_attribute('number_of_wheels')]

    for i in range(min(num_vehicles, len(spawn_points))):
        vehicle_bp = random.choice(vehicle_blueprints)
        spawn_point = spawn_points.pop()
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

        if vehicle:
            vehicle.set_autopilot(True)
            vehicles_list.append(vehicle)
            print(f"NPC Spawned: {vehicle.type_id} at {spawn_point.location}")
        else:
            print(f"Failed to spawn NPC vehicle {i+1}")

    return vehicles_list


def draw_predicted_trajectory(world, vehicles, prediction_time=1.0, step_size=0.1, fade_time=0.35):
    """
    Draws the predicted trajectory for each vehicle based on velocity and yaw.

    Args:
        world (carla.World): The CARLA world object.
        vehicles (list): List of vehicle actors (usually just the ego).
        prediction_time (float): Total prediction horizon in seconds.
        step_size (float): Time interval between predicted points (e.g., 0.1s).
        fade_time (float): How long each marker should remain visible (in seconds).
    """
    for vehicle in vehicles:
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()

        x_init = transform.location.x
        y_init = transform.location.y
        speed = math.sqrt(velocity.x**2 + velocity.y**2)
        yaw = transform.rotation.yaw
        yaw_rad = math.radians(yaw)

        color = carla.Color(255, 0, 0)  # Red for ego

        for t in range(1, int(prediction_time / step_size) + 1):
            time_step = t * step_size
            x_future = x_init + speed * math.cos(yaw_rad) * time_step
            y_future = y_init + speed * math.sin(yaw_rad) * time_step

            world.debug.draw_string(
                carla.Location(x_future, y_future, transform.location.z),
                "O",
                draw_shadow=False,
                color=color,
                life_time=fade_time,
                persistent_lines=False
            )


# === Utility Functions ===

def calculate_sides(hypotenuse, angle):
    """
    Calculates the two sides (x, y) of a right triangle given the hypotenuse and an angle.

    Args:
        hypotenuse (float): The hypotenuse length (distance behind the vehicle).
        angle (float): The yaw angle of the vehicle in degrees.

    Returns:
        tuple: The lengths of the two sides (delta_x, delta_y).
    """
    angle_radians = math.radians(angle)
    delta_x = hypotenuse * math.cos(angle_radians)
    delta_y = hypotenuse * math.sin(angle_radians)
    return delta_x, delta_y

def update_spectator_view(world, vehicle):
    """
    Updates the spectator camera to follow the Ego Vehicle from directly above.

    Args:
        world (carla.World): The CARLA world object.
        vehicle (carla.Actor): The Ego Vehicle actor.
    """
    spectator = world.get_spectator()
    vehicle_transform = vehicle.get_transform()
    
    follow_distance = -3   # Forward offset
    follow_height = 35     # Top-down height

    delta_x, delta_y = calculate_sides(follow_distance, vehicle_transform.rotation.yaw)

    spectator_transform = carla.Transform(
        vehicle_transform.location + carla.Location(x=-delta_x, y=-delta_y, z=follow_height),
        carla.Rotation(yaw=vehicle_transform.rotation.yaw, pitch=-90)
    )

    spectator.set_transform(spectator_transform)


# === Main Tracking Logic ===

def track_vehicles(vehicles_list, ego_vehicle, world):
    """
    Tracks the ego vehicle, predicts future positions, and evaluates kinematic prediction accuracy
    using ADE and MissRate metrics. Visualizes trajectory in real time and logs metrics once per second.

    Args:
        vehicles_list (list): List of NPC vehicles (can be empty).
        ego_vehicle (carla.Actor): The ego vehicle actor.
        world (carla.World): The CARLA world object.
    """
    ego_actual_traj = []       # Stores actual (GT) positions of the ego vehicle
    ego_predicted_traj = []    # Stores future predicted trajectories (lists of (x, y))
    start_time = time.time()
    last_metric_time = 0.0     # Controls 1-second interval for metric logging

    # === Prediction configuration ===
    PREDICTION_HORIZON = 1.0       # seconds
    STEP_INTERVAL = 0.1            # seconds per prediction step
    STEPS_PER_SECOND = int(PREDICTION_HORIZON / STEP_INTERVAL)  # = 10
    UPDATE_INTERVAL = 0.05         # how often this function updates (matches your sim rate)

    # Initial delay to allow ego vehicle to begin moving
    print("[INFO] Waiting for 3 seconds before starting tracking...")
    time.sleep(3.0)

    try:
        while True:
            current_time = time.time() - start_time

            # === Get current ego vehicle data ===
            transform = ego_vehicle.get_transform()
            velocity = ego_vehicle.get_velocity()
            x_now = transform.location.x
            y_now = transform.location.y
            speed = math.sqrt(velocity.x**2 + velocity.y**2)
            yaw_rad = math.radians(transform.rotation.yaw)

            # Store ground truth position
            ego_actual_traj.append((x_now, y_now))

            # Print current ground truth position
            print(f"\n[Time {current_time:.1f}s] Current GT position: x = {x_now:.2f}, y = {y_now:.2f}")

            # === Predict future trajectory (1 second ahead, with 0.1s intervals) ===
            future_pred = []
            for t in range(1, STEPS_PER_SECOND + 1):
                x_pred = x_now + speed * math.cos(yaw_rad) * (t * STEP_INTERVAL)
                y_pred = y_now + speed * math.sin(yaw_rad) * (t * STEP_INTERVAL)
                future_pred.append((x_pred, y_pred))

            # Store the prediction sequence
            ego_predicted_traj.append(future_pred)

            # Print predicted trajectory to console
            print("[Predicted Future Positions (next 1s)]:")
            for i, (px, py) in enumerate(future_pred):
                print(f"  Step {i + 1}: x = {px:.2f}, y = {py:.2f}")

            # === Visualize in CARLA ===
            draw_predicted_trajectory(world, [ego_vehicle], prediction_time=1.0, step_size=0.1)
            update_spectator_view(world, ego_vehicle)

            # === Compute metrics if 1 second has passed and enough data is available ===
            if current_time - last_metric_time >= 1.0 and \
               len(ego_predicted_traj) >= STEPS_PER_SECOND and \
               len(ego_actual_traj) >= STEPS_PER_SECOND:

                # Get prediction from 1 second ago
                past_pred = ego_predicted_traj[-STEPS_PER_SECOND]  # 10 steps ago

                # Get current GT for the same prediction horizon
                recent_gt = ego_actual_traj[-STEPS_PER_SECOND:]

                # Compute ADE and MissRate per step
                metrics_per_timestep = compute_trajectory_errors(past_pred, recent_gt)

                # Log and print
                rounded_time = round(current_time, 1)
                print(f"\n[Metrics at {rounded_time}s]")
                for i, (ade_t, mr_t) in enumerate(metrics_per_timestep):
                    print(f"  Step {i + 1}: ADE = {ade_t:.2f}, MissRate = {mr_t}")

                # === Log metrics and plot full trajectory ===
                log_metrics_to_csv(rounded_time, past_pred, recent_gt, metrics_per_timestep)
                plot_trajectories("metrics.csv", save_path="trajectory_plot.png")
                last_metric_time = current_time

            # === Print tracking summary ===
            print(f"Tracking at t = {current_time:.1f}s | Speed: {speed:.2f} m/s")

            # === Wait before next update ===
            time.sleep(UPDATE_INTERVAL)

    except KeyboardInterrupt:
        print("\nSimulation manually stopped.")
        cleanup_vehicles(vehicles_list, ego_vehicle)


# === Evaluation Utilities ===

def compute_trajectory_errors(predicted, ground_truth):
    """
    Computes ADE and MissRate for each timestep between predicted and ground truth trajectories.

    Args:
        predicted (list of tuples): List of predicted (x, y) positions.
        ground_truth (list of tuples): List of actual (x, y) positions.

    Returns:
        list: List of tuples (ADE_t, MissRate_t) for each timestep
    """
    if len(predicted) != len(ground_truth):
        print("Trajectory lengths do not match.")
        return []

    errors = []
    results = []
    for i, (p, gt) in enumerate(zip(predicted, ground_truth)):
        dist = math.hypot(p[0] - gt[0], p[1] - gt[1])
        errors.append(dist)
        ade = np.mean(errors)        # Mean of distances up to this point
        mr = 1 if dist > 2.0 else 0  # Miss rate at current timestep
        results.append((ade, mr))

    return results

def log_metrics_to_csv(time_sec, predicted, ground_truth, metrics_list, filename="metrics.csv"):
    """
    Logs predicted and ground truth positions along with ADE and MissRate per timestep to a CSV.

    Args:
        time_sec (float): Current time in seconds.
        predicted (list of (x, y)): Predicted positions.
        ground_truth (list of (x, y)): Ground truth positions.
        metrics_list (list of (ade, missrate)): Error metrics per timestep.
        filename (str): Output CSV file name.
    """
    header = ["Time (s)", "Timestep", "GT_X", "GT_Y", "Pred_X", "Pred_Y", "ADE", "MissRate"]
    write_header = not os.path.exists(filename)

    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        for i, ((gt_x, gt_y), (pred_x, pred_y), (ade, miss)) in enumerate(zip(ground_truth, predicted, metrics_list)):
            writer.writerow([
                int(time_sec),
                i + 1,
                round(gt_x, 3),
                round(gt_y, 3),
                round(pred_x, 3),
                round(pred_y, 3),
                round(ade, 3),
                miss
            ])

def plot_trajectories(csv_path, save_path="trajectory_plot.png"):
    """
    Plots the ground truth and predicted trajectories from a CSV file.

    Args:
        csv_path (str): Path to the CSV file with columns GT_X, GT_Y, Pred_X, Pred_Y.
        save_path (str): Path to save the resulting plot image.

    Saves:
        A PNG image comparing ground truth vs. predicted trajectories.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(6, 6))
    plt.plot(df["GT_X"], df["GT_Y"], label="Ground Truth", marker='o', markersize=2, linestyle='-', linewidth=0.8)
    plt.plot(df["Pred_X"], df["Pred_Y"], label="Predicted", marker='x', markersize=2, linestyle='--', linewidth=0.8)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Predicted vs Ground Truth Trajectory")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Trajectory plot saved as: {save_path}")


# === Cleanup ===

def cleanup_vehicles(vehicles_list, ego_vehicle):
    """
    Removes all spawned vehicles from the simulation.
    Args:
        vehicles_list (list): List of spawned NPC vehicles.
        ego_vehicle (carla.Actor): The ego vehicle actor.
    """
    print("\nRemoving all spawned vehicles...")
    for vehicle in vehicles_list:
        vehicle.destroy()

    if ego_vehicle:
        ego_vehicle.destroy()
        print("Ego Vehicle removed.")

    print("All vehicles removed.")


# === Main ===

def main():
    """
    Main function that initializes the CARLA world, spawns only the ego vehicle, and starts vehicle tracking.
    Runs indefinitely until interrupted.
    """
    # Connect to CARLA and ensure asynchronous mode
    world = connect_to_carla()
    ensure_async_mode(world)

    # Get spawn points and vehicle blueprints
    spawn_points = get_spawn_points(world)
    blueprint_library = world.get_blueprint_library()

    # Spawn only the ego vehicle
    ego_vehicle = spawn_ego_vehicle(world, spawn_points, blueprint_library)
    vehicles_list = []  # No NPCs

    # Wait for initial delay
    print("[INFO] Waiting for 3 seconds before checking movement...")
    time.sleep(3.0)

    # Wait until ego vehicle starts moving
    print("[INFO] Waiting for ego vehicle to start moving...")
    while True:
        velocity = ego_vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2)
        if speed > 0.1:  # Minimum threshold to consider 'moving'
            print(f"[INFO] Ego vehicle started moving at speed: {speed:.2f} m/s")
            break
        time.sleep(0.1)

    # Start tracking only the ego vehicle
    track_vehicles(vehicles_list, ego_vehicle, world)


if __name__ == "__main__":
    main()



