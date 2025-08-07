# npc_spawner.py
# ----------------
# This module handles the spawning of random NPC vehicles in the CARLA simulator.
# It selects random vehicle types and spawn points, spawns them into the world,
# and returns their actor references and initial positions.

import random  # Used for shuffling spawn points and randomly selecting vehicle types
import time    # Used for adding delays (e.g., waiting for vehicles to move)


class npc_spawner(object):
    """Spawns one or more NPC vehicles in the CARLA simulator."""

    def __init__(self, world, number_of_npcs=1):
        # Store reference to the CARLA world
        self.world = world
        # Get all valid spawn points from the map
        self.spawn_points = self.world.get_map().get_spawn_points()
        # Get all available vehicle blueprints from the CARLA library
        self.blueprint_library = self.world.get_blueprint_library()
        # Number of NPCs to spawn
        self.number_of_npcs = number_of_npcs

    def spawn_npcs(self):
        # Shuffle spawn points to randomize spawn locations
        random.shuffle(self.spawn_points)
        vehicles = []           # List to hold spawned vehicle actors
        vehicle_origins = {}    # Dictionary to store initial [x, y] positions per vehicle ID

        if not self.spawn_points:
            print("No available spawn points. Exiting.")
            exit()

        # Filter valid vehicle blueprints (those with wheels)
        vehicle_blueprints = [bp.id for bp in self.blueprint_library.filter("vehicle.*")
                              if bp.has_attribute('number_of_wheels')]

        # Spawn each vehicle
        for i in range(min(self.number_of_npcs, len(self.spawn_points))):
            vehicle_type = random.choice(vehicle_blueprints)  # Pick a random vehicle type
            vehicle = self.spawn_vehicle(vehicle_type)        # Try to spawn it

            if vehicle:
                vehicles.append(vehicle)
                transform = vehicle.get_transform()
                # Store initial location for this vehicle
                vehicle_origins[vehicle.id] = [transform.location.x, transform.location.y]

        if not vehicles:
            print("No vehicles spawned. Exiting.")
            return None, None

        print("\nAll vehicles spawned. Waiting 3 seconds for movement before predictions start...\n")
        return vehicles, vehicle_origins

    def spawn_vehicle(self, vehicle_type):
        # Get the blueprint of the requested vehicle type
        vehicle_bp = self.blueprint_library.find(vehicle_type)

        # If we run out of spawn points or blueprint is invalid
        if not self.spawn_points or not vehicle_bp:
            print(f"Could not spawn {vehicle_type}. Exiting.")
            return None

        # Pop the next spawn point
        spawn_point = self.spawn_points.pop()

        # Try to spawn the actor in the world
        vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

        if vehicle:
            # Enable autopilot
            vehicle.set_autopilot(True)
            print(
                f"Vehicle {vehicle_type} Spawned at Location(x={spawn_point.location.x:.6f}, "
                f"y={spawn_point.location.y:.6f}, z={spawn_point.location.z:.6f})")
            return vehicle  # Return the successfully spawned vehicle instance
        else:
            print(f"Failed to spawn {vehicle_type}. Retrying at a new location...")
            return None

