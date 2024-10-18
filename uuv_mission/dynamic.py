import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pandas as pd

class Submarine:
    def __init__(self):
        self.depth = 0.0

    def get_depth(self):
        return self.depth

    def transition(self, action, disturbance):
        self.depth += action + disturbance

class Trajectory:
    def __init__(self, positions):
        # Ensure positions is a 2-dimensional array
        self.position = np.array(positions).reshape(-1, 2)
        
    def plot(self):
        plt.plot(self.position[:, 0], self.position[:, 1])
        plt.show()

    def plot_completed_mission(self, mission):
        x_values = np.arange(len(self.position))
        min_depth = np.min(self.position[:, 1])
        max_height = np.max(self.position[:, 1])
        
        plt.figure(figsize=(20, 5))
        plt.fill_between(x_values, mission.cave_depth, min_depth*np.ones(len(x_values)), 
                         color='saddlebrown', alpha=0.3)
        plt.fill_between(x_values, max_height*np.ones(len(x_values)), mission.cave_height, 
                         color='saddlebrown', alpha=0.3)
        plt.plot(self.position[:, 0], self.position[:, 1], label='Trajectory')
        plt.plot(mission.reference, 'r', linestyle='--', label='Reference')
        plt.legend(loc='upper right')
        plt.xlabel('Time')
        plt.ylabel('Depth')
        plt.title('Completed Mission Trajectory')
        plt.show()

@dataclass
class Mission:
    reference: np.ndarray
    cave_height: np.ndarray
    cave_depth: np.ndarray

    @classmethod
    def random_mission(cls, duration: int, scale: float):
        (reference, cave_height, cave_depth) = generate_reference_and_limits(duration, scale)
        return cls(reference, cave_height, cave_depth)

    @classmethod
    def from_csv(cls, file_name: str):
        # Use pandas to read the CSV file
        df = pd.read_csv(file_name)
        
        # Extract the relevant columns
        reference = df['reference'].to_numpy()
        cave_height = df['cave_height'].to_numpy()
        cave_depth = df['cave_depth'].to_numpy()
        
        # Return an instance of Mission
        return cls(reference, cave_height, cave_depth)

class ClosedLoop:
    def __init__(self, plant, controller):
        self.plant = plant
        self.controller = controller

    def simulate(self, mission, disturbances):
        positions = np.zeros((len(mission.reference), 2))  # Ensure positions is 2-dimensional
        actions = np.zeros(len(mission.reference))

        for t in range(len(mission.reference)):
            observation_t = self.plant.get_depth()
            error = mission.reference[t] - observation_t
            actions[t] = self.controller.compute_action(error)
            self.plant.transition(actions[t], disturbances[t])
            positions[t, 0] = t  # Assuming time or some other x-axis value
            positions[t, 1] = self.plant.get_depth()

        return Trajectory(positions)

    def simulate_with_random_disturbances(self, mission, variance=0.5):
        disturbances = np.random.normal(0, variance, len(mission.reference))
        return self.simulate(mission, disturbances)