import numpy as np
from pymdp.agent import Agent
import os
import sys
import logging

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

logging.basicConfig(level=logging.INFO)

class SpatialNavigationAgent:
    """
    Agent responsible for navigating space and providing navigational cues.
    """

    def __init__(self, spatial_navigation_kd, emotion_kd):
        """
        Initializes the agent with references to the knowledge domains.
        """
        self.spatial_navigation_kd = spatial_navigation_kd
        self.emotion_kd = emotion_kd

        # Define the generative model parameters
        self.A = self._define_A()
        self.B = self._define_B()
        self.C = self._define_C()
        self.D = self._define_D()

        # Initialize the pymdp Agent
        self.agent = Agent(A=self.A, B=self.B, C=self.C, D=self.D)

    def _define_A(self):
        """
        Define the observation likelihood matrix A.
        """
        A_spatial = self.spatial_navigation_kd.A
        A_emotion = np.eye(len(self.emotion_kd.emotion_names))
        A_combined = np.block([
            [A_spatial],
            [A_emotion]
        ])
        A_combined = A_combined / A_combined.sum(axis=0, keepdims=True)  # Normalize A
        return A_combined

    def _define_B(self):
        """
        Define the state transition matrix B.
        """
        B_spatial = self.spatial_navigation_kd.B
        B_emotion = np.eye(len(self.emotion_kd.emotion_names))
        B_combined = np.block([
            [B_spatial],
            [B_emotion]
        ])
        B_combined = B_combined / B_combined.sum(axis=1, keepdims=True)  # Normalize B
        return B_combined

    def _define_C(self):
        """
        Define the preference matrix C.
        """
        # Define C based on the agent's goals/preferences
        C = np.ones((len(self.spatial_navigation_kd.state_names), 1))
        return C

    def _define_D(self):
        """
        Define the initial belief distribution D.
        """
        # Define D based on the initial state distribution
        D = np.ones((len(self.spatial_navigation_kd.state_names), 1)) / len(self.spatial_navigation_kd.state_names)
        return D

    def step(self):
        """
        Perform a step in the agent's decision-making process.
        """
        # Implement the step logic
        action = self.agent.step()
        logging.info(f"SpatialNavigationAgent action: {action}")
        return action