import logging
import sys
import os
import numpy as np
from pymdp.agent import Agent

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

logging.basicConfig(level=logging.INFO)

class RecallingPastEventsAgent:
    """
    Agent responsible for recalling past events and activating specific memories.
    """

    def __init__(self, episodic_memory_kd, emotion_kd, spatial_navigation_kd):
        """
        Initializes the agent with references to the knowledge domains.
        """
        self.episodic_memory_kd = episodic_memory_kd
        self.emotion_kd = emotion_kd
        self.spatial_navigation_kd = spatial_navigation_kd

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
        A_memory = self.episodic_memory_kd.A
        A_emotion = np.eye(len(self.emotion_kd.emotion_names))
        A_combined = np.block([
            [A_memory],
            [A_emotion]
        ])
        A_combined = A_combined / A_combined.sum(axis=0, keepdims=True)  # Normalize A
        return A_combined

    def _define_B(self):
        """
        Define the state transition matrix B.
        """
        B_memory = self.episodic_memory_kd.B
        B_emotion = np.eye(len(self.emotion_kd.emotion_names))

        # Print shapes for debugging
        print("B_memory shape:", B_memory.shape)
        print("B_emotion shape:", B_emotion.shape)

        # Normalize each B matrix individually along the correct axis
        B_memory = B_memory / B_memory.sum(axis=1, keepdims=True)
        B_emotion = B_emotion / B_emotion.sum(axis=1, keepdims=True)

        # Ensure B_emotion has three dimensions
        B_emotion_expanded = np.expand_dims(B_emotion, axis=0)
        B_emotion_expanded = np.repeat(B_emotion_expanded, B_memory.shape[0], axis=0)

        # Ensure shapes are compatible
        print("Normalized B_memory:", B_memory)
        print("Expanded B_emotion:", B_emotion_expanded)

        # Combine as a 3D array
        B_combined = np.zeros((2, B_memory.shape[0], B_memory.shape[1], B_memory.shape[1]))

        B_combined[0] = B_memory
        B_combined[1] = B_emotion_expanded

        return B_combined


    def _define_C(self):
        """
        Define the preference matrix C.
        """
        C_emotion = np.ones(len(self.emotion_kd.emotion_names))
        C_emotion = C_emotion / C_emotion.sum(axis=0, keepdims=True)  # Normalize C
        return C_emotion

    def _define_D(self):
        """
        Define the initial state distribution D.
        """
        D_memory = self.episodic_memory_kd.D
        D_memory = D_memory / D_memory.sum(axis=0, keepdims=True)  # Normalize D
        return D_memory

    def activate_memory(self, cue):
        """
        Retrieves and activates specific memories based on the given cue.
        """
        memory_state = self.episodic_memory_kd.step(action=cue)
        logging.info("Activated Memory State: %s", self.episodic_memory_kd.get_state_details())
        observation = self.episodic_memory_kd.get_observation()
        logging.info("Observation: %s", observation)
        return memory_state

    def update_emotional_state(self):
        """
        Updates the emotional state based on the valence and confidence of the recalled memory.
        """
        observation_index = self.episodic_memory_kd.get_observation_index()
        if observation_index is None:
            logging.error("No observation index found. Ensure that activate_memory is called before update_emotional_state.")
            return
        
        valence = self.episodic_memory_kd.observation_names[observation_index]["valence"]
        confidence = self.episodic_memory_kd.observation_names[observation_index]["confidence"]

        if valence == "positive":
            self.emotion_kd.step(action="positive")
        elif valence == "negative":
            self.emotion_kd.step(action="negative")
        else:
            self.emotion_kd.step(action="neutral")

        logging.info("Updated Emotional State: %s", self.emotion_kd.get_emotion_name())

    def provide_navigation_cues(self):
        """
        Provides navigation cues based on the content of the recalled memory.
        """
        current_memory = self.episodic_memory_kd.get_state_details()
        content = current_memory.split("Content: ")[1].split(",")[0]

        if "kitchen" in content:
            cue = "search_kitchen"
        elif "living room" in content:
            cue = "search_living_room"
        elif "bedroom" in content:
            cue = "search_bedroom"
        else:
            cue = "search_default"

        logging.info("Navigation Cue: %s", cue)
        return cue

    def step(self, cue):
        """
        Performs a single step of the agent's process.
        """
        self.activate_memory(cue)
        self.update_emotional_state()
        navigation_cue = self.provide_navigation_cues()
        
        observations = [self.episodic_memory_kd.get_observation_index(), self.emotion_kd.get_emotion_index()]
        qs = self.agent.infer_states(observations)
        q_pi, G = self.agent.infer_policies()
        action = self.agent.sample_action()
        
        logging.info("Selected Action: %s", action)
        return navigation_cue

# Example usage
if __name__ == "__main__":
    from thoughtseed_simulation.knowledge_domains.episodic_memory_KD import EpisodicMemoryKD
    from thoughtseed_simulation.knowledge_domains.emotion_KD import EmotionKD
    from thoughtseed_simulation.knowledge_domains.spatial_navigation_KD import SpatialNavigationKD

    episodic_memory_kd = EpisodicMemoryKD(seed=42)
    emotion_kd = EmotionKD(seed=42)
    spatial_navigation_kd = SpatialNavigationKD(seed=42)

    agent = RecallingPastEventsAgent(episodic_memory_kd, emotion_kd, spatial_navigation_kd)

    for _ in range(10):
        navigation_cue = agent.step(cue="recall")
        print("Navigation Cue:", navigation_cue)
