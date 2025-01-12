import pymdp
from pymdp import utils
import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

class EpisodicMemoryKD:
    """
    Simulates an episodic memory process using a probabilistic state-transition model.
    """

    def __init__(self, state_names=None, observation_names=None, B=None, A=None, D=None, C=None, seed=None):
        """
        Initializes the model with state names, observation names, and probabilistic matrices.
        Allows optional custom inputs for transition and observation matrices.
        """
        if seed is not None:
            np.random.seed(seed)

        # Define state names with additional attributes for content and time
        self.state_names = state_names if state_names else [
            {"name": "Highly Activated", "content": "left keys on the kitchen counter", "time": "last hour"},
            {"name": "Moderately Activated", "content": "left keys in the living room", "time": "yesterday"},
            {"name": "Baseline", "content": "left keys in the bedroom", "time": "last week"},
            {"name": "Moderately Suppressed", "content": "left keys in the bathroom", "time": "last month"},
            {"name": "Highly Suppressed", "content": "left keys in the office", "time": "last year"}
        ]
        self.num_states = len(self.state_names)

        # Define observation names with additional attributes for valence and confidence
        self.observation_names = observation_names if observation_names else [
            {"name": "Vivid", "valence": "positive", "confidence": "high"},
            {"name": "Moderate", "valence": "neutral", "confidence": "medium"},
            {"name": "Faint", "valence": "negative", "confidence": "low"}
        ]
        self.num_observations = len(self.observation_names)

        # Define transition matrix B
        self.B = B if B is not None else np.array([
            [0.5, 0.3, 0.1, 0.1, 0.0],  # Highly Activated
            [0.2, 0.4, 0.3, 0.1, 0.0],  # Moderately Activated
            [0.1, 0.2, 0.4, 0.2, 0.1],  # Baseline
            [0.0, 0.1, 0.3, 0.4, 0.2],  # Moderately Suppressed
            [0.0, 0.0, 0.1, 0.3, 0.6]   # Highly Suppressed
        ])
        self.B = self.B / self.B.sum(axis=1, keepdims=True)  # Normalize transition matrix

        # Define observation likelihood matrix A
        self.A = A if A is not None else np.array([
            [0.9, 0.7, 0.5, 0.3, 0.1],  # Vivid 
            [0.1, 0.2, 0.3, 0.4, 0.3],  # Moderate
            [0.0, 0.1, 0.2, 0.3, 0.6]   # Faint 
        ])
        self.A = self.A / self.A.sum(axis=1, keepdims=True)  # Normalize likelihood matrix

        # Define initial state distribution D
        self.D = D if D is not None else utils.onehot(2, self.num_states)

        # Define preference matrix C
        self.C = C if C is not None else np.ones(self.num_observations)  # Default preference is uniform

        # Initialize the state with the initial state distribution
        self.state = self.D

        # Initialize the last observation index
        self.last_observation_index = None

        # Validate the dimensions and properties of the matrices
        self.validate_matrices()

    def validate_matrices(self):
        """
        Validates the dimensions and properties of the transition and observation matrices.
        """
        assert self.B.shape == (self.num_states, self.num_states), "Transition matrix B has incorrect dimensions."
        assert self.A.shape == (self.num_observations, self.num_states), "Observation matrix A has incorrect dimensions."
        assert np.allclose(self.B.sum(axis=1), 1), "Rows of transition matrix B must sum to 1."
        assert np.allclose(self.A.sum(axis=1), 1), "Rows of observation matrix A must sum to 1."

    def step(self, action=None):
        """
        Advances the system to the next state based on the current state, action, and transition probabilities.
        """
        if action is not None:
            # Modify transition probabilities based on the action
            action_effect = np.eye(self.num_states) * 0.1  # Example: Action influences self-transitions
            modified_B = self.B + action_effect
            modified_B = modified_B / modified_B.sum(axis=1, keepdims=True)
            next_state_prob = np.dot(modified_B.T, self.state).flatten()
        else:
            next_state_prob = np.dot(self.B.T, self.state).flatten()

        next_state_prob = next_state_prob / next_state_prob.sum()
        next_state_index = utils.sample(next_state_prob)
        self.state = utils.onehot(next_state_index, self.num_states)
        logging.info("Next State Probabilities: %s", next_state_prob)
        logging.info("New State: %s", self.get_state_details())
        return self.state

    def get_observation(self):
        """
        Samples an observation based on the current state and observation likelihoods.
        """
        observation_prob = np.dot(self.A, self.state).flatten()
        observation_prob = observation_prob / observation_prob.sum()
        observation_index = utils.sample(observation_prob)
        self.last_observation_index = observation_index  # Store the observation index
        self.update_beliefs(observation_index)
        logging.info("Observation Probabilities: %s", observation_prob)
        logging.info("Observation: %s", self.get_observation_details(observation_index))
        return self.observation_names[observation_index]["name"]

    def get_observation_index(self):
        """
        Returns the index of the most recent observation.
        """
        return self.last_observation_index

    def update_beliefs(self, observation_index):
        """
        Updates beliefs about the current state using Bayesian inference.
        """
        likelihood = self.A[observation_index]
        updated_state = self.state * likelihood
        updated_state = updated_state / updated_state.sum()
        self.state = updated_state

    def get_state_details(self):
        """
        Returns the details of the current state.
        """
        state = self.state_names[np.argmax(self.state)]
        return f"{state['name']} (Content: {state['content']}, Time: {state['time']})"

    def get_observation_details(self, observation_index):
        """
        Returns the details of the observation.
        """
        observation = self.observation_names[observation_index]
        return f"{observation['name']} (Valence: {observation['valence']}, Confidence: {observation['confidence']})"

    def plot_matrices(self):
        """
        Visualizes the transition and observation matrices with state and observation names.
        """
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title("Transition Matrix (B)")
        plt.imshow(self.B, cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.xticks(ticks=np.arange(self.num_states), labels=[state["name"] for state in self.state_names], rotation=90)
        plt.yticks(ticks=np.arange(self.num_states), labels=[state["name"] for state in self.state_names])

        plt.subplot(1, 2, 2)
        plt.title("Observation Matrix (A)")
        plt.imshow(self.A, cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.xticks(ticks=np.arange(self.num_states), labels=[state["name"] for state in self.state_names], rotation=90)
        plt.yticks(ticks=np.arange(self.num_observations), labels=[obs["name"] for obs in self.observation_names])

        plt.tight_layout()
        plt.show()

    def get_action(self):
        """
        Determines an action based on the current state and a predefined policy.
        The action is influenced by the content and context of the current memory state.
        """
        current_state_index = np.argmax(self.state)
        current_state = self.state_names[current_state_index]

        # Define actions based on the content and context of the memory state
        if "kitchen" in current_state["content"]:
            action_prob = [0.7, 0.2, 0.1]  # Higher probability for action 0 (e.g., search kitchen)
        elif "living room" in current_state["content"]:
            action_prob = [0.2, 0.7, 0.1]  # Higher probability for action 1 (e.g., search living room)
        elif "bedroom" in current_state["content"]:
            action_prob = [0.1, 0.2, 0.7]  # Higher probability for action 2 (e.g., search bedroom)
        else:
            action_prob = [0.33, 0.33, 0.34]  # Default probabilities if no specific content is matched

        action = np.random.choice(len(action_prob), p=action_prob)
        logging.info("Current State: %s", current_state["name"])
        logging.info("Action Probabilities: %s", action_prob)
        logging.info("Chosen Action: %d", action)
        return action

# Testing the EpisodicMemoryKD class
if __name__ == "__main__":
    episodic_memory_kd = EpisodicMemoryKD(seed=42)  # Set a random seed for repeatability
    print("Initial State:", episodic_memory_kd.get_state_details())
    print("Initial Observation:", episodic_memory_kd.get_observation())

    for _ in range(10):
        next_state = episodic_memory_kd.step()
        print("Next State:", episodic_memory_kd.get_state_details())
        print("Observation:", episodic_memory_kd.get_observation())

    episodic_memory_kd.plot_matrices()