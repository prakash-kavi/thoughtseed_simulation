import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

class EmotionKD:
    """
    Simulates the emotional state of an agent during a key search process using a probabilistic state-transition model.
    """

    def __init__(self, emotion_names=None, B=None, A=None, D=None, seed=None):
        """
        Initializes the model with emotion names, transition matrix, observation matrix, and initial state distribution.
        Allows optional custom inputs for transition and observation matrices.
        """
        if seed is not None:
            np.random.seed(seed)

        self.emotion_names = emotion_names if emotion_names else [
            {"name": "Anxious", "intensity": "Moderate", "physiological_cues": ["Increased heart rate"]},
            {"name": "Frustrated", "intensity": "High", "physiological_cues": ["Tense muscles"]},
            {"name": "Neutral", "intensity": "Low", "physiological_cues": ["Calm"]},
            {"name": "Hopeful", "intensity": "Moderate", "physiological_cues": ["Relaxed"]},
            {"name": "Relieved", "intensity": "High", "physiological_cues": ["Deep breaths"]}
        ]
        self.num_emotions = len(self.emotion_names)

        self.B = B if B is not None else np.array([
            [0.5, 0.3, 0.1, 0.1, 0.0],  # Anxious
            [0.3, 0.4, 0.2, 0.1, 0.0],  # Frustrated
            [0.1, 0.2, 0.4, 0.2, 0.1],  # Neutral
            [0.0, 0.1, 0.2, 0.4, 0.3],  # Hopeful
            [0.0, 0.0, 0.1, 0.3, 0.6]   # Relieved
        ])
        self.B = self.B / self.B.sum(axis=1, keepdims=True)  # Normalize transition matrix

        self.A = A if A is not None else np.array([
            [0.1, 0.1, 0.3, 0.4, 0.5],  # Positive Cue
            [0.6, 0.4, 0.1, 0.0, 0.0],  # Negative Cue
            [0.3, 0.5, 0.6, 0.6, 0.5]   # Neutral Cue
        ])
        self.A = self.A / self.A.sum(axis=1, keepdims=True)  # Normalize likelihood matrix

        self.D = D if D is not None else np.ones(self.num_emotions) / self.num_emotions  # Uniform initial belief

        self.state = self.D
        self.steps = 0  # Initialize step counter

        self.validate_matrices()

    def validate_matrices(self):
        """
        Validates the dimensions and properties of the transition and observation matrices.
        """
        assert self.B.shape == (self.num_emotions, self.num_emotions), "Transition matrix B has incorrect dimensions."
        assert self.A.shape == (self.A.shape[0], self.num_emotions), "Observation matrix A has incorrect dimensions."
        assert np.allclose(self.B.sum(axis=1), 1), "Rows of transition matrix B must sum to 1."
        assert np.allclose(self.A.sum(axis=1), 1), "Rows of observation matrix A must sum to 1."

    def step(self):
        """
        Advances the system to the next emotional state based on the current state and transition probabilities.
        """
        next_emotion_prob = np.dot(self.B.T, self.state).flatten()
        next_emotion_prob = next_emotion_prob / next_emotion_prob.sum()
        next_emotion_index = np.random.choice(self.num_emotions, p=next_emotion_prob)
        self.state = np.zeros(self.num_emotions)
        self.state[next_emotion_index] = 1
        self.steps += 1  # Increment step counter
        logging.info("Next Emotion Probabilities: %s", next_emotion_prob)
        logging.info("New Emotion: %s", self.get_emotion_name())
        return self.state

    def get_observation(self):
        """
        Samples an observation based on the current emotional state and observation likelihoods.
        """
        observation_prob = np.dot(self.A, self.state).flatten()
        if observation_prob.sum() == 0:
            observation_prob = np.ones_like(observation_prob) / len(observation_prob)  # Uniform distribution if sum is zero
        else:
            observation_prob = observation_prob / observation_prob.sum()
        observation_index = np.random.choice(self.A.shape[0], p=observation_prob)
        self.update_beliefs(observation_index)
        logging.info("Observation Probabilities: %s", observation_prob)
        observation_labels = ["Positive Cue", "Negative Cue", "Neutral Cue"]
        return observation_labels[observation_index]

    def update_beliefs(self, observation_index):
        """
        Updates beliefs about the current emotional state using Bayesian inference.
        """
        likelihood = self.A[observation_index]
        updated_state = self.state * likelihood
        if updated_state.sum() == 0:
            updated_state = np.ones_like(updated_state) / len(updated_state)  # Uniform distribution if sum is zero
        else:
            updated_state = updated_state / updated_state.sum()
        self.state = updated_state

    def get_emotion_name(self):
        """
        Returns the name of the current emotional state.
        """
        return self.emotion_names[np.argmax(self.state)]["name"]

    def plot_matrices(self):
        """
        Visualizes the transition and observation matrices with emotion names.
        """
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title("Transition Matrix (B)")
        plt.imshow(self.B, cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.xticks(ticks=np.arange(self.num_emotions), labels=[emotion["name"] for emotion in self.emotion_names], rotation=90)
        plt.yticks(ticks=np.arange(self.num_emotions), labels=[emotion["name"] for emotion in self.emotion_names])

        plt.subplot(1, 2, 2)
        plt.title("Observation Matrix (A)")
        plt.imshow(self.A, cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.xticks(ticks=np.arange(self.num_emotions), labels=[emotion["name"] for emotion in self.emotion_names], rotation=90)
        plt.yticks(ticks=np.arange(self.A.shape[0]), labels=["Positive Cue", "Negative Cue", "Neutral Cue"])

        plt.tight_layout()
        plt.show()

# Testing the EmotionKD class
if __name__ == "__main__":
    emotion_kd = EmotionKD(seed=42)  # Set a random seed for repeatability
    print("Initial Emotion:", emotion_kd.get_emotion_name())
    print("Initial Observation:", emotion_kd.get_observation())

    for _ in range(10):
        next_emotion = emotion_kd.step()
        print("Next Emotion:", emotion_kd.get_emotion_name())
        print("Observation:", emotion_kd.get_observation())

    emotion_kd.plot_matrices()
    print("Total Steps Taken:", emotion_kd.steps)