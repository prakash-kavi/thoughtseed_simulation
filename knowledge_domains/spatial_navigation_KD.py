import numpy as np
import logging
import matplotlib.pyplot as plt
import networkx as nx

logging.basicConfig(level=logging.INFO)

class SpatialNavigationKD:
    """
    Simulates a spatial navigation process using a probabilistic state-transition model.
    """

    def __init__(self, room_names=None, B=None, A=None, D=None, seed=None):
        """
        Initializes the model with room names, transition matrix, observation matrix, and initial state distribution.
        Allows optional custom inputs for transition and observation matrices.
        """
        if seed is not None:
            np.random.seed(seed)

        self.room_names = room_names if room_names else [
            {"name": "Kitchen", "connections": ["Living Room", "Bathroom"], "landmarks": ["Fridge", "Oven"]},
            {"name": "Living Room", "connections": ["Kitchen", "Bedroom", "Office"], "landmarks": ["Sofa", "TV"]},
            {"name": "Bedroom", "connections": ["Living Room"], "landmarks": ["Bed", "Wardrobe"]},
            {"name": "Bathroom", "connections": ["Kitchen"], "landmarks": ["Shower", "Sink"]},
            {"name": "Office", "connections": ["Living Room"], "landmarks": ["Desk", "Computer"]}
        ]
        self.num_rooms = len(self.room_names)

        self.B = B if B is not None else np.array([
            [0.5, 0.5, 0.0, 0.5, 0.0],  # Kitchen
            [0.5, 0.5, 0.5, 0.0, 0.5],  # Living Room
            [0.0, 0.5, 0.5, 0.0, 0.0],  # Bedroom
            [0.5, 0.0, 0.0, 0.5, 0.0],  # Bathroom
            [0.0, 0.5, 0.0, 0.0, 0.5]   # Office
        ])
        self.B = self.B / self.B.sum(axis=1, keepdims=True)  # Normalize transition matrix

        self.A = A if A is not None else np.array([
            [0.6, 0.3, 0.1, 0.0, 0.0],  # Keys
            [0.1, 0.4, 0.4, 0.0, 0.1],  # Remote
            [0.0, 0.1, 0.4, 0.0, 0.5]   # Book
        ])
        self.A = self.A / self.A.sum(axis=1, keepdims=True)  # Normalize likelihood matrix

        self.D = D if D is not None else np.ones(self.num_rooms) / self.num_rooms  # Uniform initial belief

        self.state = self.D

        self.validate_matrices()

    def validate_matrices(self):
        """
        Validates the dimensions and properties of the transition and observation matrices.
        """
        assert self.B.shape == (self.num_rooms, self.num_rooms), "Transition matrix B has incorrect dimensions."
        assert self.A.shape == (self.A.shape[0], self.num_rooms), "Observation matrix A has incorrect dimensions."
        assert np.allclose(self.B.sum(axis=1), 1), "Rows of transition matrix B must sum to 1."
        assert np.allclose(self.A.sum(axis=1), 1), "Rows of observation matrix A must sum to 1."

    def step(self):
        """
        Advances the system to the next room based on the current room and transition probabilities.
        """
        next_room_prob = np.dot(self.B.T, self.state).flatten()
        next_room_prob = next_room_prob / next_room_prob.sum()
        next_room_index = np.random.choice(self.num_rooms, p=next_room_prob)
        self.state = np.zeros(self.num_rooms)
        self.state[next_room_index] = 1
        logging.info("Next Room Probabilities: %s", next_room_prob)
        logging.info("New Room: %s", self.get_room_name())
        return self.state

    def get_observation(self):
        """
        Samples an observation based on the current room and observation likelihoods.
        """
        observation_prob = np.dot(self.A, self.state).flatten()
        observation_prob = observation_prob / observation_prob.sum()
        observation_index = np.random.choice(self.A.shape[0], p=observation_prob)
        self.update_beliefs(observation_index)
        logging.info("Observation Probabilities: %s", observation_prob)
        observation_labels = ["Keys", "Remote", "Book"]
        return observation_labels[observation_index]

    def update_beliefs(self, observation_index):
        """
        Updates beliefs about the current room using Bayesian inference.
        """
        likelihood = self.A[observation_index]
        updated_state = self.state * likelihood
        updated_state = updated_state / updated_state.sum()
        self.state = updated_state

    def get_room_name(self):
        """
        Returns the name of the current room.
        """
        return self.room_names[np.argmax(self.state)]["name"]

    def plot_matrices(self):
        """
        Visualizes the transition and observation matrices with room names.
        """
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title("Transition Matrix (B)")
        plt.imshow(self.B, cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.xticks(ticks=np.arange(self.num_rooms), labels=[room["name"] for room in self.room_names], rotation=90)
        plt.yticks(ticks=np.arange(self.num_rooms), labels=[room["name"] for room in self.room_names])

        plt.subplot(1, 2, 2)
        plt.title("Observation Matrix (A)")
        plt.imshow(self.A, cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.xticks(ticks=np.arange(self.num_rooms), labels=[room["name"] for room in self.room_names], rotation=90)
        plt.yticks(ticks=np.arange(self.A.shape[0]), labels=["Keys", "Remote", "Book"])

        plt.tight_layout()
        plt.show()

    def plot_room_connections(self):
        """
        Plots the physical room connections based on the transition matrix.
        """
        G = nx.Graph()
        for i, room in enumerate(self.room_names):
            G.add_node(room["name"])
        for i in range(self.num_rooms):
            for j in range(self.num_rooms):
                if self.B[i, j] > 0:
                    G.add_edge(self.room_names[i]["name"], self.room_names[j]["name"], weight=self.B[i, j])

        pos = nx.spring_layout(G)
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold")
        labels = nx.get_edge_attributes(G, 'weight')
        # Normalize edge weights to ensure they sum to 1 for each node
        for node in G.nodes:
            total_weight = sum(G[node][neighbor]['weight'] for neighbor in G[node])
            for neighbor in G[node]:
                G[node][neighbor]['weight'] /= total_weight
        labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.title("Physical Room Connections")
        plt.show()

# Testing the SpatialNavigationKD class
if __name__ == "__main__":
    spatial_navigation_kd = SpatialNavigationKD(seed=42)  # Set a random seed for repeatability
    print("Initial Room:", spatial_navigation_kd.get_room_name())
    print("Initial Observation:", spatial_navigation_kd.get_observation())

    for _ in range(10):
        next_room = spatial_navigation_kd.step()
        print("Next Room:", spatial_navigation_kd.get_room_name())
        print("Observation:", spatial_navigation_kd.get_observation())

    spatial_navigation_kd.plot_matrices()
    spatial_navigation_kd.plot_room_connections()