# Project Overview

This project implements a 3-level hierarchical POMDP model. The goal is to simulate the cognitive processes involved in a relatable scenario: searching for lost keys.

## Model Structure

The model consists of three interconnected levels:

### Meta-awareness Agent
- **Role**: Oversees the thoughtseed agents, ensuring they work towards the goal of finding the lost keys.
- **Functionality**: Adjusts the priorities and actions of the thoughtseed agents based on the current state and observations.

### Thoughtseed Agents

#### Recalling Past Events Agent
- **Function**: Activates specific memories related to past events that might help locate the keys.
- **Interactions**:
  - **Episodic Memory KD**: Accesses and retrieves relevant memories.
  - **Emotion KD**: Influences and is influenced by the current emotional state.

#### Spatial Navigation Agent
- **Function**: Provides navigational cues and plans routes to search different locations.
- **Interactions**:
  - **Spatial Navigation KD**: Uses spatial information to navigate.
  - **Emotion KD**: Adjusts navigation strategies based on emotional states.

### Knowledge Domains (KDs)

#### Episodic Memory KD
- **Role**: Stores and retrieves episodic memories.
- **State Space**: Represents different memory states.
- **Transition Dynamics**: Defines how memories transition from one state to another.

#### Spatial Navigation KD
- **Role**: Represents spatial information and navigation strategies.
- **State Space**: Represents different locations (e.g., kitchen, living room) or grid cells.
- **Transition Dynamics**: Defines how the agent moves from one location to another.

#### Emotion KD
- **Role**: Represents the agent's emotional states.
- **State Space**: Represents different emotional states (e.g., anxious, frustrated, neutral, hopeful, relieved).
- **Transition Dynamics**: Defines how emotional states transition based on observations and interactions with other KDs.

## Detailed Descriptions

#### Recalling Past Events Agent
- **Role**: The Recalling Past Events Agent is responsible for activating specific memories related to past events that might help locate the keys.
- **Functionality**:
  - **Memory Activation**: Activates and retrieves relevant episodic memories that could provide clues about the location of the keys.
  - **Memory Evaluation**: Evaluates the relevance and confidence of the retrieved memories to prioritize the search process.
  - **Interaction with KDs**:
    - **Episodic Memory KD**: Accesses and retrieves memories stored in the Episodic Memory KD. Uses the state space and transition dynamics of the Episodic Memory KD to simulate the recall process.
    - **Emotion KD**: Influences and is influenced by the current emotional state. Emotional states can affect the recall process, and the recalled memories can, in turn, influence the emotional state.
- **Implementation**:
  - **A Matrix**: Defines the observation likelihoods by combining episodic memory and emotional states. Represents the probability of observing a particular cue given the current state.
  - **B Matrix**: Defines the state transition probabilities based on past events and emotional states. Represents how the state transitions from one to another.
  - **C Matrix**: Represents the agent's preferences or goals, influencing the decision-making process.
  - **D Matrix**: Represents the initial belief distribution over states.

#### Spatial Navigation Agent
- **Role**: The Spatial Navigation Agent is responsible for navigating space and providing navigational cues to search different locations.
- **Functionality**:
  - **Route Planning**: Plans routes and navigates through different locations to search for the keys.
  - **Spatial Awareness**: Maintains awareness of the spatial environment and adjusts navigation strategies based on observations.
  - **Interaction with KDs**:
    - **Spatial Navigation KD**: Uses spatial information to navigate and plan routes. Interacts with the state space and transition dynamics of the Spatial Navigation KD to simulate movement.
    - **Emotion KD**: Adjusts navigation strategies based on the current emotional state. Emotional states can influence decision-making and movement.
- **Implementation**:
  - **A Matrix**: Defines the observation likelihoods by combining spatial and emotional states. Represents the probability of observing a particular cue given the current state.
  - **B Matrix**: Defines the state transition probabilities based on spatial movements and emotional states. Represents how the state transitions from one location to another.
  - **C Matrix**: Represents the agent's preferences or goals, influencing the decision-making process.
  - **D Matrix**: Represents the initial belief distribution over states.


### Episodic Memory Knowledge Domain (KD)

#### Observables

Observables in the Episodic Memory KD represent the different types of observations that the agent can make about its memory states. These observations are characterized by their valence (emotional value) and confidence (certainty).

- **Vivid**:
  - **Valence**: Positive
  - **Confidence**: High
  - **Description**: Represents a clear and detailed memory that is highly confident and positively valued. For example, vividly remembering leaving the keys on the kitchen counter an hour ago.
  
- **Moderate**:
  - **Valence**: Neutral
  - **Confidence**: Medium
  - **Description**: Represents a moderately clear memory with average confidence and neutral emotional value. For example, moderately remembering leaving the keys in the living room yesterday.
  
- **Faint**:
  - **Valence**: Negative
  - **Confidence**: Low
  - **Description**: Represents a vague and unclear memory with low confidence and negative emotional value. For example, faintly remembering leaving the keys in the office last year.

#### States

States in the Episodic Memory KD represent different levels of memory activation, each associated with specific content and time attributes.

- **Highly Activated**:
  - **Content**: Left keys on the kitchen counter
  - **Time**: Last hour
  - **Description**: Represents a highly activated memory state where the agent strongly recalls leaving the keys on the kitchen counter recently.
  
- **Moderately Activated**:
  - **Content**: Left keys in the living room
  - **Time**: Yesterday
  - **Description**: Represents a moderately activated memory state where the agent recalls leaving the keys in the living room a day ago.
  
- **Baseline**:
  - **Content**: Left keys in the bedroom
  - **Time**: Last week
  - **Description**: Represents a baseline memory state where the agent has a neutral recall of leaving the keys in the bedroom a week ago.
  
- **Moderately Suppressed**:
  - **Content**: Left keys in the bathroom
  - **Time**: Last month
  - **Description**: Represents a moderately suppressed memory state where the agent has a weak recall of leaving the keys in the bathroom a month ago.
  
- **Highly Suppressed**:
  - **Content**: Left keys in the office
  - **Time**: Last year
  - **Description**: Represents a highly suppressed memory state where the agent barely recalls leaving the keys in the office a year ago.

### Spatial Navigation Knowledge Domain (KD)

#### Observables

Observables in the Spatial Navigation KD represent the different types of observations that the agent can make about its spatial environment. These observations are characterized by their location and context.

- **Kitchen**:
  - **Description**: Represents the observation of the kitchen area. For example, seeing the kitchen counter or appliances.
  
- **Living Room**:
  - **Description**: Represents the observation of the living room area. For example, seeing the sofa or television.
  
- **Bedroom**:
  - **Description**: Represents the observation of the bedroom area. For example, seeing the bed or wardrobe.
  
- **Bathroom**:
  - **Description**: Represents the observation of the bathroom area. For example, seeing the sink or shower.
  
- **Office**:
  - **Description**: Represents the observation of the office area. For example, seeing the desk or computer.

#### States

States in the Spatial Navigation KD represent different locations within the environment, each associated with specific spatial attributes.

- **Kitchen**:
  - **Description**: Represents the state of being in the kitchen area.
  
- **Living Room**:
  - **Description**: Represents the state of being in the living room area.
  
- **Bedroom**:
  - **Description**: Represents the state of being in the bedroom area.
  
- **Bathroom**:
  - **Description**: Represents the state of being in the bathroom area.
  
- **Office**:
  - **Description**: Represents the state of being in the office area.

### Emotion Knowledge Domain (KD)

#### Observables

Observables in the Emotion KD represent the different types of observations that the agent can make about its emotional states. These observations are characterized by their intensity and physiological cues.

- **Positive Cue**:
  - **Intensity**: High
  - **Physiological Cues**: Relaxed, deep breaths
  - **Description**: Represents a positive emotional state with high intensity and physiological relaxation.
  
- **Negative Cue**:
  - **Intensity**: High
  - **Physiological Cues**: Increased heart rate, tense muscles
  - **Description**: Represents a negative emotional state with high intensity and physiological tension.
  
- **Neutral Cue**:
  - **Intensity**: Low
  - **Physiological Cues**: Calm
  - **Description**: Represents a neutral emotional state with low intensity and physiological calmness.

#### States

States in the Emotion KD represent different emotional states, each associated with specific intensity and physiological cues.

- **Anxious**:
  - **Intensity**: Moderate
  - **Physiological Cues**: Increased heart rate
  - **Description**: Represents a moderate level of anxiety with increased heart rate.
  
- **Frustrated**:
  - **Intensity**: High
  - **Physiological Cues**: Tense muscles
  - **Description**: Represents a high level of frustration with tense muscles.
  
- **Neutral**:
  - **Intensity**: Low
  - **Physiological Cues**: Calm
  - **Description**: Represents a low level of emotional intensity with calm physiological cues.
  
- **Hopeful**:
  - **Intensity**: Moderate
  - **Physiological Cues**: Relaxed
  - **Description**: Represents a moderate level of hopefulness with relaxed physiological cues.
  
- **Relieved**:
  - **Intensity**: High
  - **Physiological Cues**: Deep breaths
  - **Description**: Represents a high level of relief with deep breaths.

### Summary

This document provides a comprehensive overview of the project, detailing the roles, functionalities, and interactions of the Meta-awareness Agent, Thoughtseed Agents, and Knowledge Domains. It includes detailed descriptions of the observables and states relevant to each KD, providing a clear understanding of the model's components and their roles in simulating the cognitive processes involved in searching for lost keys.