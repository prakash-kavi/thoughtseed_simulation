from higher_order_agents.metacognition import MetaCognitionAgent
from knowledge_domains.episodic_memory_KD import EpisodicMemoryKD
from knowledge_domains.emotion_KD import EmotionKD
from knowledge_domains.spatial_navigation_KD import SpatialNavigationKD
from thoughtseed_sub_agents.recalling_past_events_TS import RecallingPastEventsAgent
from thoughtseed_sub_agents.spatial_navigation_TS import SpatialNavigationAgent
from utils.visualization import plot_agent_behaviors, plot_simulation_outcomes
from utils.analysis import analyze_performance, statistical_analysis
import config
import logging

def main():
    # Set up logging
    if config.ENABLE_LOGGING:
        logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))

    # Initialize knowledge domains
    episodic_kd = EpisodicMemoryKD()
    emotion_kd = EmotionKD()
    spatial_kd = SpatialNavigationKD()

    # Initialize thoughtseed agents
    recalling_agent = RecallingPastEventsAgent(episodic_kd, emotion_kd, spatial_kd)
    navigation_agent = SpatialNavigationAgent(spatial_kd, emotion_kd)

    # Initialize meta-cognition agent
    meta_agent = MetaCognitionAgent(episodic_kd, emotion_kd, spatial_kd)

    # Simulation loop
    for step in range(config.SIMULATION_STEPS):
        # Agents interact with knowledge domains
        recalling_agent.step()
        navigation_agent.step()
        meta_agent.step()

        # (Additional simulation logic can be added here)

    # Visualize results
    plot_agent_behaviors()
    plot_simulation_outcomes()

if __name__ == "__main__":
    main()