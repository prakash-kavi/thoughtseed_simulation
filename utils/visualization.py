def plot_agent_behaviors(agent_data):
    import matplotlib.pyplot as plt

    for agent_name, behaviors in agent_data.items():
        plt.plot(behaviors['time'], behaviors['values'], label=agent_name)

    plt.xlabel('Time')
    plt.ylabel('Behavior Values')
    plt.title('Agent Behaviors Over Time')
    plt.legend()
    plt.show()


def plot_simulation_outcomes(outcome_data):
    import matplotlib.pyplot as plt

    plt.bar(outcome_data['labels'], outcome_data['values'])
    plt.xlabel('Outcomes')
    plt.ylabel('Values')
    plt.title('Simulation Outcomes')
    plt.show()