def analyze_performance(data):
    """
    Analyzes the performance of agents based on simulation data.
    
    Parameters:
    data (list): A list of performance metrics collected during the simulation.
    
    Returns:
    dict: A dictionary containing average performance metrics.
    """
    if not data:
        return {}

    average_performance = {
        'average_score': sum(d['score'] for d in data) / len(data),
        'average_time': sum(d['time'] for d in data) / len(data),
    }
    
    return average_performance


def statistical_analysis(data):
    """
    Performs statistical analysis on the simulation data.
    
    Parameters:
    data (list): A list of numerical data points from the simulation.
    
    Returns:
    dict: A dictionary containing statistical metrics such as mean and standard deviation.
    """
    import numpy as np

    if not data:
        return {}

    mean = np.mean(data)
    std_dev = np.std(data)

    return {
        'mean': mean,
        'std_dev': std_dev,
    }