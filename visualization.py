import json
import numpy as np
import matplotlib.pyplot as plt


def load_data(filepath: str) -> dict:
    """Load training results from a JSON file and compute max_score."""
    with open(filepath, "r") as f:
        training_state = json.load(f)
    max_reached = 0
    training_state['max_scores'] = []
    for i in training_state['scores']:
        max_reached = max(i, max_reached)
        training_state['max_scores'].append(max_reached)
    return training_state


def plot_performance(agent_name: str, agent_states: dict, window=50, logy=False) -> None:
    """Plot the training performance for a single agent."""
    episodes, scores, max_scores = agent_states['episodes'], agent_states['scores'], agent_states['max_scores']
    
    # Apply rolling mean and trim to match length of episodes
    smoothed_scores = np.convolve(scores, np.ones((window,)) / window, mode='same')
    smoothed_scores = smoothed_scores[:len(episodes)]  # Ensure both arrays have the same length
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title(f'Performance: {agent_name}', fontsize=16)
    plt.ylabel('Score', fontsize=14)
    plt.xlabel('Episode', fontsize=14)
    
    if logy:
        ax.set_yscale('log')
        plt.ylabel('log(Score)', fontsize=14)
        scores = [x + 1 for x in scores]
        max_scores = [x + 1 for x in max_scores]
        smoothed_scores = [x + 1 for x in smoothed_scores]

    plt.scatter(episodes, scores, label='Scores', color='b', s=3)
    plt.plot(episodes, max_scores, label='Max Scores', color='g')
    plt.plot(episodes, smoothed_scores, label='Rolling Mean', color='orange')
    
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Define filenames
    filenames = {
        'Validation Resume': 'data/validation_resume.json',
        'Training Values Epsilon': 'data/training_values_epsilon.json',
        'Training Values': 'data/training_values.json',
    }
    
    # Load and plot each file separately
    for agent_name, filepath in filenames.items():
        try:
            agent_states = load_data(filepath)
            plot_performance(agent_name, agent_states, window=50, logy=False)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading or plotting {filepath}: {e}")
