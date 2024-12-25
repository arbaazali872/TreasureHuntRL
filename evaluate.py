import json

def evaluate_model(model, env, num_episodes=100, log_file="evaluation_results.json"):
    """
    Evaluate a trained model over a fixed number of episodes and log the metrics.

    Args:
        model: Trained RL model to evaluate.
        env: The environment in which to evaluate the model.
        num_episodes (int): Number of episodes to evaluate.
        log_file (str): Path to save evaluation results.

    Returns:
        metrics (dict): Dictionary containing evaluation metrics.
    """
    total_reward = 0
    success_count = 0
    total_steps = 0

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1

        total_reward += episode_reward
        total_steps += steps
        if reward > 0:  # Success if the agent reached the treasure
            success_count += 1

    # Calculate metrics
    avg_reward = total_reward / num_episodes
    success_rate = success_count / num_episodes
    avg_steps = total_steps / num_episodes

    # Save metrics to a JSON file
    metrics = {
        "average_reward": avg_reward,
        "success_rate": success_rate,
        "average_steps": avg_steps
    }
    with open(log_file, "w") as f:
        json.dump(metrics, f)

    print(f"Evaluation results saved to {log_file}")
    return metrics
