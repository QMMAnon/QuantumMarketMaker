import tensorflow as tf
import numpy as np

from mbt_gym.agents.DDPGMarketMakerAgent import DDPGMarketMakerAgent
from mbt_gym.agents.PQCDDPGMarketMakerAgent import PQCDDPGMarketMakerAgent
from mbt_gym.agents.VQCDDPGMarketMakerAgent import VQCDDPGMarketMakerAgent

def export_actor_weights_only(agent_class, weights_path, export_path, state_size, action_size, action_bounds, **kwargs):
    # Instantiate agent and load pre-trained weights
    agent = agent_class(state_size, action_size, action_bounds, **kwargs)
    agent.actor(tf.zeros((1, state_size)))  # Build model
    agent.actor.load_weights(weights_path)

    # Save actor weights
    actor_weights_path = f"{export_path}_actor.weights.h5"
    agent.actor.save_weights(actor_weights_path)
    print(f"Saved actor weights to: {actor_weights_path}")

    # Export 1000 sample policy outputs for analysis
    sample_states = np.random.uniform(low=-1, high=1, size=(1000, state_size)).astype(np.float32)
    actions = agent.actor(sample_states).numpy()
    np.savez(f"{export_path}_policy_outputs.npz", states=sample_states, actions=actions)
    print(f"Saved 1000 sample policy outputs to: {export_path}_policy_outputs.npz")

state_size = 4
action_size = 2

# Export weights for DDPG agent
export_actor_weights_only(
    DDPGMarketMakerAgent,
    "/Users/chloejassur/mbt_gym_copy/results/pqc29_best_actor.weights.h5",
    "ddpg_policy",
    state_size, action_size, action_bounds
)

# Export weights for PQC agent
export_actor_weights_only(
    PQCDDPGMarketMakerAgent,
    "pqc_model29.weights.h5_best_actor.weights.h5",
    "pqc_policy",
    state_size, action_size, action_bounds,
    n_qubits=4
)

# Export weights for VQC agent
export_actor_weights_only(
    VQCDDPGMarketMakerAgent,
    "vqc_model37.weights.h5_best_actor.weights.h5", 
    "vqc_policy",
    state_size, action_size, action_bounds,
    n_qubits=6,
    n_layers=1
)
