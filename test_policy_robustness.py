import os
import numpy as np
import argparse
import tensorflow as tf
import random
import pandas as pd

from mbt_gym.gym.TradingEnvironment import TradingEnvironment
from mbt_gym.gym.ModelDynamics import LimitOrderModelDynamics
from mbt_gym.stochastic_processes.arrival_models import PoissonArrivalModel
from mbt_gym.stochastic_processes.fill_probability_models import ExponentialFillFunction
from mbt_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from mbt_gym.agents.DDPGMarketMakerAgent import DDPGMarketMakerAgent
from mbt_gym.agents.PQCDDPGMarketMakerAgent import PQCDDPGMarketMakerAgent
from mbt_gym.agents.VQCDDPGMarketMakerAgent import VQCDDPGMarketMakerAgent
from mbt_gym.gym.index_names import INVENTORY_INDEX

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

AGENT_CLASSES = {
    "ddpg": DDPGMarketMakerAgent,
    "pqc": PQCDDPGMarketMakerAgent,
    "vqc": VQCDDPGMarketMakerAgent
}

FIXED_SEEDS = [7, 13, 21, 42, 69]
ARRIVAL_RATES = [10, 50, 100, 200, 500]


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_env(seed, arrival_rate, step_size=0.005, n_steps=200):
    arrival_model = PoissonArrivalModel(
        intensity=np.array([arrival_rate, arrival_rate]),
        step_size=step_size,
        seed=seed
    )
    midprice_model = BrownianMotionMidpriceModel(
        step_size=step_size,
        seed=seed
    )
    fill_model = ExponentialFillFunction(
        step_size=step_size,
        seed=seed
    )

    model_dynamics = LimitOrderModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        fill_probability_model=fill_model,
        seed=seed
    )

    return TradingEnvironment(
        model_dynamics=model_dynamics,
        seed=seed,
        n_steps=n_steps,
        normalise_action_space=True,
        normalise_observation_space=True,
    )


def run_episode(agent, env, use_noise):
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0

    while not done:
        action = agent.act(state, use_noise=use_noise)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        steps += 1

    inventory_size = next_state[0, 1]
    if env.normalise_observation_space_:
        inventory_size = (
            (inventory_size + 1)
            * env._gradient_obs_norm[INVENTORY_INDEX]
            + env._intercept_obs_norm[INVENTORY_INDEX]
        )

    return float(total_reward), float(inventory_size), steps


def evaluate(agent_type, model_path_actor, episodes_per_seed):
    AgentClass = AGENT_CLASSES[agent_type]
    results = []

    for rate in ARRIVAL_RATES:
        for seed in FIXED_SEEDS:
            set_seed(seed)
            env = build_env(seed=seed, arrival_rate=rate)
            action_size = int(np.prod(env.action_space.shape))
            state_size = env.observation_space.shape[0]
            action_bounds = env.action_space.high[0]

            if agent_type == "vqc":
                agent = AgentClass(state_size, action_size, action_bounds)
            elif agent_type=="pqc":
                agent = AgentClass(state_size, action_size, action_bounds, n_qubits=4)
            else:
                agent = AgentClass(state_size, action_size, action_bounds)

            agent.actor(tf.zeros((1, state_size)))
            agent.actor.load_weights(model_path_actor)

            for ep in range(episodes_per_seed):
                exploit_reward, _, _ = run_episode(agent, env, use_noise=False)
                explore_reward, _, _ = run_episode(agent, env, use_noise=True)

                results.append({
                    "agent": agent_type,
                    "arrival_rate": rate,
                    "seed": seed,
                    "episode": ep + 1,
                    "exploit_reward": exploit_reward,
                    "explore_reward": explore_reward
                })

                print(f"[Rate {rate:>3}] Seed {seed:<4} Ep {ep+1} → Exploit: {exploit_reward:.2f}, Explore: {explore_reward:.2f}")

    return results, agent


def summarize(results):
    print("\n Summary by Arrival Rate:")
    for rate in ARRIVAL_RATES:
        rate_results = [r for r in results if r["arrival_rate"] == rate]
        expl_rewards = [r["exploit_reward"] for r in rate_results]
        expr_rewards = [r["explore_reward"] for r in rate_results]

        print(f"Arrival Rate {rate:>3}:")
        print(f"  Exploit → Avg: {np.mean(expl_rewards):.2f}, Std: {np.std(expl_rewards):.2f}")
        print(f"  Explore → Avg: {np.mean(expr_rewards):.2f}, Std: {np.std(expr_rewards):.2f}")


def inspect_policy(agent, num_inputs=5):
    print("\n Learned Policy Outputs")
    test_inputs = [
        np.zeros(agent.state_size),
        np.ones(agent.state_size),
        -np.ones(agent.state_size),
        np.linspace(-1, 1, agent.state_size),
        np.random.uniform(-1, 1, agent.state_size)
    ]
    for i, inp in enumerate(test_inputs[:num_inputs]):
        inp = np.array(inp).reshape(1, -1)
        action = agent.actor(inp).numpy()
        print(f"Input {i+1}: {inp.flatten()}")
        print(f"→ Action: {action.flatten()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustness test for DDPG-style agents under different arrival rates.")
    parser.add_argument("--agent", type=str, required=True, choices=["ddpg", "pqc", "vqc"])
    parser.add_argument("--model_path_actor", type=str, required=True)
    parser.add_argument("--episodes_per_seed", type=int, default=2)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    results, final_agent = evaluate(
        agent_type=args.agent,
        model_path_actor=args.model_path_actor,
        episodes_per_seed=args.episodes_per_seed
    )

    summarize(results)

    df = pd.DataFrame(results)
    csv_filename = f"robustness_results_{args.agent}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n Results saved to {csv_filename}")

    if args.inspect:
        inspect_policy(final_agent)
