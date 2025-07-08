"""
VQCDDPGMarketMakerAgent.py

Defines a Variational Quantum Circuit (VQC) DDPG market making agent using PennyLane and TensorFlow.
The agent integrates a quantum actor network with a classical critic network for trading in MBT-Gym environments.
"""

import numpy as np
import tensorflow as tf
import pennylane as qml
from tensorflow.keras import layers, optimizers, models
from pennylane.qnn import KerasLayer
from collections import deque

class VQCDDPGMarketMakerAgent:
    """
    Variational Quantum Circuit DDPG Market Making Agent.

    Attributes:
        state_dim: Dimension of environment state input.
        action_dim: Dimension of agent action output.
        action_bound: Maximum magnitude of actions.
    """
    def __init__(self, state_dim, action_dim, action_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.actor_lr = 5e-5
        self.critic_lr = 3e-4
        self.gamma = 0.99
        self.tau = 0.001
        self.reward_baseline = 0.0

        self.noise_std = 0.2
        self.noise_decay = 0.9998
        self.noise_min = 0.05
        self.noise = self.noise_std

        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.target_actor = self.build_actor()
        self.target_critic = self.build_critic()

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor_optimizer = optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = optimizers.Adam(learning_rate=self.critic_lr)

        self.buffer = deque(maxlen=1_000_000)

        self.last_actor_loss = None
        self.last_critic_loss = None

    def build_actor(self):
        """
        Builds the quantum actor network with AngleEmbedding and StronglyEntanglingLayers.
        """
        dev = qml.device("default.qubit", wires=self.state_dim)

        @qml.qnode(dev, interface="tf")
        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(self.state_dim))
            qml.templates.StronglyEntanglingLayers(weights, wires=range(self.state_dim))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.action_dim)]

        weight_shapes = {"weights": (6, self.state_dim, 3)}
        qlayer = KerasLayer(circuit, weight_shapes, output_dim=self.action_dim)

        model = models.Sequential([
            layers.Input(shape=(self.state_dim,)),
            qlayer,
            layers.Activation('tanh'),
            layers.Lambda(lambda x: x * self.action_bound)
        ])
        return model

    def build_critic(self):
        """
        Builds the classical critic network estimating Q-values.
        """
        state_input = layers.Input(shape=(self.state_dim,))
        action_input = layers.Input(shape=(self.action_dim,))
        concat = layers.Concatenate()([state_input, action_input])
        out = layers.Dense(400, activation='relu')(concat)
        out = layers.Dense(300, activation='relu')(out)
        out = layers.Dense(1)(out)
        return models.Model([state_input, action_input], out)

    def remember(self, state, action, reward, next_state, done, inventory=0):
        """
        Stores an experience in the replay buffer with inventory penalty.
        """
        inventory_penalty = 0.01 * (inventory ** 2)
        reward -= inventory_penalty
        self.reward_baseline = 0.99 * self.reward_baseline + 0.01 * reward
        adjusted_reward = reward - self.reward_baseline
        adjusted_reward = np.clip(adjusted_reward, -100, 100)
        self.buffer.append((state, action, adjusted_reward, next_state, done))

    def train(self, batch_size=256):
        """
        Trains critic and actor networks with a mini-batch.
        """
        if len(self.buffer) < batch_size:
            return

        minibatch = np.random.choice(len(self.buffer), batch_size, replace=False)

        states = np.array([self.buffer[i][0] for i in minibatch])
        states = np.squeeze(states, axis=1) if len(states.shape) == 3 else states
        actions = np.array([self.buffer[i][1] for i in minibatch])
        rewards = np.array([self.buffer[i][2] for i in minibatch])
        next_states = np.array([self.buffer[i][3] for i in minibatch])
        next_states = np.squeeze(next_states, axis=1) if len(next_states.shape) == 3 else next_states
        dones = np.array([self.buffer[i][4] for i in minibatch])

        states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tf = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards_tf = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones_tf = tf.convert_to_tensor(dones, dtype=tf.float32)
        next_states_tf = tf.convert_to_tensor(next_states, dtype=tf.float32)

        if len(rewards_tf.shape) > 1:
            rewards_tf = tf.squeeze(rewards_tf, axis=-1)

        actions_tf = tf.reshape(actions_tf, (batch_size, -1))
        penalty = tf.reduce_sum(tf.abs(actions_tf), axis=1)
        rewards_tf = tf.math.tanh(rewards_tf) - 0.001 * penalty

        with tf.GradientTape() as tape:
            next_actions = self.target_actor(next_states_tf)
            target_q = self.target_critic([next_states_tf, next_actions])
            target_q = tf.squeeze(target_q, axis=-1)
            y = rewards_tf + self.gamma * (1 - dones_tf) * target_q

            q = self.critic([states_tf, actions_tf])
            q = tf.squeeze(q, axis=-1)
            critic_loss = tf.reduce_mean(tf.square(y - q))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))
        self.last_critic_loss = critic_loss.numpy()

        with tf.GradientTape() as tape:
            actions_pred = self.actor(states_tf)
            actor_loss = -tf.reduce_mean(self.critic([states_tf, actions_pred]))
            entropy_reg = tf.reduce_mean(tf.math.reduce_std(actions_pred, axis=0))
            actor_loss -= 0.01 * entropy_reg

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        actor_grad, _ = tf.clip_by_global_norm(actor_grad, 1.0)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
        self.last_actor_loss = actor_loss.numpy()

        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)

    def act(self, state, use_noise=True):
        """
        Selects an action, optionally adding Gaussian noise.
        """
        state = np.squeeze(state)
        state = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        action = self.actor(state)[0].numpy()
        if use_noise:
            action += np.random.normal(0, self.noise, size=self.action_dim)
        action = np.clip(action, -self.action_bound, self.action_bound)
        if len(action.shape) == 1:
            action = action.reshape((1, self.action_dim))
        return action

    def update_noise(self, episode, total_episodes):
        """
        Updates exploration noise standard deviation.
        """
        self.noise = max(self.noise * self.noise_decay, self.noise_min)

    def soft_update(self, target, source):
        """
        Soft updates target network parameters.
        """
        weights = np.array(source.get_weights(), dtype=object)
        target_weights = np.array(target.get_weights(), dtype=object)
        new_weights = self.tau * weights + (1 - self.tau) * target_weights
        target.set_weights(new_weights)

    def save(self, path):
        """
        Saves model weights to disk.
        """
        self.actor.save_weights(path + "_actor.weights.h5")
        self.critic.save_weights(path + "_critic.weights.h5")
        print(f"Models saved to {path}_actor.weights.h5 and {path}_critic.weights.h5")

    def load(self, path):
        """
        Loads model weights from disk and synchronises targets.
        """
        self.actor.load_weights(path + "_actor.weights.h5")
        self.critic.load_weights(path + "_critic.weights.h5")
        self.target_actor.load_weights(path + "_actor.weights.h5")
        self.target_critic.load_weights(path + "_critic.weights.h5")
        print(f"Models loaded from {path}_actor.weights.h5 and {path}_critic.weights.h5")
