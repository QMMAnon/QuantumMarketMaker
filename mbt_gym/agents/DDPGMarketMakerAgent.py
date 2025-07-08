"""
DDPGMarketMakerAgent.py

Implements a Deep Deterministic Policy Gradient (DDPG) market making agent for continuous action trading environments.
Includes actor-critic networks, experience replay, target network updates, and Gaussian action noise for exploration.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random

class GaussianActionNoise:
    def __init__(self, mean, std_dev, seed=None):
        self.mean = mean
        self.std_dev = std_dev
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        return self.rng.normal(self.mean, self.std_dev)

class DDPGMarketMakerAgent:
    """
    Deep Deterministic Policy Gradient Market Making Agent.
    """
    def __init__(self, state_size, action_size, action_bounds, seed=None):
        self.state_size = state_size
        self.action_size = action_size
        self.action_bounds = action_bounds

        self.gamma = 0.99
        self.tau = 0.001
        self.buffer = deque(maxlen=1000000)
        self.batch_size = 256
        self.warmup_steps = 1000
        self.reward_baseline = 0.0
        self.train_step = 0

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.replay_rng = random.Random(seed)

        self.actor = self._build_actor(seed)
        self.critic = self._build_critic(seed)
        self.target_actor = self._build_actor(seed)
        self.target_critic = self._build_critic(seed)
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

        self.noise = GaussianActionNoise(
            mean=np.zeros(self.action_size),
            std_dev=0.2 * np.ones(self.action_size),
            seed=seed
        )

    def _build_actor(self, seed=None):
        """
        Builds the actor network.
        """
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(400, activation='swish',
                         kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(inputs)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(300, activation='swish',
                         kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(200, activation='swish',
                         kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
        x = layers.LayerNormalization()(x)
        outputs = layers.Dense(self.action_size, activation="tanh",
                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
        outputs = layers.Lambda(lambda x: x * self.action_bounds)(outputs)
        return tf.keras.Model(inputs, outputs)

    def _build_critic(self, seed=None):
        """
        Builds the critic network.
        """
        state_input = layers.Input(shape=(self.state_size,))
        action_input = layers.Input(shape=(self.action_size,))
        x = layers.Concatenate()([state_input, action_input])
        x = layers.Dense(400, activation='swish',
                         kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(300, activation='swish',
                         kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(200, activation='swish',
                         kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
        x = layers.LayerNormalization()(x)
        outputs = layers.Dense(1,
                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
        return tf.keras.Model([state_input, action_input], outputs)

    def act(self, state, use_noise=True):
        """
        Selects an action given a state.
        """
        state = np.asarray(state).reshape(1, -1).astype(np.float32)
        action = self.actor.predict(state, verbose=0)[0]
        if use_noise:
            action += self.noise()
        action = np.clip(action, -self.action_bounds, self.action_bounds)
        return action.reshape((1, -1))

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

    def train(self):
        """
        Trains actor and critic networks using sampled experiences.
        """
        if len(self.buffer) < max(self.batch_size, self.warmup_steps):
            return

        self.train_step += 1
        minibatch = self.replay_rng.sample(self.buffer, self.batch_size)
        states = np.array([item[0].reshape(-1) for item in minibatch], dtype=np.float32)
        actions = np.array([item[1].reshape(-1) for item in minibatch], dtype=np.float32)
        rewards = np.array([x[2] for x in minibatch], dtype=np.float32)
        next_states = np.array([item[3].reshape(-1) for item in minibatch], dtype=np.float32)
        dones = np.array([x[4] for x in minibatch], dtype=np.float32)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        next_actions = self.target_actor(next_states, training=False)
        noise = tf.clip_by_value(0.1 * tf.random.normal(shape=next_actions.shape), -0.2, 0.2)
        next_actions = tf.clip_by_value(next_actions + noise, -self.action_bounds, self.action_bounds)

        next_q = self.target_critic([next_states, next_actions], training=False)
        next_q = tf.squeeze(next_q, axis=1)
        target_q = rewards + self.gamma * next_q * (1 - dones)
        target_q = tf.clip_by_value(target_q, -1000, 1000)

        with tf.GradientTape() as tape:
            q_values = self.critic([states, actions], training=True)
            q_values = tf.squeeze(q_values, axis=1)
            critic_loss = tf.keras.losses.Huber()(target_q, q_values)
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_actions = self.actor(states, training=True)
            q_val = self.critic([states, new_actions], training=True)
            entropy = -tf.reduce_mean(new_actions * tf.math.log(tf.abs(new_actions) + 1e-6))
            actor_loss = -tf.reduce_mean(q_val) + 0.01 * entropy
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        self._update_target_network(self.target_actor, self.actor)
        self._update_target_network(self.target_critic, self.critic)

    def _update_target_network(self, target_model, model):
        """
        Soft updates target network parameters.
        """
        new_weights = [self.tau * w + (1 - self.tau) * t
                       for t, w in zip(target_model.get_weights(), model.get_weights())]
        target_model.set_weights(new_weights)

    def save(self, prefix="ddpg"):
        """
        Saves model weights.
        """
        self.actor.save_weights(f"{prefix}_actor.weights.h5")
        self.critic.save_weights(f"{prefix}_critic.weights.h5")

    def load(self, prefix="ddpg"):
        """
        Loads model weights and syncs targets.
        """
        self.actor.load_weights(f"{prefix}_actor.weights.h5")
        self.critic.load_weights(f"{prefix}_critic.weights.h5")
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def update_noise(self, episode, max_episodes):
        """
        Updates Gaussian noise for exploration.
        """
        decay = max(0.05, 0.2 * (1 - episode / max_episodes))
        self.noise = GaussianActionNoise(
            mean=np.zeros(self.action_size),
            std_dev=decay * np.ones(self.action_size),
            seed=self.seed
        )
