"""
PQCDDPGMarketMakerAgent.py

Implements a Parameterized Quantum Circuit (PQC) DDPG market making agent using PennyLane and TensorFlow.
Includes a quantum actor network with fixed Ry rotations and a classical critic network for continuous action trading environments.
"""

import pennylane as qml
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from pennylane.qnn import KerasLayer
from collections import deque
import random

class GaussianActionNoise:
    """
    Gaussian noise process for action exploration.
    """
    def __init__(self, mean, std_dev):
        self.mean = mean
        self.std_dev = std_dev

    def __call__(self):
        return np.random.normal(self.mean, self.std_dev)

class PQCDDPGMarketMakerAgent:
    """
    Parameterized Quantum Circuit DDPG Market Making Agent.

    Attributes:
        state_size: Dimension of environment state input.
        action_size: Dimension of agent action output.
        action_bounds: Maximum magnitude of actions.
        n_qubits: Number of qubits in the quantum circuit.
    """

    def __init__(self, state_size, action_size, action_bounds, n_qubits=4, seed=None):
        self.state_size = state_size
        self.action_size = action_size
        self.action_bounds = action_bounds
        self.n_qubits = n_qubits

        self.gamma = 0.99
        self.tau = 0.001
        self.buffer = deque(maxlen=1000000)
        self.batch_size = 256

        # Build actor and target actor networks
        self.actor = self._build_actor()
        self.target_actor = self._build_actor()
        self.target_actor.set_weights(self.actor.get_weights())

        # Build critic and target critic networks
        self.critic = self._build_critic()
        self.target_critic = self._build_critic()
        self.target_critic.set_weights(self.critic.get_weights())

        # Define optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

        # Initialise Gaussian action noise for exploration
        self.noise = GaussianActionNoise(mean=np.zeros(self.action_size), std_dev=0.2 * np.ones(self.action_size))
        self.noise_std = 0.2
        self.noise_decay = 0.9998
        self.noise_min = 0.05

    def _build_actor(self):
        """
        Builds the quantum actor network using a PQC layer with fixed Ry rotations.

        Returns:
            TensorFlow Keras model with a quantum KerasLayer.
        """
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="tf")
        def circuit(inputs):
            # Embed classical inputs as qubit rotations
            qml.templates.AngleEmbedding(inputs, wires=range(self.n_qubits))

            # Apply fixed Ry rotations as static quantum feature transformation
            fixed_angles = np.linspace(0, np.pi/4, self.n_qubits)
            for i in range(self.n_qubits):
                qml.RY(fixed_angles[i], wires=i)

            # Apply fixed CNOT entanglement between qubits
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Measure Pauli-Z expectation values as outputs
            return [qml.expval(qml.PauliZ(i)) for i in range(self.action_size)]

        qlayer = KerasLayer(circuit, {}, output_dim=self.action_size)

        # Define full actor model with classical preprocessing -> quantum layer -> output scaling
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(self.n_qubits, activation="tanh")(inputs)
        x = qlayer(x)
        x = layers.Dense(self.action_size, activation="tanh")(x)
        outputs = layers.Lambda(lambda a: a * self.action_bounds)(x)

        return tf.keras.Model(inputs, outputs)

    def _build_critic(self):
        """
        Builds the classical critic network mapping (state, action) pairs to Q-values.

        Returns:
            TensorFlow Keras model.
        """
        state_input = layers.Input(shape=(self.state_size,))
        action_input = layers.Input(shape=(self.action_size,))
        x = layers.Concatenate()([state_input, action_input])
        x = layers.Dense(400, activation="relu")(x)
        x = layers.Dense(300, activation="relu")(x)
        x = layers.Dense(1)(x)
        return tf.keras.Model([state_input, action_input], x)

    def act(self, state, use_noise=True):
        """
        Selects an action given a state, optionally adding Gaussian noise for exploration.
        """
        state = np.asarray(state).reshape(1, -1).astype(np.float32)
        action = self.actor(tf.convert_to_tensor(state))[0].numpy()
        if use_noise:
            action += self.noise()  # Add exploration noise
        return np.clip(action, -self.action_bounds, self.action_bounds).reshape((1, -1))

    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience in the replay buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def train(self):
        """
        Trains the actor and critic networks using sampled experiences from the replay buffer.
        Applies soft updates to target networks.
        """
        if len(self.buffer) < self.batch_size:
            return  # Not enough samples to train

        minibatch = random.sample(self.buffer, self.batch_size)

        # Prepare mini-batch tensors
        states = np.array([i[0].reshape(-1) for i in minibatch], dtype=np.float32)
        actions = np.array([i[1].reshape(-1) for i in minibatch], dtype=np.float32)
        rewards = np.array([i[2] for i in minibatch], dtype=np.float32)
        next_states = np.array([i[3].reshape(-1) for i in minibatch], dtype=np.float32)
        dones = np.array([i[4] for i in minibatch], dtype=np.float32)

        states_tf = tf.convert_to_tensor(states)
        actions_tf = tf.convert_to_tensor(actions)
        rewards_tf = tf.convert_to_tensor(rewards)
        next_states_tf = tf.convert_to_tensor(next_states)
        dones_tf = tf.convert_to_tensor(dones)

        # Critic training step
        next_actions_tf = self.target_actor(next_states_tf)
        target_q = self.target_critic([next_states_tf, next_actions_tf])
        target_q = rewards_tf + self.gamma * tf.squeeze(target_q) * (1 - dones_tf)

        with tf.GradientTape() as tape:
            q_values = tf.squeeze(self.critic([states_tf, actions_tf]))
            critic_loss = tf.keras.losses.MSE(target_q, q_values)
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        # Actor training step with entropy regularisation for exploration
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states_tf)
            actor_loss = -tf.reduce_mean(self.critic([states_tf, actions_pred]))
            entropy_reg = tf.reduce_mean(tf.math.reduce_std(actions_pred, axis=0))
            actor_loss -= 0.01 * entropy_reg  # Encourage diverse actions
            grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        # Soft update target networks
        self._update_target(self.target_actor, self.actor)
        self._update_target(self.target_critic, self.critic)

    def _update_target(self, target, source):
        """
        Soft update target network parameters towards source network parameters.
        """
        target.set_weights([self.tau * w + (1 - self.tau) * t for w, t in zip(source.get_weights(), target.get_weights())])

    def update_noise(self, episode, total_episodes):
        """
        Updates the exploration noise standard deviation with decay.
        """
        self.noise_std = max(self.noise_std * self.noise_decay, self.noise_min)
        self.noise = GaussianActionNoise(
            mean=np.zeros(self.action_size),
            std_dev=self.noise_std * np.ones(self.action_size)
        )

    def save(self, path):
        """
        Saves actor and critic model weights to disk.
        """
        self.actor.save_weights(path + "_actor.weights.h5")
        self.critic.save_weights(path + "_critic.weights.h5")
        print(f"Models saved to {path}_actor.weights.h5 and {path}_critic.weights.h5")

    def load(self, path):
        """
        Loads actor and critic model weights from disk and synchronises target networks.
        """
        self.actor.load_weights(path + "_actor.weights.h5")
        self.critic.load_weights(path + "_critic.weights.h5")
        self.target_actor.load_weights(path + "_actor.weights.h5")
        self.target_critic.load_weights(path + "_critic.weights.h5")
        print(f"Models loaded from {path}_actor.weights.h5 and {path}_critic.weights.h5")
