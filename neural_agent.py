"""
Biologically-Inspired Neural Trading Agent
─────────────────────────────────────────────────────────────────────────────
Architecture:
  • Deep Q-Network (DQN) with experience replay — learns optimal trade decisions
  • Hebbian synaptic strengthening — frequently co-activated paths self-reinforce
  • Spike-timing inspired eligibility traces — reward attribution across time
  • Continuous online learning — updates after every trade outcome
  • Adaptive exploration (epsilon-greedy decay) — explores early, exploits later

Actions: 0=HOLD, 1=BUY, 2=SELL
"""

import numpy as np
import os
import json
import logging
from collections import deque
import random
import config

log = logging.getLogger("neural_agent")

# ─────────────────────────────────────────────────────────────────────────────
class NeuralNet:
    """
    Lightweight feed-forward network using only NumPy.
    No heavy framework dependencies — runs fast in any env.
    Architecture: Input → Hidden1 → Hidden2 → Output
    Activation: LeakyReLU hidden, linear output (Q-values)
    """

    def __init__(self, state_size: int, hidden: int, action_size: int, lr: float):
        self.lr = lr
        self.state_size  = state_size
        self.hidden      = hidden
        self.action_size = action_size

        # Xavier initialization for stable gradients
        scale1 = np.sqrt(2.0 / state_size)
        scale2 = np.sqrt(2.0 / hidden)

        self.W1 = np.random.randn(state_size, hidden) * scale1
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, hidden) * scale2
        self.b2 = np.zeros(hidden)
        self.W3 = np.random.randn(hidden, action_size) * np.sqrt(2.0/hidden)
        self.b3 = np.zeros(action_size)

        # Hebbian eligibility traces (one per weight matrix)
        self.trace1 = np.zeros_like(self.W1)
        self.trace2 = np.zeros_like(self.W2)
        self.trace3 = np.zeros_like(self.W3)
        self.trace_decay = 0.9   # biological spike timing decay

    def _leaky_relu(self, x: np.ndarray, alpha=0.01) -> np.ndarray:
        return np.where(x > 0, x, alpha * x)

    def _leaky_relu_grad(self, x: np.ndarray, alpha=0.01) -> np.ndarray:
        return np.where(x > 0, 1.0, alpha)

    def forward(self, x: np.ndarray) -> tuple:
        """Returns (q_values, cache) where cache holds intermediate activations."""
        z1 = x @ self.W1 + self.b1
        a1 = self._leaky_relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._leaky_relu(z2)
        z3 = a2 @ self.W3 + self.b3
        return z3, (x, z1, a1, z2, a2, z3)

    def predict(self, x: np.ndarray) -> np.ndarray:
        q, _ = self.forward(x)
        return q

    def update(self, x: np.ndarray, targets: np.ndarray, hebbian_reward: float = 0.0):
        """
        Single-sample gradient descent + Hebbian update.
        hebbian_reward: +1 for win, -1 for loss — modulates synaptic potentiation
        """
        q, (inp, z1, a1, z2, a2, z3) = self.forward(x)

        # MSE loss gradient
        dL = 2 * (q - targets)   # shape: (action_size,)

        # Backprop layer 3
        dW3 = np.outer(a2, dL)
        db3 = dL
        da2 = dL @ self.W3.T

        # Layer 2
        dz2 = da2 * self._leaky_relu_grad(z2)
        dW2 = np.outer(a1, dz2)
        db2 = dz2
        da1 = dz2 @ self.W2.T

        # Layer 1
        dz1 = da1 * self._leaky_relu_grad(z1)
        dW1 = np.outer(inp, dz1)
        db1 = dz1

        # ── Hebbian modulation ────────────────────────────────────────────────
        # Update eligibility traces (like STDP: pre * post correlation)
        self.trace1 = self.trace_decay * self.trace1 + np.outer(inp, a1)
        self.trace2 = self.trace_decay * self.trace2 + np.outer(a1,  a2)
        self.trace3 = self.trace_decay * self.trace3 + np.outer(a2,  q)

        # Hebbian term: strengthen co-active synapses when rewarded
        heb_rate = 0.0001 * hebbian_reward
        hW1 = heb_rate * self.trace1
        hW2 = heb_rate * self.trace2
        hW3 = heb_rate * self.trace3

        # Apply gradient descent + Hebbian
        self.W1 -= self.lr * dW1 + hW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2 + hW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3 + hW3
        self.b3 -= self.lr * db3

    def copy_weights_from(self, other: "NeuralNet"):
        self.W1[:] = other.W1; self.b1[:] = other.b1
        self.W2[:] = other.W2; self.b2[:] = other.b2
        self.W3[:] = other.W3; self.b3[:] = other.b3

    def save(self, path: str):
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)
        log.info(f"Model saved → {path}")

    def load(self, path: str):
        data = np.load(path)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], data["b2"]
        self.W3, self.b3 = data["W3"], data["b3"]
        log.info(f"Model loaded ← {path}")


# ─────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    """Experience replay — breaks temporal correlation in training data."""

    def __init__(self, maxlen: int):
        self.buf = deque(maxlen=maxlen)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done)
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


# ─────────────────────────────────────────────────────────────────────────────
class TradingAgent:
    """
    Deep Q-Network agent with biologically inspired learning.

    Decision process:
      1. Receive market feature vector (state)
      2. Epsilon-greedy action selection (explore vs exploit)
      3. Execute trade
      4. Receive reward (realized PnL + shaped signals)
      5. Store in replay buffer
      6. Batch train on random sample of past experiences
      7. Periodic target network sync (stabilizes training)
      8. Save model weights after each update
    """

    ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}

    def __init__(self):
        ss = config.STATE_SIZE
        hs = config.HIDDEN_SIZE
        as_ = config.ACTION_SIZE

        self.policy_net  = NeuralNet(ss, hs, as_, config.LEARNING_RATE)
        self.target_net  = NeuralNet(ss, hs, as_, config.LEARNING_RATE)
        self.target_net.copy_weights_from(self.policy_net)

        self.replay      = ReplayBuffer(config.REPLAY_BUFFER_SIZE)
        self.epsilon     = config.EPSILON_START
        self.step_count  = 0
        self.total_reward= 0.0
        self.wins        = 0
        self.losses      = 0

        # Load saved weights if available
        if os.path.exists(config.MODEL_SAVE_PATH + ".npz"):
            try:
                self.policy_net.load(config.MODEL_SAVE_PATH + ".npz")
                self.target_net.copy_weights_from(self.policy_net)
                # Resume with lower epsilon since we have prior knowledge
                self.epsilon = max(config.EPSILON_MIN * 5, 0.2)
                log.info("Loaded existing model — resuming training")
            except Exception as e:
                log.warning(f"Could not load model: {e}. Starting fresh.")

    # ── Core Interface ────────────────────────────────────────────────────────
    def decide(self, state: list, rule_signal: str, rule_confidence: float) -> str:
        """
        Choose action given state features + rule-based signal as prior.
        Returns "BUY", "SELL", or "HOLD"
        """
        s = np.array(state, dtype=np.float32)

        if random.random() < self.epsilon:
            # Explore — but bias toward the rule signal (not fully random)
            if random.random() < 0.5:
                action = {"BUY": 1, "SELL": 2, "HOLD": 0}.get(rule_signal, 0)
            else:
                action = random.randint(0, config.ACTION_SIZE - 1)
        else:
            # Exploit — use Q-network
            q_vals = self.policy_net.predict(s)
            # Blend: Q-value + small bonus for rule signal agreement
            rule_idx = {"BUY": 1, "SELL": 2, "HOLD": 0}.get(rule_signal, 0)
            q_vals[rule_idx] += rule_confidence * 0.5  # soft prior injection
            action = int(np.argmax(q_vals))

        return self.ACTION_NAMES[action]

    def record_experience(self, state, action_str, reward, next_state, done=False):
        """Store experience and trigger learning."""
        action = {"HOLD": 0, "BUY": 1, "SELL": 2}[action_str]
        self.replay.push(state, action, reward, next_state, done)
        self.total_reward += reward

        if reward > 0: self.wins  += 1
        elif reward < -0.001: self.losses += 1

        # Track Hebbian signal from trade outcome
        hebbian_signal = np.sign(reward)
        self.policy_net.trace1 *= 1.0  # keep existing traces

        if len(self.replay) >= config.BATCH_SIZE:
            self._train_batch(hebbian_signal)

        # Target network sync
        self.step_count += 1
        if self.step_count % config.TARGET_UPDATE_FREQ == 0:
            self.target_net.copy_weights_from(self.policy_net)
            self.policy_net.save(config.MODEL_SAVE_PATH)
            log.info(f"🧠 Target net synced | ε={self.epsilon:.3f} | "
                     f"wins={self.wins} losses={self.losses} "
                     f"total_reward={self.total_reward:.4f}")

        # Decay exploration
        self.epsilon = max(config.EPSILON_MIN, self.epsilon * config.EPSILON_DECAY)

    def _train_batch(self, hebbian_signal: float = 0.0):
        """DQN batch update with Bellman equation."""
        states, actions, rewards, next_states, dones = self.replay.sample(config.BATCH_SIZE)

        for i in range(config.BATCH_SIZE):
            s  = states[i]
            a  = actions[i]
            r  = rewards[i]
            ns = next_states[i]
            d  = dones[i]

            q_curr  = self.policy_net.predict(s).copy()
            q_next  = self.target_net.predict(ns)
            target  = r + config.GAMMA * np.max(q_next) * (1 - d)
            q_curr[a] = target  # only update chosen action's Q-value

            self.policy_net.update(s, q_curr, hebbian_reward=hebbian_signal)

    # ── Stats ─────────────────────────────────────────────────────────────────
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "epsilon":     round(self.epsilon, 4),
            "total_reward":round(self.total_reward, 6),
            "wins":        self.wins,
            "losses":      self.losses,
            "win_rate":    round(self.win_rate(), 3),
            "replay_size": len(self.replay),
            "steps":       self.step_count,
        }
