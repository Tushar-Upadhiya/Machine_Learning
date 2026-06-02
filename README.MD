# Machine Learning, Deep Learning & RL from Scratch 🧠🚀

Welcome to my personal engineering kitchen. This repository serves as a code-first documentation of my journey mastering **Reinforcement Learning** and **Deep Learning** from foundational theory to fully custom implementations. 

Instead of treating frameworks like Black Boxes, every script here focuses on translating rigorous mathematical formulas directly into pure Python, NumPy, or low-level tensor operations.

---

## 📚 Core Curriculums & Resources

This repository follows structural tracks mapped to world-class academic and industry curricula:

1.  **Reinforcement Learning Track:**
    * 📖 *Book:* **Sutton & Barto** – *Reinforcement Learning: An Introduction*
    * 🎥 *Lectures:* **Stanford CS234** (Reinforcement Learning)
2.  **Deep Learning Track:**
    * 📖 *Book:* **The Deep Learning Book** (Goodfellow, Bengio, and Courville)
    * 🎥 *Lectures:* **CampusX** – *Deep Learning Masterclass*

---

## 🛠️ Repository Structure & Milestones

### 🔬 Part 1: Reinforcement Learning (Sutton & Barto / Stanford CS234)

Focuses on transition dynamics, evaluative feedback loops, and sequential decision-making environments.

* **Multi-Armed Bandits (Chapter 2):**
    * Implementation of the space-efficient **Incremental Sample-Average Update Rule**:
        $$Q_{k+1} = Q_k + \frac{1}{k} (R_k - Q_k)$$
    * Simulations comparing $\epsilon$-greedy exploration mechanics vs. pure exploitation.
* **Markov Decision Processes (Chapter 3):**
    * **The Recycling Robot System:** Custom simulation mapping stochastic transition graphs (States: `high`/`low` battery; Actions: `search`/`wait`/`recharge`) handled via uniform random branching.
    * **Grid World MDP:** A custom, stochastic $3 \times 3$ grid environment featuring terminal states (Goal/Pitfalls), step penalties, and custom slippery floor transition noise (veering $90^\circ$ left/right).
    * **Bellman Optimality Backups:** Direct iterative value calculations updating expected long-term rewards:
        $$q_{k+1}(s, a) = \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \max_{a'} q_k(s', a') \right]$$

### 🧠 Part 2: Deep Learning (Goodfellow / CampusX Masterclass)

Focuses on representation learning, optimization math, and architectural implementations built from the ground up using vectorization.

* **Linear & Logistic Foundations:**
    * Mathematical implementation of loss functions (Mean Squared Error, Binary Cross-Entropy) and optimization via Batch/Stochastic Gradient Descent (SGD) entirely from scratch.
* **Deep Neural Networks (DNNs) from Scratch:**
    * **Forward Propagation:** Building explicit matrix multiplication layers, tracking tensor weights $W$ and biases $b$, alongside element-wise activation functions (Sigmoid, ReLU, Tanh).
    * **Backward Propagation & Chain Rule:** Manual derivation and implementation of partial derivatives ($\frac{\partial L}{\partial W}$, $\frac{\partial L}{\partial b}$) using calculus chain rules across multiple hidden layers.
* **Optimization & Regularization Math:**
    * Low-level mechanics of advanced optimizers (Momentum, RMSprop, Adam) and numerical stability techniques (Vanishing/Exploding gradients remedies).

---

## 📈 Core Takeaways & Learning Objectives

1.  **Memory Optimization:** Understanding how incremental math allows updating statistical expectations using only three floating-point variables ($Q_k, k, R_k$) rather than storing memory-heavy arrays of past history.
2.  **Stochastic Graph Modeling:** Translating environmental transition functions ($p(s', r \mid s, a)$) into structured algorithmic conditional pathways.
3.  **Vectorized Matrix Math:** Moving away from standard Python loops to clean, vectorized linear algebra operations for network backpropagation, matching text equations line-by-line.
