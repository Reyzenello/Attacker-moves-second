# Attacker moves second

This repository provides a framework for evaluating the safety mechanisms of language models through three distinct and complementary adversarial attack techniques. The primary goal is to test whether models can uphold safety constraints, specifically by refusing to reveal a protected "canary" token (`C4N4RY`), when subjected to sophisticated prompt manipulations.
Paper: https://arxiv.org/abs/2510.09023

## 📝 Overview

The core of this project is to simulate a scenario where an attacker has the second move, allowing them to adaptively probe and attempt to bypass a model's deployed safety filters. This is achieved through a combination of black-box, gray-box, and white-box attack methodologies.

## 🚀 Attack Methodologies

The framework implements the following three attack strategies:

### 1. RL-Based Black-Box Bandit Attack
This approach uses reinforcement learning to discover effective prompt transformations without needing access to the model's internal workings.

*   **Approach**: Employs a multi-armed bandit algorithm with an epsilon-greedy exploration strategy.
*   **Strategy**: Iteratively applies various formatting operators (like Markdown, XML tags, JSON encoding, Unicode noise, and role prefixes) to the input prompt. The bandit learns which transformations are most successful at bypassing the safety filters.
*   **Key Features**:
    *   Gradient-free optimization.
    *   Dynamic evolution of adversarial suffixes.
    *   Requires only input/output access to the target model.

### 2. Search-Based LLM-Guided Evolutionary Attack
This method utilizes a genetic algorithm to evolve a population of effective adversarial prompts over multiple generations.

*   **Approach**: A population-based search guided by a genetic algorithm.
*   **Strategy**: A set of candidate prompts is initialized and then evolved through mutations, such as inserting wrappers, adding Unicode noise, or including role-based hints.
*   **Key Features**:
    *   Starts with a diverse set of stress-test scenarios.
    *   Selects the top-k performing prompts as parents for the next generation.
    *   Employs a variety of mutation operators to explore the search space.

### 3. Gradient-Based Surrogate Transfer Attack
This white-box approach optimizes an attack on a surrogate model and then transfers it to the target model.

*   **Approach**: White-box optimization performed on a surrogate model (GPT-2).
*   **Strategy**: An adversarial suffix is optimized by manipulating its continuous embedding to maximize or minimize the probability of the canary token's appearance. This soft prompt is then discretized into actual tokens for transfer.
*   **Key Features**:
    *   Soft prompt optimization.
    *   Token discretization based on cosine similarity.
    *   Generates transferable adversarial suffixes.
    *   Requires PyTorch and the Transformers library.

## 🏗️ Architecture

The general workflow of an attack is as follows:

```
Benign Task + Adversarial Suffix → [Target LLM] → Response
                                        ↓
                                 Violation Detector
                                        ↓
                                  JSONL Traces
```

Each attack logs detailed traces in JSONL format, capturing the following information:
*   Timestamp, model, and task details
*   The full prompt and the model's response
*   A binary flag indicating if a safety violation occurred
*   Latency measurements
*   Attack-specific metadata (e.g., generation, step, operator used)

## 🛠️ Usage

### Prerequisites

*   Python 3.x
*   [Ollama](https://ollama.ai/) running locally on port `11434`.

### Installation

1.  **Core Dependencies**:
    ```bash
    pip install requests
    ```

2.  **For Gradient-Based Attacks**:
    ```bash
    pip install torch transformers
    ```

### Running the Attacks

You can run the attacks with default parameters or customize them as needed.

*   **Basic Run**:
    ```bash
    python attacker_moves_second.py --model gemma3
    ```

*   **Customized Configuration**:
    ```bash
    python attacker_moves_second.py \
      --model llama3 \
      --trace_dir ./experiment_logs \
      --rl_steps 100 \
      --evo_generations 15 \
      --grad_steps 300 \
      --grad_objective maximize
    ```

## ⚙️ Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--model` | `gemma3` | The name of the Ollama model to be used. |
| `--trace_dir` | `traces` | The directory where the output JSONL logs will be stored. |
| `--rl_steps` | `50` | The number of iterations for the RL bandit attack. |
| `--rl_eps` | `0.2` | The epsilon value for exploration in the RL bandit. |
| `--evo_generations` | `10` | The number of generations for the evolutionary attack. |
| `--evo_pop` | `10` | The population size for each generation in the evolutionary attack. |
| `--evo_topk` | `3` | The number of top parents to select for the next generation. |
| `--grad_steps` | `200` | The number of optimization steps for the gradient-based attack. |
| `--grad_len` | `12` | The token length of the adversarial suffix in the gradient attack. |
| `--grad_objective` | `maximize` | The optimization objective (`maximize` or `minimize`) for the canary probability. |

## 📊 Output

For each run, the framework generates three JSONL trace files:

*   `traces_bandit_<timestamp>.jsonl`: Logs from the RL-based attack.
*   `traces_evo_<timestamp>.jsonl`: Logs from the evolutionary search attack.
*   `traces_grad_<timestamp>.jsonl`: Logs from the gradient-based attack.

A final scoreboard is displayed in the console, summarizing the violation rates for each attack type.

## 🔬 Research Context

This implementation delves into the "attacker moves second" paradigm, where adversarial methods are adapted to probe and challenge deployed safety mechanisms. The three attack families represent different threat models:

*   **Black-box (RL)**: Assumes no access to the model's internal state, relying solely on its inputs and outputs.
*   **Gray-box (Search)**: Involves query access to the model with iterative refinement of the attack prompts.
*   **White-box (Gradient)**: Requires full access to a surrogate model to craft attacks that can be transferred.
