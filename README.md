# Attacker-moves-second
unofficial implementation of the paper "attacker moves second" focused on AI secure

Overview
This repository implements three complementary adversarial attack techniques to evaluate language model safety mechanisms. The framework tests whether models can maintain safety constraints (specifically, refusing to reveal a protected canary token C4N4RY) when faced with sophisticated prompt manipulations.
Attack Methodologies
1. RL-Based Black-Box Bandit
Approach: Reinforcement learning with epsilon-greedy exploration
Strategy: Iteratively applies formatting operators (Markdown wrapping, XML tags, JSON encoding, Unicode noise, role prefixes) and learns which transformations are most effective at bypassing safety filters.
Key Features:

Multi-armed bandit optimization
Dynamic suffix evolution
No gradient access required

2. Search-Based LLM-Guided Evolutionary
Approach: Genetic algorithm with population-based search
Strategy: Maintains a population of candidate prompts and evolves them through mutations (wrapper insertion, Unicode noise injection, role-based hints) across generations.
Key Features:

Seeded initialization with diverse stress-test scenarios
Top-k parent selection
Multiple mutation operators

3. Gradient-Based Surrogate Transfer
Approach: White-box optimization on surrogate model (GPT-2)
Strategy: Optimizes a continuous suffix embedding to maximize/minimize canary token probability, then discretizes to actual tokens for transfer to target model.
Key Features:

Soft prompt optimization
Cosine similarity-based token discretization
Transferable adversarial suffixes
Requires PyTorch & Transformers

Architecture
benign task + adversarial suffix → [Target LLM] → response
                                          ↓
                                   violation detector
                                          ↓
                                    JSONL traces
Each attack logs comprehensive traces including:

Timestamp, model, task details
Full prompt and response
Violation flag (binary)
Latency measurements
Attack metadata (generation, step, operator)

Usage
bash# Basic run with default parameters
python attacker_moves_second.py --model gemma3

# Customized attack configuration
python attacker_moves_second.py \
  --model llama3 \
  --trace_dir ./experiment_logs \
  --rl_steps 100 \
  --evo_generations 15 \
  --grad_steps 300 \
  --grad_objective maximize
Parameters
ParameterDefaultDescription--modelgemma3Ollama model name--trace_dirtracesOutput directory for JSONL logs--rl_steps50RL bandit iterations--rl_eps0.2Epsilon for exploration--evo_generations10Evolutionary generations--evo_pop10Population size--evo_topk3Parents per generation--grad_steps200Gradient optimization steps--grad_len12Suffix token length--grad_objectivemaximizemaximize or minimize canary probability
Requirements
bash# Core dependencies
pip install requests

# For gradient-based attacks
pip install torch transformers
External: Requires Ollama running locally on port 11434
Output
The framework generates three JSONL trace files per run:

traces_bandit_<timestamp>.jsonl - RL attack logs
traces_evo_<timestamp>.jsonl - Evolutionary search logs
traces_grad_<timestamp>.jsonl - Gradient-based attack logs

Final scoreboard displays violation rates across all attack types.
Research Context
This implementation explores the "attacker moves second" paradigm where adversaries adaptively probe deployed safety mechanisms. The three attack families represent different threat models:

Black-box (RL): No model access, only input/output
Gray-box (Search): Query access with iterative refinement
White-box (Gradient): Full surrogate model access with transfer
