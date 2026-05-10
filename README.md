# Deep Neural Optimization for GPU Inference Scheduling

A learning-based GPU inference scheduler that combines imitation learning and reinforcement learning to optimize job placement across shared GPU clusters under dynamic workloads.

Built using Transformer-based policy networks, discrete-event simulation, and rollout-based policy-gradient refinement to improve throughput, GPU utilization, and tail latency under overloaded cluster conditions.

---

## Overview

Modern ML serving systems must schedule heterogeneous inference workloads across limited GPU resources in real time. Traditional scheduling heuristics such as FIFO and Shortest-Job-First (SJF) suffer from major trade-offs:

- **FIFO** preserves fairness but causes head-of-line blocking and poor utilization
- **SJF** minimizes average completion time but introduces severe tail latency and starvation under heavy load

This project introduces a learned scheduling framework that:
- Learns scheduling behavior from simulation data using imitation learning
- Refines policies using rollout-based reinforcement learning
- Models GPU memory fragmentation and co-location interference
- Dynamically balances throughput, latency, fairness, and utilization

---

# Key Features

- Transformer-based neural scheduling policy
- Discrete-event GPU cluster simulator
- Memory-aware scheduling constraints
- Co-location interference modeling
- Imitation learning from oracle SJF policies
- Reinforcement learning refinement via policy gradients
- Evaluation framework for FIFO, SJF, and learned schedulers

---

# System Architecture

The simulator models:
- Poisson job arrivals
- GPU memory allocation
- Multi-job co-location
- Queue dynamics
- Resource fragmentation
- Interference-aware execution slowdown

Each scheduling decision observes:
- Waiting job queue
- GPU memory availability
- GPU utilization
- Running job counts

The learned scheduler outputs feasible `(job, GPU)` assignments subject to memory constraints.

---

# Neural Scheduler

The scheduler uses a Transformer encoder to jointly reason over:
- Job features
- Cluster state
- GPU resource availability

## Job Features
- Model size
- Batch size
- Sequence length
- Utilization intensity
- Memory requirements

## GPU Features
- Free memory
- Current utilization
- Number of running jobs

The model produces a dense score matrix over all feasible `(job, GPU)` pairs and greedily selects assignments during inference.

---

# Training Pipeline

## Stage 1 — Imitation Learning

The scheduler first learns from an oracle SJF teacher policy using supervised learning.

- Teacher labels generated from simulator rollouts
- Cross-entropy loss over job-GPU assignments
- Trained across multiple workload intensities
- Learns strong scheduling priors from expert decisions

## Stage 2 — Reinforcement Learning Refinement

The imitation policy is refined using rollout-based policy-gradient updates.

Optimization objectives:
- Reduce average JCT
- Reduce P99 latency
- Increase GPU utilization

Training includes:
- Multi-rollout updates
- Entropy regularization
- Behavior cloning stabilization
- Advantage normalization

---

# Experimental Results

Evaluated on:
- 7,500 simulated jobs
- 10-GPU cluster
- 80 GB per GPU

## Heavy Load Performance (λ = 1.0)

| Scheduler | Avg JCT ↓ | P99 ↓ | Throughput ↑ | GPU Util ↑ |
|---|---|---|---|---|
| FIFO | 395.92 | 758.18 | 0.559 | 62.23% |
| SJF | 165.59 | 1343.48 | 0.538 | 60.62% |
| Neural (Refined) | 216.35 | 1308.80 | 0.595 | 66.15% |

### Highlights
- ~11% higher throughput vs SJF
- 66.6% GPU utilization achieved
- Reduced P99 latency relative to SJF
- ~7% JCT improvement after RL refinement

---

# Future Work

Potential extensions include:
- Learned runtime estimation models
- Real-world cluster trace integration
- More advanced RL optimization
- Distributed multi-node scheduling
- Production-scale inference serving

---

# Authors

- Mokshith Reddy Mannuru
- Jeevan Reji
- Henry Y. Jiang
- Nithil Balaji

Georgia Institute of Technology
