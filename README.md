# BoxJumpEvolve

A simple neuroevolution experiment where neural networks learn to navigate a side-scrolling obstacle course through genetic algorithms.

## Overview

This project demonstrates how artificial agents can learn complex behaviors without explicit programming, using only evolution and natural selection. Small neural networks control jumping agents that must navigate gaps, walls, and platforms to reach a goal. Through successive generations, the population evolves better strategies to complete the course.

## Motivation

Built as a fun learning experiment to understand:
- How neural networks can evolve without backpropagation
- Genetic algorithms and natural selection in action
- Emergent behavior from simple rules
- Neuroevolution as an alternative to traditional ML training methods

## How It Works

### The Agents

Each agent is controlled by a small neural network:
- **Inputs (7):** Distance to next gap, gap width, distance to next wall, wall height, distance to next platform, platform height, vertical velocity
- **Hidden layer:** 6 neurons with sigmoid activation
- **Output (1):** Jump probability (jumps if > 0.5)

### The Evolution Process

1. **Generation 0:** 21 agents with random neural network weights
2. **Simulation:** Agents attempt to navigate the obstacle course
3. **Fitness:** Based on distance traveled + huge bonus for reaching the goal
4. **Selection:** Top 20% of performers survive
5. **Mutation:** Survivors are cloned with random weight mutations (adaptive mutation strength based on performance)
6. **Repeat:** New generation runs with evolved brains

### Obstacles

- **Gaps:** Fall through and die
- **Walls:** Solid grey blocks that stop horizontal movement (agents must jump over)
- **Platforms:** Surfaces agents can land on above ground level
- **Timeout:** Agents have limited time to reach the goal (base time + 10% buffer)

## Installation

```bash
pip install pygame numpy
```

## Usage

```bash
python neuroevo_game.py
```

The simulation runs for 24 generations, displaying:
- Current generation number
- Number of agents still alive
- Real-time visualization of all agents attempting the course

After all generations complete, the best-performing agent does a victory lap.

## Features

- **Collision Detection:** Walls have proper hitboxes - agents get stuck until they jump/fall clear
- **Adaptive Mutation:** Mutation strength decreases as agents get closer to the goal
- **Goal Incentive:** +1000 fitness bonus for completing the course (prevents "hiding" strategies)
- **Timeout System:** Prevents infinite wall-sitting; forces forward progress
- **Visual Feedback:** Color-coded agents, generation counter, population tracking

## Parameters You Can Tweak

In `neuroevo_game.py`:
- `NUM_GENERATIONS`: How many generations to run (default: 24)
- `POP_SIZE`: Number of agents per generation (default: 21)
- `gaps`: Location and width of deadly gaps
- `walls`: Position and width of obstacles
- `platforms`: Elevated surfaces agents can use
- Evolution parameters in `evolve()`: retention rate, mutation chance, mutation strength

## Future Ideas

- Add backward movement capability for puzzle-solving
- Multiple obstacle types (moving platforms, spikes, etc.)
- Save/load best networks
- Variable difficulty levels
- Speed controls for simulation
- Real-time graph of fitness progression
- Multi-objective fitness (time + distance)

## Technical Details

- **Language:** Python 3
- **Dependencies:** pygame, numpy
- **Neural Network:** Fully connected feedforward (7→6→1)
- **Activation:** Sigmoid with gradient clipping
- **Evolution Strategy:** (μ+λ) with fitness-proportionate mutation

## License

Built for fun and learning. Do whatever you want with it!
