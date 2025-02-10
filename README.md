# SolidBlock RL

Hey there! This is a little project I built to learn about Reinforcement Learning (RL) using a simple grid-based game. Everything is designed from scratch, using Node.js and TensorFlow.js to create and train an AI agent that learns to navigate, collect rewards, and avoid penalties.

I wrote a detailed blogpost about the project which has a lot of background and technical information:
- [Solid Block, adventures with Tensorflow and Reinforcement Learning (RL)](https://knifecoat.com/Posts/Solid+Block%2C+adventures+with+Tensorflow+and+Reinforcement+Learning+(RL))

## Why Did I Build This?

I wanted to understand RL beyond just the theory. Building this from scratch helped me grasp concepts like:
- Experience replay
- Exploration-exploitation trade-offs
- How to structure rewards to guide learning
- The importance of state representation
- Model architecture
- etc

## Features

🎮 **Play It Yourself**: Use arrow keys to move around
🤖 **Watch AI Play**: Load a trained model and watch it navigate the environment
🏋️ **Train Your Own**: Experiment with different architectures and training strategies
🎯 **Multiple Layouts**: Try different obstacle patterns to test adaptability

## Getting Started

Please review the code for full flag usage.

1. Clone and install dependencies:
   ```bash

   npm install
   ```

2. Try playing it yourself:
   ```bash
   node index.js --play
   ```

3. Watch a trained agent:
   ```bash
   node index.js --auto path/to/model.json
   ```

4. Train a new agent:
   ```bash
   node index.js --train
   ```

## Why code so bad??

I would be extremely interested in PR's that result in improved trainign scores. If you do make imporvements, please share your scores and an explenation of your approach.