#!/usr/bin/env node
/**
 * Main entry point for the game application.
 * 
 * Modes:
 *  --play  : Human play mode.
 *  --auto  : Auto-play mode (using a trained model).
 *  --train : Train the model with checkpointing/restoration.
 *
 * Optional:
 *  --obstacle [normal|diamond|none] : Sets the obstacle layout in play/auto mode.
 */
// Import modules
const Game = require('./game/game.js');
const Renderer = require('./game/renderer.js');

const args = process.argv.slice(2);
if (!args.includes('--play') && !args.includes('--auto') && !args.includes('--train')) {
  console.log("Usage: node index.js --play | --auto | --train");
  process.exit(0);
}

const game = new Game();
const renderer = new Renderer();

// Update obstacle pattern for play/auto mode.
if (!args.includes('--train')) {
  let obstacleType = "normal"; // default
  const obstacleIndex = args.indexOf("--obstacle");
  if (obstacleIndex !== -1 && obstacleIndex + 1 < args.length) {
    obstacleType = args[obstacleIndex + 1];
  }
  // Call setObstaclePattern to update obstacles.
  if (typeof game.setObstaclePattern === 'function') {
    game.setObstaclePattern(obstacleType);
    console.log("Obstacle pattern set to:", obstacleType);
  }
}

if (args.includes('--train')) {
  console.log("Starting training mode.");
  const trainIndex = args.indexOf('--train');
  let resumeModelPath = null;
  // If provided, assume model checkpoint exists.
  if (trainIndex + 1 < args.length && !args[trainIndex + 1].startsWith('--')) {
    resumeModelPath = args[trainIndex + 1];
    console.log("Resuming training from model checkpoint at:", resumeModelPath);
  }
  const { train } = require('./ai/train');
  train(resumeModelPath)
    .then(() => {
      console.log("Training complete.");
    })
    .catch(err => {
      console.error("Training error:", err);
    });
} else if (args.includes('--play')) {
  renderer.render(game);
  console.log("Use arrow keys to move. Press Ctrl+C to exit.");
  process.stdin.setRawMode(true);
  process.stdin.resume();
  process.stdin.setEncoding('utf8');
  process.stdin.on('data', (key) => {
    if (key === '\u0003') {
      console.log("Exiting...");
      process.exit();
    }
    let direction = null;
    if (key === '\u001b[A') {
      direction = 'up';
    } else if (key === '\u001b[B') {
      direction = 'down';
    } else if (key === '\u001b[C') {
      direction = 'right';
    } else if (key === '\u001b[D') {
      direction = 'left';
    }
    if (direction) {
      game.movePlayer(direction);
      game.checkCollisions();
      game.tick();
      renderer.render(game);
      console.log("Score:", game.score);
    }
  });
} else if (args.includes('--auto')) {
  const autoIndex = args.indexOf('--auto');
  const modelPathArg = args[autoIndex + 1];
  if (modelPathArg) {
    const tf = require('@tensorflow/tfjs-node');
    const path = require('path');
    const modelFullPath = path.resolve(modelPathArg);
    const modelURL = "file://" + modelFullPath;
    console.log("Loading model from", modelURL);
    tf.loadLayersModel(modelURL).then((loadedModel) => {
      console.log("Model loaded successfully.");
      game.reset();
      // Updated getFeatureVector for 9x9 grid.
      function getFeatureVector(state, game) {
        const player = state.player;
        const food = state.food;
        // Use the same offset scaling (5×) as before.
        const dx = 5 * (food.x - player.x) / game.width;
        const dy = 5 * (food.y - player.y) / game.height;
        // Updated grid size for the new model architecture: 9x9.
        const gridSize = 9;
        const numClasses = 6;
        const half = Math.floor(gridSize / 2); // half = 4 (since 9/2 = 4.5, floor=4)
        let grid = [];
        for (let i = -half; i <= half; i++) {
          let row = [];
          for (let j = -half; j <= half; j++) {
            const cellX = player.x + j;
            const cellY = player.y + i;
            let cat = 5; // default category for an empty cell.
            if (cellX < 0 || cellX >= game.width || cellY < 0 || cellY >= game.height) {
              cat = 0;
            } else if (game.obstacles.some(o => o.x === cellX && o.y === cellY)) {
              cat = 0;
            } else if (game.redBlocks.some(o => o.x === cellX && o.y === cellY)) {
              cat = 1;
            } else if (game.greenBlocks.some(o => o.x === cellX && o.y === cellY)) {
              cat = 2;
            } else if (game.food && game.food.x === cellX && game.food.y === cellY) {
              cat = 3;
            } else if (game.player.x === cellX && game.player.y === cellY) {
              cat = 4;
            }
            let oneHot = new Array(numClasses).fill(0);
            oneHot[cat] = 1;
            row.push(oneHot);
          }
          grid.push(row);
        }
        return { grid: grid, offset: [dx, dy] };
      }

      // Add a step counter and table header
      let stepCount = 0;
      console.log("\x1b[1m" + "Step".padEnd(6) + "Action".padEnd(10) + "Score".padEnd(20) + "Reward" + "\x1b[0m");
      
      // Parse an optional epsilon parameter for auto-play mode
      let epsilonAuto = 0.15;
      const epsilonFlag = '--epsilon';
      const epsilonIndex = args.indexOf(epsilonFlag);
      if (epsilonIndex !== -1 && epsilonIndex + 1 < args.length) {
        epsilonAuto = parseFloat(args[epsilonIndex + 1]);
      }
      console.log("Auto-play epsilon set to:", epsilonAuto);
      
      setInterval(() => {
        stepCount++;
        const currentState = game.getState();
        const featureVector = getFeatureVector(currentState, game);
        const gridTensor = tf.tensor4d([featureVector.grid], [1, 9, 9, 6]);
        const offsetTensor = tf.tensor2d([featureVector.offset], [1, 2]);
        const qsTensor = loadedModel.predict([gridTensor, offsetTensor]);
        const qs = qsTensor.dataSync();
        qsTensor.dispose();
        gridTensor.dispose();
        offsetTensor.dispose();
        const actions = ['up', 'down', 'left', 'right'];
        let action;
        if (Math.random() < epsilonAuto) {
          // With probability epsilonAuto, choose a random action excluding the one with the highest value.
          const bestActionIndex = qs.indexOf(Math.max(...qs));
          const otherActions = actions.filter((_, index) => index !== bestActionIndex);
          action = otherActions[Math.floor(Math.random() * otherActions.length)];
        } else {
          // Otherwise, take the best action from the model.
          const bestActionIndex = qs.indexOf(Math.max(...qs));
          action = actions[bestActionIndex];
        }
        const result = game.step(action);
        renderer.render(game);
        const formattedScore = game.score.toFixed(2);
        const formattedReward = result.reward.toFixed(2);
        const row = 
          "Step: " + "\x1b[36m" + stepCount.toString().padEnd(5) + "\x1b[0m" +
          "Action: " + "\x1b[32m" + action.padEnd(7) + "\x1b[0m" +
          "Score: " + "\x1b[33m" + formattedScore.toString().padEnd(9) + "\x1b[0m" +
          "Reward: " + "\x1b[35m" + formattedReward.padEnd(7) + "\x1b[0m";
        console.log(row);
        const countersRow = "Food Blocks: " + (game.foodEaten || 0).toString().padEnd(5) +
                            "Red Blocks: " + "\x1b[31m" + (game.redBlocksEaten || 0).toString().padEnd(5) + "\x1b[0m" +
                            "Green Blocks: " + "\x1b[32m" + (game.greenBlocksEaten || 0).toString().padEnd(5) + "\x1b[0m";
        console.log(countersRow);
      }, 200);
    }).catch(err => {
      console.error("Error loading model:", err);
      process.exit(1);
    });
  } else {
    console.log("No model path provided; using random actions for auto-play mode.");
    game.reset();
    const actions = ['up', 'down', 'left', 'right'];
    setInterval(() => {
      const action = actions[Math.floor(Math.random() * actions.length)];
      const result = game.step(action);
      renderer.render(game);
      console.log("Auto-play: Action:", action, "Score:", game.score, "Reward:", result.reward);
      
      // Log item counters.
      const countersRow = "Food Blocks: " + (game.foodEaten || 0).toString().padEnd(5) +
                          "Red Blocks: " + "\x1b[31m" + (game.redBlocksEaten || 0).toString().padEnd(5) + "\x1b[0m" +
                          "Green Blocks: " + "\x1b[32m" + (game.greenBlocksEaten || 0).toString().padEnd(5) + "\x1b[0m";
      console.log(countersRow);
    }, 200);
  }
}
