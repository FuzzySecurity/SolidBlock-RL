const { BOARD_WIDTH, BOARD_HEIGHT, TICK_SPAWN_INTERVAL } = require('./constants.js');

// Define an exploration bonus constant.
const EXPLORATION_BONUS = 0.6;

/**
 * Creates a new Game instance.
 * @constructor
 * @param {boolean} [debugPatternDetection=false] - Enable debug logging for pattern detection.
 */
class Game {
  constructor(debugPatternDetection = false) {
    this.width = BOARD_WIDTH;
    this.height = BOARD_HEIGHT;
    this.tickCount = 0;
    this.score = 0;
    // Counters for items consumed
    this.foodEaten = 0;
    this.redBlocksEaten = 0;
    this.greenBlocksEaten = 0;
    
    // Initialize a cross-shaped obstacle at the board's center.
    const centerX = Math.floor((this.width - 1) / 2);
    const centerY = Math.floor((this.height - 1) / 2);
    let obstacles = [];
    for (let y = centerY - 4; y <= centerY + 4; y++) {
      for (let x = centerX - 1; x <= centerX + 1; x++) {
        obstacles.push({ x, y });
      }
    }
    for (let x = centerX - 4; x <= centerX + 4; x++) {
      for (let y = centerY - 1; y <= centerY + 1; y++) {
        obstacles.push({ x, y });
      }
    }
    this.obstacles = Array.from(new Map(obstacles.map(o => [`${o.x},${o.y}`, o])).values());

    // Initialize bonus block arrays BEFORE using getRandomFreePosition.
    this.redBlocks = [];
    this.greenBlocks = [];

    // Now it is safe to assign the player and food positions.
    this.player = this.getRandomFreePosition();
    this.food = this.getRandomFreePosition();

    this.moveHistory = []; // track last moves
    this.cyclePenaltyCount = 0;
    this.wallPenaltyCount = 0;
    this.debugPatternDetection = debugPatternDetection;

    // Track visited cells in the current episode to encourage exploration.
    this.visitedCells = new Set();
    // Mark the starting cell as visited.
    this.visitedCells.add(`${this.player.x},${this.player.y}`);
  }

  /**
   * Check if the given coordinates are within the board boundaries.
   * @param {number} x - The x-coordinate.
   * @param {number} y - The y-coordinate.
   * @returns {boolean} True if (x, y) is within bounds.
   */
  isWithinBounds(x, y) {
    return x >= 0 && x < this.width && y >= 0 && y < this.height;
  }

  /**
   * Check if the cell at (x, y) is occupied by the player, food, obstacles, or bonus blocks.
   * @param {number} x - The x-coordinate.
   * @param {number} y - The y-coordinate.
   * @returns {boolean} True if the cell is occupied.
   */
  isOccupied(x, y) {
    // Only check player if it has been initialized.
    if (this.player && this.player.x === x && this.player.y === y) return true;
    if (this.food && typeof this.food.x === 'number' && this.food.x === x && this.food.y === y) return true;
    if (this.obstacles.some(obs => obs.x === x && obs.y === y)) return true;
    if (this.redBlocks.some(rb => rb.x === x && rb.y === y)) return true;
    if (this.greenBlocks.some(gb => gb.x === x && gb.y === y)) return true;
    return false;
  }

  /**
   * Find a random free position on the board (not occupied).
   * @returns {{x: number, y: number}} An object with x and y coordinates.
   */
  getRandomFreePosition() {
    let x, y;
    let attempts = 0;
    do {
      x = Math.floor(Math.random() * this.width);
      y = Math.floor(Math.random() * this.height);
      attempts++;
      if (attempts > 1000) {
        throw new Error("Unable to find free position on board.");
      }
    } while (this.isOccupied(x, y));
    return { x, y };
  }

  /**
   * Move the player in the specified direction if the move is valid.
   * @param {string} direction - The direction ('up', 'down', 'left', or 'right').
   */
  movePlayer(direction) {
    let newX = this.player.x;
    let newY = this.player.y;
    switch (direction) {
      case 'up':
        newY -= 1;
        break;
      case 'down':
        newY += 1;
        break;
      case 'left':
        newX -= 1;
        break;
      case 'right':
        newX += 1;
        break;
      default:
        return; // Invalid direction.
    }
    if (!this.isWithinBounds(newX, newY)) return;
    if (this.obstacles.some(obs => obs.x === newX && obs.y === newY)) return;
    this.player.x = newX;
    this.player.y = newY;
  }

  /**
   * Check for collisions between the player and food or bonus blocks.
   * Updates the score and increments item counters accordingly.
   */
  checkCollisions() {
    if (this.player.x === this.food.x && this.player.y === this.food.y) {
      this.score += 40;
      this.foodEaten++; // Increment food counter
      this.food = this.getRandomFreePosition();
    }

    for (let i = 0; i < this.greenBlocks.length; i++) {
      let gb = this.greenBlocks[i];
      if (this.player.x === gb.x && this.player.y === gb.y) {
        this.score += 25;
        this.greenBlocksEaten++; // Increment green block counter
        this.greenBlocks.splice(i, 1);
        i--;
        if (this.redBlocks.length > 0) {
          const randIndex = Math.floor(Math.random() * this.redBlocks.length);
          this.redBlocks.splice(randIndex, 1);
        }
      }
    }

    for (let i = 0; i < this.redBlocks.length; i++) {
      let rb = this.redBlocks[i];
      if (this.player.x === rb.x && this.player.y === rb.y) {
        this.score -= 20;
        this.redBlocksEaten++; // Increment red block counter
        this.redBlocks.splice(i, 1);
        i--;
      }
    }
  }

  /**
   * Advance the game tick and spawn bonus blocks periodically.
   */
  tick() {
    this.tickCount++;
    if (this.tickCount % TICK_SPAWN_INTERVAL === 0) {
      this.spawnBonusPair();
    }
  }

  /**
   * Spawn a pair of bonus blocks (one red and one green) at random free positions.
   */
  spawnBonusPair() {
    const redPos = this.getRandomFreePosition();
    this.redBlocks.push(redPos);
    const greenPos = this.getRandomFreePosition();
    this.greenBlocks.push(greenPos);
  }

  /**
   * Reset the game for a new episode.
   * @returns {Object} The new game state.
   */
  reset() {
    this.tickCount = 0;
    this.score = 0;
    this.player = this.getRandomFreePosition();
    this.redBlocks = [];
    this.greenBlocks = [];
    this.food = this.getRandomFreePosition();
    this.moveHistory = [];
    // Reset visited cells for a new episode.
    this.visitedCells = new Set();
    this.visitedCells.add(`${this.player.x},${this.player.y}`);
    return this.getState();
  }

  /**
   * Process an action, updating the game state, rewards, and penalties.
   * @param {string} action - The action to take ('up', 'down', 'left', or 'right').
   * @returns {Object} Object containing the new state, reward, and a done flag.
   */
  step(action) {
    const oldX = this.player.x;
    const oldY = this.player.y;
    const previousScore = this.score;

    this.movePlayer(action);
    this.moveHistory.push(action);
    if (this.moveHistory.length > 20) {
      this.moveHistory.shift();
    }
    
    // If the player moves to a new cell in this episode, reward exploration.
    const posKey = `${this.player.x},${this.player.y}`;
    if (!this.visitedCells.has(posKey)) {
      this.visitedCells.add(posKey);
      this.score += EXPLORATION_BONUS;
    }

    // Encourage moving closer to food (manhattan distance)
    const d_old = Math.abs(oldX - this.food.x) + Math.abs(oldY - this.food.y);
    const d_new = Math.abs(this.player.x - this.food.x) + Math.abs(this.player.y - this.food.y);
    const shapingReward = 1.0 * (d_old - d_new);
    this.score += shapingReward;

    // Penalize moves that make insufficient progress toward food
    // If the decrease in distance is less than the threshold, subtract an extra penalty.
    const progressThreshold = 0.5;
    if ((d_old - d_new) < progressThreshold) {
      this.score -= 0.5;
    }

    // Apply penalties for repeated move patterns
    if (this.moveHistory.length >= 8) {
      const last8 = this.moveHistory.slice(-8);
      const firstHalf = last8.slice(0, 4).join(',');
      const secondHalf = last8.slice(4).join(',');
      if (firstHalf === secondHalf) {
        this.score -= 1.5;
        this.cyclePenaltyCount++;
        if (this.debugPatternDetection) {
          console.log("[DEBUG] 4-move cycle repeated. Penalty applied.");
        }
      }
    }

    if (this.moveHistory.length >= 10) {
      const last10 = this.moveHistory.slice(-10);
      const uniqueMoves = [...new Set(last10)];
      if (uniqueMoves.length === 1) {
        this.score -= 2;
        if (this.debugPatternDetection) {
          console.log("[DEBUG] Constant move detected. Penalty applied.");
        }
      } else if (uniqueMoves.length === 2) {
        let patternAlternates = true;
        for (let i = 0; i < last10.length - 1; i++) {
          if (last10[i] === last10[i + 1]) {
            patternAlternates = false;
            break;
          }
        }
        if (patternAlternates) {
          this.score -= 2;
          if (this.debugPatternDetection) {
            console.log("[DEBUG] Alternating 2-move pattern detected. Penalty applied.");
          }
        }
      } else {
        const candidate = last10[0];
        const countSame = last10.reduce((count, move) => count + (move === candidate ? 1 : 0), 0);
        if (countSame >= 8) {
          this.score -= 0.5;
          if (this.debugPatternDetection) {
            console.log("[DEBUG] 80% identical moves detected. Penalty applied.");
          }
        }
      }
    }

    if (this.debugPatternDetection) {
      console.log(`[DEBUG] moveHistory (last ${this.moveHistory.length}):`, this.moveHistory.join(','));
    }

    if (this.player.x === oldX && this.player.y === oldY) {
      this.score -= 3;
      this.wallPenaltyCount++;
    }

    this.checkCollisions();
    this.tick();
    const reward = this.score - previousScore;
    return { state: this.getState(), reward, done: false };
  }

  /**
   * Retrieve a snapshot of the current game state.
   * @returns {Object} The current game state including player, food, bonus blocks, obstacles, tick count, and score.
   */
  getState() {
    return {
      player: { ...this.player },
      food: this.food ? { ...this.food } : null,
      redBlocks: this.redBlocks.map(rb => ({ ...rb })),
      greenBlocks: this.greenBlocks.map(gb => ({ ...gb })),
      obstacles: this.obstacles,
      tickCount: this.tickCount,
      score: this.score
    };
  }

  /**
   * Log statistics for cycle and wall penalties.
   */
  logPenaltyStats() {
    console.log(`Cycle penalty triggered: ${this.cyclePenaltyCount} times, Wall penalty triggered: ${this.wallPenaltyCount} times`);
  }

  /**
   * Update the obstacle layout based on a given pattern without affecting training.
   * Options are:
   *   'normal' - Original cross obstacle.
   *   'diamond'- Diamond shape, 7 rows
   *   'none'   - No obstacles.
   * If an invalid pattern is given, defaults to 'normal'.
   * @param {string} pattern - The obstacle pattern.
   */
  setObstaclePattern(pattern) {
    const centerX = Math.floor((this.width - 1) / 2);
    const centerY = Math.floor((this.height - 1) / 2);
    let obstacles = [];
    switch (pattern) {
      case 'normal':
        // Original cross obstacle.
        for (let y = centerY - 4; y <= centerY + 4; y++) {
          for (let x = centerX - 1; x <= centerX + 1; x++) {
            obstacles.push({ x, y });
          }
        }
        for (let x = centerX - 4; x <= centerX + 4; x++) {
          for (let y = centerY - 1; y <= centerY + 1; y++) {
            obstacles.push({ x, y });
          }
        }
        break;
      case 'diamond':
        // Diamond shape scaled up (expanded by 2 layers) => 7 rows:
        // Top row (centerY - 3): one block at center.
        obstacles.push({ x: centerX, y: centerY - 3 });

        // Second row (centerY - 2): three blocks.
        for (let x = centerX - 1; x <= centerX + 1; x++) {
          obstacles.push({ x, y: centerY - 2 });
        }

        // Third row (centerY - 1): five blocks.
        for (let x = centerX - 2; x <= centerX + 2; x++) {
          obstacles.push({ x, y: centerY - 1 });
        }

        // Middle row (centerY): seven blocks.
        for (let x = centerX - 3; x <= centerX + 3; x++) {
          obstacles.push({ x, y: centerY });
        }

        // Fifth row (centerY + 1): five blocks.
        for (let x = centerX - 2; x <= centerX + 2; x++) {
          obstacles.push({ x, y: centerY + 1 });
        }

        // Sixth row (centerY + 2): three blocks.
        for (let x = centerX - 1; x <= centerX + 1; x++) {
          obstacles.push({ x, y: centerY + 2 });
        }

        // Bottom row (centerY + 3): one block at center.
        obstacles.push({ x: centerX, y: centerY + 3 });
        break;
      case 'none':
        // No obstacles.
        obstacles = [];
        break;
      default:
        // Fallback to normal.
        for (let y = centerY - 4; y <= centerY + 4; y++) {
          for (let x = centerX - 1; x <= centerX + 1; x++) {
            obstacles.push({ x, y });
          }
        }
        for (let x = centerX - 4; x <= centerX + 4; x++) {
          for (let y = centerY - 1; y <= centerY + 1; y++) {
            obstacles.push({ x, y });
          }
        }
        break;
    }
    this.obstacles = obstacles;
  }
}

module.exports = Game;
