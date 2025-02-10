const chalk = require('chalk').default;

class Renderer {
  /**
   * Render the current game state onto the console.
   * Clears the console and prints the board with the positions of the player,
   * food, bonus blocks, obstacles, and other elements.
   * @param {Object} game - The game instance containing the current state.
   */
  render(game) {
    // Clear the console for a clean redraw.
    console.clear();

    const width = game.width;
    const height = game.height;
    let output = '';

    // Loop through every row (including border rows).
    for (let y = -1; y <= height; y++) {
      let row = '';
      for (let x = -1; x <= width; x++) {
        // If we are on the border, always draw a white block.
        if (x === -1 || x === width || y === -1 || y === height) {
          row += chalk.bgWhite('  ');
        } else {
          // Check if the cell contains the player.
          if (game.player.x === x && game.player.y === y) {
            row += chalk.bgYellow("  ");
          }
          // Check if the cell contains the food (target block) — add a guard.
          else if (game.food && typeof game.food.x === 'number' && game.food.x === x && game.food.y === y) {
            row += chalk.bgWhite("  ");
          }
          // Check for a green bonus block.
          else if (game.greenBlocks.some(gb => gb.x === x && gb.y === y)) {
            row += chalk.bgGreen("  ");
          }
          // Check for a red bonus block.
          else if (game.redBlocks.some(rb => rb.x === x && rb.y === y)) {
            row += chalk.bgRed("  ");
          }
          // Check for a cross obstacle.
          else if (game.obstacles.some(obs => obs.x === x && obs.y === y)) {
            row += chalk.bgWhite("  ");
          }
          // Otherwise render an empty cell.
          else {
            row += chalk.bgBlack("  ");
          }
        }
      }
      output += row + '\n';
    }
    console.log(output);
  }
}

module.exports = Renderer; 
