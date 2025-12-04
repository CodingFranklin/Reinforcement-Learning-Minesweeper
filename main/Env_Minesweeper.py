import numpy as np

from Sprites import Board
from Settings import ROWS, COLS

class MinesweeperEnv:
    def __init__(self):
        self.board = None
        self.done = False

    def reset(self):
        """Start a new episode (new random board)."""
        self.board = Board()
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        """
        Encode the visible board into a 2D numpy array:
        - -2 = flagged
        - -1 = unknown
        -  0 = revealed empty (no neighboring mines)
        -  1..8 = revealed clue with that number
        """
        obs = np.zeros((ROWS, COLS), dtype=np.int8)
        for x in range(ROWS):
            for y in range(COLS):
                tile = self.board.board_list[x][y]
                if tile.flagged:
                    obs[x, y] = -2
                elif not tile.revealed:
                    obs[x, y] = -1
                elif tile.type == "/":   # empty
                    obs[x, y] = 0
                elif tile.type == "C":
                    obs[x, y] = tile.value
                elif tile.type == "X":
                    # Usually you don't see mines unless game is over;
                    # only reveal them in observation when done.
                    obs[x, y] = 9
        return obs

    def _check_win(self):
        # same idea as Game.check_win() :contentReference[oaicite:2]{index=2}
        for row in self.board.board_list:
            for tile in row:
                if tile.type != "X" and not tile.revealed:
                    return False
        return True

    def step(self, action):
        """
        action: integer in [0, ROWS * COLS - 1]
        """
        if self.done:
            raise ValueError("Call reset() before step() on a finished episode.")

        x = action // COLS
        y = action % COLS
        tile = self.board.board_list[x][y]

        reward = 0.0

        # If we re-click a revealed tile, small penalty to discourage it
        if tile.revealed:
            reward = -0.1
            return self._get_observation(), reward, self.done, {}

        # Dig the tile (same logic Board.dig uses for human play) :contentReference[oaicite:3]{index=3}
        alive = self.board.dig(x, y)

        if not alive:
            # Hit a mine
            reward = -10.0
            self.done = True
        elif self._check_win():
            reward = 10.0
            self.done = True
        else:
            # Survived; give small reward for revealing safe stuff
            reward = 1.0

        obs = self._get_observation()
        return obs, reward, self.done, {}

    # Optional: if you want a quick textual render for debugging
    def render(self):
        obs = self._get_observation()
        print(obs)