import numpy as np

from Sprites import Board
from Settings import ROWS, COLS

# rewards
MINE_PENALTY = -10.0
WIN_BONUS = 10.0
FLAG_REWARD_CORRECT = 2.0
FLAG_REWARD_WRONG = -2.0
INVALID_ACTION_PENALTY = -1.0
PERFECT_FLAG_BONUS = 50.0

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

                if tile.flagged and not tile.revealed:
                    obs[x, y] = -2 # flagged
                elif not tile.revealed:
                    obs[x, y] = -1 # unknown
                else:
                    if tile.type == "/":
                        obs[x, y] = 0
                    elif tile.type == "C":
                        obs[x, y] = int(getattr(tile, "value", 0))
                    elif tile.type == "X":
                        obs[x, y] = 9 if self.done else -1
                    else:
                        obs[x, y] = 0

        return obs

    def _check_win(self):
        # same idea as Game.check_win() :contentReference[oaicite:2]{index=2}
        for x in range(ROWS):
            for y in range(COLS):
                tile = self.board.board_list[x][y]
                if tile.type != "X" and not tile.revealed:
                    return False
        return True

    def _all_mines_flagged(self):
        for x in range(ROWS):
            for y in range(COLS):
                tile = self.board.board_list[x][y]
                if tile.type == "X" and not tile.flagged:
                    return False
        return True

    def _count_revealed(self):
        count = 0
        for x in range(ROWS):
            for y in range(COLS):
                if self.board.board_list[x][y].revealed:
                    count += 1
        return count

    def step(self, action):
        """
        action: integer in [0, ROWS * COLS - 1]
        """
        if self.done:
            raise ValueError("Call reset() before step() on a finished episode.")

        N = ROWS * COLS
        is_flag_action = (action >= N)
        tile_index = action % N

        x = tile_index // COLS
        y = tile_index % COLS
        tile = self.board.board_list[x][y]

        if is_flag_action:
            if tile.revealed:
                reward = INVALID_ACTION_PENALTY
                return self._get_observation(), reward, self.done, {}

            if not tile.flagged:
                tile.flagged = True
                if tile.type == "X":
                    reward = FLAG_REWARD_CORRECT
                else:
                    reward = FLAG_REWARD_WRONG
            else:
                tile.flagged = False
                reward = 0.0

            obs = self._get_observation()
            return obs, float(reward), self.done, {}

        if tile.flagged:
            reward = INVALID_ACTION_PENALTY
            obs = self._get_observation()
            return obs, float(reward), self.done, {}

        if tile.revealed:
            reward = INVALID_ACTION_PENALTY
            obs = self._get_observation()
            return obs, float(reward), self.done, {}

        prev_revealed = self._count_revealed()

        alive = self.board.dig(x, y)

        new_revealed = self._count_revealed()
        newly_opened = max(0, new_revealed - prev_revealed)

        if not alive:

            reward = MINE_PENALTY
            self.done = True
        elif self._check_win():
            reward = float(newly_opened) + WIN_BONUS

            if self._all_mines_flagged():
                reward += PERFECT_FLAG_BONUS

            self.done = True
        else:
            reward = float(newly_opened)

        obs = self._get_observation()
        return obs, float(reward), self.done, {}

        reward = 0.0

        # If we re-click a revealed tile, small penalty to discourage it
        if tile.revealed:
            reward = -1.0
            return self._get_observation(), reward, self.done, {}

        # Calculate the block got digged before the step
        prev_revealed = 0
        for row in self.board.board_list:
            for t in row:
                if t.revealed:
                    prev_revealed += 1

        # Dig the tile (same logic Board.dig uses for human play) :contentReference[oaicite:3]{index=3}
        alive = self.board.dig(x, y)

        # Calculate the block got digged by this step
        new_revealed = 0
        for row in self.board.board_list:
            for t in row:
                if t.revealed:
                    new_revealed += 1

        # the amount this time digged and keep alive (click the safe block)
        newly_opened = max(0, new_revealed - prev_revealed)

        if not alive:
            # Hit a mine
            reward = -100.0
            self.done = True
        elif self._check_win():
            reward = float(newly_opened) + 1000.0
            self.done = True
            print("WIN!!!")
        else:
            # Survived; give small reward for revealing safe stuff
                # new: add reward based on the amout newly opened
            reward = float(newly_opened)

        obs = self._get_observation()
        return obs, reward, self.done, {}

    # Optional: if you want a quick textual render for debugging
    def render(self):
        obs = self._get_observation()
        print(obs)

