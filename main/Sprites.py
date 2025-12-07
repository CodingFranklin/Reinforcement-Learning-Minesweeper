import pygame
import random
from Settings import *

# type list:
# "." -> unknown
# "X" -> mine
# "C" -> clue
# "/" -> empty

class Tile:
    def __init__(self, x, y, image, type, revealed=False, flagged=False):
        self.x, self.y = x * TILESIZE, y * TILESIZE
        self.image = image
        self.type = type
        self.revealed = revealed
        self.flagged = flagged
        self.value = 0

    def __repr__(self):
        return self.type

    def draw(self, board_surface):
        if not self.flagged and self.revealed:
            board_surface.blit(self.image, (self.x, self.y))
        elif self.flagged and not self.revealed:
            board_surface.blit(tile_flag, (self.x, self.y))
        elif not self.revealed:
            board_surface.blit(tile_unknown, (self.x, self.y))


class Board:
    def __init__(self):
        self.board_surface = pygame.Surface((WIDTH, HEIGHT))
        # board list: a matrix contains all the Tiles which represented by their types
        self.board_list = [[Tile(col, row, tile_empty, ".") for row in range(ROWS)] for col in range(COLS)]
        self.dug = []
        self.mines_placed = False

    def place_mines(self, safe_x = None, safe_y = None):
        def in_safe_zone(x, y):
            if safe_x is None or safe_y is None:
                return False
            else:
                return abs(x - safe_x) <= 1 and abs(y - safe_y) <= 1

        placed = 0
        while(placed < AMOUNT_OF_MINES):
            x = random.randint(0, ROWS-1)
            y = random.randint(0, COLS-1)

            if self.board_list[x][y].type == "." and not in_safe_zone(x, y):
                self.board_list[x][y].image = tile_mine
                self.board_list[x][y].type = "X"
                placed += 1
        
        self.mines_placed = True

    def place_clues(self):
        for x in range(ROWS):
            for y in range(COLS):
                if self.board_list[x][y].type != "X":
                    clue_number = self.check_neighbors(x, y)
                    if clue_number > 0:
                        tile = self.board_list[x][y]
                        tile.image = tile_numbers[clue_number - 1]
                        tile.type = "C"
                        tile.value = clue_number

    



    # Check is (x, y) is valid on the board
    @staticmethod
    def is_inside(x, y):
        return 0 <= x < ROWS and 0 <= y < COLS
    
    def check_neighbors(self, x, y):
        total_mines = 0
        for x_offset in range(-1, 2):
            for y_offset in range(-1, 2):
                neighbor_x = x + x_offset
                neighbor_y = y + y_offset
                if self.is_inside(neighbor_x, neighbor_y) and self.board_list[neighbor_x][neighbor_y].type == "X":
                    total_mines += 1
        return total_mines

    def dig(self, x, y):
        if not self.mines_placed:
            self.place_mines(x, y)
            self.place_clues()

        self.dug.append((x, y))

        # Dig out a bomb
        if self.board_list[x][y].type == "X":
            self.board_list[x][y].image = tile_exploded
            self.board_list[x][y].revealed = True
            # Game over 
            return False
        # Dig out a clue
        elif self.board_list[x][y].type == "C":
            self.board_list[x][y].revealed = True
            return True
        
        self.board_list[x][y].revealed = True
        # Dig out empty and expand and reveal all the connecting empty tiles
        # Loop through all the tiles around the current tile
        for row in range(max(0, x-1), min(x+1, ROWS-1) + 1):
            for col in range(max(0, y-1), min(y+1, COLS-1) + 1):
                if (row, col) not in self.dug:
                    self.dig(row, col)
        return True



            
            





    def display_board(self):
        for row in self.board_list:
            print(row)

    def draw(self, screen):
        for row in self.board_list:
            for tile in row:
                tile.draw(self.board_surface)   
        screen.blit(self.board_surface, (0, 0))
    