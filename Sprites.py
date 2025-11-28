import pygame
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

    def __repr__(self):
        return self.type

    def draw(self, board_surface):
        board_surface.blit(tile_unknown, (self.x, self.y))

class Board:
    def __init__(self):
        self.board_surface = pygame.Surface((WIDTH, HEIGHT))
        self.board_list = [[Tile(col, row, tile_empty, ".") for row in range(ROWS)] for col in range(COLS)]

    def display_board(self):
        for row in self.board_list:
            print(row)

    def draw(self, screen):
        for row in self.board_list:
            for tile in row:
                tile.draw(self.board_surface)   
        screen.blit(self.board_surface, (0, 0))
    