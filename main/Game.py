import pygame
from Settings import *
from Sprites import *

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()

    def reset(self):
        self.board = Board()
        self.board.display_board()

    def run(self):
        self.playing = True
        while self.playing:
            self.clock.tick(FPS)
            self.events()
            self.draw()
        else:
            self.end_screen()

    def check_win(self):
        for row in self.board.board_list:
            for tile in row:
                if tile.type != "X" and not tile.revealed:
                    return False
        return True

    def draw(self):
        self.screen.fill(BGCOLOR)
        self.board.draw(self.screen)
        pygame.display.flip()

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit(0)

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                mx //= TILESIZE
                my //= TILESIZE
                target_tile = self.board.board_list[mx][my]

                # if left mouse button down
                if event.button == 1:
                    if not target_tile.flagged:
                        if not self.board.dig(mx, my):
                            # Game Over
                            for row in self.board.board_list:
                                for tile in row:
                                    if tile.flagged and tile.type != "X":
                                        tile.flagged = False
                                        tile.image = tile_not_mine
                                        tile.revealed = True
                                    elif tile.type == "X":
                                        tile.revealed = True
                            self.playing = False

                # if right mouse button down
                if event.button == 3:
                    if not target_tile.revealed:
                        target_tile.flagged = not target_tile.flagged
                
                if self.check_win():
                    self.win = True
                    self.playing = False
                    for row in self.board.board_list:
                        for tile in row:
                            if not tile.revealed:
                                tile.flagged = True

    def end_screen(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit(0)

                if event.type == pygame.MOUSEBUTTONDOWN:
                    return





game = Game()
while True:
    game.reset()
    game.run()