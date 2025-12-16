// Copyright (c) 2024 Ole-Christoffer Granmo

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef BOARD_DIM
    #define BOARD_DIM 11
#endif

int neighbors[] = {-(BOARD_DIM+2) + 1, -(BOARD_DIM+2), -1, 1, (BOARD_DIM+2), (BOARD_DIM+2) - 1};

struct hex_game {
	int board[(BOARD_DIM+2)*(BOARD_DIM+2)*2];
	int open_positions[BOARD_DIM*BOARD_DIM];
	int number_of_open_positions;
	int moves[BOARD_DIM*BOARD_DIM];
	int connected[(BOARD_DIM+2)*(BOARD_DIM+2)*2];
};

void hg_init(struct hex_game *hg)
{
	for (int i = 0; i < BOARD_DIM+2; ++i) {
		for (int j = 0; j < BOARD_DIM+2; ++j) {
			hg->board[(i*(BOARD_DIM + 2) + j) * 2] = 0;
			hg->board[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 0;

			if (i > 0 && i < BOARD_DIM + 1 && j > 0 && j < BOARD_DIM + 1) {
				hg->open_positions[(i-1)*BOARD_DIM + j - 1] = i*(BOARD_DIM + 2) + j;
			}

			if (i == 0) {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2] = 1;
			} else {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2] = 0;
			}
			
			if (j == 0) {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 1;
			} else {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 0;
			}
		}
	}
	hg->number_of_open_positions = BOARD_DIM*BOARD_DIM;
}

int hg_connect(struct hex_game *hg, int player, int position) 
{
	hg->connected[position*2 + player] = 1;

	if (player == 0 && position / (BOARD_DIM + 2) == BOARD_DIM) {
		return 1;
	}

	if (player == 1 && position % (BOARD_DIM + 2) == BOARD_DIM) {
		return 1;
	}

	for (int i = 0; i < 6; ++i) {
		int neighbor = position + neighbors[i];
		if (hg->board[neighbor*2 + player] && !hg->connected[neighbor*2 + player]) {
			if (hg_connect(hg, player, neighbor)) {
				return 1;
			}
		}
	}
	return 0;
}

int hg_winner(struct hex_game *hg, int player, int position)
{
	for (int i = 0; i < 6; ++i) {
		int neighbor = position + neighbors[i];
		if (hg->connected[neighbor*2 + player]) {
			return hg_connect(hg, player, position);
		}
	}
	return 0;
}

int hg_place_piece_randomly(struct hex_game *hg, int player)
{
	int random_empty_position_index = rand() % hg->number_of_open_positions;

	int empty_position = hg->open_positions[random_empty_position_index];

	hg->board[empty_position * 2 + player] = 1;

	hg->moves[BOARD_DIM*BOARD_DIM - hg->number_of_open_positions] = empty_position;

	hg->open_positions[random_empty_position_index] = hg->open_positions[hg->number_of_open_positions-1];

	hg->number_of_open_positions--;

	return empty_position;
}

void hg_place_piece_based_on_tm_input(struct hex_game *hg, int player)
{
	printf("TM!\n");
}

int hg_full_board(struct hex_game *hg)
{
	return hg->number_of_open_positions == 0;
}

void hg_print(struct hex_game *hg)
{
	for (int i = 0; i < BOARD_DIM; ++i) {
		for (int j = 0; j < i; j++) {
			printf(" ");
		}

		for (int j = 0; j < BOARD_DIM; ++j) {
			if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2] == 1) {
				printf(" X");
			} else if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2 + 1] == 1) {
				printf(" O");
			} else {
				printf(" Â·");
			}
		}
		printf("\n");
	}
}

int main(int argc, char *argv[]) {
	int games = 1;
	int seed = -1;
	int verbose = 0;
	char *dump_moves_path = NULL;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--games") == 0 && i + 1 < argc) {
			games = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
			seed = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--verbose") == 0) {
			verbose = 1;
		} else if (strcmp(argv[i], "--dump-moves") == 0 && i + 1 < argc) {
			dump_moves_path = argv[++i];
		}
	}

	if (seed == -1) {
		seed = time(NULL);
	}

	FILE *csv_file = NULL;
	if (dump_moves_path) {
		csv_file = fopen(dump_moves_path, "w");
		if (!csv_file) {
			fprintf(stderr, "Error: Could not open %s for writing\n", dump_moves_path);
			return 1;
		}
		fprintf(csv_file, "game_id,move_idx,player,row,col,position,winner\n");
	}

	struct hex_game hg;

	for (int game = 0; game < games; ++game) {
		srand(seed + game);
		hg_init(&hg);

		int player = 0;
		int winner = -1;

		while (!hg_full_board(&hg)) {
			int position = hg_place_piece_randomly(&hg, player);
			
			if (hg_winner(&hg, player, position)) {
				winner = player;
				break;
			}

			player = 1 - player;
		}

		if (verbose) {
			printf("\nGame %d: Player %d wins!\n\n", game, winner);
			hg_print(&hg);
		}

		if (csv_file) {
			int total_moves = BOARD_DIM * BOARD_DIM - hg.number_of_open_positions;
			int current_player = 0;
			for (int i = 0; i < total_moves; ++i) {
				int pos = hg.moves[i];
				int row = pos / (BOARD_DIM + 2) - 1;
				int col = pos % (BOARD_DIM + 2) - 1;
				fprintf(csv_file, "%d,%d,%d,%d,%d,%d,%d\n", 
						game, i, current_player, row, col, pos, winner);
				current_player = 1 - current_player;
			}
		}
	}

	if (csv_file) {
		fclose(csv_file);
	}

	return 0;
}