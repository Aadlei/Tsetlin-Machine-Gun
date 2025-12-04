BOARD_DIM ?= 11
CFLAGS ?= -O3 -Wall -Wextra
hex: hex.c
	$(CC) $(CFLAGS) -D BOARD_DIM=$(BOARD_DIM) -o $@ $<
clean:
	rm -f hex
