#include <stdio.h>
#include <stdbool.h>

#define BOARD_SIZE 8

// Function to check if the move is within the board
bool isValidMove(int x, int y) {
    return x >= 1 && x <= BOARD_SIZE && y >= 1 && y <= BOARD_SIZE;
}

// Recursive function to check if the knights can meet
bool canKnightsMeet(int x1, int y1, int x2, int y2, int steps) {
    if (steps == 0) {
        return x1 == x2 && y1 == y2;
    }

    int moves[][2] = {
        {1, 2}, {1, -2}, {-1, 2}, {-1, -2},
        {2, 1}, {2, -1}, {-2, 1}, {-2, -1}
    };

    for (int i = 0; i < 8; ++i) {
        int new_x = x1 + moves[i][0];
        int new_y = y1 + moves[i][1];

        if (isValidMove(new_x, new_y) && canKnightsMeet(new_x, new_y, x2, y2, steps - 1)) {
            return true;
        }
    }

    return false;
}

int main() {
    int T;
    scanf("%d", &T);

    for (int i = 1; i <= T; ++i) {
        int steps, x1, y1, x2, y2;
        char coord1[3], coord2[3];

        scanf("%d %s %s", &steps, coord1, coord2);

        x1 = coord1[0] - 'A' + 1;
        y1 = coord1[1] - '0';
        x2 = coord2[0] - 'A' + 1;
        y2 = coord2[1] - '0';

        bool result = canKnightsMeet(x1, y1, x2, y2, steps);
        
        printf("Case #%d: %s\n", i, (result ? "YES" : "NO"));
    }

    return 0;
}

