#include <stdio.h>
#include <ctype.h>

int main() {
    int t;
    scanf("%d", &t);
    getchar(); // Consume the newline character after reading t

    for (int i = 1; i <= t; i++) {
        char input[1000];

        scanf(" %[^\n]", input);

        printf("Case #%d: ", i);

        char prev_char = ' ';

        for (int j = 0; input[j] != '\0'; j++) {
            char current_char = input[j];

            if (isalpha(current_char) && (prev_char == ' ' || prev_char == '0')) {
                putchar(toupper(current_char));
            }

            prev_char = current_char;
        }

        printf("\n");
    }

    return 0;
}

