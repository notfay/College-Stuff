#include <stdio.h>
#include <string.h>

int main () {
    int num;
    int num2;
    FILE *file = fopen ("testdata.in", "r");

    fscanf(file, "%d", &num);

    char first[num][200];
    char second[num][200];

    for (int i = 0; i < num; i++) {
        fscanf(file, "%s#%[^\n]", first[i], second[i]);
    }

    fscanf(file, "%d", &num2);

    for (int i = 0; i < num2; i++) {
        char search[200];
        fscanf(file, "%s", search);

        int j;
        for (j = 0; j < num; j++) {
            if (strcmp(search, first[j]) == 0) {
                printf("Case #%d: %s\n", i+1, second[j]);
                break;
            }
        }

        if (j == num) {
            printf("Case #%d: N/A\n", i+1);
        }
    }

    fclose(file);
    return 0;
}