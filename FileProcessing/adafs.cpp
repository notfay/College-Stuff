#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>

struct Data {
    char location1[100];
    char location2[100];
    int price;
    int rooms;
    int bathrooms;
    int carParks;
    char type[100];
    char furnish[100];
};

int readData(struct Data data[]) {
    FILE *file;
    file = fopen("file.csv", "r");
    if (file == NULL) {
        printf("Error opening the file.\n");
        return -1;
    }

    char buffer[1024];
    int i = 0;

    // Ignore the first line
    fgets(buffer, sizeof(buffer), file);

    // Read data from the file and store it in the structure array, up to 3942 entries
    while (fgets(buffer, sizeof(buffer), file) && i < 3942) {
        sscanf(buffer, "%[^,],%[^,],%d,%d,%d,%d,%[^,],%s",
               data[i].location1, data[i].location2, &data[i].price,
               &data[i].rooms, &data[i].bathrooms, &data[i].carParks,
               data[i].type, data[i].furnish);

        // Code for removing double quotes if present in location fields
        size_t len1 = strlen(data[i].location1);
        size_t len2 = strlen(data[i].location2);
        for (size_t j = 0; j < len1; j++) {
            if (data[i].location1[j] == '"') {
                memmove(&data[i].location1[j], &data[i].location1[j + 1], len1 - j);
                len1--;
                j--;
            }
        }
        for (size_t j = 0; j < len2; j++) {
            if (data[i].location2[j] == '"') {
                memmove(&data[i].location2[j], &data[i].location2[j + 1], len2 - j);
                len2--;
                j--;
            }
        }

        i++;
    }

    fclose(file);
    return i; // Return the number of entries read from the file
}

int main() {
    struct Data data[3942]; // Updated struct size and data array size
    int dataSize = readData(data);

    if (dataSize == -1) {
        return 1;
    }

    // Use the 'data' array with 'dataSize' elements read from the file
    // ... (rest of your code)
    
    return 0;
}

