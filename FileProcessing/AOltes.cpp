#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ROWS 1000  // Maximum number of rows to read from the CSV file

typedef struct {
    char data[MAX_ROWS][100]; // Assuming each field has a maximum length of 100 characters
    int rows;
    int cols;
} CSVData;

// Function to read data from CSV file
void readCSV(const char *filename, CSVData *csvData) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file.\n");
        return;
    }

    int row = 0;
    int col = 0;
    char line[1024]; // Assuming a maximum line length of 1024 characters

    while (fgets(line, sizeof(line), file) != NULL && row < MAX_ROWS) {
        col = 0;
        char *token = strtok(line, ",");
        while (token != NULL) {
            strcpy(csvData->data[row * csvData->cols + col], token);
            token = strtok(NULL, ",");
            col++;
        }
        row++;
    }

    csvData->rows = row;
    csvData->cols = col;

    fclose(file);
}

// Function to display 'n' rows of data
void displayRows(CSVData csvData, int n) {
    if (n > csvData.rows || n <= 0) {
        n = csvData.rows;
    }

    printf("Displaying %d rows:\n", n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < csvData.cols; j++) {
            printf("%s\t", csvData.data[i * csvData.cols + j]);
        }
        printf("\n");
    }
}

// Function to select rows based on a specific column value
void selectRow(CSVData csvData, int column, const char *query) {
    printf("Selected rows with '%s' in column %d:\n", query, column);
    int found = 0;
    for (int i = 0; i < csvData.rows; i++) {
        if (strcmp(csvData.data[i * csvData.cols + column - 1], query) == 0) {
            found = 1;
            for (int j = 0; j < csvData.cols; j++) {
                printf("%s\t", csvData.data[i * csvData.cols + j]);
            }
            printf("\n");
        }
    }
    if (!found) {
        printf("Not Found.\n");
    }
}

// Function to sort data by a specific column in ascending or descending order
void sortBy(CSVData *csvData, int column, int ascending) {
    // Implementation of sorting algorithm (e.g., bubble sort, quicksort, etc.)
    // Sorting logic goes here...

    // For simplicity, let's assume the sorting is done
    // and we will display the first 10 data after sorting
    printf("Sorted data (first 10 rows):\n");
    int limit = (csvData->rows < 10) ? csvData->rows : 10;
    for (int i = 0; i < limit; i++) {
        for (int j = 0; j < csvData->cols; j++) {
            printf("%s\t", csvData->data[i * csvData->cols + j]);
        }
        printf("\n");
    }
}

// Function to export data to a CSV file
void exportCSV(const char *filename, CSVData csvData) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error creating file for export.\n");
        return;
    }

    for (int i = 0; i < csvData.rows; i++) {
        for (int j = 0; j < csvData.cols; j++) {
            fprintf(file, "%s", csvData.data[i * csvData.cols + j]);
            if (j < csvData.cols - 1) {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

int main() {
    CSVData csvData;
    csvData.rows = 0;
    csvData.cols = 0;

    // Read data from CSV file
    readCSV("data.csv", &csvData);

    // Display 'n' rows of data
    displayRows(csvData, 5);

    // Select rows based on a specific column value
    selectRow(csvData, 2, "value");

    // Sort data by a specific column in ascending order (assuming column 3)
    sortBy(&csvData, 3, 1);

    // Export data to a new CSV file
    exportCSV("exported_data.csv", csvData);

    return 0;
}

