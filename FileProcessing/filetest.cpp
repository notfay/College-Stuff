#include<stdio.h>
#include<stdlib.h>
#include<string.h>

struct Book {
    int bookId;
    char BTitle[100];
    char BAuthor[100];
    int yearPublished;
};

void inputBook(struct Book *book) {
    printf("Please Enter Book Id: ");
    scanf("%d", &book->bookId);

    printf("Please Enter Book Title: ");
    scanf(" %[^\n]", book->BTitle);

    printf("Please Enter Author Name: ");
    scanf(" %[^\n]", book->BAuthor);

    printf("Please Enter Year Published: ");
    scanf("%d", &book->yearPublished);
}

void displayBooks(struct Book *books, int count) {
    for (int i = 0; i < count - 1; i++) {
        for (int j = 0; j < count - i - 1; j++) {
            if (books[j].bookId > books[j + 1].bookId) {
                struct Book temp = books[j];
                books[j] = books[j + 1];
                books[j + 1] = temp;
            }
        }
    }

    printf("\nBook Records (Sorted by ID):\n");
    printf("%-10s %-30s %-20s %-10s\n", "Book ID", "Title", "Author", "Year Published");
    for (int i = 0; i < count; i++) {
        printf("%-10d %-30s %-20s %-10d\n", books[i].bookId, books[i].BTitle, books[i].BAuthor, books[i].yearPublished);
    }
}

void updateBook(struct Book *books, int count) {
    int id;
    printf("Enter the Book ID to update: ");
    scanf("%d", &id);
    int found = 0;
    for (int i = 0; i < count; i++) {
        if (books[i].bookId == id) {
            found = 1;

            printf("Enter updated Book Title: ");
            scanf(" %[^\n]", books[i].BTitle);

            printf("Enter updated Author Name: ");
            scanf(" %[^\n]", books[i].BAuthor);

            printf("Enter updated Year Published: ");
            scanf("%d", &books[i].yearPublished);

            printf("Book information updated successfully!\n");
            break;
        }
    }

    if (!found) {
        printf("Book with ID %d not found.\n", id);
    }
}

void eraseBook(struct Book *books, int *count) {
    int id;
    printf("Enter the Book ID to erase: ");
    scanf("%d", &id);

    int found = 0;
    for (int i = 0; i < *count; i++) {
        if (books[i].bookId == id) {
            found = 1;

            for (int j = i; j < *count - 1; j++) {
                books[j] = books[j + 1];
            }

            (*count)--;
            printf("Book with ID %d erased successfully!\n", id);
            break;
        }
    }

    if (!found) {
        printf("Book with ID %d not found.\n", id);
    }
}

void saveAndExit(struct Book *books, int count) {
    FILE *file = fopen("library.txt", "w");
    if (file == NULL) {
        printf("Error opening file for writing.\n");
        exit(1);
    }

    for (int i = 0; i < count; i++) {
        fprintf(file, "%d,%s,%s,%d\n", books[i].bookId, books[i].BTitle, books[i].BAuthor, books[i].yearPublished);
    }

    fclose(file);
    printf("Library information saved to 'library.txt'. Exiting program.\n");
    exit(0);
}

void readFromFile(struct Book *books, int *count) {
    FILE *file = fopen("library.txt", "r");
    if (file == NULL) {
        printf("Error opening file for reading.\n");
        exit(1);
    }

    *count = 0;
    while (fscanf(file, "%d,%99[^,],%99[^,],%d\n", &books[*count].bookId, books[*count].BTitle, books[*count].BAuthor, &books[*count].yearPublished) == 4) {
        (*count)++;
    }

    fclose(file);
}

int main() {
    struct Book books[100];
    int bookCount = 0;
    int option;

    // Load data from file when the program starts
    readFromFile(books, &bookCount);

    while (1) {
        printf("\n==========Library Information System==========\n");
        printf("1. Input book record\n");
        printf("2. Display book record (Sorted)\n");
        printf("3. Update book record\n");
        printf("4. Erase book record\n");
        printf("5. Save and Exit\n");
        printf("Enter your choice (1-5): ");
        scanf("%d", &option);

        switch (option) {
            case 1:
                if (bookCount < 100) {
                    inputBook(&books[bookCount]);
                    bookCount++;
                    printf("Successfully added!\n");
                } else {
                    printf("Sorry, can't add more books.\n");
                }
                break;
            case 2:
                if (bookCount > 0) {
                    displayBooks(books, bookCount);
                } else {
                    printf("Display Book Not Available. \n");
                }
                break;
            case 3:
                if (bookCount > 0) {
                    updateBook(books, bookCount);
                } else {
                    printf("Update Not Available. \n");
                }
                break;
            case 4:
                if (bookCount > 0) {
                    eraseBook(books, &bookCount);
                } else {
                    printf("Erase Not Available. \n");
                }
                break;
            case 5:
                // Save data to file before exiting
                saveAndExit(books, bookCount);
                break;
            default:
                printf("Invalid, please enter a number between (1-5), Thank you. \n");
        }
    }

    return 0;
}
