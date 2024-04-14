#include <stdio.h>
#include <string.h>
#include <stdlib.h> 

typedef struct node {

char title[15];
char name[15];
char duration[15];

struct node* next;
struct node* prev;

}node;

node* head = NULL;

node* create(char title[], char name[], char duration[]) {
    node* head = malloc(sizeof(node));

    strcpy(head->title, title);
    strcpy(head->name, name);
    strcpy(head->duration, duration);

    head->next = NULL;
    head->prev = NULL;

    return head;
}

node* insertSong(char title[], char name[], char duration[]) {
    if(head == NULL) {
        head = create(title, name, duration);
        return head;
    }

    node* ptr = head;

    while(ptr->next != NULL) {
        ptr = ptr->next;
    }

    node* temp = create(title, name, duration);

    ptr->next = temp;
    temp->prev = ptr;

    return head;
} 


node* removeSong() {
    node* ptr = head;

    head = head->next;
    free(ptr);

    printf("Song Removed!\n");
    return head;
}

void showPlaylist() {
    node* ptr = head;

    while(ptr != NULL) {
        printf("Title: %s | Artist: %s | Duration %s\n", ptr->title, ptr->name, ptr->duration);
        ptr = ptr->next;
    }
}

void clearPlaylist() {
    node* temp;

    while(head != NULL) {
        temp = head;
        head = head->next;
        free(temp);
    }

    printf("Playlist Cleared\n");
}

void insertAfter(char title[], char newTitle[], char newArtist[], char newDuration[]) {

    node* ptr = head;

    while(ptr != NULL) {

        if(strcmp(ptr->title, title) == 0) {
            node* temp = create(newTitle, newArtist, newDuration);

            temp->prev = ptr;
            temp->next = ptr->next;
            
            if(ptr->next != NULL) {
                ptr->next->prev = temp;
            }
            ptr->next = temp;

            printf("Song '%s' inserted after '%s'.\n", newTitle, title);
            return;

        }

        ptr = ptr->next;
    } 

}

void insertBefore(char title[], char newTitle[], char newArtist[], char newDuration[]) {
    node* ptr = head;

    while(ptr != NULL) {

        if(strcmp(ptr->title, title) == 0) {
            node* temp = create(newTitle, newArtist, newDuration);

            temp->next = ptr;
            temp->prev = ptr->prev;

            if(ptr->prev != NULL) {
                ptr->prev->next = temp;
            
            }
            ptr->prev = temp;

            if(head == ptr) {
                temp = head;
            }

            printf("Song '%s' inserted before '%s'.\n", newTitle, title);
            return;
        }

        ptr = ptr->next;
    }
}


int main () {

    insertSong("Song Title 1", "Artist 1", "03:30");
    insertSong("Song Title 2", "Artist 2", "04:00");
    insertSong("Song Title 3", "Artist 3", "03:45");
    insertSong("Song Title 4", "Artist 4", "03:15");

    showPlaylist();

    insertAfter("Song Title 2", "Song Title 2.5", "Artist 2.5", "03:20");
    insertBefore("Song Title 4", "Song Title 3.5", "Artist 3.5", "03:30");

    showPlaylist();

    removeSong();

    showPlaylist();

    clearPlaylist();

    showPlaylist();

    return 0;
}