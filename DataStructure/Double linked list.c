#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Structure to represent a song
typedef struct Song {
    char title[100];
    char artist[100];
    char duration[10]; // hh:mm:ss format
    struct Song *prev;
    struct Song *next;
} Song;

// Global variable to represent the playlist
Song *playlist = NULL;



// Function to create a new song node
Song *createSong(char title[], char artist[], char duration[]) {
    Song *newSong = (Song *)malloc(sizeof(Song));
    strcpy(newSong->title, title);
    strcpy(newSong->artist, artist);
    strcpy(newSong->duration, duration);
    newSong->prev = NULL;
    newSong->next = NULL;
    return newSong;
}

// Function to insert a song at the end of the playlist
void insertSong(char title[], char artist[], char duration[]) {
    Song *newSong = createSong(title, artist, duration);
    if (playlist == NULL) {
        playlist = newSong;
    } else {
        Song *temp = playlist;
        while (temp->next != NULL) {
            temp = temp->next;
        }
        temp->next = newSong;
        newSong->prev = temp;
    }
}

// Function to remove a song from the playlist
void removeSong(char title[]) {
    Song *temp = playlist;
    while (temp != NULL) {
        if (strcmp(temp->title, title) == 0) {
            if (temp->prev != NULL) {
                temp->prev->next = temp->next;
            } else {
                playlist = temp->next;
            }
            if (temp->next != NULL) {
                temp->next->prev = temp->prev;
            }
            free(temp);
            printf("Song '%s' removed from the playlist.\n", title);
            return;
        }
        temp = temp->next;
    }
    printf("Song '%s' not found in the playlist.\n", title);
}

// Function to display all songs in the playlist
void showPlaylist() {
    Song *temp = playlist;
    if (temp == NULL) {
        printf("Playlist is empty.\n");
        return;
    }
    printf("Playlist:\n");
    while (temp != NULL) {
        printf("Title: %s | Artist: %s | Duration: %s\n", temp->title, temp->artist, temp->duration);
        temp = temp->next;
    }
}

// Function to clear the playlist
void clearPlaylist() {
    Song *temp;
    while (playlist != NULL) {
        temp = playlist;
        playlist = playlist->next;
        free(temp);
    }
    printf("Playlist cleared.\n");
}

// Function to insert a new song after a given song title
void insertAfter(char title[], char newTitle[], char newArtist[], char newDuration[]) {
    Song *temp = playlist;
    while (temp != NULL) {
        if (strcmp(temp->title, title) == 0) {
            Song *newSong = createSong(newTitle, newArtist, newDuration);
            newSong->prev = temp;
            newSong->next = temp->next;
            if (temp->next != NULL) {
                temp->next->prev = newSong;
            }
            temp->next = newSong;
            printf("Song '%s' inserted after '%s'.\n", newTitle, title);
            return;
        }
        temp = temp->next;
    }
    printf("Song '%s' not found in the playlist.\n", title);
}

// Function to insert a new song before a given song title
void insertBefore(char title[], char newTitle[], char newArtist[], char newDuration[]) {
    Song *temp = playlist;
    while (temp != NULL) {
        if (strcmp(temp->title, title) == 0) {
            Song *newSong = createSong(newTitle, newArtist, newDuration);
            newSong->next = temp;
            newSong->prev = temp->prev;
            if (temp->prev != NULL) {
                temp->prev->next = newSong;
            } else {
                playlist = newSong;
            }
            temp->prev = newSong;
            printf("Song '%s' inserted before '%s'.\n", newTitle, title);
            return;
        }
        temp = temp->next;
    }
    printf("Song '%s' not found in the playlist.\n", title);
}

int main() {
    insertSong("Song Title 1", "Artist 1", "03:30");
    insertSong("Song Title 2", "Artist 2", "04:15");
    insertSong("Song Title 3", "Artist 3", "02:50");
    insertSong("Song Title 4", "Artist 4", "05:00");

    showPlaylist();

    insertAfter("Song Title 2", "Song Title 2.5", "Artist 2.5", "01:45");
    insertBefore("Song Title 4", "Song Title 3.5", "Artist 3.5", "02:30");

    showPlaylist();

    removeSong("Song Title 2.5");
    removeSong("Song Title 4");

    showPlaylist();

    clearPlaylist();

    showPlaylist();

    return 0;
}
