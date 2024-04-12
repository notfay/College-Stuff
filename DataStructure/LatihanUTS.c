#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct node {
    
    char name[20];
    char maker[20];
    char duration[20];
    
    struct node* next;
    struct node* prev;
    
}node;


node* create(char name[], char maker[], char duration[]) {
    node* head = malloc(sizeof(node));
    
    strcpy(head->name, name);
    strcpy(head->maker, maker);
    strcpy(head->duration, duration);
    
    head->next = NULL;
    head->prev = NULL;
    
    return head;
}

node* head = NULL;

//Queue System, pushTail, delbeg

node* insertSong(char name[], char maker[], char duration[]) {
    if(head == NULL) {
        head = create(name, maker, duration);
        return head;
    }
    
    node* ptr = head;
    
    while(ptr->next != NULL) {
        ptr = ptr->next;
    }
    
    node* temp = create(name, maker, duration);
    
    ptr->next = temp;
    temp->prev = ptr;
    
    return head;
    
}

node* songFinished() { //pop beg
    if(head == NULL) {
        printf("All Song Ended\n");
        return head;
    }
    
    node* ptr = head;
    
    head = head->next;
    free(ptr);
    
    return head;
}


void showPlaylist() {
    node* ptr = head;
    
    while(ptr != NULL) {
        printf("%s %s %s\n", ptr->name, ptr->maker, ptr->duration);
        ptr = ptr->next;
    }
    
    printf("End of Playlist\n");
}

void clearPlaylist() {
    node* temp;
    
    while(head != NULL) {
        temp = head;
        head = head->next;
        free(temp);
    }
}



/////////////////////////////////////////////////////////////////////////////////////////


void insertAfter(char name[], char newTitle[], char newArtist[], char newDuration[]) {
    node *ptr = head;
    
    while (ptr != NULL) {
        if (strcmp(ptr->name, name) == 0) {
            
            node *temp = create(newTitle, newArtist, newDuration);
            
            temp->prev = ptr;
            temp->next = ptr->next;

            if (ptr->next != NULL) {
                ptr->next->prev = temp;
            }
            
            ptr->next = temp;
            
            printf("Song '%s' inserted after '%s'.\n", newTitle, name);
            return;
        }
        ptr = ptr->next;
    }
    printf("Song '%s' not found in the playlist.\n", name);
}


void insertBefore(char name[], char newTitle[], char newArtist[], char newDuration[]) {
    node *ptr = head;
    
    while (ptr != NULL) {
        if (strcmp(ptr->name, name) == 0) {
            
            node *temp = create(newTitle, newArtist, newDuration);

            temp->next = ptr;
            temp->prev = ptr->prev;

            if (ptr->prev != NULL) {
                ptr->prev->next = temp;
            }
            
            ptr->prev = temp;
            
            if (ptr == head) {
                head = temp; // Update head if inserting before the head node
            }
            printf("Song '%s' inserted before '%s'.\n", newTitle, name);
            return;
        }
        ptr = ptr->next;
    }
    printf("Song '%s' not found in the playlist.\n", name);
}






int main () {
    
    
    insertSong("Huahwi", "Gamer", "20.20");
    insertSong("sdadsi", "weamer", "20.20");
    insertSong("xcxcahwi", "Gamer", "20.20");
    insertSong("Hfdfhwi", "Gamer", "20.20");
    
    songFinished();
    
    showPlaylist();
    
    
    
    
    return 0;
}
