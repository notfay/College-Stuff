#include <stdio.h>
#include <string.h>
#include <stdlib.h>

struct node {
    char name[100];
    int damage;
    struct node *left;
    struct node *right;
};

struct node *createChar(char name[], int damage){
    struct node *newChar = (struct node *)malloc(sizeof(struct node));
    strcpy(newChar->name, name);
    newChar->damage = damage;
    newChar->left = NULL;
    newChar->right = NULL;
    
    return newChar;
}

struct node *insertChar(struct node *root, char name[], int damage){
    if(root == NULL){
        root = createChar(name, damage);
        return root;
    } else if (damage < root->damage){
        root->left = insertChar(root->left, name, damage);
    } else if (damage > root->damage){
        root->right = insertChar(root->right, name, damage);
    } 
    
    return root;
}

struct node *deleteChar(struct node *root, int damage) {
    if (root == NULL) return NULL;

    if (damage < root->damage) {
        root->left = deleteChar(root->left, damage);
    } else if (damage > root->damage) {
        root->right = deleteChar(root->right, damage);
    } else {
      // Case 1: No children
      if (root->left == NULL && root->right == NULL) {
        free(root);
        root = NULL;
      }
      // Case 2: One child
      else if (root->left == NULL) {
        struct node *temp = root->right;
        free(root);
        root = NULL;
        return temp;
      } else if (root->right == NULL) {
        struct node *temp = root->left;
        free(root);
        root = NULL;
        return temp;
      } else {
		    // successor
		    struct node *temp = root->right;
            while (temp->left != NULL) {
                temp = temp->left;
            }
            
            strcpy(root->name, temp->name);
            root->damage = temp->damage;
            root->right = deleteChar(root->right, temp->damage);
            
            // predecessor
            // struct node* temp = root->left;
            // while (temp->right != NULL){
            //      temp = temp->right;
            // }
    
            // root->left = deleteChar(root->left, temp->damage);
		}

    }
    return root;
}

void preorder(struct node *root){
    if (root == NULL) return;
    
    printf("%d ", root->damage);
    preorder(root->left);
    preorder(root->right);
}

void inorder(struct node *root){
    if (root == NULL) return;
    
    inorder(root->left);
    printf("%d ", root->damage);
    inorder(root->right);
}

void postorder(struct node *root){
    if (root == NULL) return;
    
    postorder(root->left);
    postorder(root->right);
    printf("%d ", root->damage);
}

int main() {
    
    struct node *root = NULL;
    
    root = insertChar(root, "char1", 100);
    root = insertChar(root, "char2", 80);
    root = insertChar(root, "char3", 10);
    root = deleteChar(root, 80);
    root = deleteChar(root, 10);
    
    printf("Preorder: ");
    preorder(root);
    printf("\n");
    printf("Postorder: ");
    postorder(root);
    printf("\n");
    printf("Inorder: ");
    inorder(root);
    printf("\n");
    
    return 0;
}