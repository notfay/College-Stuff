//node, create, max, height, getbal, lr, rr, insert, fidn pred, delete

#include <string.h>
#include <stdlib.h>
#include <stdio.h>  

typedef struct node {

int height;
char isbn[23];
struct node* left;
struct node* right;
}node;

node* create (char isbn[]) {
    node* head = malloc(sizeof(node));
    head->left = NULL;
    head->right = NULL;

    head->height = 1;
    strcpy(head->isbn, isbn);

    return head;
}

int max(int a, int b) {
    return (a>b) ? a:b;
}

int height(node* n) {
    if(n == NULL) {
        return 0;
    }
    return n->height;
}

int getBal(node* n) {
    if(n == NULL) {
        return 0;
    }
    return height(n->left) - height(n->right);
}

node* leftR(node* t) {
    node* s = t->right;
    node* x2 = s->left;

    s->left = t;
    t->right = x2;

    t->height = max(height(t->left), height(t->right)) + 1;
    s->height = max(height(s->left), height(s->right)) + 1;

    return s;
}

node* rightR(node* t) {
    node* s = t->left;
    node* x2 = s->right;

    s->right = t;
    t->left = x2;

    t->height = max(height(t->left), height(t->right)) + 1;
    s->height = max(height(s->left), height(s->right)) + 1;

    return s;
}

node* insert(node* root, char isbn[]) {
    if(root == NULL) {
        return create(isbn);
    }

    if(strcmp(isbn, root->isbn) < 0) {
        root->left = insert(root->left, isbn);
    }
    else if(strcmp(isbn, root->isbn) > 0) {
        root->right = insert(root->right, isbn);
    }
    else {
        return root; 
    }

    root->height = 1 + max(height(root->left), height(root->right));
    int bal = getBal(root);

    if(bal > 1 && strcmp(isbn, root->left->isbn) < 0) {
        return rightR(root);
    }
    if(bal < -1 && strcmp(isbn, root->right->isbn) > 0) {
        return leftR(root);
    }
    if(bal > 1 && strcmp(isbn, root->left->isbn) > 0) {
        root->left = leftR(root->left);
        return rightR(root);
    }
    if(bal < -1 && strcmp(isbn, root->right->isbn) < 0) {
        root->right = rightR(root->right);
        return leftR(root);
    }

    return root;
}


node* findPred(node* root) {
    node* curr = root;
    while(curr->right != NULL) {
        curr = curr->right;
    }
    return curr;
}


node* delete(node* root, char isbn[]) {
    if(root == NULL) {
        return root;
    }
    if(strcmp(isbn, root->isbn) < 0) {
        root->left = delete(root->left, isbn);
    }
    else if (strcmp(isbn, root->isbn) > 0) {
        root->right = delete(root->right, isbn);
    }
    else {
        if(!root->left || !root->right) {
            node* temp = root->left ? root->left:root->right;

            if(!temp) {
                temp = root;
                root = NULL;
            }
            else {
                *root = *temp;
            }
            free(root);
        }
        else {
            node* temp = findPred(root->left);
            strcpy(root->isbn, temp->isbn);
            root->left = delete(root->left, temp->isbn);
        }
    }

    if(root == NULL) {
        return root;
    }

    root->height = max(height(root->left), height(root->right)) + 1;
    int bal = getBal(root);

    if(bal > 1 && getBal(root->left) >= 0) {
        return rightR(root);
    }
    if(bal < -1 && getBal(root->right) <= 0) {
        return leftR(root);
    }
    if(bal > 1 && getBal(root->left) < 0) {
        root->left = leftR(root->left);
        return rightR(root);
    }
    if(bal < -1 && getBal(root->right) > 0) {
        root->right = rightR(root->right);
        return leftR(root);
    }

    return root;


}

node* search(node* root, char isbn[]) {
    if(root == NULL) {
        return NULL;
    }
    if(strcmp(isbn, root->isbn) == 0) {
        printf("Found\n");
        return root;
    }
    else if(strcmp(isbn, root->isbn) < 0) {
        return search(root->left, isbn);
    }
    else {
        return search(root->right, isbn);
    }
}

void inOrder(node* root) {
    if (root != NULL) {
        inOrder(root->left);
        printf("%s ", root->isbn);
        inOrder(root->right);
    }
}

int main() {
    node* root = NULL;

    // Inserting nodes
    root = insert(root, "ISBN1");
    root = insert(root, "ISBN2");
    root = insert(root, "ISBN3");
    root = insert(root, "ISBN4");
    root = insert(root, "ISBN5");

    // Deleting a node
    root = delete(root, "ISBN3");

    // Printing the tree or other operations can be added for testing
    // Example: Inorder traversal to print all nodes
    printf("Inorder traversal of the AVL tree:\n");
    inOrder(root);

    return 0;
}