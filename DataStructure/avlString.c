//Node, create, max, height, bal, lr, rr, insert, find pred, delete, search

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct node {

    struct node* left;
    struct node* right;
    int height;
    char isbn[20];
}node;

node* create(char isbn[]) {
    node* head = malloc(sizeof(node));
    head->left = NULL;
    head->right = NULL;
    strcpy(head->isbn, isbn);
    head->height = 1;
    return head;
}

int max(int a, int b) {
    return(a>b) ? a:b;
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

node* rightR(node* t) {
    node* s = t->left;
    node* x2 = s->right;

    s->right = t;
    t->right = x2;

    t->height = max(height(t->left), height(t->right)) + 1;
    s->height = max(height(s->left), height(s->right)) + 1;

    return s;
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

node* insert(node* root, char isbn[]) {
    if(root == NULL) {
        return create(isbn);
    }

    if(strcmp(root->isbn, isbn) < 0) {
        root->left = insert(root->left, isbn);
    }
    else if (strcmp(root->isbn, isbn) > 0) {
        root->right = insert(root->right, isbn);
    }
    else {
        return root;
    }

    root->height = max(height(root->left), height(root->right)) + 1;
    int bal = getBal(root);

    if(bal > 1 && strcmp(root->left->isbn, isbn) < 0) {
        return rightR(root);
    }
    if(bal < -1 && strcmp(root->right->isbn, isbn) > 0) {
        return leftR(root);
    }
     if(bal > 1 && strcmp(root->left->isbn, isbn) > 0) {
        root->left = leftR(root->left);
        return rightR(root);
    }
    if(bal < -1 && strcmp(root->left->isbn, isbn) < 0) {
        root->right = leftR(root->right);
        return leftR(root);
    }

    return root;


}


int main () {

    return 0;
}

