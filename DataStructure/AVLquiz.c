#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct node {

    int isbn;
    struct node* left;
    struct node* right;
    int height;
}node;

int height(node* n) {
    if(n == NULL) {
        return 0;
    }
    return n->height;
 }

 int max(int a, int b) {
    return (a>b) ? a : b;
 }

node* create(int isbn) {
    node* head = malloc(sizeof(node));
    head->isbn = isbn;
    head->left = NULL;
    head->right = NULL;

    head->height = 1;

    return head;
}

node* search(node* root, int value) {
    if(root == NULL) {
        printf("Nothing\n");
        return NULL;
    }
    if(value == root->isbn) {
        printf("yes\n");
        return root;
    }
    else if(value < root->isbn) {
        return search(root->left, value);
    }
    else return search(root->right, value);
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

int getBal(node* n) {
    if(n == NULL) return 0;
    return height(n->left) - height(n->right);
}


node* insert(node* root, int value) {
    if(root == NULL) {
        return create(value);
    }
    if(value < root->isbn) {
        root->left = insert(root->left, value);
    }
    else if (value > root->isbn) {
        root->right = insert(root->right, value);
    }
    else {
        return root;
    }
    root->height = max(height(root->left), height(root->right)) + 1;
    int balance = getBal(root);


    if(balance > 1 && value < root->left->isbn) {
        return rightR(root);
    }
   if(balance < -1 && value > root->right->isbn) {
        return leftR(root);
    } 
     if(balance > 1 && value > root->left->isbn) {
        root->left = leftR(root->left);
        return rightR(root);
    }
     if(balance < -1 && value < root->left->isbn) {
        root->right = rightR(root->right);
        return leftR(root);
    }

    return root;
}

node* findPred(node* root) {
    node* curr = root;
    while(curr && curr->right != NULL) {
        curr = curr->right;
    }
    return curr;
}


node* delete(node* root, int value) {
    if(root == NULL) {
        return NULL;
    }
    if(value < root->isbn) {
        root->left = delete(root->left, value);
    }
    else if(value > root->isbn) {
        root->right = delete(root->right, value);
    }
    else {
        if(root->left == NULL) {
            node* temp = root->right;
            free(root);
            return temp;
        }
        else if (root->right == NULL) {
            node* temp = root->left;
            free(root);
            return temp;
        }

        node* temp = findPred(root->left);
        root->isbn = temp->isbn;

        root->left = delete(root->left, temp->isbn);

    }

    root->height = max(height(root->left), height(root->right)) + 1;
    int balance = getBal(root);


    if(balance > 1 && getBal(root->left) >= 0) {
        return rightR(root);
    }
   if(balance < -1 && getBal(root->right) <= 0) {
        return leftR(root);
    } 
     if(balance > 1 && getBal(root->left) < 0) {
        root->left = leftR(root->left);
        return rightR(root);
    }
     if(balance < -1 && getBal(root->right) > 0) {
        root->right = rightR(root->right);
        return leftR(root);
    }

    return root;

}



int main () {


    return 0;
}