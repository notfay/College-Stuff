//node, create, max, height, bal, lr, rr, insert, find pred, delte, search

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct node {
    char isbn;
    int height;

    struct node* left;
    struct node* right;

}node;


node* create(char isbn[]) {
    node* head = (node*)malloc(sizeof(node));
    head->left = NULL;
    head->right = NULL;

    head->height= 1;
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

node* lr(node* n) {
    node* s = n->right;
    node* x2 = s->left;

    s->left = n;
    n->right = x2;

    s->height = max(height(n->left), height(n->right)) + 1;
    s->height = max(height(s->left), height(s->right)) + 1;

    return s;
}

node* rr(node* n) {
    node* s = n->left;
    node* x2 = s->right;

    s->right = n;
    n->left = x2;

    s->height = max(height(n->left), height(n->right)) + 1;
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
    else if (strcmp(isbn, root->isbn) > 0) {
        root->right = insert(root->right, isbn);
    }
    else {
        return root;
    }

    int bal = getBal(root);
    root->height = max(height(root->left), height(root->right)) + 1;

    if(bal > 1 && strcmp(isbn, root->left->isbn) < 0) {
        return rr(root);
    }
    if(bal < -1 && strcmp(isbn, root->right->isbn) > 0) {
        return lr(root);
    }
    if(bal > 1 && strcmp(isbn, root->left->isbn) > 0) {
        root->left = lr(root->left);
        return rr(root);
    }
    if(bal < -1 && strcmp(isbn, root->right->isbn) < 0) {
        root->right = rr(root->right);
        return lr(root);
    }

    return root;
}





/*





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
    printIn(root);

    return 0;
}
