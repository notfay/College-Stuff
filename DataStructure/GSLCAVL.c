#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//Hasil Modif dari contoh source code yang bapak berikan


typedef struct node {
    int item;
    char name[24];
    char position[24];
    int salary;

    struct node* left;
    struct node* right;
    int height;
} node;

int height(node* N) {
    if (N == NULL) {
        return 0;
    }
    return N->height;
}

int max(int a, int b) {
    return (a > b) ? a : b;
}

int search(node* root, int value) {
    if (root == NULL) {
        printf("%d is not found\n", value);
        return 0;
    }
    if (value == root->item) {
        printf("%d is found\n", value);
        return root->item;
    }
    if (value < root->item) {
        return search(root->left, value);
    }
    if (value > root->item) {
        return search(root->right, value);
    }
}

node* create(int value, char name[], char position[], int salary) {
    node* newNode = malloc(sizeof(node));
    newNode->item = value;
    strcpy(newNode->name, name);
    strcpy(newNode->position, position);
    newNode->salary = salary;

    newNode->left = NULL;
    newNode->right = NULL;
    newNode->height = 1;

    return newNode;
}

node* rightRotate(node* T) {
    node* S = T->left;
    node* X2 = S->right;

    S->right = T;
    T->left = X2;

    T->height = max(height(T->left), height(T->right)) + 1;
    S->height = max(height(S->left), height(S->right)) + 1;

    return S;
}

node* leftRotate(node* T) {
    node* S = T->right;
    node* X2 = S->left;

    S->left = T;
    T->right = X2;

    T->height = max(height(T->left), height(T->right)) + 1;
    S->height = max(height(S->left), height(S->right)) + 1;

    return S;
}

int getBalance(node* N) {
    if (N == NULL) {
        return 0;
    }
    return height(N->left) - height(N->right);
}

node* insert(node* root, int value, char name[], char position[], int salary) {
    if (root == NULL) {
        return create(value, name, position, salary);
    }
    if (value < root->item) {
        root->left = insert(root->left, value, name, position, salary);
    } else if (value > root->item) {
        root->right = insert(root->right, value, name, position, salary);
    } else return root;

    root->height = 1 + max(height(root->left), height(root->right));

    int balance = getBalance(root);

    if (balance > 1 && value < root->left->item) {
        return rightRotate(root);
    }

    if (balance < -1 && value > root->right->item) {
        return leftRotate(root);
    }

    if (balance > 1 && value > root->left->item) {
        root->left = leftRotate(root->left);
        return rightRotate(root);
    }

    if (balance < -1 && value < root->right->item) {
        root->right = rightRotate(root->right);
        return leftRotate(root);
    }

    return root;
}

node* findInorderPredecessor(node* root) {
    node* curr = root;

    while (curr && curr->right != NULL) {
        curr = curr->right;
    }
    return curr;
}

node* deleteNode(node* root, int value) {
    if (root == NULL) {
        return root;
    }
    if (value < root->item) {
        root->left = deleteNode(root->left, value);
    } else if (value > root->item) {
        root->right = deleteNode(root->right, value);
    } else {
        if (root->left == NULL) {
            node* temp = root->right;
            free(root);
            return temp;
        } else if (root->right == NULL) {
            node* temp = root->left;
            free(root);
            return temp;
        }

        node* temp = findInorderPredecessor(root->left);
        root->item = temp->item;

        root->left = deleteNode(root->left, temp->item);
    }

    root->height = 1 + max(height(root->left), height(root->right));

    int balance = getBalance(root);

    if (balance > 1 && getBalance(root->left) >= 0) {
        return rightRotate(root);
    }

    if (balance < -1 && getBalance(root->right) <= 0) {
        return leftRotate(root);
    }

    if (balance > 1 && getBalance(root->left) < 0) {
        root->left = leftRotate(root->left);
        return rightRotate(root);
    }

    if (balance < -1 && getBalance(root->right) > 0) {
        root->right = rightRotate(root->right);
        return leftRotate(root);
    }

    return root;
}

void preorder(node* root) {
    if (root == NULL) return;
    printf("ID : %d, Name %s, Position : %s, Salary : %d\n", root->item, root->name, root->position, root->salary);
    preorder(root->left);
    preorder(root->right);
}

void inOrder(node* root) {
    if (root == NULL) return;
    inOrder(root->left);
    printf("ID : %d, Name %s, Position : %s, Salary : %d\n", root->item, root->name, root->position, root->salary);
    inOrder(root->right);
}

int main() {
    node* root = NULL;
    root = insert(root, 101, "John Doe", "Manager", 5000);
    root = insert(root, 103, "Jane Doe", "Developer", 4000);
    root = insert(root, 105, "Doe John", "Analyst", 45000);
    root = insert(root, 102, "john Jane", "Designer", 48000);
    root = insert(root, 104, "Jane John", "HR", 42000);
    root = insert(root, 100, "Kujo Jotaro", "Analyst", 42000);
   

    printf("Data Karyawan (Preorder):\n");
    preorder(root);
    
    printf("\n");
    printf("Data Karyawan (Inorder):\n");
    inOrder(root);

    return 0;
}
