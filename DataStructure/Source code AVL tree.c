#include <stdio.h>
#include <stdlib.h>

typedef struct node{
    int item;
    struct node* left;
    struct node* right;
    int height;
}node;

// Function to get height value
int height(node* N){
    if(N == NULL){
        return 0;
    }
    return N->height;
}

int max(int a, int b){
    return (a > b)? a : b;
}

int search(node* root, int value){
    if(root == NULL){
        printf("%d is not found\n", value);
        return 0;
    }
    if(value == root->item){
        printf("%d is found\n", value);
        return root->item;
    }
    if(value < root->item){
        return search(root->left, value);
    }
    if(value > root->item){
        return search(root->right, value);
    }
}

node* create(int value){
    node* newNode = malloc(sizeof(node));
    newNode->item = value;
    newNode->left = NULL;
    newNode->right = NULL;
    newNode->height = 1;
    
    return newNode;
}

node* rightRotate(node* T){
    node* S = T->left;
    node* X2 = S->right;
    
    // Perform rotation
    S->right = T;
    T->left = X2;
    
    // Update height
    T->height = max(height(T->left), height(T->right)) + 1;
    S->height = max(height(S->left), height(S->right)) + 1;
    
    return S;
}

node* leftRotate(node* T){
    node* S = T->right;
    node* X2 = S->left;
    
    // Perform rotation
    S->left = T;
    T->right = X2;
    
    // Update height
    T->height = max(height(T->left), height(T->right)) + 1;
    S->height = max(height(S->left), height(S->right)) + 1;
    
    return S;
}

int getBalance(node* N){
    if(N == NULL){
        return 0;
    }
    return height(N->left) - height(N->right);
}

node* insert(node* root, int value){
    // 1. Perform the normal BST insertion
    if(root == NULL){
        return create(value);
    }
    if(value < root->item){
        root->left = insert(root->left, value);
    }
    else if(value > root->item){
        root->right = insert(root->right, value);
    }
    else return root;
    
    // 2. Update height 
    root->height = 1 + max(height(root->left), height(root->right));
    
    // 3. Get balance factor
    int balance = getBalance(root);
    
    // If not balance, there are 4 cases
    // a. Left Left case(Right Rotation)
    if(balance > 1 && value < root->left->item){
        return rightRotate(root);
    }
    
    // b. Right Right case(Left Rotation)
    if(balance < -1 && value > root->right->item){
        return leftRotate(root);
    }
    
    // c. Left Right case(Left Right Rotation)
    if(balance > 1 && value > root->left->item){
        root->left = leftRotate(root->left);
        return rightRotate(root);
    }
    
    // d. Right Left case(Right Left Rotation)
    if(balance < -1 && value < root->right->item){
        root->right = rightRotate(root->right);
        return leftRotate(root);
    }
    
    return root;
}

node* findIndorderDecc(node* root){
    node* curr = root;
    
    // Find the rightmost leaf
    while(curr && curr->right != NULL){
        curr = curr->right;
    }
    return curr;
}

node* deleteNode(node* root, int value){
    // STEP 1: Perform the normal BST deletion
    if(root == NULL){
        return root;
    }
    if(value < root->item){
        root->left = deleteNode(root->left, value);
    }
    else if(value > root->item){
        root->right = deleteNode(root->right, value);
    }else{
        // If the node is with only one child or no child
        if(root->left == NULL){
            node* temp = root->right;
            free(root);
            return temp;
        }else if(root->right == NULL){
            node* temp = root->left;
            free(root);
            return temp;
        }
        
        // If the node has two children
        node* temp = findIndorderDecc(root->left);
        
        // Place the inorder decc in position of the deleted node
        root->item = temp->item;
        
        //Delete the inorder decc node
        root->left = deleteNode(root->left, temp->item);
    }
    
    // STEP 2: Update height
    root->height = 1 + max(height(root->left), height(root->right));
    
    // STEP 3: Get Balance factor
    int balance = getBalance(root);
    
    // If node is unbalance
    // a. Left Left case(Right Rotation)
    if(balance > 1 && getBalance(root->left) >= 0){
        return rightRotate(root);
    }
    
    // b. Right Right case(Left Rotation)
    if(balance < -1 && getBalance(root->right) <= 0){
        return leftRotate(root);
    }
    
    // c. Left Right case(Left Right Rotation)
    if(balance > 1 && getBalance(root->left) < 0){
        root->left = leftRotate(root->left);
        return rightRotate(root);
    }
    
    // d. Right Left case(Right Left Rotation)
    if(balance < -1 && getBalance(root->right) > 0){
        root->right = rightRotate(root->right);
        return leftRotate(root);
    }
    
    return root;
}

void preorder(node* root){
    if(root == NULL) return;
    printf("%d ", root->item);
    preorder(root->left);
    preorder(root->right);
}

int main()
{
    node* root = NULL;
    root = insert(root, 9);
    root = insert(root, 5);
    root = insert(root, 10);
    root = insert(root, 0);
    root = insert(root, 6);
    root = insert(root, 11);
    root = insert(root, -1);
    root = insert(root, 1);
    root = insert(root, 2);
    
    preorder(root);
    
    root = deleteNode(root, 10);
    
    printf("\n");
    preorder(root);
    return 0;
}