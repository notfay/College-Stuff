//node, create, max, height, bal, lr, rr, insert, find pred, delte, search

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct node {
	
	char isbn[23];
	int height;
	
	struct node* left;
	struct node* right;		

} node;

node* create(char isbn[]) {
	node* head = (node*)malloc(sizeof(node));
	head->left = NULL;
	head->right = NULL;
	
	head->height = 1;
	strcpy(head->isbn, isbn);
	
	return head;
}

int max(int a, int b) {
	return (a>b) ? a:b;
}

int height (node* n) {
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

node* lr(node* t) {
	node* s = t->right;
	node* x2 = s->left;
	
	s->left = t;
	t->right = x2;
	
	s->height = max(height(s->left), height(s->right));
	t->height = max(height(t->left), height(t->right));
	
	return s;
}

node* rr(node* t) {
	node* s = t->left;
	node* x2 = s->right;
	
	s->right = t;
	t->left = x2;
	
	s->height = max(height(s->left), height(s->right));
	t->height = max(height(t->left), height(t->right));
	
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
	
	root->height = max(height(root->left), height(root->right));
	int bal = getBal(root);
	
	if(bal > 1 && strcmp(isbn, root->left->isbn) < 0) {
		return rr(root);
	}
	if(bal < -1 && strcmp(isbn,root->right->isbn) > 0) {
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


node* findPred(node* root) {
	node* curr = root;
	while(curr->right != NULL) {
		curr = curr->right;
	}
	return curr;
}

node* delete(node* root, char isbn[]) {
	if(root == NULL) {
		return NULL;
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
			if (!temp) {
				temp = root;
				root = NULL;
			}
			else {
				*root = *temp;
			}	
			free(temp);
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
        return rr(root);
    }
    if(bal < -1 && getBal(root->right) <= 0) {
        return lr(root);
    }
    if(bal > 1 && getBal(root->left) < 0) {
        root->left = lr(root->left);
        return rr(root);
    }
    if(bal < -1 && getBal(root->right) > 0) {
        root->right = rr(root->right);
        return lr(root);
    }
    
    return root;
	
	
}

void printIn(node* root) {
	if(root != NULL) {
		printIn(root->left);
		printf("%s ", root->isbn);
		printIn(root->right);
	}
}



int main() {
    node* root = NULL;

    // Example insertions
    root = insert(root, "ISBN1234");
    root = insert(root, "ISBN5678");
    root = insert(root, "ISBN91011");

    printf("Inorder traversal of the AVL tree before deletion:\n");
    printIn(root);

    // Delete a node
    root = delete(root, "ISBN5678");

    printf("\nInorder traversal of the AVL tree after deletion:\n");
    printIn(root);
	//testing
    return 0;
}
