#include <stdio.h>
#include <stdlib.h>

typedef struct node {
	
	int data;
	struct node* left;
	struct node* right;
	
}node;

node* create(int value) {
	node* head = malloc(sizeof(node));
	
	head->data = value;
	head->left = NULL;
	head->right = NULL;
	
	return head;
}

int search(node* root, int value) {
	if(root == NULL) {
		printf("Empty\n");
		return 0;
	}
	if(value == root->data) {
		printf("%d Found", value);
		return root->data;
	}
	if(value < root->data) {
		return search(root->left, value);
	}
	if(value > root->data) {
		return search(root->right, value);
	}
}

node* insert(node* root, int value) {
	if(root == NULL) {
		return create(value);
	}
	if(value == root->data) {
		printf("Its the same\n");
		return root;
	}
	if(value < root->data) {
		root->left = insert(root->left, value);
	}
	if(value > root->data) {
		root->right = insert(root->right, value);
	}
	
	return root;
}

node* findDecc(node* root) {
	node* curr = root;
	
	while(curr && curr->right != NULL) {
		curr = curr->right;
	}
	return curr;
}

node* delIn(node* root, int value) {
	if(root == NULL) {
		return root;
	}
	
	if(value < root->data) {
		root->left = delIn(root->left, value);
	}
	else if(value > root->data) {
		root->right = delIn(root->right, value);
	}
	else {
		if(root->left == NULL) {
			node* temp = root->right;
			free(root);
			return temp;
		}
		else if(root->right == NULL) {
			node* temp = root->left;
			free(root);
			return temp;
		}
		
		node* temp = findDecc(root->left);
		root->data = temp->data;
		root->left = delIn(root->left, temp->data);
		
	}
	return root;
}



void disIn(node* head) {
	if(head == NULL) return;
	disIn(head->left);
	printf("%d ", head->data);
	disIn(head->right);
}










int main () {
	node* head = NULL;
	
	head = insert(head, 30);
	head = insert(head, 15);
	head = insert(head, 37);
	head = insert(head, 26);
	head = insert(head, 19);
	head = insert(head, 28);
	head = insert(head, 32);
	head = insert(head, 45);
	head = insert(head, 40);
	head = insert(head, 7);
	
	printf("Inorder :");
	disIn(head);
	
	head = delIn(head, 30);
	
	printf("\nAfter : ");
	disIn(head);
	
	return 0;
}


