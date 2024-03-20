#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef struct node{
    char item;
    struct node *left;
    struct node *right;
}node;

node* create(char value){
    node* newNode = malloc(sizeof(node));
    newNode->item = value;
    newNode->left = NULL;
    newNode->right = NULL;
    
    return newNode;
}

int is_operator(char data){
    switch(data){
        case '+': return 1;
        case '-': return 1;
        case '*': return 1;
        case '/': return 1;
        case '^': return 1;
        default: return 0;
    }
}

void inorder(node* root){
    if(root == NULL) return;
    
    if(is_operator(root->item)) putchar('(');
    inorder(root->left);
    printf("%c", root->item);
    inorder(root->right);
    if(is_operator(root->item)) putchar(')');
}

void preorder(node* root){
    if(root == NULL) return;
    printf("%c", root->item);
    preorder(root->left);
    preorder(root->right);
}

void postorder(node* root){
    if(root == NULL) return;
    postorder(root->left);
    postorder(root->right);
    printf("%c", root->item);
}

node* construct_tree_prefix(char prefix[]){
    node* stack[100];
    int top = -1;
    int i = strlen(prefix)-1;
    
    while(prefix[i] != 0){
        char ch = prefix[i];
        if(ch >= '0' && ch <= '9'){
            node* newNode = create(ch);
            stack[++top] = newNode;
        }else{
            node* newNode = create(ch);
            newNode->left = stack[top--];
            newNode->right = stack[top--];
            stack[++top] = newNode;
        }
        i--;
    }
    return stack[top--];
}

node* construct_tree_postfix(char postfix[]) {
    struct node* stack[100];
    int top = -1;
    int i = 0;
    while (postfix[i] != '\0') {
        char ch = postfix[i];
        if (ch >= '0' && ch <= '9') {
            node* newNode = create(ch);
            stack[++top] = newNode;
        } else {
            node* newNode = create(ch);
            newNode->right = stack[top--];
            newNode->left = stack[top--];
            stack[++top] = newNode;
        }
        i++;
    }
    return stack[top--];
}

//INPUT PAKAI INFIX
int isOperand(char c) {
  return isdigit(c) || isalpha(c);
}

int precedence(char c) {
  switch (c) {
    case '+':
    case '-':
      return 1;
    case '*':
    case '/':
      return 2;
    case '^':
      return 3;
    default:
      return -1;
  }
}

node* construct_tree_infix(char infix[]) {
  // !!! First, we need to convert the infix to postfix (or prefix, up to you), how? using the stack algorithm that we've been learned back then !!!
  
  // We need to allocate memory for the postfix expression
  int n = strlen(infix);
  char *postfix = malloc(sizeof(char) * (n + 1));

  // Then, create a stack to store the operators
  char *stack = malloc(sizeof(char) * n);
  int top = -1;

  // Traverse the infix expression from left to right
  int i = 0, j = 0;
  while (infix[i] != '\0') {
    // If the current character is an operand, append it to the postfix expression
    if (isOperand(infix[i])) {
      postfix[j++] = infix[i];
    }
    
    // If the current character is an opening parenthesis '(', push it to the stack
    else if (infix[i] == '(') {
      stack[++top] = infix[i];
    }
    
    // If the current character is a closing parenthesis ')', pop and append the operators from the stack until an opening parenthesis is encountered
    else if (infix[i] == ')') {
      while (top != -1 && stack[top] != '(') {
        postfix[j++] = stack[top--];
      }
      // Pop and discard the opening parenthesis
      if (top != -1 && stack[top] == '(') {
        top--;
      }
    }
    
    // If the current character is an operator, pop and append the operators from the stack that have higher or equal precedence, and then push the current operator to the stack
    else {
      while (top != -1 && precedence(infix[i]) <= precedence(stack[top])) {
        postfix[j++] = stack[top--];
      }
      stack[++top] = infix[i];
    }
    
    // Move to the next character in the infix expression
    i++;
  }

  // Pop and append the remaining operators from the stack to the postfix expression
  while (top != -1) {
    postfix[j++] = stack[top--];
  }

  // Terminate the postfix expression with a null character
  postfix[j] = '\0';

  // Free the memory allocated for the stack
  free(stack);

  // Construct a binary tree from a postfix expression
  return construct_tree_postfix(postfix);
}

int main()
{
    char postfix[] = "359+2*+";
    node* root = construct_tree_postfix(postfix);
    
    printf("Inorder: ");
    inorder(root);
    
    printf("\nPreorder: ");
    preorder(root);
    
    printf("\nPostorder: ");
    postorder(root);
    
    char prefix[] = "+3*+592";
    node* root2 = construct_tree_prefix(prefix);
    
    printf("\n\nInorder: ");
    inorder(root2);
    
    printf("\nPreorder: ");
    preorder(root2);
    
    printf("\nPostorder: ");
    postorder(root2);
    
    char infix[] = "(3+((5+9)*2))";
    node* root3 = construct_tree_infix(infix);
    
    printf("\n\nInorder: ");
    inorder(root3);
    
    printf("\nPreorder: ");
    preorder(root3);
    
    printf("\nPostorder: ");
    postorder(root3);
    
    return 0;
}
