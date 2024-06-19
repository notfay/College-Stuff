#include <stdio.h>

#define MAX 100

int parent[MAX];
int rank[MAX];

//void makeSet(int v) {
//	parent[v] = v;
//}
//
//int findSet(int v) {
//	if(v == parent[v]) {
//		return v;
//	}
//	return findSet(parent[v]);
//}
//
//void unionSet(int a, int b) {
//	a = findSet(a);
//	b = findSet(b);
//	
//	if(a != b) {
//		parent[b] = a;
//	}
//}

/*==========--------------------------------==================*/

void makeSet(int v) {
	parent[v] = v;
	rank[v] = 0;
}

int findSet(int v) {
	if(v == parent[v]) {
		return v;
	}
	return parent[v] = findSet(parent[v]);
}

int swap(int a, int b) {
	int temp = a;
	a = b;
	b = temp;
	
	return temp;
}

void unionSet(int a, int b) {
	a = findSet(a);
	b = findSet(b);
	
	if(a != b) {
		if(rank[a] < rank[b]) {
			swap(a,b);	
		}
		
	parent[b] = a;
	
	if(rank[a] == rank[b]) {
		rank[a]++;
	}
   }

}

int main() {
	
	int numV = 8;
	
	for(int i = 1; i <= numV; i++) {
		makeSet(i);
	}
	
	printf("Before Union:\n");
	printf("Parent of 3 : %d\n", findSet(3));
	printf("Parent of 1 : %d\n", findSet(1));
	printf("\nDisjoint set like this:\n");
	
	for (int i = 1; i <= numV; i++) {
		printf("%d ", parent[i]);
	}
	
	printf("\n");
	
	for (int i = 1; i <= numV; i++) {
		printf("%d ", rank[i]);
	}
	
	
	unionSet(6,8);
	unionSet(5,7);
	unionSet(3,6);
	unionSet(3,5);
	unionSet(2,4);
	unionSet(2,3);
	unionSet(1,2);
	
	
	printf("\n\nAfter Union:");
	printf("\nParent of 3 : %d\n", findSet(3));
	printf("Parent of 1 : %d\n\n", findSet(1));
	
	for (int i = 1; i <= numV; i++) {
		printf("%d ", parent[i]);
	}
	
	printf("\n");
	
	for (int i = 1; i <= numV; i++) {
		printf("%d ", rank[i]);
	}

}
