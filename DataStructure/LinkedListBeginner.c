#include <stdio.h>
#include <string.h>	//Manipulasi string
#include <stdlib.h>	//Malloc function


typedef struct node{
	
	int data;				//Data yang di dalam nodenya
	struct node* next;		//Membuat node baru
	
} node;

node* create(int value) {	//Create the node
	
	node* head = NULL;					
	head = malloc(sizeof(node));		//Menentukan alamat di nodenya
	
	head->data = value;	//Value dari main
	head->next = NULL;
	
	return head;
	
}

//Satu paket di atas



void display(node* head) {			//Display each Node using Loop
	if (head == NULL) {
		printf("There's NOTHING'\n");
	}
	
	node* ptr = head;
	
	while(ptr != NULL) {			//Saat belum ketemu NULL atau diakhir
		printf("%d -> ", ptr->data);	//Akan print value 
		ptr = ptr->	next;				//Pointer akan lanjut ke alamat node selanjutnya 
	}
	
}


void nodeDisplay(node* head) {	//Display the total nodes that is created

	int count = 0;		//Standard counter line, then count++
			
	if (head == NULL) {
		printf("There's NOTHING'\n");
	}
	
	node* ptr = head;
	
	while(ptr != NULL) {			
		 
		count++;
		ptr = ptr->	next;				
	}
	printf("\nTotal Node is %d\n", count);

}





void seek(node* head) {	

	int count = 0;		
			
	if (head == NULL) {
		printf("There's NOTHING'\n");
	}
	
	node* ptr = head;
	
	while(ptr->next != NULL) {			
		ptr = ptr->	next;				
	}
	printf("Last Node is %d\n", ptr->data);

}


 	
 

int main () {
	
	node* node1 = create(45);
	node* node2 = create(95);		//Untuk membuat node dan valuenya
	node* node3 = create(11);
	node* node4 = create(23);
	
	node1 -> next = node2;
	node2 -> next = node3;			//Menyambungkan nodenya;
	node3 -> next = node4;
	
	
	
	display(node1);
	nodeDisplay(node1);
	seek(node1);
	
	
	
	return 0;
	
}
