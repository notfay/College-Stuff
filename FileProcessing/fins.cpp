#include <stdio.h> 


void write(char menu[100][100], int price[100], int count) {
	
	FILE *files= fopen("Menu.txt", "w");
	
	for (int i = 0; i < count; i++) {
		fprintf (files, "%s | %d\n", menu[i], price[i]);
	}
	
	fclose(files);
}


void read() {
		
		FILE *files= fopen("Menu.txt", "r");
		
		if (files == NULL) {
			printf ("No Data :(");
		} 
		else {
			while(!feof(files)) {
				int price, menu[100];
				fscanf(files, "%[^|] | %d\n", menu, &price);
				printf ("%s | %d\n", menu, price);
			}
		}
		
		
		fclose(files);
		
}


int main () {
	
	
	char menu[100][100] = {{"Nasi goreng"}, {"Nasi Campur"}, {"Mie Ayam"}};
	int price[100] = {5000, 20000, 10000};
	
	
	//write(menu, price, 3);
	
	read();
	
	
	
	
	
	
	return 0;
}

























