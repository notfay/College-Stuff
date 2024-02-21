#include <stdio.h> 

int main () {
	
	FILE *data;
	data = fopen("testdata.in", "r");
	
	if (data== NULL) {
		printf ("Data not found\n");
		return 1;
	}
	
	int a, b;
	
	fscanf(data, "%d %d", &a, &b);

    fclose(data);

    printf("%d\n", a + b);
	
	
	return 0;
}
