#include <stdio.h>

int main () {
	
	
int a;
int num;
int tot = 0;

scanf ("%d", &a);

for (int i = 1; i <= a; i++) {

	scanf ("%d", &num);
	tot += num;
	
}	
	
printf ("%d\n", tot); 



    return 0;
}

