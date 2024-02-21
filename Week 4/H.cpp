#include <stdio.h>

int main () {
	
	
int a;

scanf ("%d", &a);
getchar ();

int i = 0;

while (i < a) {
	int j = 0;
  while (j < a) {
  
	printf ("*");
	j++;
}
	printf("\n");
	i++;
}
    return 0;
}

