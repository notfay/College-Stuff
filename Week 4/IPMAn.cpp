#include <stdio.h>
#include <string.h>

int main() {

char n[100] ;
int a;

scanf ("%d\n", &a);    

int i = 1;

for (i = 1; i<=a; i++) {
	int len = strlen(n);
	scanf ("%[^\n]", &n);
	
	if (len == 11) {
		printf ("Case #%d : YES\n", i);
	}
	
	else {
	printf ("Case #%d : NO\n", i);
	}
}


	return 0;
}

