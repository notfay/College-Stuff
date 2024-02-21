#include <stdio.h>

int main () {
	
	
	
	int a, b;
	int t;
	int j, l;
	int tamb, tes;
	int tamb2, tes2;
	
	scanf ("%d", &t);
	getchar ();
	for (int i =1; i <=t; i++) {
		
		
		scanf ("%d %d",&j, &l);
		  for (int j = 1; j <= t; j++) {
		  	
		  	scanf ("%d", &tamb);
		  	tes += tamb;
		  }
		
		scanf ("%d %d",&a, &b);
		  for (int a = 1; a <= t; a++) {
		  	
		  	scanf ("%d", &tamb2);
		  	tes2 += tamb2;	  	
	}
	printf ("Case #%d : %d No Wishes", i, tes);
}
	
	return 0;
}
