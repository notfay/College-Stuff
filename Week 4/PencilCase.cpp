#include <stdio.h>

int main () {

int a;
int r,h;
double rm;

scanf ("%d", &a);

for (int i = 1; i <=a; i++) {
	
	scanf ("%d %d", &r, &h);
	getchar ();
	rm = (2 * 3.14 * (r * r)) + (2 * 3.14 * r * h);
	printf ("Case #%d: %.2lf\n",i, rm);
}



return 0;
}
