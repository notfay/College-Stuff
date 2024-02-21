#include <stdio.h>

int main () {
	
	int N;
	scanf("%d", &N);
	
	int jojo, lili, bibi;
	scanf ("%d %d %d", &jojo, &lili, &bibi);
	
	int total = jojo + lili + bibi;
	
	int part[N];
	
	for (int i = 0; i < N; i++) {
		
		scanf ("%d", &part[i]);
		total += part[i];
	}
	
	int avg = total / (N+3);
	
	printf ("%s\n", (jojo >= avg) ? "Jojo lolos" : "Jojo tidak lolos");
	printf ("%s\n", (lili >= avg) ? "Lili lolos" : "Lili tidak lolos");
	printf ("%s\n", (bibi >= avg) ? "Bibi lolos" : "Bibi tidak lolos");
	

	
	return 0;
}
