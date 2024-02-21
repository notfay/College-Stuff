#include <stdio.h>

int bubbleS(int arr[], int size) {
	for (int i = 0; i < size - 1; i++) {
		for (int j = 0; j < size - i -1; j++) {
			if (arr[j] > arr[j+1]) {
				int temp = arr[j];
				arr[j] = arr[j+1];
				arr[j+1] = temp;
				
			}
		}
	}
}

int main () {
	
	int n;
	
	scanf("%d", &n);
	getchar ();
	
	int arr[n];
	int size = n;
	
	for (int i = 0; i < n; i++) {
		scanf("%d", &arr[i]);
	}
	
	bubbleS(arr, size);
	
	for (int j = 0; j < n; j++) {
		printf ("%d, ", arr[j]);
	}
	
	return 0;
}
