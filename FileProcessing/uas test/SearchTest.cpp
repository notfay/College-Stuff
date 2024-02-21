#include <stdio.h>

int search(int arr[], int key, int min, int max) {
	
	int mid = (min+max)/2;
	
	if (min > max) return -1;
	
	if(arr[mid] == key) {
		return mid;
	}
	else if (arr[mid] < key) {
		return search(arr, key, mid+1, max);
	}
	else if (arr[mid] > key) {
		return search(arr, key, min, mid-1);
	}
	
}

int main () {
	
	int num;
	int key;
	
	scanf("%d", &num);
	
	int array[num];
	
	for (int i = 0; i < num; i++) {
		scanf("%d", &array[i]);
	}
	
	printf("Enter key\n");
	
	scanf("%d", &key);
	
	int res = search(array, key, 0, num-1);
	
	printf("%d", res);
	
	
	return 0;
}





