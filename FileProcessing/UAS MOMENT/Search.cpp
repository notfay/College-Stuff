// Online C compiler to run C program online
#include <stdio.h>

int linearSearch(int array[], int key, int size){
    for(int i = 0; i < size; i++){
        if (array[i] == key){
            return i;
        }
    }
    return -1;
}

int binarySearch(int array[], int key, int min, int max){
    int mid = (min+max)/2;
    
    if (min > max) return -1;
    
    if (array[mid] == key){
        return mid;
    }
    else if (array[mid]<key){
        return binarySearch(array, key, mid+1, max);
    }
    else if (array[mid]>key){
        return binarySearch(array, key, min, mid-1);
    }
}




int main() {
    int banyak;
    scanf("%d", &banyak);
    
    int arr[banyak];
    
    for (int i = 0; i < banyak; i++){
        scanf("%d", &arr[i]);
    }
    
    int k;
    scanf("%d", &k);

    int hasil = interSearch(arr, k, 0, banyak-1);
    printf("%d", hasil);
    
}
