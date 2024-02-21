#include <stdio.h> 
#include <string.h>

struct info {
	int num;
	char name[200];
};

void sort1(struct info data [], int size) {
	for (int i = 0; i < size - 1; i++) {
		for (int j = 0; j < size -i - 1; j++) {
			if(data[j].num > data[j+1].num) {
				struct info temp = data[j];
				data[j] = data[j+1];
				data[j+1] = temp;
			}
		}
	}
}

void sort2(struct info data[], int size) {
	for (int i = 0; i < size -1; i++) {
		for (int j = 0; j < size - i - 1; j++) {
			if(strcmp(data[j].name, data[j+1].name) > 0) {
				struct info temp = data[j];
				data[j] = data[j+1];
				data[j+1] = temp;
			}
		}
	}
}

int main () {
	
	int num;
	int opt;
	

	printf("1/2?\n");
	scanf("%d", &opt);
	
	if(opt == 1) {
		printf("Sort by Num\n");
		scanf("%d", &num);
		
		struct info data [num];
		
		for (int i = 0; i < num; i++) {
			scanf("%s %d", data[i].name, &data[i].num);
		}
		
		sort1(data, num);
		
		for (int j = 0; j < num; j++) {
			printf("%s %d\n", data[j].name, data[j].num);
		
		}
	}
	
	else {
		printf("sort by char\n");
		scanf("%d", &num);
		
		struct info data[num];
		for (int i = 0; i < num; i++) {
			scanf("%s %d", data[i].name, &data[i].num);
			
		}
		
		sort2(data, num);
				
		for (int j = 0; j < num; j++) {
			printf("%s %d\n", data[j].name, data[j].num);
		}
	}
	
	
	return 0;
}
