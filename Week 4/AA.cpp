#include <stdio.h>

int main() {
    int N [2001];
    scanf ("%d", &N); 
    int position = 0;

    for (int i = 0; i < N; i++) {
    	
        int roll;
        scanf("%d", &roll);            

        position += roll;
    
    if (position == 30) {
    	position =  10;
	}
    else if (position == 12) {
    	position = 28;
	}
	else if (position == 35) {
    	position = 7;
	}
//	else if (position == 53) {
//    	position = 37;
//	}
//	else if (position == 80) {
//    	position = 59;
//	}
//	else if (position == 97) {
//    	position = 88;
//	}
}

printf ("%d\n", position);
	return 0;
}

