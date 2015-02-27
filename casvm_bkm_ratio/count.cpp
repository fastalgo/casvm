#include<stdio.h>
int main(){
	int correct=0;
	int total=0;
	int i,j;
	FILE * accuracyfile = fopen("accuracyfile", "r");
	while(fscanf(accuracyfile, "%d %d", &i, &j)!=EOF){
		correct+=i;
		total+=j;
	}
	printf("\n****************************************************\n\n");
	printf("Total Accuracy = %g%% (%d/%d) (classification)\n", (double)correct/total*100,correct,total);
	printf("\n****************************************************\n");
}
