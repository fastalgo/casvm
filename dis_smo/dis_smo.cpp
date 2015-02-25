#include "mpi.h"
#include <sys/time.h>   
#include <stdio.h>   
#include <math.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <errno.h>
#include <string>
#include <sstream>
using namespace std;

#define DataType float
#define INF HUGE_VAL
#define VectorLength 16
#define MY_THREADS 12
#define ScheduleState static
#define ALIGNLENGTH 128
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

struct Kernel_params{
	DataType gamma;
	DataType coef0;
	int degree;
	DataType b;
	std::string kernel_type;
};
Kernel_params kp;
DataType parameterA = -0.125;
DataType parameterB = 1.0;
DataType parameterC = 3.0;
DataType cost = 1.0;
DataType tolerance = 1e-3;
DataType epsilon = 1e-5; 
DataType cEpsilon;

int numtasks, rank, len, rc, sub_element, level=0, gap;
int nPoints, sub_nPoints;
int * send_counts;
int * displs;

static char *line = NULL;
static int max_line_len;
char* inputalpha = NULL;
char* outputalpha = NULL;
char* outputsvs = NULL;

DataType* data;
DataType* labels;
int* data_index;
int* row_index;
DataType* alpha;

DataType* whole_data; 
DataType* whole_labels; 
int* whole_data_index; 
int* whole_row_index;
int max_index;

MPI_Comm double_comm;

enum KernelType {LINEAR, POLYNOMIAL, GAUSSIAN, SIGMOID};

static DataType selfKernel() {
	return 1.0;//only for GAUSSIAN kernel
}

static DataType kernel(int a, int b, DataType* data, int* row_index, int* data_index) {
	//printf("start kernel\n");
    	int i,j;
    	DataType accumulant=0;
    	for(i=row_index[a];i<row_index[a+1];i++){
    		accumulant+=(data[i]*data[i]);
	}
    	for(j=row_index[b];j<row_index[b+1];j++){
    		accumulant+=(data[j]*data[j]);
	}
    	i=row_index[a];
    	j=row_index[b];
    	while(i<row_index[a+1]&&j<row_index[b+1]){
		if(data_index[i]==data_index[j]){
			accumulant-=(2*data[i]*data[j]);
			i++;
			j++;
	    	}else if(data_index[i]>data_index[j]){
		    	j++;
	    	}else{
		    	i++;
	    	}
    	}    
    	//printf("end kernel\n");
    	return exp(parameterA * accumulant);
}  

void performTraining(){
	int pospoints = 0; 
	int negpoints = 0;
        for(int i=0;i<nPoints;i++){
                if((int)whole_labels[i]==1){   
			pospoints++;
		}
                else if((int)whole_labels[i]==-1){ 
			negpoints++;
		}
        }

	DataType * global_high_low;
	DataType alphaHighDiff, alphaLowDiff, bLow, bHigh, alphaHighOld, alphaLowOld, eta;
	int iLow, iHigh;
	int kType = GAUSSIAN;
	if (kp.kernel_type.compare(0,3,"rbf") == 0) {
		parameterA = -kp.gamma;
		kType = GAUSSIAN;
		//printf("Gaussian kernel: gamma = %f\n", -parameterA);
	} 

	if(rank==0){	
		printf("--rank %d, Gaussian kernel: gamma = %f, Cost: %f, Tolerance: %f, Epsilon: %f, #samples: %d (%d positve, %d negtive), #sub_samples: %d, max_index: %d\n", rank, -parameterA, cost, tolerance, epsilon, nPoints, pospoints, negpoints, sub_nPoints, max_index);
	}

	//cEpsilon = cost - epsilon; 
	DataType* devKernelDiag;
	DataType* devF;
	
	if(rank==0){
		devKernelDiag = (DataType*)malloc(sizeof(DataType) * nPoints);
		devF = (DataType*)malloc(sizeof(DataType) * nPoints);
		global_high_low = (DataType *)malloc(sizeof(DataType) * 4 * numtasks);
	}	

	DataType* sub_devF = (DataType*)malloc(sizeof(DataType) * sub_nPoints);
	//DataType* tempF = (DataType*)malloc(sizeof(DataType) * sub_nPoints);
	DataType* sub_highKernel = (DataType*)malloc(sizeof(DataType) * sub_nPoints);
	DataType* sub_lowKernel = (DataType*)malloc(sizeof(DataType) * sub_nPoints);
	DataType* high_x = (DataType*)malloc(sizeof(DataType) * max_index);
	DataType* low_x = (DataType*)malloc(sizeof(DataType) * max_index);

	//printf("rank %d, Initialization for devKernelDiag and sub_devF\n", rank);
	MPI_Barrier(MPI_COMM_WORLD);
	
	if(rank==0){	
		for (int index = 0;index < nPoints;index++) {
			devKernelDiag[index] = selfKernel();
			devF[index] = -whole_labels[index];
		}   
	}
	printf("rank %d, Initialization complete\n", rank);
	MPI_Barrier(MPI_COMM_WORLD);

	if(rank==0){
		bLow = 1;
	  	bHigh = -1;
	  	iLow = -1;
	  	iHigh = -1;
	  	for (int i = 0; i < nPoints; i++) {
		    	if (whole_labels[i] < 0) {
		      		if (iLow == -1) {
					iLow = i;
					if (iHigh > -1) {
			  		i = nPoints; //Terminate
					}
		      		}
		    	} else {
		      		if (iHigh == -1) {
					iHigh = i;
					if (iLow > -1) {
			  			i = nPoints; //Terminate
					}
		      		}
		    	}
	  	}
		eta = devKernelDiag[iHigh] + devKernelDiag[iLow];
		eta = eta - 2*kernel(iHigh, iLow, whole_data, whole_row_index, whole_data_index);
		alphaLowOld = alpha[iLow];
		alphaHighOld = alpha[iHigh];  
		//And we know eta > 0
		DataType alphaLowNew = 2/eta; //Just boil down the algebra
		if (alphaLowNew > cost) {
			alphaLowNew = cost;
		}
		//alphaHighNew == alphaLowNew for the first step
		alpha[iLow] = alphaLowNew;
		alpha[iHigh] = alphaLowNew;
		alphaLowDiff = alpha[iLow] - alphaLowOld;
  		alphaHighDiff = -whole_labels[iHigh] * whole_labels[iLow] * alphaLowDiff;	
	}
	//printf("rank %d, normal 01\n", rank);
	//MPI_Barrier(MPI_COMM_WORLD);

	MPI_Bcast(&iLow, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&iHigh, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bLow, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bHigh, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&alphaLowDiff, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&alphaHighDiff, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(alpha+iLow, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(alpha+iHigh, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Scatter (devF, sub_nPoints, MPI_FLOAT, sub_devF, sub_nPoints, MPI_FLOAT, 0, MPI_COMM_WORLD);
	if(rank==0){	
		free(devF);
	}

	//printf("rank %d, normal 02\n", rank);
	//MPI_Barrier(MPI_COMM_WORLD);

	int iteration = 1;
	DataType high_devF, low_devF;
	DataType high_low[4];
	
	//printf("rank %d, normal 03\n", rank);
	//MPI_Barrier(MPI_COMM_WORLD);
	  
	struct timeval start,finish;
	gettimeofday(&start, 0);	
 
	while(bLow > bHigh + 2*tolerance){	
		if(rank==0){
			for(int i=0;i<max_index;i++){	
				high_x[i] = low_x[i] = 0;
			}
			for(int i=whole_row_index[iHigh];i<whole_row_index[iHigh+1];i++){
				high_x[whole_data_index[i]-1] = whole_data[i];
			}
			for(int i=whole_row_index[iLow];i<whole_row_index[iLow+1];i++){
				low_x[whole_data_index[i]-1] = whole_data[i];
			}
	  	}  
	  	MPI_Bcast(high_x, max_index, MPI_FLOAT, 0, MPI_COMM_WORLD);
	  	MPI_Bcast(low_x, max_index, MPI_FLOAT, 0, MPI_COMM_WORLD);
	  	MPI_Barrier(MPI_COMM_WORLD);
	  	/*if(rank==2&&iteration<=1){
			printf("\n*******************high_x*******************\n\n");
			for(int i=0;i<max_index;i++){	
				if(high_x[i]!=0) printf("%d:%f ", i+1, high_x[i]);
			}	
			printf("\n\n*******************high_x*******************\n\n");
	  	}
	  	if(rank==2&&iteration<=1){
			printf("\n*******************low_x*******************\n\n");
			for(int i=0;i<max_index;i++){
				if(low_x[i]!=0) printf("%d:%f ", i+1, low_x[i]);
			}	
			printf("\n\n*******************low_x*******************\n\n");
	  	}*/
	  	MPI_Barrier(MPI_COMM_WORLD);
	  	//printf("rank %d, normal 04-01\n", rank);
	  	//#pragma omp parallel for schedule(ScheduleState)
	  	//#pragma vector always aligned //#pragma simd //vectorlength(VectorLength) //num_threads(MY_THREADS)  
	  	for(int i=0;i<sub_nPoints;i++){
			DataType temp = 0;
			for(int j=row_index[i];j<row_index[i+1];j++){
				temp += (data[j]*data[j] - 2*data[j]*high_x[data_index[j]-1]);
				//if(rank==0&&i==0&&iteration<=1) printf("%d:%f ", data_index[j], data[j]);
			}
			for(int j=0;j<max_index;j++){	
				temp += high_x[j]*high_x[j];
			}
			sub_highKernel[i] = exp(parameterA * temp);

			temp = 0;
			for(int j=row_index[i];j<row_index[i+1];j++){	
				temp += (data[j]*data[j] - 2*data[j]*low_x[data_index[j]-1]);
			}
			for(int j=0;j<max_index;j++){	
				temp += low_x[j]*low_x[j];
			}
			sub_lowKernel[i] = exp(parameterA * temp);
			sub_devF[i] = sub_devF[i] + alphaHighDiff * whole_labels[iHigh] * sub_highKernel[i] + alphaLowDiff * whole_labels[iLow] * sub_lowKernel[i];

	  	}
		int offsets = rank*sub_nPoints;
		high_devF = INF; 
		low_devF = -INF;
	 	//printf("rank %d, normal 04-02\n", rank);
	 	//MPI_Barrier(MPI_COMM_WORLD);  

		for(int i=0;i<sub_nPoints;i++){
			if((high_devF>sub_devF[i])&&(((whole_labels[i+offsets] > 0) && (alpha[i+offsets] < cEpsilon)) || ((whole_labels[i+offsets] < 0) && (alpha[i+offsets] > epsilon)))){
				high_devF=sub_devF[i]; 
			  	iHigh=i;
			}
		  	if((low_devF<sub_devF[i])&&(((whole_labels[i+offsets] > 0) && (alpha[i+offsets] > epsilon)) || ((whole_labels[i+offsets] < 0) && (alpha[i+offsets] < cEpsilon)))){
			  	low_devF=sub_devF[i]; 
			  	iLow=i;
			}
		}
		iHigh = offsets+iHigh;
		iLow = offsets+iLow;
		high_low[0] = high_devF;
		high_low[1] = iHigh;
		high_low[2] = low_devF;
		high_low[3] = iLow;
		MPI_Gather(high_low, 4, MPI_FLOAT, global_high_low, 4, MPI_FLOAT, 0, MPI_COMM_WORLD);

	 	//printf("rank %d, normal 04-03\n", rank);
	  	//MPI_Barrier(MPI_COMM_WORLD);  
	
		if(rank==0){	  
			bHigh = high_devF;
			for(int i=0;i<numtasks;i++){
				DataType temp_devF = global_high_low[4*i];
				if(temp_devF<bHigh){
					bHigh = temp_devF;
			  		iHigh = (int)global_high_low[4*i+1];
			  	}
		  	}

		  	bLow = low_devF;
		  	for(int i=0;i<numtasks;i++){
				DataType temp_devF = global_high_low[4*i+2];
			  	if(temp_devF>bLow){
					bLow = temp_devF;
			  		iLow = (int)global_high_low[4*i+3];
			  	}
		  	}
		  
		  	eta = devKernelDiag[iHigh] + devKernelDiag[iLow] - 2 * kernel(iHigh, iLow, whole_data, whole_row_index, whole_data_index);  
		  	alphaHighOld = alpha[iHigh];
		  	alphaLowOld = alpha[iLow];
		  	DataType alphaDiff = alphaLowOld - alphaHighOld;
		  	DataType lowLabel = whole_labels[iLow];
		  	DataType sign = whole_labels[iHigh] * lowLabel;
		  	DataType alphaLowUpperBound;
		  	DataType alphaLowLowerBound;
		  	if (sign < 0) {
	          		if (alphaDiff < 0) {
	              			alphaLowLowerBound = 0;
	              			alphaLowUpperBound = cost + alphaDiff;
	          		} else {
				  	alphaLowLowerBound = alphaDiff;
	              			alphaLowUpperBound = cost;
			  	}
	      		} else {
				DataType alphaSum = alphaLowOld + alphaHighOld;
				if (alphaSum < cost) {
					alphaLowUpperBound = alphaSum;
					alphaLowLowerBound = 0;
				} else {
					alphaLowLowerBound = alphaSum - cost;
					alphaLowUpperBound = cost;
	        		}
		  	}
	      		DataType alphaLowNew;
	      		if (eta > 0) {
				alphaLowNew = alphaLowOld + lowLabel*(bHigh - bLow)/eta;
				if (alphaLowNew < alphaLowLowerBound){ 
					alphaLowNew = alphaLowLowerBound;
				}
	        		else if (alphaLowNew > alphaLowUpperBound){ 
					alphaLowNew = alphaLowUpperBound;
				}
		  	} else {
				DataType slope = lowLabel * (bHigh - bLow);
				DataType delta = slope * (alphaLowUpperBound - alphaLowLowerBound);
				if (delta > 0) {
					if (slope > 0){	
						alphaLowNew = alphaLowUpperBound;
					}
					else{	
						alphaLowNew = alphaLowLowerBound;
					}
				} else{	
					alphaLowNew = alphaLowOld;
				}
		  	}
	      		alphaLowDiff = alphaLowNew - alphaLowOld;
	      		alphaHighDiff = -sign*(alphaLowDiff);
	      		alpha[iLow] = alphaLowNew;
	      		alpha[iHigh] = (alphaHighOld + alphaHighDiff);
		}
	 	//printf("rank %d, normal 04-04\n", rank);
	  	//MPI_Barrier(MPI_COMM_WORLD);  
		MPI_Bcast(&iLow, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&iHigh, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&bLow, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&bHigh, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&alphaLowDiff, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&alphaHighDiff, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast(alpha+iLow, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast(alpha+iHigh, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
		iteration++;
		//printf("rank %d, normal 04-05\n", rank);
	  	//MPI_Barrier(MPI_COMM_WORLD);  
		//printf("iLow: %d, bLow: %lf; iHigh: %d, bHigh: %lf\n",iLow,bLow,iHigh,bHigh);

	}
  	gettimeofday(&finish, 0);
  	DataType trainingTime = (DataType)(finish.tv_sec - start.tv_sec) + ((DataType)(finish.tv_usec - start.tv_usec)) * 1e-6;
  	//FILE * sparse_time = fopen("sparse_time","a+");
  	//fprintf(sparse_time,"%f\n", trainingTime);	
  	//printf("Training time : %f seconds\n", trainingTime);
  	//printf("--- %d iterations ---", iteration);
  	//printf("bLow: %f, bHigh: %f\t", bLow, bHigh);
  	kp.b = (bLow + bHigh) / 2;
  	printf("****** rank: %d, Time : %f s, #iter: %d, bLow: %f, bHigh: %f, b: %f ******\n", rank, trainingTime, iteration, bLow, bHigh, kp.b);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

static char* readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL){
		return NULL;
	}

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void printModel(const char* outputFile, DataType* data, int* data_index, int* row_index, DataType* labels, DataType* alpha, int nPoints) { 
	printf("Output File: %s\n", outputFile);
	FILE* outputFilePointer = fopen(outputFile, "w");
	if (outputFilePointer == NULL) {
		printf("Can't write %s\n", outputFile);
		exit(1);
	}
	int nSV = 0;
	int pSV = 0;
	for(int i = 0; i < nPoints; i++) {
		if (alpha[i] > epsilon) {
			if (labels[i] > 0) {
				pSV++;
			} else {
				nSV++;
			}
		}
	}
  	bool printGamma = false;
  	bool printCoef0 = false;
  	bool printDegree = false;
  	const char* kernelType = kp.kernel_type.c_str();
  	if (strncmp(kernelType, "polynomial", 10) == 0) {
    		printGamma = true;
    		printCoef0 = true;
    		printDegree = true;
  	} else if (strncmp(kernelType, "rbf", 3) == 0) {
    		printGamma = true;
  	} else if (strncmp(kernelType, "sigmoid", 7) == 0) {
    		printGamma = true;
    		printCoef0 = true;
  	}
	
	fprintf(outputFilePointer, "svm_type c_svc\n");
	fprintf(outputFilePointer, "kernel_type %s\n", kp.kernel_type.c_str());
  	if (printDegree) {
    		fprintf(outputFilePointer, "degree %i\n", kp.degree);
  	}
  	if (printGamma) {
    		fprintf(outputFilePointer, "gamma %f\n", kp.gamma);
  	}
  	if (printCoef0) {
    		fprintf(outputFilePointer, "coef0 %f\n", kp.coef0);
  	}
	fprintf(outputFilePointer, "nr_class 2\n");
	fprintf(outputFilePointer, "total_sv %d\n", nSV + pSV);
	fprintf(outputFilePointer, "rho %.10f\n", kp.b);
	fprintf(outputFilePointer, "label 1 -1\n");
	fprintf(outputFilePointer, "nr_sv %d %d\n", pSV, nSV);
	fprintf(outputFilePointer, "SV\n");
	//FILE * out_alpha = fopen(outputalpha, "w");
	//FILE * out_svs = fopen(outputsvs, "w");
	int nSVs = 0;
	for (int i = 0; i < nPoints; i++) {
		//fprintf(out_alpha, "%f\n", alpha[i]);
		if (alpha[i] > epsilon) {
			nSVs++;
			//fprintf(out_alpha, "%f\n", alpha[i]);
			fprintf(outputFilePointer, "%.10f ", labels[i]*alpha[i]);
			//fprintf(out_svs, "%d ", (int)labels[i]);
			for (int j = row_index[i]; j < row_index[i+1]; j++){
				fprintf(outputFilePointer, "%d:%.10f ", data_index[j], data[j]);
				//fprintf(out_svs, "%d:%.10f ", data_index[j], data[j]);
			}				
			fprintf(outputFilePointer, "\n");
			//fprintf(out_svs, "\n");
		}
	}
	fclose(outputFilePointer);
	//fclose(out_alpha);
	//fclose(out_svs);
	printf("\n---------number of Support Vectors: %d---------\n\n", nSVs);
}

void printHelp() {
	printf("Usage: dismo [options] trainingData\n");
  	printf("Options:\n");
  	printf("\t-o outputFilename\t Location of output file\n");
  	printf("Kernel types: (only support gaussian kernel for current version, please ignore this option!!!)\n");
  	printf("\t--gaussian\tGaussian or RBF kernel (default): Phi(x, y; gamma) = exp{-gamma*||x-y||^2}\n");
  	printf("\t--linear\tLinear kernel: Phi(x, y) = x . y\n");
  	printf("\t--polynomial\tPolynomial kernel: Phi(x, y; a, r, d) = (ax . y + r)^d\n");
  	printf("\t--sigmoid\tSigmoid kernel: Phi(x, y; a, r) = tanh(ax . y + r)\n");
  	printf("Parameters:\n");
  	printf("\t-c, --cost\tSVM training cost C (default = 1)\n");
  	printf("\t-g\tGamma for Gaussian kernel (default = 1/nDimension)\n");
  	printf("\t-a\tParameter a for Polynomial and Sigmoid kernels (default = 1/l)\n");
  	printf("\t-r\tParameter r for Polynomial and Sigmoid kernels (default = 1)\n");
  	printf("\t-d\tParameter d for Polynomial kernel (default = 3)\n");
  	printf("Convergence parameters:\n");
  	printf("\t--tolerance, -t\tTermination criterion tolerance (default = 0.001)\n");
  	printf("\t--epsilon, -e\tSupport vector threshold (default = 1e-5)\n");
}

static int kType = GAUSSIAN;

int main(int argc, char** argv)  {   
	MPI_Status Stat;
	char hostname[MPI_MAX_PROCESSOR_NAME];
	rc = MPI_Init(&argc,&argv);
	if (rc != MPI_SUCCESS) {
    		printf ("Error starting MPI program. Terminating.\n");
    		MPI_Abort(MPI_COMM_WORLD, rc);
    	}   
    	MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
    	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    	MPI_Get_processor_name(hostname, &len);	  	
    	//printf ("Number of tasks= %d My rank= %d Running on %s\n", numtasks,rank,hostname);
	int currentOption;
  	bool parameterASet = false;
  	bool parameterBSet = false;
  	bool parameterCSet = false;  
  	char* outputFilename = NULL; 
  	//char* inputalpha = NULL;
  	//char* outputalpha = NULL; 
  	
  	while (1) {
    		static struct option longOptions[] = {
      			{"gaussian", no_argument, &kType, GAUSSIAN},
      			{"polynomial", no_argument, &kType, POLYNOMIAL},
      			{"sigmoid", no_argument, &kType, SIGMOID},
      			{"linear", no_argument, &kType, LINEAR},
      			{"cost", required_argument, 0, 'c'},
      			{"tolerance", required_argument, 0, 't'},
      			{"epsilon", required_argument, 0, 'e'},
      			{"output", required_argument, 0, 'o'},
      			{"help", no_argument, 0, 'f'}
    		};
    		int optionIndex = 0;
    		currentOption = getopt_long(argc, (char *const*)argv, "c:t:e:o:p:q:s:a:r:d:g:f", longOptions, &optionIndex);
    		if (currentOption == -1) {
      			break;
    		}
    		int method = 3;
    		switch (currentOption) {
    			case 0:
      				break;
    			case 'v':
      				printf("Distributed SMO version: 1.0 @ Feb 24, 2015\n");
      				return(0);
    			case 'f':
      				printHelp();
      				return(0);
    			case 'c':
      				sscanf(optarg, "%f", &cost);
      				break;
    			case 't':
      				sscanf(optarg, "%f", &tolerance);
      				break;
    			case 'e':
      				sscanf(optarg, "%f", &epsilon);
      				break;
    			case 'o':
      				outputFilename = (char*)malloc(strlen(optarg));
      				strcpy(outputFilename, optarg);
      				break;
    			case 'p':
      				inputalpha = (char*)malloc(strlen(optarg));
      				strcpy(inputalpha, optarg);
      				break;
    			case 'q':
      				outputalpha = (char*)malloc(strlen(optarg));
      				strcpy(outputalpha, optarg);
      				break;
    			case 's':
      				outputsvs = (char*)malloc(strlen(optarg));
      				strcpy(outputsvs, optarg);
      				break;
    			case 'a':
      				sscanf(optarg, "%f", &parameterA);
      				parameterASet = true;
      				break;
    			case 'r':
      				sscanf(optarg, "%f", &parameterB);
      				parameterBSet = true;
      				break;
    			case 'd':
      				sscanf(optarg, "%f", &parameterC);
      				parameterCSet = true;
      				break;
    			case 'g':
      				sscanf(optarg, "%f", &parameterA);
      				parameterA = -parameterA;
      				parameterASet = true;
      				break;
    			case '?':
      				break;
    			default:
      				abort();
      				break;
    		}
  	}

  	if (optind != argc - 1) {
    		printHelp();
    		return(0);
	}

	const char* trainingFilename = argv[optind];  
  
  	if (outputFilename == NULL) {
    		int inputNameLength = strlen(trainingFilename);
    		outputFilename = (char*)malloc(sizeof(char)*(inputNameLength + 5));
    		strncpy(outputFilename, trainingFilename, inputNameLength + 4);
    		char* period = strrchr(outputFilename, '.');
    		if (period == NULL) {
      			period = outputFilename + inputNameLength;
    		}
    		strncpy(period, ".mdl\0", 5);
  	}  	

	if(rank==0){
		int inst_max_index, i, j;
		FILE *fp = fopen(trainingFilename,"r");
		char *endptr;
		char *idx, *val, *label;

		if(fp == NULL)
		{
			fprintf(stderr,"can't open input file %s\n",trainingFilename);
			exit(1);
		}

		nPoints = 0;
		int elements = 0;

		max_line_len = 1024;
		line = Malloc(char,max_line_len);
		while(readline(fp)!=NULL)
		{
			char *p = strtok(line," \t"); // label
			// features
			while(1)
			{
				p = strtok(NULL," \t");
				if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
					break;
				++elements;
			}
			//++elements;
			++nPoints;
		}
		rewind(fp);
		whole_labels = (DataType *)malloc(nPoints*sizeof(DataType));
		whole_row_index = (int *)malloc((nPoints+1)*sizeof(int));	
		whole_data = (DataType *)malloc(elements*sizeof(DataType));
		whole_data_index = (int *)malloc(elements*sizeof(int));

		max_index = 0;
		j=0;
		for(i=0;i<nPoints;i++)
		{
			inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
			readline(fp);
			//prob.x[i] = &x_space[j];
			whole_row_index[i]=j;
			label = strtok(line," \t\n");
			if(label == NULL){ // empty line
				exit_input_error(i+1);
			}
			//prob.y[i] = strtod(label,&endptr);
			whole_labels[i] = strtod(label,&endptr);
			if(endptr == label || *endptr != '\0'){
				exit_input_error(i+1);
			}

			while(1)
			{
				idx = strtok(NULL,":");
				val = strtok(NULL," \t");

				if(val == NULL){
					break;
				}

				errno = 0;
				//x_space[j].index = (int) strtol(idx,&endptr,10);
				whole_data_index[j] = (int) strtol(idx,&endptr,10);
				//if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				if(endptr == idx || errno != 0 || *endptr != '\0' || whole_data_index[j] <= inst_max_index){
					exit_input_error(i+1);
				}
				else{
					//inst_max_index = x_space[j].index;
					inst_max_index = whole_data_index[j];
				}

				errno = 0;
				//x_space[j].value = strtod(val,&endptr);
				whole_data[j] = strtod(val,&endptr);
				if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr))){
					exit_input_error(i+1);
				}

				++j;
			}

			if(inst_max_index > max_index){
				max_index = inst_max_index;
			}
			//x_space[j++].index = -1;
		}	
		whole_row_index[i]=elements;
		//if(param.gamma == 0 && max_index > 0)
		//	param.gamma = 1.0/max_index;		
		fclose(fp);
		/*
		FILE * verify_input =fopen("verify_input","w");
		for(i=0;i<nPoints;i++){
			fprintf(verify_input,"%d ",(int)labels[i]);
			for(j=row_index[i];j<row_index[i+1];j++)
				fprintf(verify_input,"%d:%f ",data_index[j],data[j]);
			fprintf(verify_input,"\n");
		}
		*/
		//printf("\n\nInput data found: %d points, %d is the maximum dimension\n", nPoints, max_index); 
	}

	MPI_Bcast(&nPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&max_index, 1, MPI_INT, 0, MPI_COMM_WORLD);

	//MPI_Barrier(MPI_COMM_WORLD);

        //double partition_start = MPI_Wtime();
	
	sub_nPoints = nPoints/numtasks;

	//printf("\n*********on node %d, sub_nPoints is %d\n", rank, sub_nPoints);

	alpha = (DataType*)calloc(nPoints, sizeof(DataType));

	cEpsilon = cost - epsilon; 

	//get_data_size(sub_nPoints);
	if(rank==0){
		send_counts = (int *)malloc(numtasks*sizeof(int));
		displs = (int *)malloc(numtasks*sizeof(int));
		for(int i=0;i<numtasks;i++){
			send_counts[i] = whole_row_index[(i+1)*sub_nPoints] - whole_row_index[i*sub_nPoints];
			//printf("@@@@@@@@@@@@@@@row_index[%d] = %d, row_index[%d] = %d, send_counts[%d] = %d\n", (i+1)*sub_nPoints, row_index[(i+1)*sub_nPoints], i*sub_nPoints, row_index[i*sub_nPoints], i, send_counts[i]);
			if(i==0){	
				displs[i] = 0;
			}
		   	else{ 
				displs[i] = displs[i-1]+send_counts[i-1];
			}
		}
	}
	MPI_Scatter(send_counts, 1, MPI_INT, &sub_element, 1, MPI_INT, 0, MPI_COMM_WORLD);

	printf("\n^^^^^^^^^^on node %d, sub_nPoints is %d, sub_element is %d\n", rank, sub_nPoints, sub_element);

	data = (DataType*)malloc(sub_element*sizeof(DataType));

	if(rank!=0){	
		whole_labels = (DataType*)malloc(nPoints*sizeof(DataType));
	}

	data_index = (int*)malloc(sub_element*sizeof(int));

	row_index = (int*)malloc((sub_nPoints+1)*sizeof(int)); 

	MPI_Scatterv(whole_data, send_counts, displs, MPI_FLOAT, data, sub_element, MPI_FLOAT, 0, MPI_COMM_WORLD);

	MPI_Scatterv(whole_data_index, send_counts, displs, MPI_INT, data_index, sub_element, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Scatter(whole_row_index, sub_nPoints, MPI_INT, row_index, sub_nPoints, MPI_INT, 0, MPI_COMM_WORLD);

	//MPI_Scatter(whole_labels, sub_nPoints, MPI_FLOAT, labels, sub_nPoints, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast (whole_labels, nPoints, MPI_FLOAT, 0, MPI_COMM_WORLD);

	//printf("\n###########on node %d, after data distribution\n", rank);

	MPI_Barrier(MPI_COMM_WORLD);

        double partition_start = MPI_Wtime();


	/*if(rank==3)
	for(int i=0;i<=sub_nPoints;i++)
		printf("*********%d\n", row_index[i]);*/
		
	
	for(int i=1;i<sub_nPoints;i++){
		row_index[i] = (row_index[i] - row_index[0]);
	}

	row_index[0] = 0;

	row_index[sub_nPoints] = sub_element;
  		
  	if(rank==0){
	  	//free(whole_data);
	  	//free(whole_data_index);
	  	//free(whole_row_index);
	  	free(send_counts);
	  	free(displs);
	  	//free(whole_labels);
	}

	if(inputalpha!=NULL){
		//printf("**********start input alpha**********\n");
		FILE * alpha_in = fopen(inputalpha, "r");
		DataType in_temp;
		int i = 0;
		while(fscanf(alpha_in, "%f", &in_temp)!=EOF){
			alpha[i++] = in_temp;
		}
	}else{
		//printf("**********no input alpha**********\n");
	}
	/*if(rank==2)
	for(int i=0;i<sub_nPoints;i++){
		printf("%d ",(int)labels[i]);
		for(int j=row_index[i];j<row_index[i+1];j++){
			printf("%d:%f ",data_index[j],data[j]);	
		}
		printf("\n\n");	
	}*/
  
  	if (kType == LINEAR) {
    		printf("Linear kernel\n");
    		kp.kernel_type = "linear";
  	} else if (kType == POLYNOMIAL) {
    		if (!(parameterCSet)) {
      			parameterC = 3.0f;
    		}
    		if (!(parameterASet)) {
      			//parameterA = 1.0/nPoints;
	  		parameterA = 1.0/max_index;
    		}
    		if (!(parameterBSet)) {
      			parameterB = 0.0f;
    		}
    		printf("Polynomial kernel: a = %f, r = %f, d = %f\n", parameterA, parameterB, parameterC);
    		if ((parameterA <= 0) || (parameterB < 0) || (parameterC < 1.0)) {
      			printf("Invalid parameters\n");
      			exit(1);
    		}
    		kp.kernel_type = "polynomial";
    		kp.gamma = parameterA;
    		kp.coef0 = parameterB;
    		kp.degree = (int)parameterC;
  	} else if (kType == GAUSSIAN) {
    		if (!(parameterASet)) {
      			//parameterA = 1.0/nPoints;
	  		parameterA = 1.0/max_index;
    		} else {
      			parameterA = -parameterA;
    		}
    		//printf("Gaussian kernel: gamma = %f\n", parameterA);
    		if (parameterA < 0) {
      			printf("Invalid parameters\n");
      			exit(1);
    		}
    		kp.kernel_type = "rbf";
    		kp.gamma = parameterA;
  	} else if (kType == SIGMOID) {
    		if (!(parameterASet)) {
      			//parameterA = 1.0/nPoints;
	  		parameterA = 1.0/max_index;
    		}
    		if (!(parameterBSet)) {
      			parameterB = 0.0f;
    		}
    		//printf("Sigmoid kernel: a = %f, r = %f\n", parameterA, parameterB);
    		if ((parameterA <= 0) || (parameterB < 0)) {
      			printf("Invalid Parameters\n");
      			exit(1);
    		}
    		kp.kernel_type = "sigmoid";
    		kp.gamma = parameterA;
    		kp.coef0 = parameterB;
  	}
	MPI_Barrier(MPI_COMM_WORLD);
	double smo_start = MPI_Wtime();
	performTraining();
	MPI_Barrier(MPI_COMM_WORLD);
	double smo_end = MPI_Wtime();

	if(rank==0)	printf("\n******************total time: %lf, partition time: %lf, smo time: %lf********************\n\n", (smo_end-partition_start), (smo_start-partition_start), (smo_end-smo_start));

	MPI_Barrier(MPI_COMM_WORLD);
	if(rank==0)	printModel(outputFilename, whole_data, whole_data_index, whole_row_index, whole_labels, alpha, nPoints);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize(); 
	return 0;
}
