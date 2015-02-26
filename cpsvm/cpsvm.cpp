#include "mpi.h"
#include <sys/time.h>   
#include <stdio.h>   
#include <math.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include <string>
#include <sstream>
#include <assert.h>
using namespace std;

#define DataType float
#define INF HUGE_VAL
#define PHIDEV 1
#define VectorLength 16
#define MY_THREADS 12
#define ScheduleState static
#define ALIGNLENGTH 128
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define kmeans_threshold 0.001

struct Kernel_params
{
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

int numtasks, rank, len, rc, sub_element, level, gap, numClusters;
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
//DataType* clusters;

MPI_Comm double_comm;
//int myrank, mycolor;

enum KernelType {LINEAR, POLYNOMIAL, GAUSSIAN, SIGMOID};

static DataType selfKernel() 
{
	return 1.0;
}

static DataType kernel(int a, int b) 
{
	//printf("start kernel\n");
    	int i,j;
    	DataType accumulant=0;
    	for(i=row_index[a];i<row_index[a+1];i++)
	{
    		accumulant+=(data[i]*data[i]);
	}
    	for(j=row_index[b];j<row_index[b+1];j++)
	{
    		accumulant+=(data[j]*data[j]);
	}
    	i=row_index[a];
    	j=row_index[b];

    	while(i<row_index[a+1]&&j<row_index[b+1])
	{
		if(data_index[i]==data_index[j])
		{
			accumulant-=(2*data[i]*data[j]);
			i++;
			j++;
	    	}
		else if(data_index[i]>data_index[j])
		{
			j++;
	    	}
		else
		{
		    	i++;
	    	}
    	}    
    	//printf("end kernel\n");
    	return exp(parameterA * accumulant);
}  

//void performTraining(DataType* data, int* data_index, int* row_index, DataType* labels, DataType* alpha, int nPoints){
void performTraining(int nPoints)
{
	int pospoints = 0; 
	int negpoints = 0;
	for(int i=0;i<nPoints;i++)
	{
		if((int)labels[i]==1)	
		{
			pospoints++;
		}
		else if((int)labels[i]==-1) 
		{
			negpoints++;
		}
	}

	/*if(level==2){
		MPI_Barrier(MPI_COMM_WORLD);
	}	
	if(level==2&&rank==3){
		FILE * out_test= fopen("ijcnn.part", "w");
		for(int i=0;i<nPoints;i++){
			fprintf(out_test, "%d ", (int)labels[i]);
			for(int j=row_index[i];j<row_index[i+1];j++){
				fprintf(out_test, "%d:%f ",data_index[j]+1,data[j]);	
			}
			fprintf(out_test, "\n");	
		}
		fclose(out_test);
	}*/
	if(level==2)
	{
		MPI_Barrier(MPI_COMM_WORLD);
	}	
	DataType alphaHighDiff, alphaLowDiff, bLow, bHigh, alphaHighOld, alphaLowOld, eta;
	int iLow, iHigh;
	int kType = GAUSSIAN;
	if (kp.kernel_type.compare(0,3,"rbf") == 0) 
	{
		parameterA = -kp.gamma;
		kType = GAUSSIAN;
		//printf("Gaussian kernel: gamma = %f\n", -parameterA);
	} 
	printf("--level: %d, rank %d, Gaussian kernel: gamma = %f, Cost: %f, Tolerance: %f, Epsilon: %f, #samples: %d (%d positve, %d negtive)\n", level, rank, -parameterA, cost, tolerance, epsilon, nPoints, pospoints, negpoints); 
	//cEpsilon = cost - epsilon; 
	
	DataType* devKernelDiag = (DataType*)malloc(sizeof(DataType) * nPoints);
	DataType* devF = (DataType*)malloc(sizeof(DataType) * nPoints);
	//DataType* tempF = (DataType*)malloc(sizeof(DataType) * nPoints);
	DataType* highKernel = (DataType*)malloc(sizeof(DataType) * nPoints);
	DataType* lowKernel = (DataType*)malloc(sizeof(DataType) * nPoints);

	printf("level: %d, rank %d, Initialization for devKernelDiag and devF\n", level, rank);
	
	for (int index = 0;index < nPoints;index++) 
	{
		devKernelDiag[index] = selfKernel();
		devF[index] = -labels[index];
	}   
	printf("level: %d, rank %d, Initialization complete\n", level, rank);

  	bLow = 1;
  	bHigh = -1;
  	iLow = -1;
  	iHigh = -1;

  	for (int i = 0; i < nPoints; i++) 
	{
    		if (labels[i] < 0) {
      			if (iLow == -1) 
			{
        			iLow = i;
        			if (iHigh > -1) 
				{
          				i = nPoints; //Terminate
        			}
      			}
    		} 
		else 
		{
      			if (iHigh == -1) 
			{
        			iHigh = i;
        			if (iLow > -1) 
				{
          				i = nPoints; //Terminate
        			}
      			}
    		}
  	} 
  	//launchTakeFirstStep(devKernelDiag);
  	printf("level: %d, rank %d, Finish First Step\n", level, rank);

  	eta = devKernelDiag[iHigh] + devKernelDiag[iLow];
	eta = eta - 2*kernel(iHigh, iLow);
	alphaLowOld = alpha[iLow];
	alphaHighOld = alpha[iHigh]; 
	//And we know eta > 0
	DataType alphaLowNew = 2/eta; //Just boil down the algebra
	if (alphaLowNew > cost) 
	{
		alphaLowNew = cost;
	}
	//alphaHighNew == alphaLowNew for the first step
	alpha[iLow] = alphaLowNew;
	alpha[iHigh] = alphaLowNew;	
  
  	alphaLowDiff = alpha[iLow] - alphaLowOld;
  	alphaHighDiff = -labels[iHigh] * labels[iLow] * alphaLowDiff;
  	//printf("rank %d, normal01\n", rank);
  	//printf("alphaLowDiff: %f\talphaHighDiff: %f\n",alphaLowDiff,alphaHighDiff);
  	//return;
  
  	printf("level: %d, rank: %d, Starting iterations\n", level, rank);
  	//MPI_Barrier(MPI_COMM_WORLD);
  	int iteration = 1;
  	DataType high_devF, low_devF;
  
  	struct timeval start,finish;
  	gettimeofday(&start, 0);	
  
	while(bLow > bHigh + 2*tolerance)
	{	  
		int i;	  
	  	//#pragma omp parallel for schedule(ScheduleState)
	  	for(i=0;i<nPoints;i++)
		{
			//#pragma vector always aligned //num_threads(MY_THREADS)
		  	highKernel[i] = kernel(i, iHigh);	
		  	lowKernel[i] = kernel(i, iLow);	
		  	devF[i] = devF[i] + alphaHighDiff * labels[iHigh] * highKernel[i] + alphaLowDiff * labels[iLow] * lowKernel[i];
	  	}
	  	//if(rank==3)	printf("level: %d, iter: %d, rank: %d, normal01\n", level, iteration, rank);
	  	//MPI_Barrier(MPI_COMM_WORLD);
	  	/*tempF[0:nPoints] = (((labels[0:nPoints] > 0) && (alpha[0:nPoints] < cEpsilon)) || ((labels[0:nPoints] < 0) && (alpha[0:nPoints] > epsilon))) ? devF[0:nPoints] : INF;
	  	iHigh = __sec_reduce_min_ind(tempF[0:nPoints]);
	  
	  	tempF[0:nPoints] = (((labels[0:nPoints] > 0) && (alpha[0:nPoints] > epsilon)) || ((labels[0:nPoints] < 0) && (alpha[0:nPoints] < cEpsilon))) ? devF[0:nPoints] : -INF;
	  	iLow = __sec_reduce_max_ind(tempF[0:nPoints]);*/
	  	high_devF = INF; 
	  	low_devF = -INF;
	  	//if(rank==3)	printf("level: %d, iter: %d, rank: %d, normal02\n", level, iteration, rank);
		//MPI_Barrier(MPI_COMM_WORLD);

	  	for(i=0;i<nPoints;i++)
		{
		  	if((high_devF>devF[i])&&(((labels[i] > 0) && (alpha[i] < cEpsilon)) || ((labels[i] < 0) && (alpha[i] > epsilon))))
			{
			  	high_devF=devF[i]; 
			  	iHigh=i;
			}
		  	if((low_devF<devF[i])&&(((labels[i] > 0) && (alpha[i] > epsilon)) || ((labels[i] < 0) && (alpha[i] < cEpsilon))))
			{
			  	low_devF=devF[i]; 
			  	iLow=i;
			}
	  	}
	  	//if(rank==3)	printf("level: %d, iter: %d, rank: %d, normal03\n", level, iteration, rank);
	  	//MPI_Barrier(MPI_COMM_WORLD);

	  	bHigh = devF[iHigh];
	  	bLow = devF[iLow];
	  	eta = devKernelDiag[iHigh] + devKernelDiag[iLow] - 2 * kernel(iHigh, iLow);  
	  	alphaHighOld = alpha[iHigh];
	  	alphaLowOld = alpha[iLow];
	  	DataType alphaDiff = alphaLowOld - alphaHighOld;
	  	DataType lowLabel = labels[iLow];
	  	DataType sign = labels[iHigh] * lowLabel;
	  	DataType alphaLowUpperBound;
	  	DataType alphaLowLowerBound;

	  	if (sign < 0) 
		{
          		if (alphaDiff < 0) 
			{
              			alphaLowLowerBound = 0;
              			alphaLowUpperBound = cost + alphaDiff;
          		} 
			else 
			{
				alphaLowLowerBound = alphaDiff;
              			alphaLowUpperBound = cost;
		  	}
      		} 
		else 
		{
			DataType alphaSum = alphaLowOld + alphaHighOld;
			if (alphaSum < cost) 
			{
				alphaLowUpperBound = alphaSum;
				alphaLowLowerBound = 0;
			} 
			else 
			{
				alphaLowLowerBound = alphaSum - cost;
				alphaLowUpperBound = cost;
        		}
	  	}

	  	//if(rank==3)	printf("level: %d, iter: %d, rank: %d, normal04\n", level, iteration, rank);
	  	//MPI_Barrier(MPI_COMM_WORLD);
      		DataType alphaLowNew;
      		if (eta > 0) 
		{
			alphaLowNew = alphaLowOld + lowLabel*(bHigh - bLow)/eta;
			if (alphaLowNew < alphaLowLowerBound) 
			{
				alphaLowNew = alphaLowLowerBound;
			}
        		else if (alphaLowNew > alphaLowUpperBound) 
			{
				alphaLowNew = alphaLowUpperBound;
			}
	  	} 
		else 
		{
			DataType slope = lowLabel * (bHigh - bLow);
			DataType delta = slope * (alphaLowUpperBound - alphaLowLowerBound);
			if (delta > 0) 
			{
				if (slope > 0)	
				{
					alphaLowNew = alphaLowUpperBound;
				}
				else	
				{
					alphaLowNew = alphaLowLowerBound;
				}
				
			} 
			else	
			{
				alphaLowNew = alphaLowOld;
			}
	  	}
    		alphaLowDiff = alphaLowNew - alphaLowOld;
    		alphaHighDiff = -sign*(alphaLowDiff);
    		alpha[iLow] = alphaLowNew;
    		alpha[iHigh] = (alphaHighOld + alphaHighDiff);
	  	//if(rank==3)	printf("level: %d, iter: %d, rank: %d, bLow: %f, bHigh: %f, normal05\n", level, iteration, rank, bLow, bHigh);
	  	//MPI_Barrier(MPI_COMM_WORLD);
	  	iteration++;
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
  	printf("level: %d, rank: %d, Time : %f s, #iter: %d, bLow: %f, bHigh: %f, b: %f\n", level, rank, trainingTime, iteration, bLow, bHigh, kp.b);
  	//MPI_Barrier(MPI_COMM_WORLD);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

static char* readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
	{
		return NULL;
	}

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
		{
			break;
		}
	}
	return line;
}

void printModel(const char* outputFile, DataType* data, int* data_index, int* row_index, DataType* labels, DataType* alpha, int nPoints) 
{ 
	printf("Output File of rank %d is %s\n", rank, outputFile);
	FILE* outputFilePointer = fopen(outputFile, "w");
	if (outputFilePointer == NULL) 
	{
		printf("Can't write %s\n", outputFile);
		exit(1);
	}
	int nSV = 0;
	int pSV = 0;
	for(int i = 0; i < nPoints; i++) 
	{
		if (alpha[i] > epsilon) 
		{
			if (labels[i] > 0) 
			{
				pSV++;
			} 
			else 
			{
				nSV++;
			}
		}
	}
  	bool printGamma = false;
  	bool printCoef0 = false;
  	bool printDegree = false;
  	const char* kernelType = kp.kernel_type.c_str();

  	if (strncmp(kernelType, "polynomial", 10) == 0) 
	{
    		printGamma = true;
    		printCoef0 = true;
    		printDegree = true;
  	} 
	else if (strncmp(kernelType, "rbf", 3) == 0) 
	{
    		printGamma = true;
  	} 
	else if (strncmp(kernelType, "sigmoid", 7) == 0) 
	{
    		printGamma = true;
    		printCoef0 = true;
  	}
	
	fprintf(outputFilePointer, "svm_type c_svc\n");
	fprintf(outputFilePointer, "kernel_type %s\n", kp.kernel_type.c_str());
  
	if (printDegree) 
	{
    		fprintf(outputFilePointer, "degree %i\n", kp.degree);
  	}
  	if (printGamma) 
	{
    		fprintf(outputFilePointer, "gamma %f\n", kp.gamma);
  	}
  	if (printCoef0) 
	{
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
	for (int i = 0; i < nPoints; i++) 
	{
		//fprintf(out_alpha, "%f\n", alpha[i]);
		if (alpha[i] > epsilon) 
		{
			nSVs++;
			//fprintf(out_alpha, "%f\n", alpha[i]);
			fprintf(outputFilePointer, "%.10f ", labels[i]*alpha[i]);
			//fprintf(out_svs, "%d ", (int)labels[i]);
			for (int j = row_index[i]; j < row_index[i+1]; j++)
			{
				fprintf(outputFilePointer, "%d:%.10f ", data_index[j]+1, data[j]);
				//fprintf(out_svs, "%d:%.10f ", data_index[j]+1, data[j]);
			}				
			fprintf(outputFilePointer, "\n");
			//fprintf(out_svs, "\n");
		}
	}
	fclose(outputFilePointer);
	//fclose(out_alpha);
	//fclose(out_svs);
	int total_svs=0;
	printf("number of Support Vectors on node %d is %d\n", rank, nSVs);
	MPI_Reduce(&nSVs, &total_svs, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	if(rank==0) 
	{
		printf("\n---------total number of Support Vectors on all nodes is %d---------\n\n", total_svs);
	}
}

void printHelp() {
        printf("Usage: dcsvm [options] trainingData\n");
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
        //printf("\t-a\tParameter a for Polynomial and Sigmoid kernels (default = 1/l)\n");
        //printf("\t-r\tParameter r for Polynomial and Sigmoid kernels (default = 1)\n");
        //printf("\t-d\tParameter d for Polynomial kernel (default = 3)\n");
        printf("Convergence parameters:\n");
        printf("\t--tolerance, -t\tTermination criterion tolerance (default = 0.001)\n");
        printf("\t--epsilon, -e\tSupport vector threshold (default = 1e-5)\n");
}

static int kType = GAUSSIAN;

float euclid_dist_2(int id, int max_index, float * center)
{
	float ans=0.0;

    	for(int i=0;i<max_index;i++)	
	{
		ans += center[i]*center[i]; 
	}

    	for(int i=row_index[id];i<row_index[id+1];i++)	
	{
		ans += data[i] * data[i];
	}

    	for(int i=row_index[id];i<row_index[id+1];i++)	
	{
		ans -= 2* data[i] * center[data_index[i]];
	}

    	return(ans);
}

int find_nearest_cluster(int numClusters, int max_index, int id, float **clusters)
{
	int index, i;
    	float dist, min_dist;

    	/* find the cluster id that has min distance to object */
    	index = 0;
    	min_dist = euclid_dist_2(id, max_index, clusters[0]);

    	for (i=1; i<numClusters; i++) 
	{
        	dist = euclid_dist_2(id, max_index, clusters[i]);
        	/* no need square root */
        	if (dist < min_dist) 
		{ /* find the min and its array index */
            		min_dist = dist;
            		index    = i;
        	}
    	}

    	return(index);
}

int main(int argc, char** argv)  
{   
	MPI_Status Stat;
	char hostname[MPI_MAX_PROCESSOR_NAME];
	rc = MPI_Init(&argc,&argv);
	if (rc != MPI_SUCCESS) 
	{
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
  	
  	while (1) 
	{
    		static struct option longOptions[] = 
		{
      			{"gaussian", no_argument, &kType, GAUSSIAN},
      			{"polynomial", no_argument, &kType, POLYNOMIAL},
      			{"sigmoid", no_argument, &kType, SIGMOID},
      			{"linear", no_argument, &kType, LINEAR},
      			{"cost", required_argument, 0, 'c'},
      			{"tolerance", required_argument, 0, 't'},
      			{"epsilon", required_argument, 0, 'e'},
      			{"output", required_argument, 0, 'o'},
      			{"version", no_argument, 0, 'v'},
      			{"help", no_argument, 0, 'f'}
    		};
    		int optionIndex = 0;
    		currentOption = getopt_long(argc, (char *const*)argv, "c:t:e:o:p:q:s:a:r:d:g:v:f", longOptions, &optionIndex);
    		if (currentOption == -1) 
		{
      			break;
    		}
    		int method = 3;

    		switch (currentOption) 
		{
    			case 0:
      				break;
    			case 'v':
      				printf("CPSVM version: 1.0\n");
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

  	if (optind != argc - 1) 
	{
    		printHelp();
    		return(0);
	}

	const char* trainingFilename = argv[optind];  
  
  	if (outputFilename == NULL) 
	{
    		int inputNameLength = strlen(trainingFilename);
    		outputFilename = (char*)malloc(sizeof(char)*(inputNameLength + 5));
    		strncpy(outputFilename, trainingFilename, inputNameLength + 4);
    		char* period = strrchr(outputFilename, '.');
    		if (period == NULL) 
		{
      			period = outputFilename + inputNameLength;
    		}
    		strncpy(period, ".mdl\0", 5);
  	}  	

	DataType* whole_data; 
	DataType* whole_labels; 
	int* whole_data_index; 
	int* whole_row_index;
	int nPoints;
	int max_index;
	int elements;

	if(rank==0)
	{
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
		elements = 0;

		max_line_len = 1024;
		line = Malloc(char,max_line_len);
		while(readline(fp)!=NULL)
		{
			char *p = strtok(line," \t"); // label
			// features
			while(1)
			{
				p = strtok(NULL," \t");
				if(p == NULL || *p == '\n') 
				{// check '\n' as ' ' may be after the last feature
					break;
				}
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
			if(label == NULL) 
			{// empty line
				exit_input_error(i+1);
			}
			//prob.y[i] = strtod(label,&endptr);
			whole_labels[i] = strtod(label,&endptr);
			if(endptr == label || *endptr != '\0')
			{
				exit_input_error(i+1);
			}

			while(1)
			{
				idx = strtok(NULL,":");
				val = strtok(NULL," \t");

				if(val == NULL)
				{
					break;
				}

				errno = 0;
				//x_space[j].index = (int) strtol(idx,&endptr,10);
				whole_data_index[j] = ((int) strtol(idx,&endptr,10)-1);
				//if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				if(endptr == idx || errno != 0 || *endptr != '\0' || (whole_data_index[j]+1) <= inst_max_index)
				{
					exit_input_error(i+1);
				}
				else
				{
					//inst_max_index = x_space[j].index;
					inst_max_index = (whole_data_index[j]+1);
				}

				errno = 0;
				//x_space[j].value = strtod(val,&endptr);
				whole_data[j] = strtod(val,&endptr);
				if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				{
					exit_input_error(i+1);
				}

				++j;
			}

			if(inst_max_index > max_index)
			{
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
				fprintf(verify_input,"%d:%f ",data_index[j]+1,data[j]);
			fprintf(verify_input,"\n");
		}
		*/
		//printf("\n\nInput data found: %d points, %d is the maximum dimension\n", nPoints, max_index); 
	}

	MPI_Bcast(&nPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&max_index, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	int sub_nPoints = nPoints/numtasks;

	//printf("\n*********on node %d, sub_nPoints is %d\n", rank, sub_nPoints);

	//alpha = (DataType*)calloc(sub_nPoints, sizeof(DataType));

	cEpsilon = cost - epsilon; 

	//get_data_size(sub_nPoints);
	if(rank==0)
	{
		send_counts = (int *)malloc(numtasks*sizeof(int));
		displs = (int *)malloc(numtasks*sizeof(int));
		for(int i=0;i<numtasks;i++)
		{
			send_counts[i] = whole_row_index[(i+1)*sub_nPoints] - whole_row_index[i*sub_nPoints];
			if(i==0)
			{	
				displs[i] = 0;
			}
		   	else 
			{
				displs[i] = displs[i-1]+send_counts[i-1];
			}
		}
	}
	MPI_Scatter(send_counts, 1, MPI_INT, &sub_element, 1, MPI_INT, 0, MPI_COMM_WORLD);

	//printf("\n^^^^^^^^^^on node %d, sub_element is %d\n", rank, sub_element);

	data = (DataType*)malloc(sub_element*sizeof(DataType));

	labels = (DataType*)malloc(sub_nPoints*sizeof(DataType));

	data_index = (int*)malloc(sub_element*sizeof(int));

	row_index = (int*)malloc((sub_nPoints+1)*sizeof(int)); 

	MPI_Scatterv(whole_data, send_counts, displs, MPI_FLOAT, data, sub_element, MPI_FLOAT, 0, MPI_COMM_WORLD);

	MPI_Scatterv(whole_data_index, send_counts, displs, MPI_INT, data_index, sub_element, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Scatter(whole_row_index, sub_nPoints, MPI_INT, row_index, sub_nPoints, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Scatter(whole_labels, sub_nPoints, MPI_FLOAT, labels, sub_nPoints, MPI_FLOAT, 0, MPI_COMM_WORLD);

	//printf("\n###########on node %d, after data distribution\n", rank);
	MPI_Barrier(MPI_COMM_WORLD);
        double curT0 = MPI_Wtime();
		
	
	for(int i=1;i<sub_nPoints;i++)
	{
		row_index[i] = (row_index[i] - row_index[0]);
	}

	row_index[0] = 0;

	row_index[sub_nPoints] = sub_element;
  		
  	
	/**************************************************************************/
	/**************************finished data distribution**********************/
	/**************************************************************************/
	numClusters = numtasks;
	DataType** clusters    = (DataType**) malloc(numClusters *             sizeof(DataType*));
    	assert(clusters != NULL);
    	clusters[0] = (DataType*)  calloc(numClusters * max_index, sizeof(DataType));
   	assert(clusters[0] != NULL);
	for (int i=1; i<numClusters; i++)
	{
        	clusters[i] = clusters[i-1] + max_index;
	}

    	/*if(rank==0)	{
	    srand (time(NULL));
	    int * cluster_id = (int *)malloc(numClusters*sizeof(int));
	    cluster_id[0] = rand()%nPoints;
	    for(int i=1;i<numClusters;i++){
		    int my_id;
		    while(1){
			my_id = rand()%nPoints;
			int id_flag=0;
			for(int k=0;k<i;k++){
				if(my_id==cluster_id[k])	id_flag=1;
			}
			if(id_flag==0)	break;
		    }
		    cluster_id[i] = my_id;
	    }
	    printf("the init clusters of kmeans are:\n");
	    for(int i=0;i<numClusters;i++){
		    printf("%d ", cluster_id[i]);
		    int id = cluster_id[i];
		    for(int j=whole_row_index[id];j<whole_row_index[id+1];j++){
			    clusters[i][whole_data_index[j]] = whole_data[j];
		    }
	    }
	    printf("\n");
	}*/

    	if(rank==0)	
	{
		for(int i=0;i<numClusters;i++)
		{
			for(int j=whole_row_index[i];j<whole_row_index[i+1];j++)
			{
				clusters[i][whole_data_index[j]] = whole_data[j];
		    	}
	    	}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(clusters[0], numClusters*max_index, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	int* membership = (int*) malloc(sub_nPoints * sizeof(int));
    	assert(membership != NULL);
    	int     *newClusterSize; 
    	int     *clusterSize;
    	DataType    delta;
    	DataType    delta_tmp;
    	DataType  **newClusters;

    	for (int i=0; i<sub_nPoints; i++) 
	{
		membership[i] = -1;
	}

    	newClusterSize = (int*) calloc(numClusters, sizeof(int));
    	assert(newClusterSize != NULL);
    	clusterSize    = (int*) calloc(numClusters, sizeof(int));
    	assert(clusterSize != NULL);

    	newClusters    = (float**) malloc(numClusters *            sizeof(float*));
    	assert(newClusters != NULL);
    	newClusters[0] = (float*)  calloc(numClusters * max_index, sizeof(float));
    	assert(newClusters[0] != NULL);

    	for (int i=1; i<numClusters; i++)
	{
        	newClusters[i] = newClusters[i-1] + max_index;
	}

    	int loop = 0;
	//MPI_Barrier(MPI_COMM_WORLD);
	//double curT0 = MPI_Wtime();
	
    	do 
	{
        	delta = 0.0;
        	for (int pi=0; pi<sub_nPoints; pi++) 
		{
            		/* find the array index of nestest cluster center */
            		int index = find_nearest_cluster(numClusters, max_index, pi, clusters);
            
            		/* if membership changes, increase delta by 1 */
            		if (membership[pi] != index) 
			{
				delta += 1.0;
			}

            		/* assign the membership to object i */
           		membership[pi] = index;

            		/* update new cluster centers : sum of objects located within */
            		newClusterSize[index]++;
            		for (int j=row_index[pi]; j<row_index[pi+1]; j++)
			{
                		newClusters[index][data_index[j]] += data[j];
			}
        	}

        	/* sum all data objects in newClusters */
        	MPI_Allreduce(newClusters[0], clusters[0], numClusters*max_index, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        	MPI_Allreduce(newClusterSize, clusterSize, numClusters, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        	/* average the sum and replace old cluster centers with newClusters */
        	for (int i=0; i<numClusters; i++) 
		{
            		for (int j=0; j<max_index; j++) 
			{
                		if (clusterSize[i] > 1)
				{
                    			clusters[i][j] /= clusterSize[i];
				}
                		newClusters[i][j] = 0.0;   /* set back to 0 */
            		}
            		newClusterSize[i] = 0;   /* set back to 0 */
        	}
            
        	MPI_Allreduce(&delta, &delta_tmp, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        	delta = delta_tmp / nPoints;
    
	} 
	while (delta > kmeans_threshold && loop++ < 500);

    	double maxTime;
    	double curT = MPI_Wtime() - curT0;
    	MPI_Reduce(&curT, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    	if (rank == 0) 
	{
		printf("rank %2d: kmeans is finished, loops=%d time=%f sec\n",rank,loop,curT);
	}

    	DataType* new_labels;
    	int* new_row_index;
    	int* new_data_index;
    	DataType* new_data;
    	int * whole_membership;

    	if(rank==0)
	{
	    	new_labels = (DataType *)malloc(nPoints*sizeof(DataType));
		new_row_index = (int *)calloc((nPoints+1), sizeof(int));	
		new_data = (DataType *)malloc(elements*sizeof(DataType));
		new_data_index = (int *)malloc(elements*sizeof(int));
		whole_membership = (int *)malloc(nPoints*sizeof(int));
		FILE * clusterFile = fopen("clusterFile.kmeans", "w");
		fprintf(clusterFile, "%d %d\n", numClusters, max_index);
		for (int i=0; i<numClusters; i++) 
		{
            		for (int j=0; j<max_index; j++) 
			{
				fprintf(clusterFile, "%f ", clusters[i][j]);
            		}
			fprintf(clusterFile, "\n");
        	}
		fclose(clusterFile);
    	}
    	MPI_Gather(membership, sub_nPoints, MPI_INT, whole_membership, sub_nPoints, MPI_INT, 0, MPI_COMM_WORLD);

    	int* displs2;

    	MPI_Barrier(MPI_COMM_WORLD);

	if(rank==0)
	{
		displs2 = (int *)malloc(numtasks*sizeof(int));
		int ni = 0, nj =0;
		for(int i=0;i<numClusters;i++)
		{
			send_counts[i] = 0;
			for(int j=0;j<nPoints;j++)
			{
				if(whole_membership[j]==i)
				{
					new_labels[ni] = whole_labels[j];
					new_row_index[ni+1] = (new_row_index[ni] + (whole_row_index[j+1]-whole_row_index[j]));
					send_counts[i] += (whole_row_index[j+1]-whole_row_index[j]);
					for(int k=whole_row_index[j];k<whole_row_index[j+1];k++)
					{
						new_data[nj] = whole_data[k];
						new_data_index[nj] = whole_data_index[k];
						nj++;
					}
					ni++;
				}
			}
			if(i==0)
			{
				displs[i] = 0;
				displs2[i] = 0;
			} 
			else 
			{
				displs[i] = displs[i-1]+send_counts[i-1];
				displs2[i] = displs2[i-1]+clusterSize[i-1];
				//printf("\n$$$$$$$$$$$$$$$$$$$ displs[%d]: %d, displs[%d]: %d, send_counts[%d]: %d, displs2[%d]: %d, displs2[%d]: %d, clusterSize[%d]: %d $$$$$$$$$$$$$$\n\n", i, displs[i], (i-1), displs[i-1], (i-1), send_counts[i-1], i, displs2[i], (i-1), displs2[i-1], (i-1), clusterSize[i-1]);
			}
		}
		//printf("\n$$$$$$$$$$$$$$$$$$$ni: %d, nj: %d$$$$$$$$$$$$$$\n\n", ni, nj);
		/*for(int i=0;i<nPoints;i++){
			printf("%d ", (int)new_labels[i]);
			for(int j=new_row_index[i];j<new_row_index[i+1];j++){
				printf("%d:%f ", new_data_index[j]+1, new_data[j]);
			}
			printf("\n");
		}*/
		free(whole_data);
	  	free(whole_data_index);
	  	free(whole_row_index);
	  	free(whole_labels);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	
	MPI_Scatter(send_counts, 1, MPI_INT, &sub_element, 1, MPI_INT, 0, MPI_COMM_WORLD);

	sub_nPoints = clusterSize[rank];

	//MPI_Barrier(MPI_COMM_WORLD);
	//printf("\n$$$$$$$$$$$$$$$$$$$$$ %d node has %d points and %d elements $$$$$$$$$$$$$$$$$$$$$\n\n", rank, sub_nPoints, sub_element);
	//MPI_Barrier(MPI_COMM_WORLD);

	alpha = (DataType*)calloc(sub_nPoints, sizeof(DataType));

	free(data);
	data = (DataType*)malloc(sub_element*sizeof(DataType));

	free(labels);
	labels = (DataType*)malloc(sub_nPoints*sizeof(DataType));

	free(data_index);
	data_index = (int*)malloc(sub_element*sizeof(int));

	free(row_index);
	row_index = (int*)malloc((sub_nPoints+1)*sizeof(int)); 

	MPI_Scatterv(new_data, send_counts, displs, MPI_FLOAT, data, sub_element, MPI_FLOAT, 0, MPI_COMM_WORLD);

	MPI_Scatterv(new_data_index, send_counts, displs, MPI_INT, data_index, sub_element, MPI_INT, 0, MPI_COMM_WORLD);

	//MPI_Scatter(whole_row_index, sub_nPoints, MPI_INT, row_index, sub_nPoints, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatterv(new_row_index, clusterSize, displs2, MPI_INT, row_index, sub_nPoints, MPI_INT, 0, MPI_COMM_WORLD);

	//MPI_Scatter(whole_labels, sub_nPoints, MPI_FLOAT, labels, sub_nPoints, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Scatterv(new_labels, clusterSize, displs2, MPI_FLOAT, labels, sub_nPoints, MPI_FLOAT, 0, MPI_COMM_WORLD);		
	
	for(int i=1;i<sub_nPoints;i++)
	{
		row_index[i] = (row_index[i] - row_index[0]);
	}

	row_index[0] = 0;

	row_index[sub_nPoints] = sub_element;

	/*for(int ri=0;ri<numtasks;ri++){
		MPI_Barrier(MPI_COMM_WORLD);
		if(rank==ri){
			printf("\n*********************************rank: %d*********************************\n\n", rank);
			for(int i=0;i<=sub_nPoints;i++){
				printf("%d ", row_index[i]);
			}
			printf("\n\n*********************************rank: %d*********************************\n", rank);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}*/

	if(rank==0)
	{
		free(send_counts);
		free(displs);
		free(displs2);
		free(new_labels);
    		free(new_row_index);
    		free(new_data_index);
    		free(new_data);
    		free(whole_membership);
	}
	free(clusters);
	free(clusterSize);
	free(newClusters);
	free(newClusterSize);
	
	/*
	if(rank==0){
		FILE* out_member = fopen("membership.txt", "w");
		for(int i=0;i<nPoints;i++)
			fprintf(out_member, "%d %d\n", i, whole_membership[i]);
	}*/
	/*for(int ri=0;ri<numtasks;ri++){
		MPI_Barrier(MPI_COMM_WORLD);
		if(rank==ri){
			printf("\n*********************************rank: %d*********************************\n", rank);
			for(int i=0;i<sub_nPoints;i++){
				printf("%d ", (int)labels[i]-1);
				for(int j=row_index[i];j<row_index[i+1];j++){
					printf("%d:%f ", data_index[j]+1, data[j]);
				}
				printf("\n");
			}
			printf("\n*********************************rank: %d*********************************\n", rank);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}*/
	
	
	/*if(rank==0){
		for(int i=0;i<numClusters;i++){
			printf("**********cluster%d has %d elements**********\n", i, clusterSize[i]);
			printf("\n");
		}
	}*/
	
	

	//printf("**********Finish Test at Rank %d**********\n", rank);
	MPI_Barrier(MPI_COMM_WORLD);
	//return 0;

	if(inputalpha!=NULL)
	{
		//printf("**********start input alpha**********\n");
		FILE * alpha_in = fopen(inputalpha, "r");
		DataType in_temp;
		int i = 0;
		while(fscanf(alpha_in, "%f", &in_temp)!=EOF)
		{
			alpha[i++] = in_temp;
		}
	}
	else
	{
		//printf("**********no input alpha**********\n");
	}
	/*if(rank==2)
	for(int i=0;i<sub_nPoints;i++){
		printf("%d ",(int)labels[i]);
		for(int j=row_index[i];j<row_index[i+1];j++){
			printf("%d:%f ",data_index[j]+1,data[j]);	
		}
		printf("\n\n");	
	}*/
  
  	if (kType == LINEAR) 
	{
    		printf("Linear kernel\n");
    		//kp.kernel_type = "linear";
  	} 
	else if (kType == POLYNOMIAL) 
	{
    		if (!(parameterCSet)) 
		{
      			parameterC = 3.0f;
    		}
    		if (!(parameterASet)) 
		{
      			//parameterA = 1.0/nPoints;
	  		parameterA = 1.0/max_index;
    		}
    		if (!(parameterBSet)) 
		{
      			parameterB = 0.0f;
    		}
    		//printf("Polynomial kernel: a = %f, r = %f, d = %f\n", parameterA, parameterB, parameterC);
    		if ((parameterA <= 0) || (parameterB < 0) || (parameterC < 1.0)) 
		{
      			printf("Invalid parameters\n");
      			exit(1);
    		}
    		kp.kernel_type = "polynomial";
    		kp.gamma = parameterA;
    		kp.coef0 = parameterB;
    		kp.degree = (int)parameterC;
  	} 
	else if (kType == GAUSSIAN) 
	{
    		if (!(parameterASet)) {
      			//parameterA = 1.0/nPoints;
	  		parameterA = 1.0/max_index;
    		} 
		else 
		{
      			parameterA = -parameterA;
    		}
    		//printf("Gaussian kernel: gamma = %f\n", parameterA);
    		if (parameterA < 0) 
		{
      			printf("Invalid parameters\n");
      			exit(1);
    		}
    		kp.kernel_type = "rbf";
    		kp.gamma = parameterA;
  	} 
	else if (kType == SIGMOID) 
	{
    		if (!(parameterASet)) 
		{
      			//parameterA = 1.0/nPoints;
	  		parameterA = 1.0/max_index;
    		}
    		if (!(parameterBSet)) 
		{
      			parameterB = 0.0f;
    		}
    		//printf("Sigmoid kernel: a = %f, r = %f\n", parameterA, parameterB);
    		if ((parameterA <= 0) || (parameterB < 0)) 
		{
      			printf("Invalid Parameters\n");
      			exit(1);
    		}
    		kp.kernel_type = "sigmoid";
    		kp.gamma = parameterA;
    		kp.coef0 = parameterB;
  	}
	//struct timeval start,finish;
	//gettimeofday(&start, 0);
	int logp = log2(numtasks);

	//MPI_Barrier(MPI_COMM_WORLD);

	/*mycolor=rank%2;
	printf("---------------------------rank: %d, myrank: %d\n", rank, myrank);
	int rank_size = numtasks/2;
  	MPI_Comm_split(MPI_COMM_WORLD,mycolor,rank,&double_comm);
  	MPI_Comm_size(double_comm,&rank_size);
  	MPI_Comm_rank(double_comm,&myrank);
  	printf("))))))))))))))))))))))))))rank: %d, myrank: %d\n", rank, myrank);*/

  	MPI_Barrier(MPI_COMM_WORLD);

	double dcsvm_start = MPI_Wtime();

	performTraining(sub_nPoints);

	MPI_Barrier(MPI_COMM_WORLD);

	double dcsvm_end = MPI_Wtime();
	double dcsvm_time = dcsvm_end - dcsvm_start;

	if (rank==0)
	{	
		printf("\n@@@@@@@@@@@@@@@@@@@@@@@@-cpsvm time is %lf sec, kmeans (%d loops) time is %lf, total time is %lf\n\n", dcsvm_time, loop, curT, (dcsvm_end-curT0));
	}
	//printf("Output File for rank %d is %s\n", rank, outputFilename);

	//if(rank==0)	printModel(outputFilename, data, data_index, row_index, labels, alpha, sub_nPoints);

	char buffer[6];
    	snprintf(buffer, sizeof(char) * 32, "%i", rank);
	char filename[32];
	int i, j;
	for(i=0;outputFilename[i]!='\0';i++) 
	{
		filename[i] = outputFilename[i];
	}
	for(j=0;buffer[j]!='\0';j++, i++) 
	{
		filename[i] = buffer[j];
	}
	filename[i] = '\0';
	
	printModel(filename, data, data_index, row_index, labels, alpha, sub_nPoints);

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Finalize(); 
	return 0;
}
