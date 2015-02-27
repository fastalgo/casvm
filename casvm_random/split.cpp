/*the data is stored in column-major*/
#include <sys/time.h>   
#include <stdio.h>   
#include <math.h>
#include <string.h>
#include <errno.h>
#include <getopt.h>
#include <stdlib.h>

#define DataType float
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

struct svm_node
{
    int index;
    DataType value;
};

static char *line = NULL;
static int max_line_len;
struct svm_node *x_space;
struct svm_node ** data;
DataType* labels;
int nPoints, nDimension, max_index;

#ifndef SVMCOMMONH
#define SVMCOMMONH
#include <string>
struct Kernel_params{
	DataType gamma;
	DataType coef0;
	int degree;
	DataType b;
	std::string kernel_type;
};
enum SelectionHeuristic {FIRSTORDER, SECONDORDER, RANDOM, ADAPTIVE};
#endif

#ifndef KernelType
#define KernelType
enum KernelType {
  LINEAR,
  POLYNOMIAL,
  GAUSSIAN,
  SIGMOID
};
#endif

#ifndef SVM_KERNELS
#define SVM_KERNELS
#include <math.h>
struct Linear {
  static DataType selfKernel(const svm_node * pointerA, DataType parameterA, DataType parameterB, DataType parameterC) {
    DataType accumulant = 0.0f;
    do {
      DataType value = pointerA->value;
      accumulant += value * value;
      pointerA = pointerA + 1;
    } while (pointerA->index!=-1);
    return accumulant;
  }
  static DataType kernel(const svm_node *x, const svm_node *y, DataType parameterA, DataType parameterB, DataType parameterC){
	DataType sum = 0;	
    while(x->index != -1 && y->index !=-1){
        if(x->index == y->index){
            sum += (x->value) * (y->value);
            ++x;
            ++y;			
        }else{            
			if(x->index > y->index)	++y;           
            else	++x;
        }
    }
    return sum;
  }
};

struct Polynomial {
  static DataType selfKernel(const svm_node * pointerA, DataType a, DataType r, DataType d) {
    DataType accumulant = 0.0f;
    do {
      DataType value = pointerA->value;
      accumulant += value * value;
      pointerA = pointerA + 1;
    } while (pointerA->index!=-1);
    accumulant = accumulant * a + r;
    DataType result = accumulant;
    for (DataType degree = 2.0f; degree <= d; degree = degree + 1.0f) {
      result *= accumulant;
    }    
    return result;
  }
  static DataType kernel(const svm_node *x, const svm_node *y, DataType a, DataType r, DataType d) {
    DataType sum = 0;	
    while(x->index != -1 && y->index !=-1){
        if(x->index == y->index){
            sum += (x->value) * (y->value);
            ++x;
            ++y;			
        }else{            
			if(x->index > y->index)	++y;           
            else	++x;
        }
    }
    sum = sum * a + r;
    DataType result = sum;
    for (DataType degree = 2.0f; degree <= d; degree = degree + 1.0f) {
      result *= sum;
    }    
    return result;
  }  
};

struct Gaussian {
  static DataType selfKernel(const svm_node * pointerA, DataType parameterA, DataType parameterB, DataType parameterC){
    return 1.0f;
  }
  static DataType kernel(const svm_node *x, const svm_node *y, DataType ngamma, DataType parameterB, DataType parameterC){
    double sum = 0;	
    while(x->index != -1 && y->index !=-1){
        if(x->index == y->index){
            double d = x->value - y->value;
            sum += d*d;
            ++x;
            ++y;			
        }else{            
			if(x->index > y->index){
                sum += y->value * y->value;
                ++y;
            }else{
                sum += x->value * x->value;
                ++x;
            }
        }
    }	
    while(x->index != -1){
        sum += x->value * x->value;
        ++x;
    }	
    while(y->index != -1){
        sum += y->value * y->value;
        ++y;
    }	
    return exp(ngamma*sum);
  }
};

struct Sigmoid {  
  static DataType selfKernel(const svm_node * pointerA, DataType a, DataType r, DataType parameterC) {
    DataType accumulant = 0.0f;
    do {
      DataType value = pointerA->value;
      accumulant += value * value;
      pointerA = pointerA + 1;
    } while (pointerA->index!=-1);
    accumulant = accumulant * a + r;
    return tanh(accumulant);
  }
  static DataType kernel(const svm_node *x, const svm_node *y, DataType a, DataType r, DataType parameterC) {
    DataType sum = 0;	
    while(x->index != -1 && y->index !=-1){
        if(x->index == y->index){
            sum += (x->value) * (y->value);
            ++x;
            ++y;			
        }else{            
			if(x->index > y->index)	++y;           
            else	++x;
        }
    }
    sum = sum * a + r;
    return tanh(sum);
  }
};
#endif

#ifndef INITIALIZE
#define INITIALIZE

template<class Kernel>
void initializeArrays(DataType parameterA, DataType parameterB, DataType parameterC, DataType* devKernelDiag, DataType* alpha, DataType* devF) { 	
	for (int index = 0;index < nPoints;index++) {
		devKernelDiag[index] = Kernel::selfKernel(data[index], parameterA, parameterB, parameterC);
		devF[index] = -labels[index];
		alpha[index] = 0;
	}
}

void launchInitialization(int kType, DataType parameterA, DataType parameterB, DataType parameterC, DataType* devKernelDiag, DataType* alpha, DataType* devF) {
  switch (kType) {
  case LINEAR:
    initializeArrays<Linear>(parameterA, parameterB, parameterC, devKernelDiag, alpha, devF);
    break;
  case POLYNOMIAL:
    initializeArrays<Polynomial>(parameterA, parameterB, parameterC, devKernelDiag, alpha, devF);
    break;
  case GAUSSIAN:
    initializeArrays<Gaussian>(parameterA, parameterB, parameterC, devKernelDiag, alpha, devF);
    break;  
  case SIGMOID:
    initializeArrays<Sigmoid>(parameterA, parameterB, parameterC, devKernelDiag, alpha, devF);
    break;
  }
}

template<class Kernel>
void takeFirstStep(DataType * alphaLowOld, DataType * alphaHighOld, DataType* devKernelDiag, DataType* alpha, DataType cost, int iLow, int iHigh, DataType parameterA, DataType parameterB, DataType parameterC) { 
                                     
	DataType eta = devKernelDiag[iHigh] + devKernelDiag[iLow] - 2*Kernel::kernel(data[iHigh], data[iLow], parameterA, parameterB, parameterC);
	//For the first step, we know alphaHighOld == alphaLowOld == 0, and we know sign == -1
	//labels[iLow] = -1
	//labels[iHigh] = 1
	//DataType sign = -1;

	*alphaLowOld = alpha[iLow];
	*alphaHighOld = alpha[iHigh];
 
	//And we know eta > 0
	DataType alphaLowNew = 2/eta; //Just boil down the algebra
	if (alphaLowNew > cost) {
		alphaLowNew = cost;
	}
	//alphaHighNew == alphaLowNew for the first step
	alpha[iLow] = alphaLowNew;
	alpha[iHigh] = alphaLowNew;	
}
void launchTakeFirstStep(DataType * alphaLowOld, DataType * alphaHighOld, DataType* devKernelDiag, DataType* alpha, DataType cost, int iLow, int iHigh, int kType, DataType parameterA, DataType parameterB, DataType parameterC) {
  switch (kType) {
  case LINEAR:
    takeFirstStep<Linear>(alphaLowOld, alphaHighOld, devKernelDiag, alpha, cost, iLow, iHigh, parameterA, parameterB, parameterC);
    break;
  case POLYNOMIAL:
    takeFirstStep<Polynomial>(alphaLowOld, alphaHighOld, devKernelDiag, alpha, cost, iLow, iHigh, parameterA, parameterB, parameterC);
    break;
  case GAUSSIAN:
    takeFirstStep<Gaussian>(alphaLowOld, alphaHighOld, devKernelDiag, alpha, cost, iLow, iHigh, parameterA, parameterB, parameterC);
    break;  
  case SIGMOID:
    takeFirstStep<Sigmoid>(alphaLowOld, alphaHighOld, devKernelDiag, alpha, cost, iLow, iHigh, parameterA, parameterB, parameterC);
    break;
  }
}
#endif

#ifndef SECONDORDERH
#define SECONDORDERH
template<class Kernel>
  void	secondOrder(DataType epsilon, DataType cEpsilon, DataType* alpha, DataType* devF, DataType * alphaHighDiff, DataType * alphaLowDiff, int * iLow, int * iHigh, DataType * bLow, DataType * bHigh, DataType parameterA, DataType parameterB, DataType parameterC, DataType * devKernelDiag, DataType cost) {
	  int i;
	  DataType highKernel,lowKernel;
	  for(i=0;i<nPoints;i++){
		  highKernel = Kernel::kernel(data[i], data[*iHigh], parameterA, parameterB, parameterC);
		  lowKernel = Kernel::kernel(data[i], data[*iLow], parameterA, parameterB, parameterC);
		  devF[i] = devF[i] + *alphaHighDiff * labels[*iHigh] * highKernel + *alphaLowDiff * labels[*iLow] * lowKernel;
	  }
	  DataType min,beta,kappa,G;
	  for(i=0;i<nPoints;i++)
		  if (((labels[i] > 0) && (alpha[i] < cEpsilon)) || ((labels[i] < 0) && (alpha[i] > epsilon))){
			  min = devF[i];
			  *iHigh = i;
			  break;
		  }
	  for(;i<nPoints;i++)
		  if (((labels[i] > 0) && (alpha[i] < cEpsilon)) || ((labels[i] < 0) && (alpha[i] > epsilon))){
			if(devF[i]<min){
			  min = devF[i];
			  *iHigh = i;
			}
		  }
	  for(i=0;i<nPoints;i++)
		  if (((labels[i] > 0) && (alpha[i] > epsilon)) || ((labels[i] < 0) && (alpha[i] < cEpsilon))){
			beta = devF[*iHigh]-devF[i];
			if(beta <= epsilon){
            //if(beta < -epsilon){
				kappa = devKernelDiag[*iHigh] + devKernelDiag[i] - 2 * Kernel::kernel(data[i], data[*iHigh], parameterA, parameterB, parameterC);//if we could reuse the highKernel???
				if (kappa <= 0)       kappa = epsilon;
				G = beta * beta / kappa;
				*iLow = i;
				break;
			}
		  }
	  for(;i<nPoints;i++)
		  if (((labels[i] > 0) && (alpha[i] > epsilon)) || ((labels[i] < 0) && (alpha[i] < cEpsilon))){
			beta = devF[*iHigh]-devF[i];
			if(beta <= epsilon){
            //if(beta < -epsilon){
				kappa = devKernelDiag[*iHigh] + devKernelDiag[i] - 2 * Kernel::kernel(data[i], data[*iHigh], parameterA, parameterB, parameterC);//maybe we should store kernel(iHigh,i) into cache
				if (kappa <= 0)       kappa = epsilon;
				DataType temp = beta * beta / kappa;
				if (temp > G){
					*iLow = i;
					G = temp;
				}
			}
		  }
	  *bHigh = devF[*iHigh];
	  *bLow = devF[*iLow];
	  DataType eta = devKernelDiag[*iHigh] + devKernelDiag[*iLow] - 2*Kernel::kernel(data[*iHigh], data[*iLow], parameterA, parameterB, parameterC);  
	  DataType alphaHighOld = alpha[*iHigh];
	  DataType alphaLowOld = alpha[*iLow];
	  DataType alphaDiff = alphaLowOld - alphaHighOld;
	  DataType lowLabel = labels[*iLow];
	  DataType sign = labels[*iHigh] * lowLabel;
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
		alphaLowNew = alphaLowOld + lowLabel*(devF[*iHigh] - devF[*iLow])/eta;
		if (alphaLowNew < alphaLowLowerBound) alphaLowNew = alphaLowLowerBound;
        else if (alphaLowNew > alphaLowUpperBound) alphaLowNew = alphaLowUpperBound;
	  } else {
		DataType slope = lowLabel * (bHigh - bLow);
		DataType delta = slope * (alphaLowUpperBound - alphaLowLowerBound);
		if (delta > 0) {
			if (slope > 0)	alphaLowNew = alphaLowUpperBound;
			else	alphaLowNew = alphaLowLowerBound;
		} else	alphaLowNew = alphaLowOld;
	  }
    *alphaLowDiff = alphaLowNew - alphaLowOld;
    *alphaHighDiff = -sign*(*alphaLowDiff);
    DataType alphaHighNew = alphaHighOld + *alphaHighDiff;
    alpha[*iLow] = alphaLowNew;
    alpha[*iHigh] = alphaHighNew;
}    

void launchSecondOrder(int kType, DataType epsilon, DataType cEpsilon, DataType* alpha, DataType* devF, DataType * alphaHighDiff, DataType * alphaLowDiff, int * iLow, int * iHigh, DataType * bLow, DataType * bHigh, DataType parameterA, DataType parameterB, DataType parameterC, DataType * devKernelDiag, DataType cost){
	switch (kType) {
      case LINEAR:
        secondOrder <Linear>(epsilon, cEpsilon, alpha, devF, alphaHighDiff, alphaLowDiff, iLow, iHigh, bLow, bHigh, parameterA, parameterB, parameterC, devKernelDiag, cost);
        break;
      case POLYNOMIAL:
        secondOrder <Polynomial>(epsilon, cEpsilon, alpha, devF, alphaHighDiff, alphaLowDiff, iLow, iHigh, bLow, bHigh, parameterA, parameterB, parameterC, devKernelDiag, cost);
        break;
      case GAUSSIAN:
        secondOrder <Gaussian>(epsilon, cEpsilon, alpha, devF, alphaHighDiff, alphaLowDiff, iLow, iHigh, bLow, bHigh, parameterA, parameterB, parameterC, devKernelDiag, cost);
        break;
      case SIGMOID:
        secondOrder <Sigmoid>(epsilon, cEpsilon, alpha, devF, alphaHighDiff, alphaLowDiff, iLow, iHigh, bLow, bHigh, parameterA, parameterB, parameterC, devKernelDiag, cost);
        break;
      }     
  }
#endif

#ifndef firstORDERH
#define firstORDERH
template<class Kernel>
  void	firstOrder(DataType epsilon, DataType cEpsilon, DataType* alpha, DataType* devF, DataType * alphaHighDiff, DataType * alphaLowDiff, int * iLow, int * iHigh, DataType * bLow, DataType * bHigh, DataType parameterA, DataType parameterB, DataType parameterC, DataType * devKernelDiag, DataType cost) {
	  int i;
	  DataType highKernel,lowKernel;
	  for(i=0;i<nPoints;i++){
		  highKernel = Kernel::kernel(data[i], data[*iHigh], parameterA, parameterB, parameterC);
		  lowKernel = Kernel::kernel(data[i], data[*iLow], parameterA, parameterB, parameterC);
		  devF[i] = devF[i] + *alphaHighDiff * labels[*iHigh] * highKernel + *alphaLowDiff * labels[*iLow] * lowKernel;
	  }
	  DataType min,max;
	  for(i=0;i<nPoints;i++)
		  if (((labels[i] > 0) && (alpha[i] < cEpsilon)) || ((labels[i] < 0) && (alpha[i] > epsilon))){
			  min = devF[i];
			  *iHigh = i;
			  break;
		  }
	  for(;i<nPoints;i++)
		  if (((labels[i] > 0) && (alpha[i] < cEpsilon)) || ((labels[i] < 0) && (alpha[i] > epsilon))){
			if(devF[i]<min){
			  min = devF[i];
			  *iHigh = i;
			}
		  }
	  for(i=0;i<nPoints;i++)
		  if (((labels[i] > 0) && (alpha[i] > epsilon)) || ((labels[i] < 0) && (alpha[i] < cEpsilon))){
			max = devF[i];
			*iLow = i;
			break;
		  }
	  for(;i<nPoints;i++)
		  if (((labels[i] > 0) && (alpha[i] > epsilon)) || ((labels[i] < 0) && (alpha[i] < cEpsilon))){
			if(devF[i]>max){
				max = devF[i];
				*iLow = i;
			}
		  }
	  *bHigh = devF[*iHigh];
	  *bLow = devF[*iLow];
	  DataType eta = devKernelDiag[*iHigh] + devKernelDiag[*iLow] - 2*Kernel::kernel(data[*iHigh], data[*iLow], parameterA, parameterB, parameterC);
	  DataType alphaHighOld = alpha[*iHigh];
	  DataType alphaLowOld = alpha[*iLow];
	  DataType alphaDiff = alphaLowOld - alphaHighOld;
	  DataType lowLabel = labels[*iLow];
	  DataType sign = labels[*iHigh] * lowLabel;
	  //2 low; 1 high
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
		alphaLowNew = alphaLowOld + lowLabel*(devF[*iHigh] - devF[*iLow])/eta;
		if (alphaLowNew < alphaLowLowerBound) alphaLowNew = alphaLowLowerBound;
        else if (alphaLowNew > alphaLowUpperBound) alphaLowNew = alphaLowUpperBound;
	  } else {
		DataType slope = lowLabel * (bHigh - bLow);
		DataType delta = slope * (alphaLowUpperBound - alphaLowLowerBound);
		if (delta > 0) {
			if (slope > 0)	alphaLowNew = alphaLowUpperBound;
			else	alphaLowNew = alphaLowLowerBound;
		} else	alphaLowNew = alphaLowOld;
	  }
    *alphaLowDiff = alphaLowNew - alphaLowOld;
    *alphaHighDiff = -sign*(*alphaLowDiff);
    DataType alphaHighNew = alphaHighOld + *alphaHighDiff;
    alpha[*iLow] = alphaLowNew;
    alpha[*iHigh] = alphaHighNew;
}    

//void launchFirstOrder(int kType, int nPoints, int nDimension, DataType* devData, DataType* devTransposedData, DataType* labels, DataType epsilon, DataType cEpsilon, DataType* alpha, DataType* devF, int * iLow, int * iHigh, DataType * bLow, DataType * bHigh, DataType parameterA, DataType parameterB, DataType parameterC, DataType* devKernelDiag, DataType cost)
//firstOrder(DataType* devData, int devDataPitchInDataTypes, DataType* labels, int nPoints, int nDimension, DataType epsilon, DataType cEpsilon, DataType* alpha, DataType* devF, DataType * alphaHighDiff, DataType * alphaLowDiff, int * iLow, int * iHigh, DataType * bLow, DataType * bHigh, DataType parameterA, DataType parameterB, DataType parameterC, DataType * devKernelDiag)
void launchFirstOrder(int kType, DataType epsilon, DataType cEpsilon, DataType* alpha, DataType* devF, DataType * alphaHighDiff, DataType * alphaLowDiff, int * iLow, int * iHigh, DataType * bLow, DataType * bHigh, DataType parameterA, DataType parameterB, DataType parameterC, DataType * devKernelDiag, DataType cost){
	switch (kType) {
      case LINEAR:
        firstOrder <Linear>(epsilon, cEpsilon, alpha, devF, alphaHighDiff, alphaLowDiff, iLow, iHigh, bLow, bHigh, parameterA, parameterB, parameterC, devKernelDiag, cost);
        break;
      case POLYNOMIAL:
        firstOrder <Polynomial>(epsilon, cEpsilon, alpha, devF, alphaHighDiff, alphaLowDiff, iLow, iHigh, bLow, bHigh, parameterA, parameterB, parameterC, devKernelDiag, cost);
        break;
      case GAUSSIAN:
        firstOrder <Gaussian>(epsilon, cEpsilon, alpha, devF, alphaHighDiff, alphaLowDiff, iLow, iHigh, bLow, bHigh, parameterA, parameterB, parameterC, devKernelDiag, cost);
        break;
      case SIGMOID:
        firstOrder <Sigmoid>(epsilon, cEpsilon, alpha, devF, alphaHighDiff, alphaLowDiff, iLow, iHigh, bLow, bHigh, parameterA, parameterB, parameterC, devKernelDiag, cost);
        break;
      }     
  }
#endif

//FILE * testout = fopen("test.txt","w");
void performTraining(DataType * alpha, Kernel_params* kp, DataType cost, SelectionHeuristic heuristicMethod, DataType epsilon, DataType tolerance){
	DataType cEpsilon = cost - epsilon;
	int kType = GAUSSIAN;
	DataType parameterA;
	DataType parameterB;
	DataType parameterC;
	if (kp->kernel_type.compare(0,3,"rbf") == 0) {
		parameterA = -kp->gamma;
		kType = GAUSSIAN;
		printf("Gaussian kernel: gamma = %f\n", -parameterA);
	} else if (kp->kernel_type.compare(0,10,"polynomial") == 0) {
		parameterA = kp->gamma;
		parameterB = kp->coef0;
		parameterC = kp->degree;
		kType = POLYNOMIAL;
		printf("Polynomial kernel: a = %f, r = %f, d = %f\n", parameterA, parameterB, parameterC);
	} else if (kp->kernel_type.compare(0,6,"linear") == 0) {
		kType = LINEAR;
		printf("Linear kernel\n");
	} else if (kp->kernel_type.compare(0,7,"sigmoid") == 0) {
		kType = SIGMOID;
		parameterA = kp->gamma;
		parameterB = kp->coef0;
		printf("Sigmoid kernel: a = %f, r = %f\n", parameterA, parameterB);
		if ((parameterA <= 0) || (parameterB < 0)) {
			printf("Invalid Parameters\n");
			exit(1);
		}
	}
	printf("--Cost: %f, Tolerance: %f, Epsilon: %f\n", cost, tolerance, epsilon); 
	DataType* devKernelDiag = (DataType*)malloc(sizeof(DataType) * nPoints);
	DataType* devF = (DataType*)malloc(sizeof(DataType) * nPoints);
	DataType alphaLowOld, alphaHighOld, alphaLowDiff, alphaHighDiff;
	launchInitialization(kType, parameterA, parameterB, parameterC, devKernelDiag, alpha, devF); 
	printf("Initialization complete\n");

  DataType bLow = 1;
  DataType bHigh = -1;
  int iteration = 0;
  int iLow = -1;
  int iHigh = -1;
  for (int i = 0; i < nPoints; i++) {
    if (labels[i] < 0) {
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
  //launchTakeFirstStep(DataType * alphaLowOld, DataType * alphaHighOld, DataType* devKernelDiag, DataType* alpha, DataType cost, int iLow, int iHigh, int kType, DataType parameterA, DataType parameterB, DataType parameterC)
  launchTakeFirstStep(&alphaLowOld, &alphaHighOld, devKernelDiag, alpha, cost, iLow, iHigh, kType, parameterA, parameterB, parameterC);
  alphaLowDiff = alpha[iLow] - alphaLowOld;
  alphaHighDiff = -labels[iHigh] * labels[iLow] * alphaLowDiff;

  printf("Starting iterations\n");
	
  for (iteration = 1; true; iteration++) {	
    if (bLow <= bHigh + 2*tolerance) {
      printf("Converged\n");
      break; //Convergence!!
    }
    //printf("oldLow: %f, oldHigh: %f	--- newLow: %f, newHigh: %f\n", alphaLowOld, alphaHighOld, alpha[iLow], alpha[iHigh]);
	//printf("iLow: %d, iHigh: %d	--- bLow: %f, bHigh: %f\n", iLow, iHigh, bLow, bHigh);
	//if(iteration == 10)
		//break;
    if ((iteration & 0x7ff) == 0) {
	  printf("iteration: %d; bLow: %f, bHigh: %f\n", iteration, bLow, bHigh);
      //printf("iteration: %d; gap: %f\n",iteration, bLow - bHigh);
    }        
    if ((iteration & 0x7f) == 0) {
      //heuristicMethod = progress.getMethod();
    }   
    /*if (heuristicMethod == FIRSTORDER) {
      launchFirstOrder(kType, data, labels, nPoints, nDimension, epsilon, cEpsilon, alpha, devF, &alphaHighDiff, &alphaLowDiff, &iLow, &iHigh, &bLow, &bHigh, parameterA, parameterB, parameterC, devKernelDiag, cost);
    } else {
      launchSecondOrder(kType, data, labels, nPoints, nDimension, epsilon, cEpsilon, alpha, devF, &alphaHighDiff, &alphaLowDiff, &iLow, &iHigh, &bLow, &bHigh, parameterA, parameterB, parameterC, devKernelDiag, cost);
    }*/
	launchSecondOrder(kType, epsilon, cEpsilon, alpha, devF, &alphaHighDiff, &alphaLowDiff, &iLow, &iHigh, &bLow, &bHigh, parameterA, parameterB, parameterC, devKernelDiag, cost);
	//launchFirstOrder(kType, epsilon, cEpsilon, alpha, devF, &alphaHighDiff, &alphaLowDiff, &iLow, &iHigh, &bLow, &bHigh, parameterA, parameterB, parameterC, devKernelDiag, cost);
  }
  //progress.addIteration(bLow-bHigh); 
  printf("--- %d iterations ---\n", iteration);
  printf("bLow: %f, bHigh: %f\n", bLow, bHigh);
  kp->b = (bLow + bHigh) / 2;
}
void exit_input_error(int line_num){
    fprintf(stderr,"Wrong input format at line %d\n", line_num);
    exit(1);
}
static char* readline(FILE *input){
    int len;
    if(fgets(line,max_line_len,input) == NULL)
        return NULL;
    while(strrchr(line,'\n') == NULL)//do not found '\n', which means we did not read the whole line
    {
        max_line_len *= 2;//max_line_len=2048
        line = (char *) realloc(line,max_line_len);
        len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    return line;
}
int readSvm(const char* filename){
	int inst_max_index, i, j;
	//struct svm_node *x_space;
    FILE *fp = fopen(filename,"r");
    char *endptr;
    char *idx, *val, *label;

    if(fp == NULL)
    {
        fprintf(stderr,"can't open input file %s\n",filename);
        exit(1);
    }
    nPoints = 0;
    nDimension = 0;
    max_line_len = 1024;
    line = Malloc(char,max_line_len);
    while(readline(fp)!=NULL)
    {
        char *p = strtok(line," \t"); // label
        // features
        while(1){
            p = strtok(NULL," \t");
            if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
                break;
            (nDimension)=(nDimension)+1;
        }
        (nDimension)=(nDimension)+1;
        (nPoints)=(nPoints)+1;
    }
    rewind(fp);//set the pointer to the head of the file
    labels = Malloc(DataType,nPoints);
    data = Malloc(struct svm_node *,nPoints);//one pointer for one sample
    x_space = Malloc(struct svm_node,nDimension);//all the features for all samples

    max_index = 0;
    j=0;
    for(i=0;i<nPoints;i++)
    {
        inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
        readline(fp);
        data[i] = &x_space[j];
        label = strtok(line," \t\n");
        if(label == NULL) // empty line
            exit_input_error(i+1);
        labels[i] = strtod(label,&endptr);//get the labels
        if(endptr == label || *endptr != '\0')
            exit_input_error(i+1);
        while(1){
            idx = strtok(NULL,":");
            val = strtok(NULL," \t");

            if(val == NULL)
                break;

            errno = 0;
            x_space[j].index = (int) strtol(idx,&endptr,10);//10 means decimal
            if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
                exit_input_error(i+1);
            else
                inst_max_index = x_space[j].index;

            errno = 0;
            x_space[j].value = strtod(val,&endptr);
            if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                exit_input_error(i+1);

            ++j;
        }

        if(inst_max_index > max_index)
            max_index = inst_max_index;
        x_space[j++].index = -1;
    }
    fclose(fp);
	return 1;
}

void printModel(const char* outputFile, Kernel_params kp, DataType* alpha, DataType epsilon){ 
	printf("Output File: %s\n", outputFile);
	FILE* outputFilePointer = fopen(outputFile, "w");
	if (outputFilePointer == NULL){
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
	if (printDegree)
		fprintf(outputFilePointer, "degree %i\n", kp.degree);
	if (printGamma)
		fprintf(outputFilePointer, "gamma %f\n", kp.gamma);
	if (printCoef0)
		fprintf(outputFilePointer, "coef0 %f\n", kp.coef0);
	fprintf(outputFilePointer, "nr_class 2\n");
	fprintf(outputFilePointer, "total_sv %d\n", nSV + pSV);
	fprintf(outputFilePointer, "rho %.10f\n", kp.b);
	fprintf(outputFilePointer, "label 1 -1\n");
	fprintf(outputFilePointer, "nr_sv %d %d\n", pSV, nSV);
	fprintf(outputFilePointer, "SV\n");
	for (int i = 0; i < nPoints; i++) {
		svm_node * x = data[i];
		if (alpha[i] > epsilon) {
			fprintf(outputFilePointer, "%.10f ", labels[i]*alpha[i]);
			do{
				fprintf(outputFilePointer, "%d:%f ", x->index, x->value);
				x=x+1;
			}while(x->index!=-1);
			fprintf(outputFilePointer, "\n");
		}
	}
	fclose(outputFilePointer);
}

void printHelp() {
  printf("Usage: svmTrain [options] trainingData.svm\n");
  printf("Options:\n");
  printf("\t-o outputFilename\t Location of output file\n");
  printf("Kernel types:\n");
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
  printf("Internal options:\n");
  printf("\t--heuristic, -h\tWorking selection heuristic:\n");
  printf("\t\t0: First order\n");
  printf("\t\t1: Second order\n");
  printf("\t\t2: Random (either first or second order)\n");
  printf("\t\t3: Adaptive (default)\n");
}

float euclid_dist_2(int id, int cluster_max_index, float * center)
{
	float ans=0.0;
    	for(int i=0;i<cluster_max_index;i++)	ans += center[i]*center[i];
	struct svm_node * data_point = data[id]; 
    	while(data_point->index!=-1){
		ans += ((data_point->value)*(data_point->value)-2*(data_point->value)*center[(data_point->index)-1]);
		data_point++;
   	}
    	return(ans);
}

int find_nearest_cluster(int numClusters, int cluster_max_index, int  id, float **clusters){
    int   index, i;
    float dist, min_dist;
    index    = 0;
    min_dist = euclid_dist_2(id, cluster_max_index, clusters[0]);

    for (i=1; i<numClusters; i++) {
        dist = euclid_dist_2(id, cluster_max_index, clusters[i]);/* no need square root */
        if (dist < min_dist) { /* find the min and its array index */
            min_dist = dist;
            index    = i;
        }
    }
    return(index);
}
static int kType = GAUSSIAN;
FILE * testout = fopen("testout","w");

int main( const int argc, const char** argv)  { 
  int currentOption;
  DataType parameterA = -0.125f;
  DataType parameterB = 1.0f;
  DataType parameterC = 3.0f;

  bool parameterASet = false;
  bool parameterBSet = false;
  bool parameterCSet = false;  
  
  SelectionHeuristic heuristicMethod = ADAPTIVE;
  DataType cost = 1.0f;
  
  DataType tolerance = 1e-3f;
  DataType epsilon = 1e-5f;
  char* outputFilename = NULL;
  while (1) {
    static struct option longOptions[] = {
      {"gaussian", no_argument, &kType, GAUSSIAN},
      {"polynomial", no_argument, &kType, POLYNOMIAL},
      {"sigmoid", no_argument, &kType, SIGMOID},
      {"linear", no_argument, &kType, LINEAR},
      {"cost", required_argument, 0, 'c'},
      {"heuristic", required_argument, 0, 'h'},
      {"tolerance", required_argument, 0, 't'},
      {"epsilon", required_argument, 0, 'e'},
      {"output", required_argument, 0, 'o'},
      {"version", no_argument, 0, 'v'},
      {"help", no_argument, 0, 'f'}
    };
    int optionIndex = 0;
    currentOption = getopt_long(argc, (char *const*)argv, "c:h:t:e:o:a:r:d:g:v:f", longOptions, &optionIndex);
    if (currentOption == -1) {
      break;
    }
    int method = 3;
    switch (currentOption) {
    case 0:
      break;
    case 'v':
      printf("GPUSVM version: 1.0\n");
      return(0);
    case 'f':
      printHelp();
      return(0);
    case 'c':
      sscanf(optarg, "%f", &cost);
      break;
    case 'h':
      sscanf(optarg, "%i", &method);
      switch (method) {
      case 0:
        heuristicMethod = FIRSTORDER;
        break;
      case 1:
        heuristicMethod = SECONDORDER;
        break;
      case 2:
        heuristicMethod = RANDOM;
        break;
      case 3:
        heuristicMethod = ADAPTIVE;
        break;
      }
      break;
    case 't':
      sscanf(optarg, "%f", &tolerance);
      break;
    case 'e':
      sscanf(optarg, "%e", &epsilon);
      break;
    case 'o':
      outputFilename = (char*)malloc(strlen(optarg));
      strcpy(outputFilename, optarg);
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
  	readSvm(trainingFilename);
  	printf("nPoints: %d, max_index: %d\n", nPoints, max_index);
	FILE * clusterFile = fopen("clusterFile.kmeans", "r");
	int numClusters, cluster_max_index;
	fscanf(clusterFile, "%d %d", &numClusters, &cluster_max_index);
	printf("numClusters: %d, cluster_max_index: %d\n", numClusters, cluster_max_index);
	DataType** clusters    = (DataType**) malloc(numClusters *             sizeof(DataType*));
    	//assert(clusters != NULL);
    	clusters[0] = (DataType*)  calloc(numClusters * cluster_max_index, sizeof(DataType));
   	//assert(clusters[0] != NULL);
	for (int i=1; i<numClusters; i++)
        	clusters[i] = clusters[i-1] + cluster_max_index;
	for (int i=0; i<numClusters; i++) {
            	for (int j=0; j<cluster_max_index; j++) {
			fscanf(clusterFile, "%f ", &clusters[i][j]);
            	}
        }
	fclose(clusterFile);
	int* membership = (int*) malloc(nPoints * sizeof(int));
	for(int i=0;i<nPoints;i++){
		membership[i] = find_nearest_cluster(numClusters, cluster_max_index, i, clusters);
	}
	FILE * output[numClusters];
	char filename[numClusters][32];
	char fileheader[] = "subdata";
	for(int k=0;k<numClusters;k++){
		char buffer[6];
    		snprintf(buffer, sizeof(char) * 32, "%i", k);
		int i, j;
		for(i=0;fileheader[i]!='\0';i++) filename[k][i] = fileheader[i];
		for(j=0;buffer[j]!='\0';j++, i++) filename[k][i] = buffer[j];
		filename[k][i] = '\0';
	}
	for(int k=0;k<numClusters;k++) output[k] = fopen(filename[k], "w");
	for(int i=0;i<nPoints;i++){
		struct svm_node * data_point = data[i];
		fprintf(output[membership[i]],"%d ", (int)labels[i]);
		while(data_point->index!=-1){
			fprintf(output[membership[i]],"%d:%f ", data_point->index, data_point->value);
			data_point++;
		}
		fprintf(output[membership[i]],"\n");
	}
	for(int k=0;k<numClusters;k++) fclose(output[k]);
	return 0;
}

