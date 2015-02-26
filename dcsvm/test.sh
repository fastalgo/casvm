#!/bin/bash -l
rm dcsvm
rm *.mdl
rm alpha_out
rm sv_file
rm libsvm_out
node=8
cost=32
gamma=2
epsilon=1e-2
tolerance=1e-1
export KMP_AFFINITY='compact'
echo $KMP_AFFINITY
#export OMP_NUM_THREADS=24
#echo $OMP_NUM_THREADS

#mpiicpc -o dcsvm -openmp -simd dcsvm.cpp;
mpic++ -o dcsvm dcsvm.cpp;

mpirun -np $node ./dcsvm -c $cost -g $gamma -e $epsilon -t $tolerance -o ijcnn.mdl -q alpha_out -s sv_file ../dataset/ijcnn6400.r
#../libsvm/svm-predict ../dataset/ijcnn.t ijcnn.mdl libsvm_out
