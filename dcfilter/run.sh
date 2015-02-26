#!/bin/bash -l
rm dcfilter
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

#mpiicpc -o dcfilter -openmp -simd dcfilter.cpp;
mpic++ -o dcfilter dcfilter.cpp;

mpirun -np $node ./dcfilter -c $cost -g $gamma -e $epsilon -t $tolerance -o ijcnn.mdl -q alpha_out -s sv_file ../dataset/ijcnn.r
../libsvm/svm-predict ../dataset/ijcnn.t ijcnn.mdl libsvm_out
