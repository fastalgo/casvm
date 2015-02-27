#!/bin/bash -l
rm casvm_fcfs
rm *.mdl*
rm subdata*
rm accuracyfile;
rm alpha_out
rm sv_file
rm libsvm_out
rm split
rm counting
rm testout
rm subdata*
rm accuracyfile
rm clusterFile.kmeans
node=8
cost=32
gamma=2
epsilon=1e-2
tolerance=1e-1
export KMP_AFFINITY='compact'
echo $KMP_AFFINITY
#export OMP_NUM_THREADS=24
#echo $OMP_NUM_THREADS

#mpiicpc -o casvm_fcfs -openmp -simd casvm_fcfs.cpp;
mpic++ -o casvm_fcfs casvm_fcfs.cpp;

mpirun -np $node ./casvm_fcfs -c $cost -g $gamma -e $epsilon -t $tolerance -o ijcnn.mdl -q alpha_out -s sv_file ../dataset/ijcnn6400.r

#g++ -o split split.cpp;
#g++ -o counting count.cpp;

#./split ../dataset/ijcnn.t;
#x=0
#while [ $x -le $(( $node - 1 )) ]
#do
  #echo "the model from $x node"
  #../libsvm2/svm-predict ./subdata$x ijcnn.mdl$x libsvm_out;
  #x=$(( $x + 1 ))
#done
#./counting;
