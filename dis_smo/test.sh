rm dissmo
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

#mpic++ -o dissmo -openmp -simd dis_smo.cpp;
mpic++ -o dissmo dis_smo.cpp;

mpirun -np $node ./dissmo -c $cost -g $gamma -e $epsilon -t $tolerance -o ijcnn.mdl ../dataset/ijcnn6400.r
#../libsvm/svm-predict ../dataset/ijcnn.t ijcnn.mdl libsvm_out
