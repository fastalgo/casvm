We do not claim this is the most efficient implementation. If you want
to use CA-SVM as a baseline, please modify our code or rewrite the code.
For example, we commented the openmp and simd lines in our code, you may
need to uncomment these lines.

If you want to use our source code, please cite the following paper:

Yang You, James Demmel, Kent Czechowski, Le Song, Rich Vuduc, CA-SVM: 
Communication-Avoiding Support Vector Machines on Distributed Systems, 
in 2015 International Symposium on Parallel & Distributed Processing 
(IPDPS). IEEE.

bibtex:

@inproceedings{casvm2015ipdps,
  title={CA-SVM: Communication-Avoiding Support Vector Machines on Distributed Systems},
  author={You, Yang and Demmel, James and Czechowski, Kent and Song, Le and Vuduc, Rich},
  booktitle={2015 International Symposium on Parallel \& Distributed Processing (IPDPS)},
  organization={IEEE}
}

