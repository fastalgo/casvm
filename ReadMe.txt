We do not claim this is the most efficient implementation. If you want
to use CA-SVM as a baseline, please modify our code or rewrite the code.
For example, we commented the openmp and simd lines in our code, you may
need to uncomment these lines.

If you want to use our source code, please cite the following paper:

Yang You, James Demmel, Kent Czechowski, Le Song, Rich Vuduc, CA-SVM: 
Communication-Avoiding Support Vector Machines on Distributed Systems, 
in 2015 International Symposium on Parallel & Distributed Processing 
(IPDPS). IEEE.

Bibtex:

@inproceedings{casvm2015ipdps,
  title={CA-SVM: Communication-Avoiding Support Vector Machines on Distributed Systems},
  author={You, Yang and Demmel, James and Czechowski, Kent and Song, Le and Vuduc, Rich},
  booktitle={2015 International Symposium on Parallel \& Distributed Processing (IPDPS)},
  organization={IEEE}
}

report:
Y. You, J. Demmel, K. Czechowski, L. Song, and R. Vuduc, "CA-SVM: Communication-Avoiding Parallel Support Vector Machines on Distributed Systems," EECS Department, University of California, Berkeley, Tech. Rep. UCB/EECS-2015-9, Feb. 2015.

Report BibTeX:
@techreport{You:EECS-2015-9,
    Author = {You, Yang and Demmel, James and Czechowski, Kenneth and Song, Le and Vuduc, Richard},
    Title = {CA-SVM: Communication-Avoiding Parallel Support Vector Machines on Distributed Systems},
    Institution = {EECS Department, University of California, Berkeley},
    Year = {2015},
    Month = {Feb},
    URL = {http://www.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-9.html},
    Number = {UCB/EECS-2015-9},
    Abstract = {We consider the problem of how to design and implement communication-efficient versions of parallel support vector machines, a widely used classifier in statistical machine learning, for distributed memory clusters and supercomputers. The main computational bottleneck is the training phase, in which a statistical model is built from an input data set. Prior to our study, the parallel isoefficiency of a state-of-the-art implementation scaled as W = Omega(P^3), where W is the problem size and P the number of processors; this scaling is worse than even a one-dimensional block row dense matrix vector multiplication, which has W = Omega(P^2).
This study considers a series of algorithmic refinements, leading ultimately to a Communication-Avoiding SVM (CASVM) method that improves the isoefficiency to nearly W = Omega(P). We evaluate these methods on 96 to 1536 processors, and show average speedups of 3 - 16x (7x on average) over Dis-SMO, and a 95% weak-scaling efficiency on six realworld datasets, with only modest losses in overall classification accuracy. The source code can be downloaded at [1].}
}



