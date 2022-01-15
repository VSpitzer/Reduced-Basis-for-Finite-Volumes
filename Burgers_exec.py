#source ../../SOLVERLAB_install/env_SOLVERLAB.sh


import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import dsolve
import scipy.sparse as sparse
import matplotlib.pyplot as plt


import base_RB_imp as brb
import Burgers_base as bb
#import NN_RB_stationary as nrs

a = 0.
b = np.pi
nx = 100



nbComp=1
nt=50
N_h=nx*nbComp
cfl = 1
dlx=(b-a)*1./nx
dlt=dlx/cfl

size_mu=5                        
mu=np.random.rand(size_mu)              
mu=mu*1./np.linalg.norm(mu)



source_term=bb.F
solver=bb.Burgers_solver
jacobian=bb.Burgers_jacobian

arg_st=(a,b)
arg_solver=(a, b, source_term)
arg_jac=(a,b)

arg=[arg_solver,arg_jac,arg_st]


N_s=N_h
eps=10e-7


mu=np.random.rand(size_mu)
mu=mu*1./np.linalg.norm(mu)

#brb.test(mu,N_h,nt,dlx,dlt,solver,arg)

#RB=brb.build_RB(N_s,size_mu,nbComp,N_h,nt,dlx,dlt,solver,arg)
#RB = np.genfromtxt('RB.csv', delimiter=',')


RB_base=brb.time_build_RB(N_s,size_mu,N_h,nt,dlx,dlt,solver,arg)

RB_idx=np.array(np.genfromtxt('RB_idx.csv', delimiter=','),dtype=int)
RB_base_csv=np.genfromtxt('RB_base.csv', delimiter=',')
RB_base=np.array([np.matrix(RB_base_csv[RB_idx[i]:RB_idx[i+1]]) for i in range(len(RB_idx)-1)])

(h,h_hrb)=brb.time_test_HRB(RB_base,mu,N_h,nt,dlx,dlt,solver,jacobian,source_term,arg)


x=np.arange(0,np.pi,dlx)
plt.plot(x,h,color='blue')
plt.plot(x,h,color='red')
plt.show()





