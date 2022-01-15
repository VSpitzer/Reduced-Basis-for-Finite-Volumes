import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import dsolve
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import cdmath

import equations_canal_bouillant as ecb
import base_RB_imp as brb

#terme source
def F(mu,N_h,dlx,dlt,args):
	nbComp=1
	N_x=N_h/nbComp
	(a,b)=args
	n=len(mu)
	res=np.zeros(nbComp*N_x)
	for i in range(N_x):
		y=np.array([k*i*dlx for k in range(1,n+1)])*np.pi*1./(2*n*(b-a))
		z=np.array([k*(i+1)*dlx for k in range(1,n+1)])*np.pi*1./(2*n*(b-a))
		res[i]=1./dlx*np.dot(mu/np.array(range(1,n+1)),np.cos(y)-np.cos(z))
	return res*dlt
	
#retourne le tuple (matrice, Right Hand Side)
def Burgers_jacobian(mu,N_h,dlx,dlt,hk,h,F,args):
	nbComp=1
	N_x=N_h/nbComp
	(a,b)=args
	
	v1=np.array(hk)
	v2=np.array(v1[1:]+v1[:-1])*0.5
	v3=-np.array(v2)
	v2[np.where(v2>0)]=0
	v3[np.where(v3>0)]=0
	v1[1:-1]=-v2[1:]-v3[:-1]
	v1[0]=-v2[0]
	v1[-1]=-v3[-1]
	M=sparse.diags([dlt/dlx*np.array(v3),dlt/dlx*np.array(v1)+np.ones(N_x),dlt/dlx*np.array(v2)],[-1,0,1],format="csc")
	
	Rhs=h+F-M.dot(hk)
	
	return (M,Rhs)


#Solveur standard
def Burgers_solver(a,N_x,N_t,dlx,dlt,args,*h_ref):
	( x, y, source_term)=args[0]

	j=0
	x=np.concatenate((np.arange(0,np.pi,dlx),np.array([np.pi])),axis=0)
	h=-1./dlx*(np.cos(x[1:])-np.cos(x[:-1]))
	
	
	if len(h_ref)>0:
		h=np.array(h_ref[0])

	newton_max=50
	eps=1e-5
	
	
	f=np.array(source_term(a,N_x,dlx,dlt,args[2]))
	#time loop
	while(j<N_t):

		dh=np.array(h)
		hk=np.array(h)
		residu=1.
		k=0
		#Newton loop
		while k<newton_max and residu > eps:
			v1=np.array(hk)
			v2=np.array(v1[1:]+v1[:-1])*0.5
			v3=-np.array(v2)
			v2[np.where(v2>0)]=0
			v3[np.where(v3>0)]=0
			v1[1:-1]=-v2[1:]-v3[:-1]
			v1[0]=-v2[0]
			v1[-1]=-v3[-1]
			M=sparse.diags([dlt/dlx*np.array(v3),dlt/dlx*np.array(v1)+np.ones(N_x),dlt/dlx*np.array(v2)],[-1,0,1],format="csc")

			dh=sparse.linalg.spsolve(M,h+f-M.dot(hk))
			
			hk+=dh
			
			residu=np.linalg.norm(dh)
			k+=1
		if np.linalg.norm(h-hk)<eps:
			h=np.array(hk)
			break
			
		h=np.array(hk)
		j+=1

	return h
	
	
	
def Flux_Godunov(u_l, u_r):
	if (u_l==u_r):
		flux = 0.5*u_l*u_l
	elif (u_l<0 and 0<u_r):
		flux = 0.
	elif (u_l<u_r):
		flux = min(0.5*u_l*u_l,0.5*u_r*u_r)
	elif (u_l>u_r):
		flux = max(0.5*u_l*u_l,0.5*u_r*u_r)
	return flux

def Godunov_solver(a,N_x,N_t,dlx,dlt,args):
	j=0
	x=np.concatenate((np.arange(0,np.pi,dlx),np.array([np.pi])),axis=0)
	h=-1./dlx*(np.cos(x[1:])-np.cos(x[:-1]))
	
	h_tmp=np.zeros(N_x)
	f=np.array(F(a,N_x,dlx,dlt,args[2]))
	#time loop
	while(j<N_t):
		for i in range(N_x):
			if (i==0):
				flux_iminus = 0.5*h[0]*h[0]#Flux at the left Neumann boundary
				flux_iplus = Flux_Godunov(h[i],h[i+1])
			if (i==N_x-1):
				flux_iplus  = 0.5*h[-1]*h[-1]#Flux at the right Neumann boundary
			else:
				flux_iplus = Flux_Godunov(h[i],h[i+1])
			pass
			h_tmp[i] = f[i]+h[i]-dlt/dlx*(flux_iplus-flux_iminus)
			flux_iminus = flux_iplus
		h = h_tmp	
		j+=1
	return h
	
