#!/usr/bin/env python
# coding: utf-8
#source ../../SOLVERLAB_install/env_SOLVERLAB.sh


import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import dsolve
import scipy
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import time
#import ecb_base as becb

"""
solver: 
	* appel: solver(mu,N_h,N_t,dlx,dlt,arg) pour
		- mu le paramétrage
		- N_h, N_t, dlx et dlt le nombre de cells, le nombre de pas de temps et les pas associés
		- arg tuple de tout autre paramètre d'entrée spécifique à la fonction solveur employée
	* retourne: array de la solution attendue
jacobian:
	* appel: jacobian(mu,N_h,N_t,dlx,dlt,h_j,h,f,arg) pour
		- mu le paramétrage
		- N_h, N_t, dlx et dlt le nombre de cells, le nombre de pas de temps et les pas associés
		- h_j le termé auquel est appliqué la jacobienne, format array, pour un schéma implicite l'itération de Newton courante. Format array
		- h pour un schéma implicite la solution au pas de temps précédent. Format array
		- f le terme source, format array
		- arg tuple de tout autre paramètre d'entrée spécifique à la fonction jacobienne employée
	* retourne: tuple (matrice de la jacobienne évaluée en h_j, résidu)
source_term: 
	* appel: source_term(mu,N_h,N_t,dlx,dlt,arg) pour
		- mu le paramétrage
		- N_h, N_t, dlx et dlt le nombre de cells, le nombre de pas de temps et les pas associés
		- args tuple de tout autre paramètre d'entrée spécifique à la fonction solveur employée
	* retourne: array du terme source, format array


args est la liste des arguments des fonctions solver, jacobian, source_term, au format suivant: arg=[arg_solver,arg_jac,arg_st]
"""


#Construction de la base réduite à partir du tableau d'échantillons tab_h et d'une précision eps_rb
def SVD_POD(tab_h,eps_rb):
	M=np.matrix(tab_h).T
	(u,s,vh)=np.linalg.svd(M, full_matrices=False)
	err=0
	i=0
	while(err<eps_rb and i<len(s)): #Pour éviter des erreurs d'absorbtion/elimination, commencer par les plus faibles valeurs singulières
		i+=1
		err+=s[-i]**2
	return np.matrix(u.T[:-i+1])




#Construction de la base réduite par échantillonnage de N_s solutions au temps N_t*dlt
def build_RB(N_s,size_a,nbComp,N_h,N_t,dlx,dlt,solver,*args):
	#sampling
	M=np.zeros((N_s,N_h))
	for i in range(N_s):
		a=np.random.rand(size_a)
		a=a*1./np.linalg.norm(a)
		M[i]=solver(a,N_h,N_t,dlx,dlt,args[0]) 
	M=np.matrix(M).T
	#SVD
	(u,s,vh)=np.linalg.svd(M, full_matrices=False)
	eps=1e-4
	err=0
	i=0
	#POD
	while(err<eps and i<len(s)): #Pour éviter des erreurs d'absorbtion/elimination, commencer par les plus faibles valeurs singulières
		i+=1
		err+=s[-i]**2
	RB=np.matrix(u.T[:-i+1])
	print("Reduction de modele: dimension reduite de",repr(len(s))," à ",repr(len(s)-i+1))
	np.savetxt('RB.csv', RB, delimiter=',')
	return RB
	
	

#Idem mais à chaque pas de temps
def time_build_RB(N_s,size_a,N_h,N_t,dlx,dlt,solver,*args):
	RB_base=[]
	RB_base.append(np.identity(N_h))
	eps_rb=1e-12
	eps_stat=1e-3
	cpt=0

	tab_a=np.random.rand(N_s,size_a)
	tab_a=np.array([a*1./np.linalg.norm(a) for a in tab_a])

	tab_h=np.array([solver(a,N_h,1,dlx,dlt,args[0]) for a in tab_a])
	idx=np.array(range(len(tab_a)),dtype=int)
	
	itr=1
	print("Debut de l'echantillonnage")
	while len(idx)>0 and itr<N_t:
		print("Pas de temps: ",itr)
		RB=np.matrix(RB_base[-1])
		idx_ref=np.array(idx,dtype=int)
		for i in idx_ref:

			a=tab_a[i]
			h_ref=np.reshape(np.array((RB.T*RB).dot(tab_h[i])),(N_h,))
			h=solver(a,N_h,1,dlx,dlt,args[0],h_ref)
			if np.linalg.norm(h-h_ref)>eps_stat:
				tab_h[i]=np.array(h)

			else:
				j=np.where(idx==i)[0][0]
				idx=np.concatenate((idx[:j],idx[j+1:]),axis=0)
				if i>0:
					tab_h[i]=tab_h[i-1]
				else:
					tab_h[i]=tab_h[i+1]
	
		# normalisation des échantillons pour travailler sur une erreur relative plutot qu'absolue?
		#M=np.matrix([h*1./np.linalg.norm(h) for h in tab_h]).T
		M=np.matrix(tab_h).T
		(u,s,vh)=np.linalg.svd(M, full_matrices=False)
		err=0
		i=0
		#POD
		while(err<eps_rb**2 and i<len(s)): #Pour éviter des erreurs d'absorbtion/elimination, commencer par les plus faibles valeurs singulières
			i+=1
			err+=s[-i]**2
		RB_base.append(np.matrix(u.T[:-i+1]))
		cpt+=len(s)-i+1
		
		itr+=1
		
	
	RB_len=np.array([rb.shape[0] for rb in RB_base])
	RB_idx=np.array([0]+[sum(RB_len[:i+1]) for i in range(len(RB_len))])
	RB_base_csv=np.concatenate(RB_base,axis=0)
	
	#enregistrement des bases réduites dans des fichiers .csv à charger ultérieurement
	np.savetxt('RB_idx.csv', RB_idx, delimiter=',')
	np.savetxt('RB_base.csv', RB_base_csv, delimiter=',')
	print("Reduction de modele: dimension reduite en moyenne de: ",cpt*1./len(RB_base))
	return RB_base
	


# OBSOLÈTE
#Solveur qui teste la résolution des bases réduites, sans gain de temps, avec application de la base réduite au pas de temps N_t uniquement
def solver_RB(RB,mu,N_h,N_t,dlx,dlt,cfl,init,jacobian,source_term,args):
	j=0
	
	newton_max=50
	eps=10e-5
	nbComp=4
	
	h0=np.reshape(init(mu,N_h,N_t,cfl,args),(N_h,1))
	h=np.reshape(RB.dot(init(mu,N_h,N_t,cfl,args)),(RB.shape[0],1))
	f=source_term(mu,N_h,N_t,dlx,dlt,args[2])
	f=np.reshape(f,(f.shape[0],1))
	#time loop
	while(j<N_t):
		dh=np.array(h)
		hk=np.array(h)
		residu=1.
		k=0
		#Newton loop
		while k<newton_max and residu > eps:
			print((k,j))
			h_j=np.reshape(np.array(RB.T*hk),(N_h,1))
			print(np.linalg.norm(h_j-h0))
			(M,Rh)=jacobian(mu,N_h,N_t,dlx,dlt,h_j,h,f,args[1])  #M et Rh de type np.array
			Rhs=np.array(np.reshape((RB.T*RB).dot(Rh),(N_h,)))[0]
			dh=np.array(np.dot(RB,sparse.linalg.spsolve(M,Rhs) ) )
			print(dh.shape)
			hk+=dh
			
			residu=np.linalg.norm(dh)/np.linalg.norm(h)
			k+=1
		h=np.array(hk)
		j+=1
	res=(RB.T).dot(h)
	return np.array(np.reshape(res,(res.shape[0],)))[0]


# OBSOLÈTE
#Solveur qui teste la résolution des bases réduites avec DEIM, avec application de la base réduite au pas de temps N_t uniquement
def solver_HRB(HRB,RBH,RB,idx_vsn,idx_val,mu,N_h,N_t,dlx,dlt,cfl,init,jacobian,source_term,args): 
	j=1
	N_hrb=RBH.shape[0]
	
	newton_max=50
	eps=10e-5
	
	h=np.reshape((RBH*RB).dot(init(mu,N_h,N_t,cfl,args)),(N_hrb,1))
	f=source_term(mu,N_h,N_t,dlx,dlt,args[2])[idx_vsn] 
	f=np.reshape(f,(f.shape[0],1))
	
	P=RBH.dot(np.linalg.inv(HRB))
	while(j<N_t):
		dh=np.array(h)
		hk=np.array(h)
		residu=1.
		k=0
		#Newton loop
		while k<newton_max and residu > eps:
			h_j=np.reshape(np.array(hk),(N_hrb,1))
			(M,Rh)=jacobian(mu,N_hrb,N_t,dlx,dlt,h_j,h,f,args[1])  #M et Rh de type np.array
			dh=sparse.linalg.spsolve(M,Rh)[idx_val]
			dh=P.dot(dh)
			hk+=dh
			
			residu=np.linalg.norm(dh)/np.linalg.norm(h)
			k+=1
		h=np.array(hk)
		j+=1

	res=(RB.T).dot(np.linalg.solve(HRB,h[idx_val]))
	return np.array(np.reshape(res,(res.shape[0],)))[0]
	
#Solveur qui teste la résolution des bases réduites avec réseaux de neurones pour une base réduite à chaque pas de temps
def time_solver_RB_NN(RB,mu,N_h,N_t,dlx,dlt,cfl,solver,jac_model,source_term,args):
	j=1
	
	newton_max=50
	eps=10e-5
	
	h=np.reshape(RB.dot(solver(mu,N_h,N_t,dlx,dlt,args)),(RB.shape[0],1))
	f=source_term(mu,N_h,N_t,dlx,dlt,args[2])
	f=np.reshape(RB.dot(f),(RB.shape[0],1))
	#time loop
	while(j<N_t):
		dh=np.array(h)
		hk=np.array(h)
		residu=1.
		k=0
		#Newton loop
		while k<newton_max and residu > eps:
			h_j=np.reshape(np.array(RB.T*hk),(N_h,))
			(M,Rh)=jac_model(RB.shape[0],h)  #M de type np.array
			dh=np.array(np.linalg.spsolve(M,Rh))
			hk+=dh
			
			residu=np.linalg.norm(dh)/np.linalg.norm(h)
			k+=1
		h=np.array(hk)
		j+=1
	res=(RB.T).dot(h)
	return np.array(np.reshape(res,(res.shape[0],)))[0]

#Solveur qui teste la résolution des bases réduites, sans gain de temps, pour une base réduite à chaque pas de temps
def time_solver_RB(RB_base,mu,N_h,ntmax,dlx,dlt,solver,jacobian,source_term,args):
	j=1

	newton_max=50
	eps_newton=1e-5
	eps_stat=1e-6
	

	nbComp=4

	h=np.reshape(RB_base[0].dot(solver(mu,N_h,1,dlx,dlt,args)),(RB_base[0].shape[0],1))

	f=source_term(mu,N_h,dlx,dlt,args[2])
	f_ref=np.reshape(f,(f.shape[0],1))
	
	
	ntmax=len(RB_base)
	#time loop
	while(j<ntmax):
		if j<len(RB_base):
			RB=np.matrix(RB_base[j-1])
			RB_n=np.matrix(RB_base[j])
		else:
			RB=np.matrix(RB_base[-1])
			RB_n=np.matrix(RB_base[-1])

		h=(RB.T).dot(h)

		hk=RB_n.dot(h)
		dh=np.array(hk)
		
		residu=1.
		k=0
		#Newton loop
		while k<newton_max and residu > eps_newton:
			hk_j=np.reshape(np.array(RB_n.T*hk),(N_h,))
			f_j=np.reshape(np.array(f_ref),(N_h,))
			h_j=np.reshape(np.array(h),(N_h,))
			(M,Rhs)=jacobian(mu,N_h,dlx,dlt,hk_j,h_j,f_j,args[1])
			M=RB_n*M*RB_n.T
			Rhs=RB_n.dot(Rhs).T
			dh=np.linalg.solve(M,Rhs)
			
			hk+=dh
			
			residu=np.linalg.norm(dh)
			
			k+=1

		if np.linalg.norm(h-(RB_n.T).dot(hk))<eps_stat:
			h=np.array(hk)
			break

		h=np.array(hk)
		j+=1

	res=(np.matrix(RB_base[-1]).T).dot(h)
	return np.array(np.reshape(res,(res.shape[0],)))[0]
	
#Solveur qui teste la résolution des bases réduites avec DEIM pour une base réduite à chaque pas de temps
def time_solver_HRB(RB_base,RB_step_base,HRB_step_base,idx_base,idx_vsn_base,idx_val_base,mu,N_h,N_t,dlx,dlt,solver,jacobian,source_term,args,*h_ref):
	nbVoisins=2

	j=1
	N_hrb=idx_vsn_base[0].shape[0]	
	idx_vsn=idx_vsn_base[0]
	
	newton_max=50
	eps_newton=1e-5
	eps_stat=1e-6
	nbComp=1

	if len(h_ref)>0:
		h=np.reshape(RB_base[0].dot(h_ref[0]),(RB_base[0].shape[0],1))
	else:
		h=np.reshape(RB_base[0].dot(solver(mu,N_h,1,dlx,dlt,args)),(RB_base[0].shape[0],1))
	
	f=source_term(mu,N_h,dlx,dlt,args[2])
	f=np.reshape(f,(f.shape[0],1))
	
	
	ntmax=len(RB_base)
	while(j<N_t):
		if j<len(RB_base):
			N_hrb=idx_vsn_base[j].shape[0]	
			idx_vsn=idx_vsn_base[j]
			idx_val=idx_val_base[j]
			idx=idx_base[j]
			
			RB=np.matrix(RB_base[j-1])
			RB_n=np.matrix(RB_base[j])
			
			RBH_n=np.matrix((RB_n.T)[idx_vsn])
		
		else:
			N_hrb=idx_vsn_base[-1].shape[0]	
			idx_vsn=idx_vsn_base[-1]
			idx_val=idx_val_base[-1]
			idx=idx_base[-1]
			
			RB=np.matrix(RB_base[-1])
			RB_n=np.matrix(RB_base[-1])
			
			RBH_n=np.matrix((RB_n.T)[idx_vsn])
			
		#hk=(RB_n*RB.T).dot(h)
		hk=RB_step_base[j-1].dot(h)
		dh=np.array(hk)
		
		f_ref=f[idx_vsn]
		#h_ref=(RB.T).dot(h)[idx_vsn]
		h_ref=(HRB_step_base[j-1]).dot(h)
		
		residu=1.
		k=0
		#Newton loop
		while k<newton_max and residu > eps_newton:
			hk_j=np.reshape(np.array(RBH_n*hk),(-1,))
			f_j=np.reshape(np.array(f_ref),(-1,))
			h_j=np.reshape(np.array(h_ref),(-1,))
			(M,Rhs)=jacobian(mu,N_hrb,dlx,dlt,hk_j,h_j,f_j,args[1])
			M=M[idx_val,:]*RBH_n
			Rhs=np.matrix(Rhs[idx_val]).T
			dh=np.linalg.solve(M,Rhs)
			hk+=dh
			residu=np.linalg.norm(dh)
			k+=1

		h=np.array(hk)
		j+=1
		
	res=(np.matrix(RB_base[-1]).T).dot(h)
	return np.array(np.reshape(res,(res.shape[0],)))[0]
	#return np.reshape(np.array(h),(-1,))

def error(h_rb,h):
    error=np.linalg.norm(h_rb-h)/np.linalg.norm(h)
    return error

##Hyper reduction

#Extraction d'une matrice carré inversible de conditionnement faible par QR
#On ne considere par les termes aux bords pour ignorer les conditions aux bords lors de la résolution.
def extract_inv(M,nbComp):
	(k,n)=M.shape
	M=M[:,nbComp:n-nbComp]
	(k,n)=M.shape
	
	(Q,R,P)=scipy.linalg.qr(M, pivoting=True)
	idx=np.array(P[:k],dtype=int)
	
	HRB=np.matrix(M[:,idx]).T
	idx=idx+nbComp
	return (HRB,np.array(idx,dtype=int),np.linalg.cond(HRB))

#Construction des indices à considérer pour la résolution, i.e. ceux des cellules d'intérêts et de leur premier voisinage
#On ignore les conditions aux bords en ne considérant que des cells intérieures dans idx, et en imposant dans idx_vsn les cells aux bords pour du non-intrusif
def build_vsn(idx,n,nbComp):
	tmp=np.ones((len(idx),3*nbComp))
	mat=np.diag(range(-nbComp,2*nbComp))
	tmp=np.matrix((tmp.dot(mat)).T+nbComp*(idx/nbComp)).T
	idx_vsn=np.array(np.reshape(tmp,(3*nbComp*len(idx),)))[0]
	idx_vsn=np.concatenate((np.arange(3*nbComp),idx_vsn,np.arange(n-3*nbComp,n)),axis=0)
	
	idx_val=np.array([nbComp*(3*i+1+3)+idx[i]%nbComp for i in range(len(idx))],dtype=int)
	#idx_val=np.array([nbComp*(3*i+1)+idx[i]%nbComp for i in range(len(idx))],dtype=int)
	return (np.array(idx_vsn,dtype=int),idx_val,idx)

#Constrution de l'hyper-reduction DEIM
def build_HRB(RB,nbComp):
	(HRB,idx,cond)=extract_inv(np.array(RB),nbComp)
	print("Conditionnement de l'hyperreduction: ",cond)
	(idx_vsn,idx_val,idx)=build_vsn(idx,RB.shape[1],nbComp)
	RBH= np.matrix(RB).T[idx_vsn]
	return (HRB,RBH,idx_vsn,idx_val,idx)
    
#Construction des matrices HRB à chaque pas de temps
def time_build_HRB(RB_base,nbComp):
	idx_base=[]
	idx_val_base=[]
	idx_vsn_base=[]
	max_cond=1
	RB_tmp=RB_base[0].T
	for RB in RB_base:
		(HRB,idx,cond)=extract_inv(np.array(RB),nbComp)
		if cond>max_cond :
			max_cond=cond
		(idx_vsn,idx_val,idx)=build_vsn(idx,RB.shape[1],nbComp)
		idx_base.append(idx)
		idx_val_base.append(idx_val)
		idx_vsn_base.append(idx_vsn)
	print("Conditionnement de l'hyperreduction: ",max_cond)
	return (idx_base,idx_vsn_base,idx_val_base)

# Test

#test du solveur standard
def test(mu,N_h,N_t,dlx,dlt,solver,args,*h_ref):
	t0=time.time()
	h1=solver(mu,N_h,N_t,dlx,dlt,args,*h_ref)
	print("temps execution: ",time.time()-t0)
	return h1
	

	
#test du solveur base réduite sans gain de temps	
def time_test_RB(RB_base,mu,N_h,ntmax,dlx,dlt,solver,jacobian,source_term,args):
	RB=np.matrix(RB_base[-1])
	

	h=solver(mu,N_h,ntmax,dlx,dlt,args)
	print("solveur exec terminé")
	h_proj=np.array(np.reshape((RB.T*RB).dot(h),(h.shape[0],)))[0]
	
	
	h_rb=time_solver_RB(RB_base,mu,N_h,ntmax,dlx,dlt,solver,jacobian,source_term,args)
	print("Erreur relative approximation RB: ",vector_error(h,h_rb))
	print("Erreur relative de projection: ",vector_error(h,h_proj))
	return
	
#test du solveur HRB 
def time_test_HRB(RB_base,mu,N_h,ntmax,dlx,dlt,solver,jacobian,source_term,args):
	(idx_base,idx_vsn_base,idx_val_base)=time_build_HRB(RB_base,1)
	print("Dimension reduite en moyenne de ",N_h," a ", sum([rb.shape[0] for rb in RB_base[1:]])*1./len(RB_base[1:]))
	RB=np.matrix(RB_base[-1])
	idx=idx_base[-1]
	HRB=np.matrix(RB[:,idx]).T

	
	t0 = time.time()
	h=solver(mu,N_h,ntmax,dlx,dlt,args)
	print("Temps execution solveur: ",time.time()-t0)
	h_proj=np.array(np.reshape((RB.T*RB).dot(h),(h.shape[0],)))[0]
	h_hyper=np.array(np.reshape((RB.T).dot(np.linalg.solve(HRB,h[idx])),(h.shape[0],)))[0]
	
	h_rb=time_solver_RB(RB_base,mu,N_h,ntmax,dlx,dlt,solver,jacobian,source_term,args)
	
	RB_step_base=[RB_base[i+1]*RB_base[i].T for i in range(len(RB_base)-1)]
	HRB_step_base=[(RB_base[i].T)[idx_vsn_base[i+1]] for i in range(len(RB_base)-1)]
	t0 = time.time()
	h_hrb=time_solver_HRB(RB_base,RB_step_base,HRB_step_base,idx_base,idx_vsn_base,idx_val_base,mu,N_h,ntmax,dlx,dlt,solver,jacobian,source_term,args)
	print("Temps execution hrb: ",time.time()-t0)
	
	print("Erreur relative approximation RB a solution exacte: ",error(h,h_rb))
	print("Erreur relative approximation HRB a solution exacte: ",error(h,h_hrb))
	print("...................................................................................................")
	print("Erreur relative approximation RB a approximation HRB: ",error(h_rb,h_hrb))
	print("Erreur relative hyper-reduction exacte a la reduction exacte: ",error(h_proj,h_hyper) )
	print("Erreur relative approximation RB a reduction exacte: ",error(h_rb,h_proj))
	print("Erreur relative approximation HRB a hyper-reduction exacte: ",error(h_hrb,h_hyper))
	print("...................................................................................................")
	print("Erreur relative de reduction exacte: ",error(h,h_proj))
	print("Erreur relative d'hyper-reduction exacte: ",error(h,h_hyper))
	return (h,h_hrb)

   
	

    




