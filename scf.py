#!/usr/bin/env python 
import numpy as np
import sys
import argparse
from scipy.special import lpmn, gegenbauer, gamma
from scipy.misc import derivative
from math import factorial,sqrt,pi
import time

np.seterr(all='raise')

parser = argparse.ArgumentParser()
parser.add_argument("-i", help="file name")

parser.add_argument("-lout", help="label output files default to name of input", default="")

parser.add_argument("-aout", help="output projection coeff", action='store_true', default=False)

parser.add_argument("-rb", help="rbasis - rescale radial functions",
                    type=float,default=1.0,metavar="")

parser.add_argument("-nr", help="number of radial basis functions",
                    type=int,default=11,metavar="")

parser.add_argument("-tout", help="output time",
                    type=float,default=1.0,metavar="")

parser.add_argument("-tsnap", help="output time for snap shot, save to file e.g. f128ksnapt0.dat",
                    type=float,default=8.0,metavar="")

parser.add_argument("-dt", help="time step ",
                    type=float,default=2**-4,metavar="")

parser.add_argument("-tend", help="termination time",
                    type=float,default=1.0,metavar="")

parser.add_argument("-rcut", help="rut off value for calculating inertia tensor",
                    type=float,default=10.0,metavar="")

#lpar = parser.add_mutually_exclusive_group(required=True)
#lpar.add_argument('-lm',  help=" e.g. -lm l0 m0 l2  m0 " ,nargs='+', type=int, default=[0,0])
parser.add_argument('-lmax',  help=" maximum value of l includes all m < l (ignored if not set) ", type=int, default=1)

#parser.add_argument('-m', nargs='+', type=int, default=[0])

args = parser.parse_args()

if args.lout ==  "":
	args.lout = args.i
#load data

dat = np.loadtxt(args.i)

# set number of particles and mass (assuming M_tot=1)
n = len(dat)
mass = 1.0/float(n)
      

# Plummer test functions
def ppot(r):
	"plummer model potential"
	return np.reciprocal(np.sqrt(r**2+1.0))

def ppacc(r):
	"plummer model force"
	return np.divide(r,(r**2+1.0)**1.5)
#
# see Hernquist & Ostriker 1992
#
def xi(r):
	return np.divide((r - 1.0),(r + 1.0))

def dxi(r):
	return np.divide(2.0,(r + 1.0)**2)

def k(l,n):
	return 0.5*n*(n + 4.*l + 3.) + (l + 1.)*(2.*l + 1.);

def A(l,n):
	nf = float(factorial(n))
	return (2.**(4.*l + 3.))*gamma(2.*l + 1.5)*sqrt( nf*(n + 2.*l + 1.5)/( k(l, n)*args.rb*gamma(n + 4.*l + 3.) ) ) 

def dcf(r,l):
	cf1 = (2.*l + 1.)*np.divide(r**l , (1. + r)**(2.*l + 2.))
	cf2 = -l*np.divide(r**(l-1) , (1. + r)**(2.*l + 1.))
	return cf1+cf2

alna = np.zeros((args.lmax+1, args.nr))
for li in range(args.lmax+1):
	for nj in range(args.nr):
		alna[li,nj] = A(li,nj)

cofshp= []
for li in range(args.lmax+1):
	tli = []
	cofshp.append(tli)
	for mi in range(li+1):
		tli.append(sqrt( ((2.*li+1.)/(4.*pi))*(factorial(li-mi)/factorial(li+mi)) ))
#access cofshp[li][mi]

poti = []
jpoti = []
#for li in args.l: #range(0,nl):
#	for mi in args.m:

if args.lmax >= 0:
	sphl = []
	for i in range(args.lmax+1):
		for j in range(i+1):
			sphl.append(i)
			sphl.append(j)	
	args.lm = sphl

def possion(x,incpot=False):
	#spherical coordinates
	r = np.sum(x**2,axis=-1)**(1./2)
	angi = np.arctan2(x[:,1],x[:,0])
	cospol = x[:,2]/r
	pol = np.arccos(cospol)
	sinpol = np.sin(pol)
	
	cosangi = np.cos(angi)	
	sinangi = np.sin(angi)	

	# units vectors in spherical coordinates
	utheta = np.zeros_like(x)
	uphi = np.zeros_like(x)
	utheta[:,0] = -sinangi
	utheta[:,1] = cosangi
		
	uphi[:,0] = cosangi*cospol
	uphi[:,1] = sinangi*cospol
	uphi[:,2] = -np.sin(pol)  		

	ur  = x/r[:,None]

	arj = np.zeros(n)
	atj = np.zeros(n) # az
	apj = np.zeros(n)
	


	#projection
	rs  = r/args.rb 
	rxi = xi(rs)
	rsinphi = r*np.sin(pol)
	cot = np.reciprocal(np.tan(pol))

	sphorder = []
	for mi in range(args.lmax+1):
		if mi == 0:
			sphorder.append(np.ones(len(r)))
			continue
		sphorder.append(np.exp(1j*mi*angi))	

	plm={}
	for mi in range(args.lmax+1):
		plmi = 1.0
		if mi > 0:
			plmi = -1.*(2.*mi - 1.)*np.sqrt(1.0 - cospol**2)*plm[(mi-1,mi-1)]
		plm1m=plmi
		plm2m=0.0
		plm[(mi,mi)] = plmi
		for li in range(mi+1,args.lmax+1):
			plmi=(cospol*(2.*li-1.)*plm1m-(li+mi-1.)*plm2m)/(li-mi)
			plm2m=plm1m
			plm1m=plmi
			plm[(li,mi)] = plmi
		
	dplmi = 0.0
	dplm={}
	tc = 1.0/(cospol*cospol-1.0)
	for mi in range(args.lmax+1):
		for li in range(mi,args.lmax+1):
			if li == 0:
				dplmi = 0.0
			elif mi == li:
				dplmi = li*cospol*plm[(li,mi)]*tc
			else:
				dplmi = (li*cospol*plm[(li,mi)]-(li+mi)*plm[(li-1,mi)] )*tc
			dplm[(li,mi)] = dplmi

	#spherical harmonics, use sph[li][mi]
	sph = {}
	for li in range(args.lmax+1):
		for mi in range(0,li+1):
			sph[(li,mi)]= cofshp[li][mi]*plm[(li,mi)]*sphorder[mi]
		
	#spherical harmonics derivative with respect to polar angle 
	dsph = {}
	for li in range(args.lmax+1):
		for mi in range(0,li+1):
			dsph[(li,mi)] = cofshp[li][mi]*dplm[(li,mi)]*sphorder[mi]*(-sinpol) 
		

	pottot = 0.0
	coefpot = []	
	for li in range(args.lmax+1):
		gcof =  2.*li + 1.5
		cf = np.divide(-1.*rs**li , (1. + rs)**(2.*li + 1.))
		for ni in range(args.nr):			
			if ni == 0:
				gegi = np.ones(len(r))
				gegm2 = gegi
			elif ni == 1:
				gegi = 2.*gcof*rxi
				gegm1 = gegi				
			else:	
				gegi = (1./ni)*( 2.*(ni+gcof-1.)*rxi*gegm1  - ( ni+2.*gcof-2.)*gegm2 )
				gegm2 = gegm1
				gegm1 = gegi

			# gegenbauer derivative
			if ni == 0:
				dgegi = np.zeros(len(r))
			elif ni ==1:
				dgegi  = np.ones(len(r))
				dgegm2 = dgegi
			elif ni ==2:
				dgegi = 2.*(gcof+1)*rxi
				dgegm1 = dgegi
			else:
				dgegi = (1./(ni-1))*( 2.*(ni+gcof-1.)*rxi*dgegm1  - ( ni+2.*gcof-1.)*dgegm2 )
				dgegm2 = dgegm1
				dgegm1 = dgegi
				
			
			#gc = gegenbauer(ni,gcof)
			#rint(gegi[:4],x[1,1])#gc(rxi[:4]))	

			rf = cf*gegi
			#print(rf)
			for mi in range(li +1):
				sphi =  sph[(li,mi)] 
				poti = alna[li,ni]*rf*sphi
			
				a = (poti.sum()).conjugate()*mass
				if incpot:
					coefpot.append(a)			
				if mi != 0:
					a*=2

				pottot += np.real( a*(poti.sum())*mass)			
				arj += np.real( a*alna[li,ni]*sphi*(cf*2.*(gcof)*dgegi*dxi(rs) + dcf(rs,li)*gegi) )/args.rb 
				atj += np.real( a*(1.j*mi*poti))/rsinphi 
				apj += np.real( a*alna[li,ni]*rf*dsph[(li,mi)])/rs

	atot = np.zeros_like(x)
	#print(atot.__array_interface__['data'][0])

	
	if incpot:
		atot = ur*arj[:,None]  + utheta*atj[:,None] + uphi*apj[:,None]	
		return 0.5*pottot, np.real(atot), coefpot

	else:
		atot = ur*arj[:,None]  + utheta*atj[:,None] + uphi*apj[:,None]	
		return atot


# initial values
t = 0.0
tout = 0.0
tsnap = args.tsnap
dt = args.dt
x = dat[:,1:4]	
v = dat[:,4:]
start = time.time()

def writeout(t,x,v,potj,cofa):

	#write coeff form proection
	if args.aout:
		fi.write(f' {t} ')
		for ci in cofa:
			fi.write(f' {ci} ')
		fi.write('\n')

	if t == 0.0:
		print("# Time 	Energy 	Kinetic   Potentail Amp-m2     L_z     L_tot    time_seconds ")
	ke = 0.5*mass*(v**2).sum()
	#potj, atot =  possion(x,incpot=True)	
	angi = np.arctan2(x[:,1],x[:,0])
	A = np.sqrt((np.sin(-2.*angi).sum())**2+(np.cos(-2.*angi).sum())**2)
	angA = np.arctan2(np.sin(-2.*angi).sum(),np.cos(-2.*angi).sum())
	e= -potj + ke
	L = mass*np.cross(x,v)
	L = np.sum(L, axis=0)
	xi,yi,zi = x[:,0], x[:,1] ,x[:,2]

	Ixx = mass*( xi[xi<args.rcut]**2 ).sum()
	Iyy = mass*( yi[yi<args.rcut]**2 ).sum()
	Izz = mass*( zi[zi<args.rcut]**2 ).sum()
	print(" {} {:8.7f} {:8.7f} {:8.7f} {:5.4E} {:5.4E} {:8.7F} {:8.7f} {:4.3f} {:4.3f} {:4.3f} {:5.1f} ".format(t,e,ke,potj,A*mass,angA, L[2], np.linalg.norm(L), Ixx, Iyy, Izz, time.time()-start))
	#print(np.sum(v, axis=0)/len(v),np.sum(x, axis=0)/len(x),len(v))

potj, atot, cofa =  possion(x,incpot=True)

if args.aout:
	fi = open(f'{args.lout}coef','w')
writeout(t,x,v,potj,cofa)
tout += args.tout
#fi = open(f'{args.lout}coef','w')

while t <= args.tend:	

		x += dt*v + 0.5*dt*dt*atot
		
		atoto = np.copy(atot)

		if t >= tout:
			potj, atot, cofa =  possion(x,incpot=True)
		else:
			atot =  possion(x)
		
		v +=  0.5*dt*(atot+atoto)
		
		if t >= tout:
			writeout(t,x,v,potj,cofa)
			tout += args.tout 

		if t >= tsnap:
			np.savetxt("{}snapt{}.dat".format(args.lout,t),dat)
			tsnap += args.tsnap
		t += dt

#write last snapshot even if not tout
if t != tout:
	writeout(t,x,v,potj,cofa)
if args.aout:
	fi.close()
