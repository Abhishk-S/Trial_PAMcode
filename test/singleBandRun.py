from frg2d import frgd
import numpy as np

nF=20
beTa=50.0
wMax=50.0
Ns=16
NW=4
NK=0
cutoffR='litim'
maxBasis=32

currentRun=frgd(nF,beTa,wMax,Ns,NW,NK,cutoffR,maxBasis)
nB=1
kX=((2*np.pi)/float(Ns))*np.arange(-Ns/2+1,Ns/2+1,1)
kBx=np.repeat(kX,Ns)
kBy=np.tile(kX,Ns)

disMat=np.zeros((Ns*Ns,nB,nB))
disMat[:,0,0]=-2*np.cos(kBx)-2*np.cos(kBy)-4*0.15*np.cos(kBx)*np.cos(kBy)

uIntMat=np.zeros((nB,nB))
uIntMat[0,0]=4.0

currentRun.initHamiltonian(2,1.0/8,disMat,uIntMat)
fileName='testRunOut'
lMax=6.0
currentRun.runFlow(fileName,lMax=lMax)
