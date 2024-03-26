from frg2d import frgd
import numpy as np

nF=20
beTa=50.0
wMax=50.0
Ns=8
NW=2
NK=0
cutoffR='litim'
maxBasis=32

currentRun=frgd(nF,beTa,wMax,Ns,NW,NK,cutoffR,maxBasis)
nB=2
kX=((2*np.pi)/float(Ns))*np.arange(-Ns/2+1,Ns/2+1,1)
kBx=np.repeat(kX,Ns)
kBy=np.tile(kX,Ns)

txy=-1.0
tz=-0.05
txyz=0.4
ez0=2.0

eX=2*txy*(np.cos(kBx)+np.cos(kBy))
eY=2*tz*(np.cos(kBx)+np.cos(kBy))
eXY=2*txyz*(np.cos(kBx)-np.cos(kBy))

disMat=np.zeros((len(kBx),nB,nB),dtype=np.complex_)
disMat[:,0,0]=eY+ez0
disMat[:,1,1]=eX
disMat[:,0,1]=eXY
disMat[:,1,0]=eXY

uIntMat=np.zeros((nB,nB))
uIntMat[:,:]=[[6.922,3.99],[3.99,4.508]]
jIntMat=np.zeros((nB,nB))
jIntMat[0,1]=0.726
jIntMat[1,0]=0.726
vIntMat=np.zeros((nB,nB))
vIntMat[:,:]=[[0.764,0.833],[0.833,0.901]]

currentRun.initHamiltonian(2,1.0/8,disMat,uIntMat,vIntMat,jIntMat)
fileName='2BandTestOut'
lMax=0.1
currentRun.runFlow(fileName,lMax=lMax)
