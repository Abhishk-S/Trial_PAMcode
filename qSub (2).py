import os 
import tempfile 
import subprocess

TEMPLATE_CURRENT = """
#!/bin/bash -l
#$ -l h_rt=16:00:00
#$ -pe omp 1

export MCR_CACHE_ROOT=$TMPDIR
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
module load anaconda2 
source activate pyNumbaF 
{script}

source deactivate

"""

TEMPLATE_PYTHON = """
import numpy as np 
from frgd.frg2d import frg 

nF=24
wMax=100.0
NB=1
nDim=3
cutoffR='litim'

{script}

currentRun=frg(nF,beTa,wMax,Ns,NW,NK,NB,cutoffR,nDim,nL)
nB=1
kX=((2*np.pi)/float(Ns))*np.arange(-Ns/2+1,Ns/2+1,1)
kBx=np.tile(kX[:,None,None],(1,Ns,Ns))
kBy=np.tile(kX[None,:,None],(Ns,1,Ns))
kBz=np.tile(kX[None,None,:],(Ns,Ns,1))

disMat=np.zeros((Ns,Ns,Ns,nB,nB))
disMat[:,:,:,0,0]=-2*np.cos(kBx)-2*np.cos(kBy)-2*np.cos(kBz)
disMat=np.reshape(disMat,(Ns*Ns*Ns,nB,nB))

vCF=np.zeros((Ns,Ns,Ns,nB,nB))
vCF[:,:,:,0,0]=V0+2*V1*(np.cos(kBx)+np.cos(kBy)+np.cos(kBz))+4*V2*np.cos(kBx)*np.cos(kBy)*np.cos(kBz)
vCF=np.reshape(vCF,(Ns*Ns*Ns,nB,nB))

uIntMat=np.zeros((nB,nB))
uIntMat[0,0]=uU

currentRun.initFreePeriodicHamilt(disMat,xDopC,vCF,xDopF)
currentRun.initInteractions(uIntMat)
fileName=outputName
currentRun.runFlow(fileName)
"""

def submit_py(code,outputName,nLoop,inputTemp,inputUint,inputXdopC,inputXdopF,inputCFhopping0,inputCFhopping1,inputCFhopping2,NW,NK,N):
    pythonScript='nL='+nLoop+'\n'+'beTa='+inputTemp+'\n'+'uU='+inputUint+'\n'+\
        'V0='+inputCFhopping0+'\n'+'V1='+inputCFhopping1+'\n'+'V2='+inputCFhopping2+'\n'+'xDopC='+inputXdopC+'\n'+'xDopF='+inputXdopF+'\n'+\
        'NW='+NW+'\n'+'NK='+NK+'\n'+'Ns='+N+'\n'+'outputName=\''+outputName+'\'\n'
    if os.path.exists(code+'.py'):
        os.remove(code+'.py')

    open(code+'.py','w').write(TEMPLATE_PYTHON.format(script=pythonScript))
    
    bashScript='python '+code+'.py'
    
    if os.path.exists('JobPY.qsub'):
        os.remove('JobPY.qsub')

    open('JobPY.qsub','w').write(TEMPLATE_CURRENT.format(script=bashScript))
    subprocess.call("qsub JobPY.qsub",shell=True)


