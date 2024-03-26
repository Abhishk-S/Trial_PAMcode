from frgFlow import fRG2D
from vertexF import vertexR
from tempfile import TemporaryFile
import numpy as np

def frgRun(args):
    """
    def uF(kPP,kPH,kPHE):
            k1x=0.5*(kPP[0]+kPH[0]-kPHE[0])
            k2x=0.5*(kPP[0]-kPH[0]+kPHE[0])
            k3x=0.5*(kPP[0]+kPH[0]+kPHE[0])

            k1y=0.5*(kPP[1]+kPH[1]-kPHE[1])
            k2y=0.5*(kPP[1]-kPH[1]+kPHE[1])
            k3y=0.5*(kPP[1]+kPH[1]+kPHE[1])

            return (args.couplingU[0]+2*args.couplingV[0]*(np.cos(kPH[0]) + np.cos(kPH[1])))

    def uFX(kPP,kPH,kPHE):
            k1x=0.5*(kPP[0]+kPH[0]-kPHE[0])
            k2x=0.5*(kPP[0]-kPH[0]+kPHE[0])
            k3x=0.5*(kPP[0]+kPH[0]+kPHE[0])

            k1y=0.5*(kPP[1]+kPH[1]-kPHE[1])
            k2y=0.5*(kPP[1]-kPH[1]+kPHE[1])
            k3y=0.5*(kPP[1]+kPH[1]+kPHE[1])

            return args.couplingU[0]+4*args.couplingV[0]*(np.cos(k1x)+np.cos(k2x)+np.cos(k3x)+np.cos(k1x+k2x-k3x)+\
                np.cos(k1y)+np.cos(k2y)+np.cos(k3y)+np.cos(k1y+k2y-k3y))
    """
    """
    filename=''.join(['couplingFRG2DU',str(int(args.couplingU[1])),'V',str(int(args.couplingV[1])),'X',
                str(int(args.fillingN[1])),'R',args.cutoffR,'UV','nf',str(args.nPatches),'nL',str(args.nL),'NW',str(args.NW),
                  'NK',str(args.NK),'nSites',str(args.Ns),'tP',str(int(args.tP[1])),'nB',str(args.nBasis),'bT',str(int(args.beTa[1]))])

    nB=1
    N=args.Ns
    kX=((2*np.pi)/float(N))*np.arange(-N/2+1,N/2+1,1)
    kBx=np.repeat(kX,N)
    kBy=np.tile(kX,N)
    disMat=np.zeros((len(kBx),nB,nB),dtype=np.complex_)
    disMat[:,0,0]=-2*np.cos(kBx)-2*np.cos(kBy)+4*args.tP[0]*np.cos(kBx)*np.cos(kBy)

    uintMat=np.zeros((nB,nB))
    uintMat[0,0]=args.couplingU[0]
    vintMat=np.zeros((nB,nB))+args.couplingV[0]
    jintMat=np.zeros((nB,nB))
    intVals=[uintMat,vintMat,jintMat]

    xVals=np.arange(args.Ns**2)/float(args.Ns**2)
    xValsDN=xVals[np.logical_and(xVals>=0.,xVals<0.1)]
    xValsN=-xValsDN[::4]
    xVals=np.arange(args.Ns**2)/float(args.Ns**2)
    xValsDN=xVals[np.logical_and(xVals>=0.1,xVals<0.24)]
    xVals=np.append(xValsN,-xValsDN[::3])

    xValsC=xVals[int(args.fillingN[0])]
    """
    """
    Pnictide two band
    """
    """
    filename=''.join(['pnictide2B','U',str(int(args.couplingU[1])),'J',\
        str(int(args.couplingV[1])),'X',str(int(args.fillingN[1])),'R',args.cutoffR,\
        'nf',str(args.nPatches),'nL',str(args.nL),'NW',str(args.NW),'NK',str(args.NK),\
        'nSites',str(args.Ns),'nB',str(args.nBasis),'bT',str(int(args.beTa[1]))])

    nB=2
    t1=-1.0
    t2=1.3
    t3=-0.85
    t4=-0.85
    N=args.Ns
    kX=((2*np.pi)/float(N))*np.arange(-N/2+1,N/2+1,1)
    kBx=np.repeat(kX,N)
    kBy=np.tile(kX,N)
    eX=-2*t1*np.cos(kBx)-2*t2*np.cos(kBy)-4*t3*np.cos(kBx)*np.cos(kBy)
    eY=-2*t2*np.cos(kBx)-2*t1*np.cos(kBy)-4*t3*np.cos(kBx)*np.cos(kBy)
    eXY=-4*t4*np.sin(kBx)*np.sin(kBy)
    eP=0.5*(eX+eY)
    eM=0.5*(eX-eY)
    disMat=np.zeros((len(kBx),nB,nB),dtype=np.complex_)
    disMat[:,0,0]=eP+eM
    disMat[:,1,1]=eP-eM
    disMat[:,0,1]=eXY
    disMat[:,1,0]=eXY

    uintMat=np.zeros((nB,nB))
    uintMat[0,:]=[args.couplingU[0],args.couplingU[0]-2*args.couplingV[0]]
    uintMat[1,:]=[args.couplingU[0]-2*args.couplingV[0],args.couplingU[0]]

    vintMat=np.zeros((nB,nB))
    jintMat=np.zeros((nB,nB))
    jintMat[0,1]=args.couplingV[0]
    jintMat[1,0]=args.couplingV[0]
    intVals=[uintMat,vintMat,jintMat]

    xVals=np.arange(args.Ns**2)/float(args.Ns**2)
    xVals=np.append(-xVals[::-1],xVals)
    xValsDN=xVals[np.logical_and(xVals>=(-0.5),xVals<0.5)]

    xVals=xValsDN[::6]
    xValsC=xVals[int(args.fillingN[0])]
    """

    """
    Cuprate two band
    """
    """
    filename=''.join(['cuprate2B','U',str(int(args.couplingU[1])),'J',\
        str(int(args.couplingV[1])),'eL',str(int(args.fillingN[1])),\
        'hyb',str(int(args.tP[1])),'R',args.cutoffR,\
        'nf',str(args.nPatches),'nL',str(args.nL),'NW',str(args.NW),'NK',str(args.NK),\
        'nSites',str(args.Ns),'nB',str(args.nBasis),'bT',str(int(args.beTa[1]))])

    nB=2
    txy1=-1.0
    tz1=-0.05
    txyz1=args.tP[0]
    exy0=0
    ez0=-args.fillingN[0]
    N=args.Ns
    kX=((2*np.pi)/float(N))*np.arange(-N/2+1,N/2+1,1)
    kBx=np.repeat(kX,N)
    kBy=np.tile(kX,N)
    eX=2*txy1*(np.cos(kBx)+np.cos(kBy))
    eY=2*tz1*(np.cos(kBx)+np.cos(kBy))
    eXY=2*txyz1*(np.cos(kBx)-np.cos(kBy))

    disMat=np.zeros((len(kBx),nB,nB),dtype=np.complex_)
    disMat[:,0,0]=eY+ez0
    disMat[:,1,1]=eX+exy0
    disMat[:,0,1]=eXY
    disMat[:,1,0]=eXY
    disMat=disMat

    uintMat=np.zeros((nB,nB))
    uintMat[0,:]=[args.couplingU[0],args.couplingU[0]-2*args.couplingV[0]]
    uintMat[1,:]=[args.couplingU[0]-2*args.couplingV[0],args.couplingU[0]]

    vintMat=np.zeros((nB,nB))
    jintMat=np.zeros((nB,nB))
    jintMat[0,1]=args.couplingV[0]
    jintMat[1,0]=args.couplingV[0]
    intVals=[uintMat,vintMat,jintMat]
    xValsC=-1.15
    """

    """
    Cuprate 3-Band
    """

    filename=''.join(['cuprate3B','U',str(int(args.couplingU[1])),'J',\
        str(int(args.couplingV[1])),'eL',str(int(args.fillingN[1])),\
        'hyB',str(int(args.tP[1])),'R',args.cutoffR,\
        'nf',str(args.nPatches),'nL',str(args.nL),'NW',str(args.NW),'NK',str(args.NK),\
        'nSites',str(args.Ns),'nB',str(args.nBasis),'bT',str(int(args.beTa[1]))])

    nB=3
    tpd=-1.0
    tpp=args.tP[0]
    ed=0
    ep=-args.fillingN[0]
    N=args.Ns
    kX=((2*np.pi)/float(N))*np.arange(-N/2+1,N/2+1,1)
    kBx=np.repeat(kX,N)
    kBy=np.tile(kX,N)

    sxF=np.sin(0.5*kBx)
    syF=np.sin(0.5*kBy)

    disMat=np.zeros((len(kBx),nB,nB),dtype=np.complex_)
    disMat[:,0,0]=ed
    disMat[:,1,1]=ep
    disMat[:,2,2]=ep
    disMat[:,0,1]=-2*1j*sxF
    disMat[:,1,0]=2*1j*sxF
    disMat[:,0,2]=2*1j*syF
    disMat[:,2,0]=-2*1j*syF
    disMat[:,1,2]=4*tpp*sxF*syF
    disMat[:,2,1]=4*tpp*sxF*syF
    disMat=disMat

    uintMat=np.zeros((nB,nB))
    uintMat[0,:]=[args.couplingU[0],0,0]

    vintMat=np.zeros((nB,nB))
    jintMat=np.zeros((nB,nB))
    intVals=[uintMat,vintMat,jintMat]
    xVals=np.arange(args.Ns**2)/float(args.Ns**2)
    xValsDN=xVals[np.logical_and(xVals>=(0),xVals<0.5)]

    xVals=xValsDN[::3]
    xValsC=-2-xVals[12]
    
    wMin=(np.pi)/args.beTa[0]
    lMax=-np.log(wMin/args.wMax)
    lMax=0
    #print(xVals,xVals[int(args.fillingN[0])],len(xVals))
    fRGrun=fRG2D(args.nPatches,args.step,args.beTa[0],xValsC,args.wMax,nB,args.Ns,args.NW,args.NK,args.cutoffR,args.nBasis)
    fRGrun.initializeFlow(disMat,intVals,args.nL)
    fRGrun.adaptiveRGFlow(lMax)
    (chargeFac,spinFac,suFac,dFill)=fRGrun.susFunctions()
    #(chargeFacI,spinFacI,suFacI)=fRGrun.susFunctions(inv=True)
    #gapC,gapS,gapSU=fRGrun.gapFunctions(fRGrun.l)

    np.savez(filename,xC=chargeFac,xS=spinFac,xSU=suFac,wB=fRGrun.UnF.wB,\
        kB=fRGrun.UnF.kB,wF=fRGrun.propG.wF,mU=fRGrun.propG.mU,\
        sE=fRGrun.propG.sE,lM=fRGrun.l,bT=args.beTa[0],uU=args.couplingU[0],vV=args.couplingV[0],\
        xU=fRGrun.xDop,tP=args.tP[0],dF=dFill,eL=args.fillingN[0])
