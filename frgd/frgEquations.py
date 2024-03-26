import numpy as np
import frgd.timeFunctions as timeF
from frgd.sparseV import sparseVert as sVert
import frgd.auxFunctions as auxF
import copy
import time

def projectedVertex(UnX,UnF,AC,flag=None):
    """Calculates projected contributions to each channel."""

    #Expand vertices in a basis set
    #for projection across channels

    UnPPI=UnF.legndExpand(UnX[0],UnF.uLeftPP,UnF.uRightPP,UnF.ppIndex,AC)
    UnPHI=UnF.legndExpand(UnX[1],UnF.uLeftPH,UnF.uRightPH,UnF.phIndex,AC)
    UnPHEI=UnF.legndExpand(UnX[2],UnF.uLeftPHE,UnF.uRightPHE,UnF.pheIndex,AC)

    if flag is 'b':
        UnF.UnPPI=UnPPI
        UnF.UnPHI=UnPHI
        UnF.UnPHEI=UnPHEI
    #Projection from each channel
    #to the other channels

    nPatches=len(UnF.wB)
    kPatches=len(UnF.kB[0])
    nB=UnF.nB
    NW=UnF.NW
    NKF=UnF.NKF
    NB=UnF.NB

    uPP=np.zeros(UnX[0].shape[:2]+(NB,NW,NKF,NB,NW,NKF),dtype=np.complex_)
    uPH=np.zeros(UnX[1].shape[:2]+(NB,NW,NKF,NB,NW,NKF),dtype=np.complex_)
    uPHE=np.zeros(UnX[2].shape[:2]+(NB,NW,NKF,NB,NW,NKF),dtype=np.complex_)

    ppIndexL,ppIndexR=auxF.indXY(UnF.ppIndex,nB**2)
    phIndexL,phIndexR=auxF.indXY(UnF.phIndex,nB**2)
    pheIndexL,pheIndexR=auxF.indXY(UnF.pheIndex,nB**2)

    uBands,bandSymm=UnF.uBands,UnF.bandSymm
    bLength=len(uBands)

    def partialU(uLeft,uRight,uX,uIndex,bandInd,symmCur,NB,chnL):
        uCur=np.zeros(UnX[0].shape[:2]+(NB,NW,NKF,NB,NW,NKF),dtype=np.complex_)
        iBand=bandInd[0]
        for j in range(NB):
            for k in range(NB):
                uMap=uLeft[:,:,:,j,uIndex[0][iBand]]*uRight[:,:,:,uIndex[1][iBand],k]
                uCur[:,:,j,:,:,k,:,:]+=uMap[None,:,None,:,None,:]*uX

        for l,sM in enumerate(symmCur):
            uX=np.reshape(uX,uX.shape[:2]+(NW*NKF,NW*NKF,))
            if sM is 'ex':
                uX=UnF.symmFuncs.applyExchange(uX,chnL)
            elif sM is 'pos':
                uX=UnF.symmFuncs.applyPositivity(uX,chnL)
            uX=np.reshape(uX,uX.shape[:2]+(NW,NKF,NW,NKF,))
            iBand=bandInd[l+1]
            for j in range(NB):
                for k in range(NB):
                    uMap=uLeft[:,:,:,j,uIndex[0][iBand]]*uRight[:,:,:,uIndex[1][iBand],k]
                    uCur[:,:,j,:,:,k,:,:]+=uMap[None,:,None,:,None,:]*uX

        return uCur

    for i in range(bLength):
        uPH1,uPHE1=UnF.projectChannel(UnPPI[i],AC,'PP')
        uPP1,uPHE2=UnF.projectChannel(UnPHI[i],AC,'PH')
        uPP2,uPH2=UnF.projectChannel(UnPHEI[i],AC,'PHE')

        uPPc=np.reshape(uPP1+uPP2,(nPatches,kPatches,NW,NKF,NW,NKF))
        uPHc=np.reshape(uPH1+uPH2,(nPatches,kPatches,NW,NKF,NW,NKF))
        uPHEc=np.reshape(uPHE1+uPHE2,(nPatches,kPatches,NW,NKF,NW,NKF))

        uPP+=partialU(UnF.uLeftPPc,UnF.uRightPPc,uPPc,(ppIndexL,ppIndexR),\
            bandSymm[i][0],bandSymm[i][1],NB,'PP')

        uPH+=partialU(UnF.uLeftPHc,UnF.uRightPHc,uPHc,(phIndexL,phIndexR),\
            bandSymm[i][0],bandSymm[i][1],NB,'PH')

        uPHE+=partialU(UnF.uLeftPHEc,UnF.uRightPHEc,uPHEc,(pheIndexL,pheIndexR),\
            bandSymm[i][0],bandSymm[i][1],NB,'PHE')

    uPP=np.reshape(uPP,(nPatches,kPatches,NB*NW*NKF,NB*NW*NKF))
    uPH=np.reshape(uPH,(nPatches,kPatches,NB*NW*NKF,NB*NW*NKF))
    uPHE=np.reshape(uPHE,(nPatches,kPatches,NB*NW*NKF,NB*NW*NKF))

    return uPP,uPH,uPHE

def basisDerv(UnF,AC):
    """Additional vertex correction for the scale dependence of the basis sets
    for frequency modes."""
    indD,scaleD,indD2,scaleD2,scaleDr,scaleD2r=UnF.scaleProjection(AC)

    nPatches=len(UnF.wB)
    kPatches=len(UnF.kB[0])

    NW=UnF.NW
    NKF=UnF.NKF
    NB=UnF.NB

    def genFUnX(UnC,preFac,scaleDc,indS):
        UnC=np.reshape(UnC,(nPatches,kPatches,NB,NW,NKF,NB,NW,NKF))
        UnC=np.swapaxes(UnC,2,3)
        UnC=np.swapaxes(np.swapaxes(np.swapaxes(UnC,6,5),5,4),4,3)
        UnC=np.reshape(UnC,(nPatches,kPatches,NW*NW,NB,NKF,NB,NKF))
        UnCs=np.zeros((nPatches,kPatches,NB,NW,NKF,NB,NW,NKF),dtype=np.complex_)
        for i in range(NW):
            for j in range(NW):
                scaleDcE=scaleDc[i][j]
                indE=indS[i][j]
                UnCs[:,:,:,i,:,:,j,:]=preFac*np.sum(UnC[:,:,indE,:,:,:,:]*\
                    scaleDcE[None,None,:,None,None,None,None],axis=2)
        UnCs=np.reshape(UnCs,(nPatches,kPatches,NB*NW*NKF,NB*NW*NKF))
        return UnCs

    def genFUnXr(UnC,preFac,scaleDc):
        UnC=np.reshape(UnC,(nPatches,kPatches,1,NB,NW,NKF))
        UnCs=np.zeros((nPatches,kPatches,1,NB,NW,NKF),dtype=np.complex_)
        for i in range(NW):
            scaleDcE=scaleDc[i]
            UnCs[:,:,:,:,i,:]=preFac*np.sum(UnC*\
                    scaleDcE[None,None,None,None,:,None],axis=4)
        UnCs=np.reshape(UnCs,(nPatches,kPatches,1,NB*NW*NKF))
        return UnCs

    fA=auxF.fScaling(AC)
    preFac=(auxF.derivFScaling(AC)/fA)
    UnPPs=genFUnX(UnF.UnPP,preFac,scaleD,indD)
    UnPHs=genFUnX(UnF.UnPH,preFac,scaleD2,indD2)
    UnPHEs=genFUnX(UnF.UnPHE,preFac,scaleD2,indD2)

    UnPs=(UnPPs,UnPHs,UnPHEs)

    suXs=genFUnXr(UnF.suGap,preFac,scaleDr)
    chXs=genFUnXr(UnF.chGap,preFac,scaleD2r)
    spXs=genFUnXr(UnF.spGap,preFac,scaleD2r)

    gapXs=(suXs,chXs,spXs)

    phXs=genFUnXr(UnF.gVert,preFac,scaleD2r)
    return UnPs,gapXs,phXs


def betaF(UnF,propT,AC,nL):
    """One-loop beta function for decoupled fRG."""

    UnPHphonon=UnF.conPhVert(AC)
    UnX=(UnF.UnPP,UnF.UnPH+UnPHphonon,UnF.UnPHE)
    UnPPX,UnPHX,UnPHEX=projectedVertex(UnX,UnF,AC,'b')

    UnPPX+=UnF.UnPPO+UnF.UnPP
    UnPHX+=UnF.UnPHO+UnF.UnPH+UnPHphonon
    UnPHEX+=UnF.UnPHEO+UnF.UnPHE
    UnPHSX=UnPHX-UnF.getPHEinPH(UnPHEX)

    dSE=calcSEfft(UnF,propT,AC)
    mixPP, mixPH=propT.xBubbles(UnF.wB,UnF.kB,dSE,AC,UnF.tPoints,UnF.sPoints)
    UnXs,gapXs,gVertXs=basisDerv(UnF,AC)

    mixC=np.reshape(auxF.projectMix(mixPP,UnF.uLeftPP,UnF.uRightPP),UnPPX.shape)
    dUnPP=UnXs[0]-auxF.sparseMul(UnPPX,mixC,UnPPX)
    dSUgap=gapXs[0]-auxF.sparseMul(UnPPX,mixC,UnF.suGap)

    mixC=np.reshape(auxF.projectMix(mixPH,UnF.uLeftPH,UnF.uRightPH),UnPHX.shape)
    dUnPH=UnXs[1]+auxF.sparseMul(UnPHSX,mixC,UnPHX)+\
        auxF.sparseMul(UnPHX,mixC,UnPHSX)
    dCHgap=gapXs[1]+auxF.sparseMul(UnPHSX+UnPHX,mixC,UnF.chGap)
    dgVert=gVertXs+auxF.sparseMul(UnPHSX+UnPHX,mixC,UnF.gVert)
    dPhononSE=-2*np.squeeze(auxF.sparseMul(UnF.phVertL(AC),mixC,UnF.gVert))

    mixC=np.reshape(auxF.projectMix(mixPH,UnF.uLeftPHE,UnF.uRightPHE),UnPHEX.shape)
    dUnPHE=UnXs[2]-auxF.sparseMul(UnPHEX,mixC,UnPHEX)
    dSPgap=gapXs[2]-auxF.sparseMul(UnPHEX,mixC,UnF.spGap)

    dSE=-AC*dSE
    dUX=(-AC*dUnPP,-AC*dUnPH,-AC*dUnPHE)
    dGapX=(-AC*dSUgap,-AC*dCHgap,-AC*dSPgap)
    dPHX=(-AC*dPhononSE,-AC*dgVert)

    return dSE, dUX, dGapX, dPHX

def betaF2L(UnF,propT,AC,nL):
    """Two-loop beta function for decoupled fRG."""

    UnPHphonon=UnF.conPhVert(AC)
    UnX=(UnF.UnPP,UnF.UnPH+UnPHphonon,UnF.UnPHE)
    UnPPX,UnPHX,UnPHEX=projectedVertex(UnX,UnF,AC,'b')

    UnPPX+=UnF.UnPPO+UnF.UnPP
    UnPHX+=UnF.UnPHO+UnF.UnPH+UnPHphonon
    UnPHEX+=UnF.UnPHEO+UnF.UnPHE
    UnPHSX=UnPHX-UnF.getPHEinPH(UnPHEX)

    gapX=UnF.suGap,UnF.chGap,UnF.spGap
    fA=auxF.fScaling(AC)
    tPointsC=(UnF.tPoints[0]/fA,UnF.tPoints[1]/fA)
    suGapF,chGapF,spGapF,kSU,kCH,kSP=auxF.calcMaxGap(gapX,UnF.wB,UnF.kB,\
        tPointsC,UnF.sPoints)

    dSE=calcSEfft(UnF,propT,AC)
    mixPP, mixPH=propT.xBubbles(UnF.wB,UnF.kB,dSE,AC,UnF.tPoints,UnF.sPoints)

    mixC=np.reshape(auxF.projectMix(mixPP,UnF.uLeftPP,UnF.uRightPP),UnPPX.shape)
    dUnPP=-auxF.sparseMul(UnPPX,mixC,UnPPX)
    dSUgap=-auxF.sparseMul(UnF.suGap,mixC,UnPPX)

    mixC=np.reshape(auxF.projectMix(mixPH,UnF.uLeftPH,UnF.uRightPH),UnPHX.shape)
    dUnPH=auxF.sparseMul(UnPHSX,mixC,UnPHX)+\
        auxF.sparseMul(UnPHX,mixC,UnPHSX)
    dCHgap=auxF.sparseMul(UnF.chGap,mixC,UnPHSX+UnPHX)
    dgVert=auxF.sparseMul(UnF.gVert,mixC,UnPHSX+UnPHX)
    dPhononSE=2*np.squeeze(auxF.sparseMul(UnF.gVert,mixC,UnF.phVertL(AC)))

    mixC=np.reshape(auxF.projectMix(mixPH,UnF.uLeftPHE,UnF.uRightPHE),UnPHEX.shape)
    dUnPHE=-auxF.sparseMul(UnPHEX,mixC,UnPHEX)
    dSPgap=-auxF.sparseMul(UnF.spGap,mixC,UnPHEX)

    #Two-Loop contributions from
    #scale derivative of vertex
    #dUnPPi=auxF.reduceVertex(dUnPP,UnF.nMax)
    #dUnPHi=auxF.reduceVertex(dUnPH,UnF.nMax)
    #dUnPHEi=auxF.reduceVertex(dUnPHE,UnF.nMax)

    #Project derviate of the vertex across channels
    #to construct two loop contribution

    dUnXi=(dUnPP,dUnPH,dUnPHE)
    mixPP,mixPH=propT.gBubbles(UnF.wB,UnF.kB,AC,UnF.tPoints,UnF.sPoints)
    dUnPPX,dUnPHX,dUnPHEX=projectedVertex(dUnXi,UnF,AC)
    dUnPHSX=dUnPHX-UnF.getPHEinPH(dUnPHEX)
    UnXs,gapXs,gVertXs=basisDerv(UnF,AC)

    mixC=np.reshape(auxF.projectMix(mixPP,UnF.uLeftPP,UnF.uRightPP),UnPPX.shape)
    dUnPP+=UnXs[0]-auxF.sparseMul(dUnPPX,mixC,UnPPX)-\
        auxF.sparseMul(UnPPX,mixC,dUnPPX)
    dSUgap+=gapXs[0]-auxF.sparseMul(UnF.suGap,mixC,dUnPPX)

    mixC=np.reshape(auxF.projectMix(mixPH,UnF.uLeftPH,UnF.uRightPH),UnPHX.shape)
    dUnPH+=UnXs[1]+auxF.sparseMul(dUnPHSX,mixC,UnPHX)+\
        auxF.sparseMul(UnPHSX,mixC,dUnPHX)+\
        auxF.sparseMul(UnPHX,mixC,dUnPHSX)+\
        auxF.sparseMul(dUnPHX,mixC,UnPHSX)
    dCHgap+=gapXs[1]+auxF.sparseMul(UnF.chGap,mixC,dUnPHSX+dUnPHX)
    dgVert+=gVertXs+auxF.sparseMul(UnF.gVert,mixC,dUnPHSX+dUnPHX)

    mixC=np.reshape(auxF.projectMix(mixPH,UnF.uLeftPHE,UnF.uRightPHE),UnPHEX.shape)
    dUnPHE+=UnXs[2]-auxF.sparseMul(dUnPHEX,mixC,UnPHEX)-\
        auxF.sparseMul(UnPHEX,mixC,dUnPHEX)
    dSPgap+=gapXs[2]-auxF.sparseMul(UnF.spGap,mixC,dUnPHEX)

    dSE=-AC*dSE
    dUX=(-AC*dUnPP,-AC*dUnPH,-AC*dUnPHE)
    dGapX=(-AC*dSUgap,-AC*dCHgap,-AC*dSPgap)
    dPHX=(-AC*dPhononSE,-AC*dgVert)

    return dSE, dUX, dGapX, dPHX

def betaFNL(UnF,propT,AC,nL):
    """Multi-Loop beta function for decoupled fRG."""

    UnX=(UnF.UnPP,UnF.UnPH,UnF.UnPHE)
    uPP,uPH,uPHE=projectedVertex(UnX,UnF,AC,'b')

    UnPPX=UnF.UnPPO+UnF.UnPP+uPP
    UnPHX=UnF.UnPHO+UnF.UnPH+uPH
    UnPHEX=UnF.UnPHEO+UnF.UnPHE+uPHE

    nFull=UnF.NW*UnF.NKF
    dSE=calcSEfft(UnF,propT,AC)
    gPP,gPH,indMap=propT.gBubbles(UnF.wB,UnF.kB,AC,UnF.tPoints,UnF.sPoints)
    mixPP, mixPH,indMap=propT.xBubbles(UnF.wB,UnF.kB,dSE,AC,UnF.tPoints,UnF.sPoints)

    dUnPP=-auxF.sparseMatMul(UnPPX,(mixPP,indMap),UnPPX,nFull)
    dUnPH=auxF.sparseMatMul(UnPHX-UnPHEX,(mixPH,indMap),UnPHX,nFull)+\
        auxF.sparseMatMul(UnPHX,(mixPH,indMap),UnPHX-UnPHEX,nFull)
    dUnPHE=-auxF.sparseMatMul(UnPHEX,(mixPH,indMap),UnPHEX,nFull)

    dUnPPn=auxF.reduceVertex(dUnPP,UnF.nMax)
    dUnPHn=auxF.reduceVertex(dUnPH,UnF.nMax)
    dUnPHEn=auxF.reduceVertex(dUnPHE,UnF.nMax)

    UnX=(dUnPPn,dUnPHn,dUnPHEn)
    dUnPP2X,dUnPH2X,dUnPHE2X=projectedVertex(UnX,UnF,AC)

    dUnPPL=-auxF.sparseMatMul(dUnPP2X,(gPP,indMap),UnPPX,nFull)
    dUnPPR=-auxF.sparseMatMul(UnPPX,(gPP,indMap),dUnPP2X,nFull)
    dUnPPC=dUnPPL+dUnPPR

    dUnPHL=auxF.sparseMatMul(dUnPH2X-dUnPHE2X,(gPH,indMap),UnPHX,nFull)+\
        auxF.sparseMatMul(dUnPH2X,(gPH,indMap),UnPHX-UnPHEX,nFull)
    dUnPHL2=auxF.sparseMatMul(dUnPH2X-dUnPHE2X,(gPH,indMap),UnPHX-UnPHEX,nFull)
    dUnPHR=auxF.sparseMatMul(UnPHX-UnPHEX,(gPH,indMap),dUnPH2X,nFull)+\
        auxF.sparseMatMul(UnPHX,(gPH,indMap),dUnPH2X-dUnPHE2X,nFull)
    dUnPHC=dUnPHL+dUnPHR

    dUnPHEL=-auxF.sparseMatMul(dUnPHE2X,(gPH,indMap),UnPHEX,nFull)
    dUnPHER=-auxF.sparseMatMul(UnPHEX,(gPH,indMap),dUnPHE2X,nFull)
    dUnPHEC=dUnPHEL+dUnPHER

    dUnPP+=dUnPPC
    dUnPH+=dUnPHC
    dUnPHE+=dUnPHEC

    for i in range(nL-2):
        dUnPPCn=auxF.reduceVertex(dUnPPC,UnF.nMax)
        dUnPHCn=auxF.reduceVertex(dUnPHC,UnF.nMax)
        dUnPHECn=auxF.reduceVertex(dUnPHEC,UnF.nMax)
        UnX=(dUnPPCn,dUnPHCn,dUnPHEn)
        dUnPPNX,dUnPHNX,dUnPHENX=projectedVertex(UnX,UnF,AC)

        dUnPPc=-auxF.sparseMatMul(UnPPX,(gPP,indMap),dUnPPL,nFull)
        dUnPPL=-auxF.sparseMatMul(dUnPPNX,(gPP,indMap),UnPPX,nFull)
        dUnPPR=-auxF.sparseMatMul(UnPPX,(gPP,indMap),dUnPPNX,nFull)
        dUnPPC=dUnPPc+dUnPPL+dUnPPR

        dUnPHc=auxF.sparseMatMul(UnPHX,(gPH,indMap),dUnPHL2,nFull)+\
            auxF.sparseMatMul(UnPHX-UnPHEX,(gPH,indMap),dUnPHL,nFull)
        dUnPHL=auxF.sparseMatMul(dUnPHNX-dUnPHENX,(gPH,indMap),UnPHX,nFull)+\
            auxF.sparseMatMul(dUnPHNX,(gPH,indMap),UnPHX-UnPHEX,nFull)
        dUnPHL2=auxF.sparseMatMul(dUnPHNX-dUnPHENX,(gPH,indMap),UnPHX-UnPHEX,nFull)
        dUnPHR=auxF.sparseMatMul(UnPHX-UnPHEX,(gPH,indMap),dUnPHNX,nFull)+\
            auxF.sparseMatMul(UnPHX,(gPH,indMap),dUnPHNX-dUnPHENX,nFull)
        dUnPHC=dUnPHc+dUnPHL+dUnPHR

        dUnPHEc=-auxF.sparseMatMul(UnPHEX,(gPH,indMap),dUnPHEL,nFull)
        dUnPHEL=-auxF.sparseMatMul(dUnPHENX,(gPH,indMap),UnPHEX,nFull)
        dUnPHER=-auxF.sparseMatMul(UnPHEX,(gPH,indMap),dUnPHENX,nFull)
        dUnPHEC=dUnPHEc+dUnPHEL+dUnPHER

        dUnPP+=dUnPPC
        dUnPH+=dUnPHC
        dUnPHE+=dUnPHEC

    UnPPs,UnPHs,UnPHEs=basisDerv(UnF,AC)

    dUnPP+=UnPPs
    dUnPH+=UnPHs
    dUnPHE+=UnPHEs

    dUX=(-dUnPP*AC,-dUnPH*AC,-dUnPHE*AC)
    return -AC*dSE, dUX

def makeWSymmetric(indVals,tAvg,beTa):
    """Make truncated frequency basis sets symmetric to time reversal."""
    tCur=tAvg[indVals]
    tNegCur=auxF.tBetaShift(-tCur,beTa)[0]
    tIndex=np.zeros(len(indVals),dtype=int)
    for i,tC in enumerate(tNegCur):
        tIndex[i]=np.where(np.round(tC,4)==np.round(tAvg,4))[0]
    indFull=np.append(tIndex,indVals)
    return np.unique(indFull)

def calcSE(UnF,propT,AC):
    """Calculation of the self energy via a direct expansion of the vertices in the
    three channels"""
    wF=propT.wF
    kFx,kFy=propT.kF

    wFX=propT.wFX
    wFG=propT.wFG

    kFXx,kFXy=propT.kF

    (wFE,wFXE)=np.meshgrid(wF,wFX,indexing='ij')
    (kFEx,kFXEx)=np.meshgrid(kFx,kFXx,indexing='ij')
    (kFEy,kFXEy)=np.meshgrid(kFy,kFXy,indexing='ij')

    wPP=wFE+wFXE
    wPHZ=wFXE-wFXE
    wPH=wFXE-wFE

    kPPx=kFEx+kFXEx
    kPHZx=kFXEx-kFXEx
    kPHx=kFXEx-kFEx

    kPPy=kFEy+kFXEy
    kPHZy=kFXEy-kFXEy
    kPHy=kFXEy-kFEy

    wFGE=np.tile(wFG[:,np.newaxis],(1,len(kFXx)))
    sProp=wFGE*propT.sF(wFX,(kFXx,kFXy),AC)

    u1=np.swapaxes(UnF.uEvaluate(wPP,wPHZ,wPH,(kPPx,kPPy),(kPHZx,kPHZy),(kPHx,kPHy),AC),1,2)
    u2=np.swapaxes(UnF.uEvaluate(wPP,wPH,wPHZ,(kPPx,kPPy),(kPHx,kPHy),(kPHZx,kPHZy),AC),1,2)

    dSE=(1.0/(propT.beTa*float(propT.N)))*np.sum((2*u1-u2)*sProp[None,None,:,:],axis=(2,3))
    return dSE

def calcSET(UnF,propT,AC):
    """Calculation of the self energy via fft. Vertices in the three channels are
    truncated according to spectral weight."""
    wF=propT.wF
    kF=propT.kF

    wFX=propT.wFX
    wFG=propT.wFG

    kFX=propT.kF
    nDim=propT.nDim

    (wFE,wFXE)=np.meshgrid(wF,wFX,indexing='ij')
    lPatches=len(kF[0])
    kPatches=len(kFX[0])
    kShape=(lPatches,kPatches)
    k1n=[np.zeros(kShape) for i in range(nDim)]
    k2n=[np.zeros(kShape) for i in range(nDim)]
    for i in range(nDim):
        k1n[i]=np.tile(kF[i][:,np.newaxis],(1,kPatches))
        k2n[i]=np.tile(kFX[i][np.newaxis,:],(lPatches,1))

    wPP=wFE+wFXE
    wPHZ=wFXE-wFXE
    wPH=wFXE-wFE

    kPP=[np.zeros(kShape) for i in range(nDim)]
    kPH=[np.zeros(kShape) for i in range(nDim)]
    kPHZ=[np.zeros(kShape) for i in range(nDim)]
    for i in range(nDim):
        kPP[i]=k1n[i]+k2n[i]
        kPH[i]=k2n[i]-k1n[i]
        kPHZ[i]=k2n[i]-k2n[i]

    nB=UnF.nB
    seIndexL,seIndexR=UnF.seIndexL,UnF.seIndexR

    wFGE=np.tile(wFG[:,np.newaxis],(1,len(kFX[0])))
    sEshape=(len(wF),len(kF[0]))
    dSE=np.zeros((nB,nB,)+sEshape,dtype=np.complex_)
    for i in range(nB):
        for j in range(nB):
            for k in range(nB):
                sProp=wFGE*propT.sF(wFX,kFX,AC,k)
                u1=UnF.uF(kPP,kPHZ,kPH,seIndexL[i][j][k])
                u2=UnF.uF(kPP,kPH,kPHZ,seIndexR[i][j][k])

                sPropR=np.sum(sProp,axis=0)
                dSET=np.sum(sPropR[None,:]*(2*u1-u2),axis=1)
                dSE[i,j,:,:]+=(1.0/(propT.beTa*propT.N))*np.tile(dSET[np.newaxis,:],(len(wF),1))

    beTa=propT.beTa

    NLmax=UnF.NLmax
    NB=UnF.NB
    NKF=UnF.NKF
    NW=UnF.NW
    N=UnF.N
    N1D=UnF.N1D

    kDis=np.zeros(N)
    for i in range(nDim):
        kDis+=UnF.kB[i]**2
    kZ=np.argmin(kDis)
    wZ=np.argmin(np.abs(UnF.wB))
    bLen=UnF.nB**4

    UnPPi=UnF.applySymmFull(UnF.UnPPI,'PP')
    UnPHi=UnF.applySymmFull(UnF.UnPHI,'PH')
    UnPHEi=UnF.applySymmFull(UnF.UnPHEI,'PHE')
    UnPHz=[[] for i in range(bLen)]
    UnPHEz=[[] for i in range(bLen)]
    aIndC=np.arange(NW*NKF*NW*NKF)
    for i in range(bLen):
        UnPHT=np.reshape(UnF.UnPH[wZ,kZ,:,:],(NB,NW,NKF,NB,NW,NKF))
        UnPHET=np.reshape(UnF.UnPHE[wZ,kZ,:,:],(NB,NW,NKF,NB,NW,NKF))
        UnPHt=np.zeros((NW,NKF,NW,NKF),dtype=np.complex_)
        UnPHEt=np.zeros((NW,NKF,NW,NKF),dtype=np.complex_)

        iphL,iphR=auxF.indXY(UnF.phIndex[i],UnF.nB*UnF.nB)
        ipheL,ipheR=auxF.indXY(UnF.pheIndex[i],UnF.nB*UnF.nB)
        for j in range(UnF.NB):
            for k in range(UnF.NB):
                UnPHt+=UnPHT[j,:,:,k,:,:]*UnF.uLeftPH[kZ,:,:,iphL,j][None,:,None,:]*\
                    UnF.uRightPH[kZ,:,:,k,iphR][None,:,None,:]
                UnPHEt+=UnPHET[j,:,:,k,:,:]*UnF.uLeftPHE[kZ,:,:,ipheL,j][None,:,None,:]*\
                    UnF.uRightPHE[kZ,:,:,k,ipheR][None,:,None,:]

        UnPHt=np.swapaxes(UnPHt,1,2)
        UnPHz[i]=np.reshape(UnPHt,(NW*NW,NKF*NKF))
        UnPHEt=np.swapaxes(UnPHEt,1,2)
        UnPHEz[i]=np.reshape(UnPHEt,(NW*NW,NKF*NKF))

    nFreqs=len(wF)
    nMoms=len(kF[0])

    def wkIndex(givIndex):
        indKL,indKR,indWL,indWR=auxF.indWKlr(givIndex,NW,NKF)
        zWIndex=indWL*NW+indWR
        zKIndex=indKL*NKF+indKR

        return zWIndex,zKIndex
    wIndexFull=np.arange(NW*NW*NLmax)

    def momSumFFT(UnXb,sEkTrans,wkTrans):
        dSEn=np.zeros(sEshape,dtype=np.complex_)
        UnXb=np.reshape(UnXb,(NW*NW*NLmax,N,NKF,NKF))
        indY0=sEkTrans[0][0][0]
        wkTransCur=wkTrans[:,:,indY0]
        for i in range(NKF):
            for j in range(NKF):
                indX=sEkTrans[i][j][1]
                UnXcur=UnXb[:,indX,i,j]

                indY=sEkTrans[i][j][0]
                sEt=np.sum(wkTransCur*UnXcur[:,None,:],axis=0)
                dSEn+=auxF.iffTransformD(sEt,N1D,nDim,axis=1)*sEkTrans[i][j][2][None,:]
        return dSEn

    def momSum(UnXb,sEkTrans,wkTrans):
        dSEn=np.zeros(sEshape,dtype=np.complex_)
        sInd=np.arange(N)
        for i,kI in enumerate(sInd):
            UnXbr=np.sum(UnXb[:,None,:]*wkTrans[:,:,i][:,:,None],axis=0)
            kInd=sEkTrans[kI][0]
            dSEn+=np.sum(UnXbr[:,kInd,None]*sEkTrans[kI][1][None,:,:],axis=1)
        return dSEn
    """
    def momSum(UnXb,sEkTrans,wkTrans):
        UnXb=np.reshape(UnXb,(NW**2,NKF,NKF))
        dSEn=np.zeros(sEshape,dtype=np.complex_)
        sInd=np.arange(N)
        for i in range(NKF):
            for j in range(NKF):
                skLoc=sEkTrans[i][j][0]
                kPhaseIJ=sEkTrans[i][j][1]
                UnXbr=np.sum(UnXb[:,None,i,j]*wkTrans[:,:,skLoc],axis=0)

                dSEn+=kPhaseIJ[None,:]*UnXbr[:,None]
        return dSEn
    """
    sETransPP1=UnF.kExpandPP1
    sETransPP2=UnF.kExpandPP2

    sETransPH=UnF.kExpandPH
    sETransPHE=UnF.kExpandPH

    sETransPHZ=UnF.kExpandPHZ
    sETransPHEZ=UnF.kExpandPHEZ

    wIrFull=np.arange(NW*NW)
    wIFull=np.arange(NLmax*NW*NW)
    kIrFull=np.arange(NKF*NKF)
    kIFull=np.arange(N*NKF*NKF)
    for i in range(nB):
        wIndex=np.arange(NW*NW*NLmax)
        indexU=(wIndex,wIndex,wIndex)
        sPropL=propT.sF(wFX,kF,AC,i)
        wTransPP1,wTransPP2,wTransPH,wTransPHz=\
            UnF.selfExpand(wF,wFX,wFG,indexU,sPropL,AC)

        for j in range(nB):
            for k in range(nB):
                seIndexl=seIndexL[j][k][i]
                seIndexr=seIndexR[j][k][i]

                uPPi1=UnPPi[seIndexl]
                dSE[j,k,:,:]+=2*momSumFFT(uPPi1,sETransPP1,wTransPP1)

                uPPi2=UnPPi[seIndexr]
                dSE[j,k,:,:]+=-momSumFFT(uPPi2,sETransPP2,wTransPP2)

                uPHz1=UnPHz[seIndexl]
                dSE[j,k,:,:]+=2*momSum(uPHz1,sETransPHZ,wTransPHz)

                uPHi2=UnPHi[seIndexr]
                dSE[j,k,:,:]+=-momSumFFT(uPHi2,sETransPH,wTransPH)

                uPHEi1=UnPHEi[seIndexl]
                dSE[j,k,:,:]+=2*momSumFFT(uPHEi1,sETransPH,wTransPH)

                uPHEz2=UnPHEz[seIndexr]
                dSE[j,k,:,:]+=-momSum(uPHEz2,sETransPHEZ,wTransPHz)

    return dSE

def calcSEfft(UnF,propT,AC):
    """Calculation of the self energy via fft. Vertices in the three channels are
    truncated according to spectral weight."""
    wF=propT.wF
    kF=propT.kF

    wFX=propT.wFX
    wFG=propT.wFG

    kFX=propT.kF
    nDim=propT.nDim

    (wFE,wFXE)=np.meshgrid(wF,wFX,indexing='ij')
    lPatches=len(kF[0])
    kPatches=len(kFX[0])
    kShape=(lPatches,kPatches)
    k1n=[np.zeros(kShape) for i in range(nDim)]
    k2n=[np.zeros(kShape) for i in range(nDim)]
    for i in range(nDim):
        k1n[i]=np.tile(kF[i][:,np.newaxis],(1,kPatches))
        k2n[i]=np.tile(kFX[i][np.newaxis,:],(lPatches,1))

    wPP=wFE+wFXE
    wPHZ=wFXE-wFXE
    wPH=wFXE-wFE

    kPP=[np.zeros(kShape) for i in range(nDim)]
    kPH=[np.zeros(kShape) for i in range(nDim)]
    kPHZ=[np.zeros(kShape) for i in range(nDim)]
    for i in range(nDim):
        kPP[i]=k1n[i]+k2n[i]
        kPH[i]=k2n[i]-k1n[i]
        kPHZ[i]=k2n[i]-k2n[i]

    nB=UnF.nB
    seIndexL,seIndexR=UnF.seIndexL,UnF.seIndexR

    wFGE=np.tile(wFG[:,np.newaxis],(1,len(kFX[0])))
    sEshape=(len(wF),len(kF[0]))
    dSEo=np.zeros((nB,nB,)+sEshape,dtype=np.complex_)
    for i in range(nB):
        for j in range(nB):
            for k in range(nB):
                sProp=wFGE*propT.sF(wFX,kFX,AC,k)
                u1=UnF.uF(kPP,kPHZ,kPH,seIndexL[i][j][k])
                u2=UnF.uF(kPP,kPH,kPHZ,seIndexR[i][j][k])

                sPropR=np.sum(sProp,axis=0)
                dSET=np.sum(sPropR[None,:]*(2*u1-u2),axis=1)
                dSEo[i,j,:,:]+=(1.0/(propT.beTa*propT.N))*np.tile(dSET[np.newaxis,:],(len(wF),1))

    beTa=propT.beTa

    fA=auxF.fScaling(AC)
    NLmax=UnF.NLmax
    NB=UnF.NB
    NKF=UnF.NKF
    NW=UnF.NW
    N=UnF.N
    N1D=UnF.N1D

    UnPPi=UnF.applySymmFull(UnF.UnPPI,'PP')
    UnPHi=UnF.applySymmFull(UnF.UnPHI,'PH')
    UnPHEi=UnF.applySymmFull(UnF.UnPHEI,'PHE')

    nFreqs=len(wF)
    nMoms=len(kF[0])

    uVertExtend=UnF.uVertExtend
    wFI=UnF.wFI
    NWs=UnF.NWfull
    def momSumFFT(UnXb,sEkTrans,sEwTrans,sPropX):
        dSEn=np.zeros(sEshape,dtype=np.complex_)
        UnXb=np.reshape(UnXb,(NLmax,NW,NW,N,NKF,NKF))

        dSEn=np.zeros((NWs,N),dtype=np.complex_)
        for l in range(NKF):
            for m in range(NKF):
                dSEkn=np.zeros((NWs,N),dtype=np.complex_)
                uTempIJ=np.zeros((NWs,NW,NW,N),dtype=np.complex_)
                uTempIJ[uVertExtend[0],:,:,:]=uVertExtend[2][:,None,None,None]*UnXb[uVertExtend[1],:,:,:,l,m]

                for i in range(NW):
                    for j in range(NW):
                        intL=sEwTrans[i][j][0].shape[0]

                        indXK=sEkTrans[l][m][1]
                        uTemp=uTempIJ[:,i,j,:][:,indXK]

                        indYK=sEkTrans[l][m][0]
                        sPropIJ=sPropX[sEwTrans[i][j][3],:]
                        sPropIJ=sPropIJ[:,indYK]

                        dSEtemp=np.zeros((NWs,N),dtype=np.complex_)
                        for k in range(intL):
                            uTempCur=uTemp[sEwTrans[i][j][0][k,:],:]
                            wPhase=np.exp(1j*wFI*sEwTrans[i][j][1][k])*sEwTrans[i][j][2][k]

                            dSEtemp+=(NWs/beTa)*wPhase[:,None]*timeF.iffTransformW(sPropIJ*uTempCur,NWs,axis=0,bos=1)
                        dSEkn+=dSEtemp
                dSEn+=auxF.iffTransformD(dSEkn,N1D,nDim,axis=1)*sEkTrans[l][m][2][None,:]

        return dSEn

    def momSumFFTalt(UnXb,sEkTrans,sEwTrans,sPropX):
        dSEn=np.zeros((NWs,N),dtype=np.complex_)
        for i in range(NW):
            for j in range(NW):
                for k in range(NKF):
                    for l in range(NKF):
                        skLoc=sEkTrans[k][l][0]
                        kPhaseKL=sEkTrans[k][l][1]
                        wPhaseIJ=sEwTrans[i][j][1]
                        sPropMultIJ=sEwTrans[i][j][0]
                        dSEfac=UnXb[i,j,k,l]*np.sum(sPropMultIJ*sPropX[:,skLoc[0]])
                        dSEn+=dSEfac*wPhaseIJ[:,None]*kPhaseKL[None,:]

        return dSEn

    sETransPP1K=UnF.kExpandPP1
    sETransPP2K=UnF.kExpandPP2
    sETransPHK=UnF.kExpandPH
    sETransPHZK=UnF.kExpandPHz

    sETransPP1W=UnF.wTransSEpp1
    sETransPP2W=UnF.wTransSEpp2
    sETransPHW=UnF.wTransSEph
    sETransPHZW=UnF.wTransSEphz

    dSE=np.zeros((nB,nB,NWs,N),dtype=np.complex_)
    for i in range(nB):
        sPropL=propT.sF(fA*wFI,kF,AC,i)
        sPropL=timeF.ffTransformW(sPropL,NWs,axis=0,bos=1)
        sPropL=auxF.ffTransformD(sPropL,N1D,nDim,axis=1)
        for j in range(nB):
            for k in range(nB):
                seIndexl=seIndexL[j][k][i]
                seIndexr=seIndexR[j][k][i]

                uPPi1=UnPPi[seIndexl]
                dSE[j,k,:,:]+=2*momSumFFT(uPPi1,sETransPP1K,sETransPP1W,sPropL)

                uPPi2=UnPPi[seIndexr]
                dSE[j,k,:,:]+=-momSumFFT(uPPi2,sETransPP2K,sETransPP2W,sPropL)

                uPHz=np.sum(np.reshape(UnPHi[seIndexl],(NLmax,NW,NW,N,NKF,NKF)),axis=(0,3))
                dSE[j,k,:,:]+=2*momSumFFTalt(uPHz,sETransPHZK,sETransPHZW,sPropL)

                uPHi2=UnPHi[seIndexr]
                dSE[j,k,:,:]+=-momSumFFT(uPHi2,sETransPHK,sETransPHW,sPropL)

                uPHEi1=UnPHEi[seIndexl]
                dSE[j,k,:,:]+=2*momSumFFT(uPHEi1,sETransPHK,sETransPHW,sPropL)

                uPHEz=np.sum(np.reshape(UnPHEi[seIndexr],(NLmax,NW,NW,N,NKF,NKF)),axis=(0,3))
                dSE[j,k,:,:]+=-momSumFFTalt(uPHEz,sETransPHZK,sETransPHZW,sPropL)
    dSE=auxF.linInterpO(fA*wFI,fA*dSE,wF,axis=2)
    return dSE+dSEo
