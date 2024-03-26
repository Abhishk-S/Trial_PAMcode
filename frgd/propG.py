import numpy as np
import frgd.auxFunctions as auxF
import time

class scaleProp:
    """
    A class for quantities related to the single particle vertex
    ...
    Attributes
    ----------
    g0 : function(wX)
        The hybridization function of the impurity Hamiltonian
    beTa : float
        The temperature of the system
    Mu : float
        The chemical potential of the system
    wF : array_like(float,ndim=1)
        The positive fermionic Ozaki-matsubara frequencies of the system
    wFX,wFG : array_like(float,ndim=1)
        The Ozaki-Matsubara frequencies and weights for matsubara sums
    sE : array_like(float, ndim=1)
        The self-energy of the system of interest.
    gF : function(wF,AC)
        The full propagator of our system at scale AC
    sF : function(wF,AC)
        The single scale propagator of our system at scale AC
    Methods
    -------
    sEInterp(wQ)
        The interpolated self-energy at the frequencies wQ
    xBubbles(wQ,dSEwMid,AC,NW)
        The single scale exchange propagators over NW basis
        functions at scale AC
    gBubbles(wQ,AC,NW)
        The full exchange propagators over the NW basis
        functions at scale AC
    susBubbles(wQ,NW)
        The full exchnage propagators for the susceptibility
        over NW basis functions at scale AC
    _gSoft(wQ,AC)
        The full propagator suppressed by a soft multiplicative
        regulator at scale AC
    _sSoft(wQ,AC)
        The softly regulated single scale propagator
    _gSharp(wQ,AC)
        The full propagator suppressed by a sharp multiplicative
        regulator at scale AC
    _sSharp(wQ,AC)
        The sharply regulated single scale propagator
    _gAdditive(wQ,AC)
        The full propagator suppressed by a soft additive
        regulator at scale AC
    _sAdditive(wQ,AC)
        The additively regulated single scale propagator
    _gLitim(wQ,AC)
        The full propagator suppressed by the additive Litim
        regulator at scale AC
    _sLitim(wQ,AC)
        The single scale Litim propagator
    """

    def __init__(self,maxW,nPatches,freeG,beTa,xDop,N1D,nB,nDim,cutoff='litim'):
        """
        Parameters
        ----------
        maxW: float
            The maximum frequency for the system of interest
        freeG: function(wX)
            The hybridization function of the system
        beTa: float
            The temperature of the sytem
        Mu : float
            The chemical potential of the sytem
        Cutoff: str,optional
            The choice of regulator for the propagator
            Default if the Litim regulator
        """
        self.wF,wB=auxF.freqPoints(beTa,maxW,2*nPatches)
        wF,wFG=auxF.padeWeights(maxW,beTa)
        self.kF=auxF.genMomPoints(N1D,nDim)

        self.N1D=N1D
        self.nDim=nDim
        self.N=N1D**nDim
        self.nB=nB

        self.wFX=np.append(-wF[::-1],wF)
        self.wFG=np.append(wFG[::-1],wFG)
        N=N1D**nDim
        singShape=(nB,nB,len(self.wF),N)
        self.sE=np.zeros(singShape,dtype=np.complex_)
        self.curGap=np.zeros(singShape[3],dtype=np.complex_)
        self.nesQ=np.zeros(1,dtype=int)


        self.g0=freeG
        self.beTa=beTa

        if cutoff=='litim':
            self.gF=self._gLitim
            self.sF=self._sScale_litim

        elif cutoff=='additive':
            self.gF=self._gAdditive
            self.sF=self._sScale_additive

        elif cutoff=='sharp':
            self.gF=self._gSharp
            self.sF=self._sScale_sharp

        elif cutoff=='soft':
            self.gF=self._gSoft
            self.sF=self._sScale_soft

        self.xDop=xDop
        self.setMu()
        self.mu0=self.mU*1

    def singInterp(self,wQ,kQ,vertexOne):
        N1D=self.N1D

        #vertexFull=auxF.fillMomVertex(vertexOne,self.iSymm,self.N,axis=1)
        wFI=np.append(-self.wF[::-1],self.wF)
        vertexFull=np.concatenate((np.conj(vertexOne[::-1,:]),vertexOne),axis=0)
        kQi=auxF.indexOfMomenta(kQ,N1D)

        vQ=auxF.linInterpO(wFI,vertexFull,wQ,axis=0)
        vR=vQ[:,kQi]

        return vR
    def setMu(self):
        wFX,wFG=self.wFX,self.wFG
        kF=self.kF
        wZ=np.zeros(1)+np.abs(wFX).min()
        nB=self.nB
        N=self.N
        beTa=self.beTa
        xDop=self.xDop

        gMax=0
        for i in range(nB):
            gMax=max([gMax,np.abs(self.g0(wZ,kF,i)).max()])
        mMax=gMax
        mMin=-gMax

        def occpFunc(xDop):
            fCur=-xDop
            for i in range(nB):
                gProp=self.gF(wFX,kF,0.0,i)
                fCur+=-(2.0/(beTa*N))*np.sum(wFG[:,None]*gProp)
            return fCur

        self.mU=mMax
        fMax=occpFunc(xDop)

        self.mU=mMin
        fMin=occpFunc(xDop)

        toL=0.004
        while np.abs(fMax)>toL or np.abs(fMin)>toL:
            mNew=(mMax-mMin)*np.random.rand(1)+mMin
            self.mU=mNew
            fNew=occpFunc(xDop)

            if np.sign(fNew)==np.sign(fMax):
                mMax,fMax=mNew,fNew

            else:
                mMin,fMin=mNew,fNew

        if np.abs(mMax-mMin)>toL:
            sl=(fMax-fMin)/(mMax-mMin)
            bL=fMax-sl*mMax
            self.mU=-bL/sl
        else:
            self.mU=0.5*(mMin+mMax)

    def _gAdditive(self,wQ,kQ,AC):
        wShape=wQ.shape
        kShape=kQ[0].shape
        wQ=np.reshape(wQ,wQ.size)

        sEQ=self.singInterp(wQ,kQ,self.sE)
        freeG=self.g0(wQ,kQ)

        regL=auxF.additiveR0(wQ,AC)

        gProp=1/(1j*wQ[:,None]+freeG+self.mU+regL[:,None]-sEQ)
        return np.reshape(gProp,wShape+kShape)

    def _sScale_additive(self,wQ,kQ,AC):
        wShape=wQ.shape
        kShape=kQ[0].shape
        wQ=np.reshape(wQ,wQ.size)

        sEQ=self.singInterp(wQ,kQ,self.sE)
        freeG=self.g0(wQ,kQ)

        regL=auxF.additiveR0(wQ,AC)

        dAdditive=auxF.dAdditiveR0(wQ,AC)

        sProp=-(dAdditive[:,None])/((1j*wQ[:,None]+freeG+self.mU+regL[:,None]-sEQ)**2)
        return np.reshape(sProp,wShape+kShape)

    def _gSoft(self,wQ,kQ,AC,bI):
        wShape=wQ.shape
        kShape=kQ[0].shape
        wQ=np.reshape(wQ,wQ.size)

        nB=self.nB
        regL=auxF.softR(wQ,AC)
        gInv=np.zeros((len(wQ),len(kQ[0]),nB,nB),dtype=np.complex_)
        for i in range(nB):
            freeG=self.g0(wQ,kQ,i)
            gInv[:,:,i,i]=(1j*wQ[:,None]-freeG+self.mU)*regL[:,None]
            for j in range(nB):
                gInv[:,:,i,j]+=-self.singInterp(wQ,kQ,self.sE[i,j,:,:])

        gProp=np.linalg.inv(gInv)[:,:,bI,bI]
        return np.reshape(gProp,wShape+kShape)

    def _sScale_soft(self,wQ,kQ,AC,bI):
        wShape=wQ.shape
        kShape=kQ[0].shape
        wQ=np.reshape(wQ,wQ.size)

        nB=self.nB
        regL=auxF.softR(wQ,AC)
        dSoft=auxF.dSoftR(wQ,AC)
        gInv=np.zeros((len(wQ),len(kQ[0]),nB,nB),dtype=np.complex_)
        dL=np.zeros((len(wQ),len(kQ[0]),nB,nB),dtype=np.complex_)
        for i in range(nB):
            freeG=self.g0(wQ,kQ,i)
            gInv[:,:,i,i]=(1j*wQ[:,None]-freeG+self.mU)*regL[:,None]
            dL[:,:,i,i]=-dSoft[:,None]*(1j*wQ[:,None]-freeG+self.mU)
            for j in range(nB):
                gInv[:,:,i,j]+=-self.singInterp(wQ,kQ,self.sE[i,j,:,:])

        gProp=np.linalg.inv(gInv)

        sProp=np.matmul(gProp,np.matmul(dL,gProp))[:,:,bI,bI]
        return np.reshape(sProp,wShape+kShape)

    def _gLitim(self,wQ,kQ,AC,bI):
        wShape=wQ.shape
        kShape=kQ[0].shape
        wQ=np.reshape(wQ,wQ.size)

        nB=self.nB
        regL=auxF.litim0(wQ,AC)
        gInv=np.zeros((len(wQ),len(kQ[0]),nB,nB),dtype=np.complex_)
        for i in range(nB):
            freeGi=self.g0(wQ,kQ,i)
            gInv[:,:,i,i]=1j*wQ[:,None]-freeGi+self.mU+regL[:,None]
            for j in range(nB):
                gInv[:,:,i,j]+=-self.singInterp(wQ,kQ,self.sE[i,j,:,:])

        gProp=np.linalg.inv(gInv)[:,:,bI,bI]

        return np.reshape(gProp,wShape+kShape)

    def _sScale_litim(self,wQ,kQ,AC,bI):
        wShape=wQ.shape
        kShape=kQ[0].shape
        wQ=np.reshape(wQ,wQ.size)

        nB=self.nB
        dLitim=auxF.dLitim0(wQ,AC)
        regL=auxF.litim0(wQ,AC)
        gInv=np.zeros((len(wQ),len(kQ[0]),nB,nB),dtype=np.complex_)
        dL=np.zeros((len(wQ),len(kQ[0]),nB,nB),dtype=np.complex_)
        for i in range(nB):
            freeGi=self.g0(wQ,kQ,i)
            gInv[:,:,i,i]=1j*wQ[:,None]-freeGi+self.mU+regL[:,None]
            dL[:,:,i,i]=-dLitim[:,None]
            for j in range(nB):
                gInv[:,:,i,j]+=-self.singInterp(wQ,kQ,self.sE[i,j,:,:])

        gProp=np.linalg.inv(gInv)
        sProp=dL[:,:,bI,bI]*(gProp[:,:,bI,bI]**2)
        #sProp=np.matmul(dL,np.matmul(gProp,gProp))[:,:,bI,bI]

        return np.reshape(sProp,wShape+kShape)
    def _gLitim_gap(self,wQ,kQ,AC,bI):
        curGap=self.curGap
        nesQ=self.nesQ
        wShape=wQ.shape
        kShape=kQ[0].shape
        wQ=np.reshape(wQ,wQ.size)

        nB=self.nB
        regL=auxF.litim0(wQ,AC)
        gInv=np.zeros((len(wQ),len(kQ[0])),dtype=np.complex_)
        freeGi=self.g0(wQ,kQ,0)
        g1=1j*wQ[:,None]-freeGi+self.mU+regL[:,None]-self.singInterp(wQ,kQ,self.sE[0,0,:,:])

        kQn=[None for i in range(len(kQ))]
        for i in range(len(kQ)):
            kQn[i]=kQ[i]+kQ[i][nesQ]

        freeGi=self.g0(wQ,kQn,0)
        g2=1j*wQ[:,None]-freeGi+self.mU+regL[:,None]-self.singInterp(wQ,kQn,self.sE[0,0,:,:])
        gProp=1/(g1-(curGap**2)[None,:]/g2)

        return np.reshape(gProp,wShape+kShape)

    def _sScale_litim_gap(self,wQ,kQ,AC,bI):
        curGap=self.curGap
        nesQ=self.nesQ
        wShape=wQ.shape
        kShape=kQ[0].shape
        wQ=np.reshape(wQ,wQ.size)

        nB=self.nB
        dLitim=auxF.dLitim0(wQ,AC)
        regL=auxF.litim0(wQ,AC)

        gInv=np.zeros((len(wQ),len(kQ[0])),dtype=np.complex_)
        freeGi=self.g0(wQ,kQ,0)
        g1=1j*wQ[:,None]-freeGi+self.mU+regL[:,None]-self.singInterp(wQ,kQ,self.sE[0,0,:,:])

        kQn=[None for i in range(len(kQ))]
        for i in range(len(kQ)):
            kQn[i]=kQ[i]+kQ[i][nesQ]

        freeGi=self.g0(wQ,kQn,0)
        g2=1j*wQ[:,None]-freeGi+self.mU+regL[:,None]-self.singInterp(wQ,kQn,self.sE[0,0,:,:])
        gProp=1/(g1-(curGap**2)[None,:]/g2)

        sProp=-dLitim[:,None]*(gProp**2)*(1+(curGap**2)[None,:]/(g2**2))

        return np.reshape(sProp,wShape+kShape)
