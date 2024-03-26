import numpy as np
import multiprocessing as mp
from frgd.sparseV import sparseVert as sVert
from frgd.symmFunctions import symmV
import frgd.auxFunctions as auxF
import frgd.timeFunctions as timeF
import frgd.frgEquations as frgE
import time
import copy

class vertexR:
    """
    A class to contain the two particle vertex

    ...
    Attributes
    ----------
    wB : array_like(float, ndim=1)
        An array of bosonic frequencies at which the
        value of the vertex is known

    beTa : float
        Inverse temperature of the system

    NLmax : int
        Max number of basis functions used to expand
        the vertices in each channel

    NW : int
        Current number of basis functions used to
        approximate the vertex

    uF : function(wPP,wPH,wPHE)
        Initial two particle vertex

    UnPP, UnPPO : array_like(complex, ndim=3)
        Intial and generated particle-particle components
        of the vertex

    UnPH, UnPHO : array_like(complex, ndim=3)
        Particle-particle component of the vertex

    UnPHE : array_like(complex, ndim=3)
        Particle-particle component of the vertex

    zFX,zFG : array_like(float, ndim=1)
        Gauss-Legendre points and weights for integration
        over Legendre polynomials

    scaleDerv : array_like(float, ndim=4)
        Projection of the derivative of scale dependent
        basis functions

    wTransXtoY : arraylike(float, nidm=5)
        Six internal variables for projecting between the
        the three channels of the vertex

    Methods
    -------
    initializeVertex()
        Projects initial two particle vertex over the three
        channels

    projectionW()
        One shot calculation of projection arrays between the
        various channels. Calculates wTransXtoY for run.

    legndExpand(UnX,AC)
        Expands the bosonic frequency dependence of the vertex
        at scale AC via NLmax basis functions

    uEvaluate(wPP,wPH,wPHE,AC)
        Evaluates the full vertex at scale AC at the given
        frequecies

    _expandChannel(wS,wSX,wSY,AC,chnL)
        Fully expands a channel of the vertex at scale AC

    projectChannel(UnL,AC,chnL):
        Projects a channel at scale AC into the other two
        channels
    """
    def __init__(self,maxW,nPatches,N1D,nDim,nB,NW,NK,NB,beTa):
        """
        Parameters
        ----------
        wB : array_like(float, ndim=1)
            An array of bosonic frequencies at which the
            value of the vertex is known
        NW : int
            Current number of frequency basis functions
        NK : int
            Current number of momentum basis functions
        NB : int
            Current number of singular values for decompostion of bands
        beTa : float
            Inverse temperature of the system
        UnF : function(wPP,wPH,wPHE)
            Initial two particle vertex
        """
        self.nB=nB
        self.uBands,self.bandSymm=auxF.findUniqueBands(nB)
        #self.ppI,self.phI,self.pheI=auxF.bandIndCalc(nB)
        #self.seIndex=auxF.selfEIndCalc(nB)

        #self.seIndPP,self.seIndPH,self.seIndPHE=auxF.selfRedIndCalc(nB)
        #self.uBandsPP,self.uBandsPH,self.uBandsPHE=auxF.findRedBands(nB)
        #self.ppI,self.phI,self.pheI=auxF.redBandIndCalc(nB)
        self.seIndexL,self.seIndexR=auxF.selfEIndCalc(nB)
        wF,wB=auxF.freqPoints(beTa,maxW,nPatches)
        self.wB=np.append(-wB[:0:-1],wB)
        self.beTa=beTa

        self.N1D=N1D
        self.nDim=nDim

        self.kBX=auxF.genMomPoints(N1D,nDim)
        self.kB=auxF.genMomPoints(N1D,nDim)
        self.N=N1D**nDim

        tMin=0.001
        nMax=(wB.max()*beTa)/(2*np.pi)
        nMax=2**(np.log2(nMax).astype(int)+1)
        self.NWT=min([256,nMax])
        dT=(np.pi/3)

        #self.NWfull=2**(np.log2((2*beTa)/dT).astype(int))
        #self.NLmax=max(64,2*NW)

        self.NWfull=2**(np.log2((2*beTa)/dT).astype(int))
        self.NWfull=64
        self.NLmax=max(32,2*NW)

        NW=min(NW,self.NLmax)
        self.NW=NW
        self.NB=NB

        self.tPoints=timeF.genTPoints(tMin,beTa,NW,dT)
        self.NW=len(self.tPoints[0])
        self.NLmax=min(self.NWT,self.NLmax)
        self.tPointsL=timeF.genTPoints(tMin,beTa,self.NLmax,dT)
        self.sPoints=auxF.genXDPoints(N1D,NK,nDim)

        self.NLmax=len(self.tPointsL[0])
        self.NK=NK
        NKF=len(self.sPoints[0])
        self.NKF=NKF

        wBI=(np.pi/beTa)*2*np.arange(self.NWfull/2+1)
        self.wBI=np.append(-wBI[-2:0:-1],wBI)

        wBX=(np.pi/beTa)*2*np.arange(self.NWT/2+1)
        self.wBX=np.append(-wBX[-2:0:-1],wBX)

        wFI=(np.pi/beTa)*(2*np.arange(self.NWfull/2)+1)
        self.wFI=np.append(-wFI[::-1],wFI)

        nPatches=len(self.wB)
        kPatches=len(self.kB[0])

        self.symmFuncs=symmV(self.wB,self.kB,self.NW,self.NLmax,\
            self.NKF,self.N1D,self.nDim,self.sPoints,self.tPoints,self.tPointsL)
        self.freqProjectionFFT()
        self.selfExpandFFT()
        self.singExpand()
        self.momProjectionFFTnD()
        self.momLegSEnD()
        self.momLegSEFnD()

        NW=self.NW
        self.chGap,self.spGap,self.suGap=self.initializeGap(0.01)
        uShape=(nPatches,kPatches,NW*NKF*NB,NW*NKF*NB)

        self.UnPP=np.zeros(uShape,dtype=np.complex_)
        self.UnPH=np.zeros(uShape,dtype=np.complex_)
        self.UnPHE=np.zeros(uShape,dtype=np.complex_)


    def initializeGap(self,initGap):
        nPatches=len(self.wB)
        kPatches=len(self.kB[0])
        NW,NKF,NB=self.NW,self.NKF,self.NB

        uShape=(nPatches,kPatches,1,NW*NKF)
        wInd=np.argmin(np.abs(self.tPoints[0]))
        chGap=np.zeros(uShape,dtype=np.complex_)
        spGap=np.zeros(uShape,dtype=np.complex_)
        suGap=np.zeros(uShape,dtype=np.complex_)
        for i in range(NKF):
            chGap[:,:,0,wInd*NKF+i]+=initGap
            spGap[:,:,0,wInd*NKF+i]+=initGap
            suGap[:,:,0,wInd*NKF+i]+=initGap

        return chGap,spGap,suGap

    def initializePhonons(self,gEph=None,pDisper=None):
        wB,NW=self.wB,self.NW
        kB=self.kB
        kBS=self.kB
        NKF=self.NKF
        NB=self.NB
        sPoints=self.sPoints
        N=self.N
        nDim=len(kB)

        if gEph is None:
            gEph=np.zeros((N,N),dtype=np.complex_)
            pDisper=np.zeros(N)+0.1

        nPatches=len(wB)
        kPatches=N

        wInd=np.argmin(np.abs(self.tPoints[0]))
        momBasisR=np.ones((kPatches,NKF),dtype=np.complex_)
        for j in range(NKF):
            for i in range(nDim):
                momBasisR[:,j]=momBasisR[:,j]*np.exp(-1j*sPoints[i][j]*kB[i])

        self.phononSE=np.zeros((nPatches,kPatches),dtype=np.complex_)
        gVert=np.zeros((nPatches,kPatches,1,NW*NKF),dtype=np.complex_)

        for j in range(kPatches):
            for k in range(NKF):
                gVert[:,j,0,wInd*NKF+k]=(momBasisR[j,k]/N)*np.sum(gEph[:,j]*momBasisR[:,k])

        self.gVert=gVert
        self.pDisper=pDisper

    def initializeVertex(self,UnF):
        """Calculates the initial vertex in the three channels"""

        self.uF=UnF

        wB,NW=self.wB,self.NW
        kB=self.kBX
        kBS=self.kB
        NKF=self.NKF
        NB=self.NB
        sPoints=self.sPoints
        N=self.N
        nDim=self.nDim
        nPatches=len(wB)
        kPatches=len(kBS[0])
        lPatches=len(kB[0])

        nB=self.nB
        uShape=(kPatches,NKF,NKF,nB**2,nB**2)
        UnPPO=np.zeros(uShape,dtype=np.complex_)
        UnPHO=np.zeros(uShape,dtype=np.complex_)
        UnPHOs=np.zeros(uShape,dtype=np.complex_)
        UnPHEO=np.zeros(uShape,dtype=np.complex_)

        self.ppIndex=np.zeros(nB**4,dtype=int)
        self.phIndex=np.zeros(nB**4,dtype=int)
        self.pheIndex=np.zeros(nB**4,dtype=int)

        momBasisL=np.ones((lPatches,NKF),dtype=np.complex_)
        momBasisR=np.ones((lPatches,NKF),dtype=np.complex_)
        for j in range(NKF):
            for i in range(nDim):
                momBasisL[:,j]=momBasisL[:,j]*np.exp(1j*sPoints[i][j]*kB[i])
                momBasisR[:,j]=momBasisR[:,j]*np.exp(-1j*sPoints[i][j]*kB[i])
        momBasisL=(1.0/N)*momBasisL
        momBasisR=(1.0/N)*momBasisR

        momBasisL=np.tile(momBasisL[:,None,:],(1,lPatches,1))
        momBasisR=np.tile(momBasisR[None,:,:],(lPatches,1,1))

        kShape=(lPatches,lPatches)
        kS1n=[np.zeros(kShape) for i in range(nDim)]
        kS2n=[np.zeros(kShape) for i in range(nDim)]
        kZ=[np.zeros(kShape) for i in range(nDim)]
        for i in range(nDim):
            kS1n[i]=np.tile(kB[i][:,np.newaxis],(1,lPatches))
            kS2n[i]=np.tile(kB[i][np.newaxis,:],(lPatches,1))

        kPP=[np.zeros(kShape) for i in range(nDim)]
        kPH=[np.zeros(kShape) for i in range(nDim)]
        kPHE=[np.zeros(kShape) for i in range(nDim)]
        for i in range(nB**4):
            bandInd=auxF.genBandIndicies(i,nB)
            wInd=np.argmin(np.abs(self.tPoints[0]))

            for k in range(kPatches):
                kSingPhase=np.ones((NKF,NKF),dtype=np.complex_)
                for j in range(nDim):
                    kSingPhase=kSingPhase*np.exp(-0*1j*kB[j][k]*0.5*(sPoints[j][:,None]-sPoints[j][None,:]))

                for j in range(nDim):
                    kPP[j]=kB[j][k]+kZ[j]
                    kPH[j]=kS2n[j]-kS1n[j]
                    kPHE[j]=-kB[j][k]+kS2n[j]+kS1n[j]

                uPP=self.uF(kPP,kPH,kPHE,i)
                ppL=bandInd[0]*nB+bandInd[1]
                ppR=bandInd[3]*nB+bandInd[2]
                self.ppIndex[i]=ppL*(nB**2)+ppR
                UnPPO[k,:,:,ppL,ppR]=kSingPhase*auxF.calcMomProjection(uPP,momBasisL,momBasisR)

                for j in range(nDim):
                    kPP[j]=kS1n[j]+kS2n[j]-kB[j][k]
                    kPH[j]=kB[j][k]+kZ[j]
                    kPHE[j]=kS2n[j]-kS1n[j]

                uPH=self.uF(kPP,kPH,kPHE,i)
                phL=bandInd[0]*nB+bandInd[3]
                phR=bandInd[1]*nB+bandInd[2]
                self.phIndex[i]=phL*(nB**2)+phR
                UnPHO[k,:,:,phL,phR]=kSingPhase*auxF.calcMomProjection(uPH,momBasisL,momBasisR)

                for j in range(nDim):
                    kPP[j]=kS1n[j]+kS2n[j]-kB[j][k]
                    kPH[j]=kS2n[j]-kS1n[j]
                    kPHE[j]=kB[j][k]+kZ[j]

                uPHE=self.uF(kPP,kPH,kPHE,i)
                pheL=bandInd[1]*nB+bandInd[3]
                pheR=bandInd[0]*nB+bandInd[2]
                self.pheIndex[i]=pheL*(nB**2)+pheR
                UnPHEO[k,:,:,pheL,pheR]=kSingPhase*auxF.calcMomProjection(uPHE,momBasisL,momBasisR)

        """
        uLeftPP,sPP,uRightPP=np.linalg.svd(UnPPO)
        uLeftPH,sPH,uRightPH=np.linalg.svd(UnPHO)
        uLeftPHE,sPHE,uRightPHE=np.linalg.svd(UnPHEO)
        """

        uLeftPP=np.zeros((N,NKF,NKF,nB,nB),dtype=np.complex_)+1
        uRightPP=np.zeros((N,NKF,NKF,nB,nB),dtype=np.complex_)+1
        sPP=UnPPO[...,0]

        uLeftPH=np.zeros((N,NKF,NKF,nB,nB),dtype=np.complex_)+1
        uRightPH=np.zeros((N,NKF,NKF,nB,nB),dtype=np.complex_)+1
        sPH=UnPHO[...,0]

        uLeftPHE=np.zeros((N,NKF,NKF,nB,nB),dtype=np.complex_)+1
        uRightPHE=np.zeros((N,NKF,NKF,nB,nB),dtype=np.complex_)+1
        sPHE=UnPHEO[...,0]

        self.uLeftPP=uLeftPP[:,:,:,:,:NB]
        self.uRightPP=uRightPP[:,:,:,:NB,:]
        self.uLeftPH=uLeftPH[:,:,:,:,:NB]
        self.uRightPH=uRightPH[:,:,:,:NB,:]
        self.uLeftPHE=uLeftPHE[:,:,:,:,:NB]
        self.uRightPHE=uRightPHE[:,:,:,:NB,:]

        rShape=self.uRightPP.shape
        lShape=self.uLeftPP.shape
        self.uLeftPPc=np.zeros(rShape,dtype=np.complex_)
        self.uLeftPHc=np.zeros(rShape,dtype=np.complex_)
        self.uLeftPHEc=np.zeros(rShape,dtype=np.complex_)
        self.uRightPPc=np.zeros(lShape,dtype=np.complex_)
        self.uRightPHc=np.zeros(lShape,dtype=np.complex_)
        self.uRightPHEc=np.zeros(lShape,dtype=np.complex_)
        for i in range(N):
            for j in range(NKF):
                for k in range(NKF):
                    self.uLeftPPc[i,j,k,:,:]=np.conj(np.transpose(self.uLeftPP[i,j,k,:,:]))
                    self.uRightPPc[i,j,k,:,:]=np.conj(np.transpose(self.uRightPP[i,j,k,:,:]))

                    self.uLeftPHc[i,j,k,:,:]=np.conj(np.transpose(self.uLeftPH[i,j,k,:,:]))
                    self.uRightPHc[i,j,k,:,:]=np.conj(np.transpose(self.uRightPH[i,j,k,:,:]))

                    self.uLeftPHEc[i,j,k,:,:]=np.conj(np.transpose(self.uLeftPHE[i,j,k,:,:]))
                    self.uRightPHEc[i,j,k,:,:]=np.conj(np.transpose(self.uRightPHE[i,j,k,:,:]))

        fShape=(kPatches,NW*NKF*NB,NW*NKF*NB)
        UnPPOF=np.zeros(fShape,dtype=np.complex_)
        UnPHOF=np.zeros(fShape,dtype=np.complex_)
        UnPHEOF=np.zeros(fShape,dtype=np.complex_)
        for i in range(NB):
            for k in range(NKF):
                for l in range(NKF):
                    UnPPOF[:,i*NW*NKF+wInd*NKF+k,i*NW*NKF+wInd*NKF+l]=sPP[:,k,l,i]
                    UnPHOF[:,i*NW*NKF+wInd*NKF+k,i*NW*NKF+wInd*NKF+l]=sPH[:,k,l,i]
                    UnPHEOF[:,i*NW*NKF+wInd*NKF+k,i*NW*NKF+wInd*NKF+l]=sPHE[:,k,l,i]

        UnPPOF=np.tile(UnPPOF[None,:,:,:],(nPatches,1,1,1))
        UnPHOF=np.tile(UnPHOF[None,:,:,:],(nPatches,1,1,1))
        UnPHEOF=np.tile(UnPHEOF[None,:,:,:],(nPatches,1,1,1))

        self.calcBandMap()
        self.UnPPO=UnPPOF
        self.UnPHO=UnPHOF
        self.UnPHEO=UnPHEOF

    def getPHEinPH(self,UnPHE):
        nPatches=len(self.wB)
        kPatches=len(self.kB[0])
        NB=self.NB
        NW=self.NW
        NKF=self.NKF

        UnPHE=np.reshape(UnPHE,(nPatches,kPatches,NB,NW,NKF,NB,NW,NKF))
        fTrans=self.bandPHEtoPH

        UnPHEf=np.zeros(UnPHE.shape,dtype=np.complex_)
        for i in range(NB):
            for j in range(NB):
                UnPHEf[:,:,i,:,:,j,:,:]=np.sum(fTrans[:,:,:,:,:,i,j][None,:,:,None,:,:,None,:]*\
                    UnPHE,axis=(2,5))
        return np.reshape(UnPHEf,(nPatches,kPatches,NB*NW*NKF,NB*NW*NKF))

    def calcBandMap(self):
        NB=self.NB
        NW=self.NW
        NKF=self.NKF
        nPatches=len(self.wB)
        kPatches=len(self.kB[0])

        fTrans=np.zeros((kPatches,NB,NKF,NB,NKF,NB,NB),dtype=np.complex_)
        nB=self.nB
        for i in range(nB**4):
            bandInd=auxF.genBandIndicies(i,nB)
            bLabel=auxF.genBandLabel([bandInd[1],bandInd[0],bandInd[2],bandInd[3]],\
                nB)
            pheL,pheR=auxF.indXY(self.pheIndex,nB**2)
            transTemp=np.zeros((kPatches,NB,NKF,NB,NKF),dtype=np.complex_)
            for j in range(NB):
                for k in range(NB):
                    transTemp[:,j,:,k,:]=self.uLeftPHE[:,:,:,pheL[bLabel],j]*self.\
                        uRightPHE[:,:,:,k,pheR[bLabel]]

            phL,phR=auxF.indXY(self.phIndex,nB**2)
            for j in range(NB):
                for k in range(NB):
                    phMap=self.uLeftPHc[:,:,:,j,phL[i]]*self.uRightPHc[:,:,:,phR[i],k]
                    fTrans[:,:,:,:,:,j,k]=phMap[:,None,:,None,:]*transTemp

        self.bandPHEtoPH=fTrans

    def phVertL(self,AC):
        wB=self.wB
        kB=self.kB
        nDim=len(kB)
        N1D=self.N1D

        nWrange=auxF.locIndex(-wB,wB)
        nkB=[None for i in range(nDim)]
        for i in range(nDim):
            nkB[i]=-kB[i]
        nKrange=auxF.indexOfMomenta(nkB,N1D)

        sPoints=self.sPoints
        tInd=auxF.trueInd(sPoints,N1D,nDim)

        newindKLN=[None for i in range(nDim)]
        for i in range(nDim):
            newindKLN[i]=-self.sPoints[i]
        newInd=auxF.trueInd(newindKLN,N1D,nDim)
        indKLNi=auxF.locIndex(newInd,tInd)

        NB=self.NB
        NW=self.NW
        NKF=self.NKF
        wIndexCur=np.repeat(np.arange(NW),NKF)
        kIndexCur=np.tile(indKLNi,NW)
        newIndex=wIndexCur*NKF+kIndexCur

        phVertLcur=np.zeros(self.gVert.shape[:2]+(NB*NW*NKF,1,),dtype=np.complex_)
        phVertLcur[:,:,:,0]=self.gVert[:,:,0,:][np.ix_(nWrange,nKrange,newIndex)]

        kPhaseFac=np.ones((self.N,NKF),dtype=np.complex_)
        for i in range(nDim):
            kPhaseFac=kPhaseFac*np.exp(1j*2*kB[i][:,None]*sPoints[i][None,:])
        kPhaseFac=np.reshape(np.tile(kPhaseFac[:,None,:],(1,NW,1)),(self.N,NB*NW*NKF,1))

        phVertLcur=phVertLcur*kPhaseFac[None,...]

        return phVertLcur
    def conPhVert(self,AC):
        phVertl=self.phVertL(AC)

        wB=self.wB
        phononProp=1/(1j*wB[:,None]-self.pDisper[None,:])-\
            1/(1j*wB[:,None]+self.pDisper[None,:])
        phononProp=1/(1/phononProp-self.phononSE)
        phVertl=phVertl*phononProp[:,:,None,None]
        uPHphonon=phVertl[:,:,:,0][:,:,None,:]*(self.gVert[:,:,0,:][:,:,:,None])
        return uPHphonon

    def applySymm(self,UnX,chnL):
        bandSymm=self.bandSymm
        nB=self.nB
        UnR=[[] for i in range(nB**4)]
        kRed=auxF.indexofMomenta(self.kB,self.N1D)
        wRange=np.arange(len(self.wB))
        for i in range(len(bandSymm)):
            UnR[bandSymm[i][0][0]]=UnX[i]
            symmCur=bandSymm[i][1]
            UnXn=auxF.compVertex(UnX[i],self.iSymm,self.N1D,self.NW,self.NKF,self.sPoints)
            for j,sM in enumerate(symmCur):
                if sM is 'ex':
                    UnXn=self.symmFuncs.applyExchange(UnXn,chnL)
                elif sM is 'pos':
                    UnXn=self.symmFuncs.applyPositivity(UnXn,chnL)
                aSort=np.arange(self.NW*self.NKF)
                UnR[bandSymm[i][0][j+1]]=UnXn[np.ix_(wRange,kRed,aSort,aSort)]
        return UnR
    def applySymmFull(self,UnX,chnL):
        bandSymm=self.bandSymm
        nB=self.nB
        UnR=[[] for i in range(nB**4)]

        for i in range(len(bandSymm)):
            UnR[bandSymm[i][0][0]]=UnX[i]
            symmCur=bandSymm[i][1]
            UnXn=UnX[i]
            aIndn=np.arange(UnXn.size)
            for j,sM in enumerate(symmCur):
                if sM is 'ex':
                    UnXn=self.symmFuncs.applyExchangeFull(UnXn,chnL)

                elif sM is 'pos':
                    UnXn=self.symmFuncs.applyPositivityFull(UnXn,chnL)

                UnR[bandSymm[i][0][j+1]]=UnXn

        return UnR

    def waveletTransformSE(self):
        def addRedundant(indexF,wTransF):
            sortInd=np.argsort(indexF)
            uniqInd,loC=np.unique(indexF[sortInd],return_index=True)
            loCE=np.append(loC,len(indexF))
            masK=loCE[1:]-loCE[:-1]

            indexSort=indexF[sortInd]
            wTransSort=wTransF[sortInd,:]

            indexR=indexSort[loC[masK==1]]
            wTN=wTransSort[loC[masK==1],:]

            indexRemain=loC[masK>1]
            indicies=np.arange(len(loC))[masK>1]

            indexL=np.zeros(len(indexRemain),dtype=int)
            wTNL=np.zeros((len(indexRemain),wTN.shape[1]),dtype=np.complex_)
            for k in range(len(indexRemain)):
                indK=indicies[k]

                indexL[k]=indexSort[indexRemain[k]]
                wTNL[k,:]=np.sum(wTransSort[loCE[indK]:loCE[indK+1],:],axis=0)

            indexFull=np.append(indexR,indexL)
            wTransFull=np.concatenate((wTN,wTNL),axis=0)
            return indexFull,wTransFull

        def reduceSETrans(wTransSECurF,mInd):
            NLmax=self.NLmax
            wIndexT=[[] for i in range(NLmax)]
            wTransT=[[] for i in range(NLmax)]

            for i in range(NLmax):
                lenM=len(mInd[i][0])
                iC=mInd[i][0][0]
                wgT=mInd[i][1][0]
                indexN=np.zeros(0,dtype=int)
                indexN=np.append(indexN,wTransSECurF[iC][0])
                wTransWF=wgT*wTransSECurF[iC][1]
                for j in range(lenM-1):
                    iC=mInd[i][0][j+1]
                    wgT=mInd[i][1][j+1]
                    indexN=np.append(indexN,wTransSECurF[iC][0])
                    wTransWF=np.concatenate((wTransWF,wgT*wTransSECurF[iC][1]),axis=0)

                wIndexT[i],wTransT[i]=addRedundant(indexN,wTransWF)

            return wIndexT,wTransT

        NLmax=self.NLmax

        iMerge=timeF.debArray(NLmax)
        indexPP1,transPP1=reduceSETrans(self.wTransSEpp1,iMerge)
        indexPP2,transPP2=reduceSETrans(self.wTransSEpp2,iMerge)
        indexPH,transPH=reduceSETrans(self.wTransSEph,iMerge)
        indexPHz,transPHz=reduceSETrans(self.wTransSEphz,iMerge)

        self.wTransSEpp1=[[] for i in range(NLmax)]
        self.wTransSEpp2=[[] for i in range(NLmax)]
        self.wTransSEph=[[] for i in range(NLmax)]
        self.wTransSEphz=[[] for i in range(NLmax)]
        for i in range(NLmax):
            self.wTransSEpp1[i]=[indexPP1[i],transPP1[i]]
            self.wTransSEpp2[i]=[indexPP2[i],transPP2[i]]
            self.wTransSEph[i]=[indexPH[i],transPH[i]]
            self.wTransSEphz[i]=[indexPHz[i],transPHz[i]]

    def singExpand(self):
        tAvg,deltaT=self.tPointsL
        NLmax=self.NLmax
        beTa=self.beTa

        NWT=self.NWT
        wBX=(2*np.pi/beTa)*np.arange(NWT/2)
        wBX=np.append(-wBX[:0:-1],wBX)

        tCur=(beTa/NWT)*np.arange(-NWT/2,NWT/2+1)

        wArrF=[[] for i in range(NLmax)]
        toL=10**(-2)
        for i in range(NLmax):
            cond1=np.where(tCur>=(tAvg[i]-0.5*deltaT[i]))[0]
            cond2=np.where(tCur<=(tAvg[i]+0.5*deltaT[i]))[0]

            wLoc=np.intersect1d(cond1,cond2)
            wArrF[i]=[wLoc,(np.zeros(len(wLoc))+1.0)]
        self.wArrF=wArrF

    def scaleProjection(self,AC):
        NW=self.NW

        indexPP=[[[] for i in range(NW)] for j in range(NW)]
        scaleDervPP=[[[] for i in range(NW)] for j in range(NW)]
        indexPH=[[[] for i in range(NW)] for j in range(NW)]
        scaleDervPH=[[[] for i in range(NW)] for j in range(NW)]

        fA=auxF.fScaling(AC)
        tAvg=self.tPoints[0]/fA
        deltaT=self.tPoints[1]/fA
        beTa=self.beTa

        wBX=2*(np.pi/beTa)*np.arange(40000)
        wBX=np.append(-wBX[:0:-1],wBX)

        intM1=np.zeros((NW,NW),dtype=np.complex_)
        intM2=np.zeros((NW,NW),dtype=np.complex_)

        derF1=np.zeros((len(wBX),NW),dtype=np.complex_)
        derF2=np.zeros((len(wBX),NW),dtype=np.complex_)
        tMinLoc=np.argmin(np.abs(tAvg))
        for i in range(NW):
            if i!=tMinLoc:
                fBas1=np.exp(1j*wBX*tAvg[i])*np.sinc((0.5*deltaT[i]*wBX)/np.pi)
                derF1[:,i]=((1j*wBX*tAvg[i])*fBas1+\
                    (np.exp(1j*tAvg[i]*wBX)*np.cos(0.5*wBX*deltaT[i])-fBas1))

                fBas2=np.exp(-1j*wBX*tAvg[i])*np.sinc((0.5*deltaT[i]*wBX)/np.pi)
                derF2[:,i]=(-(1j*wBX*tAvg[i])*fBas2+\
                    (np.exp(-1j*tAvg[i]*wBX)*np.cos(0.5*wBX*deltaT[i])-fBas2))

        for i in range(NW):
            fJ1=deltaT[i]*np.exp(-1j*wBX*tAvg[i])*np.sinc((0.5*deltaT[i]*wBX)/np.pi)

            intM1[:,i]=(1/beTa)*np.round(np.sum(fJ1[:,None]*derF1,axis=0),5)
            intM2[:,i]=(1/beTa)*np.round(np.sum(np.conj(fJ1)[:,None]*derF2,axis=0),5)

        scaleDervPPr=[[] for i in range(NW)]
        scaleDervPHr=[[] for i in range(NW)]
        for i in range(NW):
            tempI=intM2[:,i]
            scaleDervPPr[i]=tempI
            tempI=intM1[:,i]
            scaleDervPHr[i]=tempI
            for j in range(NW):
                tTemp1=np.zeros((NW,NW),dtype=np.complex_)
                tTemp1[i,:]+=intM1[:,j]
                tTemp1[:,j]+=intM2[:,i]
                tTemp1=np.reshape(tTemp1,NW*NW)
                indexPP[i][j]=np.nonzero(tTemp1)[0]
                scaleDervPP[i][j]=tTemp1[indexPP[i][j]]
                tTemp1=np.zeros((NW,NW),dtype=np.complex_)
                tTemp1[i,:]+=intM1[:,j]
                tTemp1[:,j]+=intM1[:,i]
                tTemp1=np.reshape(tTemp1,NW*NW)
                indexPH[i][j]=np.nonzero(tTemp1)[0]
                scaleDervPH[i][j]=tTemp1[indexPH[i][j]]

        return indexPP,scaleDervPP,indexPH,scaleDervPH,scaleDervPPr,scaleDervPHr

    def legndExpand(self,UnX,uLeftC,uRightC,uIndexC,AC):
        wB=self.wB
        NW,NKF,NB=self.NW,self.NKF,self.NB
        nB=self.nB
        nDim=self.nDim
        beTa=self.beTa

        NLmax=self.NLmax
        sPoints=self.sPoints
        N=self.N
        N1D=self.N1D

        nPatches=len(wB)
        kPatches=len(self.kB[0])

        uBands=self.uBands
        bLength=len(uBands)
        UnXi=[[] for i in range(bLength)]
        UnXc=np.reshape(UnX,(nPatches,kPatches,NB,NW,NKF,NB,NW,NKF))
        UnXc=np.swapaxes(np.swapaxes(UnXc,5,6),6,7)
        UnXc=np.swapaxes(np.swapaxes(np.swapaxes(np.swapaxes(UnXc,2,3),3,4),4,5),5,6)

        for i in range(bLength):
            iLeft,iRight=auxF.indXY(uIndexC[uBands[i]],nB*nB)
            uTemp=np.squeeze(np.matmul(uLeftC[:,:,:,iLeft,:][None,:,None,:,None,:,None,:],\
                np.matmul(UnXc,uRightC[:,:,:,:,iRight][None,:,None,:,None,:,:,None])))
            uTemp=np.reshape(uTemp,(nPatches,N,NW,NKF,NW,NKF))

            fA=auxF.fScaling(AC)
            wBX=self.wBX
            NWT=self.NWT
            nIndex=np.append(np.arange(NWT/2,NWT),np.arange(NWT/2+1))
            UnXiF=auxF.linInterpO(wB,uTemp,fA*wBX,axis=0)

            UnLw=timeF.ffTransformW(UnXiF,NWT,axis=0)[nIndex.astype(int),...]
            wArrF=self.wArrF
            UnL=np.zeros((NLmax,)+UnLw.shape[1:],dtype=np.complex_)
            for iW in range(NLmax):
                UnL[iW,...]=np.sum(UnLw[wArrF[iW][0],...],axis=0)

            UnLw=auxF.ffTransformD(UnL,N1D,nDim,axis=1)

            UnLw=np.swapaxes(np.swapaxes(np.swapaxes(UnLw,1,2),4,3),3,2)
            UnLw=np.reshape(UnLw,(NLmax*NW*NW,N*NKF*NKF))

            UnXi[i]=UnLw
        return UnXi

    def uEvaluate(self,wPP,wPH,wPHE,kPP,kPH,kPHE,AC):
        """Evaluates the vertex at the given frequency and scale"""

        uShape=wPP.shape+kPP[0].shape
        nDim=self.nDim
        wPP=np.reshape(wPP,wPP.size)
        wPH=np.reshape(wPH,wPH.size)
        wPHE=np.reshape(wPHE,wPHE.size)
        kSize=kPP[0].size
        k1=[np.zeros(kSize) for i in range(nDim)]
        k2=[np.zeros(kSize) for i in range(nDim)]
        k3=[np.zeros(kSize) for i in range(nDim)]
        k4=[np.zeros(kSize) for i in range(nDim)]

        for i in range(nDim):
            kPPc=np.reshape(kPP[i],kPP[i].size)
            kPHc=np.reshape(kPH[i],kPH[i].size)
            kPHEc=np.reshape(kPHE[i],kPHE[i].size)

            k1[i]=0.5*(kPPc+kPHc-kPHEc)
            k2[i]=0.5*(kPPc-kPHc+kPHEc)
            k3[i]=0.5*(kPPc+kPHc+kPHEc)
            k4[i]=0.5*(kPPc-kPHc-kPHEc)

        uPP=self._expandChannel(wPP,-(wPH-wPHE),(wPH+wPHE),kPP,k2,k3,AC,'PP')
        uPH=self._expandChannel(wPH,(wPP-wPHE),(wPP+wPHE),kPH,k1,k3,AC,'PH')
        uPHE=self._expandChannel(wPHE,(wPP-wPH),(wPP+wPH),kPHE,k2,k3,AC,'PHE')

        u0=self.uF(kPP,kPH,kPHE,0)
        u0=np.tile(u0[np.newaxis,:],(len(wPP),1))
        UnV=u0+uPP+uPH+uPHE

        return np.reshape(UnV,uShape)

    def _expandChannel(self,wS,wSX,wSY,kS,kSX,kSY,AC,chnL):
        if chnL is 'PP':
            UnX = self.UnPP
        elif chnL is 'PH':
            UnX = self.UnPH
        elif chnL is 'PHE':
            UnX = self.UnPHE

        NW=self.NW
        NKF=self.NKF
        nDim=self.nDim
        N1D=self.N1D

        nWPoints=len(wS)
        nKPoints=len(kS[0])
        nPatches=len(self.wB)
        kPatches=len(self.kB[0])

        kSIndex=auxF.indexOfMomenta(kS,N1D)
        fA=auxF.fScaling(AC)

        lTempKL=np.ones((nKPoints,NKF),np.complex_)
        lTempKR=np.ones((nKPoints,NKF),np.complex_)

        sPoints=self.sPoints
        for i in range(NKF):
            for j in range(nDim):
                lTempKL[:,i]=lTempKL[:,i]*np.exp(-1j*sPoints[j][i]*kSX[j])
                lTempKR[:,i]=lTempKR[:,i]*np.exp(1j*sPoints[j][i]*kSY[j])

        lTempL=np.zeros((nWPoints,NW),dtype=np.complex_)
        lTempR=np.zeros((nWPoints,NW),dtype=np.complex_)
        tAvg,deltaT=self.tPoints
        for i in range(NW):
            lTempL[:,i]=np.exp(1j*wSX*(tAvg[i]/fA))*\
                np.sinc((0.5*(deltaT[i]/fA)*wSX)/np.pi)
            lTempR[:,i]=np.exp(1j*wSY*(tAvg[i]/fA))*\
                np.sinc((0.5*(deltaT[i]/fA)*wSY)/np.pi)

        UnX=np.reshape(UnX,(nPatches,kPatches,NW,NKF,NW,NKF))
        #UnXc,UnXindC=auxF.compVertex(UnX,self.kB,self.ikBr,NW,NKF,N1D,self.sPoints)
        #UnXc=auxF.compVertex(UnX,self.iSymm,self.N1D,NW,NKF,self.sPoints)
        #indKL,indKR,indWL,indWR=auxF.indWKlr(UnXindC,NW,NKF)
        UnV=np.zeros((nWPoints,nKPoints),dtype=np.complex_)
        for i in range(NW):
            for j in range(NW):
                wTemp=lTempL[:,i]*lTempR[:,j]
                for k in range(NKF):
                    for l in range(NKF):
                        kTemp=lTempKL[:,k]*lTempKR[:,l]
                        UnIs=auxF.linInterpO(self.wB,UnX[:,:,i,k,j,l],wS)
                        UnV+=UnIs[:,kSIndex]*wTemp*kTemp

        return UnV

    def projectChannelDir(self,UnX,AC,chnL):
        if chnL is 'PP':
            kTrans1=self.kTransPPtoPH
            kTrans2=self.kTransPPtoPHE

            wTrans1=self.wTransPPtoPH
            wTrans2=self.wTransPPtoPHE
        elif chnL is 'PH':
            kTrans1=self.kTransPHtoPP
            kTrans2=self.kTransPHtoPHE

            wTrans1=self.wTransPHtoPP
            wTrans2=self.wTransPHtoPHE
        elif chnL is 'PHE':
            kTrans1=self.kTransPHEtoPP
            kTrans2=self.kTransPHEtoPH

            wTrans1=self.wTransPHEtoPP
            wTrans2=self.wTransPHEtoPH

        NW,NKF=self.NW,self.NKF
        wB,kB=self.wB,self.kBX
        nPatches=len(wB)
        kPatches=len(kB[0])
        NLmax=self.NLmax
        N=self.N
        N1D=self.N1D
        nDim=self.nDim

        kIndexLoc=auxF.indexOfMomenta(self.kB,N1D)

        NKFr=N*NKF*NKF
        UnF1=np.zeros((nPatches,NKFr,NW,NW),dtype=np.complex_)
        UnF2=np.zeros((nPatches,NKFr,NW,NW),dtype=np.complex_)

        fA=np.sqrt(AC**2+1)
        for i in range(NW):
            for j in range(NW):
                indexCur=wTrans1[i][j][0]
                UnF1[:,:,i,j]=np.sum(UnX[indexCur,None,:]*(wTrans1[i][j][1][:,:,None]),axis=0)

                indexCur=wTrans2[i][j][0]
                UnF2[:,:,i,j]=np.sum(UnX[indexCur,None,:]*(wTrans2[i][j][1][:,:,None]),axis=0)

        UnF1=auxF.linInterpO(fA*wB,UnF1,wB,axis=0)
        UnF2=auxF.linInterpO(fA*wB,UnF2,wB,axis=0)

        UnP1=np.zeros((nPatches,kPatches,NW,NKF,NW,NKF),dtype=np.complex_)
        uPhase1=np.zeros((kPatches,NKF,NKF),dtype=np.complex_)
        UnP2=np.zeros((nPatches,kPatches,NW,NKF,NW,NKF),dtype=np.complex_)
        uPhase2=np.zeros((kPatches,NKF,NKF),dtype=np.complex_)

        for i in range(NKF):
            for j in range(NKF):
                curIndex=kTrans1[i][j][0]
                fftIndex=kTrans1[i][j][1]
                kPhase=kTrans1[i][j][2]

                uPhase1[:,i,j]=kPhase
                uTemp=np.zeros((nPatches,kPatches,NW,NW),dtype=np.complex_)
                uTemp[:,fftIndex,:,:]=UnF1[:,curIndex,:,:]
                UnP1[:,:,:,i,:,j]=uTemp

                curIndex=kTrans2[i][j][0]
                fftIndex=kTrans2[i][j][1]
                kPhase=kTrans2[i][j][2]

                uPhase2[:,i,j]=kPhase
                uTemp=np.zeros((nPatches,kPatches,NW,NW),dtype=np.complex_)
                uTemp[:,fftIndex,:,:]=UnF2[:,curIndex,:,:]
                UnP2[:,:,:,i,:,j]=uTemp

        UnP1=auxF.iffTransformD(UnP1,N1D,nDim,axis=1)*uPhase1[None,:,None,:,None,:]
        UnP2=auxF.iffTransformD(UnP2,N1D,nDim,axis=1)*uPhase2[None,:,None,:,None,:]
        UnP1=np.reshape(UnP1,(nPatches,kPatches,NW*NKF,NW*NKF))
        UnP2=np.reshape(UnP2,(nPatches,kPatches,NW*NKF,NW*NKF))
        return UnP1,UnP2

    def projectChannel(self,UnX,AC,chnL):
        if chnL is 'PP':
            kTrans1=self.kTransPPtoPH
            kTrans2=self.kTransPPtoPHE

            wTrans1=self.wTransPPtoPH
            wTrans2=self.wTransPPtoPHE
        elif chnL is 'PH':
            kTrans1=self.kTransPHtoPP
            kTrans2=self.kTransPHtoPHE

            wTrans1=self.wTransPHtoPP
            wTrans2=self.wTransPHtoPHE
        elif chnL is 'PHE':
            kTrans1=self.kTransPHEtoPP
            kTrans2=self.kTransPHEtoPH

            wTrans1=self.wTransPHEtoPP
            wTrans2=self.wTransPHEtoPH

        NW,NKF=self.NW,self.NKF
        wB,kB=self.wB,self.kBX
        nPatches=len(wB)
        kPatches=len(kB[0])

        wBI=self.wBI
        NWs=self.NWfull
        NWs2=int(NWs/2)
        NLmax=self.NLmax
        N=self.N
        N1D=self.N1D
        nDim=self.nDim

        NWFr=NLmax*NW*NW
        UnP1=np.zeros((NWFr,kPatches,NKF,NKF),dtype=np.complex_)
        uPhase1=np.zeros((kPatches,NKF,NKF),dtype=np.complex_)
        UnP2=np.zeros((NWFr,kPatches,NKF,NKF),dtype=np.complex_)
        uPhase2=np.zeros((kPatches,NKF,NKF),dtype=np.complex_)

        for i in range(NKF):
            for j in range(NKF):
                curIndex=kTrans1[i][j][0]
                fftIndex=kTrans1[i][j][1]
                kPhase=kTrans1[i][j][2]

                uPhase1[:,i,j]=kPhase
                uTemp=np.zeros((NWFr,kPatches),dtype=np.complex_)
                uTemp[:,fftIndex]=UnX[:,curIndex]
                UnP1[:,:,i,j]=uTemp

                curIndex=kTrans2[i][j][0]
                fftIndex=kTrans2[i][j][1]
                kPhase=kTrans2[i][j][2]

                uPhase2[:,i,j]=kPhase
                uTemp=np.zeros((NWFr,kPatches),dtype=np.complex_)
                uTemp[:,fftIndex]=UnX[:,curIndex]
                UnP2[:,:,i,j]=uTemp

        UnP1=auxF.iffTransformD(UnP1,N1D,nDim,axis=1)*uPhase1[None,:,:,:]
        UnP2=auxF.iffTransformD(UnP2,N1D,nDim,axis=1)*uPhase2[None,:,:,:]


        kIndexLoc=auxF.indexOfMomenta(self.kB,N1D)
        pFac=np.exp(1j*np.arange(NWs2)*np.pi)

        NKFr=N*NKF*NKF
        UnF1=np.zeros((NWs,kPatches,NW,NKF,NW,NKF),dtype=np.complex_)
        UnF2=np.zeros((NWs,kPatches,NW,NKF,NW,NKF),dtype=np.complex_)

        fA=auxF.fScaling(AC)
        for i in range(NW):
            for j in range(NW):
                lenInt=len(wTrans1[i][j][0])
                uIndexCur,weightCur=wTrans1[i][j][2][0],wTrans1[i][j][2][1]
                uCurVal=weightCur[:,None,None,None]*UnP1[uIndexCur,:,:,:]
                for intL in range(lenInt):
                    wPhase=wTrans1[i][j][0][intL]
                    curIndex=wTrans1[i][j][1][intL]
                    fftIndex=wTrans1[i][j][3][intL]

                    UnRt=np.zeros((NWs,kPatches,NKF,NKF),dtype=np.complex_)
                    UnRt[fftIndex,:,:,:]=np.sum(uCurVal[curIndex,:,:,:],axis=0)
                    UnRN=timeF.iffTransformW(UnRt[:NWs2,:,:,:]*pFac[:,None,None,None],NWs2,axis=0)
                    UnRN+=timeF.iffTransformW(UnRt[NWs2:,:,:,:]*pFac[:,None,None,None],NWs2,axis=0)

                    UnF1[:,:,i,:,j,:]+=wPhase[:,None,None,None]*np.append(UnRN,UnRN,axis=0)

                lenInt=len(wTrans2[i][j][0])
                uIndexCur,weightCur=wTrans2[i][j][2][0],wTrans2[i][j][2][1]
                uCurVal=weightCur[:,None,None,None]*UnP2[uIndexCur,:,:,:]
                for intL in range(lenInt):
                    wPhase=wTrans2[i][j][0][intL]
                    curIndex=wTrans2[i][j][1][intL]
                    fftIndex=wTrans2[i][j][3][intL]

                    UnRt=np.zeros((NWs,kPatches,NKF,NKF),dtype=np.complex_)
                    UnRt[fftIndex,:,:,:]=np.sum(uCurVal[curIndex,:,:,:],axis=0)
                    UnRN=timeF.iffTransformW(UnRt[:NWs2,:,:,:]*pFac[:,None,None,None],NWs2,axis=0)
                    UnRN+=timeF.iffTransformW(UnRt[NWs2:,:,:,:]*pFac[:,None,None,None],NWs2,axis=0)

                    UnF2[:,:,i,:,j,:]+=wPhase[:,None,None,None]*np.append(UnRN,UnRN,axis=0)

        UnF1=auxF.linInterpO(fA*wBI[:-1],UnF1[:-1,...],wB,axis=0)
        UnF2=auxF.linInterpO(fA*wBI[:-1],UnF2[:-1,...],wB,axis=0)

        UnF1=np.reshape(UnF1,(nPatches,kPatches,NW*NKF,NW*NKF))
        UnF2=np.reshape(UnF2,(nPatches,kPatches,NW*NKF,NW*NKF))
        return UnF1,UnF2

    def newSEtransv2(self,wTransCur,nIndex):
        mnN=len(wTransCur)
        lI,mnI=auxF.indXY(nIndex,mnN)
        mnR=np.unique(mnI)
        wRed=[[] for i in range(len(mnR))]
        for i in range(len(mnR)):
            indCur=wTransCur[mnR[i]][0]
            aSort=np.argsort(indCur)
            lRun=np.arange(len(indCur))[aSort]
            wInew=np.nonzero(np.in1d(indCur[aSort],nIndex))[0]
            iInew=np.nonzero(np.in1d(nIndex,indCur[aSort]))[0]

            wRed[i]=[iInew,lRun[wInew],wTransCur[mnR[i]][1]]

        return wRed,mnR
    def newSEtrans(self,wTransCur,nIndex):
        NLmax=len(wTransCur)
        wRed=[[] for i in range(NLmax)]
        for i in range(NLmax):
            wInew=np.nonzero(np.in1d(wTransCur[i][0],nIndex))[0]
            iInew=np.nonzero(np.in1d(nIndex,wTransCur[i][0]))[0]

            wRed[i]=[iInew,wTransCur[i][1][wInew,:]]

        return wRed

    def newVertexTrans(self,wTransCur,nIndex):
        NKF=len(wTransCur)
        wRed=[[] for i in range(NKF*NKF)]
        iC=0
        relInd=np.zeros(0,dtype=int)
        for i in range(NKF):
            for j in range(NKF):
                wInew=np.nonzero(np.in1d(wTransCur[i][j][0],nIndex))[0]
                iInew=np.nonzero(np.in1d(nIndex,wTransCur[i][j][0]))[0]

                if len(iInew)>0:
                    wRed[iC]=[iInew,wTransCur[i][j][1][wInew,:]]
                    relInd=np.append(relInd,i*NKF+j)
                    iC=iC+1

        return wRed[:iC],relInd
    def newVertexTransv2(self,wTransCur,nIndex):
        NKF=len(wTransCur)
        wRed=[[] for i in range(NKF*NKF)]
        iC=0
        relInd=np.zeros(0,dtype=int)
        for i in range(NKF):
            for j in range(NKF):
                indexIJ=wTransCur[i][j][0]
                aSort=np.argsort(indexIJ)
                indexIJ=indexIJ[aSort]
                fftIndexIJ=wTransCur[i][j][1][aSort]
                wInew=np.nonzero(np.in1d(indexIJ,nIndex))[0]
                iInew=np.nonzero(np.in1d(nIndex,indexIJ))[0]

                if len(iInew)>0:
                    wRed[iC]=[iInew,fftIndexIJ[wInew],wTransCur[i][j][2]]
                    relInd=np.append(relInd,i*NKF+j)
                    iC=iC+1

        return wRed[:iC],relInd

    def freqProjectionFFT(self):
        NWF=self.NW
        NWs=self.NWfull
        beTa=self.beTa
        wBI=self.wBI

        dT=beTa/NWs
        tPointsDT=np.arange(-NWs/2+1,NWs/2+1,1)*dT

        tAvg,deltaT=self.tPoints
        tPointsM,dTpointsM,indexM=timeF.indiciesInInterval(tAvg,deltaT)
        tPointsN,dTpointsN,indexN=timeF.indiciesInInterval(tAvg,deltaT)

        tAvgL,deltaTL=self.tPointsL
        tPointsL,dTpointsL,indexL=timeF.indiciesInInterval(tAvgL,deltaTL)

        tPointsME,dTpointsME,indexME=timeF.clusterPoints(tPointsM,dTpointsM*deltaT[indexM],tPointsDT)

        NWm=len(tPointsM)
        NWn=len(tPointsN)
        NWsL=len(tPointsL)

        lCur=np.tile(tPointsL[:,np.newaxis,np.newaxis],(1,NWm,NWn))
        lIndexC=np.tile(indexL[:,np.newaxis,np.newaxis],(1,NWm,NWn))
        lCur=np.reshape(lCur,NWsL*NWm*NWn)
        lIndexC=np.reshape(lIndexC,NWsL*NWm*NWn)

        mCur=np.tile(tPointsM[np.newaxis,:,np.newaxis],(NWsL,1,NWn))
        mIndexC=np.tile(indexM[np.newaxis,:,np.newaxis],(NWsL,1,NWn))
        mIndexCE=np.tile(np.arange(len(indexM))[np.newaxis,:,np.newaxis],(NWsL,1,NWn))

        mCur=np.reshape(mCur,NWsL*NWm*NWn)
        mIndexC=np.reshape(mIndexC,NWsL*NWm*NWn)
        mIndexCE=np.reshape(mIndexCE,NWsL*NWm*NWn)

        nCur=np.tile(tPointsN[np.newaxis,np.newaxis,:],(NWsL,NWm,1))
        nIndexC=np.tile(indexN[np.newaxis,np.newaxis,:],(NWsL,NWm,1))

        nCur=np.reshape(nCur,NWsL*NWm*NWn)
        nIndexC=np.reshape(nIndexC,NWsL*NWm*NWn)

        uIndexFullV=lIndexC*NWF*NWF+mIndexC*NWF+nIndexC
        dTpointsM=np.zeros(len(tPointsM))
        for i in range(len(tPointsM)):
            dTpointsM[i]=dTpointsME[np.where(i==indexME)[0][0]]*dTpointsN[i]
        weightFullV=(dTpointsM[:,None]*dTpointsN[None,:])[None,:,:]*dTpointsL[:,None,None]
        weightFullV=np.reshape(weightFullV,NWsL*NWm*NWn)

        def wIndexM(indexWi,sgn=1):
            indexWcur=(sgn*indexWi/dT).astype(int)
            indexWn=np.mod(indexWcur,NWs)
            return indexWn.astype(int)

        toL=0.001
        def findInd(XtoYA,XtoYB,deltaTA,deltaTB,sgnW):
            zLoc1a=np.where(np.abs(XtoYA)<=0.5*deltaTA)[0]
            zLoc1b=np.where(np.abs(XtoYA)>=(beTa-0.5*deltaTA))[0]
            zLoc1=np.append(zLoc1a,zLoc1b)

            zLoc2a=np.where(np.abs(XtoYB)<=0.5*deltaTB)[0]
            zLoc2b=np.where(np.abs(XtoYB)>=(beTa-0.5*deltaTB))[0]
            zLoc2=np.append(zLoc2a,zLoc2b)

            uIndex=np.intersect1d(zLoc1,zLoc2)
            tmValid=mCur[uIndex]
            tnValid=nCur[uIndex]

            mValid=mIndexC[uIndex]
            nValid=nIndexC[uIndex]
            lValid=lIndexC[uIndex]

            mValidE=mIndexCE[uIndex]

            lUnique=np.unique(lValid)
            if sgnW==(-1):
                tmnValid=np.mod(-tmValid+tnValid,beTa)
            elif sgnW==1:
                tmnValid=np.mod(tmValid+tnValid,beTa)
            tmnValid=np.round(tmnValid,4)
            tmnUnique=np.unique(tmnValid)
            lenPoints=len(lUnique)*len(tmnUnique)

            uIndexFull=[]
            fftIndexFull=[]
            weightFull=[]
            wPhase=[]

            for j in range(len(tmnUnique)):
                weightMatrixLM=np.zeros((len(lUnique),NWs))
                uIndexMatrixLM=np.zeros((len(lUnique),NWs),dtype=int)
                runner=0
                mIndexFFTlist=np.zeros(0,dtype=int)
                for i in range(len(lUnique)):
                    locL=np.where(lValid==lUnique[i])[0]
                    locMN=np.where(tmnValid==tmnUnique[j])[0]
                    loc=np.intersect1d(locL,locMN)

                    uIndexList=uIndexFullV[uIndex][loc]
                    weightList=weightFullV[uIndex][loc]
                    mIndexList=mValidE[loc]

                    mIndexUnique=np.unique(mIndexList)
                    uIndexUnique=np.zeros(len(mIndexUnique),dtype=int)
                    weightUnique=np.zeros(len(mIndexUnique))

                    for k in range(len(mIndexUnique)):
                        locRR=np.where(mIndexUnique[k]==mIndexList)[0]
                        uIndexUnique[k]=uIndexList[locRR[0]]
                        for l in locRR:
                            weightUnique[k]+=weightList[l]

                    if len(mIndexUnique)>0:
                        runner+=1
                        uIndexUnqExt=np.zeros(0,dtype=int)
                        weightUnqExt=np.zeros(0)
                        tmFFText=np.zeros(0)

                        for k in range(len(mIndexUnique)):
                            locExtV=np.where(mIndexUnique[k]==indexME)[0]
                            tmFFText=np.append(tmFFText,tPointsME[locExtV])
                            weightUnqExt=np.append(weightUnqExt,np.zeros(len(locExtV))+weightUnique[k])
                            uIndexUnqExt=np.append(uIndexUnqExt,np.zeros(len(locExtV),dtype=int)+uIndexUnique[k])

                        fftLoc=wIndexM(tmFFText,sgnW)
                        mIndexFFTlist=np.append(mIndexFFTlist,fftLoc)
                        mIndexFFTlist=np.unique(mIndexFFTlist)
                        weightMatrixLM[i,fftLoc]=weightUnqExt
                        uIndexMatrixLM[i,fftLoc]=uIndexUnqExt
                if runner>0 and weightMatrixLM.max()>0.1:
                    wPhase.append(tmnUnique[j])
                    uIndexFull.append(uIndexMatrixLM)
                    weightFull.append(weightMatrixLM)
                    fftIndexFull.append(mIndexFFTlist)

            wPhaseF=[]
            uIndexFullF=[]
            weightFullF=[]
            fftIndexFullF=[]
            while len(wPhase)>0:
                wPhaseCur=np.exp(1j*wBI*wPhase[0])
                fftIndexCur=fftIndexFull[0]

                origLoc=0
                iLocS=np.zeros(1,dtype=int)
                for i in range(1,len(wPhase)):
                    overLap=np.intersect1d(fftIndexCur,fftIndexFull[i])
                    conD1=len(overLap)>0.8*len(fftIndexCur)
                    value2=np.abs(weightFull[i][:,overLap]-weightFull[origLoc][:,overLap])

                    if conD1 and value2.max()<0.01:
                        fftIndexCur=np.intersect1d(fftIndexCur,fftIndexFull[i])
                        iTest=[origLoc,i]
                        origLoc=iTest[np.argmax([np.sum(weightFull[origLoc]),np.sum(weightFull[i])])]
                        wPhaseCur+=np.exp(1j*wBI*wPhase[i])
                        iLocS=np.append(iLocS,i)
                wPhaseF.append(wPhaseCur)
                uIndexFullF.append(uIndexFull[origLoc][:,fftIndexCur])
                weightFullF.append(weightFull[origLoc][:,fftIndexCur])
                fftIndexFullF.append(fftIndexCur)

                for i in range(len(iLocS)):
                    wPhase.pop(iLocS[i]-i)
                    fftIndexFull.pop(iLocS[i]-i)
                    uIndexFull.pop(iLocS[i]-i)
                    weightFull.pop(iLocS[i]-i)

            uniqueVertTemp=np.zeros(0,dtype=int)
            for i in range(len(wPhaseF)):
                uniqueVertTemp=np.append(uniqueVertTemp,np.unique(np.reshape(uIndexFullF[i],uIndexFullF[i].size)))
                uniqueVertTemp=np.unique(uniqueVertTemp)

            uniqueVertIndex=np.zeros(0,dtype=int)
            uniqueWeightVert=np.zeros(0)
            for i in range(len(uniqueVertTemp)):
                weightVertTemp=np.zeros(0)
                for j in range(len(wPhaseF)):
                    locUind=np.where(uniqueVertTemp[i]==np.reshape(uIndexFullF[j],uIndexFullF[j].size))[0]
                    weightVertCur=np.unique(np.reshape(weightFullF[j],uIndexFullF[j].size)[locUind])
                    weightVertTemp=np.unique(np.append(weightVertTemp,weightVertCur))
                uniqueVertIndex=np.append(uniqueVertIndex,np.zeros(len(weightVertTemp),dtype=int)+uniqueVertTemp[i])
                uniqueWeightVert=np.append(uniqueWeightVert,weightVertTemp)

            uIndexFullFf=[]
            for i in range(len(wPhaseF)):
                uShape=uIndexFullF[i].shape
                uIndexTemp=np.zeros(uShape,dtype=int)
                for j in range(uShape[0]):
                    for k in range(uShape[1]):
                        locUjk=np.where(uIndexFullF[i][j][k]==uniqueVertIndex)[0]
                        locWjk=np.where(weightFullF[i][j][k]==uniqueWeightVert)[0]

                        uIndexTemp[j,k]=np.intersect1d(locUjk,locWjk)
                uIndexFullFf.append(uIndexTemp)
            return [wPhaseF,uIndexFullFf,[uniqueVertIndex,uniqueWeightVert],fftIndexFullF]

        self.wTransPPtoPH=[[[] for i in range(NWF)] for j in range(NWF)]
        self.wTransPHEtoPH=[[[] for i in range(NWF)] for j in range(NWF)]
        self.wTransPPtoPHE=[[[] for i in range(NWF)] for j in range(NWF)]
        self.wTransPHtoPHE=[[[] for i in range(NWF)] for j in range(NWF)]
        self.wTransPHtoPP=[[[] for i in range(NWF)] for j in range(NWF)]
        self.wTransPHEtoPP=[[[] for i in range(NWF)] for j in range(NWF)]

        for a in range(NWF):
            for b in range(NWF):
                XtoYA=np.mod(0.5*lCur-0.5*mCur+0.5*nCur-tAvg[a],beTa)
                XtoYB=np.mod(0.5*lCur+0.5*mCur-0.5*nCur+tAvg[b],beTa)

                self.wTransPHtoPP[a][b]=findInd(XtoYA,XtoYB,deltaT[a],deltaT[b],-1)

                XtoYA=np.mod(-0.5*lCur+0.5*mCur-0.5*nCur-tAvg[a],beTa)
                XtoYB=np.mod(0.5*lCur+0.5*mCur-0.5*nCur+tAvg[b],beTa)

                self.wTransPHEtoPP[a][b]=findInd(XtoYA,XtoYB,deltaT[a],deltaT[b],-1)

                XtoYA=np.mod(0.5*lCur+0.5*mCur+0.5*nCur+tAvg[a],beTa)
                XtoYB=np.mod(0.5*lCur-0.5*mCur-0.5*nCur+tAvg[b],beTa)

                self.wTransPPtoPH[a][b]=findInd(XtoYA,XtoYB,deltaT[a],deltaT[b],1)

                XtoYA=np.mod(-0.5*lCur-0.5*mCur-0.5*nCur+tAvg[a],beTa)
                XtoYB=np.mod(0.5*lCur-0.5*mCur-0.5*nCur+tAvg[b],beTa)

                self.wTransPHEtoPH[a][b]=findInd(XtoYA,XtoYB,deltaT[a],deltaT[b],1)

                XtoYA=np.mod(0.5*lCur-0.5*mCur+0.5*nCur+tAvg[a],beTa)
                XtoYB=np.mod(0.5*lCur+0.5*mCur-0.5*nCur+tAvg[b],beTa)

                self.wTransPPtoPHE[a][b]=findInd(XtoYA,XtoYB,deltaT[a],deltaT[b],-1)

                XtoYA=np.mod(-0.5*lCur-0.5*mCur-0.5*nCur+tAvg[a],beTa)
                XtoYB=np.mod(0.5*lCur-0.5*mCur-0.5*nCur+tAvg[b],beTa)

                self.wTransPHtoPHE[a][b]=findInd(XtoYA,XtoYB,deltaT[a],deltaT[b],1)

    def freqProjection(self):
        wB=self.wB
        nPatches=len(wB)
        tAvg,deltaT=self.tPoints
        beTa=self.beTa

        NW=self.NW
        NLmax=self.NLmax

        wIndexPHtoPP=[[np.zeros(0,dtype=int) for i in range(NW)] for j in range(NW)]
        wIndexPHEtoPP=[[np.zeros(0,dtype=int) for i in range(NW)] for j in range(NW)]
        wIndexPPtoPH=[[np.zeros(0,dtype=int) for i in range(NW)] for j in range(NW)]
        wIndexPHEtoPH=[[np.zeros(0,dtype=int) for i in range(NW)] for j in range(NW)]
        wIndexPPtoPHE=[[np.zeros(0,dtype=int) for i in range(NW)] for j in range(NW)]
        wIndexPHtoPHE=[[np.zeros(0,dtype=int) for i in range(NW)] for j in range(NW)]

        wTransPHtoPP=[[np.zeros((0,nPatches),dtype=np.complex_) for i in range(NW)] for j in range(NW)]
        wTransPPtoPHE=[[np.zeros((0,nPatches),dtype=np.complex_) for i in range(NW)] for j in range(NW)]
        wTransPHEtoPP=[[np.zeros((0,nPatches),dtype=np.complex_) for i in range(NW)] for j in range(NW)]
        wTransPHtoPHE=[[np.zeros((0,nPatches),dtype=np.complex_) for i in range(NW)] for j in range(NW)]
        wTransPPtoPH=[[np.zeros((0,nPatches),dtype=np.complex_) for i in range(NW)] for j in range(NW)]
        wTransPHEtoPH=[[np.zeros((0,nPatches),dtype=np.complex_) for i in range(NW)] for j in range(NW)]

        tFX,deltaTX=timeF.pointsInInterval(tAvg,deltaT)
        tAvgL,deltaTL=self.tPointsL
        tFXl,dTl,dIndl=timeF.indiciesInInterval(tAvgL,deltaTL)

        #tFXl=self.tAvgV
        #dTl=np.zeros(len(tAvgL))+1
        #dIndl=np.arange(len(tFXl))

        def fillWIndex(wIndCur,wTransCur,tA,tB,tS,indexV,deltaTx,sgnW):
            aLIndex=np.zeros(NW,dtype=int)-1
            bLIndex=np.zeros(NW,dtype=int)-1

            indA=[[] for i in range(NW)]
            indB=[[] for i in range(NW)]
            for a in range(NW):
                t1=tA-sgnW*tAvg[a]
                t1=auxF.tBetaShift(t1,beTa)[0]
                indL,indM,indN=np.where(np.abs(t1)<(0.5*deltaT[a]))

                if len(indL)!=0:
                    aLIndex[a]=a
                    indA[a]=[indL,indM,indN]

                t1=tB-tAvg[a]
                t1=auxF.tBetaShift(t1,beTa)[0]
                indL,indM,indN=np.where(np.abs(t1)<(0.5*deltaT[a]))

                if len(indL)!=0:
                    bLIndex[a]=a
                    indB[a]=[indL,indM,indN]
            aLIndex=np.tile(aLIndex[:,None],(1,NW))
            bLIndex=np.tile(bLIndex[None,:],(NW,1))
            conD=np.logical_and(aLIndex>(-1),bLIndex>(-1))

            aLIndex=aLIndex[conD]
            bLIndex=bLIndex[conD]
            lenC=len(aLIndex)

            for a in range(lenC):
                aC=aLIndex[a]
                bC=bLIndex[a]
                lSizeA=t1.shape[1]
                lSizeB=t1.shape[2]
                inDW1=indA[aC][0]*lSizeA*lSizeB+indA[aC][1]*lSizeB+indA[aC][2]
                inDW2=indB[bC][0]*lSizeA*lSizeB+indB[bC][1]*lSizeB+indB[bC][2]
                inDV=np.intersect1d(inDW1,inDW2)

                if len(inDV)!=0:
                    lINew=np.floor(inDV/(lSizeA*lSizeB)).astype(int)
                    mINew=np.floor(np.mod(inDV,lSizeA*lSizeB)/lSizeB).astype(int)
                    nINew=np.mod(np.mod(inDV,lSizeA*lSizeB),lSizeB).astype(int)

                    dIndlE=np.tile(dIndl[:,None,None],(1,tA.shape[1],tA.shape[2]))
                    dIndlE=dIndlE[lINew,mINew,nINew]
                    tSr=tS[lINew,mINew,nINew]
                    deltaTxr=deltaTx[lINew,mINew,nINew]
                    for b in range(NLmax):
                        indCurL=np.where(dIndlE==b)[0]
                        if len(indCurL)!=0:
                            tSc=tSr[indCurL]
                            tSe=auxF.tBetaShift(tSc,beTa)[0]
                            deltaTxN=deltaTxr[indCurL]
                            phaseFac=np.sum(deltaTxN[None,:]*np.exp(1j*wB[:,None]*tSe[None,:]),axis=1)
                            wIndCur[aC][bC]=np.append(wIndCur[aC][bC],indexV+b*NW**2)
                            wTransCur[aC][bC]=np.concatenate((wTransCur[aC][bC],phaseFac[None,:]),axis=0)

        for m in range(NW):
            tM=tFX[m]
            deltaTM=deltaTX[m]

            for n in range(NW):
                tN=tFX[n]
                deltaTN=deltaTX[n]

                tLE=np.tile(tFXl[:,None,None],(1,len(tM),len(tN)))
                tME=np.tile(tM[None,:,None],(len(tFXl),1,len(tN)))
                tNE=np.tile(tN[None,None,:],(len(tFXl),len(tM),1))

                deltaTME=np.tile(deltaTM[None,:,None],(len(tFXl),1,len(tN)))
                deltaTNE=np.tile(deltaTN[None,None,:],(len(tFXl),len(tM),1))

                deltaTx=deltaTME*deltaTNE*dTl[:,None,None]
                indexMNL=m*NW+n

                tPHtoPPA=-0.5*(tLE-tME+tNE)
                tPHtoPPB=-0.5*(tLE+tME-tNE)
                tS=(tME+tNE)
                fillWIndex(wIndexPHtoPP,wTransPHtoPP,tPHtoPPA,tPHtoPPB,tS,indexMNL,deltaTx,-1)

                tPHEtoPPA=0.5*(tLE-tME+tNE)
                tPHEtoPPB=0.5*(-tLE-tME+tNE)
                tS=(tME+tNE)
                fillWIndex(wIndexPHEtoPP,wTransPHEtoPP,tPHEtoPPA,tPHEtoPPB,tS,indexMNL,deltaTx,-1)

                tPPtoPHA=0.5*(-tLE-tME-tNE)
                tPPtoPHB=0.5*(-tLE+tME+tNE)
                tS=(-tME+tNE)
                fillWIndex(wIndexPPtoPH,wTransPPtoPH,tPPtoPHA,tPPtoPHB,tS,indexMNL,deltaTx,1)

                tPPtoPHEA=0.5*(-tLE+tME-tNE)
                tPPtoPHEB=0.5*(-tLE-tME+tNE)
                tS=(tME+tNE)
                fillWIndex(wIndexPPtoPHE,wTransPPtoPHE,tPPtoPHEA,tPPtoPHEB,tS,indexMNL,deltaTx,1)

                tPHEtoPHA=0.5*(tLE+tME+tNE)
                tPHEtoPHB=0.5*(-tLE+tME+tNE)
                tS=(-tME+tNE)
                fillWIndex(wIndexPHEtoPH,wTransPHEtoPH,tPHEtoPHA,tPHEtoPHB,tS,indexMNL,deltaTx,1)

        wInd=np.argmin(np.abs(wB))
        toL=0.0
        self.wTransPHtoPP=[[[] for i in range(NW)] for j in range(NW)]
        self.wTransPHEtoPP=[[[] for i in range(NW)] for j in range(NW)]
        self.wTransPPtoPH=[[[] for i in range(NW)] for j in range(NW)]
        self.wTransPHEtoPH=[[[] for i in range(NW)] for j in range(NW)]
        self.wTransPPtoPHE=[[[] for i in range(NW)] for j in range(NW)]
        self.wTransPHtoPHE=[[[] for i in range(NW)] for j in range(NW)]
        for i in range(NW):
            for j in range(NW):
                wIndexCur=wIndexPHtoPP[i][j]
                wZ=wTransPHtoPP[i][j]
                aSort=np.argsort(wIndexCur)
                self.wTransPHtoPP[i][j]=[np.asarray(wIndexCur[aSort]),wZ[aSort,:]]

                wIndexCur=wIndexPPtoPH[i][j]
                wZ=wTransPPtoPH[i][j]
                aSort=np.argsort(wIndexCur)
                self.wTransPPtoPH[i][j]=[np.asarray(wIndexCur[aSort]),wZ[aSort,:]]

                wIndexCur=wIndexPHEtoPP[i][j]
                wZ=wTransPHEtoPP[i][j]
                aSort=np.argsort(wIndexCur)
                self.wTransPHEtoPP[i][j]=[np.asarray(wIndexCur[aSort]),wZ[aSort,:]]

                wIndexCur=wIndexPHEtoPH[i][j]
                wZ=wTransPHEtoPH[i][j]
                aSort=np.argsort(wIndexCur)
                self.wTransPHEtoPH[i][j]=[np.asarray(wIndexCur[aSort]),wZ[aSort,:]]
                self.wTransPHtoPHE[i][j]=[np.asarray(wIndexCur[aSort]),wZ[aSort,:]]

                wIndexCur=wIndexPPtoPHE[i][j]
                wZ=wTransPPtoPHE[i][j]
                aSort=np.argsort(wIndexCur)
                self.wTransPPtoPHE[i][j]=[np.asarray(wIndexCur[aSort]),wZ[aSort,:]]

    def selfExpandFFT(self):
        NWF=self.NW
        NWs=self.NWfull
        beTa=self.beTa

        dT=beTa/NWs
        tPointsDT=np.arange(-NWs/2+1,NWs/2+1,1)*dT
        tAvg,deltaT=self.tPoints
        #tPointsMN,dTpointsMN,indexMN=timeF.clusterPoints(tAvg,deltaT,tPointsDT)
        tPointsMN,dTpointsMN,indexMN=timeF.indiciesInInterval(tAvg,deltaT)

        tAvgL,deltaTL=self.tPointsL
        tPointsL,dTpointsL,indexL=timeF.clusterPoints(tAvgL,deltaTL,tPointsDT)

        tPointsDA=dT*np.arange(NWs)
        uIndexFull=np.zeros(0,dtype=int)
        fftIndex=np.zeros(0,dtype=int)
        weightFull=np.zeros(0)

        for i in range(NWs):
            locL=np.where(tPointsDA[i]==np.mod(tPointsL,beTa))[0]
            if len(locL)>0:
                uIndexFull=np.append(uIndexFull,indexL[locL])
                fftIndex=np.append(fftIndex,i)
                weightFull=np.append(weightFull,dTpointsL[locL])

        self.uVertExtend=[fftIndex,uIndexFull,weightFull]

        def findIndex(tXcur):
            indexCurrent=np.zeros(tXcur.shape,dtype=int)
            for i in range(tXcur.shape[0]):
                for j in range(tXcur.shape[1]):
                    lTcur=np.mod(tXcur[i,j],beTa)-tPointsDA
                    indexCurrent[i,j]=np.argmin(np.abs(lTcur))
            return indexCurrent

        self.wTransSEpp1=[[[] for i in range(NWF)] for j in range(NWF)]
        self.wTransSEpp2=[[[] for i in range(NWF)] for j in range(NWF)]
        self.wTransSEph=[[[] for i in range(NWF)] for j in range(NWF)]
        for i in range(NWF):
            tPointsI=tPointsMN[indexMN==i]
            weightsI=dTpointsMN[indexMN==i]
            for j in range(NWF):
                tPointsJ=tPointsMN[indexMN==j]
                weightsJ=dTpointsMN[indexMN==j]

                tPointsIe=np.repeat(tPointsI,len(tPointsJ))
                tPointsJe=np.tile(tPointsJ,len(tPointsI))

                weightsIJ=np.repeat(weightsI,len(weightsJ))
                weightsIJ=weightsIJ*np.tile(weightsJ,len(weightsI))

                lenPoints=len(tPointsIe)

                tPP1L=(tPointsIe+tPointsJe)[:,None]+tPointsDA[None,:]
                indPP1L=findIndex(tPP1L)
                wPhasePP1=-2*(tPointsIe+tPointsJe)
                weightsPP1=-weightsIJ
                sIndexL=np.mod(-np.arange(NWs),NWs)
                self.wTransSEpp1[i][j]=[indPP1L,wPhasePP1,weightsPP1,sIndexL]

                tPP2L=(tPointsJe-tPointsIe)[:,None]+tPointsDA[None,:]
                indPP2L=findIndex(tPP2L)
                wPhasePP2=-2*(tPointsJe-tPointsIe)
                weightsPP2=-weightsIJ
                sIndexL=np.mod(-np.arange(NWs),NWs)
                self.wTransSEpp2[i][j]=[indPP2L,wPhasePP2,weightsPP2,sIndexL]

                tPHL=(tPointsIe+tPointsJe)[:,None]-tPointsDA[None,:]
                indPHL=findIndex(tPHL)
                wPhasePH=2*(tPointsIe+tPointsJe)
                weightsPH=1*weightsIJ
                sIndexL=np.arange(NWs)
                self.wTransSEph[i][j]=[indPHL,wPhasePH,weightsPH,sIndexL]

        wFI=self.wFI
        self.wTransSEphz=[[[] for i in range(NWF)] for j in range(NWF)]
        Nws=200000
        wF=(np.pi/beTa)*(2*np.arange(Nws)+1)
        wF=np.append(-wF[::-1],wF)
        tSpropL=(beTa/NWs)*np.arange(NWs)
        for j in range(NWF):
            tCurJ=tSpropL-2*tAvg[j]
            tCurJE,tCurSignE=auxF.tBetaShift(tCurJ,beTa)
            validLoc=np.where(np.abs(tCurJE)<(1*deltaT[j]))[0]
            sPropFac=np.zeros(NWs,dtype=np.complex_)
            fBasI=np.exp(-1j*wF*0.2*deltaT[j])*auxF.filtSinc(2*wF,deltaT[j])
            enT=np.sum(fBasI)

            for i in validLoc:
                sPropFac[i]=(tCurSignE[i]/beTa)*enT
            sPropFac[0]=0

            for i in range(NWF):
                wPhase=np.exp(1j*2*wFI*tAvg[i])*auxF.filtSinc(2*wFI,deltaT[i])
                self.wTransSEphz[i][j]=[sPropFac,wPhase]


    def selfExpand(self,wF,wFX,wFG,indexU,sPropKL,AC):
        nPatches=len(wF)
        beTa=self.beTa
        N=self.N
        nMoms=sPropKL.shape[1]
        NW=self.NW
        NLmax=self.NLmax

        fA=auxF.fScaling(AC)
        tAvgL=self.tPointsL[0]/fA
        deltaTL=self.tPointsL[1]/fA

        tAvg=self.tPoints[0]/fA
        deltaT=self.tPoints[1]/fA
        indexPP,indexPH,indexPHE=indexU

        wS=np.tile(wF[:,np.newaxis],(1,len(wFX)))
        wX=np.tile(wFX[np.newaxis,:],(len(wF),1))

        fSincL=np.zeros(wS.shape+(NLmax,))
        filT1=np.zeros(wS.shape+(NLmax,))
        filT2=np.zeros(wS.shape+(NLmax,))
        for i in range(NLmax):
            fSincL[...,i]=np.sinc((0.5*wX*deltaTL[i])/np.pi)
            filT=np.zeros(wS.shape)+1
            filT[np.abs(wS+wX)>((np.pi)/deltaTL[i])]=0
            filT1[...,i]=filT
            filT=np.zeros(wS.shape)+1
            #filT[np.abs(wS-wX)>((np.pi)/deltaTL[i])]=0
            filT2[...,i]=filT
        fSinc1=np.zeros(wS.shape+(NW,))
        fSinc2=np.zeros(wS.shape+(NW,))
        fSinc3=np.zeros(wS.shape+(NW,))
        fSinc4=np.zeros(wS.shape+(NW,))
        for i in range(NW):
            fSinc1[...,i]=np.sinc((0.5*(wS-wX)*deltaT[i])/np.pi)
            fSinc2[...,i]=np.sinc((0.5*(wS+wX)*deltaT[i])/np.pi)
            fSinc3[...,i]=np.sinc((0.5*(2*wS)*deltaT[i])/np.pi)
            fSinc4[...,i]=np.sinc((0.5*(2*wX)*deltaT[i])/np.pi)

        wIndexCur=indexPP
        lInd=np.floor(wIndexCur/NW**2).astype(int)
        mInd=np.floor(np.mod(wIndexCur,NW**2)/NW).astype(int)
        nInd=np.mod(np.mod(wIndexCur,NW**2),NW)

        tFull=tAvgL[lInd]+tAvg[mInd]+tAvg[nInd]
        deltaTFull=self.tPointsL[1][lInd]*self.tPoints[1][mInd]*self.tPoints[1][nInd]
        tO,indO=auxF.findUniquePoints(tFull,deltaTFull)

        sPropX=np.zeros((len(wIndexCur),nPatches,nMoms),dtype=np.complex_)
        for i in range(len(tO)):
            tFe,tFsign=auxF.tBetaShift(tO[i],beTa)
            iCur=indO[i][0]
            wInt=tFsign*np.exp(1j*wFX*tFe)[:,None]*sPropKL
            wInt=wInt[None,:,:]*(fSinc1[...,mInd[iCur]]*fSinc1[...,nInd[iCur]]*\
                filT1[...,lInd[iCur]])[:,:,None]

            sPropX[i,:,:]=(1.0/beTa)*np.sum(wFG[None,:,None]*wInt,axis=1)

        def createArray(tX,tS):
            sPropCur=np.zeros((len(wIndexCur),nPatches,nMoms),dtype=np.complex_)
            curLoc=auxF.locInArray(tX,deltaTFull,tO,indO)
            tS,tSsign=auxF.tBetaShift(tS,beTa)
            phaseCur=tSsign[:,None]*np.exp(1j*wF[None,:]*tS[:,None])
            for i in range(nMoms):
                sPropCur[:,:,i]=sPropX[curLoc,:,i]*phaseCur

            return sPropCur
        tX=-tAvgL[lInd]+tAvg[mInd]+tAvg[nInd]
        tS=-tAvgL[lInd]-tAvg[mInd]-tAvg[nInd]
        sPropPP1=createArray(tX,tS)
        #sPropPP1=auxF.fillMomVertex(sPropPP1,self.iSymm,self.N,axis=2)
        sPropPP1=auxF.ffTransformD(sPropPP1,self.N1D,self.nDim,axis=2)

        tX=-tAvgL[lInd]-tAvg[mInd]+tAvg[nInd]
        tS=-tAvgL[lInd]+tAvg[mInd]-tAvg[nInd]
        sPropPP2=createArray(tX,tS)
        #sPropPP2=auxF.fillMomVertex(sPropPP2,self.iSymm,self.N,axis=2)
        sPropPP2=auxF.ffTransformD(sPropPP2,self.N1D,self.nDim,axis=2)

        wIndexCur=indexPH
        lInd=np.floor(wIndexCur/NW**2).astype(int)
        mInd=np.floor(np.mod(wIndexCur,NW**2)/NW).astype(int)
        nInd=np.mod(np.mod(wIndexCur,NW**2),NW)

        tX=-tAvgL[lInd]+tAvg[mInd]+tAvg[nInd]
        tS=tAvgL[lInd]+tAvg[mInd]+tAvg[nInd]
        sPropPH=createArray(tX,tS)
        #sPropPH=auxF.fillMomVertex(sPropPH,self.iSymm,self.N,axis=2)
        sPropPH=auxF.ffTransformD(sPropPH,self.N1D,self.nDim,axis=2)

        wIndexCur=np.arange(NW**2)
        mInd=np.floor(wIndexCur/NW).astype(int)
        nInd=np.mod(wIndexCur,NW)
        sPropPHz=np.zeros((len(wIndexCur),nPatches,nMoms),dtype=np.complex_)
        for i in range(len(mInd)):
            tXe=2*tAvg[nInd[i]]
            tYe=2*tAvg[mInd[i]]
            tXe,tXsign=auxF.tBetaShift(tXe,beTa)
            tYe,tYsign=auxF.tBetaShift(tYe,beTa)

            wInt=tXsign*tYsign*np.exp(1j*wX*tXe+1j*wS*tYe)*\
                fSinc3[...,mInd[i]]*fSinc4[...,nInd[i]]
            sPropPHz[i,:,:]=(1/beTa)*np.sum(wFG[None,:,None]*wInt[:,:,None]*sPropKL[None,:,:],axis=1)
        #sPropPHz=auxF.fillMomVertex(sPropPHz,self.iSymm,self.N,axis=2)
        sPropPHz=auxF.ffTransformD(sPropPHz,self.N1D,self.nDim,axis=2)

        return sPropPP1,sPropPP2,sPropPH,sPropPHz

    def momProjectionFFTnD(self):
        NKF=self.NKF
        N=self.N
        N1D=self.N1D
        nDim=self.nDim
        kB=self.kB
        sPoints=self.sPoints

        indeX=np.arange(N1D)
        lIndO=[None for i in range(nDim)]
        lInd=[None for i in range(nDim)]
        mInd=[None for i in range(nDim)]
        nInd=[None for i in range(nDim)]
        lenRed=len(sPoints[0])
        for i in range(nDim):
            lCur=np.zeros((N1D,)*nDim,dtype=int)
            for j in range(N1D):
                lCur[j]=indeX[j]
            lCur=np.reshape(np.swapaxes(lCur,0,i),N1D**nDim)
            lIndO[i]=lCur
            lCur=np.tile(lCur[:,np.newaxis,np.newaxis],(1,lenRed,lenRed))
            lInd[i]=np.reshape(lCur,N*NKF*NKF)

            mCur=np.tile(sPoints[i][np.newaxis,:,np.newaxis],(N,1,lenRed))
            mInd[i]=np.mod(np.reshape(mCur,N*NKF*NKF),N1D)
            nCur=np.tile(sPoints[i][np.newaxis,np.newaxis,:],(N,lenRed,1))
            nInd[i]=np.mod(np.reshape(nCur,N*NKF*NKF),N1D)

        lF=np.arange(N)*NKF*NKF
        mF=np.arange(NKF)*NKF
        nF=np.arange(NKF)
        indexFull=lF[:,None,None]+mF[None,:,None]+nF[None,None,:]
        indexFull=np.reshape(indexFull,(N*NKF*NKF))

        self.kTransPHtoPP = [[[] for i in range(NKF)] for j in range(NKF)]
        self.kTransPHEtoPP = [[[] for i in range(NKF)] for j in range(NKF)]
        self.kTransPPtoPH = [[[] for i in range(NKF)] for j in range(NKF)]
        self.kTransPHEtoPH = [[[] for i in range(NKF)] for j in range(NKF)]
        self.kTransPPtoPHE = [[[] for i in range(NKF)] for j in range(NKF)]
        self.kTransPHtoPHE = [[[] for i in range(NKF)] for j in range(NKF)]

        def kIndexM(indexLi,sgn=1):
            indexLk=np.zeros(len(indexLi[0]))
            for i in range(nDim):
                indexLk+=np.mod(sgn*indexLi[i],N1D)*(N1D**(nDim-1-i))
            return indexLk.astype(int)

        def findInd(XtoY1,XtoY2):
            zLoc=[None for i in range(nDim)]
            for i in range(nDim):
                zLoc1=np.where(XtoY1[i]==0)[0]
                zLoc2=np.where(XtoY2[i]==0)[0]
                zLoc[i]=np.intersect1d(zLoc1,zLoc2)

            uIndex=zLoc[0]
            for i in range(1,nDim):
                uIndex=np.intersect1d(uIndex,zLoc[i])

            validEnt=len(uIndex)
            mValid=[None for i in range(nDim)]
            nValid=[None for i in range(nDim)]
            lValid=[None for i in range(nDim)]
            for i in range(nDim):
                mValid[i]=mInd[i][uIndex]
                nValid[i]=nInd[i][uIndex]
                lValid[i]=lInd[i][uIndex]

            return mValid,nValid,lValid,uIndex

        XtoY1=[None for i in range(nDim)]
        XtoY2=[None for i in range(nDim)]
        for a in range(lenRed):
            for b in range(lenRed):
                kPhase=np.ones(N,dtype=np.complex_)
                for i in range(nDim):
                    XtoY1[i]=np.mod(-lInd[i]-mInd[i]-sPoints[i][a],N1D)
                    XtoY2[i]=np.mod(lInd[i]-nInd[i]+sPoints[i][b],N1D)
                    kPhase=kPhase*np.exp(1j*kB[i]*sPoints[i][a])
                mValid,nValid,lValid,uIndex=findInd(XtoY1,XtoY2)
                self.kTransPHtoPP[a][b]=[uIndex,kIndexM(lValid,-1),kPhase]

                kPhase=np.ones(N,dtype=np.complex_)
                for i in range(nDim):
                    XtoY1[i]=np.mod(lInd[i]+mInd[i]-sPoints[i][a],N1D)
                    XtoY2[i]=np.mod(lInd[i]-nInd[i]+sPoints[i][b],N1D)
                    kPhase=kPhase*np.exp(1j*0*kB[i])
                mValid,nValid,lValid,uIndex=findInd(XtoY1,XtoY2)
                self.kTransPHEtoPP[a][b]=[uIndex,kIndexM(lValid,-1),kPhase]

                kPhase=np.ones(N,dtype=np.complex_)
                for i in range(nDim):
                    XtoY1[i]=np.mod(lInd[i]-sPoints[i][a],N1D)
                    XtoY2[i]=np.mod(lInd[i]+mInd[i]-nInd[i]+sPoints[i][b],N1D)
                    kPhase=kPhase*np.exp(1j*kB[i]*sPoints[i][a])
                mValid,nValid,lValid,uIndex=findInd(XtoY1,XtoY2)
                self.kTransPPtoPH[a][b]=[uIndex,kIndexM(mValid,-1),kPhase]

                kPhase=np.ones(N,dtype=np.complex_)
                for i in range(nDim):
                    XtoY1[i]=np.mod(-lInd[i]-sPoints[i][a],N1D)
                    XtoY2[i]=np.mod(lInd[i]+mInd[i]-nInd[i]+sPoints[i][b],N1D)
                    kPhase=kPhase*np.exp(1j*kB[i]*0)
                mValid,nValid,lValid,uIndex=findInd(XtoY1,XtoY2)
                self.kTransPHEtoPH[a][b]=[uIndex,kIndexM(mValid,-1),kPhase]

                kPhase=np.ones(N,dtype=np.complex_)
                for i in range(nDim):
                    XtoY1[i]=np.mod(lInd[i]+mInd[i]-sPoints[i][a],N1D)
                    XtoY2[i]=np.mod(lInd[i]-nInd[i]+sPoints[i][b],N1D)
                    kPhase=kPhase*np.exp(1j*kB[i]*0)
                mValid,nValid,lValid,uIndex=findInd(XtoY1,XtoY2)
                self.kTransPPtoPHE[a][b]=[uIndex,kIndexM(lValid,-1),kPhase]

                kPhase=np.ones(N,dtype=np.complex_)
                for i in range(nDim):
                    XtoY1[i]=np.mod(-lInd[i]-sPoints[i][a],N1D)
                    XtoY2[i]=np.mod(lInd[i]+mInd[i]-nInd[i]+sPoints[i][b],N1D)
                    kPhase=kPhase*np.exp(1j*kB[i]*0)
                mValid,nValid,lValid,uIndex=findInd(XtoY1,XtoY2)
                self.kTransPHtoPHE[a][b]=[uIndex,kIndexM(mValid,-1),kPhase]

    def momLegSEFnD(self):
        NKF=self.NKF
        N=self.N
        N1D=self.N1D
        nDim=self.nDim
        kB=self.kB
        sPoints=self.sPoints

        indeX=np.arange(N1D)
        lIndSing=[None for i in range(nDim)]
        for i in range(nDim):
            lCur=np.zeros((N1D,)*nDim,dtype=int)
            for j in range(N1D):
                lCur[j]=indeX[j]
            lCur=np.reshape(np.swapaxes(lCur,0,i),N1D**nDim)
            lIndSing[i]=lCur

        lF=np.arange(N)*NKF*NKF
        mF=np.arange(NKF)*NKF
        nF=np.arange(NKF)
        indexFull=lF[:,None,None]+mF[None,:,None]+nF[None,None,:]
        indexFull=np.reshape(indexFull,N*NKF*NKF)

        def kIndexM(indexLi,sgn=1):
            indexLk=np.zeros(len(indexLi[0]))
            for i in range(nDim):
                indexLk+=np.mod(sgn*indexLi[i],N1D)*(N1D**(nDim-1-i))
            return indexLk.astype(int)

        def findInd(XtoY1):
            zLoc=[None for i in range(nDim)]
            for i in range(nDim):
                zLoc[i]=np.where(XtoY1[i]==0)[0]

            uIndex=zLoc[0]
            for i in range(1,nDim):
                uIndex=np.intersect1d(uIndex,zLoc[i])

            validEnt=len(uIndex)
            lValid=[None for i in range(nDim)]
            for i in range(nDim):
                lValid[i]=lIndSing[i][uIndex]

            return kIndexM(lValid,1)

        self.kExpandPP1=[[[] for i in range(NKF)] for j in range(NKF)]
        self.kExpandPP2=[[[] for i in range(NKF)] for j in range(NKF)]
        self.kExpandPH=[[[] for i in range(NKF)] for j in range(NKF)]
        self.kExpandPHz=[[[] for i in range(NKF)] for j in range(NKF)]

        facXY=[None for i in range(nDim)]
        for a in range(NKF):
            for b in range(NKF):
                phaseCur=np.ones(N,dtype=np.complex_)
                lIndSE=np.zeros(N,dtype=int)
                for i in range(nDim):
                    facCur=(sPoints[i][b]-sPoints[i][a])
                    phaseCur=phaseCur*np.exp(-1j*kB[i]*facCur)
                for j in range(N):
                    for i in range(nDim):
                        facXY[i]=np.mod(lIndSing[i]+sPoints[i][a]-sPoints[i][b]-lIndSing[i][j],N1D)
                    lIndSE[j]=findInd(facXY)

                aIndSE=kIndexM(lIndSing,-1)
                self.kExpandPP1[a][b]=[aIndSE,lIndSE,phaseCur]

                phaseCur=np.ones(N,dtype=np.complex_)
                lIndSE=np.zeros(N,dtype=int)
                for i in range(nDim):
                    facCur=(sPoints[i][b]+sPoints[i][a])
                    phaseCur=phaseCur*np.exp(-1j*kB[i]*facCur)
                for j in range(N):
                    for i in range(nDim):
                        facXY[i]=np.mod(lIndSing[i]-sPoints[i][b]-lIndSing[i][j],N1D)
                    lIndSE[j]=findInd(facXY)
                aIndSE=kIndexM(lIndSing,-1)
                self.kExpandPP2[a][b]=[aIndSE,lIndSE,phaseCur]

                phaseCur=np.ones(N,dtype=np.complex_)
                lIndSE=np.zeros(N,dtype=int)
                for i in range(nDim):
                    facCur=-(sPoints[i][b]-sPoints[i][a])
                    phaseCur=phaseCur*np.exp(-1j*kB[i]*facCur)

                for j in range(N):
                    for i in range(nDim):
                        facXY[i]=np.mod(lIndSing[i]+sPoints[i][a]-sPoints[i][b]+lIndSing[i][j],N1D)
                    lIndSE[j]=findInd(facXY)
                aIndSE=kIndexM(lIndSing,1)
                self.kExpandPH[a][b]=[aIndSE,lIndSE,phaseCur]

                phaseCur=np.ones(N,dtype=np.complex_)
                for i in range(nDim):
                    phaseCur=phaseCur*np.exp(-1j*kB[i]*sPoints[i][a])

                sPropIndN=kIndexM(sPoints,1)
                sPropIndN=np.zeros(1,dtype=int)+sPropIndN[b]
                self.kExpandPHz[a][b]=[sPropIndN,phaseCur]


    def momLegSEnD(self):
        NKF=self.NKF
        N=self.N
        N1D=self.N1D
        nDim=self.nDim
        kB=self.kB
        sPoints=self.sPoints

        indeX=np.arange(N1D)
        lIndO=[None for i in range(nDim)]
        lInd=[None for i in range(nDim)]
        mInd=[None for i in range(nDim)]
        mIndr=[None for i in range(nDim)]
        nInd=[None for i in range(nDim)]
        nIndr=[None for i in range(nDim)]
        lenRed=len(sPoints[0])
        for i in range(nDim):
            lCur=np.zeros((N1D,)*nDim,dtype=int)
            for j in range(N1D):
                lCur[j]=indeX[j]
            lCur=np.reshape(np.swapaxes(lCur,0,i),N1D**nDim)
            lIndO[i]=lCur
            lCur=np.tile(lCur[:,np.newaxis,np.newaxis],(1,lenRed,lenRed))
            lInd[i]=np.reshape(lCur,N*NKF*NKF)

            mCur=np.tile(sPoints[i][:,np.newaxis],(1,lenRed))
            mIndr[i]=np.reshape(mCur,lenRed*lenRed)
            mCur=np.tile(sPoints[i][np.newaxis,:,np.newaxis],(N,1,lenRed))
            mInd[i]=np.mod(np.reshape(mCur,N*NKF*NKF),N1D)

            nCur=np.tile(sPoints[i][np.newaxis,:],(lenRed,1))
            nIndr[i]=np.reshape(nCur,lenRed*lenRed)
            nCur=np.tile(sPoints[i][np.newaxis,np.newaxis,:],(N,lenRed,1))
            nInd[i]=np.mod(np.reshape(nCur,N*NKF*NKF),N1D)

        lF=np.arange(N)*NKF*NKF
        mF=np.arange(NKF)*NKF
        nF=np.arange(NKF)
        indexFull=lF[:,None,None]+mF[None,:,None]+nF[None,None,:]
        indexFull=np.reshape(indexFull,N*NKF*NKF)

        def transCur(XtoYcur,phaseInd):
            indexC=np.where(XtoYcur[0]==0)[0]
            for i in range(1,nDim):
                indexCcur=np.where(XtoYcur[i]==0)[0]
                indexC=np.intersect1d(indexC,indexCcur)

            kPhase=np.ones((len(indexC),N),dtype=np.complex_)
            for i in range(nDim):
                kPhase=kPhase*np.exp(-1j*kB[i][None,:]*(phaseInd[i][indexC][:,None]))
            kIndex=indexFull[indexC]

            aSort=np.argsort(kIndex)
            return [kIndex[aSort],kPhase[aSort,:]]

        self.kExpandPP1=[[] for i in range(N)]
        self.kExpandPP2=[[] for i in range(N)]
        self.kExpandPH=[[] for i in range(N)]
        self.kExpandPHZ=[[] for i in range(N)]
        self.kExpandPHE=[[] for i in range(N)]
        self.kExpandPHEZ=[[] for i in range(N)]

        facXY=[None for i in range(nDim)]
        phaseXY=[None for i in range(nDim)]
        for a in range(N):
            for i in range(nDim):
                facXY[i]=np.mod(lInd[i]+mInd[i]-nInd[i]+lIndO[i][a],N1D)
                phaseXY[i]=np.mod(lInd[i],N1D)
            self.kExpandPP1[a]=transCur(facXY,phaseXY)

            for i in range(nDim):
                facXY[i]=np.mod(lInd[i]-nInd[i]+lIndO[i][a],N1D)
                phaseXY[i]=np.mod(lInd[i]+mInd[i],N1D)
            self.kExpandPP2[a]=transCur(facXY,phaseXY)

            for i in range(nDim):
                facXY[i]=np.mod(lInd[i]+mInd[i]-nInd[i]+lIndO[i][a],N1D)
                phaseXY[i]=np.mod(-lInd[i],N1D)

            self.kExpandPH[a]=transCur(facXY,phaseXY)

            for i in range(nDim):
                facXY[i]=np.mod(-nIndr[i]+lIndO[i][a],N1D)
                phaseXY[i]=np.mod(mIndr[i],N1D)
            self.kExpandPHZ[a]=transCur(facXY,phaseXY)

            for i in range(nDim):
                facXY[i]=np.mod(lInd[i]+mInd[i]-nInd[i]+lIndO[i][a],N1D)
                phaseXY[i]=np.mod(-lInd[i],N1D)
            self.kExpandPHE[a]=transCur(facXY,phaseXY)

            for i in range(nDim):
                facXY[i]=np.mod(-nIndr[i]+lIndO[i][a],N1D)
                phaseXY[i]=np.mod(mIndr[i],N1D)
            self.kExpandPHEZ[a]=transCur(facXY,phaseXY)
