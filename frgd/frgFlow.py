import copy
import multiprocessing as mp
import numpy as np
import frgd.auxFunctions as auxF
import frgd.timeFunctions as timeF
import frgd.frgEquations as frgE
from frgd.vertexF import vertexR
from frgd.propGF import scalePropF
import time

class fRG2D:
    """
    A class for solving the fRG equations via a standard N order
    adaptive step Runge-Kutta solver.

    Attributes
    ----------
    nPatches : int
        A number of logarthmically spacedfrequencies we retain
        from the full matsubara set

    deltal : float
        Initial step size for the RK solver

    beTa : float
        Initial temperature of the system

    maxW : float
        The maximum frequency of interest

    NT : int
        The number of basis functions for the vertex

    cutoffR : str
        The choice of cutoff for the regulator

    wB : array_like(float, ndim=1)
        An array of logarthmically spaced matsubara
        frequecies

    lStart : float,optional
        Start of the flow (Set to 0)

    Mu : float
        Chemical potential of the system

    UnF : vertexR
        A class to contain the two particle vertex

    propG : scaleProp
        A class to contain the single particle vertex

    nLoop : int
        The number of loops in the beta function

    Methods
    -------
    initializeFlow(hybriD,uVertex,nLoop,Mu=0,lStart=0)
        Initializes the single and two-particle vertex of the
        system

    advanceRK4()
        A step of size deltal in the Runge-Kutta solver

    susFunctions()
        Calculates the susceptibilities of the system at
        the surrent scale

    adaptiveRGFlow(lMax):
        An adaptive step RK solver until RG time is lMax

    advanceRKF():
        Single Runge-Kutta step
    """
    def __init__(self, nPatches, beTa, maxW, N, NT, NK, NB, cutoffR, nDim, nLoop):
        self.step=0.2
        self.eRR=0
        self.beTa=beTa
        self.NW=NT
        self.NK=NK
        self.NB=NB
        self.N=N
        self.nDim=nDim
        self.nPatches=nPatches
        self.nL=nLoop
        self.l=0

        if nLoop is 1:
            self.betaFC=frgE.betaF
        elif nLoop is 2:
            self.betaFC=frgE.betaF2L
        elif (nLoop>=3):
            self.betaFC=frgE.betaFNL

        self.maxW=maxW
        self.cutoffT=cutoffR
        self.modelParameters={}

    def initSinglePartVert(self,disperMat,xFill):
        self.modelParameters['disperMat']=disperMat
        self.modelParameters['xFill']=xFill

        eV,uFor=np.linalg.eigh(disperMat)
        nB=disperMat.shape[1]
        self.nB=nB

        for j in range(nB):
            for i in range(uFor.shape[0]-1):
                uSignC=np.sign(np.sum(uFor[0,:,j]*uFor[0,:,j]))
                uSign=np.sign(np.sum(uFor[0,:,j]*uFor[i+1,:,j]))

                if uSign.real!=uSignC.real and np.abs(uSign.real)>0:
                    uFor[i+1,:,j]=-uFor[i+1,:,j]

        uBack=np.linalg.inv(uFor)

        self.uFor=uFor
        self.uBack=uBack

        N1D=self.N
        def disP(wQ,kS,nB):
            kSind=auxF.indexOfMomenta(kS,N1D)
            eK=eV[kSind,nB]

            return eK[None,:]+0*wQ[:,None]

        self.propG=scalePropF(self.maxW,self.nPatches,disP,self.beTa,xFill,self.N,self.nB,self.nDim,cutoff=self.cutoffT)

    def initSinglePartVertPA(self,disperMat,xFillC,cfHopping,xFillF):
        self.modelParameters['disperMat']=disperMat
        self.modelParameters['xFillC']=xFillC
        self.modelParameters['cfHopping']=cfHopping
        self.modelParameters['xFillF']=xFillF

        nB=disperMat.shape[1]
        self.nB=nB
        N=self.N**self.nDim
        self.uBack=np.zeros((N,nB,nB))+1
        self.uFor=np.zeros((N,nB,nB))+1
        muC=auxF.setMu(xFillC,self.beTa,self.maxW,disperMat[:,:,0])
        N1D=self.N
        def disP(wQ,kS,nB):
            kSind=auxF.indexOfMomenta(kS,N1D)
            eK=disperMat[kSind,0,0]
            vK=cfHopping[kSind,0,0]

            return (vK**2)[None,:]/(1j*wQ[:,None]-eK[None,:]+muC)

        self.propG=scalePropF(self.maxW,self.nPatches,disP,self.beTa,xFillF,self.N,self.nB,self.nDim,cutoff=self.cutoffT)

    def initSinglePartVertDPA(self,disperMatC,xFillC,cfHopping,disperMatF,xFillF):
        self.modelParameters['disperMatF']=disperMatF
        self.modelParameters['disperMatC']=disperMatC
        self.modelParameters['xFillC']=xFillC
        self.modelParameters['cfHopping']=cfHopping
        self.modelParameters['xFillF']=xFillF

        nB=disperMatF.shape[1]
        self.nB=nB
        N=self.N**self.nDim
        self.uBack=np.zeros((N,nB,nB))+1
        self.uFor=np.zeros((N,nB,nB))+1
        muC=auxF.setMu(xFillC,self.beTa,self.maxW,disperMatC[:,:,0])
        N1D=self.N
        def disP(wQ,kS,nB):
            kSind=auxF.indexOfMomenta(kS,N1D)
            eK=disperMatC[kSind,0,0]
            vK=cfHopping[kSind,0,0]
            eKF=disperMatF[kSind,0,0]

            return eKF[None,:]+(vK**2)[None,:]/(1j*wQ[:,None]-eK[None,:]+muC)

        self.propG=scalePropF(self.maxW,self.nPatches,disP,self.beTa,xFillF,self.N,self.nB,self.nDim,cutoff=self.cutoffT)

    def initTwoPartVert(self,uintVals):
        self.modelParameters['interactions']=uintVals
        self.UnF=vertexR(self.maxW,self.nPatches,self.N,self.nDim,self.nB,self.NW,self.NK,self.NB,self.beTa)

        NW=self.NW
        N=self.N
        nB=self.nB
        uBack=self.uBack
        uFor=self.uFor

        def uVertex(kPP,kPH,kPHE,bInd):
            kShape=kPP[0].shape
            nDim=len(kPP)
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

            k1ind=auxF.indexOfMomenta(k1,N)
            k2ind=auxF.indexOfMomenta(k2,N)
            k3ind=auxF.indexOfMomenta(k3,N)
            k4ind=auxF.indexOfMomenta(k4,N)

            bandInd=auxF.genBandIndicies(bInd,nB)
            uV=np.zeros(k1[0].shape,dtype=np.complex_)
            for i in range(nB):
                bandMap=uBack[k1ind,bandInd[0],i]*uBack[k2ind,bandInd[1],i]*\
                    uFor[k3ind,i,bandInd[2]]*uFor[k4ind,i,bandInd[3]]
                uV+=bandMap*uintVals[0][i,i]
                for j in range(nDim):
                    uV+=bandMap*2*uintVals[1][i,i]*np.cos(k3[j]-k2[j])

            for i in range(nB):
                for j in range(nB):
                    if i!=j:
                        bandMap=uBack[k1ind,bandInd[0],i]*uBack[k2ind,bandInd[1],j]*\
                            uFor[k3ind,j,bandInd[2]]*uFor[k4ind,i,bandInd[3]]
                        uV+=bandMap*(uintVals[0][i,j]+2*uintVals[1][i,j]*(np.cos(kPHx)+\
                            np.cos(kPHy)))

                        bandMap=uBack[k1ind,bandInd[0],i]*uBack[k2ind,bandInd[1],j]*\
                            uFor[k3ind,i,bandInd[2]]*uFor[k4ind,j,bandInd[3]]
                        uV+=bandMap*uintVals[2][i,j]

                        bandMap=uBack[k1ind,bandInd[0],i]*uBack[k2ind,bandInd[1],i]*\
                            uFor[k3ind,j,bandInd[2]]*uFor[k4ind,j,bandInd[3]]
                        uV+=bandMap*uintVals[2][i,j]
            return np.reshape(uV,kShape)
        self.UnF.initializeVertex(uVertex)
        self.UnF.initializePhonons()

    def addPhonons(self,gEph,pDisper):
        self.modelParameters['pDisper']=pDisper

        N=self.N
        def uVertexPh(wPH,kPP,kPH,kPHE):
            kShape=kPP[0].shape
            nDim=len(kPP)
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

            k1ind=auxF.indexOfMomenta(k1,N)
            k2ind=auxF.indexOfMomenta(k2,N)
            k3ind=auxF.indexOfMomenta(k3,N)
            k4ind=auxF.indexOfMomenta(k4,N)

            kPHp=[np.zeros(kSize) for i in range(nDim)]
            kPHn=[np.zeros(kSize) for i in range(nDim)]
            for i in range(nDim):
                kPHp[i]=k3[i]-k2[i]
                kPHn[i]=-k3[i]+k2[i]
            kPHpInd=auxF.indexOfMomenta(kPHp,N)
            kPHnInd=auxF.indexOfMomenta(kPHn,N)

            wQ=pDisper[kPHpInd]
            ephCoupL=gEph[k1ind,kPHpInd]
            ephCoupR=gEph[k2ind,kPHnInd]

            dQ=1/(wPH**2+wQ**2)
            vertexW=np.reshape(dQ*ephCoupL*ephCouplR,kShape)

            return vertexW
        self.UnF.initializePhonons(gEph,pDisper)

    def multiStepRGFlow(self,lMax):
        nMax=int(lMax/self.step)+2

        while self.l<lMax and self.step>0.01:
            sEf,UnXf,eRR=self.advanceMS4()
            sE=sEf
            UnPP,UnPH,UnPHE=UnXf

            #print(self.l,eRR,self.step,sE[0][0][:,0].imag)
            #print 'errpr',((0.01/eRR)**0.2)*0.35
            maxVert=max([np.abs(UnPP).max(),np.abs(UnPH).max(),\
                np.abs(UnPHE).max()])

            #print('Error:',eRR,absERR)
            newStep=((0.1/eRR)**0.5)*self.step
            self.eRR=np.append(self.eRR,eRR)
            if eRR>1:
                break
            else:
                self.l+=self.step
                self.propG.sE=sE

                self.UnF.UnPP=UnPP
                self.UnF.UnPH=UnPH
                self.UnF.UnPHE=UnPHE

                self.propG.setMu()
                self.step=0.5*newStep


        return self.UnF.UnPP,self.UnF.UnPH,self.UnF.UnPHE,self.propG.sE

    def advanceMS4(self):
        UnFTemp=copy.deepcopy(self.UnF)
        propT=copy.deepcopy(self.propG)
        lC=copy.copy(self.l)

        AC=auxF.aScale(lC,self.maxW)
        sERK=[[] for i in range(5)]

        UnPPRK=[[] for i in range(5)]
        UnPHRK=[[] for i in range(5)]
        UnPHERK=[[] for i in range(5)]

        indPPRK=[[] for i in range(5)]
        indPHRK=[[] for i in range(5)]
        indPHERK=[[] for i in range(5)]

        totLen=min([10**7,self.UnF.UnPP.size])
        indLen=int(totLen/float(np.prod(self.UnF.UnPP.shape[:2])))

        NW=self.UnF.NW
        NKF=self.UnF.NKF
        NB=self.UnF.NB
        nB=self.UnF.nB

        nShape=NB*NW*NKF

        propT.setMu()
        sEr,UnXr=self.betaFC(UnFTemp,propT,AC,self.nL)
        sERK[0]=sEr
        UnPPRK[0],indPPRK[0]=auxF.reduceVertex(UnXr[0],indLen)
        UnPHRK[0],indPHRK[0]=auxF.reduceVertex(UnXr[1],indLen)
        UnPHERK[0],indPHERK[0]=auxF.reduceVertex(UnXr[2],indLen)

        lC=self.l+self.step
        AC=auxF.aScale(lC,self.maxW)
        propT.sE=copy.deepcopy(self.propG.sE)
        propT.sE+=self.step*sEr

        UnFTemp.UnPP=copy.deepcopy(self.UnF.UnPP)
        UnFTemp.UnPP+=self.step*auxF.expandVertex(UnPPRK[0],indPPRK[0],nShape)

        UnFTemp.UnPH=copy.deepcopy(self.UnF.UnPH)
        UnFTemp.UnPH+=self.step*auxF.expandVertex(UnPHRK[0],indPHRK[0],nShape)

        UnFTemp.UnPHE=copy.deepcopy(self.UnF.UnPHE)
        UnFTemp.UnPHE+=self.step*auxF.expandVertex(UnPHERK[0],indPHERK[0],nShape)

        propT.setMu()
        sEr,UnXr=self.betaFC(UnFTemp,propT,AC,self.nL)

        propT.sE=copy.deepcopy(self.propG.sE)
        propT.sE+=0.5*self.step*(sERK[0]+sEr)

        UnFTemp.UnPP=copy.deepcopy(self.UnF.UnPP)
        UnFTemp.UnPP+=0.5*self.step*(auxF.expandVertex(UnPPRK[0],indPPRK[0],nShape)+UnXr[0])

        UnFTemp.UnPH=copy.deepcopy(self.UnF.UnPH)
        UnFTemp.UnPH+=0.5*self.step*(auxF.expandVertex(UnPHRK[0],indPHRK[0],nShape)+UnXr[1])

        UnFTemp.UnPHE=copy.deepcopy(self.UnF.UnPHE)
        UnFTemp.UnPHE+=0.5*self.step*(auxF.expandVertex(UnPHERK[0],indPHERK[0],nShape)+UnXr[2])

        propT.setMu()
        sEr,UnXr=self.betaFC(UnFTemp,propT,AC,self.nL)

        sE=copy.deepcopy(self.propG.sE)
        UnPP=copy.deepcopy(self.UnF.UnPP)
        UnPH=copy.deepcopy(self.UnF.UnPH)
        UnPHE=copy.deepcopy(self.UnF.UnPHE)

        sE+=0.5*self.step*(sERK[0]+sEr)
        sEerr=np.abs(sE-propT.sE).max()
        UnPP+=0.5*self.step*(auxF.expandVertex(UnPPRK[0],indPPRK[0],nShape)+UnXr[0])
        UnPPerr=np.abs(UnPP-UnFTemp.UnPP).max()
        UnPH+=0.5*self.step*(auxF.expandVertex(UnPHRK[0],indPHRK[0],nShape)+UnXr[1])
        UnPHerr=np.abs(UnPH-UnFTemp.UnPH).max()
        UnPHE+=0.5*self.step*(auxF.expandVertex(UnPHERK[0],indPHERK[0],nShape)+UnXr[2])
        UnPHEerr=np.abs(UnPHE-UnFTemp.UnPHE).max()
        UnXf=(UnPP,UnPH,UnPHE)
        eRR=max([UnPPerr,UnPHerr,UnPHEerr,sEerr])
        return sE,UnXf,eRR

    def susFunctions(self,lM):
        nDim=self.nDim
        kBX=self.UnF.kBX
        wB=self.UnF.wB
        kB=self.UnF.kB
        NB,NW,NKF=self.UnF.NB,self.UnF.NW,self.UnF.NKF

        nPatches=len(wB)
        kPatches=len(kB[0])
        if nDim==1:
            orderHarmonics=[np.zeros(kBX[0].shape)+1,
                np.cos(kBX[0]),np.sin(kBX[0])]

        if nDim==2:
            orderHarmonics=[np.zeros(kBX[0].shape)+1,
                np.cos(kBX[0])+np.cos(kBX[1]),
                np.cos(kBX[0])-np.cos(kBX[1]),
                np.sin(kBX[0]),np.sin(kBX[1]),
                np.sin(kBX[0])*np.sin(kBX[1])]

        if nDim==3:
            orderHarmonics=[np.zeros(kBX[0].shape)+1,
                np.cos(kBX[0])+np.cos(kBX[1])+np.cos(kBX[2]),
                np.cos(kBX[0])-np.cos(kBX[1]),
                -np.cos(kBX[0])-np.cos(kBX[1])+2*np.cos(kBX[2]),
                np.sin(kBX[0]),np.sin(kBX[1]),np.sin(kBX[2]),
                np.cos(kBX[0])*np.cos(kBX[1])+np.cos(kBX[0])*np.cos(kBX[2])+np.cos(kBX[1])*np.cos(kBX[2])]
        orderLen=len(orderHarmonics)

        AC=auxF.aScale(lM,self.maxW)
        UnPHphonon=self.UnF.conPhVert(AC)
        UnX=(self.UnF.UnPP,self.UnF.UnPH+UnPHphonon,self.UnF.UnPHE)
        UnPPX,UnPHX,UnPHEX=frgE.projectedVertex(UnX,self.UnF,AC,'b')

        UnPPX=self.UnF.UnPPO+self.UnF.UnPP+UnPPX
        UnPHX=self.UnF.UnPHO+self.UnF.UnPH+UnPHX+UnPHphonon
        UnPHEX=self.UnF.UnPHEO+self.UnF.UnPHE+UnPHEX
        UnPHSX=UnPHX-self.UnF.getPHEinPH(UnPHEX)

        wInd=np.argmin(np.abs(self.UnF.tPoints[0]))
        kInd=np.zeros(len(self.UnF.sPoints[0]))
        for i in range(nDim):
            kInd+=self.UnF.sPoints[i]**2
        kInd=np.argmin(kInd)
        UnPPXr1=auxF.linInterpO(wB,UnPPX[:,:,wInd*NKF+kInd,wInd*NKF+kInd],self.UnF.wBX,axis=0)
        UnPPXr1=timeF.ffTransformW(UnPPXr1,self.UnF.NWT,axis=0)[0,:]
        UnPHXr1=auxF.linInterpO(wB,UnPHX[:,:,wInd*NKF+kInd,wInd*NKF+kInd],self.UnF.wBX,axis=0)
        UnPHXr1=timeF.ffTransformW(UnPHXr1,self.UnF.NWT,axis=0)[0,:]
        UnPHEXr1=auxF.linInterpO(wB,UnPHEX[:,:,wInd*NKF+kInd,wInd*NKF+kInd],self.UnF.wBX,axis=0)
        UnPHEXr1=timeF.ffTransformW(UnPHEXr1,self.UnF.NWT,axis=0)[0,:]
        self.UnF.UnPHV=[UnPPXr1,UnPHXr1,UnPHEXr1]

        UnPPXr=np.sum(np.reshape(UnPPX,(nPatches,kPatches,NW,NKF,NW,NKF)),axis=(2,4))[19,:,kInd,kInd]
        UnPHXr=np.sum(np.reshape(UnPHX,(nPatches,kPatches,NW,NKF,NW,NKF)),axis=(2,4))[19,:,kInd,kInd]
        UnPHEXr=np.sum(np.reshape(UnPHEX,(nPatches,kPatches,NW,NKF,NW,NKF)),axis=(2,4))[19,:,kInd,kInd]
        UnPPXr2=np.sum(np.reshape(UnPPX,(nPatches,kPatches,NW,NKF,NW,NKF)),axis=(2,3,4,5))[19,:]
        UnPHXr2=np.sum(np.reshape(UnPHX,(nPatches,kPatches,NW,NKF,NW,NKF)),axis=(2,3,4,5))[19,:]
        UnPHEXr2=np.sum(np.reshape(UnPHEX,(nPatches,kPatches,NW,NKF,NW,NKF)),axis=(2,3,4,5))[19,:]
        self.UnF.UnPHLV=[UnPPXr,UnPHXr,UnPHEXr,UnPPXr2,UnPHXr2,UnPHEXr2]

        nB=self.nB
        N=self.UnF.N
        uCharge=np.reshape(UnPHX+UnPHSX,\
            (nPatches,kPatches,NB,NW,NKF,NB,NW,NKF))
        uSpin=np.reshape(-UnPHEX,(nPatches,kPatches,NB,NW,NKF,NB,NW,NKF))
        uSU=np.reshape(UnPPX,(nPatches,kPatches,NB,NW,NKF,NB,NW,NKF))


        def matMulRed(gL,uC,gR):
            return np.squeeze(np.matmul(gL,np.matmul(uC,gR)))


        def calcSus(gL,uXf,gR,typE):
            uXn=[[[] for i in range(nB)] for j in range(nB)]
            for i in range(nB):
                for j in range(nB):
                    uXn[i][j]=calcSusBand(gL[i],uXf,gR[j],typE)

            return uXn
        def calcSusR(gL,uXf,gR,typE,totBands):
            uXn=[[[] for i in range(nB)] for j in range(nB)]
            uXn2=[[[] for i in range(nB)] for j in range(nB)]
            uXn3=[[[] for i in range(nB)] for j in range(nB)]
            wMin=np.argmin(np.abs(wB))

            beTa=self.UnF.beTa
            N1D=self.UnF.N1D
            nPatches=len(wB)
            wBX=(2*np.pi/beTa)*np.arange(512+1)
            wBX=np.append(-wBX[-2:0:-1],wBX)

            staticSus=np.zeros((nB,nB,N1D**nDim),dtype=np.complex_)
            dynamicSusMax=np.zeros((nB,nB,nPatches),dtype=np.complex_)
            maxMomIndex=np.zeros((nB,nB),dtype=int)
            strucFactor=np.zeros((nB,nB,N1D**nDim),dtype=np.complex_)
            for i in range(nB):
                for j in range(nB):
                    uXnCur=calcSusBand(gL[i],uXf,gR[j],typE,totBands)
                    kLoc=np.argmax(np.abs(uXnCur[wMin,:]))
                    staticSus[i,j,:]=uXnCur[wMin,:]
                    dynamicSusMax[i,j,:]=uXnCur[:,kLoc]
                    maxMomIndex[i,j]=kLoc

                    xSusI=auxF.linInterpO(wB,uXnCur,wBX)
                    strucFactor[i,j,:]=(1.0/beTa)*np.sum(xSusI,axis=0)

            susOut=[staticSus,dynamicSusMax,maxMomIndex,strucFactor]
            return susOut

        def calcSusBand(gXL,uFull,gXR,typE,bandIndex):
            susCur=np.zeros((nPatches,kPatches),dtype=np.complex_)
            NB=self.UnF.NB
            nB=self.UnF.nB
            NW=self.UnF.NW
            NKF=self.UnF.NKF
            if typE is 'chargeO':
                uLeft=self.UnF.uLeftPH
                uRight=self.UnF.uRightPH
                for i in range(nB):
                    for j in range(nB):
                        for k in range(nB):
                            for l in range(nB):
                                bandCurInd=[i,j,k,l]
                                bLabel=auxF.genBandLabel(bandCurInd,nB)
                                uFullc=np.zeros(uFull.shape[:2]+(NW*NKF,NW*NKF,),dtype=np.complex_)
                                indL,indR=auxF.indXY(bandIndex,nB**2)
                                for m in range(NB):
                                    for n in range(NB):
                                        uMap=uLeft[:,:,:,indL[bLabel],m]*uRight[:,:,:,n,indR[bLabel]]
                                        uFullc+=np.reshape(uMap[None,:,None,:,None,:]*uFull[:,:,m,:,:,n,:,:]\
                                            ,(nPatches,kPatches,NW*NKF,NW*NKF))
                                susCur+=2*matMulRed(gXL[...,i,l],uFullc,gXR[...,j,k])
            elif typE is 'spinO':
                uLeft=self.UnF.uLeftPHE
                uRight=self.UnF.uRightPHE
                for i in range(nB):
                    for j in range(nB):
                        for k in range(nB):
                            for l in range(nB):
                                bandCurInd=[j,i,k,l]
                                bLabel=auxF.genBandLabel(bandCurInd,nB)
                                uFullc=np.zeros(uFull.shape[:2]+(NW*NKF,NW*NKF,),dtype=np.complex_)
                                indL,indR=auxF.indXY(bandIndex,nB**2)
                                for m in range(NB):
                                    for n in range(NB):
                                        uMap=uLeft[:,:,:,indL[bLabel],m]*uRight[:,:,:,n,indR[bLabel]]
                                        uFullc+=np.reshape(uMap[None,:,None,:,None,:]*uFull[:,:,m,:,:,n,:,:]\
                                            ,(nPatches,kPatches,NW*NKF,NW*NKF))
                                susCur+=2*matMulRed(gXL[...,i,l],uFullc,gXR[...,j,k])
            elif typE is 'pairO':
                uLeft=self.UnF.uLeftPP
                uRight=self.UnF.uRightPP
                for i in range(nB):
                    for j in range(nB):
                        for k in range(nB):
                            for l in range(nB):
                                bandCurInd=[i,j,k,l]
                                bLabel=auxF.genBandLabel(bandCurInd,nB)
                                uFullc=np.zeros(uFull.shape[:2]+(NW*NKF,NW*NKF,),dtype=np.complex_)
                                indL,indR=auxF.indXY(bandIndex,nB**2)
                                for m in range(NB):
                                    for n in range(NB):
                                        uMap=uLeft[:,:,:,indL[bLabel],m]*uRight[:,:,:,n,indR[bLabel]]
                                        uFullc+=np.reshape(uMap[None,:,None,:,None,:]*uFull[:,:,m,:,:,n,:,:]\
                                            ,(nPatches,kPatches,NW*NKF,NW*NKF))

                                susCur+=2*matMulRed(gXL[...,i,j],uFullc,gXR[...,k,l])
            return susCur

        staticSusF=np.zeros((3,orderLen,nB,nB,N),dtype=np.complex_)
        dynamicSusF=np.zeros((3,orderLen,nB,nB,nPatches),dtype=np.complex_)
        maxMomIndF=np.zeros((3,orderLen,nB,nB),dtype=int)
        strucFacF=np.zeros((3,orderLen,nB,nB,N),dtype=np.complex_)

        tPoints=self.UnF.tPoints
        sPoints=self.UnF.sPoints
        uFor=self.uFor
        uBack=self.uBack

        vertX=[uCharge,uSpin,uSU]
        indVertX=[self.UnF.phIndex,self.UnF.pheIndex,self.UnF.ppIndex]
        channelS=['chargeO','spinO','pairO']
        for i in range(orderLen):
            gPHL,gPHR,gPPL,gPPR=self.propG.susBubblesMB(wB,kB,tPoints,sPoints,\
                orderHarmonics[i],uFor,uBack)
            gL=[gPHL,gPHL,gPPL]
            gR=[gPHR,gPHR,gPPR]
            for j in range(len(channelS)):
                curSus=calcSusR(gL[j],vertX[j],gR[j],channelS[j],indVertX[j])

                staticSusF[j,i,...]=curSus[0]
                dynamicSusF[j,i,...]=curSus[1]
                maxMomIndF[j,i,...]=curSus[2]
                strucFacF[j,i,...]=curSus[3]

        return staticSusF,dynamicSusF,strucFacF,maxMomIndF

    def adaptiveRGFlow(self,lMax):
        nMax=int(lMax/self.step)+1
        nMax=80

        iCount=0
        while self.l<lMax and self.step>0.01:
            sEf,UnXf,gapX,phX,eRR=self.advanceRKF()
            sE=sEf
            UnPP,UnPH,UnPHE=UnXf
            suGap,chGap,spGap=gapX
            phSE,gVert=phX
            ACc=auxF.aScale(self.l,self.maxW)

            fA=auxF.fScaling(ACc)
            tPointsC=(self.UnF.tPoints[0]/fA,self.UnF.tPoints[1]/fA)
            suGapF,chGapF,spGapF,kSU,kCH,kSP=auxF.calcMaxGap(gapX,self.UnF.wB,self.UnF.kB,\
                tPointsC,self.UnF.sPoints)

            maxGap=max([np.abs(chGapF).min(),np.abs(spGapF).min(),\
                np.abs(suGapF).min()])

            absERR=0.05
            stepN=self.step*(absERR/eRR)**0.2

            self.eRR=np.append(self.eRR,eRR)
            if maxGap<ACc and eRR<absERR:
                self.l+=self.step
                self.propG.sE=sE

                self.UnF.UnPP=UnPP
                self.UnF.UnPH=UnPH
                self.UnF.UnPHE=UnPHE

                self.UnF.phononSE=phSE
                self.UnF.gVert=gVert

                self.UnF.suGap=suGap
                self.UnF.chGap=chGap
                self.UnF.spGap=spGap
                iCount=iCount+1

                self.step=0.5*stepN
                self.propG.setMu()

            else:
                self.step=0.5*self.step

        return self.UnF.UnPP,self.UnF.UnPH,self.UnF.UnPHE,self.propG.sE

    def advanceRKF(self):
        UnFTemp=copy.deepcopy(self.UnF)
        propT=copy.deepcopy(self.propG)
        lC=copy.copy(self.l)

        AC=auxF.aScale(lC,self.maxW)
        sERK=[[] for i in range(6)]

        UnPPRK=[[] for i in range(6)]
        UnPHRK=[[] for i in range(6)]
        UnPHERK=[[] for i in range(6)]

        indPPRK=[[] for i in range(6)]
        indPHRK=[[] for i in range(6)]
        indPHERK=[[] for i in range(6)]

        suGapRK=[[] for i in range(6)]
        chGapRK=[[] for i in range(6)]
        spGapRK=[[] for i in range(6)]

        phSERK=[[] for i in range(6)]
        gVertRK=[[] for i in range(6)]

        totLen=min([10**7,self.UnF.UnPP.size])
        indLen=int(totLen/float(np.prod(self.UnF.UnPP.shape[:2])))

        propT.setMu()
        sEr,UnXr,gapX,phononX=self.betaFC(UnFTemp,propT,AC,self.nL)

        sERK[0]=sEr
        UnPPRK[0],indPPRK[0]=auxF.reduceVertex(UnXr[0],indLen)
        UnPHRK[0],indPHRK[0]=auxF.reduceVertex(UnXr[1],indLen)
        UnPHERK[0],indPHERK[0]=auxF.reduceVertex(UnXr[2],indLen)
        suGapRK[0],chGapRK[0],spGapRK[0]=gapX
        phSERK[0],gVertRK[0]=phononX

        stepB=np.zeros(5)
        stepB[:]=[1.0/4,3.0/8,12.0/13,1.0,1.0/2]

        stepB=stepB*self.step
        bTable=np.zeros(15)
        bTable[:]=[1.0/4,3.0/32,9.0/32,1932.0/2197,-7200.0/2197,
                   7296.0/2197,439.0/216,-8.0,3680.0/513,-845.9/4104,
                   -8.0/27,2.0,-3544.0/2565,1859.0/4104,-11.0/40]

        NW=self.UnF.NW
        NKF=self.UnF.NKF
        NB=self.UnF.NB
        nB=self.UnF.nB

        nShape=NB*NW*NKF
        for i in range(5):
            lC=self.l+stepB[i]
            propT.sE=copy.deepcopy(self.propG.sE)

            UnFTemp.UnPP=copy.deepcopy(self.UnF.UnPP)
            UnFTemp.UnPH=copy.deepcopy(self.UnF.UnPH)
            UnFTemp.UnPHE=copy.deepcopy(self.UnF.UnPHE)

            UnFTemp.suGap=copy.deepcopy(self.UnF.suGap)
            UnFTemp.chGap=copy.deepcopy(self.UnF.chGap)
            UnFTemp.spGap=copy.deepcopy(self.UnF.spGap)

            UnFTemp.gVert=copy.deepcopy(self.UnF.gVert)
            UnFTemp.phononSE=copy.deepcopy(self.UnF.phononSE)
            for j in range(i+1):
                bWeight=bTable[sum(range(i+1))+j]

                propT.sE+=stepB[i]*bWeight*sERK[j]

                UnFTemp.phononSE+=(stepB[i]*bWeight)*phSERK[j]
                UnFTemp.gVert+=(stepB[i]*bWeight)*gVertRK[j]

                UnFTemp.suGap+=(stepB[i]*bWeight)*suGapRK[j]
                UnFTemp.chGap+=(stepB[i]*bWeight)*chGapRK[j]
                UnFTemp.spGap+=(stepB[i]*bWeight)*spGapRK[j]

                UnFTemp.UnPP+=auxF.expandVertex(UnPPRK[j],indPPRK[j],nShape)*\
                    (stepB[i]*bWeight)
                UnFTemp.UnPH+=auxF.expandVertex(UnPHRK[j],indPHRK[j],nShape)*\
                    (stepB[i]*bWeight)
                UnFTemp.UnPHE+=auxF.expandVertex(UnPHERK[j],indPHERK[j],nShape)*\
                    (stepB[i]*bWeight)


            AC=auxF.aScale(lC,self.maxW)
            propT.setMu()
            sEr,UnXr,gapX,phononX=self.betaFC(UnFTemp,propT,AC,self.nL)
            sERK[i+1]=sEr
            phSERK[i+1],gVertRK[i+1]=phononX
            suGapRK[i+1],chGapRK[i+1],spGapRK[i+1]=gapX
            UnPPRK[i+1],indPPRK[i+1]=auxF.reduceVertex(UnXr[0],indLen)
            UnPHRK[i+1],indPHRK[i+1]=auxF.reduceVertex(UnXr[1],indLen)
            UnPHERK[i+1],indPHERK[i+1]=auxF.reduceVertex(UnXr[2],indLen)

        o5=np.zeros(6)
        o4=np.zeros(6)
        o5[:]=[16.0/135.0,0.0,6656.0/12825.0,28561.0/56430.0,-9.0/50.0,2.0/55.0]
        o4[:]=[25.0/216,0.0,1408.0/2565,2197.0/4104,-1.0/5,0.0]

        o5=self.step*o5
        o4=self.step*o4

        sE=copy.deepcopy(self.propG.sE)

        gVert=copy.deepcopy(self.UnF.gVert)
        phononSE=copy.deepcopy(self.UnF.phononSE)

        suGap=copy.deepcopy(self.UnF.suGap)
        chGap=copy.deepcopy(self.UnF.chGap)
        spGap=copy.deepcopy(self.UnF.spGap)

        UnPP=copy.deepcopy(self.UnF.UnPP)
        UnPH=copy.deepcopy(self.UnF.UnPH)
        UnPHE=copy.deepcopy(self.UnF.UnPHE)

        pSEerr=self.UnF.phononSE*0
        gVerterr=self.UnF.gVert*0
        sEerr=self.propG.sE*0
        UnPPerr=self.UnF.UnPP*0
        UnPHerr=self.UnF.UnPH*0
        UnPHEerr=self.UnF.UnPHE*0

        for i in range(6):
            sE+=o5[i]*sERK[i]
            sEerr+=(o5[i]-o4[i])*sERK[i]

            phononSE+=o5[i]*phSERK[i]
            pSEerr+=(o5[i]-o4[i])*phSERK[i]

            gVert+=o5[i]*gVertRK[i]
            gVerterr+=(o5[i]-o4[i])*gVertRK[i]

            suGap+=o5[i]*suGapRK[i]
            chGap+=o5[i]*chGapRK[i]
            spGap+=o5[i]*spGapRK[i]

            UnPP+=auxF.expandVertex(UnPPRK[i],indPPRK[i],nShape)*o5[i]
            UnPPerr+=auxF.expandVertex(UnPPRK[i],indPPRK[i],nShape)*(o5[i]-o4[i])

            UnPH+=auxF.expandVertex(UnPHRK[i],indPHRK[i],nShape)*o5[i]
            UnPHerr+=auxF.expandVertex(UnPHRK[i],indPHRK[i],nShape)*(o5[i]-o4[i])

            UnPHE+=auxF.expandVertex(UnPHERK[i],indPHERK[i],nShape)*o5[i]
            UnPHEerr+=auxF.expandVertex(UnPHERK[i],indPHERK[i],nShape)*(o5[i]-o4[i])

        pSEerrMax=np.abs(pSEerr).max()
        sEerrMax=np.abs(sEerr).max()
        gVerterrMax=np.abs(gVerterr).max()
        UnPPerrMax=np.abs(UnPPerr).max()
        UnPHerrMax=np.abs(UnPHerr).max()
        UnPHEerrMax=np.abs(UnPHEerr).max()

        eRR=max([sEerrMax,pSEerrMax,gVerterrMax,UnPPerrMax,UnPHerrMax,UnPHEerrMax])
        sEf=sE
        UnXf=(UnPP,UnPH,UnPHE)
        gapX=(suGap,chGap,spGap)
        phX=(phononSE,gVert)

        return sEf,UnXf,gapX,phX,eRR
