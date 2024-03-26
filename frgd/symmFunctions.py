import numpy as np
import frgd.auxFunctions as auxF

class symmV:
    def __init__(self,wB,kB,NW,NLmax,NKF,N1D,nDim,sPoints,tPoints,tPointsL):
        self.wB=wB
        self.kB=kB

        self.sPoints=sPoints
        self.tPoints=tPoints

        self.NW=NW
        self.NLmax=NLmax
        self.NKF=NKF
        self.N1D=N1D
        self.nDim=nDim
        self.N=N1D**nDim

        self.tPointsL=tPointsL

        self.calcPositivity('PP')
        self.calcPositivity('PH')
        self.calcPositivity('PHE')

        self.calcPositivityFull('PP')
        self.calcPositivityFull('PH')
        self.calcPositivityFull('PHE')

        self.calcExchange('PP')
        self.calcExchange('PH')
        self.calcExchange('PHE')

        self.calcExchangeFull('PP')
        self.calcExchangeFull('PH')
        self.calcExchangeFull('PHE')

    def applyExchangeFull(self,UnCur,chnL):
        if chnL is 'PP':
            indN=self.indNEFPP
        else:
            indN=self.indNEFPHX

        aSortW=np.argsort(indN[0])
        aSortK=np.argsort(indN[1])

        return UnCur[np.ix_(aSortW,aSortK)]
    def applyExchange(self,UnCur,chnL):
        if chnL is 'PP':
            indO=self.indOEPP
            phaseC=self.phaseEPP

        elif chnL is 'PH':
            indO=self.indOEPHX
            phaseC=self.phaseEPHX

        elif chnL is 'PHE':
            indO=self.indOEPHX
            phaseC=self.phaseEPHX

        NW=self.NW
        NKF=self.NKF
        uN=np.reshape(UnCur,UnCur.shape[:2]+(NW*NKF*NW*NKF,))
        uN=uN[np.ix_(indO[0],indO[1],indO[2])]
        uN=phaseC[None,...]*uN
        uN=np.reshape(uN,uN.shape[:2]+(NW*NKF,NW*NKF,))

        return uN
    def applyPositivityFull(self,UnCur,chnL):
        if chnL is 'PP':
            indN=self.indNPFPX
        elif chnL is 'PH':
            indN=self.indNPFPH
        elif chnL is 'PHE':
            indN=self.indNPFPX

        aSortW=np.argsort(indN[0])
        aSortK=np.argsort(indN[1])

        return np.conj(UnCur)[np.ix_(aSortW,aSortK)]

    def applyPositivity(self,UnCur,chnL):
        if chnL is 'PP':
            indO=self.indOPPX
            phaseC=self.phasePPX

        elif chnL is 'PH':
            indO=self.indOPPH
            phaseC=self.phasePPH

        elif chnL is 'PHE':
            indO=self.indOPPX
            phaseC=self.phasePPX

        NW=self.NW
        NKF=self.NKF

        uN=np.reshape(UnCur,UnCur.shape[:2]+(NW*NKF*NW*NKF,))
        uN=uN[np.ix_(indO[0],indO[1],indO[2])]
        uN=phaseC[None,...]*np.conj(uN)
        uN=np.reshape(uN,uN.shape[:2]+(NW*NKF,NW*NKF,))

        return uN
    def trueInd(self,sList,N1D,nDim):
        tInd=np.zeros(len(sList[0]),dtype=int)
        for i in range(nDim):
            tInd+=np.mod(sList[i],N1D)*(N1D**(nDim-1-i))
        return tInd
    def calcPositivity(self,chnL):
        sI=self.sPoints
        tI=self.tPoints
        NW=self.NW
        NKF=self.NKF
        N1D=self.N1D
        nDim=self.nDim
        kB=self.kB
        wB=self.wB

        aInd=np.arange(NKF*NKF)
        indKL,indKR=auxF.indXY(aInd,NKF)
        aInd=np.arange(NW*NW)
        indWL,indWR=auxF.indXY(aInd,NW)

        indKLN=[None for i in range(nDim)]
        indKRN=[None for i in range(nDim)]
        for i in range(nDim):
            indKLN[i]=sI[i][indKL]
            indKRN[i]=sI[i][indKR]

        indWLN=tI[0][indWL]
        indWRN=tI[0][indWR]
        pWrange=np.arange(len(wB))
        nWrange=auxF.locIndex(-wB,wB)
        nkB=[None for i in range(nDim)]
        for i in range(nDim):
            nkB[i]=-kB[i]

        pKrange=np.arange(N1D**nDim)
        nKrange=auxF.indexOfMomenta(nkB,N1D)

        sIe=self.trueInd(sI,N1D,nDim)
        if chnL is 'PH':
            newindKLN=[None for i in range(nDim)]
            newindKRN=[None for i in range(nDim)]
            for i in range(nDim):
                newindKLN[i]=-indKLN[i]
                newindKRN[i]=-indKRN[i]

            newindKLNf=self.trueInd(newindKLN,N1D,nDim)
            newindKRNf=self.trueInd(newindKRN,N1D,nDim)

            indKLNi=auxF.locIndex(newindKLNf,sIe)
            indKRNi=auxF.locIndex(newindKRNf,sIe)

            newIndWLN=indWLN
            newIndWRN=indWRN
            indWLN=auxF.locIndex(newIndWLN,tI[0])
            indWRN=auxF.locIndex(newIndWRN,tI[0])

            indWLNext=np.repeat(indWLN,len(indKLNi))
            indKLNext=np.tile(indKLNi,len(indWLN))
            indWRNext=np.repeat(indWRN,len(indKRNi))
            indKRNext=np.tile(indKRNi,len(indWRN))

            indNPPH=(indWLNext*NKF+indKLNext)*NW*NKF+(indWRNext*NKF+indKRNext)

            self.indOPPH=(pWrange,nKrange,indNPPH)
            phasePPH=np.ones((N1D**nDim,len(indKLNi)),dtype=np.complex_)
            for i in range(nDim):
                phasePPH=phasePPH*\
                    np.exp(1j*kB[i][:,None]*(indKLN[i]-indKRN[i])[None,:])
            phasePPH=np.reshape(phasePPH,(N1D**nDim,NKF,NKF))
            phasePPH=np.reshape(np.tile(phasePPH[:,None,:,None,:],(1,NW,1,NW,1)),(N1D**nDim,NW*NKF*NW*NKF))
            self.phasePPH=phasePPH

        else:
            newindKLN=[None for i in range(nDim)]
            newindKRN=[None for i in range(nDim)]
            for i in range(nDim):
                newindKLN[i]=indKLN[i]
                newindKRN[i]=indKRN[i]

            newindKLNf=self.trueInd(newindKLN,N1D,nDim)
            newindKRNf=self.trueInd(newindKLN,N1D,nDim)

            indKLNi=auxF.locIndex(newindKLNf,sIe)
            indKRNi=auxF.locIndex(newindKRNf,sIe)

            newIndWLN=indWRN
            newIndWRN=indWLN
            indWLN=auxF.locIndex(newIndWLN,tI[0])
            indWRN=auxF.locIndex(newIndWRN,tI[0])

            indWLNext=np.repeat(indWLN,len(indKLNi))
            indKLNext=np.tile(indKLNi,len(indWLN))
            indWRNext=np.repeat(indWRN,len(indKRNi))
            indKRNext=np.tile(indKRNi,len(indWRN))

            indNPPX=(indWLNext*NKF+indKLNext)*NW*NKF+(indWRNext*NKF+indKRNext)

            self.indOPPX=(nWrange,pKrange,indNPPX)
            phasePPX=np.ones((N1D**nDim,len(indKLNi)),dtype=np.complex_)
            for i in range(nDim):
                phasePPX=phasePPX*np.exp(0*1j*kB[i][:,None]*indKLNi[None,:])
            phasePPX=np.reshape(phasePPX,(N1D**nDim,NKF,NKF))
            phasePPX=np.reshape(np.tile(phasePPX[:,None,:,None,:],(1,NW,1,NW,1)),(N1D**nDim,NW*NKF*NW*NKF))

            self.phasePPX=phasePPX

    def calcExchange(self,chnL):
        sI=self.sPoints
        tI=self.tPoints
        NW=self.NW
        NKF=self.NKF
        N1D=self.N1D
        nDim=self.nDim
        kB=self.kB
        wB=self.wB

        aInd=np.arange(NKF*NKF)
        indKL,indKR=auxF.indXY(aInd,NKF)
        aInd=np.arange(NW*NW)
        indWL,indWR=auxF.indXY(aInd,NW)

        indKLN=[None for i in range(nDim)]
        indKRN=[None for i in range(nDim)]
        for i in range(nDim):
            indKLN[i]=sI[i][indKL]
            indKRN[i]=sI[i][indKR]

        indWLN=tI[0][indWL]
        indWRN=tI[0][indWR]
        pWrange=np.arange(len(wB))
        nWrange=auxF.locIndex(-wB,wB)
        nkB=[None for i in range(nDim)]
        for i in range(nDim):
            nkB[i]=-kB[i]

        pKrange=np.arange(N1D**nDim)
        nKrange=auxF.indexOfMomenta(nkB,N1D)

        sIe=self.trueInd(sI,N1D,nDim)

        if chnL is 'PP':
            newindKLN=[None for i in range(nDim)]
            newindKRN=[None for i in range(nDim)]
            for i in range(nDim):
                newindKLN[i]=-indKLN[i]
                newindKRN[i]=-indKRN[i]

            newindKLNf=self.trueInd(newindKLN,N1D,nDim)
            newindKRNf=self.trueInd(newindKRN,N1D,nDim)

            indKLNi=auxF.locIndex(newindKLNf,sIe)
            indKRNi=auxF.locIndex(newindKRNf,sIe)

            newIndWLN=-indWLN
            newIndWRN=-indWRN
            indWLN=auxF.locIndex(newIndWLN,tI[0])
            indWRN=auxF.locIndex(newIndWRN,tI[0])

            indWLNext=np.repeat(indWLN,len(indKLNi))
            indKLNext=np.tile(indKLNi,len(indWLN))
            indWRNext=np.repeat(indWRN,len(indKRNi))
            indKRNext=np.tile(indKRNi,len(indWRN))

            indNEPP=(indWLNext*NKF+indKLNext)*NW*NKF+(indWRNext*NKF+indKRNext)

            self.indOEPP=(pWrange,pKrange,indNEPP)
            phaseEPP=np.ones((N1D**nDim,len(indKLNi)),dtype=np.complex_)
            for i in range(nDim):
                phaseEPP=phaseEPP*np.exp(1j*kB[i][:,None]*(indKLN[i]-indKRN[i])[None,:])
            phaseEPP=np.reshape(phaseEPP,(N1D**nDim,NKF,NKF))
            phaseEPP=np.reshape(np.tile(phaseEPP[:,None,:,None,:],(1,NW,1,NW,1)),(N1D**nDim,NW*NKF*NW*NKF))

            self.phaseEPP=phaseEPP

        else:
            newindKLN=[None for i in range(nDim)]
            newindKRN=[None for i in range(nDim)]
            for i in range(nDim):
                newindKLN[i]=-indKRN[i]
                newindKRN[i]=-indKLN[i]

            newindKLNf=self.trueInd(newindKLN,N1D,nDim)
            newindKRNf=self.trueInd(newindKRN,N1D,nDim)

            indKLNi=auxF.locIndex(newindKLNf,sIe)
            indKRNi=auxF.locIndex(newindKRNf,sIe)

            newIndWLN=indWRN
            newIndWRN=indWLN
            indWLN=auxF.locIndex(newIndWLN,tI[0])
            indWRN=auxF.locIndex(newIndWRN,tI[0])

            indWLNext=np.repeat(indWLN,len(indKLNi))
            indKLNext=np.tile(indKLNi,len(indWLN))
            indWRNext=np.repeat(indWRN,len(indKRNi))
            indKRNext=np.tile(indKRNi,len(indWRN))

            indN=(indWLNext*NKF+indKLNext)*NW*NKF+(indWRNext*NKF+indKRNext)

            phaseEPHX=np.ones((N1D**nDim,len(indKLNi)),dtype=np.complex_)
            for i in range(nDim):
                phaseEPHX=phaseEPHX*np.exp(1j*kB[i][:,None]*(indKLN[i]-indKRN[i])[None,:])
            phaseEPHX=np.reshape(phaseEPHX,(N1D**nDim,NKF,NKF))
            phaseEPHX=np.reshape(np.tile(phaseEPHX[:,None,:,None,:],(1,NW,1,NW,1)),(N1D**nDim,NW*NKF*NW*NKF))

            self.phaseEPHX=phaseEPHX
            self.indOEPHX=(nWrange,nKrange,indN)

    def calcExchangeFull(self,chnL):
        sI=self.sPoints
        tI=self.tPoints
        tIL=self.tPointsL
        NW=self.NW
        NKF=self.NKF
        N=self.N
        N1D=self.N1D
        nDim=self.nDim
        NLmax=self.NLmax

        wIndexFull=np.arange(NLmax*NW*NW)
        indWLsing,rem=auxF.indXY(wIndexFull,NW*NW)
        indWL,indWR=auxF.indXY(rem,NW)

        kIndexFull=np.arange(N*NKF*NKF)
        indKLsing,rem=auxF.indXY(kIndexFull,NKF*NKF)
        indKL,indKR=auxF.indXY(rem,NKF)

        indexX=np.arange(N1D)
        mIndex=[None for i in range(nDim)]
        for i in range(nDim):
            lCur=np.zeros((N1D,)*nDim,dtype=int)
            for j in range(N1D):
                lCur[j]=indexX[j]
            lCur=np.reshape(np.swapaxes(lCur,0,i),N1D**nDim)
            mIndex[i]=lCur

        indKLsingN=[None for i in range(nDim)]
        indKLN=[None for i in range(nDim)]
        indKRN=[None for i in range(nDim)]
        for i in range(nDim):
            indKLsingN[i]=mIndex[i][indKLsing]
            indKLN[i]=sI[i][indKL]
            indKRN[i]=sI[i][indKR]

        indWLN=tI[0][indWL]
        indWRN=tI[0][indWR]
        indWLs=tIL[0][indWLsing]
        sIe=self.trueInd(sI,N1D,nDim)
        mIndexE=self.trueInd(mIndex,N1D,nDim)
        if chnL is 'PP':
            newindKLN=[None for i in range(nDim)]
            newindKRN=[None for i in range(nDim)]
            newindKsing=[None for i in range(nDim)]
            for i in range(nDim):
                newindKLN[i]=-indKLN[i]
                newindKRN[i]=-indKRN[i]
                newindKsing[i]=np.mod(indKLsingN[i]+indKLN[i]-indKRN[i],N1D)

            newindKLNf=self.trueInd(newindKLN,N1D,nDim)
            newindKRNf=self.trueInd(newindKRN,N1D,nDim)
            newindKsingf=self.trueInd(newindKsing,N1D,nDim)

            indKLN=auxF.locIndex(newindKLNf,sIe)
            indKRN=auxF.locIndex(newindKRNf,sIe)
            indKsing=auxF.locIndex(newindKsingf,mIndexE)

            newIndWLN=-indWLN
            newIndWRN=-indWRN
            indWLN=auxF.locIndex(newIndWLN,tI[0])
            indWRN=auxF.locIndex(newIndWRN,tI[0])
            indWsing=auxF.locIndex(indWLs,tIL[0])

            self.indNEFPP=(indWsing*NW*NW+indWLN*NW+indWRN,\
                indKsing*NKF*NKF+indKLN*NKF+indKRN)
        elif chnL is 'PH':
            newindKLN=[None for i in range(nDim)]
            newindKRN=[None for i in range(nDim)]
            newindKsing=[None for i in range(nDim)]
            for i in range(nDim):
                newindKLN[i]=-indKRN[i]
                newindKRN[i]=-indKLN[i]
                newindKsing[i]=np.mod(-indKLsingN[i]-indKLN[i]+indKRN[i],N1D)

            newindKsingf=self.trueInd(newindKsing,N1D,nDim)
            newindKLNf=self.trueInd(newindKLN,N1D,nDim)
            newindKRNf=self.trueInd(newindKRN,N1D,nDim)

            indKLN=auxF.locIndex(newindKLNf,sIe)
            indKRN=auxF.locIndex(newindKRNf,sIe)
            indKsing=auxF.locIndex(newindKsingf,mIndexE)

            newIndWLN=indWRN
            newIndWRN=indWLN
            indWLN=auxF.locIndex(newIndWLN,tI[0])
            indWRN=auxF.locIndex(newIndWRN,tI[0])
            indWsing=auxF.locIndex(-indWLs,tIL[0])
            self.indNEFPHX=(indWsing*NW*NW+indWLN*NW+indWRN,\
                indKsing*NKF*NKF+indKLN*NKF+indKRN)

    def calcPositivityFull(self,chnL):
        sI=self.sPoints
        tI=self.tPoints
        tIL=self.tPointsL
        NW=self.NW
        NKF=self.NKF
        N=self.N
        N1D=self.N1D
        nDim=self.nDim
        NLmax=self.NLmax

        wIndexFull=np.arange(NLmax*NW*NW)
        indWLsing,rem=auxF.indXY(wIndexFull,NW*NW)
        indWL,indWR=auxF.indXY(rem,NW)

        kIndexFull=np.arange(N*NKF*NKF)
        indKLsing,rem=auxF.indXY(kIndexFull,NKF*NKF)
        indKL,indKR=auxF.indXY(rem,NKF)

        indexX=np.arange(N1D)
        mIndex=[None for i in range(nDim)]
        for i in range(nDim):
            lCur=np.zeros((N1D,)*nDim,dtype=int)
            for j in range(N1D):
                lCur[j]=indexX[j]
            lCur=np.reshape(np.swapaxes(lCur,0,i),N1D**nDim)
            mIndex[i]=lCur

        indKLsingN=[None for i in range(nDim)]
        indKLN=[None for i in range(nDim)]
        indKRN=[None for i in range(nDim)]
        for i in range(nDim):
            indKLsingN[i]=mIndex[i][indKLsing]
            indKLN[i]=sI[i][indKL]
            indKRN[i]=sI[i][indKR]

        indWLN=tI[0][indWL]
        indWRN=tI[0][indWR]
        indWLs=tIL[0][indWLsing]
        sIe=self.trueInd(sI,N1D,nDim)
        mIndexE=self.trueInd(mIndex,N1D,nDim)
        if chnL is 'PH':
            newindKLN=[None for i in range(nDim)]
            newindKRN=[None for i in range(nDim)]
            newindKsing=[None for i in range(nDim)]
            for i in range(nDim):
                newindKLN[i]=-indKLN[i]
                newindKRN[i]=-indKRN[i]
                newindKsing[i]=np.mod(indKLsingN[i]+indKLN[i]-indKRN[i],N1D)

            newindKLNf=self.trueInd(newindKLN,N1D,nDim)
            newindKRNf=self.trueInd(newindKRN,N1D,nDim)
            newindKsingf=self.trueInd(newindKsing,N1D,nDim)

            indKLN=auxF.locIndex(newindKLNf,sIe)
            indKRN=auxF.locIndex(newindKRNf,sIe)
            indKsing=auxF.locIndex(newindKsingf,mIndexE)

            newIndWLN=indWLN
            newIndWRN=indWRN
            indWLN=auxF.locIndex(newIndWLN,tI[0])
            indWRN=auxF.locIndex(newIndWRN,tI[0])
            indWsing=auxF.locIndex(-indWLs,tIL[0])
            self.indNPFPH=(indWsing*NW*NW+indWLN*NW+indWRN,\
                indKsing*NKF*NKF+indKLN*NKF+indKRN)
        else:
            newindKLN=[None for i in range(nDim)]
            newindKRN=[None for i in range(nDim)]
            newindKsing=[None for i in range(nDim)]
            for i in range(nDim):
                newindKLN[i]=indKRN[i]
                newindKRN[i]=indKLN[i]
                newindKsing[i]=np.mod(-indKLsingN[i],N1D)

            newindKLNf=self.trueInd(newindKLN,N1D,nDim)
            newindKRNf=self.trueInd(newindKRN,N1D,nDim)
            newindKsingf=self.trueInd(newindKsing,N1D,nDim)

            indKLN=auxF.locIndex(newindKLNf,sIe)
            indKRN=auxF.locIndex(newindKRNf,sIe)
            indKsing=auxF.locIndex(newindKsingf,mIndexE)

            newIndWLN=indWRN
            newIndWRN=indWLN
            indWLN=auxF.locIndex(newIndWLN,tI[0])
            indWRN=auxF.locIndex(newIndWRN,tI[0])
            indWsing=auxF.locIndex(indWLs,tIL[0])
            self.indNPFPX=(indWsing*NW*NW+indWLN*NW+indWRN,\
                indKsing*NKF*NKF+indKLN*NKF+indKRN)
