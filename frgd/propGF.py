import frgd.auxFunctions as auxF
from frgd.propG import scaleProp
import numpy as np
import time
class scalePropF(scaleProp):
    def susBubblesMB(self,wQ,kQ,tPoints,sPoints,fK,uFor,uBack):
        nB=self.nB
        nPatches=len(wQ)
        kPatches=len(kQ[0])
        NW=len(tPoints[0])
        NKF=len(sPoints[0])
        gPPL=[np.zeros((nPatches,kPatches,1,NW*NKF,nB,nB),dtype=np.complex_) for i in range(nB)]
        gPPR=[np.zeros((nPatches,kPatches,NW*NKF,1,nB,nB),dtype=np.complex_) for i in range(nB)]
        gPHL=[np.zeros((nPatches,kPatches,1,NW*NKF,nB,nB),dtype=np.complex_) for i in range(nB)]
        gPHR=[np.zeros((nPatches,kPatches,NW*NKF,1,nB,nB),dtype=np.complex_) for i in range(nB)]

        for i in range(nB):
            for j in range(nB):
                for k in range(nB):
                    uBandMap=[uBack[:,j,i],uFor[:,i,k]]
                    gPHLt,gPHRt,gPPLt,gPPRt=self.susBubbles(wQ,kQ,tPoints,sPoints,fK,uBandMap,j,k)
                    gPPL[i][...,j,k]=gPPLt
                    gPPR[i][...,j,k]=gPPRt
                    gPHL[i][...,j,k]=gPHLt
                    gPHR[i][...,j,k]=gPHRt
        return gPHL,gPHR,gPPL,gPPR

    def xBubblesD(self,wQ,kQ,dSEwMid,AC,tPoints,sPoints):
        """Calculate the single scale exchange propagator over
            basis set via convolutions"""
        wFX=self.wFX
        wFG=self.wFG

        kFIx,kFIy=self.kF
        kQx,kQy=kQ
        kIndexLoc=auxF.locIndex2(kQ,self.kF)

        nB=self.nB
        N=self.N
        N1D=self.N1D
        fA=auxF.fScaling(AC)
        beTa=self.beTa

        nPatches=len(wQ)
        ePatches=len(wFX)
        kPatches=len(kQx)
        lPatches=len(kFIx)

        tAvg=tPoints[0]/fA
        deltaT=tPoints[1]/fA
        NW=len(tAvg)

        tK=auxF.tBetaShift((tAvg[:,None]+tAvg[None,:]),beTa)[0]
        tKr,itK=auxF.uniquePoints(tK,deltaT)
        NWr=len(tKr)

        (wS,wX)=np.meshgrid(wQ,wFX,indexing='ij')
        lTempPP=np.zeros((len(wQ),len(wFX),NWr),dtype=np.complex_)
        lTempPH=np.zeros((len(wQ),len(wFX),NWr),dtype=np.complex_)
        for i in range(NWr):
            iCr=itK[i][0][0]
            jCr=itK[i][1][0]
            expE=np.exp(1j*(-2*wX+wS)*tKr[i])
            lTempPP[...,i]=expE*np.sinc((0.5*(-2*wX+wS)*deltaT[iCr])/np.pi)*\
                np.sinc((0.5*(-2*wX+wS)*deltaT[jCr])/np.pi)

            expE=np.exp(1j*(2*wX+wS)*tKr[i])
            lTempPH[...,i]=expE*np.sinc((0.5*(2*wX+wS)*deltaT[iCr])/np.pi)*\
                np.sinc((0.5*(2*wX+wS)*deltaT[jCr])/np.pi)

        NKF=len(sPoints[0])
        momKx=np.mod(sPoints[0][:,None]-sPoints[0][None,:],N1D)
        momKy=np.mod(sPoints[1][:,None]-sPoints[1][None,:],N1D)
        momKx=np.reshape(momKx,momKx.size)
        momKy=np.reshape(momKy,momKy.size)

        momKr,imomK=auxF.unique2DPoints((momKx,momKy),NKF)
        NKFn=len(momKr[0])

        rIndexX=momKr[0]
        rIndexY=momKr[1]

        retIndexF=np.zeros(NW*NKF*NW*NKF,dtype=int)
        for i in range(NWr):
            for j in range(NKFn):
                wL,wR=itK[i][0],itK[i][1]
                kL,kR=imomK[j][0],imomK[j][1]
                leftIndex=np.repeat(wL,len(kL))*NKF+np.tile(kL,len(wL))
                rightIndex=np.repeat(wR,len(kR))*NKF+np.tile(kR,len(wR))
                retIndexF[leftIndex*NW*NKF+rightIndex]=i*NKFn+j

        arrayShape=(nPatches,ePatches,lPatches,nB)
        sPropPP1=np.zeros(arrayShape,dtype=np.complex_)
        sPropPP2=np.zeros(arrayShape,dtype=np.complex_)
        gPropPP1=np.zeros(arrayShape,dtype=np.complex_)
        gPropPP2=np.zeros(arrayShape,dtype=np.complex_)

        sPropPH1=np.zeros(arrayShape,dtype=np.complex_)
        sPropPH2=np.zeros(arrayShape,dtype=np.complex_)
        gPropPH1=np.zeros(arrayShape,dtype=np.complex_)
        gPropPH2=np.zeros(arrayShape,dtype=np.complex_)

        mixPPTemp=np.zeros((nPatches,ePatches,nB*nB,NKFn),dtype=np.complex_)
        mixPHTemp=np.zeros((nPatches,ePatches,nB*nB,NKFn),dtype=np.complex_)
        mixPPt=np.zeros((nPatches,kPatches,nB*nB,NWr,NKFn),dtype=np.complex_)
        mixPHt=np.zeros((nPatches,kPatches,nB*nB,NWr,NKFn),dtype=np.complex_)
        for i,(kQCx,kQCy) in enumerate(zip(kQx,kQy)):
            for j in range(nB):
                gProp=self.gF(wFX,(kFIx,kFIy),AC,j)
                sProp=self.sF(wFX,(kFIx,kFIy),AC,j)
                dSEI=self.singInterp(wFX,(kFIx,kFIy),dSEwMid[j,j,:,:])

                for k,wQc in enumerate(wQ):
                    dSEIpp=self.singInterp(wQc-wFX,(kQCx-kFIx,kQCy-kFIy),dSEwMid[j,j,:,:])
                    dSEIph=self.singInterp(wQc+wFX,(kQCx+kFIx,kQCy+kFIy),dSEwMid[j,j,:,:])

                    sPropPP1[k,:,:,j]=wFG[:,None]*(sProp+dSEI*(gProp**2))
                    sPropPP2[k,:,:,j]=wFG[:,None]*(self.sF(wQc-wFX,(kQCx-kFIx,kQCy-kFIy),AC,j)+\
                        dSEIpp*(self.gF(wQc-wFX,(kQCx-kFIx,kQCy-kFIy),AC,j)**2))
                    gPropPP1[k,:,:,j]=self.gF(wQc-wFX,(kQCx-kFIx,kQCy-kFIy),AC,j)
                    gPropPP2[k,:,:,j]=self.gF(wFX,(kFIx,kFIy),AC,j)

                    sPropPH1[k,:,:,j]=wFG[:,None]*(sProp+dSEI*(gProp**2))
                    sPropPH2[k,:,:,j]=wFG[:,None]*(self.sF(wQc+wFX,(kQCx+kFIx,kQCy+kFIy),AC,j)+\
                        dSEIph*(self.gF(wQc+wFX,(kQCx+kFIx,kQCy+kFIy),AC,j)**2))
                    gPropPH1[k,:,:,j]=self.gF(wQc+wFX,(kQCx+kFIx,kQCy+kFIy),AC,j)
                    gPropPH2[k,:,:,j]=self.gF(wFX,(kFIx,kFIy),AC,j)

            for j,(jX,jY) in enumerate(zip(rIndexX,rIndexY)):
                mBasisPP=np.exp(1j*(kQCx-kFIx)*jX+1j*(kQCy-kFIy)*jY)
                mBasisPH=np.exp(1j*(kQCx+kFIx)*jX+1j*(kQCy+kFIy)*jY)
                for k in range(nB):
                    for l in range(nB):
                        curProd=sPropPP1[...,k]*gPropPP1[...,l]+gPropPP2[...,k]*sPropPP2[...,l]
                        mixPPTemp[:,:,k*nB+l,j]=(1.0/N)*np.sum(mBasisPP[None,None,:]*curProd,axis=2)
                        curProd=sPropPH1[...,k]*gPropPH1[...,l]+gPropPH2[...,k]*sPropPH2[...,l]
                        mixPHTemp[:,:,k*nB+l,j]=(1.0/N)*np.sum(mBasisPH[None,None,:]*curProd,axis=2)

            for j in range(NWr):
                mixPPt[:,i,:,j,:]=(1.0/beTa)*np.sum(mixPPTemp*lTempPP[:,:,j][:,:,None,None],axis=1)
                mixPHt[:,i,:,j,:]=(1.0/beTa)*np.sum(mixPHTemp*lTempPH[:,:,j][:,:,None,None],axis=1)

        mixPPt=np.reshape(mixPPt,(nPatches,kPatches,nB**2,NKFn*NWr))
        mixPPt=np.reshape(mixPPt[:,:,:,retIndexF],(nPatches,kPatches,nB**2,NW*NKF,NW*NKF))
        mixPHt=np.reshape(mixPHt,(nPatches,kPatches,nB**2,NKFn*NWr))
        mixPHt=np.reshape(mixPHt[:,:,:,retIndexF],(nPatches,kPatches,nB**2,NW*NKF,NW*NKF))

        wRange=np.arange(nPatches)
        nRange=np.arange(NW*NKF)
        nBrange=np.arange(nB**2)
        mixPPt=mixPPt[np.ix_(wRange,kIndexLoc,nBrange,nRange,nRange)]
        mixPHt=mixPHt[np.ix_(wRange,kIndexLoc,nBrange,nRange,nRange)]

        return mixPPt,mixPHt

    def xBubbles(self,wQ,kQ,dSEwMid,AC,tPoints,sPoints):
        """Calculate the single scale exchange propagator over
            basis set via convolutions"""
        wFX=self.wFX
        wFG=self.wFG

        kFI=self.kF

        nB=self.nB
        N=self.N
        N1D=self.N1D
        nDim=self.nDim
        kIndexLoc=auxF.indexOfMomenta(kQ,N1D)

        fA=auxF.fScaling(AC)
        beTa=self.beTa

        nPatches=len(wQ)
        ePatches=len(wFX)
        kPatches=len(kQ[0])
        lPatches=len(kFI[0])

        arrayShape=(nPatches,ePatches,lPatches,nB)
        sPropPP1=np.zeros(arrayShape,dtype=np.complex_)
        sPropPP2=np.zeros(arrayShape,dtype=np.complex_)
        gPropPP1=np.zeros(arrayShape,dtype=np.complex_)
        gPropPP2=np.zeros(arrayShape,dtype=np.complex_)

        sPropPH1=np.zeros(arrayShape,dtype=np.complex_)
        sPropPH2=np.zeros(arrayShape,dtype=np.complex_)
        gPropPH1=np.zeros(arrayShape,dtype=np.complex_)
        gPropPH2=np.zeros(arrayShape,dtype=np.complex_)
        for i in range(nB):
            gProp=self.gF(wFX,kFI,AC,i)
            sProp=self.sF(wFX,kFI,AC,i)
            dSEI=self.singInterp(wFX,kFI,dSEwMid[i,i,:,:])

            for j,wQc in enumerate(wQ):
                dSEIpp=self.singInterp(wQc-wFX,kFI,dSEwMid[i,i,:,:])
                dSEIph=self.singInterp(wQc+wFX,kFI,dSEwMid[i,i,:,:])

                sPropPP1[j,:,:,i]=wFG[:,None]*(sProp+dSEI*(gProp**2))
                sPropPP2[j,:,:,i]=wFG[:,None]*(self.sF(wQc-wFX,kFI,AC,i)+\
                    dSEIpp*(self.gF(wQc-wFX,kFI,AC,i)**2))
                gPropPP1[j,:,:,i]=self.gF(wQc-wFX,kFI,AC,i)
                gPropPP2[j,:,:,i]=self.gF(wFX,kFI,AC,i)

                sPropPH1[j,:,:,i]=wFG[:,None]*(sProp+dSEI*(gProp**2))
                sPropPH2[j,:,:,i]=wFG[:,None]*(self.sF(wQc+wFX,kFI,AC,i)+\
                    dSEIph*(self.gF(wQc+wFX,kFI,AC,i)**2))
                gPropPH1[j,:,:,i]=self.gF(wQc+wFX,kFI,AC,i)
                gPropPH2[j,:,:,i]=self.gF(wFX,kFI,AC,i)

        sPropPP1=auxF.ffTransformD(sPropPP1,N1D,nDim,axis=2)
        sPropPP2=auxF.ffTransformD(sPropPP2,N1D,nDim,axis=2)
        sPropPH1=auxF.ffTransformD(sPropPH1,N1D,nDim,axis=2)
        sPropPH2=auxF.ffTransformD(sPropPH2,N1D,nDim,axis=2)

        gPropPP1=auxF.ffTransformD(gPropPP1,N1D,nDim,axis=2)
        gPropPP2=auxF.ffTransformD(gPropPP2,N1D,nDim,axis=2)
        gPropPH1=auxF.ffTransformD(gPropPH1,N1D,nDim,axis=2)
        gPropPH2=auxF.ffTransformD(gPropPH2,N1D,nDim,axis=2)

        indexX=np.arange(N1D)
        mIndex=[None for i in range(nDim)]
        for i in range(nDim):
            lCur=np.zeros((N1D,)*nDim,dtype=int)
            for j in range(N1D):
                lCur[j]=indexX[j]
            lCur=np.reshape(np.swapaxes(lCur,0,i),N1D**nDim)
            mIndex[i]=lCur

        tAvg=tPoints[0]/fA
        deltaT=tPoints[1]/fA
        NW=len(tAvg)

        tK=auxF.tBetaShift((tAvg[:,None]+tAvg[None,:]),beTa)[0]
        tKr,itK=auxF.uniquePoints(tK,deltaT)
        NWr=len(tKr)

        (wS,wX)=np.meshgrid(wQ,wFX,indexing='ij')
        lTempPP=np.zeros((len(wQ),len(wFX),NWr),dtype=np.complex_)
        lTempPH=np.zeros((len(wQ),len(wFX),NWr),dtype=np.complex_)
        for i in range(NWr):
            iCr=itK[i][0][0]
            jCr=itK[i][1][0]
            expE=np.exp(1j*(-2*wX+wS)*tKr[i])
            lTempPP[...,i]=expE*np.sinc((0.5*(-2*wX+wS)*deltaT[iCr])/np.pi)*\
                np.sinc((0.5*(-2*wX+wS)*deltaT[jCr])/np.pi)

            expE=np.exp(1j*(2*wX+wS)*tKr[i])
            lTempPH[...,i]=expE*np.sinc((0.5*(2*wX+wS)*deltaT[iCr])/np.pi)*\
                np.sinc((0.5*(2*wX+wS)*deltaT[jCr])/np.pi)

        NKF=len(sPoints[0])
        momKn=[None for i in range(nDim)]
        for i in range(nDim):
            momCurInd=np.mod(sPoints[i][:,None]-sPoints[i][None,:],N1D)
            momKn[i]=np.reshape(momCurInd,momCurInd.size)

        momKr,imomK=auxF.uniqueDPoints(momKn,NKF,nDim)
        NKFn=len(momKr[0])

        retIndexF=np.zeros(NW*NKF*NW*NKF,dtype=int)
        for i in range(NWr):
            for j in range(NKFn):
                wL,wR=itK[i][0],itK[i][1]
                kL,kR=imomK[j][0],imomK[j][1]
                leftIndex=np.repeat(wL,len(kL))*NKF+np.tile(kL,len(wL))
                rightIndex=np.repeat(wR,len(kR))*NKF+np.tile(kR,len(wR))
                retIndexF[leftIndex*NW*NKF+rightIndex]=i*NKFn+j

        ppIndex=np.arange(N1D**nDim)
        phIndex=np.zeros(N1D**nDim,dtype=int)
        for i in range(nDim):
            phIndex+=np.mod(-mIndex[i],N1D)*(N1D**(nDim-1-i))

        ppIndexF=np.zeros((N1D**nDim,NKFn),dtype=int)
        phIndexF=np.zeros((N1D**nDim,NKFn),dtype=int)

        for i in range(NKFn):
            iCD=np.zeros(N1D**nDim,dtype=int)
            inPPcD=np.zeros(N1D**nDim,dtype=int)
            inPHcD=np.zeros(N1D**nDim,dtype=int)
            for j in range(nDim):
                iCD+=np.mod(mIndex[j],N1D)*(N1D**(nDim-1-j))
                inPPcD+=np.mod(mIndex[j]+momKr[j][i],N1D)*(N1D**(nDim-1-j))
                inPHcD+=np.mod(mIndex[j]+momKr[j][i],N1D)*(N1D**(nDim-1-j))

            ppIndexF[:,i]=inPPcD
            phIndexF[:,i]=inPHcD

        mixPPt=(1/beTa)*auxF.calcFreqSumSS(lTempPP,sPropPP1,gPropPP1,gPropPP2,sPropPP2,ppIndex,ppIndexF)
        mixPHt=(1/beTa)*auxF.calcFreqSumSS(lTempPH,sPropPH1,gPropPH1,gPropPH2,sPropPH2,phIndex,phIndexF)

        mixPPt=auxF.iffTransformD(mixPPt,N1D,nDim,axis=1)
        mixPHt=auxF.iffTransformD(mixPHt,N1D,nDim,axis=1)

        mixPPt=np.reshape(mixPPt,(nPatches,kPatches,nB**2,NKFn*NWr))
        mixPPt=np.reshape(mixPPt[:,:,:,retIndexF],(nPatches,kPatches,nB**2,NW*NKF,NW*NKF))
        mixPHt=np.reshape(mixPHt,(nPatches,kPatches,nB**2,NKFn*NWr))
        mixPHt=np.reshape(mixPHt[:,:,:,retIndexF],(nPatches,kPatches,nB**2,NW*NKF,NW*NKF))

        wRange=np.arange(nPatches)
        nRange=np.arange(NW*NKF)
        nBrange=np.arange(nB**2)
        mixPPt=mixPPt[np.ix_(wRange,kIndexLoc,nBrange,nRange,nRange)]
        mixPPt=np.swapaxes(np.swapaxes(mixPPt,2,3),3,4)
        mixPPt=np.reshape(mixPPt,(nPatches,kPatches,NW,NKF,NW,NKF,nB**2))

        mixPHt=mixPHt[np.ix_(wRange,kIndexLoc,nBrange,nRange,nRange)]
        mixPHt=np.swapaxes(np.swapaxes(mixPHt,2,3),3,4)
        mixPHt=np.reshape(mixPHt,(nPatches,kPatches,NW,NKF,NW,NKF,nB**2))

        return mixPPt,mixPHt

    def gBubblesD(self,wQ,kQ,AC,tPoints,sPoints):
        """Calculate the single scale exchange propagator over
            basis set via convolutions"""
        wFX=self.wFX
        wFG=self.wFG

        kFIx,kFIy=self.kF
        kQx,kQy=kQ
        kIndexLoc=auxF.locIndex2(kQ,self.kF)

        nB=self.nB
        N=self.N
        N1D=self.N1D
        fA=auxF.fScaling(AC)
        beTa=self.beTa

        nPatches=len(wQ)
        ePatches=len(wFX)
        kPatches=len(kQx)
        lPatches=len(kFIx)

        tAvg=tPoints[0]/fA
        deltaT=tPoints[1]/fA
        NW=len(tAvg)

        tK=auxF.tBetaShift((tAvg[:,None]+tAvg[None,:]),beTa)[0]
        tKr,itK=auxF.uniquePoints(tK,deltaT)
        NWr=len(tKr)

        (wS,wX)=np.meshgrid(wQ,wFX,indexing='ij')
        lTempPP=np.zeros((len(wQ),len(wFX),NWr),dtype=np.complex_)
        lTempPH=np.zeros((len(wQ),len(wFX),NWr),dtype=np.complex_)
        for i in range(NWr):
            iCr=itK[i][0][0]
            jCr=itK[i][1][0]
            expE=np.exp(1j*(-2*wX+wS)*tKr[i])
            lTempPP[...,i]=expE*np.sinc((0.5*(-2*wX+wS)*deltaT[iCr])/np.pi)*\
                np.sinc((0.5*(-2*wX+wS)*deltaT[jCr])/np.pi)

            expE=np.exp(1j*(2*wX+wS)*tKr[i])
            lTempPH[...,i]=expE*np.sinc((0.5*(2*wX+wS)*deltaT[iCr])/np.pi)*\
                np.sinc((0.5*(2*wX+wS)*deltaT[jCr])/np.pi)

        NKF=len(sPoints[0])
        momKx=np.mod(sPoints[0][:,None]-sPoints[0][None,:],N1D)
        momKy=np.mod(sPoints[1][:,None]-sPoints[1][None,:],N1D)
        momKx=np.reshape(momKx,momKx.size)
        momKy=np.reshape(momKy,momKy.size)

        momKr,imomK=auxF.unique2DPoints((momKx,momKy),NKF)
        NKFn=len(momKr[0])

        rIndexX=momKr[0]
        rIndexY=momKr[1]

        retIndexF=np.zeros(NW*NKF*NW*NKF,dtype=int)
        for i in range(NWr):
            for j in range(NKFn):
                wL,wR=itK[i][0],itK[i][1]
                kL,kR=imomK[j][0],imomK[j][1]
                leftIndex=np.repeat(wL,len(kL))*NKF+np.tile(kL,len(wL))
                rightIndex=np.repeat(wR,len(kR))*NKF+np.tile(kR,len(wR))
                retIndexF[leftIndex*NW*NKF+rightIndex]=i*NKFn+j

        arrayShape=(nPatches,ePatches,lPatches,nB)
        gPropPP1=np.zeros(arrayShape,dtype=np.complex_)
        gPropPP2=np.zeros(arrayShape,dtype=np.complex_)
        gPropPH1=np.zeros(arrayShape,dtype=np.complex_)
        gPropPH2=np.zeros(arrayShape,dtype=np.complex_)

        gPPTemp=np.zeros((nPatches,ePatches,nB*nB,NKFn),dtype=np.complex_)
        gPHTemp=np.zeros((nPatches,ePatches,nB*nB,NKFn),dtype=np.complex_)
        gPPt=np.zeros((nPatches,kPatches,nB*nB,NWr,NKFn),dtype=np.complex_)
        gPHt=np.zeros((nPatches,kPatches,nB*nB,NWr,NKFn),dtype=np.complex_)
        for i,(kQCx,kQCy) in enumerate(zip(kQx,kQy)):
            for j in range(nB):
                gProp=self.gF(wFX,(kFIx,kFIy),AC,j)
                for k,wQc in enumerate(wQ):
                    gPropPP2[k,:,:,j]=self.gF(wQc-wFX,(kQCx-kFIx,kQCy-kFIy),AC,j)
                    gPropPP1[k,:,:,j]=wFG[:,None]*self.gF(wFX,(kFIx,kFIy),AC,j)

                    gPropPH2[k,:,:,j]=self.gF(wQc+wFX,(kQCx+kFIx,kQCy+kFIy),AC,j)
                    gPropPH1[k,:,:,j]=wFG[:,None]*self.gF(wFX,(kFIx,kFIy),AC,j)

            for j,(jX,jY) in enumerate(zip(rIndexX,rIndexY)):
                mBasisPP=np.exp(1j*(kQCx-kFIx)*jX+1j*(kQCy-kFIy)*jY)
                mBasisPH=np.exp(1j*(kQCx+kFIx)*jX+1j*(kQCy+kFIy)*jY)
                for k in range(nB):
                    for l in range(nB):
                        curProd=gPropPP1[...,k]*gPropPP2[...,l]
                        gPPTemp[:,:,k*nB+l,j]=(1.0/N)*np.sum(mBasisPP[None,None,:]*curProd,axis=2)
                        curProd=gPropPH1[...,k]*gPropPH2[...,l]
                        gPHTemp[:,:,k*nB+l,j]=(1.0/N)*np.sum(mBasisPH[None,None,:]*curProd,axis=2)

            for j in range(NWr):
                gPPt[:,i,:,j,:]=(1.0/beTa)*np.sum(gPPTemp*lTempPP[:,:,j][:,:,None,None],axis=1)
                gPHt[:,i,:,j,:]=(1.0/beTa)*np.sum(gPHTemp*lTempPH[:,:,j][:,:,None,None],axis=1)

        gPPt=np.reshape(gPPt,(nPatches,kPatches,nB**2,NKFn*NWr))
        gPPt=np.reshape(gPPt[:,:,:,retIndexF],(nPatches,kPatches,nB**2,NW*NKF,NW*NKF))
        gPHt=np.reshape(gPHt,(nPatches,kPatches,nB**2,NKFn*NWr))
        gPHt=np.reshape(gPHt[:,:,:,retIndexF],(nPatches,kPatches,nB**2,NW*NKF,NW*NKF))

        wRange=np.arange(nPatches)
        nRange=np.arange(NW*NKF)
        nBrange=np.arange(nB**2)
        gPPt=gPPt[np.ix_(wRange,kIndexLoc,nBrange,nRange,nRange)]
        gPHt=gPHt[np.ix_(wRange,kIndexLoc,nBrange,nRange,nRange)]

        return gPPt,gPHt

    def gBubbles(self,wQ,kQ,AC,tPoints,sPoints):
        """Calculate the single scale exchange propagator over
            basis set via convolutions"""
        wFX=self.wFX
        wFG=self.wFG

        kFI=self.kF
        fA=auxF.fScaling(AC)
        nB=self.nB
        N=self.N
        N1D=self.N1D
        nDim=self.nDim
        kIndexLoc=auxF.indexOfMomenta(kQ,N1D)
        beTa=self.beTa

        nPatches=len(wQ)
        ePatches=len(wFX)
        kPatches=len(kQ[0])
        lPatches=len(kFI[0])

        arrayShape=(nPatches,ePatches,lPatches,nB)
        gPropPP1=np.zeros(arrayShape,dtype=np.complex_)
        gPropPP2=np.zeros(arrayShape,dtype=np.complex_)
        gPropPH1=np.zeros(arrayShape,dtype=np.complex_)
        gPropPH2=np.zeros(arrayShape,dtype=np.complex_)
        for i in range(nB):
            for j,wQc in enumerate(wQ):
                gPropPP1[j,:,:,i]=wFG[:,None]*self.gF(wFX,kFI,AC,i)
                gPropPP2[j,:,:,i]=self.gF(wQc-wFX,kFI,AC,i)
                gPropPH1[j,:,:,i]=wFG[:,None]*self.gF(wFX,kFI,AC,i)
                gPropPH2[j,:,:,i]=self.gF(wQc+wFX,kFI,AC,i)

        gPropPP1=auxF.ffTransformD(gPropPP1,N1D,nDim,axis=2)
        gPropPP2=auxF.ffTransformD(gPropPP2,N1D,nDim,axis=2)
        gPropPH1=auxF.ffTransformD(gPropPH1,N1D,nDim,axis=2)
        gPropPH2=auxF.ffTransformD(gPropPH2,N1D,nDim,axis=2)

        indexX=np.arange(N1D)
        mIndex=[None for i in range(nDim)]
        for i in range(nDim):
            lCur=np.zeros((N1D,)*nDim,dtype=int)
            for j in range(N1D):
                lCur[j]=indexX[j]
            lCur=np.reshape(np.swapaxes(lCur,0,i),N1D**nDim)
            mIndex[i]=lCur

        tAvg=tPoints[0]/fA
        deltaT=tPoints[1]/fA
        NW=len(tAvg)

        tK=auxF.tBetaShift((tAvg[:,None]+tAvg[None,:]),beTa)[0]
        tKr,itK=auxF.uniquePoints(tK,deltaT)
        NWr=len(tKr)

        (wS,wX)=np.meshgrid(wQ,wFX,indexing='ij')
        lTempPP=np.zeros((len(wQ),len(wFX),NWr),dtype=np.complex_)
        lTempPH=np.zeros((len(wQ),len(wFX),NWr),dtype=np.complex_)
        for i in range(NWr):
            iCr=itK[i][0][0]
            jCr=itK[i][1][0]
            expE=np.exp(1j*(-2*wX+wS)*tKr[i])
            lTempPP[...,i]=expE*np.sinc((0.5*(-2*wX+wS)*deltaT[iCr])/np.pi)*\
                np.sinc((0.5*(-2*wX+wS)*deltaT[jCr])/np.pi)

            expE=np.exp(1j*(2*wX+wS)*tKr[i])
            lTempPH[...,i]=expE*np.sinc((0.5*(2*wX+wS)*deltaT[iCr])/np.pi)*\
                np.sinc((0.5*(2*wX+wS)*deltaT[jCr])/np.pi)

        NKF=len(sPoints[0])
        momKn=[None for i in range(nDim)]
        for i in range(nDim):
            momCurInd=np.mod(sPoints[i][:,None]-sPoints[i][None,:],N1D)
            momKn[i]=np.reshape(momCurInd,momCurInd.size)

        momKr,imomK=auxF.uniqueDPoints(momKn,NKF,nDim)
        NKFn=len(momKr[0])

        retIndexF=np.zeros(NW*NKF*NW*NKF,dtype=int)
        for i in range(NWr):
            for j in range(NKFn):
                wL,wR=itK[i][0],itK[i][1]
                kL,kR=imomK[j][0],imomK[j][1]
                leftIndex=np.repeat(wL,len(kL))*NKF+np.tile(kL,len(wL))
                rightIndex=np.repeat(wR,len(kR))*NKF+np.tile(kR,len(wR))
                retIndexF[leftIndex*NW*NKF+rightIndex]=i*NKFn+j

        ppIndex=np.arange(N1D**nDim)
        phIndex=np.zeros(N1D**nDim,dtype=int)
        for i in range(nDim):
            phIndex+=np.mod(-mIndex[i],N1D)*(N1D**(nDim-1-i))

        ppIndexF=np.zeros((N1D**nDim,NKFn),dtype=int)
        phIndexF=np.zeros((N1D**nDim,NKFn),dtype=int)
        for i in range(NKFn):
            iCD=np.zeros(N1D**nDim,dtype=int)
            inPPcD=np.zeros(N1D**nDim,dtype=int)
            inPHcD=np.zeros(N1D**nDim,dtype=int)
            for j in range(nDim):
                iCD+=np.mod(mIndex[j],N1D)*(N1D**(nDim-1-j))
                inPPcD+=np.mod(mIndex[j]+momKr[j][i],N1D)*(N1D**(nDim-1-j))
                inPHcD+=np.mod(mIndex[j]+momKr[j][i],N1D)*(N1D**(nDim-1-j))

            ppIndexF[:,i]=inPPcD
            phIndexF[:,i]=inPHcD

        gPPt=(1.0/beTa)*auxF.calcFreqSumSF(lTempPP,gPropPP1,gPropPP2,ppIndex,ppIndexF)
        gPHt=(1.0/beTa)*auxF.calcFreqSumSF(lTempPH,gPropPH1,gPropPH2,phIndex,phIndexF)

        gPPt=auxF.iffTransformD(gPPt,N1D,nDim,axis=1)
        gPHt=auxF.iffTransformD(gPHt,N1D,nDim,axis=1)

        gPPt=np.reshape(gPPt,(nPatches,kPatches,nB**2,NWr*NKFn))
        gPPt=np.reshape(gPPt[:,:,:,retIndexF],(nPatches,kPatches,nB**2,NW*NKF,NW*NKF))
        gPHt=np.reshape(gPHt,(nPatches,kPatches,nB**2,NWr*NKFn))
        gPHt=np.reshape(gPHt[:,:,:,retIndexF],(nPatches,kPatches,nB**2,NW*NKF,NW*NKF))
        wRange=np.arange(nPatches)
        nRange=np.arange(NW*NKF)
        nBrange=np.arange(nB*nB)
        gPPt=gPPt[np.ix_(wRange,kIndexLoc,nBrange,nRange,nRange)]
        gPPt=np.swapaxes(np.swapaxes(gPPt,2,3),3,4)
        gPPt=np.reshape(gPPt,(nPatches,kPatches,NW,NKF,NW,NKF,nB**2))
        gPHt=gPHt[np.ix_(wRange,kIndexLoc,nBrange,nRange,nRange)]
        gPHt=np.swapaxes(np.swapaxes(gPHt,2,3),3,4)
        gPHt=np.reshape(gPHt,(nPatches,kPatches,NW,NKF,NW,NKF,nB**2))

        return gPPt,gPHt

    def susBubblesD(self,wQ,kQ,tPoints,sPoints,fK,uBandMap,bL,bR):
        wFX=self.wFX
        wFG=self.wFG

        kFx,kFy=self.kF
        kQx,kQy=kQ
        N=self.N
        N1D=self.N1D
        beTa=self.beTa

        nPatches=len(wQ)
        ePatches=len(wFX)
        kPatches=len(kFx)
        lPatches=len(kFx)

        tAvg=tPoints[0]
        deltaT=tPoints[1]

        xAvg=sPoints[0]
        yAvg=sPoints[1]

        NW=len(tAvg)
        NKF=len(xAvg)
        gPPL=np.zeros((nPatches,kPatches,1,NW*NKF),dtype=np.complex_)
        gPPR=np.zeros((nPatches,kPatches,NW*NKF,1),dtype=np.complex_)
        gPHL=np.zeros((nPatches,kPatches,1,NW*NKF),dtype=np.complex_)
        gPHR=np.zeros((nPatches,kPatches,NW*NKF,1),dtype=np.complex_)

        fFacL=fK[0]((kFx,kFy))
        fFacR=fK[1]((kFx,kFy))

        gPPLt=np.zeros((nPatches,ePatches,NKF),dtype=np.complex_)
        gPPRt=np.zeros((nPatches,ePatches,NKF),dtype=np.complex_)
        gPHLt=np.zeros((nPatches,ePatches,NKF),dtype=np.complex_)
        gPHRt=np.zeros((nPatches,ePatches,NKF),dtype=np.complex_)

        gPropPP1=np.zeros((nPatches,ePatches,lPatches),dtype=np.complex_)
        gPropPP2=np.zeros(gPropPP1.shape,dtype=np.complex_)
        gPropPH1=np.zeros(gPropPP1.shape,dtype=np.complex_)
        gPropPH2=np.zeros(gPropPP1.shape,dtype=np.complex_)

        (wS,wX)=np.meshgrid(wQ,wFX,indexing='ij')

        for i,(kQCx,kQCy) in enumerate(zip(kQx,kQy)):
            gProp=self.gF(wFX,(kFx,kFy),0.0,bL)
            gPropPPL=wFG[:,None]*fFacL[None,:]*uBandMap[0][None,:]*gProp
            gProp=self.gF(wFX,(kFx,kFy),0.0,bR)
            gPropPPR=wFG[:,None]*fFacR[None,:]*uBandMap[1][None,:]*gProp

            gProp=self.gF(wFX,(kFx,kFy),0.0,bL)
            gPropL=wFG[:,None]*fFacL[None,:]*uBandMap[0][None,:]*gProp
            gPropR=wFG[:,None]*fFacR[None,:]*uBandMap[0][None,:]*gProp
            for j,wQc in enumerate(wQ):
                kQppInd=auxF.indexOfMomenta((kQCx-kFx,kQCy-kFy),N1D)
                gPropPP1[j,:,:]=self.gF(wQc-wFX,(kQCx-kFx,kQCy-kFy),0.0,bR)*uBandMap[0][kQppInd][None,:]
                gPropPP2[j,:,:]=self.gF(wQc-wFX,(kQCx-kFx,kQCy-kFy),0.0,bL)*uBandMap[1][kQppInd][None,:]

                kQphInd=auxF.indexOfMomenta((-kQCx+kFx,-kQCy+kFy),N1D)
                gPropPH1[j,:,:]=self.gF(-wQc+wFX,(-kQCx+kFx,-kQCy+kFy),0.0,bR)*uBandMap[1][kQphInd][None,:]
                kQphInd=auxF.indexOfMomenta((kQCx+kFx,kQCy+kFy),N1D)
                gPropPH2[j,:,:]=self.gF(wQc+wFX,(kQCx+kFx,kQCy+kFy),0.0,bR)*uBandMap[1][kQphInd][None,:]

            for j,(jX,jY) in enumerate(zip(xAvg,yAvg)):
                momPP=np.exp(-1j*(kQCx-kFx)*jX-1j*(kQCy-kFy)*jY)
                gPPLt[:,:,j]=(1.0/N)*np.sum(momPP[None,None,:]*gPropPPL*gPropPP1,axis=2)
                momPP=np.exp(1j*(kQCx-kFx)*jX+1j*(kQCy-kFy)*jY)
                gPPRt[:,:,j]=(1.0/N)*np.sum(momPP[None,None,:]*gPropPPR*gPropPP2,axis=2)

                momPH=np.exp(-1j*kFx*jX-1j*kFy*jY)
                gPHLt[:,:,j]=(1.0/N)*np.sum(momPH[None,None,:]*gPropL*gPropPH1,axis=2)
                momPH=np.exp(1j*(kQCx+kFx)*jX+1j*(kQCy+kFy)*jY)
                gPHRt[:,:,j]=(1.0/N)*np.sum(momPH[None,None,:]*gPropR*gPropPH2,axis=2)

            for k in range(NW):
                lTempPPL=np.exp(-1j*tAvg[k]*(2*wX-wS))*np.sinc((0.5*(2*wX-wS)*deltaT[k])/np.pi)
                lTempPPR=np.exp(1j*tAvg[k]*(-2*wX+wS))*np.sinc((0.5*(-2*wX+wS)*deltaT[k])/np.pi)

                lTempPHL=np.exp(1j*tAvg[k]*(2*wX-wS))*np.sinc((0.5*(2*wX-wS)*deltaT[k])/np.pi)
                lTempPHR=np.exp(1j*tAvg[k]*(2*wX+wS))*np.sinc((0.5*(2*wX+wS)*deltaT[k])/np.pi)

                gPPL[:,i,0,(k*NKF):((k+1)*NKF)]=(1/beTa)*np.sum(lTempPPL[:,:,None]*gPPLt,axis=1)
                gPPR[:,i,(k*NKF):((k+1)*NKF),0]=(1/beTa)*np.sum(lTempPPR[:,:,None]*gPPRt,axis=1)

                gPHL[:,i,0,(k*NKF):((k+1)*NKF)]=(1/beTa)*np.sum(lTempPHL[:,:,None]*gPHLt,axis=1)
                gPHR[:,i,(k*NKF):((k+1)*NKF),0]=(1/beTa)*np.sum(lTempPHR[:,:,None]*gPHRt,axis=1)

        NWrange=np.arange(NW*NKF)
        kIndexLoc=auxF.locIndex2(kQ,self.kF)
        gPPL=gPPL[:,kIndexLoc,:,:]
        gPPR=gPPR[:,kIndexLoc,:,:]
        gPHL=gPHL[:,kIndexLoc,:,:]
        gPHR=gPHR[:,kIndexLoc,:,:]
        return gPHL,gPHR,gPPL,gPPR

    def susBubbles(self,wQ,kQ,tPoints,sPoints,fK,uBandMap,bL,bR):
        wFX=self.wFX
        wFG=self.wFG

        kF=self.kF
        N=self.N
        N1D=self.N1D
        nDim=self.nDim
        beTa=self.beTa

        nPatches=len(wQ)
        ePatches=len(wFX)
        kPatches=len(kF[0])
        lPatches=len(kF[0])

        tAvg=tPoints[0]
        deltaT=tPoints[1]

        NW=len(tAvg)
        NKF=len(sPoints[0])
        gPPL=np.zeros((nPatches,kPatches,1,NW*NKF),dtype=np.complex_)
        gPPR=np.zeros((nPatches,kPatches,NW*NKF,1),dtype=np.complex_)
        gPHL=np.zeros((nPatches,kPatches,1,NW*NKF),dtype=np.complex_)
        gPHR=np.zeros((nPatches,kPatches,NW*NKF,1),dtype=np.complex_)

        AC=0.0
        fFac=fK
        gProp=self.gF(wFX,kF,AC,bL)
        gPropPPL=wFG[:,None]*fFac[None,:]*uBandMap[0][None,:]*gProp
        gProp=self.gF(wFX,kF,AC,bR)
        gPropPPR=wFG[:,None]*fFac[None,:]*uBandMap[1][None,:]*gProp
        gProp=self.gF(wFX,kF,AC,bL)
        gPropL=wFG[:,None]*fFac[None,:]*uBandMap[0][None,:]*gProp
        gPropR=wFG[:,None]*fFac[None,:]*uBandMap[0][None,:]*gProp

        gPropPP1=np.zeros((nPatches,ePatches,lPatches),dtype=np.complex_)
        gPropPP2=np.zeros(gPropPP1.shape,dtype=np.complex_)
        gPropPH1=np.zeros(gPropPP1.shape,dtype=np.complex_)
        gPropPH2=np.zeros(gPropPP1.shape,dtype=np.complex_)
        for i,wQc in enumerate(wQ):
            gPropPP1[i,:,:]=self.gF(wQc-wFX,kF,0.0,bR)*uBandMap[0][None,:]
            gPropPP2[i,:,:]=self.gF(wQc-wFX,kF,0.0,bL)*uBandMap[1][None,:]
            gPropPH1[i,:,:]=self.gF(wQc+wFX,kF,0.0,bR)*uBandMap[1][None,:]
            gPropPH2[i,:,:]=self.gF(-wQc+wFX,kF,0.0,bR)*uBandMap[1][None,:]

        gPropKL=auxF.ffTransformD(gPropL,N1D,nDim,axis=1)
        gPropKR=auxF.ffTransformD(gPropR,N1D,nDim,axis=1)
        gPropPPKL=auxF.ffTransformD(gPropPPL,N1D,nDim,axis=1)
        gPropPPKR=auxF.ffTransformD(gPropPPR,N1D,nDim,axis=1)
        gPropPP1K=auxF.ffTransformD(gPropPP1,N1D,nDim,axis=2)
        gPropPP2K=auxF.ffTransformD(gPropPP2,N1D,nDim,axis=2)
        gPropPH1K=auxF.ffTransformD(gPropPH1,N1D,nDim,axis=2)
        gPropPH2K=auxF.ffTransformD(gPropPH2,N1D,nDim,axis=2)

        (wS,wX)=np.meshgrid(wQ,wFX,indexing='ij')
        indexX=np.arange(N1D)
        mIndex=[None for i in range(nDim)]
        for i in range(nDim):
            lCur=np.zeros((N1D,)*nDim,dtype=int)
            for j in range(N1D):
                lCur[j]=indexX[j]
            lCur=np.reshape(np.swapaxes(lCur,0,i),N1D**nDim)
            mIndex[i]=lCur

        gPPLt=np.zeros((nPatches,lPatches,NW),dtype=np.complex_)
        gPPRt=np.zeros((nPatches,lPatches,NW),dtype=np.complex_)
        gPHLt=np.zeros((nPatches,lPatches,NW),dtype=np.complex_)
        gPHRt=np.zeros((nPatches,lPatches,NW),dtype=np.complex_)

        for j in range(NKF):
            jCD=np.zeros(N1D**nDim,dtype=int)
            inPPlD=np.zeros(N1D**nDim,dtype=int)
            inPPrD=np.zeros(N1D**nDim,dtype=int)
            for i in range(nDim):
                jCD+=np.mod(mIndex[i],N1D)*(N1D**(nDim-1-i))
                inPPlD+=np.mod(mIndex[i]-sPoints[i][j],N1D)*(N1D**(nDim-1-i))
                inPPrD+=np.mod(mIndex[i]+sPoints[i][j],N1D)*(N1D**(nDim-1-i))

            gPPlt=gPropPPKL[None,:,jCD]*gPropPP1K[:,:,inPPlD]
            gPPrt=gPropPPKR[None,:,jCD]*gPropPP2K[:,:,inPPrD]

            jCD=np.zeros(N1D**nDim,dtype=int)
            inPHlD=np.zeros(N1D**nDim,dtype=int)
            inPHrD=np.zeros(N1D**nDim,dtype=int)
            for i in range(nDim):
                jCD+=np.mod(-mIndex[i],N1D)*(N1D**(nDim-1-i))
                inPHlD+=np.mod(mIndex[i]-sPoints[i][j],N1D)*(N1D**(nDim-1-i))
                inPHrD+=np.mod(mIndex[i]+sPoints[i][j],N1D)*(N1D**(nDim-1-i))

            gPHlt=gPropKL[None,:,inPHlD]*gPropPH2K[:,:,jCD]
            gPHrt=gPropKR[None,:,jCD]*gPropPH1K[:,:,inPHrD]

            for k in range(NW):
                lTempPPL=np.exp(-1j*tAvg[k]*(2*wX-wS))*np.sinc((0.5*(2*wX-wS)*deltaT[k])/np.pi)
                lTempPPR=np.exp(1j*tAvg[k]*(-2*wX+wS))*np.sinc((0.5*(-2*wX+wS)*deltaT[k])/np.pi)

                lTempPHL=np.exp(1j*tAvg[k]*(2*wX-wS))*np.sinc((0.5*(2*wX-wS)*deltaT[k])/np.pi)
                lTempPHR=np.exp(1j*tAvg[k]*(2*wX+wS))*np.sinc((0.5*(2*wX+wS)*deltaT[k])/np.pi)

                gPPLt[:,:,k]=(1/beTa)*np.sum(gPPlt*lTempPPL[:,:,None],axis=1)
                gPPRt[:,:,k]=(1/beTa)*np.sum(gPPrt*lTempPPR[:,:,None],axis=1)

                gPHLt[:,:,k]=(1/beTa)*np.sum(gPHlt*lTempPHL[:,:,None],axis=1)
                gPHRt[:,:,k]=(1/beTa)*np.sum(gPHrt*lTempPHR[:,:,None],axis=1)
            gPPL[:,:,0,j::NKF]=auxF.iffTransformD(gPPLt,N1D,nDim,axis=1)
            gPPR[:,:,j::NKF,0]=auxF.iffTransformD(gPPRt,N1D,nDim,axis=1)

            gPHL[:,:,0,j::NKF]=auxF.iffTransformD(gPHLt,N1D,nDim,axis=1)
            gPHR[:,:,j::NKF,0]=auxF.iffTransformD(gPHRt,N1D,nDim,axis=1)

        NWrange=np.arange(NW*NKF)
        kIndexLoc=auxF.indexOfMomenta(kQ,N1D)
        gPPL=gPPL[:,kIndexLoc,:,:]
        gPPR=gPPR[:,kIndexLoc,:,:]
        gPHL=gPHL[:,kIndexLoc,:,:]
        gPHR=gPHR[:,kIndexLoc,:,:]
        return gPHL,gPHR,gPPL,gPPR
