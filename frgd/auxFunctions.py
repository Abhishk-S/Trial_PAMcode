import numpy as np
from numba import jit
import time
import sys

#@jit(nopython=True)
def sparseMulAlt(UnXL,mixC,UnXR):
    #return np.sum(np.matmul(UnXL,np.matmul(mixC,UnXR)),axis=2)

    uShape=UnXL.shape[:2]+UnXL.shape[3:]
    UnX=np.zeros(uShape,dtype=np.complex_)
    nB=UnXL.shape[2]

    curBasis=UnXL.shape[3]
    wRange=np.arange(mixC.shape[0])
    kRange=np.arange(mixC.shape[1])

    toL=0.99
    for i in range(nB):
        indLx,indLy=relSpecWght(UnXL[:,:,i,:,:],np.zeros(1,dtype=int),toL,axis=(2,3,),sumAxis=(0,1,))
        indRx,indRy=relSpecWght(UnXR[:,:,i,:,:],np.zeros(1,dtype=int),toL,axis=(2,3,),sumAxis=(0,1,))

        indNewX=np.unique(indLx)
        indNewY=np.unique(indRy)

        indMidX=np.unique(indLy)
        indMidY=np.unique(indRx)

        UnXLc=UnXL[:,:,i,:,:][np.ix_(wRange,kRange,indNewX,indMidX)]
        mixCc=mixC[:,:,i,:,:][np.ix_(wRange,kRange,indMidX,indMidY)]
        UnXRc=UnXR[:,:,i,:,:][np.ix_(wRange,kRange,indMidY,indNewY)]

        UnX[np.ix_(wRange,kRange,indNewX,indNewY)]+=np.matmul(UnXLc,np.matmul(mixCc,UnXRc))

    return UnX

@jit(nopython=True)
def maskVertex(uLeft,UnX,uRight):
    NB=uLeft.shape[3]
    NW=UnX.shape[2]
    NKF=UnX.shape[3]
    uCur=np.zeros(UnX.shape[:2]+(NB,NW,NKF,NB,NW,NKF,),dtype=np.complex_)
    for i in range(NB):
        for j in range(NB):
            uMap=uLeft[:,:,:,i]*uRight[:,:,:,j]
            for k in range(UnX.shape[0]):
                for l in range(UnX.shape[2]):
                    for m in range(UnX.shape[2]):
                        uCur[k,:,i,l,:,j,m,:]+=uMap*UnX[k,:,l,:,m,:]

    return uCur
@jit(nopython=True)
def genVertex(uLeft,UnX,uRight):
    uTemp=np.zeros(UnX.shape[:6],dtype=np.complex_)
    NB=UnX.shape[2]
    for i in range(NB):
        for j in range(NB):
            uMap=uLeft[:,:,:,i]*uRight[:,:,:,j]
            for k in range(UnX.shape[0]):
                for l in range(UnX.shape[2]):
                    for m in range(UnX.shape[2]):
                        uTemp[k,:,l,:,m,:]+=uMap*UnX[k,:,l,:,m,:,i,j]
    return uTemp

def redBandVertex(uLeft,uX,uRight):
    wLen=uX.shape[0]
    NB=uLeft.shape[3]
    NKF=uLeft.shape[1]
    N=uLeft.shape[0]
    uRed=np.zeros((wLen,NB,NKF,NB,NKF),dtype=np.complex_)
    for i in range(NB):
        for j in range(NB):
            uMap=uLeft[:,:,:,i]*uRight[:,:,:,j]
            uRed+=uMap[None,:,:,:]*uX

    return uRed
def fullVertex(uTemp,kTrans,indK,uShape):
    uTempFull=np.zeros(uShape,dtype=np.complex_)
    kPhaseFull=np.zeros(uShape[1:],dtype=np.complex_)
    for j in range(len(indK)):
        curIndex=kTrans[j][0]
        fftIndex=kTrans[j][1]
        kPhase=kTrans[j][2]

        indKL,indKR=indXY(indK,uShape[2])
        uTempFull[:,fftIndex,indKL,indKR]=uTemp[:,curIndex]
        kPhaseFull[:,indKL,indKR]=kPhase
    uTempFull=iffTransform2D(uTempFUll,N1D,axis=1)*kPhaseFull[None,...]
    return uTempFull
@jit(nopython=True)
def projectMix(mixCur,uLeft,uRight):
    NW=mixCur.shape[2]
    NKF=uLeft.shape[1]
    NB=uLeft.shape[4]
    nPatches=mixCur.shape[0]
    kPatches=mixCur.shape[1]
    nL=mixCur.shape[6]
    mixF=np.zeros((nPatches,kPatches,NB,NW,NKF,NB,NW,NKF),dtype=np.complex_)
    for i in range(NB):
        for j in range(NB):
            uMap=uRight[:,:,:,i,:]*uLeft[:,:,:,:,j]
            for k in range(nPatches):
                for l in range(NW):
                    for m in range(NW):
                        mixF[k,:,i,l,:,j,m,:]+=np.sum(uMap*mixCur[k,:,l,:,m,:,:],axis=3)

    return mixF

def sparseMul(UnXL,mixC,UnXR):
    return np.matmul(UnXL,np.matmul(mixC,UnXR))

"""
    nB=UnXL.shape[2]
    nPatches=mixC.shape[0]
    kPatches=mixC.shape[1]
    curBasis=UnXL.shape[3]
    wRange=np.arange(mixC.shape[0])
    kRange=np.arange(mixC.shape[1])

    UnX=np.zeros((nPatches*kPatches,curBasis,curBasis),dtype=np.complex_)
    toL=0.999
    for i in range(nB):
        mixNB=np.reshape(mixC[:,:,i,:,:],(nPatches*kPatches,curBasis,curBasis))

        uLTemp=np.reshape(UnXL[:,:,i,:,:],(nPatches*kPatches,curBasis,curBasis))
        uRTemp=np.reshape(UnXR[:,:,i,:,:],(nPatches*kPatches,curBasis,curBasis))
        indWKL,indLx,indLy=spectralWeight(uLTemp,toL)
        indWKR,indRx,indRy=spectralWeight(uRTemp,toL)

        indNewX=np.unique(indLx)
        indNewY=np.unique(indRy)

        for j in range(len(indNewX)):
            locL=np.where(indNewX[j]==indLx)[0]
            indWKLred=np.unique(indWKL[locL])
            for k in range(len(indNewY)):
                locR=np.where(indNewY[k]==indRy)[0]
                indWKRred=np.unique(indWKR[locR])

                indWKcur=np.intersect1d(indWKLred,indWKRred)

                for l in range(len(indWKcur)):
                    locLcur=np.where(indWKcur[l]==indWKL)[0]
                    locRcur=np.where(indWKcur[l]==indWKR)[0]

                    mixRed=mixNB[indWKcur[l],:,:]
                    matL=uLTemp[indWKcur[l],indNewX[j],indLy[locLcur]]
                    matR=uRTemp[indWKcur[l],indRx[locRcur],indNewY[k]]
                    UnX[indWKcur[l],indNewX[j],indNewY[k]]+=matmul(matL,mixRed[np.ix_(indLy[locLcur],indRx[locRcur])],matR)

    UnX=np.reshape(UnX,(nPatches,kPatches,curBasis,curBasis))

    return UnX
"""
def calcMaxGap(gapX,wB,kB,tPoints,sPoints):
    NKF=len(sPoints[0])
    NW=len(tPoints[0])

    kPatches=len(kB[0])
    nDim=len(kB)
    suGap=np.zeros(gapX[0].shape[:2]+(1,kPatches),dtype=np.complex_)
    chGap=np.zeros(gapX[0].shape[:2]+(1,kPatches),dtype=np.complex_)
    spGap=np.zeros(gapX[0].shape[:2]+(1,kPatches),dtype=np.complex_)

    wMin=np.argmin(np.abs(tPoints[0]))
    for k in range(NW):
        for j in range(NKF):
            for i in range(nDim):
                fBasWK=np.exp(-1j*kB[i]*sPoints[i][j])
                suGap+=gapX[0][:,:,0,k*NKF+j][:,:,None,None]*fBasWK[None,None,None,:]
                fBasWK=np.exp(-1j*kB[i]*sPoints[i][j])
                chGap+=gapX[1][:,:,0,k*NKF+j][:,:,None,None]*fBasWK[None,None,None,:]
                spGap+=gapX[2][:,:,0,k*NKF+j][:,:,None,None]*fBasWK[None,None,None,:]

    wZero=np.argmin(np.abs(wB))
    chGapF=chGap[wZero,:,0,:]
    spGapF=spGap[wZero,:,0,:]
    suGapF=suGap[wZero,:,0,:]

    locCH=np.argmax(np.sum(np.abs(chGapF),axis=1))
    locSP=np.argmax(np.sum(np.abs(spGapF),axis=1))
    locSU=np.argmax(np.sum(np.abs(suGapF),axis=1))

    return suGapF[locSU,:],chGapF[locCH,:],spGapF[locSP,:],locSU,locCH,locSP

#@jit(nopython=True)
def relSpecWght1D(uVal,tol):
    """Returns the indices of uVal that add up to a tol portion of uVal"""
    totWght=np.sum(uVal)
    nIndex=np.argsort(uVal)[::-1]
    totaL=0.0
    i=0
    if totWght==0:
        i=1
    else:
        uLen=len(uVal)
        for i in range(uLen):
            totaL+=uVal[nIndex[i]]/totWght
            if np.abs(totaL)>tol:
                break

        i=i+1
        #while len(nIndex[:i])<0.1*len(nIndex):
        #    i=i+1
    nIndex=nIndex[:i]
    return nIndex

def setMu(xDop,beTa,maxW,eV):
    nB=eV.shape[1]
    N=eV.shape[0]
    gMax=0
    for i in range(nB):
        gMax=max([gMax,np.abs(eV[:,i]).max()])
    mMax=gMax
    mMin=-gMax

    wF,wFG=padeWeights(maxW,beTa)
    wFX=np.append(-wF[::-1],wF)
    wFG=np.append(wFG[::-1],wFG)
    def occpFunc(xDop,mU):
        fCur=-xDop
        for i in range(nB):
            gProp=1/(1j*wFX[:,None]-eV[:,i][None,:]+mU)
            fCur+=-(2.0/(beTa*N))*np.sum(wFG[:,None]*gProp)
        return fCur

    fMax=occpFunc(xDop,mMax)
    fMin=occpFunc(xDop,mMin)

    toL=0.004
    while np.abs(fMax)>toL or np.abs(fMin)>toL:
        mNew=(mMax-mMin)*np.random.rand(1)+mMin
        fNew=occpFunc(xDop,mNew)

        if np.sign(fNew)!=np.sign(fMin):
            mMax,fMax=mNew,fNew

        else:
            mMin,fMin=mNew,fNew

    if np.abs(mMax-mMin)>toL:
        sl=(fMax-fMin)/(mMax-mMin)
        bl=fMax-sl*mMax
        mU=-bl/sl
    else:
        mU=0.5*(mMin+mMax)
    return mU
#@jit(nopython=True)
def spectralWeight(uX,toL):
    shape=uX.shape
    ind1=np.reshape(np.tile(np.arange(shape[0])[:,None,None],(1,shape[1],shape[2])),uX.size)
    ind2=np.reshape(np.tile(np.arange(shape[1])[None,:,None],(shape[0],1,shape[2])),uX.size)
    ind3=np.reshape(np.tile(np.arange(shape[2])[None,None,:],(shape[0],shape[1],1)),uX.size)

    ind=relSpecWght1D(np.reshape(uX,uX.size),toL)
    return ind1[ind],ind2[ind],ind3[ind]
def relSpecWghtU(uArr,tol):
    """Returns the indices of uVal that add up to a tol portion of uVal"""

    uArrWght=np.sum(np.conj(uArr)*uArr,axis=(0,1))
    curBasis=uArrWght.shape[1]
    uArrWght=np.reshape(uArrWght,(curBasis*curBasis))
    totWght=np.sum(uArrWght)

    if totWght==0:
        nIndex=np.zeros(1,dtype=int)
        i=1
    else:
        nIndex=np.argsort(uArrWght)[::-1]
        totaL=0
        for i in range(curBasis*curBasis):
            totaL+=uArrWght[nIndex[i]]/totWght
            if np.abs(totaL)>tol:
                break
    indX=np.floor(nIndex[:i]/curBasis).astype(int)
    indY=np.mod(nIndex[:i],curBasis).astype(int)


    return indX,indY

#@jit(nopython=True)
def findUniquePoints(tFull,deltaTF):
    """Repeated frequency basis functions in the exchange propagator"""

    countIndex=np.arange(len(tFull))
    tKr=np.zeros(0)
    deltaTr=np.zeros(0)
    itK=[]

    while len(countIndex)>0:
        iC=countIndex[0]
        indCur1=np.where(tFull[iC]==tFull)[0]
        indCur2=np.where(deltaTF[iC]==deltaTF)[0]
        indCur=np.intersect1d(indCur1,indCur2)

        itK.append(indCur)
        tKr=np.append(tKr,tFull[iC])
        deltaTr=np.append(deltaTr,deltaTF[iC])
        indTemp=np.zeros(len(indCur),dtype=int)
        for i,indeX in enumerate(indCur):
            indTemp[i]=np.where(indeX==countIndex)[0]

        countIndex=np.delete(countIndex,indTemp)

    return tKr,itK

def locInArray(tNew,deltaTF,tFull,indFull):
    deltaTFred=np.zeros(len(tFull))
    for i in range(len(tFull)):
        deltaTFred[i]=deltaTF[indFull[i][0]]

    tLoc=np.zeros(len(tNew),dtype=int)
    for i in range(len(tNew)):
        locCur1=np.where(tNew[i]==tFull)[0]
        locCur2=np.where(deltaTF[i]==deltaTFred)[0]

        locCur=np.intersect1d(locCur1,locCur2)

        tLoc[i]=locCur
    return tLoc

@jit(nopython=True)
def locIndex(ind,indFull):
    """Finds indices for location of ind in indFull"""
    nLoc=np.zeros(len(ind),dtype=np.int32)
    for i in range(len(ind)):
        nLoc[i]=np.where(ind[i]==indFull)[0][0]

    return nLoc

@jit(nopython=True)
def calcMomProjection(uX,momL,momR):
    uShape=(momL.shape[2],momL.shape[2])
    NKF=momL.shape[2]
    uR=np.zeros(uShape,dtype=np.complex_)

    for i in range(NKF):
        for j in range(NKF):
            momCur=momL[:,:,i]*momR[:,:,j]
            for k in range(uShape[0]):
                uR[i,j]=np.sum(uX*momCur)

    return uR

@jit(nopython=True)
def calcFreqSumSS(wTempX,propCurLA,propCurRA,propCurLB,propCurRB,indexCur,indexCurE):
    nB=propCurLA.shape[3]
    nPatches=propCurLA.shape[0]
    kPatches=propCurLA.shape[2]
    wIndLen=wTempX.shape[2]
    kIndLen=indexCurE.shape[1]
    propX=np.zeros((nPatches,kPatches,nB*nB,wIndLen,kIndLen),dtype=np.complex_)
    for i in range(nB):
        for j in range(nB):
            for k in range(kPatches):
                for l in range(kIndLen):
                    propCur=propCurLA[:,:,indexCur[k],i]*propCurRA[:,:,indexCurE[k,l],j]+\
                        propCurLB[:,:,indexCur[k],j]*propCurRB[:,:,indexCurE[k,l],i]
                    for m in range(wIndLen):
                        propX[:,k,i*nB+j,m,l]=np.sum(propCur*wTempX[:,:,m],axis=1)

    return propX

@jit(nopython=True)
def calcFreqSumSF(wTempX,propCurLA,propCurRA,indexCur,indexCurE):
    nB=propCurLA.shape[3]
    nPatches=propCurLA.shape[0]
    kPatches=propCurLA.shape[2]
    wIndLen=wTempX.shape[2]
    kIndLen=indexCurE.shape[1]

    propX=np.zeros((nPatches,kPatches,nB*nB,wIndLen,kIndLen),dtype=np.complex_)
    for i in range(nB):
        for j in range(nB):
            for k in range(kPatches):
                for l in range(kIndLen):
                    propCur=propCurLA[:,:,indexCur[k],i]*propCurRA[:,:,indexCurE[k,l],j]
                    for m in range(wIndLen):
                        propX[:,k,i*nB+j,m,l]=np.sum(propCur*wTempX[:,:,m],axis=1)

    return propX

@jit(nopython=True)
def linInterpN(xI,yI,xQ):
    y1=yI.shape[1]
    yQ = np.zeros((len(xQ),y1),dtype=np.complex_)
    for i in range(y1):
        yQ[:,i]=np.interp(xQ,xI,yI[:,i].real)
        yQ[:,i]+=1j*np.interp(xQ,xI,yI[:,i].imag)

    return yQ

def trueInd(sList,N1D,nDim):
    tInd=np.zeros(len(sList[0]),dtype=int)
    for i in range(nDim):
        tInd+=np.mod(sList[i],N1D)*(N1D**(nDim-1-i))
    return tInd

def indexOfMomenta(kIn,N1D):
    nDim=len(kIn)
    momFac=N1D/(2*np.pi)

    kQi=np.zeros(len(kIn[0]),dtype=int)
    for i in range(nDim):
        kQc=np.mod((momFac*kIn[i]).astype(int)+int(N1D/2-1),N1D).astype(int)
        kQi+=kQc*(N1D**(nDim-1-i))

    return kQi

def genMomPoints(N1D,nDim):
    momFac=(2*np.pi/float(N1D))
    kF=momFac*np.arange(-N1D/2+1,N1D/2+1,1)

    kCur=[np.zeros(N1D**nDim) for i in range(nDim)]
    for i in range(nDim):
        kCurV=np.zeros((N1D,)*nDim)
        for j in range(N1D):
            kCurV[j]=kF[j]
        kCurV=np.swapaxes(kCurV,0,i)
        kCur[i]=np.reshape(kCurV,N1D**nDim)

    return kCur

def printBar(steP,current):
    """A status bar for duration of an fRG implementation."""
    width=20
    per=int(current*100)
    curWidth=int(current*width)
    remWidth=width-curWidth
    sys.stdout.write('Step: '+str(steP)+\
                         ' [%s]'%('#'*curWidth+ ' '*remWidth)+\
                         str(per)+'%\r')
    sys.stdout.flush()
def genBandIndicies(i,nB):
    i1=np.floor(i/nB**3).astype(int)
    iRem=np.mod(i,nB**3)

    i2=np.floor(iRem/nB**2).astype(int)
    iRem=np.mod(iRem,nB**2)

    i3=np.floor(iRem/nB).astype(int)
    iRem=np.mod(iRem,nB)

    return [i1,i2,i3,iRem]
def genBandLabel(iList,nB):
    return iList[0]*(nB**3)+iList[1]*(nB**2)+iList[2]*nB+iList[3]

def bandApprox(nB,uBands,appX):
    nBI=np.arange(nB)
    bIred=np.zeros(0,dtype=int)

    for i in range(len(uBands)):
        bandInd=genBandIndicies(uBands[i],nB)
        nBandSum=0

        if bandInd[0]==nBI[-1]:
            nBandSum+=1
        if bandInd[1]==nBI[-1]:
            nBandSum+=1
        if bandInd[2]==nBI[-1]:
            nBandSum+=1
        if bandInd[3]==nBI[-1]:
            nBandSum+=1

        if appX==0:
            if nBandSum>=4:
                bIred=np.append(bIred,i)
        if appX==1:
            if nBandSum>=3:
                bIred=np.append(bIred,i)
        if appX==2:
            if nBandSum>=0:
                bIred=np.append(bIred,i)
    return bIred

def findRedBands(nB):
    ppBands=np.zeros(0,dtype=int)
    phBands=np.zeros(0,dtype=int)
    pheBands=np.zeros(0,dtype=int)
    """
    for i in range(nB):
        for j in range(nB):
            bandLabel=genBandLabel([i,i,j,j],nB)
            ppBands=np.append(ppBands,bandLabel)

            bandLabel=genBandLabel([i,j,j,i],nB)
            phBands=np.append(phBands,bandLabel)

            bandLabel=genBandLabel([j,i,j,i],nB)
            pheBands=np.append(pheBands,bandLabel)
    """
    for i in range(nB):
        for j in range(nB):
            for k in range(nB):
                for l in range(nB):
                    bandLabel=genBandLabel([i,j,k,l],nB)
                    ppBands=np.append(ppBands,bandLabel)

                    bandLabel=genBandLabel([i,j,k,l],nB)
                    phBands=np.append(phBands,bandLabel)

                    bandLabel=genBandLabel([j,i,k,l],nB)
                    pheBands=np.append(pheBands,bandLabel)

    return ppBands,phBands,pheBands
def selfEIndCalc(nB):
    uBandsPP,uBandsPH,uBandsPHE=findRedBands(nB)
    sePPL=[[np.zeros(nB,dtype=int) for i in range(nB)] for j in range(nB)]
    sePPR=[[np.zeros(nB,dtype=int) for i in range(nB)] for j in range(nB)]

    sePHL=[[np.zeros(nB,dtype=int) for i in range(nB)] for j in range(nB)]
    sePHR=[[np.zeros(nB,dtype=int) for i in range(nB)] for j in range(nB)]

    sePHEL=[[np.zeros(nB,dtype=int) for i in range(nB)] for j in range(nB)]
    sePHER=[[np.zeros(nB,dtype=int) for i in range(nB)] for j in range(nB)]

    for j in range(nB):
        for k in range(nB):
            sePPL[j][k]=np.zeros(0,dtype=int)
            sePPR[j][k]=np.zeros(0,dtype=int)

            sePHL[j][k]=np.zeros(0,dtype=int)
            sePHR[j][k]=np.zeros(0,dtype=int)

            sePHEL[j][k]=np.zeros(0,dtype=int)
            sePHER[j][k]=np.zeros(0,dtype=int)
            for i in range(nB):
                sL=genBandLabel([j,i,i,k],nB)
                sePPL[j][k]=np.append(sePPL[j][k],sL)#np.where(sL==ppBands)[0])
                sePHL[j][k]=np.append(sePHL[j][k],sL)#np.where(sL==phBands)[0])
                sePHEL[j][k]=np.append(sePHEL[j][k],sL)#np.where(sL==pheBands)[0])

                sR=genBandLabel([i,j,i,k],nB)
                sePPR[j][k]=np.append(sePPR[j][k],sR)#np.where(sR==ppBands)[0])
                sePHR[j][k]=np.append(sePHR[j][k],sR)#np.where(sR==phBands)[0])
                sePHER[j][k]=np.append(sePHER[j][k],sR)#np.where(sR==pheBands)[0])

    return sePPL,sePPR
def redBandIndCalc(nB):
    uBandsPP,uBandsPH,uBandsPHE=findRedBands(nB)
    bLen=len(uBandsPP)
    ppL=np.zeros((bLen,nB**2),dtype=int)
    ppR=np.zeros((bLen,nB**2),dtype=int)

    phL=np.zeros((bLen,nB**2),dtype=int)
    phR=np.zeros((bLen,nB**2),dtype=int)

    pheL=np.zeros((bLen,nB**2),dtype=int)
    pheR=np.zeros((bLen,nB**2),dtype=int)

    for i in range(bLen):
        inputBandPP=genBandIndicies(uBandsPP[i],nB)
        inputBandPH=genBandIndicies(uBandsPH[i],nB)
        inputBandPHE=genBandIndicies(uBandsPHE[i],nB)
        for j in range(nB):
            for k in range(nB):
                ppl=genBandLabel([inputBandPP[0],inputBandPP[1],j,k],nB)
                ppL[i,j*nB+k]=np.where(ppl==uBandsPP)[0][0]
                ppr=genBandLabel([k,j,inputBandPP[2],inputBandPP[3]],nB)
                ppR[i,j*nB+k]=np.where(ppr==uBandsPP)[0][0]

                phl=genBandLabel([inputBandPH[0],j,k,inputBandPH[3]],nB)
                phL[i,j*nB+k]=np.where(phl==uBandsPH)[0][0]
                phr=genBandLabel([k,inputBandPH[1],inputBandPH[2],j],nB)
                phR[i,j*nB+k]=np.where(phr==uBandsPH)[0][0]

                phel=genBandLabel([j,inputBandPHE[1],k,inputBandPHE[3]],nB)
                pheL[i,j*nB+k]=np.where(phel==uBandsPHE)[0][0]
                pher=genBandLabel([inputBandPHE[0],k,inputBandPHE[2],j],nB)
                pheR[i,j*nB+k]=np.where(pher==uBandsPHE)[0][0]

    return [ppL,ppR],[phL,phR],[pheL,pheR]
def findUniqueBands(nB):
    def exchangePart(iL):
        return [iL[1],iL[0],iL[3],iL[2]]
    def flip(iL):
        return [iL[3],iL[2],iL[1],iL[0]]

    countIndex=np.arange(nB**4)
    symmList=['ex','pos','ex','pos']

    uniqueBands=np.zeros(0,dtype=int)
    iSymm=[]

    iCount=0
    while len(countIndex)>0:
        bandLabel=countIndex[0]
        bandList=np.zeros(0,dtype=int)
        bandList=np.append(bandList,bandLabel)

        curBandLabel=countIndex[0]
        iSCur=[]
        for i in range(len(symmList)):
            if symmList[i] is 'ex':
                newBand=exchangePart(genBandIndicies(curBandLabel,nB))
                newBandLabel=genBandLabel(newBand,nB)
            elif symmList[i] is 'pos':
                newBand=flip(genBandIndicies(curBandLabel,nB))
                newBandLabel=genBandLabel(newBand,nB)
            dLoc=np.where(newBandLabel==bandList)[0]
            if len(dLoc)==0:
                curBandLabel=newBandLabel

                bandList=np.append(bandList,newBandLabel)
                iSCur.append(symmList[i])

        iSymm.append([bandList,iSCur])

        uniqueBands=np.append(uniqueBands,countIndex[0])

        indTemp=np.zeros(len(bandList),dtype=int)
        for i,indeX in enumerate(bandList):
            indTemp[i]=np.where(indeX==countIndex)[0]

        iCount+=1
        countIndex=np.delete(countIndex,indTemp)
    return uniqueBands,iSymm

def bandIndCalc(nB):
    uniqueBands=findUniqueBands(nB)[0]
    lenB=len(uniqueBands)

    ppL=[[np.zeros(lenB,dtype=int) for i in range(nB)] for j in range(nB)]
    ppR=[[np.zeros(lenB,dtype=int) for i in range(nB)] for j in range(nB)]

    phLa=[[np.zeros(lenB,dtype=int) for i in range(nB)] for j in range(nB)]
    phLb=[[np.zeros(lenB,dtype=int) for i in range(nB)] for j in range(nB)]
    phRa=[[np.zeros(lenB,dtype=int) for i in range(nB)] for j in range(nB)]
    phRb=[[np.zeros(lenB,dtype=int) for i in range(nB)] for j in range(nB)]

    pheL=[[np.zeros(lenB,dtype=int) for i in range(nB)] for j in range(nB)]
    pheR=[[np.zeros(lenB,dtype=int) for i in range(nB)] for j in range(nB)]

    for i in range(lenB):
        inputBand=genBandIndicies(uniqueBands[i],nB)
        for j in range(nB):
            for k in range(nB):
                ppl=[inputBand[0],inputBand[1],j,k]
                ppL[j][k][i]=genBandLabel(ppl,nB)
                ppr=[k,j,inputBand[2],inputBand[3]]
                ppR[j][k][i]=genBandLabel(ppr,nB)

                phl=[inputBand[0],j,k,inputBand[3]]
                phLa[j][k][i]=genBandLabel(phl,nB)
                phl=[j,inputBand[0],k,inputBand[3]]
                phLb[j][k][i]=genBandLabel(phl,nB)
                phr=[k,inputBand[1],inputBand[2],j]
                phRa[j][k][i]=genBandLabel(phr,nB)
                phr=[inputBand[1],k,inputBand[2],j]
                phRb[j][k][i]=genBandLabel(phr,nB)

                phel=[j,inputBand[1],k,inputBand[3]]
                pheL[j][k][i]=genBandLabel(phel,nB)
                pher=[inputBand[0],k,inputBand[2],j]
                pheR[j][k][i]=genBandLabel(pher,nB)

    return [ppL,ppR],[phLa,phLb,phRa,phRb],[pheL,pheR]
"""
def selfEIndCalc(nB):
    uBandsPP,uBandsPH,uBandsPHE=findRedBands(nB)
    sEL=[[[] for i in range(nB)] for j in range(nB)]
    sER=[[[] for i in range(nB)] for j in range(nB)]

    for j in range(nB):
        for k in range(nB):
            ppInd1=np.zeros(0,dtype=int)
            ppInd2=np.zeros(0,dtype=int)
            phInd1=np.zeros(0,dtype=int)
            phInd2=np.zeros(0,dtype=int)
            pheInd1=np.zeros(0,dtype=int)
            pheInd2=np.zeros(0,dtype=int)
            for i in range(nB):

                sL=genBandLabel([j,i,i,k],nB)
                ppInd1=np.append(ppInd1,np.where(sL==uBandsPP)[0])
                phInd1=np.append(phInd1,np.where(sL==uBandsPH)[0])
                pheInd1=np.append(pheInd1,np.where(sL==uBandsPHE)[0])

                sR=genBandLabel([i,j,i,k],nB)
                ppInd2=np.append(ppInd2,np.where(sR==uBandsPP)[0])
                phInd2=np.append(phInd2,np.where(sR==uBandsPH)[0])
                pheInd2=np.append(pheInd2,np.where(sR==uBandsPHE)[0])
            sEL[j][k]=[ppInd1,phInd1,pheInd1]
            sER[j][k]=[ppInd2,phInd2,pheInd2]

    return sEL,sER
"""
def fillSpArray(lIndex,rIndex,arraY):
    """Forms a square[lLen by rLen] array from a sparse vertex."""
    lIndexU=np.unique(lIndex)
    rIndexU=np.unique(rIndex)
    fArray=np.zeros((len(lIndexU),len(rIndexU)),dtype=np.complex_)
    for i,lI in enumerate(lIndexU):
        locCur=np.where(lI==lIndex)[0]
        rIndexCur=rIndex[locCur]
        arrayT=arraY[locCur]
        for j in range(len(rIndexCur)):
            rLoc=np.where(rIndexCur[j]==rIndexU)[0]
            fArray[i,rLoc]=arrayT[j]

    return fArray

def squareSymmetries(N1D,nDim):
    kXcur=genMomPoints(N1D,nDim)
    N=N1D**nDim
    countIndex=[True for i in range(N)]
    kDict={}
    for i in range(N):
        if countIndex[i] is True:
            kDict[i]=[i]
            for j in range(nDim):
                kSign=[1]*nDim
                kSign[j]=-1
                kNew=[np.zeros(1) for k in range(nDim)]
                for k in range(nDim):
                    kNew[k][0]=kSign[k]*kXcur[k][i]
                kNewLoc=indexOfMomenta(kNew,N1D)
                kDict[i].append(kNewLoc)
                countIndex[kNewLoc[0]]=False
    count=0
    for i in range(N):
        if countIndex[i] is True:
            count+=1
    print(count)
def genNewSquareMom(kX,syMM):
    """Transforms of a 2D square lattice. Swaps and flips along square axis."""
    toL=10**(-3)
    if syMM is 'swapP':
        kXN=kX[1]
        kYN=kX[0]
    if syMM is 'swapN':
        kXN=np.mod(-kX[1]+np.pi-toL,2*np.pi)-np.pi+toL
        kYN=np.mod(-kX[0]+np.pi-toL,2*np.pi)-np.pi+toL
    elif syMM is 'flipX':
        kXN=np.mod(-kX[0]+np.pi-toL,2*np.pi)-np.pi+toL
        kYN=kX[1]
    elif syMM is 'flipY':
        kXN=kX[0]
        kYN=np.mod(-kX[1]+np.pi-toL,2*np.pi)-np.pi+toL
    return (kXN,kYN)
def squareLatPath(kB):
    ang=np.arctan2(kB[1],kB[0])
    conD1=(ang==0)
    conD2=(kB[1]==0)

    cond=np.logical_and(conD1,conD2)
    kBxC=kB[0][cond]
    kByC=kB[1][cond]
    aSort=np.argsort(kBxC)

    kBxF=kBxC[aSort]
    kByF=kByC[aSort]

    conD1=(ang>0)
    conD2=(ang<(np.pi/4))
    cond=np.logical_and(conD1,conD2)
    cond=np.logical_and(cond,kB[0]==np.pi)
    kBxC=kB[0][cond]
    kByC=kB[1][cond]
    aSort=np.argsort(kByC)

    kBxF=np.append(kBxF,kBxC[aSort])
    kByF=np.append(kByF,kByC[aSort])

    conD1=(ang==np.pi/4)
    conD2=(kB[0]<=np.pi)
    cond=np.logical_and(conD1,conD2)
    kBxC=kB[0][cond]
    kByC=kB[1][cond]
    aSort=np.argsort(kBxC**2+kByC**2)[::-1]

    kBxF=np.append(kBxF,kBxC[aSort])
    kByF=np.append(kByF,kByC[aSort])

    return (kBxF,kByF)

def genUnSquareMom(kX):
    """Generates the unique set of momenta for a square lattice. Along with a list of the
       redundant momenta and symmetry relations."""
    countIndex=np.arange(len(kX[0]))
    imomK=[]
    toL=10**(-6)

    uniqueMomx=np.zeros(0)
    uniqueMomy=np.zeros(0)
    redIndex=np.zeros(0,dtype=int)

    for i in countIndex:
        ang=np.arctan2(kX[1][i],kX[0][i])
        conD1=(ang>=0)
        conD2=(ang<=(0.25*np.pi))
        angleCond=np.logical_and(conD1,conD2)
        if angleCond:
            redIndex=np.append(redIndex,i)
            uniqueMomx=np.append(uniqueMomx,kX[0][i])
            uniqueMomy=np.append(uniqueMomy,kX[1][i])

    for i,indexR in enumerate(redIndex):
        momLoc=np.zeros(0,dtype=int)
        momLoc=np.append(momLoc,indexR)
        kC=(kX[0][momLoc],kX[1][momLoc])

        symmList=['swapP','flipX','swapN','flipY',\
            'swapP','flipX','swapN','flipY']

        symmRed=[]
        for j in symmList:
            kNc=genNewSquareMom(kC,j)
            conD1=np.where(np.abs(kNc[0]-kX[0])<toL)[0]
            conD2=np.where(np.abs(kNc[1]-kX[1])<toL)[0]
            conD=np.intersect1d(conD1,conD2)

            inLoc=np.where(conD[0]==momLoc)[0]
            if len(inLoc)==0:
                momLoc=np.append(momLoc,conD)
                symmRed.append(j)

            kC=kNc
        imomK.append([momLoc,symmRed])

    uniqueMom=(uniqueMomx,uniqueMomy)
    return uniqueMom,imomK

def fScaling(AC):
    kPar=20
    rampF=(AC-1)*(0.5+0.5*np.tanh(kPar*(AC-1)))+1

    nN=4
    rampF=(AC**nN+1)**(1.0/nN)
    return rampF

def derivFScaling(AC):
    kPar=20
    dRampF=(0.5+0.5*np.tanh(kPar*(AC-1)))+0*kPar*0.5*(AC-1)*(1-np.tanh(kPar*(AC-1))**2)

    nN=4
    dRampF=(AC**(nN-1))*((AC**nN+1)**(1.0/nN-1))
    return dRampF

def fillMomVertex(singV,iSymm,N,axis=0):
    """Construsts full momentum dependence of one body vertices."""

    if singV.shape[axis]==N:
        singVe=singV*1
    else:
        singV=np.swapaxes(singV,0,axis)
        singVe=np.zeros((N,)+singV.shape[1:],dtype=np.complex_)
        for i in range(len(iSymm)):
            iS=iSymm[i][0]
            singVe[iS,...]=singV[i,...]

        singV=np.swapaxes(singV,0,axis)
        singVe=np.swapaxes(singVe,0,axis)
    return singVe
def compVertexP(UnX,iSymm,N1D,NW,NKF,sI):
    """Constructs full momentum dependence of two body vertices"""
    nPatches=UnX.shape[0]
    if UnX.shape[1]==(N1D**2):
        UnXf=1*UnX
    else:
        aInd=NW*NKF*NW*NKF
        aIndo=np.arange(aInd)
        UnXo=np.reshape(UnX,UnX.shape[:2]+(aInd,))
        UnXf=np.zeros((nPatches,N1D**2,aInd),dtype=np.complex_)
        for i in range(len(iSymm)):
            iIndex=iSymm[i][0]
            symmList=iSymm[i][1]
            UnXf[:,iIndex[0],:]=UnXo[:,i,:]

            for j in range(len(iIndex)-1):
                newIndex=genSquareIndex(aIndo,N1D,NW,NKF,sI,symmList[:(j+1)])
                newLoc=locIndex(newIndex,aIndo)
                oldLoc=locIndex(aIndo[newLoc],newIndex)
                UnXf[:,iIndex[j+1],oldLoc]=UnXo[:,i,newLoc]

    return np.reshape(UnXf,UnXf.shape[:2]+(NW*NKF,NW*NKF,))
def compVertex(UnX,iSymm,N1D,NW,NKF,sI):
    """Constructs full momentum dependence of two body vertices"""
    nPatches=UnX.shape[0]
    if UnX.shape[1]==(N1D**2):
        UnXf=1*UnX
    else:
        aInd=NW*NKF*NW*NKF
        aIndo=np.arange(aInd)
        UnXo=np.reshape(UnX,UnX.shape[:2]+(aInd,))
        UnXf=np.zeros((nPatches,N1D**2,aInd),dtype=np.complex_)
        for i in range(len(iSymm)):
            iIndex=iSymm[i][0]
            symmList=iSymm[i][1]
            UnXf[:,iIndex[0],:]=UnXo[:,i,:]

            for j in range(len(iIndex)-1):
                newIndex=genSquareIndex(aIndo,N1D,NW,NKF,sI,symmList[:(j+1)])
                newLoc=locIndex(newIndex,aIndo)
                oldLoc=locIndex(aIndo[newLoc],newIndex)
                UnXf[:,iIndex[j+1],oldLoc]=UnX[:,i,newLoc]

    return np.reshape(UnXf,UnX.shape[:2]+(NW*NKF,NW*NKF,))
def genNewSquareIndex(kI,syMM):
    """Spatial indicies corresponding to symmetries of the square lattice"""
    if syMM is 'swapP':
        indKNx=kI[1]
        indKNy=kI[0]

    if syMM is 'swapN':
        indKNx=-kI[1]
        indKNy=-kI[0]

    elif syMM is 'flipX':
        indKNx=-kI[0]
        indKNy=kI[1]

    elif syMM is 'flipY':
        indKNx=kI[0]
        indKNy=-kI[1]

    return indKNx,indKNy
def genSquareIndex(ind,N1D,NW,NKF,sI,syMM):
    """Generates the new indicies corresponding with symmetry transforms for
        sqaure lattice on two body vertices."""
    indKL,indKR,indWL,indWR=indWKlr(ind,NW,NKF)
    indKLNx=sI[0][indKL]
    indKLNy=sI[1][indKL]
    indKRNx=sI[0][indKR]
    indKRNy=sI[1][indKR]

    for i in syMM:
        indKLNx,indKLNy=genNewSquareIndex((indKLNx,indKLNy),i)
        indKRNx,indKRNy=genNewSquareIndex((indKRNx,indKRNy),i)

    sIe=np.mod(sI[0],N1D)*N1D+np.mod(sI[1],N1D)
    newIndKLN=np.mod(indKLNx,N1D)*N1D+np.mod(indKLNy,N1D)
    indKLN=locIndex(newIndKLN,sIe)
    newIndKRN=np.mod(indKRNx,N1D)*N1D+np.mod(indKRNy,N1D)
    indKRN=locIndex(newIndKRN,sIe)
    indN=(indWL*NKF+indKLN)*NW*NKF+\
        (indWR*NKF+indKRN)
    return indN
def genRelatedIndex(ind,N1D,NW,NKF,sI,syMM):
    indKL,indKR,indWL,indWR=indWKlr(ind,NW,NKF)
    indKLNx=sI[0][indKL]
    indKLNy=sI[1][indKL]
    indKRNx=sI[0][indKR]
    indKRNy=sI[1][indKR]

    indKLNfx=np.zeros(0,dtype=int)
    indKLNfx=np.append(indKLNfx,indKLNx)
    indKRNfx=np.zeros(0,dtype=int)
    indKRNfx=np.append(indKRNfx,indKRNx)
    indKLNfy=np.zeros(0,dtype=int)
    indKLNfy=np.append(indKLNfy,indKLNy)
    indKRNfy=np.zeros(0,dtype=int)
    indKRNfy=np.append(indKRNfy,indKRNy)
    for i in syMM:
        indKLNx,indKLNy=genNewSquareIndex((indKLNx,indKLNy),i)
        indKLNfx=np.append(indKLNfx,indKLNx)
        indKLNfy=np.append(indKLNfy,indKLNy)
        indKRNx,indKRNy=genNewSquareIndex((indKRNx,indKRNy),i)
        indKRNfx=np.append(indKRNfx,indKRNx)
        indKRNfy=np.append(indKRNfy,indKRNy)


    sIe=np.mod(sI[0],N1D)*N1D+np.mod(sI[1],N1D)
    newIndKLN=np.mod(indKLNx,N1D)*N1D+np.mod(indKLNy,N1D)
    indKLN=locIndex(newIndKLN,sIe)
    newIndKRN=np.mod(indKRNx,N1D)*N1D+np.mod(indKRNy,N1D)
    indKRN=locIndex(newIndKRN,sIe)
    indN=(indWL*NKF+indKLN)*NW*NKF+\
        (indWR*NKF+indKRN)
    return indN

def reduceVertex(UnX,nMax):
    UnX=np.reshape(UnX,UnX.shape[:2]+(np.prod(UnX.shape[2:]),))
    aNew=relSpecWght(UnX,np.zeros(1,dtype=int),1,axis=(2,),sumAxis=(0,1))

    UnX=UnX[:,:,aNew[:nMax]]
    indR=aNew[:nMax]
    aSort=np.argsort(indR)

    return UnX[:,:,aSort],indR[aSort]

def expandVertex(UnX,indC,nShape):
    UnXe=np.zeros(UnX.shape[:2]+(nShape*nShape,),dtype=np.complex_)
    UnXe[:,:,indC]=UnX
    return np.reshape(UnXe,UnX.shape[:2]+(nShape,nShape,))


def uniqueDPoints(sI,NKF,nDim):
    """Repeated momenta basis functions in the exchange propagator
        on the square lattice"""
    countIndex=np.arange(len(sI[0]))
    imomK=[]
    sIn=[np.zeros(0,dtype=int) for i in range(nDim)]
    while len(countIndex)>0:
        iC=countIndex[0]
        indCur=np.where(sI[0][iC]==sI[0])[0]
        for i in range(1,nDim):
            conDcur=np.where(sI[i][iC]==sI[i])[0]
            indCur=np.intersect1d(conDcur,indCur)
        for i in range(nDim):
            sIn[i]=np.append(sIn[i],sI[i][iC])
        imomK.append([np.floor(indCur/NKF).astype(int),np.mod(indCur,NKF)])

        indTemp=np.zeros(len(indCur),dtype=int)
        for i,indeX in enumerate(indCur):
            indTemp[i]=np.where(indeX==countIndex)[0]

        countIndex=np.delete(countIndex,indTemp)
    return sIn,imomK
def uniquePoints(tK,deltaT):
    """Repeated frequency basis functions in the exchange propagator"""
    tK=np.reshape(np.round(tK,8),tK.size)
    NW=len(deltaT)
    deltaTE=np.round(np.tile(deltaT,NW)*np.repeat(deltaT,NW),8)

    countIndex=np.arange(len(tK))
    tKr=np.zeros(0)
    deltaTr=np.zeros(0)
    itK=[]

    while len(countIndex)>0:
        iC=countIndex[0]
        indCur1=np.where(tK[iC]==tK)[0]
        indCur2=np.where(deltaTE[iC]==deltaTE)[0]
        indCur=np.intersect1d(indCur1,indCur2)

        itK.append([np.floor(indCur/NW).astype(int),np.mod(indCur,NW)])
        tKr=np.append(tKr,tK[iC])
        deltaTr=np.append(deltaTr,deltaTE[iC])
        indTemp=np.zeros(len(indCur),dtype=int)
        for i,indeX in enumerate(indCur):
            indTemp[i]=np.where(indeX==countIndex)[0]

        countIndex=np.delete(countIndex,indTemp)

    return tKr,itK
def indXY(ind,N):
    """Splitting indicies"""
    indX=np.floor(ind/N).astype(int)
    indY=np.mod(ind,N).astype(int)

    return indX,indY
def indWKlr(ind,NW,NKF):
    """Labels for frequency and momenta basis functions that make up the
    two body vertex"""
    indKL=np.mod(np.floor(ind/(NW*NKF)),NKF).astype(int)
    indKR=np.mod(np.mod(ind,NW*NKF),NKF).astype(int)

    indWL=np.floor(np.floor(ind/(NW*NKF))/NKF).astype(int)
    indWR=np.floor(np.mod(ind,NW*NKF)/NKF).astype(int)

    return indKL,indKR,indWL,indWR

def interpMomentum(UnX,nC,nD,axis=0):
    """Upscaling or downscaling the 2D square lattice via a linear interpolator"""
    UnXn=np.swapaxes(UnX,0,axis)

    kBe=(2*np.pi/nC)*np.arange(-nC/2,nC/2+1)
    kBn=(2*np.pi/nD)*np.arange(-nD/2+1,nD/2+1)

    UnXn=np.swapaxes(UnX,0,axis)
    UnXn=np.reshape(UnXn,(nC,nC)+UnXn.shape[1:])

    UnXn=np.concatenate((UnXn[-1:,...],UnXn),axis=0)
    UnXn=np.concatenate((UnXn[:,-1:,...],UnXn),axis=1)

    UnXi=linInterpO(kBe,UnXn,kBn,axis=0)
    UnXi=linInterpO(kBe,UnXi,kBn,axis=1)

    UnXi=np.reshape(UnXi,(nD**2,)+UnXn.shape[2:])

    UnXi=np.swapaxes(UnXi,0,axis)
    return UnXi

def expandAdd(arrL,axis=0):
    """Adds multiple sparse arrays with respect to their indicies to return a sparse sum"""
    arrF,indF=expandAdd2(arrL[0],arrL[1],axis=axis)
    for i in range(2,len(arrL)):
        arrF,indF=expandAdd2((arrF,indF),arrL[i])

    return (arrF,indF)
def expandAdd2(arr1,arr2,axis=0):
    """Adds two sparse arrays according to their index and returns a sparse array
    sorted according to the expanded index"""
    arrInt=np.intersect1d(arr1[1],arr2[1])

    elem12a=np.in1d(arr1[1],arrInt)
    elem12b=np.in1d(arr2[1],arrInt)

    elem1=np.invert(elem12a)
    elem2=np.invert(elem12b)

    indF=np.concatenate((arr1[1][elem1],arr2[1][elem2],arr1[1][elem12a]),axis=0)

    arr1N=np.swapaxes(arr1[0],0,axis)
    arr2N=np.swapaxes(arr2[0],0,axis)

    arrF1=arr1N[elem1,...]
    arrF2=arr2N[elem2,...]
    arrFI=arr1N[elem12a,...]
    arrFI+=arr2N[elem12b,...]

    arrF=np.concatenate((arrF1,arrF2,arrFI),axis=0)
    indSort=np.argsort(indF)

    arrF=arrF[indSort,...]
    indF=indF[indSort]

    arrF=np.swapaxes(arrF,0,axis)
    return (arrF,indF)

def calcScale(xV,yV,cScaleS):
    """A simple estimate (via Newton's Method) of scale for an algebraic map."""
    diff=cScaleS
    tol=0.001

    deltaW=min(xV[1::])/2.0
    wPrev=xV[1:]-deltaW
    wPrev=np.append(xV[0],wPrev)

    uPrev=np.interp(wPrev,xV,yV)

    wFor=xV[:-1]+deltaW
    wFor=np.append(wFor,xV[-1])
    uFor=np.interp(wFor,xV,yV)

    uDer=(uFor-uPrev)/(wFor-wPrev)
    uDer=np.append(uDer,np.zeros(1))

    while diff>tol:

        zFull=np.linspace(0,1,100)
        (zFullX,gFullX)=gaussianInt(zFull,2)

        zPos=cScaleS*(zFullX+1)/(1-zFullX)
        zNeg=cScaleS*(-zFullX+1)/(1+zFullX)

        wMax=zPos.max()
        xVa=np.append(xV,wMax)

        uPos=np.interp(zPos,xVa,uDer)
        uNeg=np.interp(zNeg,xVa,uDer)

        uValu=np.append(yV,np.zeros(1))
        uSPos=np.interp(zPos,xVa,uValu)
        uSNeg=np.interp(zNeg,xVa,uValu)

        jaCBN=2.0/(1.0+zFullX)**2
        jaCBP=2.0/(1.0-zFullX)**2

        signU=np.sign(np.sum(gFullX*(uSPos*jaCBP*cScaleS-uSNeg*jaCBN*cScaleS)))

        fXv=np.abs(np.sum(gFullX*(jaCBP*uSPos*cScaleS-jaCBN*uSNeg*cScaleS)))
        fDerv=signU*np.sum(gFullX*(cScaleS*jaCBP*uPos*(1+zFullX)/(1-zFullX)-\
                                 cScaleS*jaCBN*uNeg*(-zFullX+1)/(1+zFullX)+\
                                       jaCBP*uSPos-jaCBN*uSNeg))

        cScale=cScaleS-fXv/(fDerv+tol/2)
        diff=np.abs(cScale-cScaleS)

        cScaleS=cScale
    return cScale
def genXDPoints(N1D,NK=-1,nDim=1):
    """Generates reduced basis modes for the square lattice."""
    xAvgO=np.arange(-N1D/2+1,N1D/2+1)
    lInd=[np.zeros(N1D*nDim,dtype=int) for i in range(nDim)]
    dX2=np.zeros(N1D**nDim)
    for i in range(nDim):
        lCur=np.zeros((N1D,)*nDim,dtype=int)
        for j in range(N1D):
            lCur[j]=xAvgO[j]
        lCur=np.swapaxes(lCur,0,i)
        lInd[i]=np.reshape(lCur,N1D**nDim)
        dX2+=lInd[i]**2
    dX=np.sqrt(dX2)
    dXU=np.unique(dX)

    if NK==(-1) or NK>(N1D/2):
        kInd=np.arange(len(dX))
    else:
        kInd=dX<=dXU[NK]

    for i in range(nDim):
        lInd[i]=lInd[i][kInd]

    return lInd

def relSpecWght(uArr,zInd,toL,axis=(0,),sumAxis=None):
    """Calculates the spectral weight along a particular axis and truncates up
    to the given tolerance. In case of extra axeses truncation is based on the
    forbenius norm of sumAxis dimensions."""
    if sumAxis is None:
        uArrWght=uArr*np.conj(uArr)
    else:
        uArrWght=np.sum(np.conj(uArr)*uArr,axis=sumAxis)

    if len(axis)==1:
        totWght=np.sum(uArrWght)

        if totWght==0:
            iIndex=np.zeros(1,dtype=int)+zInd
        else:
            iIndex=relSpecWght1D(uArrWght,toL)
    else:
        totWght=np.sum(uArrWght)

        if totWght==0:
            iIndexX=np.zeros(1,dtype=int)+zInd
            iIndexY=np.zeros(1,dtype=int)+zInd

            iIndex=(iIndexX,iIndexY)
        else:
            uShape=uArrWght.shape
            iIndexX=np.repeat(np.arange(uShape[0]),uShape[1])
            iIndexY=np.tile(np.arange(uShape[1]),uShape[0])
            uArrWght=uArrWght.reshape(uArrWght.size)

            nIndex=relSpecWght1D(uArrWght,toL)
            iIndex=(iIndexX[nIndex],iIndexY[nIndex])
    return iIndex

def linInterp2(xI,yI,xQ):
    """A vectorized linear interpolator along 0 dimension of a 3-d array"""
    yL=yI.shape[1]

    yI=np.concatenate((yI[0:1,...],yI,yI[-1::,...]),axis=0)
    xEnd=max([max(abs(xQ)),max(abs(xI))])+0.1
    xI=np.append(-xEnd,np.append(xI,xEnd))
    xQI=np.searchsorted(xI,xQ)

    y1=yI[xQI-1,:]
    y2=yI[xQI,:]

    x1=xI[xQI-1]
    x2=xI[xQI]
    mulT=(xQ-x1)/(x1-x2)

    yQ=(y1-y2)*mulT[:,None]+y1

    return yQ

def linInterpO(xI,yI,xQ,axis=0,left=None,right=None):
    """Linear interpolation along any axis with extrapolation beyond grid fixed
       to end points"""
    yShape=yI.shape
    if len(yShape)==1:
        yQ=np.zeros(len(xQ),dtype=np.complex_)
        yQ+=np.interp(xQ,xI,yI.real,left,right)
        yQ+=1j*np.interp(xQ,xI,yI.imag,left,right)
    else:
        yO=yShape[axis]
        yShape=list(yShape)
        yShape.pop(axis)
        yF=np.asarray(yShape).prod()

        yI=np.swapaxes(yI,0,axis)
        yI=yI.reshape((yO,yF))

        xEnd=max([max(abs(xQ)),max(abs(xI))])+0.1
        if left!=None:
            yI=np.append(yI[0:1,...]*left,yI,axis=0)
            xI=np.append(-xEnd,xI)

        if right!=None:
            yI=np.append(yI,yI[0:1,...]*right,axis=0)
            xI=np.append(xI,xEnd)

        yQ = linInterpN(xI,yI,xQ)

        yQ=yQ.reshape((len(xQ),)+tuple(yShape))
        yQ=np.swapaxes(yQ,0,axis)

    return yQ
def iffTransformD(uV,N1D,nDim,axis=0):
    """Shifted numpy inverse fft along a specified axis"""
    uShape=uV.shape

    nL=uShape[axis]

    curAxes=(0,)
    for i in range(1,nDim):
        curAxes=curAxes+(i,)

    shifT=np.arange(N1D)*((2*np.pi)/N1D)*(N1D/2-1)
    phaseShift=np.ones(N1D**nDim,dtype=np.complex_)
    for i in range(nDim):
        shiftCur=np.zeros((N1D,)*nDim)
        for j in range(N1D):
            shiftCur[j]=shifT[j]
        shiftCur=np.reshape(np.swapaxes(shiftCur,0,i),N1D**nDim)
        phaseShift=phaseShift*np.exp(1j*shiftCur)

    if len(uShape)==1:
        uVn=phaseShift*uV
        uVn=np.reshape(uVn,(N1D,)*nDim)
        uVn=np.fft.fftn(uVn,axes=curAxes)
        uVn=np.reshape(uVn,N1D**nDim)

    else:
        uV=np.swapaxes(uV,0,axis)
        eP=np.reshape(phaseShift,((N1D**nDim,)+(1,)*(len(uShape)-1)))
        uVn=eP*uV
        uVn=np.reshape(uVn,(N1D,)*nDim+uV.shape[1:])
        uVn=np.fft.fftn(uVn,axes=curAxes)
        uVn=np.reshape(uVn,(N1D**nDim,)+uV.shape[1:])
        uVn=np.swapaxes(uVn,0,axis)

    return uVn
def ffTransformD(uV,N1D,nDim,axis=0):
    """Shifted numpy fft along a specified axis"""
    uShape=uV.shape

    nL=uShape[axis]

    curAxes=(0,)
    for i in range(1,nDim):
        curAxes=curAxes+(i,)

    shifT=np.arange(N1D)*((2*np.pi)/N1D)*(-N1D/2+1)
    phaseShift=np.ones(N1D**nDim,dtype=np.complex_)
    for i in range(nDim):
        shiftCur=np.zeros((N1D,)*nDim)
        for j in range(N1D):
            shiftCur[j]=shifT[j]
        shiftCur=np.reshape(np.swapaxes(shiftCur,0,i),N1D**nDim)
        phaseShift=phaseShift*np.exp(1j*shiftCur)

    if len(uShape)==1:
        uVn=np.reshape(uV,(N1D,)*nDim)
        uVn=np.fft.ifftn(uVn,axes=curAxes)
        uVn=np.reshape(uVn,N1D**nDim)
        uVn=phaseShift*uVn

    else:
        uV=np.swapaxes(uV,0,axis)

        uVn=np.reshape(uV,(N1D,)*nDim+uV.shape[1:])
        uVn=np.fft.ifftn(uVn,axes=curAxes)

        uVn=np.reshape(uVn,(N1D**nDim,)+uV.shape[1:])
        eP=np.reshape(phaseShift,((N1D**nDim,)+(1,)*(len(uShape)-1)))
        uVn=eP*uVn
        uVn=np.swapaxes(uVn,0,axis)

    return uVn

def padeWeights(wMax,beTa):
    """Okazaki matsubara frequencies and weights for fermionic matsubara sums"""
    tol=0.01
    xV=beTa*wMax*2
    eRR=np.tanh(xV/2.0)
    iMax=10
    while eRR>tol:
        bB=np.zeros((iMax,iMax))
        for i in range(iMax-1):
            bB[i,i+1]=1.0/(2.0*np.sqrt((2.0*(i+1)-1.0)*(2.0*(i+1)+1.0)))
            bB[i+1,i]=bB[i,i+1]

        bEig,aV=np.linalg.eig(bB)
        freQ=1.0/bEig
        wghT=np.zeros(iMax)
        for i in range(iMax):
            wghT[i]=(aV[0,i]**2)/(4*(bEig[i]**2))

            fW=zip(np.abs(freQ),freQ,wghT)
            fW=sorted(fW)
        for i in range(iMax):
            (a,freQ[i],wghT[i])=fW[i]
        est=0.0
        for i in range(iMax):
            est+=(2*wghT[i]*xV)/(xV**2+freQ[i]**2)
        eRR=np.abs(est-np.tanh(xV/2.0))

        iMax+=2
    return np.abs(freQ[::2])/beTa,wghT[::2]

def fExpSinc(wX,tCur,dTCur):
    """Sinc function corresponding to a box centered at tCur with width dTCur"""
    return np.exp(-1j*tCur*wX)*np.sinc((0.5*wX*dTCur)/np.pi)
def hermiteExpansion(wV,j):
    """Orthogonal Hermite functions."""
    lInJ=np.zeros(j+1)
    lInJ[j]=1

    return (1/np.sqrt(np.sqrt(np.pi)*(2**j)*np.math.factorial(j)))*np.exp(-0.5*wV**2)*np.polynomial.hermite.hermval(wV,lInJ)
def freqExpansion(zV,j):
    """Orthonormal Legendre functions."""
    lInJ=np.zeros(j+1)
    lInJ[j]=1

    return np.sqrt((2*j+1)/2.0)*np.polynomial.legendre.legval(zV,lInJ)

def momLegExpansion(kV,j):
    lInJ=np.zeros(j+1)
    lInJ[j]=1

    kV=np.mod(kV+np.pi,2*np.pi)-np.pi

    preFac=np.sqrt((2*j+1)/(2.0*np.pi))
    return preFac*np.polynomial.legendre.legval(kV/np.pi,lInJ)

def gaussianInt(xGrid,nG):
    """Gauss-Legendre points and weights for smooth integrals."""
    lIn=np.zeros(nG+1)
    lIn[nG]=1

    xI=np.polynomial.legendre.legroots(lIn)
    dlIn=np.polynomial.legendre.legder(lIn)
    dLi=np.polynomial.legendre.legval(xI,dlIn)
    wI=2.0/((1-xI**2)*(dLi**2))

    xG=np.zeros(nG*(len(xGrid)-1))
    wG=np.zeros(nG*(len(xGrid)-1))

    for i in range(len(xGrid)-1):
        xG[(i*nG):((i+1)*nG)]=(0.5*(xGrid[i+1]-xGrid[i])*xI+\
                                   0.5*(xGrid[i+1]+xGrid[i]))
        wG[(i*nG):((i+1)*nG)]=0.5*(xGrid[i+1]-xGrid[i])*wI

    return xG,wG

def tBetaShiftO(tF,beTa):
    """Shifts imaginary time to the interval 0 to beTa. Returns the shifted
    and the sign"""
    tSign=(-1)**np.floor(tF/beTa)
    tFn=tF-beTa*np.floor(tF/beTa)

    return tFn,tSign
def tBetaShift(tF,beTa):
    """Shifts imaginary time to the interval -beTa/2 to beTa/2.Returns shifted
    time and sign"""
    toL=10**(-5)
    tSign=(-1)**np.floor((tF-toL)/beTa)
    tFn=tF-beTa*np.floor((tF-toL)/beTa)

    tSign=tSign*((-1)**np.floor(2*(tFn-toL)/beTa))
    tFn=tFn-beTa*np.floor(2*(tFn-toL)/beTa)

    return tFn,tSign
def filtSinc(wFX,deltaT):
    """A filtered Sinc Function. Currently limited to a box Car filter."""
    wCut=4*np.pi/deltaT
    filT=np.zeros(wFX.shape)+0

    filT1=np.cos((np.pi*wFX)/(2*wCut))**1
    filT[np.abs(wFX)<wCut]=filT1[np.abs(wFX)<wCut]
    fSinc=np.sinc((0.5*wFX*deltaT)/np.pi)
    return fSinc

def freqPoints(beTa,wMax,nPoints):
    """Logarthmic set of fermionic and bosonic matsubara frequencies."""
    nMax=np.floor(0.5*((wMax*beTa/np.pi)-1))

    wTF=(np.pi/beTa)*(2*np.arange(0,nMax,1)+1)
    wTB=(np.pi/beTa)*2*np.arange(0,nMax,1)

    wIndex=np.logspace(0,np.log(len(wTF))/np.log(10),nPoints)-1
    wIndex=np.unique(wIndex.astype(int))

    wF=wTF[wIndex]
    wB=wTB[wIndex]

    i=1
    while len(wF)<nPoints:
        if len(wF)<len(wTF):
            wA=wIndex+i
            wA=wA[wA<len(wTF)]

            i+=1
            lEnd=min([len(wA),nPoints-len(wF)])
            wF=np.unique(np.append(wF,wTF[wA[:lEnd]]))
        else:
            wA=(np.pi/beTa)*(2*np.arange(nMax,nMax+nPoints-len(wF),1)+1)
            wF=np.append(wF,wA)

    i=1
    while len(wB)<nPoints:
        if len(wB)<len(wTB):
            wA=wIndex+i
            wA=wA[wA<len(wTB)]

            i+=1
            lEnd=min([len(wA),nPoints-len(wB)])
            wB=np.unique(np.append(wB,wTB[wA[:lEnd]]))
        else:
            wA=(np.pi/beTa)*2*np.arange(nMax,nMax+nPoints-len(wB),1)
            wB=np.append(wB,wA)


    return wF[wF<=wMax],wB[wB<=wMax]
def forMap(wV,cScale):
    """Algebraic map to finite domain."""
    return wV/np.sqrt(wV**2+cScale)
def backMap(zV,cScale):
    """Inverse algebraic map to unbounded domain."""
    return zV/np.sqrt(cScale-zV**2)

def aScale(lC,A0):
    return A0*np.exp(-lC)

def litim0(wQ,AC):
    """Litim Regulator for frequency domain."""
    stepR=0.5*(np.sign(AC-np.abs(wQ))+1)
    return 1j*(np.sign(wQ)*AC-wQ)*stepR

def dLitim0(wQ,AC):
    """Scale derivative of Litim regulator."""
    stepR=0.5*(np.sign(AC-np.abs(wQ))+1)
    return 1j*np.sign(wQ)*stepR

def additiveR0(wQ,AC):
    """Smooth additive regualtor for frequency domain."""
    return 1j*(np.sign(wQ)*np.sqrt(wQ**2+AC**2)-wQ)

def dAdditiveR0(wQ,AC):
    """Scale derivative of additive regulator."""
    return (1j*np.sign(wQ)*AC)/np.sqrt(wQ**2+AC**2)

def sharpR(wQ,AC):
    """A smoothed hard regulator for frequency domain."""
    kSm=1
    return (1-np.exp(-(np.abs(wQ)/AC)**kSm))

def dSharpR(wQ,AC):
    """Scale derivative of sharp regulator."""
    kSm=1
    return ((kSm*np.abs(wQ)**kSm)/AC**(kSm+1))*np.exp(-(np.abs(wQ)/AC)**kSm)

def softR(wQ,AC):
    """A soft regulator for frequency domain."""
    return (wQ**2+AC**2)/(wQ**2)

def dSoftR(wQ,AC):
    """Scale derivative of smooth regulator."""
    return 2*AC/(wQ**2)
