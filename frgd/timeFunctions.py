import numpy as np
import copy
import frgd.auxFunctions as auxF

def iffTransformW(uV,Ns,axis=0,bos=0):
    uShape=uV.shape
    shifT=np.arange(Ns)*((2*np.pi)/Ns)*(Ns/2-1)
    phaseShift=np.exp(1j*shifT+1j*bos*np.arange(Ns)*(np.pi/Ns))

    if len(uShape)==1:
        uVn=phaseShift*uV
        uVn=np.fft.fft(uVn)

    else:
        uV=np.swapaxes(uV,0,axis)
        eP=np.reshape(phaseShift,((Ns,)+(1,)*(len(uShape)-1)))
        uVn=eP*uV
        uVn=np.fft.fft(uVn,axis=0)
        uVn=np.swapaxes(uVn,0,axis)

    return uVn

def ffTransformW(uV,Ns,axis=0,bos=0):
    uShape=uV.shape
    shifT=np.arange(Ns)*((2*np.pi)/Ns)*(-Ns/2+1)
    phaseShift=np.exp(1j*shifT-1j*bos*np.arange(Ns)*(np.pi/Ns))

    if len(uShape)==1:
        uVn=np.fft.ifft(uV)
        uVn=phaseShift*uVn
    else:
        uV=np.swapaxes(uV,0,axis)
        uVn=np.fft.ifft(uV,axis=0)
        eP=np.reshape(phaseShift,((Ns,)+(1,)*(len(uShape)-1)))
        uVn=eP*uVn
        uVn=np.swapaxes(uVn,0,axis)
    return uVn

def genTPoints(tMin,beTa,NW,dT=None):
    if dT==None:
        eTa=(0.5*beTa-tMin)/float(NW/2-1)
    else:
        dTmin=2*(0.5*beTa-2*tMin)/NW
        eTa=min([dTmin,dT])

    tB=np.arange(tMin,eTa*(NW/2),eTa)
    tB=np.append(tB,0.5*beTa)
    tB=np.append(-tB[::-1],tB)
    tAvg=0.5*(tB[:-1]+tB[1:])
    deltaT=tB[1:]-tB[:-1]

    return tAvg,deltaT
def genRPoints(tMin,beTa,NW,eTa):
    tB=np.linspace(tMin,eTa*NW/2,NW/2)
    tB=np.append(-tB[::-1],tB)
    tAvg=0.5*(tB[:-1]+tB[1:])
    tAvg=np.append(tAvg,0.5*beTa)
    deltaT=tB[1:]-tB[:-1]
    deltaT=np.append(deltaT,beTa-np.sum(deltaT))

    return tAvg,deltaT
def insertPoints(tAvg,deltaT,nAdded):
    indZero=np.argmin(np.abs(tAvg))
    tStart=tAvg[indZero]+0.5*deltaT[indZero]
    tEnd=tAvg[indZero+1]+0.5*deltaT[indZero+1]

    tPoints=np.zeros(nAdded)
    deltaTP=np.zeros(nAdded)
    for i in range(nAdded-1,0,-1):
        deltaTP[i]=(tEnd-tStart)*0.5
        tPoints[i]=tEnd-0.5*deltaTP[i]

        tEnd=tPoints[i]-0.5*deltaTP[i]
    tPoints[0]=0.5*(tEnd+tStart)
    deltaTP[0]=tEnd-tStart
    tAvgN=np.concatenate((tAvg[indZero:(indZero+1)],tPoints,tAvg[(indZero+2):]))
    tAvgN=np.append(-tAvgN[-2:0:-1],tAvgN)
    deltaTN=np.concatenate((deltaT[indZero:(indZero+1)],deltaTP,deltaT[(indZero+2):]))
    deltaTN=np.append(deltaTN[-2:0:-1],deltaTN)

    return tAvgN,deltaTN

def numPointsIn(tCur,tAvg,deltaT):
    tAvgN=np.append(-(tAvg[-1]-0.25*deltaT[-1]),tAvg[:-1])
    tAvgN=np.append(tAvgN,tAvg[-1]-0.25*deltaT[-1])
    deltaTN=np.append(0.5*deltaT[-1],np.append(deltaT[:-1],0.5*deltaT[-1]))

    nPoints=np.zeros(len(tAvg)+1)
    for i in range(len(tAvg)+1):
        tCurN=np.mod(tCur-tAvgN[i]+0.5*beTa,beTa)-0.5*beTa
        indeX=np.arange(len(tCur))[np.logical_and(tCurN>=(-0.5*deltaTN[i]),tCurN<=(0.5*deltaTN[i]))]

        nPoints[i]=len(indeX)
    nPoints[-1]+=nPoints[0]
    return nPoints[1:]

def mergeTIntervals(tAvg,deltaT,NWx,AC,eTa):
    tMin=0.5*deltaT[np.argmin(np.abs(tAvg))]

    beTa=np.sum(deltaT)
    fA=np.sqrt(AC**2+1)
    beTa=np.sum(deltaT)
    tAvgN,deltaTN=genTPoints(tMin,beTa,NWx,fA,eTa)

    tAvgNN=np.append(-(tAvgN[-1]-0.25*deltaTN[-1]),tAvgN[:-1])
    tAvgNN=np.append(tAvgNN,tAvgN[-1]-0.25*deltaTN[-1])
    deltaTNN=np.append(0.5*deltaTN[-1],np.append(deltaTN[:-1],0.5*deltaTN[-1]))

    indX=[]
    for i in range(len(tAvgNN)):
        cond1=tAvg>=(tAvgNN[i]-0.5*deltaTNN[i])
        cond2=tAvg<=(tAvgNN[i]+0.5*deltaTNN[i])
        conD=np.logical_and(cond1,cond2)
        indeX=np.arange(len(tAvg))[conD]
        if len(indeX)>0:
            indX.append(indeX)

    NWn=len(indX)-1
    if NWn<NWx:
        inB=indX[0]
        indX[0]=inB[:(len(inB)-int((NWx-NWn)/2))]

        inE=indX[-1]
        indX[-1]=inE[int((NWx-NWn)/2):]
        for i in range(int((NWx-NWn)/2)):
            iR=len(inB)-i-1
            indX.insert(1,inB[iR:(iR+1)])
            indX.insert(-1,inE[i:(i+1)])

    tAvgM=np.zeros(NWx)
    deltaTM=np.zeros(NWx)
    for i in range(NWx-1):
        deltaTM[i]=np.sum(deltaT[indX[i+1]])
        tAvgM[i]=0.5*(tAvg[indX[i+1]][0]-0.5*deltaT[indX[i+1]][0]+\
            tAvg[indX[i+1]][-1]+0.5*deltaT[indX[i+1]][-1])

    deltaTM[-1]=np.sum(deltaT[indX[0]])+np.sum(deltaT[indX[-1]])
    tAvgM[-1]=0.5*beTa

    mergInd=[]
    for i in range(NWx-1):
        aI=np.zeros((2,len(indX[i+1])))
        aI[0,:]=indX[i+1]
        aI[1,:]=deltaT[indX[i+1]]/deltaTM[i]
        mergInd.append(aI)

    lenF=len(indX[0])+len(indX[-1])
    aI=np.zeros((2,lenF))
    aI[0,:]=np.append(indX[0],indX[-1])
    aI[1,:]=np.append(deltaT[indX[0]]/deltaTM[-1],deltaT[indX[-1]]/deltaTM[-1])
    mergInd.append(aI)
    return tAvgM,deltaTM,mergInd

def logspaceS(aMin,aMax,Nl):
    return np.logspace(np.log(aMin)/np.log(10),np.log(aMax)/np.log(10),Nl)

def haarArray(nT):
    nLevels=int(np.log2(nT))

    hArr=np.diag(np.zeros(nT)+1)
    for i in range(nLevels,0,-1):
        lCur=2**i
        index1=np.arange(0,lCur,2)
        index2=np.arange(1,lCur,2)

        hArrCur=np.diag(np.zeros(nT)+1)
        hArrCur[np.ix_(np.arange(lCur),np.arange(lCur))]=np.zeros((lCur,lCur))
        for j in range(lCur/2):
            iArr=np.zeros(nT)
            iArr[index1[j]]=1
            iArr[index2[j]]=1
            hArrCur[j,:]=(1.0/np.sqrt(2))*iArr
            iArr=np.zeros(nT)
            iArr[index1[j]]=1
            iArr[index2[j]]=-1
            hArrCur[(lCur/2)+j,:]=(1.0/np.sqrt(2))*iArr

        hArr=np.matmul(hArrCur,hArr)
    mInd=[[] for i in range(nT)]
    iFull=np.arange(nT)
    for i in range(nT):
        iCur=iFull[np.abs(hArr[i,:])>0]
        wCur=hArr[i,:][iCur]
        mInd[i]=[iCur,wCur]
    return mInd
def debArray(nT):
    nLevels=int(np.log2(nT))

    alpha=[(1+np.sqrt(3))/(4*np.sqrt(2)),(np.sqrt(3)+3)/(4*np.sqrt(2)),
        (3-np.sqrt(3))/(4*np.sqrt(2)),(1-np.sqrt(3))/(4*np.sqrt(2))]

    beTa=[(1-np.sqrt(3))/(4*np.sqrt(2)),(np.sqrt(3)-3)/(4*np.sqrt(2)),
        (3+np.sqrt(3))/(4*np.sqrt(2)),(-1-np.sqrt(3))/(4*np.sqrt(2))]

    hArr=np.diag(np.zeros(nT)+1)
    for i in range(nLevels,0,-1):
        lCur=2**i
        index1=np.mod(np.arange(0,lCur,2),lCur)
        index2=np.mod(np.arange(1,lCur,2),lCur)
        index3=np.mod(np.arange(2,lCur+1,2),lCur)
        index4=np.mod(np.arange(3,lCur+2,2),lCur)

        hArrCur=np.diag(np.zeros(nT)+1)
        hArrCur[np.ix_(np.arange(lCur),np.arange(lCur))]=np.zeros((lCur,lCur))
        for j in range(lCur/2):
            iArr=np.zeros(nT)
            iArr[index1[j]]+=alpha[0]
            iArr[index2[j]]+=alpha[1]
            iArr[index3[j]]+=alpha[2]
            iArr[index4[j]]+=alpha[3]
            hArrCur[j,:]=iArr
            iArr=np.zeros(nT)
            iArr[index1[j]]+=beTa[0]
            iArr[index2[j]]+=beTa[1]
            iArr[index3[j]]+=beTa[2]
            iArr[index4[j]]+=beTa[3]
            hArrCur[(lCur/2)+j,:]=iArr
        hArr=np.matmul(hArrCur,hArr)
    mInd=[[] for i in range(nT)]
    iFull=np.arange(nT)
    for i in range(nT):
        iCur=iFull[np.abs(hArr[i,:])>0]
        wCur=hArr[i,:][iCur]
        mInd[i]=[iCur,wCur]
    return mInd
def haarTransformF(uV,axis=0):
    uT=np.swapaxes(1*uV,0,axis)
    nLevels=int(np.log2(uT.shape[0]))

    for i in range(nLevels,0,-1):
        lCur=2**i
        index1=np.arange(0,lCur,2)
        index2=np.arange(1,lCur,2)

        uA=(1/np.sqrt(2))*(uT[index1,...]+uT[index2,...])
        uD=(1/np.sqrt(2))*(uT[index1,...]-uT[index2,...])
        uT[:(lCur/2),...]=uA
        uT[(lCur/2):lCur,...]=uD

    uT=np.swapaxes(uT,0,axis)
    return uT
def debTransformF(uV,axis=0):
    uT=np.swapaxes(1*uV,0,axis)
    nLevels=int(np.log2(uT.shape[0]))

    alpha=[(1+np.sqrt(3))/(4*np.sqrt(2)),(np.sqrt(3)+3)/(4*np.sqrt(2)),
        (3-np.sqrt(3))/(4*np.sqrt(2)),(1-np.sqrt(3))/(4*np.sqrt(2))]

    beTa=[(1-np.sqrt(3))/(4*np.sqrt(2)),(np.sqrt(3)-3)/(4*np.sqrt(2)),
        (3+np.sqrt(3))/(4*np.sqrt(2)),(-1-np.sqrt(3))/(4*np.sqrt(2))]
    for i in range(nLevels,0,-1):
        lCur=2**i
        index1=np.mod(np.arange(0,lCur,2),lCur)
        index2=np.mod(np.arange(1,lCur,2),lCur)
        index3=np.mod(np.arange(2,lCur+1,2),lCur)
        index4=np.mod(np.arange(3,lCur+2,2),lCur)

        uA=alpha[0]*uT[index1,...]+alpha[1]*uT[index2,...]+\
            alpha[2]*uT[index3,...]+alpha[3]*uT[index4,...]
        uD=beTa[0]*uT[index1,...]+beTa[1]*uT[index2,...]+\
            beTa[2]*uT[index3,...]+beTa[3]*uT[index4,...]

        uT[:(lCur/2),...]=uA
        uT[(lCur/2):lCur,...]=uD

    uT=np.swapaxes(uT,0,axis)
    return uT
def coifTransformF(uV,axis=0):
    uT=np.swapaxes(1*uV,0,axis)
    nLevels=int(np.log2(uT.shape[0]))

    alpha=[(1-np.sqrt(7))/(16*np.sqrt(2)),(np.sqrt(7)+5)/(16*np.sqrt(2)),
        (14+2*np.sqrt(7))/(16*np.sqrt(2)),(14-2*np.sqrt(7))/(16*np.sqrt(2)),
        (1-np.sqrt(7))/(16*np.sqrt(2)),(-3+np.sqrt(7))/(16*np.sqrt(2))]

    beTa=[alpha[5],-alpha[4],alpha[3],-alpha[2],alpha[1],-alpha[0]]
    for i in range(nLevels,0,-1):
        lCur=2**i
        index1=np.mod(np.arange(-2,lCur-2,2),lCur)
        index2=np.mod(np.arange(-1,lCur-1,2),lCur)
        index3=np.mod(np.arange(0,lCur,2),lCur)
        index4=np.mod(np.arange(1,lCur,2),lCur)
        index5=np.mod(np.arange(2,lCur+1,2),lCur)
        index6=np.mod(np.arange(3,lCur+2,2),lCur)

        uA=alpha[0]*uT[index1,...]+alpha[1]*uT[index2,...]+\
            alpha[2]*uT[index3,...]+alpha[3]*uT[index4,...]+\
            alpha[4]*uT[index5,...]+alpha[5]*uT[index6,...]
        uD=beTa[0]*uT[index1,...]+beTa[1]*uT[index2,...]+\
            beTa[2]*uT[index3,...]+beTa[3]*uT[index4,...]+\
            beTa[4]*uT[index5,...]+beTa[5]*uT[index6,...]

        uT[:(lCur/2),...]=uA
        uT[(lCur/2):lCur,...]=uD

    uT=np.swapaxes(uT,0,axis)
    return uT
def haarTransform(uV,axis=0):
    uV=np.swapaxes(uV,0,axis)
    nLevels=int(np.log2(uV.shape[0]))

    uCur=np.zeros(uV.shape,dtype=np.complex_)
    haarTransform=np.zeros(uV.shape,dtype=np.complex_)
    nDepth=np.zeros(uV.shape[0],dtype=int)
    nShift=np.zeros(uV.shape[0],dtype=int)

    uCur[:]=uV[:]
    lStart=0
    for i in range(nLevels):
        lCur=2**(nLevels-1-i)
        lEnd=lStart+lCur
        nDepth[lStart:lEnd]=nLevels-i-1
        nShift[lStart:lEnd]=np.arange(lCur)

        haarTransform[lStart:lEnd,...]=(1.0/np.sqrt(2))*(uCur[::2,...]-\
            uCur[1::2,...])
        uCur=(1.0/np.sqrt(2))*(uCur[::2,...]+uCur[1::2,...])
        lStart=lEnd

    nDepth[-1]=0
    nShift[-1]=-1
    haarTransform[-1,...]=uCur

    enG=haarTransform*np.conj(haarTransform)
    engT=np.sum(enG,axis=0)
    engT=np.reshape(engT,engT.size)
    enG=np.reshape(enG,(enG.shape[0],np.prod(enG.shape[1:])))

    cIndex=np.abs(engT)>0
    nRel=2**nDepth+nShift

    if len(engT[cIndex])>0:
        engT=engT[cIndex]
        sIndex=np.argsort(enG,axis=0)[::-1,:]
        engN=np.sort(enG,axis=0)[::-1,:]

        eCur=np.zeros(engT.shape,dtype=np.complex_)
        for i in range(sIndex.shape[0]):
            eVal=engN[i,...]
            eCur+=eVal[cIndex]/engT
            if eCur.min()>0.99:
                break

        indexF=sIndex[:,np.argmin(eCur)]
        nRel=nRel[indexF[:i]]
        haarTransform=np.swapaxes(haarTransform[indexF[:i],...],0,axis)
    else:
        haarTransform=haarTransform[-2:,...]
        nRel=nRel[-2:]

    return haarTransform,nRel

def avgVert(tCur,UnL,tAvg,deltaT):
    UnA=np.zeros((len(tAvg)+1,)+UnL.shape[1:],dtype=np.complex_)
    beTa=np.sum(deltaT)

    wInd=np.argmin(np.abs(tAvg))
    deltaT[wInd-1]+=1.2*deltaT[wInd]
    deltaT[wInd+1]+=1.2*deltaT[wInd]

    tAvgN=np.append(-(tAvg[-1]-0.25*deltaT[-1]),tAvg[:-1])
    tAvgN=np.append(tAvgN,tAvg[-1]-0.25*deltaT[-1])
    deltaTN=np.append(0.5*deltaT[-1],np.append(deltaT[:-1],0.5*deltaT[-1]))
    for i in range(len(tAvg)+1):
        tStart=tAvgN[i]-0.5*deltaTN[i]
        tEnd=tAvgN[i]+0.5*deltaTN[i]
        indeX=np.arange(len(tCur))[np.logical_and(tCur>=tStart,tCur<=tEnd)]

        if len(indeX)>0:
            tPoints=np.mod(tCur[indeX]+0.5*beTa,beTa)-0.5*beTa
            dT=tPoints[1:]-tPoints[:-1]
            tBegin=tAvgN[i]-0.5*deltaTN[i]
            tEnd=tAvgN[i]+0.5*deltaTN[i]

            dT=np.append((tPoints[0]-tBegin),np.append(dT,(tEnd-tPoints[-1])))
            fPoints=0.5*(UnL[indeX[1:],...]+UnL[indeX[:-1]])
            fPoints=np.concatenate((UnL[indeX[0:1],...],fPoints),axis=0)
            fPoints=np.concatenate((fPoints,UnL[indeX[-1:],...]),axis=0)

            UnA[i,...]=(1.0/deltaTN[i])*np.sum(dT[:,None,None]*fPoints,axis=0)
    UnAn=UnA[1:,...]
    UnAn[-1,...]+=UnA[0,...]
    return UnAn

def genLogPoints(tMin,tMax,nPoints):
    xMin=np.log(tMin/(tMax-tMin))
    xR=np.linspace(xMin,-xMin,nPoints)

    tV=tMax*np.exp(xR)/(1+np.exp(xR))

    return tV
def genGeoPoints(nPoints,beTa,tMin,tZ):
    deltaT=[tZ,tMin]
    nCount=nPoints/2-2
    aPow=np.exp(np.log(beTa/(2*tMin))/nCount)

    conD=np.sum(deltaT)+aPow*deltaT[-1]
    while conD<(beTa/2) and nCount>1 :
        deltaT.append(aPow*deltaT[-1])
        conD=np.sum(deltaT)+aPow*deltaT[-1]
        nCount-=1
    deltaT.append(0.5*beTa-np.sum(deltaT))
    tAvg=[0.5*deltaT[0]]
    for i in range(1,len(deltaT)):
        tAvg.append(0.5*(2*tAvg[i-1]+deltaT[i-1]+\
            deltaT[i]))
    tAvg=np.asarray(tAvg)
    deltaT=np.asarray(deltaT)
    tAvg=np.append(tAvg,beTa-tAvg[::-1])
    deltaT=np.append(deltaT,deltaT[::-1])
    return tAvg,deltaT
def disAvgFunc(UnL,tCur,tAvg,deltaT):
    tAvgN=np.append(-(tAvg[-1]-0.25*deltaT[-1]),tAvg[:-1])
    tAvgN=np.append(tAvgN,tAvg[-1]-0.25*deltaT[-1])
    deltaTN=np.append(0.5*deltaT[-1],np.append(deltaT[:-1],0.5*deltaT[-1]))

    if len(UnL.shape)==1:
        UnLa=np.zeros(len(tAvg)+1,dtype=np.complex_)
        for i in range(len(tAvg)):
            conD1=(tCur>=(tAvgN[i]-0.5*deltaTN[i]))
            conD2=(tCur<=(tAvgN[i]+0.5*deltaTN[i]))
            indeX=np.arange(len(tCur))[np.logical_and(conD1,conD2)]

            UnLa[i]=np.sum(UnL[indeX])
        UnLa[-1]+=UnLa[0]
        UnLa=UnLa[1:]
    else:
        UnLa=np.zeros((len(tAvg)+1,UnL.shape[1],UnL.shape[2],UnL.shape[3]),dtype=np.complex_)
        for i in range(len(tAvg)+1):
            conD1=(tCur>=(tAvgN[i]-0.5*deltaTN[i]))
            conD2=(tCur<=(tAvgN[i]+0.5*deltaTN[i]))
            indeX=np.arange(len(tCur))[np.logical_and(conD1,conD2)]

            UnLa[i,...]=np.sum(UnL[indeX,...],axis=0)
        UnLa[-1,...]+=UnLa[0,...]
        UnLa=UnLa[1:,...]
    return UnLa
def clusterPoints(tCenter,tWidth,tPoints):
    tPointsIn=np.zeros(0)
    wPointsIn=np.zeros(0)
    indPointsIn=np.zeros(0,dtype=int)

    dt0=tPoints[1]-tPoints[0]
    for i,(tCur,tWidCur) in enumerate(zip(tCenter,tWidth)):
        tLeft=tCur-0.5*tWidCur
        tRight=tCur+0.5*tWidCur

        cond1=np.where(tPoints>tLeft)[0]
        cond2=np.where(tPoints<tRight)[0]
        conD=np.intersect1d(cond1,cond2)

        tPointsCur=tPoints[conD]
        dtCur=np.zeros(len(tPointsCur))
        for j,tRedCur in enumerate(tPointsCur):
            tLeftr=max([tRedCur-0.5*dt0,tLeft])
            tRightr=min([tRedCur+0.5*dt0,tRight])
            tDiff=tRightr-tLeftr
            dtCur[j]=1.0/len(tPointsCur)

        tPointsIn=np.append(tPointsIn,tPointsCur)
        wPointsIn=np.append(wPointsIn,dtCur)
        indPointsIn=np.append(indPointsIn,np.zeros(len(tPointsCur),dtype=int)+i)

    return tPointsIn,wPointsIn,indPointsIn
def indiciesInInterval(tAvg,deltaT):
    tFX=np.zeros(0)
    dT=np.zeros(0)
    dInd=np.zeros(0,dtype=int)

    tV,dTV=pointsInInterval(tAvg,deltaT)
    for i in range(len(tV)):
        tFX=np.append(tFX,tV[i])
        dT=np.append(dT,dTV[i])
        dInd=np.append(dInd,i+np.zeros(len(tV[i]),dtype=int))
    return tFX,dT,dInd
def pointsInInterval(tAvg,deltaT):
    #tAvgN=np.append(-(tAvg[-1]-0.25*deltaT[-1]),tAvg[:-1])
    #tAvgN=np.append(tAvgN,tAvg[-1]-0.25*deltaT[-1])
    #deltaTN=np.append(0.5*deltaT[-1],np.append(deltaT[:-1],0.5*deltaT[-1]))
    tAvgN=1*tAvg
    deltaTN=1*deltaT
    deltaTN=np.round(deltaTN,5)
    deltaTmax=np.unique(deltaTN)[1]

    tPoint=[[] for i in range(len(tAvg))]
    dTvals=[[] for i in range(len(tAvg))]
    for i in range(len(tAvg)):
        if deltaTN[i]>deltaTmax:
            nCur=int(4*deltaTN[i]/deltaTmax)
            nCur=4
            tStart=(tAvgN[i]-0.5*deltaTN[i])
            tEnd=(tAvgN[i]+0.5*deltaTN[i])
            tPoints=np.linspace(tStart,tEnd,nCur+1)
            tPoint[i]=0.5*(tPoints[1:]+tPoints[:-1])
            dTvals[i]=tPoints[1:]-tPoints[:-1]
            #tPoint[i],dTvals[i]=auxF.gaussianInt([tStart,tEnd],nCur)
        elif deltaTN[i]==deltaTmax:
            nCur=int(4*deltaTN[i]/deltaTmax)
            nCur=2
            tStart=(tAvgN[i]-0.5*deltaTN[i])
            tEnd=(tAvgN[i]+0.5*deltaTN[i])
            tPoints=np.linspace(tStart,tEnd,nCur+1)
            tPoint[i]=0.5*(tPoints[1:]+tPoints[:-1])
            dTvals[i]=tPoints[1:]-tPoints[:-1]
            #tPoint[i],dTvals[i]=auxF.gaussianInt([tStart,tEnd],nCur)
        else:
            nCur=1

            tStart=(tAvgN[i]-0.5*deltaTN[i])
            tEnd=(tAvgN[i]+0.5*deltaTN[i])
            tPoints=np.linspace(tStart,tEnd,nCur+1)
            tPoint[i]=0.5*(tPoints[1:]+tPoints[:-1])
            dTvals[i]=tPoints[1:]-tPoints[:-1]

    #tPoint[-1]=np.append(tPoint[0],tPoint[-1])
    #dTvals[-1]=np.append(dTvals[0],dTvals[-1])

    for i in range(len(deltaT)):
        dTvals[i]=dTvals[i]/deltaT[i]

    return tPoint,dTvals
