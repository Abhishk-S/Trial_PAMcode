import unittest
from frgd.vertexF import vertexR
import frgd.auxFunctions as auxF
import numpy as np
from numpy.testing import assert_almost_equal as assertAE

class testVertex(unittest.TestCase):
    def setUp(self):
        maxW,beTa=100.0,5.0
        nPatches=20
        N1D=8
        nDim=2
        NW,NK,NB=4,0,1
        nLoop=2
        nB=1
        """
        nB=2
        kX=((2*np.pi)/float(N1D))*np.arange(-N1D/2+1,N1D/2+1,1)
        kBx=np.repeat(kX,N1D)
        kBy=np.tile(kX,N1D)

        disMat=np.zeros((N1D*N1D,nB,nB))
        txy1=-1.0
        tz1=-0.2
        exy0=0
        txyz=0.2
        ez0=-0.6

        eX=2*txy1*(np.cos(kBx)+np.cos(kBy))
        eY=2*tz1*(np.cos(kBx)+np.cos(kBy))
        eXY=2*txyz*(np.cos(kBx)-np.cos(kBy))

        disMat[:,0,0]=eY-ez0
        disMat[:,1,1]=eX+exy0
        disMat[:,0,1]=eXY
        disMat[:,1,0]=eXY

        eV,uFor=np.linalg.eigh(disMat)

        for j in range(nB):
            for i in range(uFor.shape[0]-1):
                uSignC=np.sign(np.sum(uFor[0,:,j]*uFor[0,:,j]))
                uSign=np.sign(np.sum(uFor[0,:,j]*uFor[i+1,:,j]))

                if uSign.real!=uSignC.real and np.abs(uSign.real)>0:
                    uFor[i+1,:,j]=-uFor[i+1,:,j]

        uBack=np.linalg.inv(uFor)

        def indexOfMomenta(kIn,N1D):
            nDim=len(kIn)
            momFac=N1D/(2*np.pi)

            kQi=np.zeros(len(kIn[0]),dtype=int)
            for i in range(nDim):
                kQc=np.mod((momFac*kIn[i]).astype(int)+int(N1D/2-1),N1D).astype(int)
                kQi+=kQc*(N1D**(nDim-1-i))

            return kQi

        uintVV=np.zeros((nB,nB))+2.2
        vintVV=np.zeros((nB,nB))+0.45
        jintVV=np.zeros((nB,nB))+0.45
        uintVals=(uintVV,vintVV,jintVV)
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

            k1ind=indexOfMomenta(k1,N1D)
            k2ind=indexOfMomenta(k2,N1D)
            k3ind=indexOfMomenta(k3,N1D)
            k4ind=indexOfMomenta(k4,N1D)

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
                        uV+=bandMap*uintVals[0][i,j]
                        for k in range(nDim):
                            uV+=bandMap*2*uintVals[1][i,j]*np.cos(k3[k]-k2[k])

                        bandMap=uBack[k1ind,bandInd[0],i]*uBack[k2ind,bandInd[1],j]*\
                            uFor[k3ind,i,bandInd[2]]*uFor[k4ind,j,bandInd[3]]
                        uV+=bandMap*uintVals[2][i,j]

                        bandMap=uBack[k1ind,bandInd[0],i]*uBack[k2ind,bandInd[1],i]*\
                            uFor[k3ind,j,bandInd[2]]*uFor[k4ind,j,bandInd[3]]
                        uV+=bandMap*uintVals[2][i,j]
            return np.reshape(uV,kShape)
        """


        uConst=5
        def singFunc(x):
            val=np.zeros(x.shape)
            for i in range(1,3):
                val+=i*np.cos(i*x)
            return val
        def auxFunc(x):
            val=np.zeros(x.shape)
            for i in range(1,2):
                val+=i*np.cos(i*x)
            return val
            #return np.cos(x)
            #return np.zeros(x.shape)+1

        def uVertex(kPP,kPH,kPHE,bI):
            kShape=kPP[0].shape
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

            funcVal=np.zeros(kPP[i].size,dtype=np.complex_)+uConst
            #for i in range(nDim):
                #funcVal+=auxFunc(k1[i])+auxFunc(k2[i])+auxFunc(k3[i])+auxFunc(k4[i])
                #funcVal+=auxFunc(k3[i]-k2[i])#*auxFunc(k3[i]-k1[i])*auxFunc(k1[i]+k2[i])
            return np.reshape(funcVal,kShape)

        self.UnF=vertexR(maxW,nPatches,N1D,nDim,nB,NW,NK,NB,beTa)
        self.UnF.initializeVertex(uVertex)

    def test_uEvaluate(self):
        wZ=np.zeros(1)
        kZ=self.UnF.kB
        uBaseline=self.UnF.uEvaluate(wZ,wZ,wZ,kZ,kZ,kZ,0.0)

        #pp Test
        self.UnF.UnPP=1*self.UnF.UnPPO
        uPP=self.UnF.uEvaluate(wZ,wZ,wZ,kZ,kZ,kZ,0.0)-uBaseline
        self.UnF.UnPP=0*self.UnF.UnPPO
        np.testing.assert_almost_equal(uPP,uBaseline)

        #phe Test
        self.UnF.UnPHE=1*self.UnF.UnPHEO
        uPHE=self.UnF.uEvaluate(wZ,wZ,wZ,kZ,kZ,kZ,0.0)-uBaseline
        self.UnF.UnPHE=0*self.UnF.UnPHEO
        np.testing.assert_almost_equal(uPHE,uBaseline)

        #ph Test
        self.UnF.UnPH=1*self.UnF.UnPHO
        uPH=self.UnF.uEvaluate(wZ,wZ,wZ,kZ,kZ,kZ,0.0)-uBaseline
        self.UnF.UnPH=0*self.UnF.UnPHO
        np.testing.assert_almost_equal(uPH,uBaseline)

    def test_momLegSEnD(self):
        nDim=self.UnF.nDim
        N1D=self.UnF.N1D
        NKF=self.UnF.NKF
        def sProp(x):
            val=np.zeros(x[0].shape)
            for j in range(nDim):
                for i in range(1,3):
                    val+=i*np.cos(i*x[j])
            return val
        wZ=np.zeros(1)
        kZ=self.UnF.kB
        sPropK=auxF.ffTransformD(sProp(kZ),N1D,nDim,axis=0)
        N=self.UnF.N

        kPP=[None for i in range(nDim)]
        kPH=[None for i in range(nDim)]
        kPHE=[None for i in range(nDim)]

        sEPP1=np.zeros(N,dtype=np.complex_)
        for i in range(N):
            for j in range(nDim):
                kPP[j]=kZ[j][i]+kZ[j]
                kPH[j]=kZ[j]-kZ[j]
                kPHE[j]=kZ[j]-kZ[j][i]
            uBaseline=self.UnF.uEvaluate(wZ,wZ,wZ,kPP,kPH,kPHE,0.0)
            self.UnF.UnPP=1*self.UnF.UnPPO
            uPP=self.UnF.uEvaluate(wZ,wZ,wZ,kPP,kPH,kPHE,0.0)-uBaseline
            self.UnF.UnPP=0*self.UnF.UnPPO
            sEPP1[i]=(1.0/N)*np.sum(sProp(kZ)*uPP[0,:])


        UnPPi=self.UnF.legndExpand(self.UnF.UnPPO,self.UnF.uLeftPP,self.UnF.uRightPP,self.UnF.ppIndex,0)
        def momSum(UnX,sEkTrans,sPropK):
            dSE=np.zeros(N,dtype=np.complex_)
            UnXc=np.sum(UnX,axis=0)

            for i in range(N):
                indX=sEkTrans[i][0]
                dSE+=sPropK[i]*np.sum(UnXc[indX][:,None]*sEkTrans[i][1],axis=0)

            return dSE
        def momSumFFT(UnX,sEkTrans,sPropK):
            dSE=np.zeros(N,dtype=np.complex_)
            UnXc=np.reshape(np.sum(UnX,axis=0),(N,NKF,NKF))
            for i in range(NKF):
                for j in range(NKF):
                    indX=sEkTrans[i][j][1]
                    indY=sEkTrans[i][j][0]
                    dSE+=auxF.iffTransformD(UnXc[indX,i,j]*sPropK[indY],N1D,nDim,axis=0)*sEkTrans[i][j][2]
            return dSE
        sETransPP1=self.UnF.kExpandPP1
        sEPP1a=momSumFFT(UnPPi[0],sETransPP1,sPropK)

        sEPP2=np.zeros(N,dtype=np.complex_)
        for i in range(N):
            for j in range(nDim):
                kPP[j]=kZ[j][i]+kZ[j]
                kPHE[j]=kZ[j]-kZ[j]
                kPH[j]=kZ[j]-kZ[j][i]
            uBaseline=self.UnF.uEvaluate(wZ,wZ,wZ,kPP,kPH,kPHE,0.0)
            self.UnF.UnPP=1*self.UnF.UnPPO
            uPP=self.UnF.uEvaluate(wZ,wZ,wZ,kPP,kPH,kPHE,0.0)-uBaseline
            self.UnF.UnPP=0*self.UnF.UnPPO
            sEPP2[i]=(1.0/N)*np.sum(sProp(kZ)*uPP[0,:])

        UnPPi=self.UnF.legndExpand(self.UnF.UnPPO,self.UnF.uLeftPP,self.UnF.uRightPP,self.UnF.ppIndex,0)
        sETransPP2=self.UnF.kExpandPP2
        sEPP2a=momSumFFT(UnPPi[0],sETransPP2,sPropK)

        sEPH=np.zeros(N,dtype=np.complex_)
        for i in range(N):
            for j in range(nDim):
                kPP[j]=kZ[j][i]+kZ[j]
                kPHE[j]=kZ[j]-kZ[j]
                kPH[j]=kZ[j]-kZ[j][i]
            uBaseline=self.UnF.uEvaluate(wZ,wZ,wZ,kPP,kPH,kPHE,0.0)
            self.UnF.UnPH=1*self.UnF.UnPHO
            uPP=self.UnF.uEvaluate(wZ,wZ,wZ,kPP,kPH,kPHE,0.0)-uBaseline
            self.UnF.UnPH=0*self.UnF.UnPHO
            sEPH[i]=(1.0/N)*np.sum(sProp(kZ)*uPP[0,:])

        UnPHi=self.UnF.legndExpand(self.UnF.UnPHO,self.UnF.uLeftPH,self.UnF.uRightPH,self.UnF.phIndex,0)
        sETransPH=self.UnF.kExpandPH
        sEPHa=momSumFFT(UnPHi[0],sETransPH,sPropK)


        sEPHE=np.zeros(N,dtype=np.complex_)
        for i in range(N):
            for j in range(nDim):
                kPP[j]=kZ[j][i]+kZ[j]
                kPH[j]=kZ[j]-kZ[j]
                kPHE[j]=kZ[j]-kZ[j][i]
            uBaseline=self.UnF.uEvaluate(wZ,wZ,wZ,kPP,kPH,kPHE,0.0)
            self.UnF.UnPHE=1*self.UnF.UnPHEO
            uPP=self.UnF.uEvaluate(wZ,wZ,wZ,kPP,kPH,kPHE,0.0)-uBaseline
            self.UnF.UnPHE=0*self.UnF.UnPHEO
            sEPHE[i]=(1.0/N)*np.sum(sProp(kZ)*uPP[0,:])

        UnPHEi=self.UnF.legndExpand(self.UnF.UnPHEO,self.UnF.uLeftPHE,self.UnF.uRightPHE,self.UnF.pheIndex,0)
        sETransPHE=self.UnF.kExpandPH
        sEPHEa=momSumFFT(UnPHEi[0],sETransPHE,sPropK)

        np.testing.assert_almost_equal(sEPP1a,sEPP1)
        np.testing.assert_almost_equal(sEPP2a,sEPP2)
        np.testing.assert_almost_equal(sEPHa,sEPH)
        np.testing.assert_almost_equal(sEPHEa,sEPHE)

    def test_legndExpand(self):
        N1D=self.UnF.N1D
        nDim=self.UnF.nDim
        UnPPi=self.UnF.legndExpand(self.UnF.UnPPO,self.UnF.uLeftPP,self.UnF.uRightPP,self.UnF.ppIndex,0)
        UnPHi=self.UnF.legndExpand(self.UnF.UnPHO,self.UnF.uLeftPH,self.UnF.uRightPH,self.UnF.phIndex,0)
        UnPHEi=self.UnF.legndExpand(self.UnF.UnPHEO,self.UnF.uLeftPHE,self.UnF.uRightPHE,self.UnF.pheIndex,0)

        wZ=np.zeros(1)
        kZ=self.UnF.kB
        uBaseline=self.UnF.uEvaluate(wZ,wZ,wZ,kZ,kZ,kZ,0.0)

        for i in range(len(UnPPi)):
            uPHpp,uPHEpp=self.UnF.projectChannel(UnPPi[i],0.0,'PP')

            #ph Test
            self.UnF.UnPH=1*uPHpp
            uPH=self.UnF.uEvaluate(wZ,wZ,wZ,kZ,kZ,kZ,0.0)-uBaseline
            self.UnF.UnPH=0*uPHpp

            #phe Test
            self.UnF.UnPHE=1*uPHEpp
            uPHE=self.UnF.uEvaluate(wZ,wZ,wZ,kZ,kZ,kZ,0.0)-uBaseline
            self.UnF.UnPHE=0*uPHEpp

            np.testing.assert_almost_equal(uPH,uPHE)
            np.testing.assert_almost_equal(uPH,uBaseline)
            np.testing.assert_almost_equal(uPHE,uBaseline)
