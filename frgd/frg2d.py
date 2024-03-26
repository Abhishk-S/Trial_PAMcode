from frgd.frgFlow import fRG2D
from frgd.vertexF import vertexR
import frgd.auxFunctions as auxF
from tempfile import TemporaryFile
import numpy as np
import time

class frg:
    """
    Class for the parameters and output of
    a functional Renormalization Group run.

    ...
    Methods
    -------
    initHamiltonian()
        Initializes the single particle hopping Hamiltonian by defining
        disperMat. Initializes the coupling between the orbitals in the system by
        setting uMat, vMat, jMat.

    couplePhonons()
        Uses the electron phonon coupling and the phonon dispersion to initialize
        a two particle vertex

    runFlow()
        Runs the fully initialized fRG flow. Saves the final vertex (optional),
        the self energy and susceptibilities to a file.
    """
    def __init__(self,nF,beTa,wMax,Ns,NW,NK,NB,cutoffR,nDim,nLoop):
        """
        Parameters
        ----------

        nF : int
            Maximum number of Matsubara frequencies

        beTa : float
            Inverse temperature of the system

        wMax : float
            Scale at which the fRG flow is initialized

        Ns : int
            Number of sites

        NW : int
            Number of basis sets for the expansion of the auxiliary frequency
            channels

        NK : int
            Number of basis sets for the expansion of the auxiliary momentum
            channels

        NB : int
            Number of singular modes retained in each channel after SVD
            decomposition of band dependence

        cutoffR : str
            Type of regulator for flow: Litim, Additive, Sharp, Soft

        nDim : int
            Spatial dimension of the system

        nLoop : int
            Loop order at which flow is constructed
        """
        self.fRGrun=fRG2D(nF,beTa,wMax,Ns,NW,NK,NB,cutoffR,nDim,nLoop)

    def initFreePeriodicHamilt(self,disMat,xFillc,vCF,xFillf):
        """
        Parameters
        ----------

        disMat : array_like(complex, ndim=3)
            Matrix that defines single particle hopping of c electrons

        xFillc : float
            The level of doping of the delocalized c electrons

        vCF: array_like(complex, ndim=3)
            Matrix that defines hopping between c and f sites

        xFillf : float
            The level of doping of the localized f electrons

        """

        self.fRGrun.initSinglePartVertPA(disMat,xFillc,vCF,xFillf)

    def initFreeDPeriodicHamilt(self,disperMatC,xFillc,cfHopping,disperMatF,xFillf):
        """
        Parameters
        ----------

        disMat : array_like(complex, ndim=3)
            Matrix that defines single particle hopping of c electrons

        xFillc : float
            The level of doping of the delocalized c electrons

        vCF: array_like(complex, ndim=3)
            Matrix that defines hopping between c and f sites

        xFillf : float
            The level of doping of the localized f electrons

        """

        self.fRGrun.initSinglePartVertDPA(disperMatC,xFillc,cfHopping,disperMatF,xFillf)

    def initFreeHamilt(self,disMat,xFill):
        """
        Parameters
        ----------

        xFill : float
            The level of doping of the system

        disMat : array_like(complex, ndim=3)
            Matrix that defines single particle hopping between orbitals
        """
        self.fRGrun.initSinglePartVert(disMat,xFill)
    def initInteractions(self,uMat,vMat=None,jMat=None):
        """
        Parameters
        ----------

        uMat : array_like(float, ndim=2)
            Matrix that defines onsite density density coupling (Hubbard - U)
            between orbitals

        vMat : array_like(float, ndim=2)
            Matrix that defines nearest neighbor density density coupling (Extended - V)
            between orbitals

        jMat : array_like(float, ndim=2)
            Matrix that defines the exchange interaction (J) between the orbitals
        """

        if vMat is None:
            vMat=np.zeros(uMat.shape)
        if jMat is None:
            jMat=np.zeros(uMat.shape)
        intVals=[uMat,vMat,jMat]
        self.fRGrun.initTwoPartVert(intVals)

    def couplePhonons(self,gEph,phononDisper):
        """
        Parameters
        ----------
        gEph : array_like(complex,ndim=3)
            Matrix that defines the electron-phonon vertex

        phononDisper: array_like(complex,ndim=3)
            Matrix that defines the phonon self energy
        """

        self.fRGrun.addPhonons(gEph,phononDisper)

    def runFlow(self,modelName,lMax=None,vOutput=None):
        """
        Parameters
        ----------

        modelName : str
            Name of file into which the final output (self Energy, susceptibilities,
            vertex (optional)) is saved

        lMax : float
            Optional, sets the scale for terminating the flow

        vOutput : bool
            Determines whether the vertex is saved to the output
        """

        if lMax is None:
            beTa=self.fRGrun.beTa
            wMax=self.fRGrun.maxW
            wMin=(np.pi)/beTa
            lMax=-np.log(wMin/wMax)+1

        t1=time.time()
        self.fRGrun.adaptiveRGFlow(lMax)
        tTot=time.time()-t1

        #Susceptibilities of the system
        staticSus,dynamicSus,strucFac,maxMomInd=self.fRGrun.susFunctions(self.fRGrun.l)

        #Gap estimates of the system
        gapX=self.fRGrun.UnF.suGap,self.fRGrun.UnF.chGap,self.fRGrun.UnF.spGap
        suGapF,chGapF,spGapF,kSU,kCH,kSP=auxF.calcMaxGap(gapX,self.fRGrun.UnF.wB,self.fRGrun.UnF.kB,\
            self.fRGrun.UnF.tPoints,self.fRGrun.UnF.sPoints)

        nDim=self.fRGrun.UnF.nDim
        Ns=self.fRGrun.UnF.N1D**nDim
        NKF=self.fRGrun.UnF.NKF
        NW=self.fRGrun.UnF.NW
        nLoop=self.fRGrun.nL
        cutoffR=self.fRGrun.cutoffT
        outputName=modelName+f'{cutoffR}nL{nLoop}nDim{nDim}NW{NW}NKF{NKF}NS{Ns}'

        wInd=np.argmin(np.abs(self.fRGrun.UnF.tPoints[0]))
        kInd=np.zeros(len(self.fRGrun.UnF.sPoints[0]))
        for i in range(nDim):
            kInd+=self.fRGrun.UnF.sPoints[i]**2
        kInd=np.argmin(kInd)
        phononV=self.fRGrun.UnF.gVert[19,:,0,wInd*NKF+kInd]

        containerD=dict(staticSus=staticSus,dynamicSus=dynamicSus,\
            maxMomInd=maxMomInd,strucFac=strucFac,wB=self.fRGrun.UnF.wB,\
            kB=self.fRGrun.UnF.kB,wF=self.fRGrun.propG.wF,mU=self.fRGrun.propG.mU,\
            sE=self.fRGrun.propG.sE,bT=self.fRGrun.beTa,lM=self.fRGrun.l,eRR=self.fRGrun.eRR,forwardLU=self.fRGrun.UnF.UnPHLV,\
            tPoints=self.fRGrun.UnF.tPoints,tToT=tTot/len(self.fRGrun.eRR),phononSE=self.fRGrun.UnF.phononSE,phononV=phononV,forwardU=self.fRGrun.UnF.UnPHV,\
            suGap=suGapF,suVecLoc=kSU,chGap=chGapF,chVecLoc=kCH,spGap=spGapF,spVecLoc=kSP)
        fileOut={**containerD,**self.fRGrun.modelParameters}
        np.savez(outputName,**fileOut)
