import frgd.auxFunctions as auxF

class sparseVert:
    def __init__(self,aEnt,aInd,axis=0):
        self.arraY=aEnt
        self.aInd=aInd
        self.axis=axis

    def __add__(self,b):
        arrayN,aIndN=auxF.expandAdd2([self.arraY,self.aInd],\
            [b.arraY,b.aInd],axis=self.axis)

        return sparseVert(arrayN,aIndN,self.axis)

    def __sub__(self,b):
        arrayN,aIndN=auxF.expandAdd2([self.arraY,self.aInd],\
            [-b.arraY,b.aInd],axis=self.axis)

        return sparseVert(arrayN,aIndN,self.axis)

    def __mul__(self,b):
        return sparseVert(b*self.arraY,self.aInd,self.axis)
    def __neg__(self):
        return sparseVert(-self.arraY,self.aInd,self.axis)

    def __iadd__(self,b):
        arrayN,aIndN=auxF.expandAdd2([self.arraY,self.aInd],\
            [b.arraY,b.aInd],axis=self.axis)

        return sparseVert(arrayN,aIndN,self.axis)
