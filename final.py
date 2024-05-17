import qSub
import numpy as np
 
t = 1
uU= 6*t

v0= [0.66]#[0.66,0.75,1.03,1.1]
v1= [0.84]#[0.84,0.96,1.395,1.5]


nL=2
beTa=40  
Ns=8
NW=4
NK=0
 
xDopC=0
xDopF= [0.1]#np.linspace(0,0.25,15)
 
runner=0
for j,v0Cur in enumerate(v0):
    for i,v1Cur in enumerate(v1):
        for k,tCur in enumerate(xDopF):
            qSub.submit_py('frg2Cur'+str(runner),'frg2DPAT'+str(k)+'N8U3v0'+str(j)+'v1'+str(i),str(nL),str(beTa),str(uU),\
                str(xDopC),str(tCur),str(v0Cur),str(v1Cur),str(0),str(NW),str(NK),str(Ns))
            runner=runner+1
