#!/usr/bin/python

import numpy as np
import time
import numpy.random as random
import scipy.sparse as sprs
import preProcess
import sys, os
from scipy.signal import convolve2d

cimport cython
cimport numpy as np
include 'utl.pxi'


#------------------------------------------------------------------
# FUNCTION/CLASS DEFINITIONS
#------------------------------------------------------------------

def genDistNew(sparseMatA,itrsWR,totalPts):
    '''
    comments
    '''
    newLocs = np.zeros((totalPts,itrsWR),dtype=int)
    indp = np.linspace(0,sparseMatA.shape[1]-1,sparseMatA.shape[1]).astype(int)
    for jjj in range(sparseMatA.shape[0]):
        aaa = np.ravel(sparseMatA[jjj,:].todense())
        ind = aaa!=0
        #newLocs[jjj,:] = random.choice(indp[ind],p=aaa[ind],size=itrsWR).astype(int)
        rind = random.randint(0,high=ind.sum(),size=itrsWR)
        newLocs[jjj,:] = indp[ind][rind]
    return newLocs


def fastGrad(oldI,oldJ,newI,newJ,oldGrad,pOld,pNew):
    '''
    comments
    '''
    nRows = pOld.shape[0]
    nCols = pOld.shape[1]
    newGrad = oldGrad
    for i in (-1,0,1):
        for j in (-1,0,1):
            if (abs(i) + abs(j) <2 ):
                if (oldI+i>=0) and (oldI+i <nRows) and (oldJ+j>=0) and (oldJ+j<nCols):
                    dif = abs(pOld[oldI,oldJ] - pOld[oldI+i,oldJ+j])
                    newGrad = newGrad - dif
                    dif = abs(pNew[oldI,oldJ] - pNew[oldI+i,oldJ+j])
                    newGrad = newGrad + dif

                if (newI+i>=0) and (newI+i <nRows) and (newJ+j>=0) and (newJ+j<nCols):
                    dif = abs(pOld[newI,newJ] - pOld[newI+i,newJ+j])
                    newGrad = newGrad - dif
                    dif = abs(pNew[newI,newJ] - pNew[newI+i,newJ+j])
                    newGrad = newGrad + dif
    return newGrad



def genSparseMat(allDoubles, nX, nZ, totalPts, numBlocks = 50):
    '''
    sparseMat = genSparseMat(allDoubles, nX, nZ, totalPts, numBlocks)

    generates the sparse matrix for use in the SOE routines; returns a csr matrix

    input:
        allDoubles:
            data set of doubles (edep1,x1,y1,z1,edep2,x2,y2,z2)
        nX,nZ:
            pixel size in the X and Z direction of the reconstruction
        totalPts:
            how many datapoints from allDoubles to use
        numBlocks:
            number of blocks of data used when computing the sparse matrix; this was tested to
            reduce memory footprint during backprojection

    output:
        sparseMat:
            a csr_matrix that contains the sparse matrix "backprojections" of the doubles; it is sized
            (totalPts, nX * nZ).  Each row is essentially the simple back projection of a single point, only vectorized
    '''
    oldstdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    nTot = nX * nZ
    blockSize = int(totalPts / numBlocks)
    sysMatTemp = np.zeros((blockSize, nTot))
    cntr = 0
    blkNum = 0
    gotBlock = 0
    sqrT = 0.3 * np.sqrt(nTot)

    for iii in range(len(allDoubles)):
        if np.logical_and(int(cntr / blockSize) == float(cntr / blockSize), blkNum == 1):
            if gotBlock == 1:
                sparseMat = sprs.csc_matrix(sysMatTemp).copy()
                sysMatTemp = np.zeros((blockSize, nTot))
                gotBlock = 0
        elif np.logical_and(int(cntr / blockSize) == float(cntr / blockSize), blkNum > 1):
            if gotBlock == 1:
                qS = sprs.csc_matrix(sysMatTemp)
                sparseMat = sprs.vstack((sparseMat, qS.copy()))
                sysMatTemp = np.zeros((blockSize, nTot))
                gotBlock = 0

        #print(iii)
        #h = preProcess.conicSection(allDoubles[iii:iii + 1, :], nx = nX, nz = nZ)
        h = preProcess.conicSectionMM(allDoubles[iii:iii + 1, :], nx = nX, nz = nZ, res = 2)
        if h.sum() > sqrT:
            q = np.reshape(h, (nTot,))
            q = q / h.sum()
            ind = cntr - blockSize * blkNum
            sysMatTemp[ind, :] = q.copy()
            cntr += 1
            if int(cntr / blockSize) == float(cntr / blockSize):
                blkNum += 1
                gotBlock = 1
        if cntr >= totalPts:
            qS = sprs.csc_matrix(sysMatTemp)
            sparseMat = sprs.vstack((sparseMat, qS.copy()))
            sysMatTemp = np.zeros((blockSize, nTot))
            break
    tStop = time.time()
    sys.stdout = oldstdout
    return sprs.csr_matrix(sparseMat)

def normSOEFast(sparseMat, int nX, int nZ,int loopTot = 500, itrsWR = 20):
    '''
    rcon = normSOEFast( sparseMat, int nX, int nZ,int loopTot, itrsWR)

    performs a simplistic SOE recontruction on the sparse matrix

    input:
        sparseMat:
            csr_matrix output from genSparseMat
        nX,nZ:
            pixel size in the X and Z direction of the reconstruction
        loopTot:
            total number of SOE loops to perform; each loop is over all data points
        itrsWR:
            ITeRationS Worth of Rands; number of random numbers generated at a time to save memory space

    output:
        rcon:
            reconstructed image, view with imshow(rcon)
    '''

    totalPts = sparseMat.shape[0]
    nTot = sparseMat.shape[1]
    newLocs = np.zeros((totalPts,itrsWR),dtype=int)
    oldLocs = np.zeros((totalPts,),dtype=int)
    p = np.zeros((nTot,))

    kludge = np.arange(nTot)
    cnt = 0

    for iii in range(loopTot):
        mvCount = 0
        if (iii == 0) or (iii%itrsWR)==0:
            newLocs = genDistNew(sparseMat,itrsWR,totalPts)
            cnt = 0
        if iii==0:
            p = p + np.bincount(np.concatenate((newLocs[:,cnt],kludge)),np.ones((nTot+totalPts,))).astype(int)
            p = p - 1
            oldLocs = newLocs[:,cnt].copy()
        else:
            myRands = random.rand(totalPts)
            #p += 1
    #        rats = (p[newLocs[:,cnt]]+1)/p[oldLocs]
    #        inds = rats > myRands
    #        p[oldLocs[inds]] -= 1
    #        p[newLocs[inds,cnt]] += 1
    #        oldLocs[inds] = newLocs[inds,cnt].copy()
            for jjj in range(totalPts):
                if (p[newLocs[jjj,cnt]]+1)/p[oldLocs[jjj]] > myRands[jjj]:
                    p[oldLocs[jjj]] -= 1
                    p[newLocs[jjj,cnt]] += 1
                    oldLocs[jjj] = newLocs[jjj,cnt]
                    mvCount += 1
            #p -= 1
        cnt += 1
        print('Finished iteration %i' % iii)
        print('Moved Perc: ', float(mvCount)/totalPts)

    rcon = np.reshape(p,(nZ,nX))
    return rcon

def normSOEFastG( sparseMat, int nX, int nZ,int loopTot=500, itrsWR = 20,beta=1):
    '''
    rcon = normSOEFastG( sparseMat, int nX, int nZ,int loopTot, itrsWR,beta)

    performs a SOE recontruction on the sparse matrix using gradient biasing

    input:
        sparseMat:
            csr_matrix output from genSparseMat
        nX,nZ:
            pixel size in the X and Z direction of the reconstruction
        loopTot:
            total number of SOE loops to perform; each loop is over all data points
        itrsWR:
            ITeRationS Worth of Rands; number of random numbers generated at a time to save memory space
        beta:
            the power the gradient bias term is raised to

    output:
        rcon:
            reconstructed image, view with imshow(rcon)
    '''
    totalPts = sparseMat.shape[0]
    nTot = sparseMat.shape[1]
    newLocs = np.zeros((totalPts,itrsWR),dtype=int)
    oldLocs = np.zeros((totalPts,),dtype=int)
    p = np.zeros((nTot,))
    rconNew = np.reshape(p,(nZ,nX))
    kludge = np.arange(nTot)
    cnt = 0
    penOld = 1
    penNew = 1

    for iii in range(loopTot):
        mvCount = 0
        if (iii == 0) or (iii%itrsWR)==0:
            newLocs = genDistNew(sparseMat,itrsWR,totalPts)
            cnt = 0
            gz,gx = np.abs(np.gradient(rconNew))
            penOld = np.sum(gz)+np.sum(gx)
        if iii==0:
            p = p + np.bincount(np.concatenate((newLocs[:,cnt],kludge)),np.ones((nTot+totalPts,))).astype(int)
            p = p - 1
            rconOld = np.reshape(p,(nZ,nX))
            rconNew = np.reshape(p,(nZ,nX))
            oldLocs = newLocs[:,cnt].copy()
            gz,gx = np.abs(np.gradient(rconNew))
            penOld = np.sum(gz)+np.sum(gx)
            penStart = penOld
        else:
            myRands = random.rand(totalPts)
            for jjj in range(totalPts):
                c = newLocs[jjj,cnt]
                z = int(c/nX)
                x = int(c-nX*z)
                cOld = oldLocs[jjj]
                zOld = int(cOld/nX)
                xOld = int(cOld-nX*zOld)
                if iii > loopTot/2:

                    rconTemp = rconNew.copy()
                    rconTemp[z,x] += 1
                    rconTemp[zOld,xOld] -= 1
                    #rconOld[zOld,xOld] -= 1
                    penNew = fastGrad(zOld,xOld,z,x,penOld,rconOld,rconTemp)
                    if penNew == 0:
                        penNew = np.mean(gz+gx)
                        R = penOld/penNew
                    else:
                        R = abs(penOld/penNew)
                        R = R**beta

                    if R*(rconNew[z,x]+1)/(rconOld[zOld,xOld]) > myRands[jjj]:
                        rconNew[z,x] += 1
                        rconNew[zOld,xOld] -= 1
                        oldLocs[jjj] = newLocs[jjj,cnt]
                        mvCount += 1
                        penOld = penNew
                        rconOld = rconNew

                else:
                    if (rconNew[z,x]+1)/(rconOld[zOld,xOld]) > myRands[jjj]:
                        rconNew[z,x] += 1
                        rconNew[zOld,xOld] -= 1
                        rconOld = rconNew
                        oldLocs[jjj] = newLocs[jjj,cnt]
                        mvCount += 1
            #p -= 1
        cnt += 1
        print('Finished iteration %i' % iii)
        print('Moved Perc: ', float(mvCount)/totalPts)

    #rcon = np.reshape(p,(nZ,nX))
    return rconNew

def normSOEFastSparse( sparseMat, totalPts, int nX, int nZ,int loopTot=5000, itrsWR = 500,beta = 1):
    '''
    rcon = normSOEFastSparse( sparseMat, int nX, int nZ,int loopTot, itrsWR,beta)

    performs a SOE recontruction on the sparse matrix using sparsity biasing

    input:
        sparseMat:
            csr_matrix output from genSparseMat
        nX,nZ:
            pixel size in the X and Z direction of the reconstruction
        loopTot:
            total number of SOE loops to perform; each loop is over all data points
        itrsWR:
            ITeRationS Worth of Rands; number of random numbers generated at a time to save memory space
        beta:
            the power the sparsity bias term is raised to

    output:
        rcon:
            reconstructed image, view with imshow(rcon)
    '''

    #totalPts = sparseMat.shape[0]
    nTot = nX*nZ
    newLocs = np.zeros((totalPts,itrsWR),dtype=int)
    oldLocs = np.zeros((totalPts,),dtype=int)
    p = np.zeros((nTot,))
    penOld = 1
    penNew = 1
    kludge = np.arange(nTot)
    cnt = 0

    for iii in range(loopTot):
        mvCount = 0
        if (iii == 0) or (iii%itrsWR)==0:

            #sparseMat = sprs.csr_matrix(genSparseMat(smearPixU(doubles),nX,nZ,totalPts,itrsWR))
            newLocs = genDistNew(sparseMat,itrsWR,totalPts)
            cnt = 0
            penOld = np.sum(p>0)
        if iii==0:
            p = p + np.bincount(np.concatenate((newLocs[:,cnt],kludge)),np.ones((nTot+totalPts,))).astype(int)
            p = p - 1
            oldLocs = newLocs[:,cnt].copy()
            penOld = np.sum(p>0)
        else:
            myRands = random.rand(totalPts)
            #p += 1
    #        rats = (p[newLocs[:,cnt]]+1)/p[oldLocs]
    #        inds = rats > myRands
    #        p[oldLocs[inds]] -= 1
    #        p[newLocs[inds,cnt]] += 1
    #        oldLocs[inds] = newLocs[inds,cnt].copy()
            for jjj in range(totalPts):
                if iii >= loopTot/2:

                    if(iii == loopTot/2 and jjj==0):
                        penOld = np.sum(p>0)

                    penNew = penOld
                    if newLocs[jjj,cnt]!=oldLocs[jjj]:
                        if p[newLocs[jjj,cnt]]==0:
                            penNew += 1
                        if p[oldLocs[jjj]] == 1:
                            penNew -= 1

                    R = penOld/penNew
                    R = R**beta
                else:
                    R = 1

                if R*(p[newLocs[jjj,cnt]]+1)/p[oldLocs[jjj]] > myRands[jjj]:
                    p[oldLocs[jjj]] -= 1
                    p[newLocs[jjj,cnt]] += 1
                    oldLocs[jjj] = newLocs[jjj,cnt]
                    mvCount += 1
                    penOld = penNew
            #p -= 1
        cnt += 1
        print('Finished iteration %i' % iii)
        print('Moved Perc: ', float(mvCount)/totalPts)

    rcon = np.reshape(p,(nZ,nX))
    return rcon

def normSOEFastGS( sparseMat, int nX, int nZ,int loopTot=500, itrsWR = 20,betaG=1,betaS=1):
    '''
    rcon = normSOEFastGS( sparseMat, int nX, int nZ,int loopTot, itrsWR,betaG,betaS)

    performs a SOE recontruction on the sparse matrix using combined gradient and sparsity biasing

    input:
        sparseMat:
            csr_matrix output from genSparseMat
        nX,nZ:
            pixel size in the X and Z direction of the reconstruction
        loopTot:
            total number of SOE loops to perform; each loop is over all data points
        itrsWR:
            ITeRationS Worth of Rands; number of random numbers generated at a time to save memory space
        betaG:
            the power the gradient bias term is raised to
        betaS:
            the power the sparisty bias term is raised to

    output:
        rcon:
            reconstructed image, view with imshow(rcon)
    '''
    totalPts = sparseMat.shape[0]
    nTot = sparseMat.shape[1]
    newLocs = np.zeros((totalPts,itrsWR),dtype=int)
    oldLocs = np.zeros((totalPts,),dtype=int)
    p = np.zeros((nTot,))
    rconNew = np.reshape(p,(nZ,nX))
    kludge = np.arange(nTot)
    cnt = 0
    penOldG = 1
    penNewG = 1
    penOldS = 1
    penNewS = 1

    for iii in range(loopTot):
        mvCount = 0
        if (iii == 0) or (iii%itrsWR)==0:
            newLocs = genDistNew(sparseMat,itrsWR,totalPts)
            cnt = 0
            gz,gx = np.abs(np.gradient(rconNew))
            penOldG = np.sum(gz)+np.sum(gx)
            penOldS = np.sum(rconNew>0)
        if iii==0:
            p = p + np.bincount(np.concatenate((newLocs[:,cnt],kludge)),np.ones((nTot+totalPts,))).astype(int)
            p = p - 1
            rconOld = np.reshape(p,(nZ,nX))
            rconNew = np.reshape(p,(nZ,nX))
            oldLocs = newLocs[:,cnt].copy()
            gz,gx = np.abs(np.gradient(rconNew))
            penOldG = np.sum(gz)+np.sum(gx)
            penStart = penOldG
            penOldS = np.sum(rconNew>0)
        else:
            myRands = random.rand(totalPts)
            for jjj in range(totalPts):
                c = newLocs[jjj,cnt]
                z = int(c/nX)
                x = int(c-nX*z)
                cOld = oldLocs[jjj]
                zOld = int(cOld/nX)
                xOld = int(cOld-nX*zOld)

                if(iii == loopTot/2 and jjj==0):
                    penOldS = np.sum(rconNew>0)

                if iii > loopTot/2:
                    penNewS = penOldS
                    if newLocs[jjj,cnt]!=oldLocs[jjj]:
                        if rconNew[z,x]==0:
                            penNewS += 1
                        if rconNew[zOld,xOld] == 1:
                            penNewS -= 1

                    Rs = penOldS/penNewS
                    Rs = Rs**betaS


                    rconTemp = rconNew.copy()
                    rconTemp[z,x] += 1
                    rconTemp[zOld,xOld] -= 1
                    #rconOld[zOld,xOld] -= 1
                    penNewG = fastGrad(zOld,xOld,z,x,penOldG,rconOld,rconTemp)
                    if penNewG == 0:
                        penNewG = np.mean(gz+gx)
                        Rg = penOldG/penNewG
                    else:
                        Rg = abs(penOldG/penNewG)
                        Rg = Rg**betaG

                    if Rg*Rs*(rconNew[z,x]+1)/(rconOld[zOld,xOld]) > myRands[jjj]:
                        rconNew[z,x] += 1
                        rconNew[zOld,xOld] -= 1
                        oldLocs[jjj] = newLocs[jjj,cnt]
                        mvCount += 1
                        penOldG = penNewG
                        rconOld = rconNew
                        penOldS = penNewS

                else:
                    if (rconNew[z,x]+1)/(rconOld[zOld,xOld]) > myRands[jjj]:
                        rconNew[z,x] += 1
                        rconNew[zOld,xOld] -= 1
                        rconOld = rconNew
                        oldLocs[jjj] = newLocs[jjj,cnt]
                        mvCount += 1
            #p -= 1
        cnt += 1
        print('Finished iteration %i' % iii)
        print('Moved Perc: ', float(mvCount)/totalPts)

    #rcon = np.reshape(p,(nZ,nX))
    return rconNew
