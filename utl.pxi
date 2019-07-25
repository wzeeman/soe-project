from libc.stdio cimport fopen, fclose, FILE, fscanf, fgetc, fgets, fprintf, feof, ftell, fseek, SEEK_SET, printf
from libc.math cimport sin, cos, acos, sqrt, fabs, tan, atan, exp, log
# from libc.stdlib cimport atoi
import numpy as np
import time, struct
cimport numpy as np

from cython.parallel import parallel, prange
cimport cython


PI = np.pi
ZERO = 1e-5
PARALLEL = 1e5
OUTOFBOUNDARY = 1e6
IMPOSSIBLE = -1e6
rhoCZT = 5.8
eCZT = np.array( [  1.00000000e-03,   1.00300000e-03,   1.00600000e-03,
                    1.00601000e-03,   1.01300000e-03,   1.02000000e-03,
                    1.02001000e-03,   1.03100000e-03,   1.04300000e-03,
                    1.04301000e-03,   1.11600000e-03,   1.19400000e-03,
                    1.19401000e-03,   1.50000000e-03,   2.00000000e-03,
                    3.00000000e-03,   3.53700000e-03,   3.53701000e-03,
                    3.63100000e-03,   3.72700000e-03,   3.72701000e-03,
                    4.00000000e-03,   4.01800000e-03,   4.01801000e-03,
                    4.17700000e-03,   4.34100000e-03,   4.34101000e-03,
                    4.47500000e-03,   4.61200000e-03,   4.61201000e-03,
                    4.77300000e-03,   4.93900000e-03,   4.93901000e-03,
                    5.00000000e-03,   6.00000000e-03,   8.00000000e-03,
                    9.65900000e-03,   9.65901000e-03,   1.00000000e-02,
                    1.50000000e-02,   2.00000000e-02,   2.67100000e-02,
                    2.67101000e-02,   3.00000000e-02,   3.18100000e-02,
                    3.18101000e-02,   4.00000000e-02,   5.00000000e-02,
                    6.00000000e-02,   8.00000000e-02,   1.00000000e-01,
                    1.50000000e-01,   2.00000000e-01,   3.00000000e-01,
                    4.00000000e-01,   5.00000000e-01,   6.00000000e-01,
                    8.00000000e-01,   1.00000000e+00,   1.02200000e+00,
                    1.25000000e+00,   1.50000000e+00,   2.00000000e+00,
                    2.04400000e+00,   3.00000000e+00,   4.00000000e+00,
                    5.00000000e+00,   6.00000000e+00,   7.00000000e+00,
                    8.00000000e+00,   9.00000000e+00,   1.00000000e+01 ], dtype=np.float64 )

attenCZT = np.array ( [ 7.60300000e+03,   7.55300000e+03,   7.50500000e+03,
                        7.68200000e+03,   7.56700000e+03,   7.45400000e+03,
                        7.55600000e+03,   7.43700000e+03,   7.34500000e+03,
                        7.43200000e+03,   6.42800000e+03,   5.56400000e+03,
                        5.61500000e+03,   3.36400000e+03,   1.69800000e+03,
                        6.24800000e+02,   4.12100000e+02,   7.69600000e+02,
                        7.21500000e+02,   6.76100000e+02,   8.45300000e+02,
                        7.10400000e+02,   7.02500000e+02,   7.79000000e+02,
                        7.06800000e+02,   6.41400000e+02,   9.01600000e+02,
                        8.41800000e+02,   7.86100000e+02,   9.08600000e+02,
                        8.36700000e+02,   7.70600000e+02,   8.32200000e+02,
                        8.07200000e+02,   5.08100000e+02,   2.39400000e+02,
                        1.45400000e+02,   1.56300000e+02,   1.42700000e+02,
                        4.82500000e+01,   2.22100000e+01,   1.01700000e+01,
                        2.89900000e+01,   2.14800000e+01,   1.84500000e+01,
                        3.36800000e+01,   1.85900000e+01,   1.02700000e+01,
                        6.29800000e+00,   2.90700000e+00,   1.61100000e+00,
                        5.88100000e-01,   3.16300000e-01,   1.60300000e-01,
                        1.13700000e-01,   9.24900000e-02,   8.02500000e-02,
                        6.60300000e-02,   5.75300000e-02,   5.67900000e-02,
                        5.05500000e-02,   4.60400000e-02,   4.07900000e-02,
                        4.04700000e-02,   3.64900000e-02,   3.51700000e-02,
                        3.49800000e-02,   3.52800000e-02,   3.58700000e-02,
                        3.65900000e-02,   3.73900000e-02,   3.82300000e-02], dtype=np.float64)

peCZT = np.array([  7.59500000e+03,   7.54600000e+03,   7.49700000e+03,
                    7.67400000e+03,   7.55900000e+03,   7.44600000e+03,
                    7.54800000e+03,   7.42900000e+03,   7.33700000e+03,
                    7.42400000e+03,   6.42100000e+03,   5.55600000e+03,
                    5.60700000e+03,   3.35700000e+03,   1.69100000e+03,
                    6.18700000e+02,   4.06400000e+02,   7.64000000e+02,
                    7.15900000e+02,   6.70600000e+02,   8.39800000e+02,
                    7.05100000e+02,   6.97300000e+02,   7.73700000e+02,
                    7.01700000e+02,   6.36300000e+02,   8.96500000e+02,
                    8.36900000e+02,   7.81200000e+02,   9.03700000e+02,
                    8.32000000e+02,   7.66000000e+02,   8.27500000e+02,
                    8.02600000e+02,   5.04100000e+02,   2.36200000e+02,
                    1.42700000e+02,   1.53600000e+02,   1.40100000e+02,
                    4.64800000e+01,   2.09300000e+01,   9.27600000e+00,
                    2.81000000e+01,   2.07100000e+01,   1.77300000e+01,
                    3.29600000e+01,   1.80500000e+01,   9.85000000e+00,
                    5.95600000e+00,   2.65400000e+00,   1.40500000e+00,
                    4.37400000e-01,   1.91200000e-01,   6.08800000e-02,
                    2.79500000e-02,   1.57200000e-02,   1.00500000e-02,
                    5.19400000e-03,   3.24300000e-03,   3.08600000e-03,
                    2.08200000e-03,   1.49100000e-03,   9.08200000e-04,
                    8.76300000e-04,   4.82500000e-04,   3.20100000e-04,
                    2.36900000e-04,   1.87100000e-04,   1.54200000e-04,
                    1.30900000e-04,   1.13600000e-04,   1.00300000e-04], dtype=np.float64)



cdef double norm( double x, double y, double z ):
    '''
    n = norm( double x, double y, double z )
    compute the norm of a vector
    '''
    return sqrt( x * x + y * y + z * z )


cpdef np.ndarray cross( double x1, double y1, double z1, double x2, double y2, double z2 ):
    '''
    c = cross( double x1, double y1, double z1, double x2, double y2, double z2 )

    compute the cross product of two vectors

    input:
        x1, y1, z1:
            double, vector 1
        x2, y2, z2:
            double, vector 2

    ouput:
        c:
            np.array contains x, y, z
    '''

    cdef double x, y, z
    x = y1 * z2 - y2 * z1
    y = x2 * z1 - x1 * z2
    z = x1 * y2 - x2 * y1
    return np.array( [ x, y,  z] )


cpdef double vectorDot( double x1, double y1, double z1,
                        double x2, double y2, double z2 ):
    '''
    d = vectorDot(  double x1, double y1, double z1,
                    double x2, double y2, double z2 )

    compute the dot product of two vectors
    input:
        x1, y1, z1:
            double, vector 1
        x2, y2, z2:
            double, vector 2

    output:
        d:
            double, the dot product of vector 1 & 2
    '''
    return  (x1 * x2 + y1 * y2 + z1 * z2)


cpdef np.ndarray matrixVectorDot(   double a11, double a12, double a13,
                                    double a21, double a22, double a23,
                                    double a31, double a32, double a33,
                                    double x, double y, double z ):
    '''
    d =  matrixVectorDot(    double a11, double a12, double a13,
                                    double a21, double a22, double a23,
                                    double a31, double a32, double a33,
                                    double x, double y, double z )

    compute the dot product of a 3x3 matrix and a vector of 3 elements.

    input:
        a11, a12, a13:
            double, the first row of the 3x3 matrix
        a21, a22, a23:
            double, the second row
        a31, a32, a33:
            double, the third row
        x, y, z:
            double, the vector
    output:
        d:
            np.array, x', y', z' of the transformed x, y, z

    '''
    cdef:
        double X, Y, Z

    X = a11 * x + a12 * y + a13 * z
    Y = a21 * x + a22 * y + a23 * z
    Z = a31 * x + a32 * y + a33 * z

    return np.array( [ X, Y, Z ] )


cpdef np.ndarray swap( double x, double y):
    '''
    s = swap( double x, double y)

    swap the value of two variables

    input:
        x, y:
            double, the two variables to be swapped

    output:
        s:
            ndarray of swapped x and y
    '''
    cdef:
        double a, b
    a = y
    b = x
    return np.array( [ a, b ] )


cpdef double lineDistance(  double x1, double y1, double z1,
                            double x2, double y2, double z2,
                            double x3, double y3, double z3,
                            double x4, double y4, double z4 ):
    '''
    d = lineDistance(   double x1, double y1, double z1,
                        double x2, double y2, double z2,
                        double x3, double y3, double z3,
                        double x4, double y4, double z4 )


    compute the distance between two vectors

    input:
        [x1, y1, z1] and [x2, y2, z2]:  defines one vector, v
        [x3, y3, z3] and [x4, y4, z4]:  defines the other vector, u

    ouput:
        d: the distance between vector u and vector v
    '''


    cdef:
        double   vx, vy, vz, ux, uy, uz
        double   vnorm, nunorm
        double   Nx, Ny, Nz
        double   temp

    # vector u and v
    ux = x2 - x1
    uy = y2 - y1
    uz = z2 - z1
    vx = x4 - x3
    vy = y4 - y3
    vz = z4 - z3
    unorm = norm( ux, uy, uz )
    vnorm = norm( vx, vy, vz )
    ux = ux / unorm
    uy = uy / unorm
    uz = uz / unorm
    vx = vx / vnorm
    vy = vy / vnorm
    vz = vz / vnorm

    # vector normal to both u and v
    Nx, Ny, Nz = cross( ux, uy, uz, vx, vy, vz )
    Nnorm = norm( Nx, Ny, Nz )
    Nx = Nx / Nnorm
    Ny = Ny / Nnorm
    Nz = Nz / Nnorm

    # if the dot product of u and v is 1, then the two vector parallel
    if fabs( fabs(vectorDot( ux, uy, uz, vx, vy, vz )) - 1.0 ) < 1e-6:
        temp = vectorDot( x3 - x1, y3 - y1, z3 - z1, ux, uy, uz )
        return sqrt( norm( x3 - x1, y3 - y1, z3 - z1 )**2 - temp**2 )
    else:
        d = vectorDot( x3 - x1, y3 - y1, z3 - z1, Nx, Ny, Nz )
        return fabs( d )


cpdef tuple lineDistanceAdvanced(   float Ax, float Ay, float Az,
                                    float Bx, float By, float Bz,
                                    float Cx, float Cy, float Cz,
                                    float Dx, float Dy, float Dz ):
    '''
    PQnorm, Px, Py, Pz, Qx, Qy, Qz = lineDistanceAdvanced(  float Ax, float Ay, float Az,
                                                            float Bx, float By, float Bz,
                                                            float Cx, float Cy, float Cz,
                                                            float Dx, float Dy, float Dz )

    computes the distance between two vectors/lines and also returns the two points P and Q
    on vector v and u

    input:
        Ax, Ay, Az, Bx, By, Bz: defines the vector v
        Cx, Cy, Cz, Dx, Cy, Cz: defines the vector u

    output:
        PQnorm: the distance between vectors v and u
        Px, Py, Pz: the point on vector v where PQ perpendicular to both v and u
        Qx, Qy, Qz: the point on vector u where PQ perpendicular to both v and u
    '''

    cdef:
        float   ux=0.0, uy=0.0, uz=0.0, vx=0.0, vy=0.0, vz=0.0
        float   ABnorm= 0.0, CDnorm=0.0, uvDot=0.0
        float   vDBdot=0.0, uDBdot=0.0
        float   Px=0.0, Py=0.0, Pz=0.0, Qx=0.0, Qy=0.0, Qz=0.0
        float   PQnorm=0.0

    ABnorm = norm( Ax - Bx, Ay - By, Az - Bz )
    CDnorm = norm( Cx - Dx, Cy - Dy, Cz - Dz )
    vx = ( Ax - Bx ) / ABnorm
    vy = ( Ay - By ) / ABnorm
    vz = ( Az - Bz ) / ABnorm
    ux = ( Cx - Dx ) / CDnorm
    uy = ( Cy - Dy ) / CDnorm
    uz = ( Cz - Dz ) / CDnorm

    uvDot = vectorDot( vx, vy, vz, ux, uy, uz )
    if fabs( uvDot ) == 1:
        return OUTOFBOUNDARY, Px, Py, Pz, Qx, Qy, Qz

    vDBdot = vectorDot( Bx - Dx, By - Dy, Bz - Dz, vx, vy, vz )
    uDBdot = vectorDot( Bx - Dx, By - Dy, Bz - Dz, ux, uy, uz )

    s = ( vDBdot * uvDot - uDBdot ) / ( uvDot * uvDot - 1.0 )
    t = ( vDBdot - uDBdot * uvDot ) / ( uvDot * uvDot - 1.0 )

    Px = Bx + t * vx
    Py = By + t * vy
    Pz = Bz + t * vz
    Qx = Dx + s * ux
    Qy = Dy + s * uy
    Qz = Dz + s * uz
    PQnorm = norm( Qx - Px, Qy - Py, Qz - Pz )
    return PQnorm, Px, Py, Pz, Qx, Qy, Qz


def formEventMatrix( eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz):
    '''
    events = formEventMatrix( eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz)

    form a data matrix

    input:
        eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz: each is a 1-D vector
    output:
        events: the event matrix
    '''
    #
    # cdef int nEvents, i
    # nEvents = len( eDepA )
    # events = np.zeros( (nEvents, 8) )
    # for i in range( nEvents ):
    #     events[i] =  np.array( [ eDepA[i], Ax[i], Ay[i], Az[i], eDepB[i], Bx[i], By[i], Bz[i] ] )
    #     # events[i, 0] = eDepA[i]
    #     # events[i, 1] = Ax[i]
    #     # events[i, 2] = Ay[i]
    #     # events[i, 3] = Az[i]
    #     # events[i, 4] = eDepB[i]
    #     # events[i, 5] = Bx[i]
    #     # events[i, 6] = By[i]
    #     # events[i, 7] = Bz[i]

    events = np.vstack( ( eDepA, Ax ) )
    events = np.vstack( ( events, Ay ) )
    events = np.vstack( ( events, Az ) )
    events = np.vstack( ( events, eDepB ) )
    events = np.vstack( ( events, Bx ) )
    events = np.vstack( ( events, By ) )
    events = np.vstack( ( events, Bz ) )
    return events.T

def pseudoCamera( events, angle=PI/2):
    '''
    pseudoEvents = pseudoCamera( events, angle=PI/2 )

    rotate events around z axis
    NOTE: rotate events/vector, not the coordinates

    input:
        events: the event matrix to be rotated
        angle:  the angle to be rotated
    output:
        pseudoEvents: rotated event coordinates
    '''

    pseudoEvents = events.copy()
    pseudoEvents[:,1] = cos( angle ) * events[:,1] - sin( angle ) * events[:,2]
    pseudoEvents[:,2] = sin( angle ) * events[:,1] + cos( angle ) * events[:,2]
    pseudoEvents[:,5] = cos( angle ) * events[:,5] - sin( angle ) * events[:,6]
    pseudoEvents[:,6] = sin( angle ) * events[:,5] + cos( angle ) * events[:,6]
    return pseudoEvents

def pseudoCamera2( events, angle=90.0, proportion=0.5):
    '''
    pseudoEvents, index = pseudoCamera( events, angle=90.0, proportion=0.5 )

    take a data matrix, randomly select a portion (proportion parametr) of it
    and rotate them around z axis
    NOTE: rotate events/vector, not the coordinates

    input:
        events: the event matrix; THIS IS THE ENTIRE MATRIX, A PORTION OF IT ()
        angle:  the angle to be rotated
        proportion: float, the proportion of the data to be rotated
    output:
        pseudoEvents: pseudo two camera events
        index: the index of the events in the input data matrix got rotated.
    '''
    angle = angle / 180 * PI

    N = len( events )
    n = int( N * proportion )
    ind = np.sort( np.random.permutation( N )[:n] )

    pseudoEvents = events.copy()
    pseudoEvents[ind,1] = cos( angle ) * events[ind,1] - sin( angle ) * events[ind,2]
    pseudoEvents[ind,2] = sin( angle ) * events[ind,1] + cos( angle ) * events[ind,2]
    pseudoEvents[ind,5] = cos( angle ) * events[ind,5] - sin( angle ) * events[ind,6]
    pseudoEvents[ind,6] = sin( angle ) * events[ind,5] + cos( angle ) * events[ind,6]
    return pseudoEvents, ind


def swapEvent( np.ndarray event ):
    '''
    ev = swapEvent( np.ndarray event )

    swap event ordering

    input:
        event: event matrix
    output:
        ev:    swapped event matrix
    '''
    ev = np.zeros( np.shape( event ) )
    ev[:, 0:4] = event[:,4:]
    ev[:, 4:]  = event[:,0:4]
    return ev


cpdef double interp1d( np.ndarray xData, np.ndarray yData, double x):
    '''
    y = interp1d( np.ndarray xData, np.ndarray yData, double x)
    interpolating in 1D
    '''

    cdef:
        int N = len( xData ), i=0
        double x0, y0, x1, y1, y=0.0


    if xData.ndim != 1:
        print( 'xData must be 1D array' )
        return -OUTOFBOUNDARY
    if yData.ndim != 1:
        print( 'yData must be 1D array' )
        return -OUTOFBOUNDARY

    if x < xData[0]:
        x0 = xData[0]
        y0 = yData[0]
        x1 = xData[1]
        y1 = yData[1]
    elif x > xData[N-1]:
        x0 = xData[N-2]
        y0 = yData[N-2]
        x1 = xData[N-1]
        y1 = yData[N-1]
    else:
        while( x > xData[i]  and i < N-1 ):
            i += 1
        x0 = xData[i-1]
        y0 = yData[i-1]
        x1 = xData[i]
        y1 = yData[i]

    y = ( y1 - y0 ) / ( x1 - x0 ) * ( x - x0 ) + y0
    return y


def KleinNishita_solidAngle( E0 ):
    '''
    theta, kn = KleinNishita_solidAngle( E0 )

    computes Klein Nishita differential cross section with respect to
    solid angle, d_sigma / d_Omega

    input:
        E0: the initial energy of the incoming photon

    output:
        theta:  the scatter angle, from 0 to Pi, in steps of 0.001
        kn:     the Klein Nishita equation
    '''

    theta = np.arange( 0, np.pi, 0.001 )
    E1 = 1 / ( 1 / E0 + ( 1 - np.cos( theta) ) / 0.511 )
    kn = ( E1 / E0 )**2 * ( E1 / E0 + E0 / E1 - np.sin( theta)**2 )
    return theta, kn


def KleinNishita_theta( E0 ):
    '''
    theta, kn = KleinNishita_theta( E0 )

    computes the Klen Nishita differential cross section with respect to scatter
    angle, d_sigma / d_theta

    input:
        E0: the initial energy of the incoming photon

    output:
        theta:  the scatter angle, from 0 to Pi, in steps of 0.001
        kn:     the Klein Nishita equation
    '''

    theta = np.arange( 0, np.pi, 0.001 )
    E1 = 1 / ( 1 / E0 + ( 1 - np.cos( theta) ) / 0.511 )
    kn = ( E1 / E0 )**2 * ( E1 / E0 + E0 / E1 - np.sin( theta)**2 ) * 2 * np.pi * np.sin( theta )
    return theta, kn


def KleinNishita_eDep( E0 ):
    '''
    eDep, kn = KleinNishita_eDep( E0 )

    computes the Klen Nishita differential cross section with respect to deposited
    energy, d_sigma / d_eDep

    input:
        E0: the initial energy of the incoming photon

    output:
        eDep:   the deposited energy, from 0 to Compton edge, in steps of 0.001
        kn:     the Klein Nishita equation
    '''

    eDep_max = E0 * ( 1 - 1 / ( 1 + 2 * E0 / 0.511 ) )
    eDep = np.arange( 0, eDep_max, 0.001 )
    E1 = E0 - eDep
    theta = np.arccos( 1 - 0.511 * ( 1 / E1 - 1/ E0 ) )
    p = ( E1 / E0 )**2 * ( E1 / E0 + E0 / E1 - np.sin(theta)**2 ) * 2 * np.pi * 0.511 / E1**2
    return eDep, p


# cpdef double entraceOD( double sx=0.0, double sy=0.0, double sz=0.0,
#                         double Ax=0.0, double Ay=0.0, double Az=0.0,
#                         double cameraX=0.0, double cameraY=0.0, double cameraZ=0.0):
#     '''
#         d = entranceOD(   double sx=0.0, double sy=0.0, double sz=0.0,
#                             double Ax=0.0, double Ay=0.0, double Az=0.0,
#                             double cameraX=0.0, double cameraY=0.0, double cameraZ=0.0)
#
#
#         compute the optical depth inside the crystals
#
#         input:
#             sx, sy, sz, Ax, Ay, Az:
#                 double, coordiantes of source and the first event
#             cameraX, cameraZ:
#                 double, the x and z coordiantes of projection of isocenter on the camera
#             cameraY:
#                 double, the distacne from isocenter to camera surface.
#         output:
#             d:
#             double, the entrace optical depth
#     '''
#
#     cdef:
#         # these are the boundaries of each crystal
#         float       crystal_x1=0.0, crystal_x2=0.0, crystal_y1=0.0, crystal_y2=0.0, crystal_z1=0.0, crystal_z2=0.0
#         # t, x, y, z is the solution if the line intercepts with a crystal surface
#         float       t=0.0, x=0.0, y=0.0, z=0.0
#         float       dis=0.0
#         int         i=0, N=64
#         # point1 and point2 are used to check if there is one or two intercepts with a crystal
#         # p1 and p2 are the coordinates of interceptions.
#         bint        point1=False, point2=False
#         np.ndarray[np.float64_t, ndim=1]  p1 = np.zeros( 3, dtype=np.float64 )
#         np.ndarray[np.float64_t, ndim=1]  p2 = np.zeros( 3, dtype=np.float64 )
#         # this is the crystal boundaries, centered, in the format of
#         # ( lower_x, upper_x, lower_y, upper_y, lower_z, upper_z )
#         # in the coordiantes of CORE
#         # camera surface is y=0.
#         np.ndarray[np.float64_t, ndim=2] crystals=np.array( [   [-48, -28, 38.32, 48.32, -101, -81],
#                                                                 [-48, -28, 38.32, 48.32, -78, -58],
#                                                                 [-48, -28, 38.32, 48.32, -48, -28],
#                                                                 [-48, -28, 38.32, 48.32, -25, -5],
#                                                                 [-48, -28, 38.32, 48.32, 5, 25],
#                                                                 [-48, -28, 38.32, 48.32, 28, 48],
#                                                                 [-48, -28, 38.32, 48.32, 58, 78],
#                                                                 [-48, -28, 38.32, 48.32, 81, 101],
#                                                                 [-25, -5, 38.32, 48.32, -101, -81],
#                                                                 [-25, -5, 38.32, 48.32, -78, -58],
#                                                                 [-25, -5, 38.32, 48.32, -48, -28],
#                                                                 [-25, -5, 38.32, 48.32, -25, -5],
#                                                                 [-25, -5, 38.32, 48.32, 5, 25],
#                                                                 [-25, -5, 38.32, 48.32, 28, 48],
#                                                                 [-25, -5, 38.32, 48.32, 58, 78],
#                                                                 [-25, -5, 38.32, 48.32, 81, 101],
#                                                                 [5, 25, 38.32, 48.32, -101, -81],
#                                                                 [5, 25, 38.32, 48.32, -78, -58],
#                                                                 [5, 25, 38.32, 48.32, -48, -28],
#                                                                 [5, 25, 38.32, 48.32, -25, -5],
#                                                                 [5, 25, 38.32, 48.32, 5, 25],
#                                                                 [5, 25, 38.32, 48.32, 28, 48],
#                                                                 [5, 25, 38.32, 48.32, 58, 78],
#                                                                 [5, 25, 38.32, 48.32, 81, 101],
#                                                                 [28, 48, 38.32, 48.32, -101, -81],
#                                                                 [28, 48, 38.32, 48.32, -78, -58],
#                                                                 [28, 48, 38.32, 48.32, -48, -28],
#                                                                 [28, 48, 38.32, 48.32, -25, -5],
#                                                                 [28, 48, 38.32, 48.32, 5, 25],
#                                                                 [28, 48, 38.32, 48.32, 28, 48],
#                                                                 [28, 48, 38.32, 48.32, 58, 78],
#                                                                 [28, 48, 38.32, 48.32, 81, 101],
#                                                                 [-48, -28, 73.88, 88.88, -101, -81],
#                                                                 [-48, -28, 73.88, 88.88, -78, -58],
#                                                                 [-48, -28, 73.88, 88.88, -48, -28],
#                                                                 [-48, -28, 73.88, 88.88, -25, -5],
#                                                                 [-48, -28, 73.88, 88.88, 5, 25],
#                                                                 [-48, -28, 73.88, 88.88, 28, 48],
#                                                                 [-48, -28, 73.88, 88.88, 58, 78],
#                                                                 [-48, -28, 73.88, 88.88, 81, 101],
#                                                                 [-25, -5, 73.88, 88.88, -101, -81],
#                                                                 [-25, -5, 73.88, 88.88, -78, -58],
#                                                                 [-25, -5, 73.88, 88.88, -48, -28],
#                                                                 [-25, -5, 73.88, 88.88, -25, -5],
#                                                                 [-25, -5, 73.88, 88.88, 5, 25],
#                                                                 [-25, -5, 73.88, 88.88, 28, 48],
#                                                                 [-25, -5, 73.88, 88.88, 58, 78],
#                                                                 [-25, -5, 73.88, 88.88, 81, 101],
#                                                                 [5, 25, 73.88, 88.88, -101, -81],
#                                                                 [5, 25, 73.88, 88.88, -78, -58],
#                                                                 [5, 25, 73.88, 88.88, -48, -28],
#                                                                 [5, 25, 73.88, 88.88, -25, -5],
#                                                                 [5, 25, 73.88, 88.88, 5, 25],
#                                                                 [5, 25, 73.88, 88.88, 28, 48],
#                                                                 [5, 25, 73.88, 88.88, 58, 78],
#                                                                 [5, 25, 73.88, 88.88, 81, 101],
#                                                                 [28, 48, 73.88, 88.88, -101, -81],
#                                                                 [28, 48, 73.88, 88.88, -78, -58],
#                                                                 [28, 48, 73.88, 88.88, -48, -28],
#                                                                 [28, 48, 73.88, 88.88, -25, -5],
#                                                                 [28, 48, 73.88, 88.88, 5, 25],
#                                                                 [28, 48, 73.88, 88.88, 28, 48],
#                                                                 [28, 48, 73.88, 88.88, 58, 78],
#                                                                 [28, 48, 73.88, 88.88, 81, 101]],
#                                                                 dtype=np.float64 )
#
#     # move the crystals to current camera posiiton
#     crystals[:, :2] = crystals[:,:2] -cameraX
#     crystals[:, 2:4] = crystals[:, 2:4] + cameraY
#     crystals[:, 4:] = crystals[:, 4:] - cameraZ
#
#     dis = 0.0
#     # for each crystal, if it intercepts with line AB twice, it's a pass through
#     # if it intercepts just once, then it's the end segment, this is the crystal
#     # event B happens:
#     for i in range( N ):
#         # reset, no interactions
#         point1 = False
#         point2 = False
#
#         # crystal boundaries
#         crystal_x1, crystal_x2, crystal_y1, crystal_y2, crystal_z1, crystal_z2 = crystals[i]
#
#         # if Ax - Bx is 0, line AB either on or parallel to
#         # the plane x = x1 ( or x = x2 ), there is no interactions of line AB
#         # with the plane
#         if fabs( Ax - sx ) > 0:
#             t = ( crystal_x1 - sx ) / ( Ax - sx )
#             if t > 0.0 and t < 1.0:
#                 y = sy + t * ( Ay - sy )
#                 z = sz + t * ( Az - sz )
#                 if  ( y > crystal_y1 ) and ( y < crystal_y2 ) and ( z > crystal_z1 ) and ( z < crystal_z2 ):
#                     point1 = True
#                     p1 = np.array( [ crystal_x1, y, z ])
#
#             t = ( crystal_x2 - sx ) / ( Ax - sx )
#             if t > 0.0 and t < 1.0:
#                 y = sy + t * ( Ay - sy )
#                 z = sz + t * ( Az - sz )
#                 if  ( y > crystal_y1 ) and ( y < crystal_y2 ) and ( z > crystal_z1 ) and ( z < crystal_z2 ):
#                     if point1:
#                         point2 = True
#                         p2 = np.array( [ crystal_x2, y, z ])
#                     else:
#                         point1 = True
#                         p1 = np.array( [ crystal_x2, y, z ])
#
#         if fabs( Ay - sy ) > 0:
#             t = ( crystal_y1 - sy ) / ( Ay - sy )
#             if t > 0.0 and t < 1.0:
#                 x = sx + t * ( Ax - sx )
#                 z = sz + t * ( Az - sz )
#                 if  ( x > crystal_x1 ) and ( x < crystal_x2 ) and ( z > crystal_z1 ) and ( z < crystal_z2 ):
#                     if point1:
#                         point2 = True
#                         p2 = np.array( [ x, crystal_y1, z ])
#                     else:
#                         point1 = True
#                         p1 = np.array( [ x, crystal_y1, z ])
#
#             t = ( crystal_y2 - sy ) / ( Ay - sy )
#             if t > 0.0 and t < 1.0:
#                 x = sx + t * ( Ax - sx )
#                 z = sz + t * ( Az - sz )
#                 if  ( x > crystal_x1 ) and ( x < crystal_x2 ) and ( z > crystal_z1 ) and ( z < crystal_z2 ):
#                     if point1:
#                         point2 = True
#                         p2 = np.array( [ x, crystal_y2, z ])
#                     else:
#                         point1 = True
#                         p1 = np.array( [ x, crystal_y2, z ])
#
#         if fabs( Az - sz ) > 0:
#             t = ( crystal_z1 - sz ) / ( Az - sz )
#             if t > 0.0 and t < 1.0:
#                 x = sx + t * ( Ax - sx )
#                 y = sy + t * ( Ay - sy )
#                 if  ( x > crystal_x1 ) and ( x < crystal_x2 ) and ( y > crystal_y1 ) and ( y < crystal_y2 ):
#                     if point1:
#                         point2 = True
#                         p2 = np.array( [ x, y, crystal_z1 ])
#                     else:
#                         point1 = True
#                         p1 = np.array( [ x, y, crystal_z1 ])
#
#             t = ( crystal_z2 - sz ) / ( Az - sz )
#             if t > 0.0 and t < 1.0:
#                 x = sx + t * ( Ax - sx )
#                 y = sy + t * ( Ay - sy )
#                 if  ( x > crystal_x1 ) and ( x < crystal_x2 ) and ( y > crystal_y1 ) and ( y < crystal_y2 ):
#                     if point1:
#                         point2 = True
#                         p2 = np.array( [ x, y, crystal_z2 ])
#                     else:
#                         point1 = True
#                         p1 = np.array( [ x, y, crystal_z2 ])
#
#         if ( point1 ) and ( point2 ):
#             dis = dis + norm( p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2] )
#         elif ( point1 ) and not ( point2 ):
#             dis = dis + norm( Ax - p1[0], Ay - p1[1], Az - p1[2] )
#
#     return dis



cpdef double opticalDepthApprox(    double Ax=0.0, double Ay=0.0, double Az=0.0,
                                    double Bx=0.0, double By=0.0, double Bz=0.0,
                                    double cameraX=0.0, double cameraY=0.0, double cameraZ=0.0,
                                    double coneAngle=0.0, int nC=8, mode='exit' ):
    '''
    eod = exitODApprox( double Ax=0.0, double Ay=0.0, double Az=0.0,
                        double Bx=0.0, double By=0.0, double Bz=0.0,
                        double cameraX=0.0, double cameraY=0.0, double cameraZ=0.0,
                        double coneAngle=0.0, int nC=8, mode='exit' )

    approximate the entrance or exit optical depth when the entrance and exit directions are unknown.

    in the case of exit, nC number of points is placed evenly on a ring on the exit cone surface,
    the ring plane is 2000 mm from the last interaction point B, optical depth from point B to
    these points are calculated and then average is used as an approximate for the exit optical depth.

    in the case of entrance, nC number of points are place on the entrace cone
    on a ring, the ring plane is located at 20000 mm from first interaction point A, optical depth from
    these points to point A is calculated and the average is used as an approximate
    for the entrace optical depth.


    input:
        Ax, Ay, Az, Bx, By, Bz:
            double, if it is for the entrance, then A and B are the first
            and the second interaction coordiates;
            if it is for the exit, then A and B are the second last and the last
            interaction coordinates.
        cameraX, cameraY, cameraZ:
            double, cameraY is camera surface location, cameraX and cameraZ is
            the porjection of isocenter on camera
        coneAngle:
            double, the cone angle of the exiting photon
        nC:
            int, the number of 3rd events will be used to compute the average
        mode:
            string, 'exit' or 'entrance'
    ouput:
        eod:
            double, the exit optical depth estimate
    '''

    cdef:
        # for coordinates rotation
        float   sinAlpha=0.0, cosAlpha=0.0, sinBeta=0.0, cosBeta=0.0
        # coordiates of the 3rd 'event'
        float   Cx=0.0, Cy=0.0, Cz=0.0
        # assumes the 3rd events is located on a ring that is 2000 mm away
        float   h=20000, r=0.0
        float   dis=0

    if ( Bx - Ax ) **2 + ( Bz - Az )**2 > 0.0:
        sinAlpha = ( Bx - Ax ) / sqrt( ( Bx - Ax)**2 + ( Bz - Az )**2 )
        cosAlpha = ( Bz - Az ) / sqrt( ( Bx - Ax)**2 + ( Bz - Az )**2 )
    else:
        sinAlpha = 0
        cosAlpha = 1
    if ( Bx - Ax )**2 + ( By - Ay )**2 + ( Bz - Az )**2 > 0.0:
        sinBeta = sqrt( ( Bx - Ax )**2 + ( By - Ay )**2 ) / sqrt( ( Bx - Ax )**2 + ( By - Ay )**2 + ( Bz - Az )**2 )
        cosBeta = ( By - Ay ) / sqrt( ( Bx - Ax )**2 + ( By - Ay )**2 + ( Bz - Az )**2 )
    else:
        sinBeta = 0.0
        cosBeta = 1.0

    # cone angle can be larger than Pi / 2
    r = fabs( h * tan( coneAngle ) )
    for i in range( nC ):
        # 3rd event coordinates in cone axis
        if mode == 'exit':
            if coneAngle < PI / 2:
                Cy = h + sqrt( ( Bx - Ax )**2 + ( By - Ay )**2 + ( Bz - Az )**2 )
            else:
                Cy = -h + sqrt( ( Bx - Ax )**2 + ( By - Ay )**2 + ( Bz - Az )**2 )
        elif mode == 'entrance':
            if coneAngle < PI / 2:
                Cy = -h + sqrt( ( Bx - Ax )**2 + ( By - Ay )**2 + ( Bz - Az )**2 )
            else:
                Cy = h + sqrt( ( Bx - Ax )**2 + ( By - Ay )**2 + ( Bz - Az )**2 )
        Cz = r * cos( PI * 2 * i / nC )
        Cx = r * sin( PI * 2 * i / nC )
        # 3rd event coordinates in CORE input coordinates
        Cx, Cy, Cz = matrixVectorDot( 1, 0, 0, 0, cosBeta, -sinBeta, 0, sinBeta, cosBeta, Cx, Cy, Cz )
        Cx, Cy, Cz = matrixVectorDot( cosAlpha, 0, sinAlpha, 0, 1, 0, -sinAlpha, 0, cosAlpha, Cx, Cy, Cz )
        Cx = Cx + Ax
        Cy = Cy + Ay
        Cz = Cz + Az

        if mode == 'exit':
            dis = dis + opticalDepth( Bx, By, Bz, Cx, Cy, Cz, cameraX, cameraY, cameraZ )
        elif mode == 'entrance':
            dis = dis + opticalDepth( Cx, Cy, Cz, Ax, Ay, Az, cameraX, cameraY, cameraZ )

    return dis / nC



cpdef double opticalDepth(  double Ax=0.0, double Ay=0.0, double Az=0.0,
                            double Bx=0.0, double By=0.0, double Bz=0.0,
                            double cameraX=0.0, double cameraY=0.0, double cameraZ=0.0 ):
    '''
    od = opticalDepth(  double Ax=0.0, double Ay=0.0, double Az=0.0,
                        double Bx=0.0, double By=0.0, double Bz=0.0,
                        double cameraX=0.0, double cameraY=0.0, double cameraZ=0.0)

    compute optical depth inside crystals along AB direction
    NOTE: direction matters, the line/vector always goes from point A to point B

    input:
        Ax, Ay, Az:
            double, starting point coordinates
        Bx, By, Bz:
            double, the ending point coordiantes
        cameraX, cameraY, cameraZ:
            double, cameraY is camera surface location, cameraX and cameraZ is
            the porjection of isocenter on camera

    ouput:
        od:
            double, the optical depth between point A and B
    '''

    cdef:
        float   dis=0.0
        # x, y and z is the coordinates of interception with crystal surface
        float   x=0.0, y=0.0, z=0.0
        float   crystal_x1=0.0, crystal_x2=0.0, crystal_y1=0.0, crystal_y2=0.0, crystal_z1=0.0, crystal_z2=0.0
        int     i=0, N=64

        np.ndarray[np.float64_t, ndim=2] crystals=np.array( [   [-48, -28, 38.32, 48.32, -101, -81],
                                                                [-48, -28, 38.32, 48.32, -78, -58],
                                                                [-48, -28, 38.32, 48.32, -48, -28],
                                                                [-48, -28, 38.32, 48.32, -25, -5],
                                                                [-48, -28, 38.32, 48.32, 5, 25],
                                                                [-48, -28, 38.32, 48.32, 28, 48],
                                                                [-48, -28, 38.32, 48.32, 58, 78],
                                                                [-48, -28, 38.32, 48.32, 81, 101],
                                                                [-25, -5, 38.32, 48.32, -101, -81],
                                                                [-25, -5, 38.32, 48.32, -78, -58],
                                                                [-25, -5, 38.32, 48.32, -48, -28],
                                                                [-25, -5, 38.32, 48.32, -25, -5],
                                                                [-25, -5, 38.32, 48.32, 5, 25],
                                                                [-25, -5, 38.32, 48.32, 28, 48],
                                                                [-25, -5, 38.32, 48.32, 58, 78],
                                                                [-25, -5, 38.32, 48.32, 81, 101],
                                                                [5, 25, 38.32, 48.32, -101, -81],
                                                                [5, 25, 38.32, 48.32, -78, -58],
                                                                [5, 25, 38.32, 48.32, -48, -28],
                                                                [5, 25, 38.32, 48.32, -25, -5],
                                                                [5, 25, 38.32, 48.32, 5, 25],
                                                                [5, 25, 38.32, 48.32, 28, 48],
                                                                [5, 25, 38.32, 48.32, 58, 78],
                                                                [5, 25, 38.32, 48.32, 81, 101],
                                                                [28, 48, 38.32, 48.32, -101, -81],
                                                                [28, 48, 38.32, 48.32, -78, -58],
                                                                [28, 48, 38.32, 48.32, -48, -28],
                                                                [28, 48, 38.32, 48.32, -25, -5],
                                                                [28, 48, 38.32, 48.32, 5, 25],
                                                                [28, 48, 38.32, 48.32, 28, 48],
                                                                [28, 48, 38.32, 48.32, 58, 78],
                                                                [28, 48, 38.32, 48.32, 81, 101],
                                                                [-48, -28, 73.88, 88.88, -101, -81],
                                                                [-48, -28, 73.88, 88.88, -78, -58],
                                                                [-48, -28, 73.88, 88.88, -48, -28],
                                                                [-48, -28, 73.88, 88.88, -25, -5],
                                                                [-48, -28, 73.88, 88.88, 5, 25],
                                                                [-48, -28, 73.88, 88.88, 28, 48],
                                                                [-48, -28, 73.88, 88.88, 58, 78],
                                                                [-48, -28, 73.88, 88.88, 81, 101],
                                                                [-25, -5, 73.88, 88.88, -101, -81],
                                                                [-25, -5, 73.88, 88.88, -78, -58],
                                                                [-25, -5, 73.88, 88.88, -48, -28],
                                                                [-25, -5, 73.88, 88.88, -25, -5],
                                                                [-25, -5, 73.88, 88.88, 5, 25],
                                                                [-25, -5, 73.88, 88.88, 28, 48],
                                                                [-25, -5, 73.88, 88.88, 58, 78],
                                                                [-25, -5, 73.88, 88.88, 81, 101],
                                                                [5, 25, 73.88, 88.88, -101, -81],
                                                                [5, 25, 73.88, 88.88, -78, -58],
                                                                [5, 25, 73.88, 88.88, -48, -28],
                                                                [5, 25, 73.88, 88.88, -25, -5],
                                                                [5, 25, 73.88, 88.88, 5, 25],
                                                                [5, 25, 73.88, 88.88, 28, 48],
                                                                [5, 25, 73.88, 88.88, 58, 78],
                                                                [5, 25, 73.88, 88.88, 81, 101],
                                                                [28, 48, 73.88, 88.88, -101, -81],
                                                                [28, 48, 73.88, 88.88, -78, -58],
                                                                [28, 48, 73.88, 88.88, -48, -28],
                                                                [28, 48, 73.88, 88.88, -25, -5],
                                                                [28, 48, 73.88, 88.88, 5, 25],
                                                                [28, 48, 73.88, 88.88, 28, 48],
                                                                [28, 48, 73.88, 88.88, 58, 78],
                                                                [28, 48, 73.88, 88.88, 81, 101]],
                                                                dtype=np.float64 )


    crystals[:, :2] = crystals[:,:2] -cameraX
    crystals[:, 2:4] = crystals[:, 2:4] + cameraY
    crystals[:, 4:] = crystals[:, 4:] - cameraZ

    # for each crystal, if it intercepts with line AB twice, it's a pass through
    # if it intercepts just once, then it's the end segment, this is the crystal
    # event B happens:
    for i in range( N ):
        # reset, start with no intersecption with this crystal
        point1 = False
        point2 = False

        # crystal boundaries
        crystal_x1, crystal_x2, crystal_y1, crystal_y2, crystal_z1, crystal_z2 = crystals[i]

        # first check if line AB intercepts with cyrstal surface at crystal_x1 and crystal_x2
        # if Ax - Bx is 0, line AB either on or parallel to
        # the plane x = x1 ( or x = x2 ), there is no interactions of line AB
        # with the plane crystal_x1 and plane crystal_x2
        if fabs( Ax - Bx ) > 0:
            t = ( crystal_x1 - Ax ) / ( Bx - Ax )
            if t > 0.0 and t < 1.0:
                y = Ay + t * ( By - Ay )
                z = Az + t * ( Bz - Az )
                if  ( y > crystal_y1 ) and ( y < crystal_y2 ) and ( z > crystal_z1 ) and ( z < crystal_z2 ):
                    point1 = True
                    p1 = np.array( [ crystal_x1, y, z ])

            t = ( crystal_x2 - Ax ) / ( Bx - Ax )
            if t > 0.0 and t < 1.0:
                y = Ay + t * ( By - Ay )
                z = Az + t * ( Bz - Az )
                if  ( y > crystal_y1 ) and ( y < crystal_y2 ) and ( z > crystal_z1 ) and ( z < crystal_z2 ):
                    if point1:
                        point2 = True
                        p2 = np.array( [ crystal_x2, y, z ])
                    else:
                        point1 = True
                        p1 = np.array( [ crystal_x2, y, z ])

        # now check if line AB intercepts with crystal surface at crystal_y1 and crystal_y2
        if fabs( Ay - By ) > 0:
            t = ( crystal_y1 - Ay ) / ( By - Ay )
            if t > 0.0 and t < 1.0:
                x = Ax + t * ( Bx - Ax )
                z = Az + t * ( Bz - Az )
                if  ( x > crystal_x1 ) and ( x < crystal_x2 ) and ( z > crystal_z1 ) and ( z < crystal_z2 ):
                    if point1:
                        point2 = True
                        p2 = np.array( [ x, crystal_y1, z ])
                    else:
                        point1 = True
                        p1 = np.array( [ x, crystal_y1, z ])

            t = ( crystal_y2 - Ay ) / ( By - Ay )
            if t > 0.0 and t < 1.0:
                x = Ax + t * ( Bx - Ax )
                z = Az + t * ( Bz - Az )
                if  ( x > crystal_x1 ) and ( x < crystal_x2 ) and ( z > crystal_z1 ) and ( z < crystal_z2 ):
                    if point1:
                        point2 = True
                        p2 = np.array( [ x, crystal_y2, z ])
                    else:
                        point1 = True
                        p1 = np.array( [ x, crystal_y2, z ])

        # if line AB intercepts with crystal surface at crystal_z1 and crystal_z2
        if fabs( Az - Bz ) > 0:
            t = ( crystal_z1 - Az ) / ( Bz - Az )
            if t > 0.0 and t < 1.0:
                x = Ax + t * ( Bx - Ax )
                y = Ay + t * ( By - Ay )
                if  ( x > crystal_x1 ) and ( x < crystal_x2 ) and ( y > crystal_y1 ) and ( y < crystal_y2 ):
                    if point1:
                        point2 = True
                        p2 = np.array( [ x, y, crystal_z1 ])
                    else:
                        point1 = True
                        p1 = np.array( [ x, y, crystal_z1 ])

            t = ( crystal_z2 - Az ) / ( Bz - Az )
            if t > 0.0 and t < 1.0:
                x = Ax + t * ( Bx - Ax )
                y = Ay + t * ( By - Ay )
                if  ( x > crystal_x1 ) and ( x < crystal_x2 ) and ( y > crystal_y1 ) and ( y < crystal_y2 ):
                    if point1:
                        point2 = True
                        p2 = np.array( [ x, y, crystal_z2 ])
                    else:
                        point1 = True
                        p1 = np.array( [ x, y, crystal_z2 ])

        # when the line intercepts with this crystal surface twice:
        if ( point1 ) and ( point2 ):
            dis = dis + norm( p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2] )

        # when the line intercepts with this crystal surface only once:
        elif ( point1 ) and not ( point2 ):
            # check if A is inside this crystal:
            if ( ( Ax > crystal_x1 ) and ( Ax < crystal_x2 ) and
                 ( Ay > crystal_y1 ) and ( Ay < crystal_y2 ) and
                 ( Az > crystal_z1 ) and ( Az < crystal_z2 ) ):
                dis = dis + norm( Ax - p1[0], Ay - p1[1], Az - p1[2] )
            # or if B is inside this crystal
            elif ( ( Bx > crystal_x1 ) and ( Bx < crystal_x2 ) and
                   ( By > crystal_y1 ) and ( By < crystal_y2 ) and
                   ( Bz > crystal_z1 ) and ( Bz < crystal_z2 ) ):
                dis = dis + norm( Bx - p1[0], By - p1[1], Bz - p1[2] )
        # when the line does not intercept with this crystal, then only
        # when both A and B are inside this crystal:
        elif ( not point1 ) and ( not point2 ):
            if ( ( Ax > crystal_x1 ) and ( Ax < crystal_x2 ) and
                 ( Ay > crystal_y1 ) and ( Ay < crystal_y2 ) and
                 ( Az > crystal_z1 ) and ( Az < crystal_z2 ) and
                 ( Bx > crystal_x1 ) and ( Bx < crystal_x2 ) and
                 ( By > crystal_y1 ) and ( By < crystal_y2 ) and
                 ( Bz > crystal_z1 ) and ( Bz < crystal_z2 ) ):
                dis = dis + norm( Bx - Ax, By - Ay, Bz - Az )

    return dis



def readDoseText( fileName ):
    '''
    data, rx, ry, rz = readDoseText( fileName )

    read the output of CORE

    input:
        fileName:
            string, the file name of the CORE output, be mindful about the path

    output:
        data:
            ndarray, 3D array in the format of [nSlices, nRows, nCols]
        rx, ry, rz:
            the dose grid in x, y, z directions


    '''
    fid = open( fileName, 'r')
    dim = np.array( list(map(int, fid.readline().strip('\n').split(' ') )) )
    nCols = dim[0]
    nRows = dim[1]
    nSlices = dim[2]
#    if not MC:
#        x = np.array( list(map(float, fid.readline().strip(' \n').split(' ') )))
#        y = np.array( list(map(float, fid.readline().strip(' \n').split(' ') )))
#        z = np.array( list(map(float, fid.readline().strip(' \n').split(' ') )))
#    else:
#        x = np.array( list(map(float, fid.readline().strip(' \n').split(' ') )))
#        y = np.array( list(map(float, fid.readline().strip(' \n').split(' ') )))
#        z = np.array( list(map(float, fid.readline().strip(' \n').split(' ') )))

    x = np.array( list(map(float, fid.readline().strip(' \n').split(' ') )))
    y = np.array( list(map(float, fid.readline().strip(' \n').split(' ') )))
    z = np.array( list(map(float, fid.readline().strip(' \n').split(' ') )))
    x = ( x[0:-1] + x[1:] ) / 2
    y = ( y[0:-1] + y[1:] ) / 2
    z = ( z[0:-1] + z[1:] ) / 2

    # if CM:
    #     x = x * 10.0
    #     y = y * 10.0
    #     z = z * 10.0  # convert to mm

    data = fid.readlines()
    if data[0].find( ' ' ) > 0:
        ending = ' \n'
        delimiter = ' '
    elif data[0].find( ',' ) > 0:
        ending = ',\n'
        delimiter = ','
    else:
        print( 'unknown file formate' )
        return

    dataArray = np.zeros( (nSlices, nRows, nCols) )

    if len( data ) == nRows * nSlices:
        for n in range(nRows * nSlices ):
            i = n // nRows
            j = n % nRows
#            dataArray[i, :, j] = np.array(list(map(float, data[n].strip( ending ).split( delimiter ) ) ) )
            dataArray[i, j, :] = np.array(list(map(float, data[n].strip( ending ).split( delimiter ) ) ) )
    elif len( data ) == nSlices:
        for n in range(nSlices):
            dataArray[n] = np.reshape( np.array(list(map(float, data[n].strip( ending ).split( delimiter ) ) ) ), (nRows, nCols) )
    else:
        print( 'unknow file formate' )
        return

#    if MC:
#        for i in range( nSlices ):
#            dataArray[i] = dataArray[i].T

    return dataArray, x, y, z


cpdef np.ndarray[np.float_t, ndim=2] countsRate( np.ndarray[np.float_t, ndim=1] t, float windowWidth=1.0  ):
    '''
        cr = countsRate( float t, float 1.0  )
        compute the counts rate, using the specified window width.
        For example, if the window width is 1.0 second,  it returns
        counts from [0.5 * windowWidth, 1.5 * windowWidth),
        [1.5 * windowWidth, 2.5 * windowWidth)... at time point 1.0 * windowWidth,
        2.0 * windowWidth...

        input:
            t: np.array, float, time stamp
            windowWidth: float, counts rate widnow width in second
        output:
            cr: np.ndarray, float, first column is the time, the second column
                is the counts rate.
    '''
    cdef:
        unsigned long long nWindows = np.uint64( ( t[-1] - t[0] )*1e-8 / windowWidth )
        unsigned long long nEvents=0, iEvent=0, iWindow=1, s=0
        np.ndarray[np.float_t, ndim=2] cr=np.zeros( (nWindows,2), dtype=np.float )

    t = t * 1e-8
    t = t - t[0]

    nEvents = len( t )

    while t[iEvent] < 0.5 * windowWidth:
        iEvent += 1

    while iEvent < nEvents and iWindow <= nWindows:
        s = 0
        while iEvent < nEvents and t[iEvent] < ( iWindow + 0.5 ) * windowWidth :
            iEvent += 1
            s += 1
        cr[iWindow-1, 0] = iWindow * windowWidth
        cr[iWindow-1, 1] = s
        iWindow += 1

    # for i in range( 1, N ):
    #     ind = np.logical_and( t >= i - 0.5, t < t + 0.5  )
    #     cr[i-1,0] = i
    #     cr[i-1,1] = ind.sum()

    return cr


cpdef  timeResolution(  np.ndarray[np.float_t, ndim=1] t,
                        np.ndarray[np.float_t, ndim=1] eDep,
                        np.ndarray[np.int64_t, ndim=1] npx,
                        float lowerEmissionLine,
                        float upperEmissionLine,
                        float windowWidthMax  ):
    '''
        dt, index, npxDetermined, npxMarked = timeResolution(   np.ndarray[np.float_t, ndim=1] t,
                                                                np.ndarray[np.float_t, ndim=1] eDep,
                                                                np.ndarray[np.int64_t, ndim=1] npx,
                                                                float lowerEmissionLine,
                                                                float upperEmissionLine,
                                                                float windowWidthMax  )

        determine time resolution of camera.
        if the deposited energies of consecutive interactions sums to be within the range
        defined by lowerEmissionLine and upperEmissionLine, return the time difference (dt)
        between the first and last interaction. A histogram of dt would reveal the
        time resolution.
    '''
    cdef:
        unsigned long long  N=len(t), i=0
        int                 nInteractions=0
        float               eDepSum=0.0
        np.ndarray          tr=np.array([])
        bint                boundaryExceeded=False

    # change window width from nanosecond to time stamp click.
    windowWidthMax = windowWidthMax / 10.0

    dt = []
    index = []
    npxDetermined = []
    npxMarked = []

    while i < N - 1:
        nInteractions = 1
        while t[i+nInteractions] - t[i] < windowWidthMax:
            nInteractions += 1
            if i + nInteractions >= N:
                boundaryExceeded = True
                break
        if boundaryExceeded:
            break
        if nInteractions > 1:
            eDepSum = eDep[i:i+nInteractions].sum()
            if ( eDepSum > lowerEmissionLine ) and ( eDepSum < upperEmissionLine ):
                dt.append( t[i+nInteractions-1] - t[i] )
                index.append( i )
                npxDetermined.append( nInteractions )
                npxMarked.append( npx[i]  )

        i += nInteractions

    return dt, index, npxDetermined, npxMarked



def unpackMeasurementNpz( npzFiles ):
    '''
    npx, mdl, edep, x, y, z, t = unpackMeasurementNpz( npzFiles )

    unpack a measurement save as npz files.

    input:
        npzFiles:
            the npz files from load( file name )
        npx, mdl, edep, x, y, z, t:
            1D array, info of captured events.

    '''
    npx = npzFiles['npx']
    mdl = npzFiles['mdl']
    edep = npzFiles['edep']
    x = npzFiles['x']
    y = npzFiles['y']
    z = npzFiles['z']
    t = npzFiles['t']
    return npx, mdl, edep, x, y, z, t


def readSaveBEFAllEventsFile( fn, moduleNumber ):
    '''
    events = readStreamingAllEventsFile( fn, moduleNumber )

    read a single AllEvents.txt file extracted from SaveBEF during streaming

    input:
        fn:
            string, file name. It should comtain full path.
        moduleNumber:
            int, the module number.
    output:
        events:
            np array, event array in the format:
            [ npx, mdlNumber, edep, x, y, z, t ]
    '''

    fid = open( fn, 'r' )
    a = fid.readlines()
    fid.close()

    n = len( a )
    i = 0

    events = []

    while i < n:

        b = a[i].strip('\n').split('\t')

        if int( b[0] ) == 1:
            temp = [float(bb) for bb in b[:-1]]
            events.append( temp )
            i += 1
        else:
            nEvents = int( b[0] )
            eventT = float( a[ i + nEvents - 1].strip('\n').split('\t')[-2] )
            temp = [float(bb) for bb in b]
            temp.append( eventT )
            events.append( temp )
            for j in range( nEvents - 2):
                b = a[i + j + 1].strip('\n').split('\t')
                temp = [float(bb) for bb in b]
                temp.append( eventT )
                events.append( temp )
            b = a[i + nEvents - 1].strip('\n').split('\t')
            temp = [float(bb) for bb in b[:-1]]
            events.append( temp )
            i += nEvents
    events = np.array( events )
    events = np.insert( events, 1, moduleNumber, axis=1 )
    return events


def readCameraBinary( fn, moduleNumber ):
    '''
    events, sync = readCameraBinary( fn, moduleNumber )

    read binary file produce either by netcast or use SaveBEF.

    NOTE, this function is for reading binary files from a SINGLE module.
    The events coordinates are for THAT module.

    input:
        fn:
            string, file name
        moduleNumber:
            module number
    output:
        events:
            np array, each row is an interaction follow the format:
            [npx, moduleNumber, chip, edep, x, y, z, t]
        sync:
            np array, snyc pulse index and time stamp
    '''
    fid = open( fn, 'rb' )
    b = fid.read()
    fid.close()

    N = len( b )

    events = []
    sync = []
    pos = 0
    while pos < N :
        if b[pos] == 122:
            if pos + 18 > len( b ):
                break
            pos += 2
            syncIndex, syncTime = struct.unpack( '>QQ', b[ pos:pos + 16 ] )
            pos += 16
            sync.append( [ moduleNumber, syncIndex, syncTime ] )
            # events.append( temp )
        elif b[pos] == 0:
            if pos + 24 > len( b ):
                break
            pos += 24
        else:
            npx = b[pos]
            if pos + 1 + 8 + npx * 17 > len( b ):
                break
            pos += 1
            t = struct.unpack( '>Q', b[ pos:pos + 8 ] )[0]
            pos += 8
            for i in range( npx ):
                chip, x, y, z, edep = struct.unpack( '>Bllll', b[ pos:pos + 17 ] )
                pos += 17
                temp = [npx, moduleNumber, chip, edep, x, y, z, t]
                events.append( temp )
    return np.array( events ), np.array( sync )


# reads in binary data from file and store events and sync pulses
#  - code copied directly from Haijian's code (utl.pxi)
#  - modifed to also return timestamps in the events array
#  - added by Steve Peterson (31 May 2019)
def readCameraBinary2( fn, moduleNumber ):
    '''
    events, sync = readCameraBinary( fn, moduleNumber )

    read binary file produce either by netcast or use SaveBEF.

    NOTE, this function is for reading binary files from a SINGLE module.
    The events coordinates are for THAT module.

    input:
        fn:
            string, file name
        moduleNumber:
            module number
    output:
        events:
            np array, each row is an interaction follow the format:
            [npx, moduleNumber, chip, edep, x, y, z, t]
        sync:
            np array, sync pulse index and time stamp
    '''
    fid = open( fn, 'rb' )
    b = fid.read()
    fid.close()

    N = len( b )

    events = []
    sync = []
    pos = 0
    while pos < N :
        if b[pos] == 122:
            if pos + 18 > len( b ):
                break
            pos += 2
            syncIndex, syncTime = struct.unpack( '>QQ', b[ pos:pos + 16 ] )
            pos += 16
            sync.append( [ moduleNumber, syncIndex, syncTime ] )
            events.append( [ 122, moduleNumber, syncIndex, syncTime ] )
            # events.append( temp )
        elif b[pos] == 0:
            if pos + 24 > len( b ):
                break
            pos += 24
        else:
            npx = b[pos]
            if pos + 1 + 8 + npx * 17 > len( b ):
                break
            pos += 1
            t = struct.unpack( '>Q', b[ pos:pos + 8 ] )[0]
            pos += 8
            for i in range( npx ):
                chip, x, y, z, edep = struct.unpack( '>Bllll', b[ pos:pos + 17 ] )
                pos += 17
                temp = [npx, moduleNumber, chip, edep, x, y, z, t]
                events.append( temp )
    return np.array( events ), np.array( sync )



def extractDouble( npx, edep, x, y, z ):
    '''
    eDepA, xA, yA, zA, eDepB,  xB, yB, zB = extractDouble( npx, x, y, z, edep )
    find those marked as doubles in AllEventsCombined.txt

    NOTHING IS DONE TO THESE DOUBLES. This is only a convenient function to get
    these doubles for purposes like spectrum comparison.

    input:
        npx, x, y, z, edep:
            each of them is a np array, these are tyically the returns from readInC
    output:
        eDepA, xA, yA, zA, eDepB,  xB, yB, zB:
            edep and coordiantes of two interactions in a double;
    '''
    ind = npx==2
    x = x[ind]
    y = y[ind]
    z = z[ind]
    edep = edep[ind]
    xA = x[:-1:2]
    xB = x[1::2]
    yA = y[:-1:2]
    yB = y[1::2]
    zA = z[:-1:2]
    zB = z[1::2]
    eDepA = edep[:-1:2]
    eDepB = edep[1::2]
    return eDepA, xA, yA, zA, eDepB,  xB, yB, zB



def extractTriple( npx, edep, x, y, z ):
    '''
    eDepA, xA, yA, zA, eDepB, xB, yB, zB, eDepC, xC, yC, zC = extractTriple( npx, x, y, z, edep )

    similar to extractDouble function, this is to find those marked as triples
    in AllEventsCombined.txt
    '''
    ind = npx==3
    x = x[ind]
    y = y[ind]
    z = z[ind]
    edep = edep[ind]
    xA = x[:-2:3]
    xB = x[1:-1:3]
    xC = x[2::3]
    yA = x[:-2:3]
    yB = x[1:-1:3]
    yC = x[2::3]
    zA = x[:-2:3]
    zB = x[1:-1:3]
    zC = x[2::3]
    eDepA = edep[:-2:3]
    eDepB = edep[1:-1:3]
    eDepC = edep[2::3]
    return eDepA, xA, yA, zA, eDepB, xB, yB, zB, eDepC, xC, yC, zC


def findOverlapIndex( evA, evB ):
    '''
    index = findOverlapIndex( evA, evB )

    A doulbe A and B, or a triple A, B and C could be coming from more than one
    emission lines.
    If we run getDoubles to find doubles belong to a line E1, and find A and B
    and then run the same getDoubles to find doubles belong to E2 and also find A
    and B, we need at this point exclude AB unless we have a way to determine
    whether AB is from E1 or E2.
    This is to find those AB's ( or ABC's ) show up in both E1 and E2.


    input:
        evA, evB:
            np array. These are the return from getDoubles or getTriples.
            For a double A and B, in the AllEventsCombined file, A appear first
            before B ( even though the algorithm may determine the correct order is BA).
            Similarly, for triples, A, B and C appear in AllEventsCombined file
            in order, regardless the events order the algorithm may determine.
            The last column in getDoubles or getTriples is the line number of A in
            the AllEventsCombined file.
    return:
        index:
            list, index[i] is a row number for evA, the doubles (or triples) in
            that row is also in evB.
    '''
    sA = set( evA[:,-1] )
    sB = set( evB[:,-1] )
    sAB = sA.intersection( sB )
    sAB = np.sort( np.array( list( sAB ) ) )
    jsA = 0
    jsAB = 0
    index = []
    while jsAB < len(sAB):
        if evA[jsA, -1] == sAB[jsAB]:
            index.append( jsA )
            jsAB += 1
            jsA += 1
        else:
            jsA += 1
    return index

def centerSolidAngle(a=20.0, b=20.0, d=574.72 ):
    alpha = a / ( 2 * d )
    beta = b / ( 2 * d )
    s = 4 * np.arccos( sqrt( ( 1 + alpha * alpha + beta * beta ) / ( 1 + alpha * alpha ) / ( 1 + beta * beta ) ) )
    return s


def offCenterSolidAngle(a=20.0, b=20.0, A=0.0, B=0.0, d=574.72 ):
    s = centerSolidAngle( a=2*(A+a), b=2*(B+b), d=d )
    s = s - centerSolidAngle( a=2*A, b=2*(B+b), d=d )
    s = s - centerSolidAngle( a=2*(A+a), b=2*B, d=d )
    s = s + centerSolidAngle( a=2*A, b=2*B, d=d )
    s = s / 4.0
    return s

def crossAxisSolidAngle1(a=20.0, b=20.0, A=0.0, B=0.0, d=574.72 ):
    if 2 * B - b < 0 or B < 0:
        raise Exception( 'use the other side for B' )

    s = centerSolidAngle( a=2*(A+a), b=2*(b-B), d=d )
    s = s - centerSolidAngle( a=2*A, b=2*(b-B), d=d )
    s = s + centerSolidAngle( a=2*(A+a), b=2*B, d=d )
    s = s - centerSolidAngle( a=2*A, b=2*B, d=d )
    s = s / 4.0
    return s

def crossAxisSolidAngle2(a=20.0, b=20.0, A=0.0, B=0.0, d=574.72 ):
    if 2 * B - b < 0 or B < 0:
        raise Exception( 'use the other side for B' )
    if  2 * A -a < 0 or A < 0:
        raise Exception( 'use the other side for A' )

    s = centerSolidAngle( a=2*(a-A), b=2*(b-B), d=d )
    s = s + centerSolidAngle( a=2*A, b=2*(b-B), d=d )
    s = s + centerSolidAngle( a=2*(a-A), b=2*B, d=d )
    s = s + centerSolidAngle( a=2*A, b=2*B, d=d )
    s = s / 4.0
    return s
