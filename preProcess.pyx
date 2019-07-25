#!/usr/bin/python

include 'utl.pxi'

def readPar( settingsFileName='testSettings.txt'):
    '''
    par = readPar( settingsFileName='testSettings.txt')
    reading the parameters file

    input:
        settingsFileName:
            string, the parametr file name, include path if the file is not
            in the current path

    output:
        par:
            python dictionay, contains all parameters
    '''

    # read all lines from the file, includes comments
    try:
        with open( settingsFileName ) as fSettings:
            settings = fSettings.readlines()
    except:
        print( 'file does not exit' )
        return

    nLines = len( settings )

    # first delete information not used, these are at the end of the setting file
    ind = None
    for i in range( nLines ):
        if settings[i].strip().startswith( '****' ) or settings[i].lstrip().startswith( '####' ):
            ind = i
            break
    if ind is not None:
        del settings[ind:]

    # now delete comments,  delete in reversed order
    nLines = len( settings )
    for i in range( nLines-1, -1, -1 ):
        settings[i] = settings[i].strip()
        if settings[i].strip().startswith( '#' ):
            del settings[i]

    # getPar will find the values of parameters from lines read from the file
    # when a parameter name passed to the function is not found,
    # an exception is thrown, and the progaram will exit.
    def getPar( parName ):
        parValue = None
        for l in settings:
            if l.startswith( parName ):
                ind = l.find( '=' )
                parValue = l[ind+1:].strip()
                break
        if parValue is None:
            raise Exception(' parameter ' + parName + ' not found!')
        else:
            return parValue

    # find input and output file names:
    par = { 'inputFolder': getPar( 'inputFolder' ) }
    if not par['inputFolder'].endswith( '/' ):
        par['inputFolder'] = par['inputFolder'] + '/'
    par['inputFileName'] = getPar( 'inputFileName' )
    par['outputDir'] = getPar( 'outputFolder' )
    if not par['outputDir'].endswith( '/' ):
        par['outputDir'] = par['outputDir'] + '/'
    par['outputFileName'] = getPar( 'outputFileName' )
    par['numberOfFiles'] = np.int( getPar( 'numberOfFiles' ) )
    if par['numberOfFiles'] > 1:
        raise Exception( 'more then one file is not supported yet' )
        # return

    # detector position:
    # x, z is the projection of isocenter on camera
    # y is the distance from isocenter to camera surface
    temp = np.array( list( map( np.float64, getPar( 'det').strip().split(' ') ) ) )
    if len( temp ) != 4:
        raise Exception( 'detector position should be in x, y, z and camera angle' )
        # return
    # temp[3] = temp[3] / 180.0 * PI
    par['det'] = temp
    # if use time stamp for detecting coincidence events.
    # if it does, also find the timing window
    # par['timeStamp'] = True if getPar( 'timeStamp' ) == 'y' else False
    # if par['timeStamp']:
    #     par['timeWindow'] = np.float64( getPar( 'timeWindow' ) )


    # if use just events from a single module.
    # if it does, find the module number.
    par['singleModule'] = True if getPar( 'singleModule' ) == 'y' else False
    if par['singleModule']:
        par['moduleNumber'] = np.int( getPar( 'moduleNumber' ) )

    # if use just modules from one plane
    # if it does, find the plane number, plane number 0 is the one that is closer to the source
    par['singlePlane'] = True if getPar( 'singlePlane' ) == 'y' else False
    if par['singlePlane']:
        par['planeNumber'] = np.int( getPar( 'planeNumber' ) )

    # point source or beam measurements:
    # for beam measurements, need to know which slice ( specified by y coording )
    # the beam axial is in
    # nODApproximalPoints is needed to approximate exit optical depth and entrace
    # optical depth in case of beam measurement
    par['beamMeasurement'] = True if getPar( 'beamMeasurement' ) == 'y' else False
    if par['beamMeasurement']:
        par['slicePosition'] = np.float64( getPar( 'slicePosition' ) )
    par['nODApproximalPoints'] = np.int( getPar( 'nODApproximalPoints' ) )
    # for point source, need to know where the source is
    par['pointSource'] = True if getPar( 'pointSource' ) == 'y' else False
    if par['pointSource']:
        temp = getPar( 'pointSourcePosition')
        par['pointSourcePosition'] = np.array( list( map( np.float64, temp.strip().split(' ') ) ) )
        if len( par['pointSourcePosition']) != 3:
            raise Exception( 'point source coord must be in 3-D (x, y, z)!')
    # cannot be both point source and beam measurements
    if par['pointSource'] and par['beamMeasurement']:
        raise Exception( 'It can only be either point source or beam measurements, can not be both!' )


    # events type
    par['singles'] = True if getPar( 'singles' ) == 'y' else False
    par['timeStampOnly'] = True if getPar( 'timeStampOnly' ) == 'y' else False
    if par['singles'] or par['timeStampOnly']:
        par['timeWindow'] = np.float64( getPar( 'timeWindow' ) )

    par['doubles'] = True if getPar( 'doubles' ) == 'y' else False

    par['triples'] = True if getPar( 'triples' ) == 'y' else False
    if par['triples']:
        # use E0 computed from triples as E0:
        par['useTripleE0'] = True if getPar( 'useTripleE0' ) == 'y' else False


    # if use dca scan. if used, then find the E0 such that the source is exactly
    # on cone surface. for point source, it's the source on the cone, for line
    # source, it's the line just touch the cone surface.
    # par['dcaScan'] = True if getPar( 'dcaScan' ) == 'y' else False
    # the E0 can be solved exactly. no need to scan it.
    # if par['dcaScan']:
    #     temp = getPar( 'dcaScanRange' )
    #     par['dcaScanRange'] = np.array( list( map( np.float, temp.strip().split(' ') ) ) )

    # use sum of eDep1 and eDep2 as E0:
    par['useEdepSum'] = True if getPar( 'useEdepSum' ) == 'y' else False

    # use the emission lines and eDep1 for dca.
    par['useEmissionLines'] = True if getPar( 'useEmissionLines' ) == 'y' else False
    if par['useEmissionLines']:
        par['nGamma'] = int( getPar( 'nGamma') )
        par['gammaE'] = np.array( list( map( np.float64, getPar('gammaE').strip().split(' ') ) ) )
        if par['nGamma'] != len( par['gammaE'] ):
            print( 'missing value in gammaE!' )
            return
    par['fullAbsorptionOnly'] = True if getPar( 'fullAbsorptionOnly' ) == 'y' else False
    par['fullAbsorptionThreshold'] = np.float64( getPar( 'fullAbsorptionThreshold' ) )

    # whether use the sum of eDep1 and eDep2 or use emission lines as E0, this is
    # the cut off to determin events should be included or not.
    par['dcaThreshold'] = np.float64( getPar( 'dcaThreshold' ) )

    par['useTripleE0'] = True if getPar( 'useTripleE0' )== 'y' else False

    # correct eDept2.
    # use the emission line which give rise to the minimal dca as E0 and E0 - eDep1 as eDep2
    par['dopplerCorrection'] = True if getPar( 'doppler' ) == 'y' else False

    # the following are not used for now.
    # par['clFiltter'] = True if getPar( 'clfilt' ) == 'y' else False
    # par['pcaFiltter'] = True if getPar( 'pcafilt' ) == 'y' else False
    # par['E2filter'] = True if getPar( 'E2filt' ) == 'y' else False
    # par['dcaSourcePosition'] = np.array( list( map( float, getPar( 'dcaSourcePosition' ).strip().split(' ') ) ) )
    par['scale'] = np.float64( getPar( 'scale' ) )

    # par['couchAngle'] = np.float( getPar( 'couchAngle' ) ) / 180.0 * PI
    # par['gantryAngle'] = np.float( getPar( 'gantryAngle' ) ) / 180.0 * PI
    par['couchAngle'] = np.float( getPar( 'couchAngle' ) )
    par['gantryAngle'] = np.float( getPar( 'gantryAngle' ) )
    # after find the point(s) on the cone that is closest to the z-axis (the assumed beam axis),
    # we need to determin if the point(s) is inside the recon volume:
    par['boundaryX'] = np.array( list( map( np.float64, getPar('boundaryX').strip().split(' ') ) ) )
    par['boundaryY'] = np.array( list( map( np.float64, getPar('boundaryY').strip().split(' ') ) ) )
    par['boundaryZ'] = np.array( list( map( np.float64, getPar('boundaryZ').strip().split(' ') ) ) )

    return par


cpdef int getNumberOfLine( par ):
    '''
    n = getNumberOfLine( par )

    find how many lines/events are there in the event file

    input:
        par:
            a python dictionary, the diectionary of all parameters, which is output of function readPar
    output:
        n:
            integer, the total number of lines/events in the event file
    '''
    cdef:
        FILE    *fileIn
        int     nLines=0
        char    line[512]
    inFileName = par['inputFolder'] + par['inputFileName']
    fileIn = fopen( inFileName.encode(), 'r' )
    if fileIn == NULL:
        print( 'can not open input file' )
        return 0

    while( fgets(line, 512, fileIn) ):
        nLines += 1
    fclose( fileIn )

    return nLines


cpdef tuple readInC( par, nEvents, convert2room=True ):
    '''
    npx, mdl, edep, x, y, z, t = readInC( par, nEvents, convert2room=True )

    read in events from a file, also convert from camera
    coordinates to room/phantom coordinates by default

    input:
        par:
            a python dictionary, the parameter dictionary, output of function readPar
        nEvents:
            integer, total number of events in the event file, it's the return from funciton getNumberOfLine
        convert2room:
            boolean, if convert from camera coordinates to room coordinates or not. default is to
            convert to CORE input coordinates, which is a right hand system, if the camera is placed under the
            couch, then y points down, towards the camera, in the same direction as DICOM y; x points away from the
            gantry, z points toward gantry 90;
            if it is set to False, the output x, y, z are in camera coordinates, note that's a left hand system,
            and earch row of modules has its own coordinates.


    output:
        a tuple of 7 one dimenssional numpy ndarray, in the order of:
        number of pixel
        module number
        deposited energy
        x coordinate
        y coordinate
        z coordinate
        time stamp
    '''
    cdef:
        FILE    *fileIn
        int     nPixel=0, module=0
        int     iEvent=0
        double  eDep=0.0, x=0.0, y=0.0, z=0.0
        long    t=0
        int     nRead=0
        np.ndarray[np.int_t, ndim=1]    nPixelReturn=np.zeros( nEvents, dtype=np.int )
        np.ndarray[np.int_t, ndim=1]    moduleReturn=np.zeros( nEvents, dtype=np.int )
        np.ndarray[np.float_t, ndim=1]  eDepReturn=np.zeros( nEvents, dtype=np.float )
        np.ndarray[np.float_t, ndim=1]  xReturn=np.zeros( nEvents, dtype=np.float )
        np.ndarray[np.float_t, ndim=1]  yReturn=np.zeros( nEvents, dtype=np.float )
        np.ndarray[np.float_t, ndim=1]  zReturn=np.zeros( nEvents, dtype=np.float )
        np.ndarray[np.float_t, ndim=1]  tReturn=np.zeros( nEvents, dtype=np.float )

    inFileName = par['inputFolder'] + par['inputFileName']
    fileIn = fopen( inFileName.encode(), 'r' )
    if fileIn == NULL:
        raise Exception( 'can not open input file' )
        # return

    # for timing how long it takes to read the file:
    start_time = time.time()

    ox, oy, oz, cameraAngle = par['det']

    while( not feof( fileIn) ):
        nRead = fscanf(fileIn, "%i %i %lf %lf %lf %lf %li\n", &nPixel, &module, &eDep, &x, &y, &z, &t )
        if module == 2 and y < 80.0:
            y = y + 53.0 * 2.0

        if module == 3 and y < 135.0:
            y = y + 53.0 * 3.0

        if convert2room:
            x, y, z = camera2CORE( x, y, z, module, ox, oy, oz, cameraAngle )

        nPixelReturn[iEvent] = nPixel
        moduleReturn[iEvent] = module
        eDepReturn[iEvent] =  eDep
        xReturn[iEvent] = x
        yReturn[iEvent] = y
        zReturn[iEvent] = z
        tReturn[iEvent] = t
        iEvent += 1

    end_time = time.time()
    print( 'number of events: ', nEvents )
    print( ( end_time - start_time ) * 1000 )

    return nPixelReturn, moduleReturn, eDepReturn, xReturn, yReturn, zReturn, tReturn


def readStreamingBinary(    measurementName, par,
                            moduleNumberList=[51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66],
                            returnSyncPulse=False ):
    '''
    events, sync = readStreamingBinary( measurementName, par,
                                        moduleNumberList=[51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66],
                                        returnSyncPulse=True )
    or ( whe returnSyncPulse is False ):
    events = readStreamingBinary(   measurementName, par,
                                    moduleNumberList=[51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66],
                                    returnSyncPulse=True )

    read events binary file produced from streaming GUI. Also converts to CORE coordinate.


    input:
        measurementName:
            string, name of the measurement. It is also the folder name
            where the data from each module is saved.
            NOTE: PAY ATTENTION TO PATH.
        moduleNumberList:
            list, module numbers, default is all modules from 51 to 66:
            [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66],
        returnSyncPulse:
            boolean, weather or not return the sync pulse information from each
            module, default is not, since the time stamp in the binary file is
            already extrapolated

    output:
        events:
            numpy array, all events in the format: [npx, mdl, chip, edep, x, y, z, t]
        sync:
            list, each element has the sync pulse info from a module
    '''

    if not measurementName.endswith( '/'):
        measurementName = measurementName + '/'

    ox, oy, oz, cameraAngle = par['det']
    events = []
    syncPulse = []

    for mdlNumber in moduleNumberList:
        fn = measurementName + 'mod' + str( mdlNumber ) + '.dat'
        fid = open( fn, 'rb' )
        b = fid.read()
        fid.close()

        N = len( b )
        pos = 0
        syncPulseMdl = []
        while pos < N :
            if b[pos] == 0:
                if pos + 18 > len( b ):
                    break
                pos += 1
                moduleNumber = b[pos]
                pos += 1
                syncIndex, syncTime = struct.unpack( '>QQ', b[ pos:pos + 16 ] )
                pos += 16
                temp = [ moduleNumber, syncIndex, syncTime ]
                syncPulseMdl.append( temp )

            else:
                npx = b[pos]
                if pos + 1 + 8 + npx * 17 > len( b ):
                    break
                pos += 1
                moduleNumber = b[pos]
                pos += 1
                t = struct.unpack( '>Q', b[ pos:pos + 8 ] )[0]
                pos += 8
                for i in range( npx ):
                    chip, x, y, z, edep = struct.unpack( '>Bllll', b[ pos:pos + 17 ] )
                    pos += 17
                    temp = [npx, moduleNumber, chip, edep, x, y, z, t]
                    events.append( temp )

        syncPulse.append( np.array( syncPulseMdl ) )
    events = np.array( events, dtype=np.float )
    ind = np.argsort( events[:,-1] )
    events = events[ind]
    events[:,3] = events[:,3] / 1000.0
    temp = events[:,4:7].copy()
    temp = temp / 1000.0

    # module 51 to 54
    ind = events[:,1] < 4
    events[ind, 4] = -temp[ind, 0] + 53 * 0.5
    events[ind, 5] = oy + 38.22 + temp[ind, 2]
    ind = events[:,1] == 0
    events[ind, 6] = -temp[ind, 1] + 53 * 1.5
    ind = events[:,1] == 1
    events[ind, 6] = -temp[ind, 1] + 53 * 0.5
    ind = events[:,1] == 2
    events[ind, 6] = -temp[ind, 1] - 53 * 0.5
    ind = events[:,1] == 3
    events[ind, 6] = -temp[ind, 1] - 53 * 1.5

    # module 55 to 58:
    ind = np.logical_and( events[:,1] > 3, events[:,1] < 8 )
    events[ind, 4] = temp[ind, 0] - 53 * 0.5
    events[ind, 5] = oy + 38.22 + temp[ind, 2]
    ind = events[:,1] == 4
    events[ind, 6] = temp[ind, 1] - 53 * 1.5
    ind = events[:,1] == 5
    events[ind, 6] = temp[ind, 1] - 53 * 0.5
    ind = events[:,1] == 6
    events[ind, 6] = temp[ind, 1] + 53 * 0.5
    ind = events[:,1] == 7
    events[ind, 6] = temp[ind, 1] + 53 * 1.5

    # module 59 t0 62
    ind = np.logical_and( events[:,1] > 7, events[:,1] < 12 )
    events[ind, 4] = -temp[ind, 0] + 53 * 0.5
    events[ind, 5] = oy + 88.88 - temp[ind, 2]
    ind = events[:,1] == 8
    events[ind, 6] = temp[ind, 1] - 53 * 1.5
    ind = events[:,1] == 9
    events[ind, 6] = temp[ind, 1] - 53 * 0.5
    ind = events[:,1] == 10
    events[ind, 6] = temp[ind, 1] + 53 * 0.5
    ind = events[:,1] == 11
    events[ind, 6] = temp[ind, 1] + 53 * 1.5

    # module 63 t0 66
    ind = events[:,1] > 11
    events[ind, 4] = temp[ind, 0] - 53 * 0.5
    events[ind, 5] = oy + 88.88 - temp[ind, 2]
    ind = events[:,1] == 12
    events[ind, 6] = -temp[ind, 1] + 53 * 1.5
    ind = events[:,1] == 13
    events[ind, 6] = -temp[ind, 1] + 53 * 0.5
    ind = events[:,1] == 14
    events[ind, 6] = -temp[ind, 1] - 53 * 0.5
    ind = events[:,1] == 15
    events[ind, 6] = -temp[ind, 1] - 53 * 1.5

    events[:,4] = events[:,4] - ox
    events[:,6] = events[:,6] - oz

    if fabs( cameraAngle ) > ZERO:
        tx = events[:,4]
        ty = events[:,6]
        events[:,4] =  tx * cos( cameraAngle ) + ty * sin( cameraAngle )
        events[:,6] = -tx * sin( cameraAngle ) + ty * cos( cameraAngle )

    if returnSyncPulse:
        return events, syncPulse
    else:
        return events



cpdef tuple camera2CORE(    double x, double y, double z,
                            int module,
                            double ox, double oy, double oz,
                            double cameraAngle):
    '''
    X, Y, Z = camera2CORE(  double x, double y, double z,
                            int module,
                            double ox, double oy, double oz,
                            double cameraAngle)

    transform from camera coordinates to CORE coordinates:
        assumes the camera is mounted under the couch:
            origin is at isocenter
            y pionts down
            x points away from the gantry
            z points towards 3 o'clock ( gantry 90 ) direction

        in the case where the camera is not facing up, but tilted by an
        cameraAngle, then there is rotation about z axis. after the shifting and
        switching x, y, z, rotate events coordinates back to CORE coordiates


    input:
        x, y, z:
            double, event coordinates in camera coordinates
            NOTE, camera coordiates is a LEFT HAND system, each row uses the
            center of the first module as origin.
        module:
            integer, module number
        ox, oy, oz,
            double, it is assumed that isocenter is at (0, 0, 0):
            oy---- the distance from camera surface to isocenter
            ox, ox----the projection of the isocenter on the camera
        cameraAngle:
            if the camera normal dose not point up, then there is a rotation
            of the camera about z axis, the angle is defined as right hand
            about z axis, for plus and minus sign, the degrees needed rotate
            the y axis so that it points down.
            NOTE this is a rotation of coordinates.

    output:
        a tuple consists of X, Y, Z:
            double, event coordinates in CORE
    '''
    cdef double tx, ty, X, Y, Z
    X = 0.0
    Y = 0.0
    Z = 0.0
    ty = 0.0
    tz = 0.0

    if module//4 == 0:
        X = -x + 53.0 / 2.0 - ox
        Y =  z + 38.32 + oy
        Z = -y + 53.0 * 1.5 -oz

    elif module//4 == 1:
        X =  x - 53.0 / 2.0 - ox
        Y =  z + 38.32 + oy
        Z =  y - 53.0 * 1.5 -oz

    elif module//4 == 2:
        X = -x + 53.0 / 2.0 - ox
        Y = -z + 88.88 + oy
        Z =  y - 53.0 * 1.5 -oz

    elif module//4 == 3:
        X =  x - 53.0 / 2.0 - ox
        Y = -z + 88.88 + oy
        Z = -y + 53.0 * 1.5 -oz

    if cameraAngle != 0:
        # this is a rotation of axis.
        tx = X
        ty = Y
        X =  tx * cos( cameraAngle / 180 * PI ) + ty * sin( cameraAngle / 180 * PI )
        Y = -tx * sin( cameraAngle / 180 * PI ) + ty * cos( cameraAngle / 180 * PI )

    return X, Y, Z



cpdef double scatterAngle( double e0, double eDep ):
    '''
    coneAngle = scatterAngle( double e0, double eDep )

    given initial and deposited energy, compute scatter angle ( cone angle )

    input:
        e0:
            double, the initial energy of Gamma's coming towards the camera
        eDep:
            double, the energy deposited during interaction
    output:
        scatterAngle:
            double, scattering angle (in radian) computed using Compton scatter equation.
            in the event the input deposited energy eDep is larger than the
            most energy a Compton scatter can deposit, returns -1
    '''
    cdef double a
    if e0 - eDep <= 0:
        return -1.0
    a = 1.0 - 0.511 * ( 1 / (e0 - eDep) - 1 / e0 )
    if fabs( a ) - 1 > 0:
        return -1.0
    else:
        return acos(  a )



cpdef double E0Triple(  double Ax, double Ay, double Az,
                        double Bx, double By, double Bz,
                        double Cx, double Cy, double Cz,
                        double eDepA, double eDepB ):
    '''
    E0 = E0Triple(  double Ax, double Ay, double Az,
                    double Bx, double By, double Bz,
                    double Cx, double Cy, double Cz,
                    double eDepA, double eDepB )

    compute the initial energy of the Gamma photon using the Compton equation,
    assuming all three interactions are Compton scatter.

    input:
        Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz:
            double, the coordinates of 3 interactions. event order is important.
        eDepA, eDepB:
            double, the depsited energy for the first and the second interactions

    output:
        E0:
            double, the initial energy of the Gamma photon
    '''

    cdef:
        double   ABx, ABy, ABz
        double   BCx, BCy, BCz
        double   cosTheta2
        double   E1
    ABx = Bx - Ax
    ABy = By - Ay
    ABz = Bz - Az
    BCx = Cx - Bx
    BCy = Cy - By
    BCz = Cz - Bz
    cosTheta2 = vectorDot( ABx, ABy, ABz, BCx, BCy, BCz ) / norm( ABx, ABy, ABz ) / norm( BCx, BCy, BCz )
    # the problem is that when the 3-pixel events are aligned along a line
    # the cos( theta_2) is 1.0, and ( 1-cos(theta_2) ) is 0.0 and raises
    # devide by 0 error.
    # in theory, this happens only when eDepC is 0, which means it's
    # actually an 2-pixel event, and the calculation of E0 will break.
    if fabs( cosTheta2 -1 ) < ZERO:
        # print( 'here is devide by 0, eDepB is ', eDepB)
        return -1.0
    else:
        E1 = ( eDepB + sqrt( eDepB * eDepB + 4 * eDepB * 0.511 / ( 1 - cosTheta2 ) ) ) / 2.0
        return E1 + eDepA



cpdef double pointDCAVector(    double Ax, double Ay, double Az,
                                double Bx, double By, double Bz,
                                double sx, double sy, double sz,
                                double coneAngle ):
    '''
    DCA = pointDCAVector(   double Ax, double Ay, double Az,
                            double Bx, double By, double Bz,
                            double sx, double sy, double sz,
                            double coneAngle )

    compute the distance from a point source to the cone surface.
    this is the same calculation as the function pointDCA except that, here,
    intead of rotating to cone axis, the calculation is done using vector operation.

    input:
        Ax, Ay, Az, Bx, By, Bz:
            double, event coordinates that defines the cone axis, A is the apex of the cone; order is important
        sx, sy, sz:
            double, the assumed point source position
        coneAngle:
            double, cone angle

    output:
        DCA:
            double, the distance from the point source to the cone surface, this is the shortest distance;
            a possitive DCA means the point source is outside the cone
            a negative DCA means the point is inside the cone
            an OUTOFBOUNDARY is returned when shortest distance is on the wrong side of the cone.
    '''

    cdef:
        double   Nx, Ny, Nz
        double   ps
        double   Nnorm, ABnorm, d

    ABnorm = norm( Bx - Ax, By - Ay, Bz - Az )

    # projection of source onto cone axis AB
    ps = vectorDot( sx - Ax, sy - Ay, sz - Az, Bx - Ax, By - Ay, Bz - Az ) /ABnorm

    if ( ( coneAngle < PI / 2 ) and ( ps < 0 ) ) or ( ( coneAngle > PI / 2 ) and ( ps > 0 ) ):
        Nx, Ny, Nz = cross( Bx - Ax, By - Ay, Bz - Az, sx - Ax, sy - Ay, sz - Az)
        Nnorm = norm( Nx, Ny, Nz )
        d = Nnorm / ABnorm

        if coneAngle > PI/2:
            coneAngle = PI - coneAngle

        return d * cos( coneAngle ) - fabs( ps ) * sin( coneAngle )
    else:
        return OUTOFBOUNDARY



# cpdef np.ndarray conicSectionSmeared(   np.ndarray events, double Cy=0,
cpdef tuple conicSectionSmeared(   np.ndarray events, double Cy=0,
                                        int nx=201, int nz=401, float dx=1.0, float dz=1.0,
                                        double threshold=3.0):
    '''
    h, s = conicSectionSmeared(   np.ndarray events, double Cy=0,
                                            int nx=201, int nz=401, float dx=1.0, float dz=1.0,
                                            double threshold=3.0)

    compute conic section. ellipse and hyperbola only.
    the section is smeared according an extimated pdf of speading. note the estimated
    speading is approximated to the first order, and is under estimated for the side
    that's further away from the apex of the cone.
    assumed cone back project along -y direction, the conic section is the intercept
    of cone with xz plane specified at y=Cy
    nx, nz define the size of xz plane. in x direction, it goes from -(nx-1)/2
    to (nx-1)/2, and the unit is in mm
    threshold is the parameter to determine if the trajectory passes through a pixel or not
    when a pixel is on the trajectory, its value is increased by 1.

    input:
        events:
            numpy ndarray, 2D events array, similar to that CORE takes as input, each row is a 2-pixel events,
            following the format of [eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz]
            note, it is assumed that the initial energy E0 is eDepA + eDepB
        Cy:
            double, the position of the xz plane.
            note, the methods works the best at Cy=0 plane.
        nx, nz:
            integer, number of pixels in x and z direction
        dx, dz:
            float, the pixel size in the output image, defaut 1.0
        threshold:
            double, DETERMINE IF A PIXEL IS ON THE INTERSECTION, IN TERMS HOW MANY SIGMA THE PIXLE
            IS AWAY

    output:
        h:
            numpy ndarray, a 2D array with nz row, nx collumn, a heat map of conic sections
        s:
            numpy ndarray, sigma of each cone
    '''

    cdef:
        double      eDepA=0.0, Ax=0.0, Ay=0.0, Az=0.0, eDepB=0.0, Bx=0.0, By=0.0, Bz=0.0
        double      t=0.0, Cx=0.0, Cz=0.0, Dx=0.0, Dy=0.0, Dz=0.0, Ex=0.0, Ey=0.0, Ez=0.0
        # A1 and C1 are A and C in the transformed axes:
        double      A1x=0.0, A1y=0.0, A1z=0.0, C1x=0.0, C1y=0.0, C1z=0.0
        double      coneAngle=0.0
        double      sinBeta=0.0, cosBeta=0.0, phi=0.0, r=0.0
        # the length:
        double      ACnorm=0.0, ADnorm=0.0, DGnorm=0.0, DFnorm=0.0, DEnorm=0.0
        double      CDnorm=0.0, CM1norm=0.0, DF1norm=0.0, DF2norm=0.0
        double      focalLength=0.0, majorAxis=0.0, minorAxis=0.0
        double      M1x=0.0, M1y=0.0, M1z=0.0, M2x=0.0, M2y=0.0, M2z=0.0
        double      F1x=0.0, F1y=0.0, F1z=0.0, F2x=0.0, F2y=0.0, F2z=0.0
        int         nEvents=len(events), iEvent=0, i=0, j=0
        # double      dx=(nx-1)/nx, dz =(nz-1)/nz
        # double      dx=1.0, dz=1.0
        bint        rightSide=False
        double      x=0.0, z=0.0, distLeft=0.0, distRight=0.0

        # the following are used for compute the uncertainties:
        float   sigma=0.0, theta1=0.0,
        float   hy=0.0, E0=0.0, E1=0.0, dSum=0.0
        int     nDf=20
        int     badCones=0

        np.ndarray[np.float64_t, ndim=1]  df = np.arange( -1, 1, 0.1 )
        np.ndarray[np.float64_t, ndim=1]  pDf = np.zeros( nDf )
        np.ndarray[np.float64_t, ndim=1]  pFitCoeff = np.zeros( 3 )
        np.ndarray[np.float64_t, ndim=1]  sigmaReturn = np.zeros( nEvents )
        # np.ndarray[np.float64_t, ndim=1]  sigmaReturn2 = np.zeros( nEvents )

        # h is the output
        np.ndarray[np.float64_t, ndim=2]  h=np.zeros( [nz, nx], dtype=np.float64 )


    # sigmaReturn = []

    # h = np.zeros( [nz, nx], dtype=np.int )
    # nEvents = len( events )

    start_time = time.time()
    for iEvent in range( nEvents ):
        eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz = events[iEvent]
        E0 = eDepA + eDepB

        coneAngle = scatterAngle( E0, eDepA )

        # sigma = sqrt( 25e-6 + 2.35**2 * E1 * 5e-6 )
        # theta1 = acos( 1 - 0.511 * ( 1 / ( E0 - E1 ) - 1 / E0 ) )

        # there should be no negative coneAngle, negative coneAngle means
        # for the depsited energy, it's can not be a Compton scatter.
        if coneAngle < 0:
            badCones += 1
            # print( 'impossible coneAngle detected!', coneAngle)
            continue

        # if coneAngle > PI / 2:
        #     coneAngle = PI - coneAngle

        # when By = Ay, the cone axis parallel to xz plane, current method fail
        if fabs( By - Ay ) < ZERO:
            print( iEvent )
            print( 'cone axis parallel to x-z plane, the current method fail!' )
            continue

        # find coordinates for point C and D.
        # C is the intersection of line BA with xz plane
        # D is the projection of point A on xz plane
        Dx = Ax
        Dy = Cy
        Dz = Az
        t = ( By - Cy ) / ( By - Ay )
        Cx = Bx + t * ( Ax - Bx )
        Cz = Bz + t * ( Az - Bz )

        ADnorm = norm( Dx - Ax, Dy - Ay, Dz - Az )
        CDnorm = norm( Dx - Cx, Dy - Cy, Dz - Cz)

        # when C and D overlap, then AB is perpendicular to xz plane
        # the intersection is a circle:
        if CDnorm < ZERO:
            if coneAngle < PI / 2:
                r = ( Ay - Cy ) * tan( coneAngle )
            else:
                r = ( Ay - Cy ) * tan( PI - coneAngle )

            E1 = eDepA
            sigma = sqrt( 25e-6 + 2.35**2 * E1 * 5e-6 )
            theta1 = fabs( coneAngle )
            hy = fabs( Ay - Cy )
            # sigmaReturn2[iEvent] = (0.511 + E0 * ( 1 - cos(theta1) ) )**2 * hy / 0.511 / E0**2 / sin(theta1) / cos(theta1)**2 * sigma

            for i in range( nDf ):
                # pDf[i] = log( 1 / sqrt( 2 * PI ) / sigma ) - ( E0**2 * (1 - cos(theta1 + cos(theta1)**2 * df[i] / hy ) ) / ( 0.511 + E0 * ( 1- cos( theta1 + cos(theta1)**2 * df[i] / hy ) ) )  - E1 )**2 / 2 / sigma**2
                # pDf[i] = pDf[i] + log( fabs( ( 0.511 * E0**2 * sin( theta1 + cos(theta1)**2 * df[i] / hy ) ) * cos( theta1 )**2 / ( 0.511 + E0 * ( 1 - cos( theta1 + cos(theta1)**2 * df[i] / hy ) ) )**2 / hy ) )
                pDf[i] = log( 1 / sqrt( 2 * PI ) / sigma ) - ( 0.511 * E0**2  * sin( theta1 ) * cos( theta1 )**2 / (0.511 + E0 *( 1 - cos(theta1) ) )**2 / hy * df[i] )**2 / sigma**2 / 2
                pDf[i] = pDf[i] + log( 0.511 * E0**2 * sin( theta1 ) * cos( theta1 )**2 / ( 0.511 + E0 * ( 1 - cos( theta1 ) ) )**2 / hy )
            pFitCoeff = np.polyfit( df, pDf, 2)
            if pFitCoeff[0] > 0:
                continue
            sigma = sqrt( -1 / 2 / pFitCoeff[0] )
            sigmaReturn[iEvent] = sigma

            # sigmaReturn.append( sigma )

            for i in prange( nz, nogil=True ):
                z = ( i - ( nz - 1 ) / 2 ) * dz
                for j in range( nx ):
                    x = ( j - ( nx - 1 ) / 2 ) * dx
                    dSum = fabs( sqrt( (x -Cx)**2 + (z - Cz)**2 ) - r )
                    # dSum = fabs( norm( x -Cx, 0, z - Cz ) - r )
                    if dSum < threshold * sigma:
                        h[i, j] += 1 / sqrt( 2 * 3.14159 ) / sigma * exp( -dSum**2 / 2 / sigma**2 )
                        # h[i, j] += 1 / sqrt( 2 * PI ) / sigma * exp( -dSum**2 / 2 / sigma**2 )
            continue

        # C and D not overlapping, either ellipse or hyperbola:
        sinBeta = ( Dx - Cx ) / CDnorm
        cosBeta = ( Dz - Cz ) / CDnorm

        if coneAngle < PI/2:
            phi = atan( CDnorm / ADnorm ) - coneAngle
        else:
            phi = atan( CDnorm / ADnorm ) - ( PI - coneAngle )

        # theta1 = fabs( phi )
        theta1 = coneAngle if coneAngle < PI / 2 else PI - coneAngle
        # E1 = E0 - 1 / ( 1 / E0 + ( 1 - cos(theta1) ) / 0.511 )
        E1 = eDepA
        sigma = sqrt( 25e-6 + 2.35**2 * E1 * 5e-6 )
        hy = fabs( Ay - Cy )
        # sigmaReturn2[iEvent] = (0.511 + E0 * ( 1 - cos(theta1) ) )**2 * hy / 0.511 / E0**2 / sin(theta1) / cos(phi)**2 * sigma
        # print( 'tehta1 ', theta1 )
        # print( 'hy ', hy)
        # print( 'sigma before ', sigma )
        for i in range( nDf ):
            # pDf[i] = log( 1 / sqrt( 2 * PI ) / sigma ) - ( E0**2 * (1 - cos(theta1 + cos(theta1)**2 * df[i] / hy ) ) / (0.511 + E0 * ( 1- cos( theta1 + cos(theta1)**2 * df[i] / hy ) ) )  - E1 )**2 / 2 / sigma**2
            # pDf[i] = pDf[i] + log( fabs( ( 0.511 * E0**2 * sin( theta1 + cos(theta1)**2 * df[i] / hy ) ) * cos( theta1 )**2 / ( 0.511 + E0 * (1 - cos( theta1 + cos(theta1)**2 * df[i] / hy ) ) )**2 / hy ) )
            pDf[i] = log( 1 / sqrt( 2 * PI ) / sigma ) - ( 0.511 * E0**2  * sin( theta1 ) * cos( phi )**2 / (0.511 + E0 *( 1 - cos(theta1) ) )**2 / hy * df[i] )**2 / sigma**2 / 2
            pDf[i] = pDf[i] + log( 0.511 * E0**2 * sin( theta1 ) * cos( phi )**2 / ( 0.511 + E0 * ( 1 - cos( theta1 ) ) )**2 / hy )

            # print( pDf[i], ',' )
        pFitCoeff = np.polyfit( df, pDf, 2)
        # print(iEvent, theta1, pFitCoeff )
        if pFitCoeff[0] > 0:
            continue
        sigma = sqrt( -1 / 2 / pFitCoeff[0] )
        sigmaReturn[iEvent] = sigma
        # sigmaReturn.append( sigma )
        # print( 'sigma ', sigma)
        # print( 'sigma after ', sigma)
        # when the intersection is an ellipse:
        if ( ( By > Ay and coneAngle < PI / 2 and 2 * coneAngle + phi < PI / 2 )
             or
             ( Ay > By and coneAngle > PI / 2 and 2 * ( PI - coneAngle ) + phi < PI / 2 )
            ):

            # print( 'in ellipse!')
            if Ay > By and coneAngle > PI / 2:
                coneAngle = PI - coneAngle

            DGnorm = ADnorm * tan( 2 * coneAngle + phi )
            DFnorm = ADnorm * tan( phi )
            majorAxis = ( DGnorm - DFnorm ) / 2.0
            DEnorm = DFnorm + majorAxis
            Ex = Dx + DEnorm * ( Cx - Dx ) / CDnorm
            Ey = Dy + DEnorm * ( Cy - Dy ) / CDnorm
            Ez = Dz + DEnorm * ( Cz - Dz ) / CDnorm

            A1x, A1y, A1z = matrixVectorDot( cosBeta, 0, -sinBeta, 0, 1, 0, sinBeta, 0, cosBeta, Ax - Ex, Ay - Ey, Az - Ez )
            C1x, C1y, C1z = matrixVectorDot( cosBeta, 0, -sinBeta, 0, 1, 0, sinBeta, 0, cosBeta, Cx - Ex, Cy - Ey, Cz - Ez )

            ACnorm = norm( A1x - C1x, A1y - C1y, A1z - C1z )
            minorAxis = sqrt( ( A1y * A1y + A1z * ( A1z - C1z ) )**2 / ( ACnorm * cos( coneAngle ) )**2 - A1y * A1y - A1z * A1z )
            focalLength = sqrt( majorAxis * majorAxis - minorAxis * minorAxis )

            M1x = Ex + focalLength * ( Cx - Dx ) / CDnorm
            M1y = Ey + focalLength * ( Cy - Dy ) / CDnorm
            M1z = Ez + focalLength * ( Cz - Dz ) / CDnorm
            M2x = Ex - focalLength * ( Cx - Dx ) / CDnorm
            M2y = Ey - focalLength * ( Cy - Dy ) / CDnorm
            M2z = Ez - focalLength * ( Cz - Dz ) / CDnorm
            # print( 'E coord ', Ex, Ey, Ez )
            # print( 'M1 coord ', M1x, M1y, M1z )
            # print( 'M2 coord ', M2x, M2y, M2z )
            # print( 'majorAxis ', majorAxis )
            # print( 'minorAxis ', minorAxis )
            # print( 'focalLength ', focalLength )

            for i in prange( nz, nogil=True ):
                z = ( i - ( nz - 1 ) / 2 ) * dz
                for j in range( nx ):
                    x = ( j - ( nx - 1 ) / 2 ) * dx
                    distLeft = sqrt( (x - M1x)**2 + (z - M1z)**2 )
                    distRight = sqrt( (x - M2x)**2 + (z - M2z)**2 )
                    dSum = fabs( distLeft +  distRight - 2 * majorAxis )
                    if  dSum < threshold * sigma:
                        h[i, j] += 1 / sqrt( 2 * 3.14159 ) / sigma * exp( -dSum**2 / 2 / sigma**2 )
                        # h[i, j] += 1 / sqrt( 2 * PI ) / sigma * exp( -dSum**2 / 2 / sigma**2 )


        # in case the trajectory is a proballa, it is  not defined
        elif ( ( By > Ay  and coneAngle < PI / 2 and fabs( 2 * coneAngle + phi - PI / 2 ) < ZERO )
               or
               ( Ay > By  and coneAngle > PI / 2 and fabs( 2 * (PI - coneAngle) + phi - PI / 2 ) < ZERO )
             ):
            print( 'prabolla needed ')
            continue

        # intersection is hyperbola
        elif ( ( By > Ay and coneAngle < PI / 2 and 2 * coneAngle + phi > PI / 2)
               or
               ( By > Ay and coneAngle > PI / 2 )
               or
               ( Ay > By and coneAngle > PI / 2 and 2 * ( PI - coneAngle ) + phi > PI /2 )
               or
               ( Ay > By and coneAngle < PI / 2 )
             ):

            # print( 'in hyperbola')
            if By > Ay and coneAngle < PI / 2 and 2 * coneAngle + phi > PI / 2:
                rightSide = False
            elif By > Ay and coneAngle > PI / 2:
                coneAngle = PI - coneAngle
                rightSide = True
            elif Ay > By and coneAngle > PI / 2:
                # need to make sure that this is a hyperbola
                if 2 * ( PI - coneAngle ) + phi > PI /2:
                    # print( 'Ay is larger and sure of hyperbola')
                    coneAngle = PI - coneAngle
                    rightSide = False
                # elif 2 * ( PI - coneAngle ) + phi > PI:
                #     coneAngle = PI - coneAngle
                #     rightSide = True
            elif Ay > By and coneAngle < PI / 2:
                rightSide = True

            # in this case, there will be no right side hyperbola as it won't
            # intercept with xz plane
            if rightSide and PI - 2 * coneAngle - phi > PI / 2:
                continue

            ACnorm = norm( Cx - Ax, Cy - Ay, Cz - Az )
            r = ADnorm * ACnorm / ( ACnorm + ADnorm/sin(coneAngle) )
            if r < 0:
                print( 'red alert: r is negative' )
                continue

            CM1norm = r / ADnorm * CDnorm
            DF1norm = ADnorm * tan( phi )

            M1x = Cx + CM1norm * ( Dx - Cx ) / CDnorm
            M1y = Cy + CM1norm * ( Dy - Cy ) / CDnorm
            M1z = Cz + CM1norm * ( Dz - Cz ) / CDnorm

            F1x = Dx + DF1norm * ( Cx - Dx ) / CDnorm
            F1y = Dy + DF1norm * ( Cy - Dy ) / CDnorm
            F1z = Dz + DF1norm * ( Cz - Dz ) / CDnorm

            DF2norm = ADnorm * tan( PI - 2 * coneAngle - phi )
            F2x = Dx + DF2norm * ( Dx - Cx ) / CDnorm
            F2y = Dy + DF2norm * ( Dy - Cy ) / CDnorm
            F2z = Dz + DF2norm * ( Dz - Cz ) / CDnorm

            Ex = ( F1x + F2x ) / 2.0
            Ey = ( F1y + F2y ) / 2.0
            Ez = ( F1z + F2z ) / 2.0

            focalLength = norm( Ex - M1x, Ey - M1y, Ez - M1z )
            majorAxis = norm( Ex - F1x, Ey - F1y, Ez - F1z )
            minorAxis = sqrt( focalLength * focalLength - majorAxis * majorAxis )

            M2x = Ex + focalLength * ( Dx - Cx ) / CDnorm
            M2y = Ey + focalLength * ( Dy - Cy ) / CDnorm
            M2z = Ez + focalLength * ( Dz - Cz ) / CDnorm
            # print( 'CM1 norm ', CM1norm, ' DF1 norm ', DF1norm )
            # print( 'DF2 norm ', DF2norm )
            # print( 'C ', Cx, Cy, Cz )
            # print( 'D ', Dx, Dy, Dz )
            # print( 'F1 coord ', F1x, F1y, F1z )
            # print( 'F2 coord ', F2x, F2y, F2z )
            # print( 'M1 coord ', M1x, M1y, M1z )
            # print( 'M2 coord ', M2x, M2y, M2z )
            # print( 'E coord ', Ex, Ey, Ez )
            # print( 'focalLength ', focalLength )
            # print( ' major aixs', majorAxis )
            # print(' minor axis', minorAxis )
            # print( 'coneAngle ', coneAngle )
            # print( 'phi ', phi)
            # print( 'rightSide is ', rightSide)


            # for i in range( nz ):
            #     z = ( i - (nz-1)/2 ) * dz
            #     for j in range( nx ):
            #         x = ( j - (nx-1)/2 ) * dx
            #         distLeft = norm( x - M1x, 0, z - M1z )
            #         distRight = norm( x - M2x, 0, z - M2z )
            #         if fabs( fabs( distLeft - distRight ) - 2 * majorAxis ) < 1:
            #         # if fabs( ( norm( x - M2x, 0, z - M2z ) - norm( x - M1x, 0, z - M1z ) ) - 2 * majorAxis ) < 1:
            #
            #             h[i, j] += 1

            if not rightSide:
                # print( 'not right side')
                for i in prange( nz, nogil=True ):
                    z = ( i - ( nz- 1 ) / 2 ) * dz
                    for j in range( nx ):
                        x = ( j - (nx - 1 ) / 2 ) * dx
                        distLeft = sqrt( (x - M1x)**2 + (z - M1z)**2 )
                        distRight = sqrt( (x - M2x)**2 + (z - M2z)**2 )
                        dSum = fabs( distRight - distLeft - 2 * majorAxis )
                        if distRight > distLeft and  dSum < threshold * sigma:
                            h[i, j] += 1 / sqrt( 2 * 3.14159 ) / sigma * exp( -dSum**2 / 2 / sigma**2 )
                            # h[i, j] += 1 / sqrt( 2 * PI ) / sigma * exp( -dSum**2 / 2 / sigma**2 )
            else:
                # print( 'right side ')
                for i in prange( nz, nogil=True ):
                    z = ( i - (nz-1)/2 ) * dz
                    for j in range( nx ):
                        x = ( j - (nx-1)/2 ) * dx
                        distLeft = sqrt( (x - M1x)**2 + (z - M1z)**2 )
                        distRight = sqrt( (x - M2x)**2 + (z - M2z)**2 )
                        dSum = fabs( distLeft - distRight - 2 * majorAxis )
                        if distLeft > distRight and dSum < threshold:
                            h[i, j] += 1 / sqrt( 2 * 3.14159 ) / sigma * exp( -dSum**2 / 2 / sigma**2 )
                            # h[i, j] += 1 / sqrt( 2 * PI ) / sigma * exp( -dSum**2 / 2 / sigma**2 )
    end_time = time.time()
    print( 'total number of cones', nEvents )
    print( 'number of bad cones (impossible cone angle)', badCones)
    print( 'time to compute back projection ', end_time - start_time )
    # return h
    return h, sigmaReturn
    # return h, sigmaReturn



# cpdef tuple conicSectionSmearedVaryingSigma(   np.ndarray events, double Cy=0,
# cpdef np.ndarray conicSectionSmearedVaryingSigma(   np.ndarray events, double Cy=0,
#                                         int nx=201, int nz=401, float dx=1.0, float dz=1.0,
#                                         double threshold=3.0):
#     '''
#     h = conicSectionSmearedVaryingSigma(   np.ndarray events, double Cy=0,
#                                             int nx=201, int nz=401, float dx=1.0, float dz=1.0,
#                                             double threshold=3.0)
#
#     compute conic section. ellipse and hyperbola only.
#     the section is smeared according an extimated pdf of speading. note the estimated
#     speading is approximated to the first order, and is under estimated for the side
#     that's further away from the apex of the cone.
#     assumed cone back project along -y direction, the conic section is the intercept
#     of cone with xz plane specified at y=Cy
#     nx, nz define the size of xz plane. in x direction, it goes from -(nx-1)/2
#     to (nx-1)/2, and the unit is in mm
#     threshold is the parameter to determine if the trajectory passes through a pixel or not
#     when a pixel is on the trajectory, its value is increased by 1.
#
#     input:
#         events:
#             numpy ndarray, 2D events array, similar to that CORE takes as input, each row is a 2-pixel events,
#             following the format of [eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz]
#             note, it is assumed that the initial energy E0 is eDepA + eDepB
#         Cy:
#             double, the position of the xz plane.
#             note, the methods works the best at Cy=0 plane.
#         nx, nz:
#             integer, number of pixels in x and z direction
#         dx, dz:
#             float, the pixel size in the output image, defaut 1.0
#         threshold:
#             double, DETERMINE IF A PIXEL IS ON THE INTERSECTION, IN TERMS HOW MANY SIGMA THE PIXLE
#             IS AWAY
#
#     output:
#         h:
#             numpy ndarray, a 2D array with nz row, nx collumn, a heat map of conic sections
#
#     '''
#
#     cdef:
#         double      eDepA=0.0, Ax=0.0, Ay=0.0, Az=0.0, eDepB=0.0, Bx=0.0, By=0.0, Bz=0.0
#         double      t=0.0, Cx=0.0, Cz=0.0, Dx=0.0, Dy=0.0, Dz=0.0, Ex=0.0, Ey=0.0, Ez=0.0
#         # A1 and C1 are A and C in the transformed axes:
#         double      A1x=0.0, A1y=0.0, A1z=0.0, C1x=0.0, C1y=0.0, C1z=0.0
#         double      coneAngle=0.0
#         double      sinBeta=0.0, cosBeta=0.0, phi=0.0, r=0.0
#         # the length:
#         double      ACnorm=0.0, ADnorm=0.0, DGnorm=0.0, DFnorm=0.0, DEnorm=0.0
#         double      CDnorm=0.0, CM1norm=0.0, DF1norm=0.0, DF2norm=0.0
#         double      focalLength=0.0, majorAxis=0.0, minorAxis=0.0
#         double      M1x=0.0, M1y=0.0, M1z=0.0, M2x=0.0, M2y=0.0, M2z=0.0
#         double      F1x=0.0, F1y=0.0, F1z=0.0, F2x=0.0, F2y=0.0, F2z=0.0
#         int         nEvents=len(events), iEvent=0, i=0, j=0
#         # double      dx=(nx-1)/nx, dz =(nz-1)/nz
#         # double      dx=1.0, dz=1.0
#         bint        rightSide=False
#         double      x=0.0, z=0.0, distLeft=0.0, distRight=0.0
#
#         # the following are used for compute the uncertainties:
#         float   sigma=0.0, theta1=0.0, sigma1=0.0, sigma2=0.0
#         float   hy=0.0, E0=0.0, E1=0.0, dSum=0.0
#         # int     nDf=20
#
#         # np.ndarray[np.float64_t, ndim=1]  df = np.arange( -1, 1, 0.1 )
#         # np.ndarray[np.float64_t, ndim=1]  pDf = np.zeros( nDf )
#         # np.ndarray[np.float64_t, ndim=1]  pFitCoeff = np.zeros( 3 )
#         # np.ndarray[np.float64_t, ndim=1]  sigmaReturn = np.zeros( nEvents )
#         # np.ndarray[np.float64_t, ndim=1]  sigmaReturn2 = np.zeros( nEvents )
#
#         # h is the output
#         np.ndarray[np.float64_t, ndim=2]  h=np.zeros( [nz, nx], dtype=np.float64 )
#
#
#     # sigmaReturn = []
#
#     # h = np.zeros( [nz, nx], dtype=np.int )
#     # nEvents = len( events )
#
#     start_time = time.time()
#     for iEvent in range( nEvents ):
#         eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz = events[iEvent]
#         E0 = eDepA + eDepB
#
#         coneAngle = scatterAngle( E0, eDepA )
#
#         # sigma = sqrt( 25e-6 + 2.35**2 * E1 * 5e-6 )
#         # theta1 = acos( 1 - 0.511 * ( 1 / ( E0 - E1 ) - 1 / E0 ) )
#
#         # there should be no negative coneAngle, negative coneAngle means
#         # for the depsited energy, it's can not be a Compton scatter.
#         if coneAngle < 0:
#             print( 'impossible coneAngle detected!', coneAngle)
#             continue
#
#         # if coneAngle > PI / 2:
#         #     coneAngle = PI - coneAngle
#
#         # when By = Ay, the cone axis parallel to xz plane, current method fail
#         if fabs( By - Ay ) < ZERO:
#             print( iEvent )
#             print( 'cone axis parallel to x-z plane, the current method fail!' )
#             continue
#
#         # find coordinates for point C and D.
#         # C is the intersection of line BA with xz plane
#         # D is the projection of point A on xz plane
#         Dx = Ax
#         Dy = Cy
#         Dz = Az
#         t = ( By - Cy ) / ( By - Ay )
#         Cx = Bx + t * ( Ax - Bx )
#         Cz = Bz + t * ( Az - Bz )
#
#         ADnorm = norm( Dx - Ax, Dy - Ay, Dz - Az )
#         CDnorm = norm( Dx - Cx, Dy - Cy, Dz - Cz)
#
#         # when C and D overlap, then AB is perpendicular to xz plane
#         # the intersection is a circle:
#         if CDnorm < ZERO:
#             if coneAngle < PI / 2:
#                 r = ( Ay - Cy ) * tan( coneAngle )
#             else:
#                 r = ( Ay - Cy ) * tan( PI - coneAngle )
#
#             E1 = eDepA
#             sigma = sqrt( 25e-6 + 2.35**2 * E1 * 5e-6 )
#             theta1 = fabs( coneAngle )
#             hy = fabs( Ay - Cy )
#             sigma = (0.511 + E0 * ( 1 - cos(theta1) ) )**2 * hy / 0.511 / E0**2 / sin(theta1) / cos(theta1)**2 * sigma
#
#             # for i in range( nDf ):
#             #     # pDf[i] = log( 1 / sqrt( 2 * PI ) / sigma ) - ( E0**2 * (1 - cos(theta1 + cos(theta1)**2 * df[i] / hy ) ) / ( 0.511 + E0 * ( 1- cos( theta1 + cos(theta1)**2 * df[i] / hy ) ) )  - E1 )**2 / 2 / sigma**2
#             #     # pDf[i] = pDf[i] + log( fabs( ( 0.511 * E0**2 * sin( theta1 + cos(theta1)**2 * df[i] / hy ) ) * cos( theta1 )**2 / ( 0.511 + E0 * ( 1 - cos( theta1 + cos(theta1)**2 * df[i] / hy ) ) )**2 / hy ) )
#             #     pDf[i] = log( 1 / sqrt( 2 * PI ) / sigma ) - ( 0.511 * E0**2  * sin( theta1 ) * cos( theta1 )**2 / (0.511 + E0 *( 1 - cos(theta1) ) )**2 / hy * df[i] )**2 / 2 / sigma**2
#             #     pDf[i] = pDf[i] + log( 0.511 * E0**2 * sin( theta1 ) * cos( theta1 )**2 / ( 0.511 + E0 * ( 1 - cos( theta1 ) ) )**2 / hy )
#             # pFitCoeff = np.polyfit( df, pDf, 2)
#             # if pFitCoeff[0] > 0:
#             #     continue
#             # sigma = sqrt( -1 / 2 / pFitCoeff[0] )
#             # sigmaReturn[iEvent] = sigma
#
#             # sigmaReturn.append( sigma )
#
#             for i in range( nz ):
#                 z = ( i - ( nz - 1 ) / 2 ) * dz
#                 for j in range( nx ):
#                     x = ( j - ( nx - 1 ) / 2 ) * dx
#                     dSum = fabs( norm( x -Cx, 0, z - Cz ) - r )
#                     if dSum < threshold * sigma:
#                         h[i, j] += 1 / sqrt( 2 * PI ) / sigma * exp( -dSum**2 / 2 / sigma**2 )
#             continue
#
#         # C and D not overlapping, either ellipse or hyperbola:
#         sinBeta = ( Dx - Cx ) / CDnorm
#         cosBeta = ( Dz - Cz ) / CDnorm
#
#         if coneAngle < PI/2:
#             phi = atan( CDnorm / ADnorm ) - coneAngle
#         else:
#             phi = atan( CDnorm / ADnorm ) - ( PI - coneAngle )
#
#         # theta1 = fabs( phi )
#         theta1 = coneAngle if coneAngle < PI / 2 else PI - coneAngle
#         # E1 = E0 - 1 / ( 1 / E0 + ( 1 - cos(theta1) ) / 0.511 )
#         E1 = eDepA
#         sigma = sqrt( 25e-6 + 2.35**2 * E1 * 5e-6 )
#         hy = fabs( Ay - Cy )
#         # sigmaReturn2[iEvent] = (0.511 + E0 * ( 1 - cos(theta1) ) )**2 * hy / 0.511 / E0**2 / sin(theta1) / cos(phi)**2 * sigma
#         # print( 'tehta1 ', theta1 )
#         # print( 'hy ', hy)
#         # print( 'sigma before ', sigma )
#         # for i in range( nDf ):
#         #     # pDf[i] = log( 1 / sqrt( 2 * PI ) / sigma ) - ( E0**2 * (1 - cos(theta1 + cos(theta1)**2 * df[i] / hy ) ) / (0.511 + E0 * ( 1- cos( theta1 + cos(theta1)**2 * df[i] / hy ) ) )  - E1 )**2 / 2 / sigma**2
#         #     # pDf[i] = pDf[i] + log( fabs( ( 0.511 * E0**2 * sin( theta1 + cos(theta1)**2 * df[i] / hy ) ) * cos( theta1 )**2 / ( 0.511 + E0 * (1 - cos( theta1 + cos(theta1)**2 * df[i] / hy ) ) )**2 / hy ) )
#         #     pDf[i] = log( 1 / sqrt( 2 * PI ) / sigma ) - ( 0.511 * E0**2  * sin( theta1 ) * cos( phi )**2 / (0.511 + E0 *( 1 - cos(theta1) ) )**2 / hy * df[i] )**2 / 2 / sigma**2
#         #     pDf[i] = pDf[i] + log( 0.511 * E0**2 * sin( theta1 ) * cos( phi )**2 / ( 0.511 + E0 * ( 1 - cos( theta1 ) ) )**2 / hy )
#         #
#         #     # print( pDf[i], ',' )
#         # pFitCoeff = np.polyfit( df, pDf, 2)
#         # # print(iEvent, theta1, pFitCoeff )
#         # if pFitCoeff[0] > 0:
#         #     continue
#         # sigma = sqrt( -1 / 2 / pFitCoeff[0] )
#         # sigmaReturn[iEvent] = sigma
#         # sigmaReturn.append( sigma )
#         # print( 'sigma ', sigma)
#         # print( 'sigma after ', sigma)
#
#         # when the intersection is an ellipse:
#         if ( ( By > Ay and coneAngle < PI / 2 and 2 * coneAngle + phi < PI / 2 )
#              or
#              ( Ay > By and coneAngle > PI / 2 and 2 * ( PI - coneAngle ) + phi < PI / 2 )
#             ):
#
#             # print( 'in ellipse!')
#             if Ay > By and coneAngle > PI / 2:
#                 coneAngle = PI - coneAngle
#
#             theta1 = coneAngle
#             E1 = eDepA
#             sigma = sqrt( 25e-6 + 2.35**2 * E1 * 5e-6 )
#             hy = fabs( Ay - Cy )
#             sigma1 = (0.511 + E0 * ( 1 - cos(theta1) ) )**2 * hy / 0.511 / E0**2 / sin(theta1) / cos(phi)**2 * sigma
#             sigma2 = (0.511 + E0 * ( 1 - cos(theta1) ) )**2 * hy / 0.511 / E0**2 / sin(theta1) / cos(phi + 2 * theta1)**2 * sigma
#
#             DGnorm = ADnorm * tan( 2 * coneAngle + phi )
#             DFnorm = ADnorm * tan( phi )
#             majorAxis = ( DGnorm - DFnorm ) / 2.0
#             DEnorm = DFnorm + majorAxis
#             Ex = Dx + DEnorm * ( Cx - Dx ) / CDnorm
#             Ey = Dy + DEnorm * ( Cy - Dy ) / CDnorm
#             Ez = Dz + DEnorm * ( Cz - Dz ) / CDnorm
#
#             A1x, A1y, A1z = matrixVectorDot( cosBeta, 0, -sinBeta, 0, 1, 0, sinBeta, 0, cosBeta, Ax - Ex, Ay - Ey, Az - Ez )
#             C1x, C1y, C1z = matrixVectorDot( cosBeta, 0, -sinBeta, 0, 1, 0, sinBeta, 0, cosBeta, Cx - Ex, Cy - Ey, Cz - Ez )
#
#             ACnorm = norm( A1x - C1x, A1y - C1y, A1z - C1z )
#             minorAxis = sqrt( ( A1y * A1y + A1z * ( A1z - C1z ) )**2 / ( ACnorm * cos( coneAngle ) )**2 - A1y * A1y - A1z * A1z )
#             focalLength = sqrt( majorAxis * majorAxis - minorAxis * minorAxis )
#
#             M1x = Ex + focalLength * ( Cx - Dx ) / CDnorm
#             M1y = Ey + focalLength * ( Cy - Dy ) / CDnorm
#             M1z = Ez + focalLength * ( Cz - Dz ) / CDnorm
#             M2x = Ex - focalLength * ( Cx - Dx ) / CDnorm
#             M2y = Ey - focalLength * ( Cy - Dy ) / CDnorm
#             M2z = Ez - focalLength * ( Cz - Dz ) / CDnorm
#             # print( 'E coord ', Ex, Ey, Ez )
#             # print( 'M1 coord ', M1x, M1y, M1z )
#             # print( 'M2 coord ', M2x, M2y, M2z )
#             # print( 'majorAxis ', majorAxis )
#             # print( 'minorAxis ', minorAxis )
#             # print( 'focalLength ', focalLength )
#
#             for i in range( nz ):
#                 z = ( i - ( nz - 1 ) / 2 ) * dz
#                 for j in range( nx ):
#                     x = ( j - ( nx - 1 ) / 2 ) * dx
#                     dSum = fabs( norm( x - M1x, 0, z - M1z ) + norm( x - M2x, 0, z - M2z ) - 2 * majorAxis )
#                     if  dSum < threshold * sigma:
#                         h[i, j] += 1 / sqrt( 2 * PI ) / sigma * exp( -dSum**2 / 2 / sigma**2 )
#
#
#         # in case the trajectory is a proballa, it is  not defined
#         elif ( ( By > Ay  and coneAngle < PI / 2 and fabs( 2 * coneAngle + phi - PI / 2 ) < ZERO )
#                or
#                ( Ay > By  and coneAngle > PI / 2 and fabs( 2 * (PI - coneAngle) + phi - PI / 2 ) < ZERO )
#              ):
#             print( 'prabolla needed ')
#             continue
#
#         # intersection is hyperbola
#         elif ( ( By > Ay and coneAngle < PI / 2 and 2 * coneAngle + phi > PI / 2)
#                or
#                ( By > Ay and coneAngle > PI / 2 )
#                or
#                ( Ay > By and coneAngle > PI / 2 and 2 * ( PI - coneAngle ) + phi > PI /2 )
#                or
#                ( Ay > By and coneAngle < PI / 2 )
#              ):
#
#             # print( 'in hyperbola')
#             if By > Ay and coneAngle < PI / 2 and 2 * coneAngle + phi > PI / 2:
#                 rightSide = False
#             elif By > Ay and coneAngle > PI / 2:
#                 coneAngle = PI - coneAngle
#                 rightSide = True
#             elif Ay > By and coneAngle > PI / 2:
#                 # need to make sure that this is a hyperbola
#                 if 2 * ( PI - coneAngle ) + phi > PI /2:
#                     # print( 'Ay is larger and sure of hyperbola')
#                     coneAngle = PI - coneAngle
#                     rightSide = False
#                 # elif 2 * ( PI - coneAngle ) + phi > PI:
#                 #     coneAngle = PI - coneAngle
#                 #     rightSide = True
#             elif Ay > By and coneAngle < PI / 2:
#                 rightSide = True
#
#             # in this case, there will be no right side hyperbola as it won't
#             # intercept with xz plane
#             if rightSide and PI - 2 * coneAngle - phi > PI / 2:
#                 continue
#
#             ACnorm = norm( Cx - Ax, Cy - Ay, Cz - Az )
#             r = ADnorm * ACnorm / ( ACnorm + ADnorm/sin(coneAngle) )
#             if r < 0:
#                 print( 'red alert: r is negative' )
#                 continue
#
#             CM1norm = r / ADnorm * CDnorm
#             DF1norm = ADnorm * tan( phi )
#
#             M1x = Cx + CM1norm * ( Dx - Cx ) / CDnorm
#             M1y = Cy + CM1norm * ( Dy - Cy ) / CDnorm
#             M1z = Cz + CM1norm * ( Dz - Cz ) / CDnorm
#
#             F1x = Dx + DF1norm * ( Cx - Dx ) / CDnorm
#             F1y = Dy + DF1norm * ( Cy - Dy ) / CDnorm
#             F1z = Dz + DF1norm * ( Cz - Dz ) / CDnorm
#
#             DF2norm = ADnorm * tan( PI - 2 * coneAngle - phi )
#             F2x = Dx + DF2norm * ( Dx - Cx ) / CDnorm
#             F2y = Dy + DF2norm * ( Dy - Cy ) / CDnorm
#             F2z = Dz + DF2norm * ( Dz - Cz ) / CDnorm
#
#             Ex = ( F1x + F2x ) / 2.0
#             Ey = ( F1y + F2y ) / 2.0
#             Ez = ( F1z + F2z ) / 2.0
#
#             focalLength = norm( Ex - M1x, Ey - M1y, Ez - M1z )
#             majorAxis = norm( Ex - F1x, Ey - F1y, Ez - F1z )
#             minorAxis = sqrt( focalLength * focalLength - majorAxis * majorAxis )
#
#             M2x = Ex + focalLength * ( Dx - Cx ) / CDnorm
#             M2y = Ey + focalLength * ( Dy - Cy ) / CDnorm
#             M2z = Ez + focalLength * ( Dz - Cz ) / CDnorm
#             # print( 'CM1 norm ', CM1norm, ' DF1 norm ', DF1norm )
#             # print( 'DF2 norm ', DF2norm )
#             # print( 'C ', Cx, Cy, Cz )
#             # print( 'D ', Dx, Dy, Dz )
#             # print( 'F1 coord ', F1x, F1y, F1z )
#             # print( 'F2 coord ', F2x, F2y, F2z )
#             # print( 'M1 coord ', M1x, M1y, M1z )
#             # print( 'M2 coord ', M2x, M2y, M2z )
#             # print( 'E coord ', Ex, Ey, Ez )
#             # print( 'focalLength ', focalLength )
#             # print( ' major aixs', majorAxis )
#             # print(' minor axis', minorAxis )
#             # print( 'coneAngle ', coneAngle )
#             # print( 'phi ', phi)
#             # print( 'rightSide is ', rightSide)
#
#
#             # for i in range( nz ):
#             #     z = ( i - (nz-1)/2 ) * dz
#             #     for j in range( nx ):
#             #         x = ( j - (nx-1)/2 ) * dx
#             #         distLeft = norm( x - M1x, 0, z - M1z )
#             #         distRight = norm( x - M2x, 0, z - M2z )
#             #         if fabs( fabs( distLeft - distRight ) - 2 * majorAxis ) < 1:
#             #         # if fabs( ( norm( x - M2x, 0, z - M2z ) - norm( x - M1x, 0, z - M1z ) ) - 2 * majorAxis ) < 1:
#             #
#             #             h[i, j] += 1
#
#             if not rightSide:
#                 # print( 'not right side')
#                 for i in range( nz ):
#                     z = ( i - ( nz- 1 ) / 2 ) * dz
#                     for j in range( nx ):
#                         x = ( j - (nx - 1 ) / 2 ) * dx
#                         distLeft = norm( x - M1x, 0, z - M1z )
#                         distRight = norm( x - M2x, 0, z - M2z )
#                         dSum = fabs( distRight - distLeft - 2 * majorAxis )
#                         if distRight > distLeft and  dSum < threshold * sigma:
#                             h[i, j] += 1 / sqrt( 2 * PI ) / sigma * exp( -dSum**2 / 2 / sigma**2 )
#             else:
#                 # print( 'right side ')
#                 for i in range( nz ):
#                     z = ( i - (nz-1)/2 ) * dz
#                     for j in range( nx ):
#                         x = ( j - (nx-1)/2 ) * dx
#                         distLeft = norm( x - M1x, 0, z - M1z )
#                         distRight = norm( x - M2x, 0, z - M2z )
#                         dSum = fabs( distLeft - distRight - 2 * majorAxis )
#                         if distLeft > distRight and dSum < threshold:
#                             h[i, j] += 1 / sqrt( 2 * PI ) / sigma * exp( -dSum**2 / 2 / sigma**2 )
#     end_time = time.time()
#     print( end_time - start_time )
#     # return h
#     return h, sigmaReturn, sigmaReturn2
#     # return h, sigmaReturn
#
#




cpdef np.ndarray conicSection( np.ndarray events, double Cy=0, int nx=201, int nz=401, double threshold=1.0):
    '''
    h = conicSection( np.ndarray events, double Cy=0, int nx=201, int nz=401, double threshold=1.0)

    compute conic section. ellipse and hyperbola only.
    assumed cone back project along -y direction, the conic section is the intercept
    of cone with xz plane specified at y=Cy
    nx, nz define the size of xz plane. in x direction, it goes from -(nx-1)/2
    to (nx-1)/2, and the unit is in mm
    threshold is the parameter to determine if the trajectory passes through a pixel or not
    when a pixel is on the trajectory, its value is increased by 1.

    input:
        events:
            numpy ndarray, 2D events array, similar to that CORE takes as input, each row is a 2-pixel events,
            following the format of [eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz]
            note, it is assumed that the initial energy E0 is eDepA + eDepB
        Cy:
            double, the position of the xz plane.
            note, the methods works the best at Cy=0 plane.
        nx, nz:
            integer, number of pixels in x and z direction
            note, it's assumed the pixel pitch is 1 mm
        threshold:
            double, it is used to determine if a pixel is on the conic section.

    output:
        h:
            numpy ndarray, a 2D array with nz row, nx collumn, a heat map of conic sections

    '''

    cdef:
        double      eDepA=0.0, Ax=0.0, Ay=0.0, Az=0.0, eDepB=0.0, Bx=0.0, By=0.0, Bz=0.0
        double      t=0.0, Cx=0.0, Cz=0.0, Dx=0.0, Dy=0.0, Dz=0.0, Ex=0.0, Ey=0.0, Ez=0.0
        # A1 and C1 are A and C in the transformed axes:
        double      A1x=0.0, A1y=0.0, A1z=0.0, C1x=0.0, C1y=0.0, C1z=0.0
        double      coneAngle=0.0
        double      sinBeta=0.0, cosBeta=0.0, phi=0.0, r=0.0
        # the length:
        double      ACnorm=0.0, ADnorm=0.0, DGnorm=0.0, DFnorm=0.0, DEnorm=0.0
        double      CDnorm=0.0, CM1norm=0.0, DF1norm=0.0, DF2norm=0.0
        double      focalLength=0.0, majorAxis=0.0, minorAxis=0.0
        double      M1x=0.0, M1y=0.0, M1z=0.0, M2x=0.0, M2y=0.0, M2z=0.0
        double      F1x=0.0, F1y=0.0, F1z=0.0, F2x=0.0, F2y=0.0, F2z=0.0
        int         nEvents=0, iEvent=0, i=0, j=0
        # double      dx=(nx-1)/nx, dz =(nz-1)/nz
        double      dx=1.0, dz=1.0
        bint        rightSide=False
        double      distLeft=0.0, distRight=0.0
        float       x=0.0, z=0.0
        int         badCones=0
        np.ndarray[np.int_t, ndim=2]  h=np.zeros( [nz, nx], dtype=np.int )
        # np.ndarray[np.float_t, ndim=2] xv=np.zeros( [nz, nx], dtype=np.float )
        # np.ndarray[np.float_t, ndim=2] zv=np.zeros( [nz, nx], dtype=np.float )
        # np.ndarray[np.float_t, ndim=2]  hTemp=np.zeros( [nz, nx], dtype=np.float )
        # np.ndarray[np.float_t, ndim=2]  hTemp2=np.zeros( [nz, nx], dtype=np.float )

    # h = np.zeros( [nz, nx], dtype=np.int )
    nEvents = len( events )
    # xv, zv = np.meshgrid( np.arange( -( nx - 1 ) / 2.0, ( nx + 1 ) / 2.0, dtype=np.float), np.arange( -( nz - 1) / 2.0, (nz + 1) / 2.0, dtype=np.float ), indexing='xy' )

    start_time = time.time()
    for iEvent in range( nEvents ):
        eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz = events[iEvent]
        coneAngle = scatterAngle( eDepA + eDepB, eDepA )

        # there should be no negative coneAngle, negative coneAngle means
        # for the depsited energy, it's can not be a Compton scatter.
        if coneAngle < 0:
            # print( 'impossible coneAngle detected!', coneAngle)
            badCones += 1
            continue

        # if coneAngle > PI / 2:
        #     coneAngle = PI - coneAngle

        # when By = Ay, the cone axis parallel to xz plane, current method fail
        if fabs( By - Ay ) < ZERO:
            print( iEvent )
            print( 'cone axis parallel to x-z plane, the current method fail!' )
            continue

        # find coordinates for point C and D.
        # C is the intersection of line BA with xz plane
        # D is the projection of point A on xz plane
        Dx = Ax
        Dy = Cy
        Dz = Az
        t = ( By - Cy ) / ( By - Ay )
        Cx = Bx + t * ( Ax - Bx )
        Cz = Bz + t * ( Az - Bz )

        ADnorm = norm( Dx - Ax, Dy - Ay, Dz - Az )
        CDnorm = norm( Dx - Cx, Dy - Cy, Dz - Cz)

        # when C and D overlap, then AB is perpendicular to xz plane
        # the intersection is a circle:
        if CDnorm < ZERO:
            if coneAngle < PI / 2:
                r = ( Ay - Cy ) * tan( coneAngle )
            else:
                r = ( Ay - Cy ) * tan( PI - coneAngle )

            # for i in range( nz ):
            for i in prange( nz, nogil=True ):
                z = ( i - ( nz - 1 ) / 2 ) * dz
                for j in range( nx ):
                    x = ( j - ( nx - 1 ) / 2 ) * dx
                    distLeft = sqrt( (x - Cx)**2 + (z - Cz)**2 )
                    # if fabs( norm( x -Cx, 0, z - Cz ) - r ) < threshold:
                    if fabs( distLeft - r ) < threshold:
                        h[i, j] += 1
            # hTemp = np.sqrt( (xv - Cx)**2 + (zv - Cz)**2 )
            # h[ np.fabs( hTemp - r ) < threshold ] += 1
            continue

        # C and D not overlapping, either ellipse or hyperbola:
        sinBeta = ( Dx - Cx ) / CDnorm
        cosBeta = ( Dz - Cz ) / CDnorm

        if coneAngle < PI/2:
            phi = atan( CDnorm / ADnorm ) - coneAngle
        else:
            phi = atan( CDnorm / ADnorm ) - ( PI - coneAngle )

        # when the intersection is an ellipse:
        if ( ( By > Ay and coneAngle < PI / 2 and 2 * coneAngle + phi < PI / 2 )
             or
             ( Ay > By and coneAngle > PI / 2 and 2 * ( PI - coneAngle ) + phi < PI / 2 )
            ):

            # print( 'in ellipse!')
            if Ay > By and coneAngle > PI / 2:
                coneAngle = PI - coneAngle

            DGnorm = ADnorm * tan( 2 * coneAngle + phi )
            DFnorm = ADnorm * tan( phi )
            majorAxis = ( DGnorm - DFnorm ) / 2.0
            DEnorm = DFnorm + majorAxis
            Ex = Dx + DEnorm * ( Cx - Dx ) / CDnorm
            Ey = Dy + DEnorm * ( Cy - Dy ) / CDnorm
            Ez = Dz + DEnorm * ( Cz - Dz ) / CDnorm

            A1x, A1y, A1z = matrixVectorDot( cosBeta, 0, -sinBeta, 0, 1, 0, sinBeta, 0, cosBeta, Ax - Ex, Ay - Ey, Az - Ez )
            C1x, C1y, C1z = matrixVectorDot( cosBeta, 0, -sinBeta, 0, 1, 0, sinBeta, 0, cosBeta, Cx - Ex, Cy - Ey, Cz - Ez )

            ACnorm = norm( A1x - C1x, A1y - C1y, A1z - C1z )
            minorAxis = sqrt( ( A1y * A1y + A1z * ( A1z - C1z ) )**2 / ( ACnorm * cos( coneAngle ) )**2 - A1y * A1y - A1z * A1z )
            focalLength = sqrt( majorAxis * majorAxis - minorAxis * minorAxis )

            M1x = Ex + focalLength * ( Cx - Dx ) / CDnorm
            M1y = Ey + focalLength * ( Cy - Dy ) / CDnorm
            M1z = Ez + focalLength * ( Cz - Dz ) / CDnorm
            M2x = Ex - focalLength * ( Cx - Dx ) / CDnorm
            M2y = Ey - focalLength * ( Cy - Dy ) / CDnorm
            M2z = Ez - focalLength * ( Cz - Dz ) / CDnorm
            # print( 'E coord ', Ex, Ey, Ez )
            # print( 'M1 coord ', M1x, M1y, M1z )
            # print( 'M2 coord ', M2x, M2y, M2z )
            # print( 'majorAxis ', majorAxis )
            # print( 'minorAxis ', minorAxis )
            # print( 'focalLength ', focalLength )


            # for i in range( nz ):
            for i in prange( nz, nogil=True ):
                z = ( i - ( nz - 1 ) / 2 ) * dz
                for j in range( nx ):
                    x = ( j - ( nx - 1 ) / 2 ) * dx
                    distLeft = sqrt( ( x - M1x)**2 + (z - M1z)**2 )
                    distRight = sqrt( (x - M2x)**2 + (z - M2z)**2 )
                    # if fabs( norm( x - M1x, 0, z - M1z ) + norm( x - M2x, 0, z - M2z ) - 2 * majorAxis ) < threshold:
                    if fabs( distLeft + distRight - 2 * majorAxis ) < threshold:
                        h[i, j] += 1
            # hTemp = np.sqrt( (xv-M1x)**2 + (zv-M1z)**2 ) + np.sqrt( ( xv-M2x)**2 + (zv-M2z)**2 )
            # h[ np.fabs( hTemp - 2*majorAxis ) < threshold] += 1

        # in case the trajectory is a proballa, it is  not defined
        elif ( ( By > Ay  and coneAngle < PI / 2 and fabs( 2 * coneAngle + phi - PI / 2 ) < ZERO )
               or
               ( Ay > By  and coneAngle > PI / 2 and fabs( 2 * (PI - coneAngle) + phi - PI / 2 ) < ZERO )
             ):
            print( 'prabolla needed ')
            continue

        # intersection is hyperbola
        elif ( ( By > Ay and coneAngle < PI / 2 and 2 * coneAngle + phi > PI / 2)
               or
               ( By > Ay and coneAngle > PI / 2 )
               or
               ( Ay > By and coneAngle > PI / 2 and 2 * ( PI - coneAngle ) + phi > PI /2 )
               or
               ( Ay > By and coneAngle < PI / 2 )
             ):

            # print( 'in hyperbola')
            if By > Ay and coneAngle < PI / 2 and 2 * coneAngle + phi > PI / 2:
                rightSide = False
            elif By > Ay and coneAngle > PI / 2:
                coneAngle = PI - coneAngle
                rightSide = True
            elif Ay > By and coneAngle > PI / 2:
                # need to make sure that this is a hyperbola
                if 2 * ( PI - coneAngle ) + phi > PI /2:
                    # print( 'Ay is larger and sure of hyperbola')
                    coneAngle = PI - coneAngle
                    rightSide = False
                # elif 2 * ( PI - coneAngle ) + phi > PI:
                #     coneAngle = PI - coneAngle
                #     rightSide = True
            elif Ay > By and coneAngle < PI / 2:
                rightSide = True

            # in this case, there will be no right side hyperbola as it won't
            # intercept with xz plane
            if rightSide and PI - 2 * coneAngle - phi > PI / 2:
                continue

            ACnorm = norm( Cx - Ax, Cy - Ay, Cz - Az )
            r = ADnorm * ACnorm / ( ACnorm + ADnorm/sin(coneAngle) )
            if r < 0:
                print( 'red alert: r is negative' )
                continue

            CM1norm = r / ADnorm * CDnorm
            DF1norm = ADnorm * tan( phi )

            M1x = Cx + CM1norm * ( Dx - Cx ) / CDnorm
            M1y = Cy + CM1norm * ( Dy - Cy ) / CDnorm
            M1z = Cz + CM1norm * ( Dz - Cz ) / CDnorm

            F1x = Dx + DF1norm * ( Cx - Dx ) / CDnorm
            F1y = Dy + DF1norm * ( Cy - Dy ) / CDnorm
            F1z = Dz + DF1norm * ( Cz - Dz ) / CDnorm

            DF2norm = ADnorm * tan( PI - 2 * coneAngle - phi )
            F2x = Dx + DF2norm * ( Dx - Cx ) / CDnorm
            F2y = Dy + DF2norm * ( Dy - Cy ) / CDnorm
            F2z = Dz + DF2norm * ( Dz - Cz ) / CDnorm

            Ex = ( F1x + F2x ) / 2.0
            Ey = ( F1y + F2y ) / 2.0
            Ez = ( F1z + F2z ) / 2.0

            focalLength = norm( Ex - M1x, Ey - M1y, Ez - M1z )
            majorAxis = norm( Ex - F1x, Ey - F1y, Ez - F1z )
            minorAxis = sqrt( focalLength * focalLength - majorAxis * majorAxis )

            M2x = Ex + focalLength * ( Dx - Cx ) / CDnorm
            M2y = Ey + focalLength * ( Dy - Cy ) / CDnorm
            M2z = Ez + focalLength * ( Dz - Cz ) / CDnorm
            # print( 'CM1 norm ', CM1norm, ' DF1 norm ', DF1norm )
            # print( 'DF2 norm ', DF2norm )
            # print( 'C ', Cx, Cy, Cz )
            # print( 'D ', Dx, Dy, Dz )
            # print( 'F1 coord ', F1x, F1y, F1z )
            # print( 'F2 coord ', F2x, F2y, F2z )
            # print( 'M1 coord ', M1x, M1y, M1z )
            # print( 'M2 coord ', M2x, M2y, M2z )
            # print( 'E coord ', Ex, Ey, Ez )
            # print( 'focalLength ', focalLength )
            # print( ' major aixs', majorAxis )
            # print(' minor axis', minorAxis )
            # print( 'coneAngle ', coneAngle )
            # print( 'phi ', phi)
            # print( 'rightSide is ', rightSide)


            # for i in range( nz ):
            #     z = ( i - (nz-1)/2 ) * dz
            #     for j in range( nx ):
            #         x = ( j - (nx-1)/2 ) * dx
            #         distLeft = norm( x - M1x, 0, z - M1z )
            #         distRight = norm( x - M2x, 0, z - M2z )
            #         if fabs( fabs( distLeft - distRight ) - 2 * majorAxis ) < 1:
            #         # if fabs( ( norm( x - M2x, 0, z - M2z ) - norm( x - M1x, 0, z - M1z ) ) - 2 * majorAxis ) < 1:
            #
            #             h[i, j] += 1

            # hTemp = np.sqrt( ( xv - M1x )**2 + ( zv - M1z )**2 )
            # hTemp2 = np.sqrt( ( xv - M2x )**2 + ( zv - M2z )**2 )
            # if not rightSide:
            #     h[ np.logical_and( hTemp2 > hTemp, np.fabs( hTemp2 - hTemp - 2 * majorAxis ) < threshold ) ] += 1
            # else:
            #     h[ np.logical_and( hTemp > hTemp2, np.fabs( hTemp - hTemp2 - 2 * majorAxis ) < threshold ) ] += 1
            if not rightSide:
                # print( 'not right side')
                # for i in range( nz ):
                for i in prange( nz, nogil=True ):
                    z = ( i - ( nz- 1 ) / 2 ) * dz
                    for j in range( nx ):
                        x = ( j - (nx - 1 ) / 2 ) * dx
                        distLeft = sqrt( ( x - M1x )**2 + ( z - M1z )**2 )
                        distRight = sqrt( ( x - M2x )**2 + ( z - M2z )**2 )
                        # distLeft = norm( x - M1x, 0, z - M1z )
                        # distRight = norm( x - M2x, 0, z - M2z )
                        if distRight > distLeft and fabs( distRight - distLeft - 2 * majorAxis ) < threshold:
                            h[i, j] += 1
            else:
                # print( 'right side ')
                for i in prange( nz, nogil=True ):
                    z = ( i - (nz-1)/2 ) * dz
                    for j in range( nx ):
                        x = ( j - (nx-1)/2 ) * dx
                        distLeft = sqrt( (x - M1x)**2 + (z - M1z)**2 )
                        distRight = sqrt( (x - M2x)**2 + (z - M2z)**2 )
                        # distLeft = norm( x - M1x, 0, z - M1z )
                        # distRight = norm( x - M2x, 0, z - M2z )
                        if distLeft > distRight and fabs( distLeft - distRight - 2 * majorAxis ) < threshold:
                            h[i, j] += 1
    end_time = time.time()
    print( 'total number of cones:', nEvents )
    print( 'number of bad cones (impossible cone angle):', badCones )
    print( 'computing time:', end_time - start_time )
    return h



cpdef np.ndarray conicSectionMM( np.ndarray events, double Cy=0, int nx=201, int nz=401, double threshold=1.0, double res=0.1):
    '''
    h = conicSectionMM( np.ndarray events, double Cy=0, int nx=201, int nz=401, double threshold=1.0, double res=0.1)

    compute conic section. ellipse and hyperbola only.
    assumed cone back project along -y direction, the conic section is the intercept
    of cone with xz plane specified at y=Cy
    nx, nz define the size of xz plane. in x direction, it goes from -(nx-1)/2
    to (nx-1)/2, and the unit is in mm
    threshold is the parameter to determine if the trajectory passes through a pixel or not
    when a pixel is on the trajectory, its value is increased by 1.

    input:
        events:
            numpy ndarray, 2D events array, similar to that CORE takes as input, each row is a 2-pixel events,
            following the format of [eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz]
            note, it is assumed that the initial energy E0 is eDepA + eDepB
        Cy:
            double, the position of the xz plane.
            note, the methods works the best at Cy=0 plane.
        nx, nz:
            integer, number of pixels in x and z direction
            note, it's assumed the pixel pitch is 1 mm
        threshold:
            double, it is used to determine if a pixel is on the conic section.
        res:
            pixel pitch, in [mm]

    output:
        h:
            numpy ndarray, a 2D array with nz row, nx collumn, a heat map of conic sections

    '''

    cdef:
        double      eDepA=0.0, Ax=0.0, Ay=0.0, Az=0.0, eDepB=0.0, Bx=0.0, By=0.0, Bz=0.0
        double      t=0.0, Cx=0.0, Cz=0.0, Dx=0.0, Dy=0.0, Dz=0.0, Ex=0.0, Ey=0.0, Ez=0.0
        # A1 and C1 are A and C in the transformed axes:
        double      A1x=0.0, A1y=0.0, A1z=0.0, C1x=0.0, C1y=0.0, C1z=0.0
        double      coneAngle=0.0
        double      sinBeta=0.0, cosBeta=0.0, phi=0.0, r=0.0
        # the length:
        double      ACnorm=0.0, ADnorm=0.0, DGnorm=0.0, DFnorm=0.0, DEnorm=0.0
        double      CDnorm=0.0, CM1norm=0.0, DF1norm=0.0, DF2norm=0.0
        double      focalLength=0.0, majorAxis=0.0, minorAxis=0.0
        double      M1x=0.0, M1y=0.0, M1z=0.0, M2x=0.0, M2y=0.0, M2z=0.0
        double      F1x=0.0, F1y=0.0, F1z=0.0, F2x=0.0, F2y=0.0, F2z=0.0
        int         nEvents=0, iEvent=0, i=0, j=0
        # double      dx=(nx-1)/nx, dz =(nz-1)/nz
        double      dx=1.0, dz=1.0
        bint        rightSide=False
        double      distLeft=0.0, distRight=0.0
        float       x=0.0, z=0.0
        int         badCones=0
        np.ndarray[np.int_t, ndim=2]  h=np.zeros( [nz, nx], dtype=np.int )
        # np.ndarray[np.float_t, ndim=2] xv=np.zeros( [nz, nx], dtype=np.float )
        # np.ndarray[np.float_t, ndim=2] zv=np.zeros( [nz, nx], dtype=np.float )
        # np.ndarray[np.float_t, ndim=2]  hTemp=np.zeros( [nz, nx], dtype=np.float )
        # np.ndarray[np.float_t, ndim=2]  hTemp2=np.zeros( [nz, nx], dtype=np.float )

    #    if mm:
    dx = res
    dz = res
    # h = np.zeros( [nz, nx], dtype=np.int )
    nEvents = len( events )
    # xv, zv = np.meshgrid( np.arange( -( nx - 1 ) / 2.0, ( nx + 1 ) / 2.0, dtype=np.float), np.arange( -( nz - 1) / 2.0, (nz + 1) / 2.0, dtype=np.float ), indexing='xy' )

    start_time = time.time()
    for iEvent in range( nEvents ):
        eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz = events[iEvent]
        coneAngle = scatterAngle( eDepA + eDepB, eDepA )

        # there should be no negative coneAngle, negative coneAngle means
        # for the depsited energy, it's can not be a Compton scatter.
        if coneAngle < 0:
            # print( 'impossible coneAngle detected!', coneAngle)
            badCones += 1
            continue

        # if coneAngle > PI / 2:
        #     coneAngle = PI - coneAngle

        # when By = Ay, the cone axis parallel to xz plane, current method fail
        if fabs( By - Ay ) < ZERO:
            print( iEvent )
            print( 'cone axis parallel to x-z plane, the current method fail!' )
            continue

        # find coordinates for point C and D.
        # C is the intersection of line BA with xz plane
        # D is the projection of point A on xz plane
        Dx = Ax
        Dy = Cy
        Dz = Az
        t = ( By - Cy ) / ( By - Ay )
        Cx = Bx + t * ( Ax - Bx )
        Cz = Bz + t * ( Az - Bz )

        ADnorm = norm( Dx - Ax, Dy - Ay, Dz - Az )
        CDnorm = norm( Dx - Cx, Dy - Cy, Dz - Cz)

        # when C and D overlap, then AB is perpendicular to xz plane
        # the intersection is a circle:
        if CDnorm < ZERO:
            if coneAngle < PI / 2:
                r = ( Ay - Cy ) * tan( coneAngle )
            else:
                r = ( Ay - Cy ) * tan( PI - coneAngle )

            # for i in range( nz ):
            for i in prange( nz, nogil=True ):
                z = ( i - ( nz - 1 ) / 2 ) * dz
                for j in range( nx ):
                    x = ( j - ( nx - 1 ) / 2 ) * dx
                    distLeft = sqrt( (x - Cx)**2 + (z - Cz)**2 )
                    # if fabs( norm( x -Cx, 0, z - Cz ) - r ) < threshold:
                    if fabs( distLeft - r ) < threshold:
                        h[i, j] += 1
            # hTemp = np.sqrt( (xv - Cx)**2 + (zv - Cz)**2 )
            # h[ np.fabs( hTemp - r ) < threshold ] += 1
            continue

        # C and D not overlapping, either ellipse or hyperbola:
        sinBeta = ( Dx - Cx ) / CDnorm
        cosBeta = ( Dz - Cz ) / CDnorm

        if coneAngle < PI/2:
            phi = atan( CDnorm / ADnorm ) - coneAngle
        else:
            phi = atan( CDnorm / ADnorm ) - ( PI - coneAngle )

        # when the intersection is an ellipse:
        if ( ( By > Ay and coneAngle < PI / 2 and 2 * coneAngle + phi < PI / 2 )
             or
             ( Ay > By and coneAngle > PI / 2 and 2 * ( PI - coneAngle ) + phi < PI / 2 )
            ):

            # print( 'in ellipse!')
            if Ay > By and coneAngle > PI / 2:
                coneAngle = PI - coneAngle

            DGnorm = ADnorm * tan( 2 * coneAngle + phi )
            DFnorm = ADnorm * tan( phi )
            majorAxis = ( DGnorm - DFnorm ) / 2.0
            DEnorm = DFnorm + majorAxis
            Ex = Dx + DEnorm * ( Cx - Dx ) / CDnorm
            Ey = Dy + DEnorm * ( Cy - Dy ) / CDnorm
            Ez = Dz + DEnorm * ( Cz - Dz ) / CDnorm

            A1x, A1y, A1z = matrixVectorDot( cosBeta, 0, -sinBeta, 0, 1, 0, sinBeta, 0, cosBeta, Ax - Ex, Ay - Ey, Az - Ez )
            C1x, C1y, C1z = matrixVectorDot( cosBeta, 0, -sinBeta, 0, 1, 0, sinBeta, 0, cosBeta, Cx - Ex, Cy - Ey, Cz - Ez )

            ACnorm = norm( A1x - C1x, A1y - C1y, A1z - C1z )
            minorAxis = sqrt( ( A1y * A1y + A1z * ( A1z - C1z ) )**2 / ( ACnorm * cos( coneAngle ) )**2 - A1y * A1y - A1z * A1z )
            focalLength = sqrt( majorAxis * majorAxis - minorAxis * minorAxis )

            M1x = Ex + focalLength * ( Cx - Dx ) / CDnorm
            M1y = Ey + focalLength * ( Cy - Dy ) / CDnorm
            M1z = Ez + focalLength * ( Cz - Dz ) / CDnorm
            M2x = Ex - focalLength * ( Cx - Dx ) / CDnorm
            M2y = Ey - focalLength * ( Cy - Dy ) / CDnorm
            M2z = Ez - focalLength * ( Cz - Dz ) / CDnorm
            # print( 'E coord ', Ex, Ey, Ez )
            # print( 'M1 coord ', M1x, M1y, M1z )
            # print( 'M2 coord ', M2x, M2y, M2z )
            # print( 'majorAxis ', majorAxis )
            # print( 'minorAxis ', minorAxis )
            # print( 'focalLength ', focalLength )


            # for i in range( nz ):
            for i in prange( nz, nogil=True ):
                z = ( i - ( nz - 1 ) / 2 ) * dz
                for j in range( nx ):
                    x = ( j - ( nx - 1 ) / 2 ) * dx
                    distLeft = sqrt( ( x - M1x)**2 + (z - M1z)**2 )
                    distRight = sqrt( (x - M2x)**2 + (z - M2z)**2 )
                    # if fabs( norm( x - M1x, 0, z - M1z ) + norm( x - M2x, 0, z - M2z ) - 2 * majorAxis ) < threshold:
                    if fabs( distLeft + distRight - 2 * majorAxis ) < threshold:
                        h[i, j] += 1
            # hTemp = np.sqrt( (xv-M1x)**2 + (zv-M1z)**2 ) + np.sqrt( ( xv-M2x)**2 + (zv-M2z)**2 )
            # h[ np.fabs( hTemp - 2*majorAxis ) < threshold] += 1

        # in case the trajectory is a proballa, it is  not defined
        elif ( ( By > Ay  and coneAngle < PI / 2 and fabs( 2 * coneAngle + phi - PI / 2 ) < ZERO )
                or
                ( Ay > By  and coneAngle > PI / 2 and fabs( 2 * (PI - coneAngle) + phi - PI / 2 ) < ZERO ) ):
            print( 'parabola needed ')
            continue

        # intersection is hyperbola
        elif ( ( By > Ay and coneAngle < PI / 2 and 2 * coneAngle + phi > PI / 2)
                or
                ( By > Ay and coneAngle > PI / 2 )
                or
                ( Ay > By and coneAngle > PI / 2 and 2 * ( PI - coneAngle ) + phi > PI / 2 )
                or
                ( Ay > By and coneAngle < PI / 2 ) ):

            # print( 'in hyperbola')
            if By > Ay and coneAngle < PI / 2 and 2 * coneAngle + phi > PI / 2:
                rightSide = False
            elif By > Ay and coneAngle > PI / 2:
                coneAngle = PI - coneAngle
                rightSide = True
            elif Ay > By and coneAngle > PI / 2:
                # need to make sure that this is a hyperbola
                if 2 * ( PI - coneAngle ) + phi > PI /2:
                    # print( 'Ay is larger and sure of hyperbola')
                    coneAngle = PI - coneAngle
                    rightSide = False
                # elif 2 * ( PI - coneAngle ) + phi > PI:
                #     coneAngle = PI - coneAngle
                #     rightSide = True
            elif Ay > By and coneAngle < PI / 2:
                rightSide = True

            # in this case, there will be no right side hyperbola as it won't
            # intercept with xz plane
            if rightSide and PI - 2 * coneAngle - phi > PI / 2:
                continue

            ACnorm = norm( Cx - Ax, Cy - Ay, Cz - Az )
            r = ADnorm * ACnorm / ( ACnorm + ADnorm/sin(coneAngle) )
            if r < 0:
                print( 'red alert: r is negative' )
                continue

            CM1norm = r / ADnorm * CDnorm
            DF1norm = ADnorm * tan( phi )

            M1x = Cx + CM1norm * ( Dx - Cx ) / CDnorm
            M1y = Cy + CM1norm * ( Dy - Cy ) / CDnorm
            M1z = Cz + CM1norm * ( Dz - Cz ) / CDnorm

            F1x = Dx + DF1norm * ( Cx - Dx ) / CDnorm
            F1y = Dy + DF1norm * ( Cy - Dy ) / CDnorm
            F1z = Dz + DF1norm * ( Cz - Dz ) / CDnorm

            DF2norm = ADnorm * tan( PI - 2 * coneAngle - phi )
            F2x = Dx + DF2norm * ( Dx - Cx ) / CDnorm
            F2y = Dy + DF2norm * ( Dy - Cy ) / CDnorm
            F2z = Dz + DF2norm * ( Dz - Cz ) / CDnorm

            Ex = ( F1x + F2x ) / 2.0
            Ey = ( F1y + F2y ) / 2.0
            Ez = ( F1z + F2z ) / 2.0

            focalLength = norm( Ex - M1x, Ey - M1y, Ez - M1z )
            majorAxis = norm( Ex - F1x, Ey - F1y, Ez - F1z )
            minorAxis = sqrt( focalLength * focalLength - majorAxis * majorAxis )

            M2x = Ex + focalLength * ( Dx - Cx ) / CDnorm
            M2y = Ey + focalLength * ( Dy - Cy ) / CDnorm
            M2z = Ez + focalLength * ( Dz - Cz ) / CDnorm
            # print( 'CM1 norm ', CM1norm, ' DF1 norm ', DF1norm )
            # print( 'DF2 norm ', DF2norm )
            # print( 'C ', Cx, Cy, Cz )
            # print( 'D ', Dx, Dy, Dz )
            # print( 'F1 coord ', F1x, F1y, F1z )
            # print( 'F2 coord ', F2x, F2y, F2z )
            # print( 'M1 coord ', M1x, M1y, M1z )
            # print( 'M2 coord ', M2x, M2y, M2z )
            # print( 'E coord ', Ex, Ey, Ez )
            # print( 'focalLength ', focalLength )
            # print( ' major aixs', majorAxis )
            # print(' minor axis', minorAxis )
            # print( 'coneAngle ', coneAngle )
            # print( 'phi ', phi)
            # print( 'rightSide is ', rightSide)


            # for i in range( nz ):
            #     z = ( i - (nz-1)/2 ) * dz
            #     for j in range( nx ):
            #         x = ( j - (nx-1)/2 ) * dx
            #         distLeft = norm( x - M1x, 0, z - M1z )
            #         distRight = norm( x - M2x, 0, z - M2z )
            #         if fabs( fabs( distLeft - distRight ) - 2 * majorAxis ) < 1:
            #         # if fabs( ( norm( x - M2x, 0, z - M2z ) - norm( x - M1x, 0, z - M1z ) ) - 2 * majorAxis ) < 1:
            #
            #             h[i, j] += 1

            # hTemp = np.sqrt( ( xv - M1x )**2 + ( zv - M1z )**2 )
            # hTemp2 = np.sqrt( ( xv - M2x )**2 + ( zv - M2z )**2 )
            # if not rightSide:
            #     h[ np.logical_and( hTemp2 > hTemp, np.fabs( hTemp2 - hTemp - 2 * majorAxis ) < threshold ) ] += 1
            # else:
            #     h[ np.logical_and( hTemp > hTemp2, np.fabs( hTemp - hTemp2 - 2 * majorAxis ) < threshold ) ] += 1
            if not rightSide:
                # print( 'not right side')
                # for i in range( nz ):
                for i in prange( nz, nogil=True ):
                    z = ( i - ( nz- 1 ) / 2 ) * dz
                    for j in range( nx ):
                        x = ( j - (nx - 1 ) / 2 ) * dx
                        distLeft = sqrt( ( x - M1x )**2 + ( z - M1z )**2 )
                        distRight = sqrt( ( x - M2x )**2 + ( z - M2z )**2 )
                        # distLeft = norm( x - M1x, 0, z - M1z )
                        # distRight = norm( x - M2x, 0, z - M2z )
                        if distRight > distLeft and fabs( distRight - distLeft - 2 * majorAxis ) < threshold:
                            h[i, j] += 1
            else:
                # print( 'right side ')
                for i in prange( nz, nogil=True ):
                    z = ( i - (nz-1)/2 ) * dz
                    for j in range( nx ):
                        x = ( j - (nx-1)/2 ) * dx
                        distLeft = sqrt( (x - M1x)**2 + (z - M1z)**2 )
                        distRight = sqrt( (x - M2x)**2 + (z - M2z)**2 )
                        # distLeft = norm( x - M1x, 0, z - M1z )
                        # distRight = norm( x - M2x, 0, z - M2z )
                        if distLeft > distRight and fabs( distLeft - distRight - 2 * majorAxis ) < threshold:
                            h[i, j] += 1
    end_time = time.time()
    print( 'total number of cones:', nEvents )
    print( 'number of bad cones (impossible cone angle):', badCones )
    print( 'computing time:', end_time - start_time )
    return h





cpdef zAxisDCA(   double Ax, double Ay, double Az,
                        double Bx, double By, double Bz,
                        double coneAngle, double Cy=0.0,
                        double boundaryX1=-100.0, double boundaryX2=100.0,
                        double boundaryY1=-150.0, double boundaryY2=150.0,
                        double boundaryZ1=-200.0, double boundaryZ2=200.0):
    '''
    DCA, coord = zAxisDCA(  double Ax, double Ay, double Az,
                            double Bx, double By, double Bz,
                            double coneAngle, double Cy,
                            double boundaryX1=-100.0, double boundaryX2=100.0,
                            double boundaryY1=-150.0, double boundaryY2=150.0,
                            double boundaryZ1=-200.0, double boundaryZ2=200.0)


    compute DCA to z-axis

    input:
        Ax, Ay Az, Bx, By, Bz:
            double, event coordinates for the first ( A ) and the second ( B ) events;
        coneAngle:
            double, cone angle
        Cy:
            double, the location of xz plane, defined by the y coordinates of C; C is the
            extension of AB onto the xz plane
        boundaryX1, boundaryX2, boundaryY1, boundaryY2, boundaryZ1, boundaryZ2:
            double, the boundary of the recon volume, only those minimum distace
            inside the volume will be allowed to return.

    output:
        DCA:
            double, the DCA
        coord:
            the coordinates of the two point ( one on cone surface, the other on z-axis)
            that give rise to the returned DCA. it's a 2D array, the first row is the
            point on the cone, the second row is the point on z-axis.
            note if DCA is 0.0, when the z-axis is intersecting with the conic section,
            the coord may return two pair of points, each pair is a intersection point
            on the cone surface and z-axis
    '''

    cdef:
        # int     nIntercept, nTouch, nAway

        # C is where the cone axis intercept with xz plane
        # D is the projection of A on xz plane
        double   t=0.0, Cx=0.0, Cz=0.0, Dx=0.0, Dy=0.0, Dz=0.0
        double   sinBeta=0.0, cosBeta=0.0, phi=0.0
        double   ADnorm=0.0, CDnorm=0.0,
        double   DCA=1000.0, temp=0.0
        np.ndarray coord=np.array( [ ] )

    # when By = Ay, the cone axis parallel to xz plane, current method fail
    if fabs( By - Ay ) < ZERO:
        # print( 'cone axis does not intersecpt with xz plane' )
        return PARALLEL, np.array( [ ] )

    # when coneAngle is pi/2, then the 'cone' becomes a plane that is perpendicular
    # to AB.
    if fabs( coneAngle / PI - 0.5 ) < ZERO:
        # when AB is perpendicular to z-axis, then the plane will be parallel to
        # z-axis:
        if fabs( Az - Bz ) < ZERO:
            return PARALLEL, np.array( [ ] )
        else:
            temp = Az + ( Ax * ( Bx - Ax ) + Ay * ( By - Ay ) ) / ( Bz - Az )
            if temp > boundaryZ1 and temp < boundaryZ2:
                return 0.0, np.array( [ [0.0, 0.0, temp], [0.0, 0.0, temp] ] )
            else:
                return OUTOFBOUNDARY, np.array( [ ] )


    # find the coordinates for points C and D
    # C is the intersecpt of AB with the xz plane
    # D is the projection of A onto xz plane
    Dx = Ax
    Dy = Cy
    Dz = Az
    t = ( By - Cy ) / ( By - Ay )
    Cx = Bx + t * ( Ax - Bx )
    Cz = Bz + t * ( Az - Bz )

    ADnorm = norm( Ax - Dx, Ay - Dy, Az - Dz )
    CDnorm = norm( Cx - Dx, Cy - Dy, Cz - Dz )

    # a special case is when point C and D overlay, which means AB is perpendicular
    # to the xz plane, the conic section is a circle, and CD length is 0.0, which
    # will cause devide by zero error later:
    if CDnorm < ZERO:
        DCA, coord = zAxisCircleDCA(    Ay, By, Cx, Cy, Cz, coneAngle,
                                        boundaryX1, boundaryX2,
                                        boundaryY1, boundaryY2,
                                        boundaryZ1, boundaryZ2 )
        return DCA, coord

    # if C and D do not overlap, i.e. vector AB is not perpendicular to xz plane
    if By - Ay > 0:
        if coneAngle < PI / 2:
            # phi is the angle between AF and AD
            phi = atan( CDnorm / ADnorm ) - coneAngle

            # the intersection is ellipse
            if 2 * coneAngle + phi < PI / 2:
                # print( 'ellipse 1')
                DCA, coord = zAxisEllipseDCA(   Ax, Ay, Az, Bx, By, Bz,
                                                Cx, Cy, Cz, Dx, Dy, Dz,
                                                coneAngle, phi,
                                                boundaryX1, boundaryX2,
                                                boundaryY1, boundaryY2,
                                                boundaryZ1, boundaryZ2 )
                return DCA, coord
            # intersection is hyperbola
            elif 2 * coneAngle + phi  > PI / 2:
                # return OUTOFBOUNDARY, np.array( [ ] )

                # print( 'hyperbola 1')
                DCA, coord = zAxisHyperbolaDCA( Ax, Ay, Az, Bx, By, Bz,
                                                Cx, Cy, Cz, Dx, Dy, Dz,
                                                coneAngle, phi,
                                                boundaryX1, boundaryX2,
                                                boundaryY1, boundaryY2,
                                                boundaryZ1, boundaryZ2,
                                                rightSide=False )
                return DCA, coord
            # intersection is parabola
            else:
                print( 'need parabolla' )
                return OUTOFBOUNDARY, np.array( [ ] )

        # when By > Ay and coneAngle > PI / 2:
        else:
            # return OUTOFBOUNDARY, np.array( [ ] )

            # phi is the angle between AF and AD
            phi = atan( CDnorm / ADnorm ) - (PI - coneAngle)
            # print( 'hyperbola 2')
            DCA, coord = zAxisHyperbolaDCA( Ax, Ay, Az, Bx, By, Bz,
                                            Cx, Cy, Cz, Dx, Dy, Dz,
                                            PI - coneAngle, phi,
                                            boundaryX1, boundaryX2,
                                            boundaryY1, boundaryY2,
                                            boundaryZ1, boundaryZ2,
                                            rightSide=True)
            return DCA, coord

    # when By is less than Ay:
    else:
        if coneAngle < PI / 2:
            # return OUTOFBOUNDARY, np.array( [ ] )

            phi = atan( CDnorm / ADnorm ) - coneAngle
            # print( 'hyperbola 3')
            DCA, coord = zAxisHyperbolaDCA( Ax, Ay, Az, Bx, By, Bz,
                                            Cx, Cy, Cz, Dx, Dy, Dz,
                                            coneAngle, phi,
                                            boundaryX1, boundaryX2,
                                            boundaryY1, boundaryY2,
                                            boundaryZ1, boundaryZ2,
                                            rightSide=True)
            return DCA, coord
        else:
            phi = atan( CDnorm / ADnorm ) - ( PI - coneAngle )

            # the intersection is ellipse
            # if 2 * (PI - coneAngle) + phi < PI / 2:
            if PI / 2 - 2 * (PI - coneAngle) - phi > 0:
                # print( 'ellipse 2')
                DCA, coord = zAxisEllipseDCA(   Ax, Ay, Az, Bx, By, Bz,
                                                Cx, Cy, Cz, Dx, Dy, Dz,
                                                PI - coneAngle, phi,
                                                boundaryX1, boundaryX2,
                                                boundaryY1, boundaryY2,
                                                boundaryZ1, boundaryZ2 )
                # print( PI - coneAngle )
                return DCA, coord
            # intersection is parabola
            elif 2 * (PI - coneAngle) + phi -PI / 2 > 0:
                # return OUTOFBOUNDARY, np.array( [ ] )

                # print( 'hyperbola 4')
                DCA, coord = zAxisHyperbolaDCA( Ax, Ay, Az, Bx, By, Bz,
                                                Cx, Cy, Cz, Dx, Dy, Dz,
                                                PI - coneAngle, phi,
                                                boundaryX1, boundaryX2,
                                                boundaryY1, boundaryY2,
                                                boundaryZ1, boundaryZ2,
                                                rightSide=False )
                return DCA, coord
            else:
                print( 'need parabolla' )
                return OUTOFBOUNDARY, np.array( [ ] )
            # intersection is hyperbola



cpdef zAxisCircleDCA(   double Ay, double By,
                        double Cx, double Cy, double Cz,
                        double coneAngle,
                        double boundaryX1=-100.0, double boundaryX2=100.0,
                        double boundaryY1=-150.0, double boundaryY2=150.0,
                        double boundaryZ1=-200.0, double boundaryZ2=200.0):
    '''
    DCA, coord = zAxisCircleDCA(    double Ay, double By,
                                    double Cx, double Cy, double Cz,
                                    double coneAngle,
                                    double boundaryX1=-100.0, double boundaryX2=100.0,
                                    double boundaryY1=-150.0, double boundaryY2=150.0,
                                    double boundaryZ1=-200.0, double boundaryZ2=200.0)


    compute DCA when the cone axis is perpendicular to xz plane, where the conic section
    is a circle.

    input:
        Ay, By:
            double, the y coordinates of the first  and the second events;
        Cx, Cy, Cz:
            double, point C is the where line AB intercepts with xz plane
            note the location of the xz plane is defined by Cy
        coneAngle:
            double, cone angle
    output:
        DCA:
            double, the DCA from the z-axis to cone surface;
        coord:
            numpy ndarray, in the formate of
            np.array( [ [ xc, yc, zc ], [ x0, y0, z0 ] ] )
            where ( x0, y0, z0 ) is the point on z-axis that has the shortest DCA
            and ( xc, yc, zc ) is the corresponding point on the cone surface.
    '''


    cdef:
        double  r=0.0, temp=0.0
        double  DCA=0.0
        # coordinates: first is the point on the z axis; second is on the cone surface
        # the distace from these two points are the minimal distance between
        # z axis and cone surface
        double  xCone1=0.0, yCone1=0.0, zCone1=0.0,
        double  xZAxis1=0.0, yZAxis1=0.0, zZAxis1=0.0
        double  xCone2=0.0, yCone2=0.0, zCone2=0.0,
        double  xZAxis2=0.0, yZAxis2=0.0, zZAxis2=0.0
        np.ndarray coordA = np.array([])
        np.ndarray coordB = np.array([])

    # make sure the we are using the correct side of the cone to compute DCA
    if By - Ay > 0 and coneAngle > PI / 2:
        return OUTOFBOUNDARY, np.array([])
    if Ay - By > 0 and coneAngle < PI / 2:
        return OUTOFBOUNDARY, np.array([])

    # radius of the circle:
    if coneAngle > PI / 2:
        coneAngle = PI - coneAngle
    r = (Ay - Cy) * tan( coneAngle )

    # the circle intercepts with z axis:
    if r - fabs( Cx ) > 0:
        DCA = 0.0
        temp = sqrt( r * r - Cx * Cx )

        # xCone1, yCone1, xCone2 and yCone2 are by default 0, but still need to make sure
        # there are with the recon volume
        if ( ( xCone1 > boundaryX1 ) and ( xCone1 < boundaryX2 )
             and
             ( yCone1 > boundaryY1 ) and ( yCone1 < boundaryY2 )
             and
             ( Cz - temp ) > boundaryZ1 and ( Cz - temp ) < boundaryZ2
           ):
            zCone1 = Cz - temp
            zZAxis1 = Cz - temp
            coordA = np.array( [ [xCone1, yCone1, zCone1], [xZAxis1, yZAxis1, zZAxis1] ] )
        if ( ( xCone2 > boundaryX1 ) and ( xCone2 < boundaryX2 )
             and
             ( yCone2 > boundaryY1 ) and ( yCone2 < boundaryY2 )
             and
             ( ( Cz + temp ) > boundaryZ1 ) and ( ( Cz + temp ) < boundaryZ2 )
           ):
            zCone2 = Cz + temp
            zZAxis2 = Cz + temp
            coordB = np.array( [ [xCone2, yCone2, zCone2], [xZAxis1, yZAxis2, zZAxis2] ] )

        if len( coordA ) > 0 and len( coordB ) == 0:
            return DCA, coordA
        elif len( coordA ) == 0 and len( coordB ) > 0:
            return DCA, coordB
        elif len( coordA ) > 0 and len( coordB ) > 0:
            return DCA, np.vstack( ( coordA, coordB ) )
        else:
            return OUTOFBOUNDARY, np.array( [] )

    # the circle is away from z axis:
    # there is only one point on the cone and one point on the z-axis
    elif fabs( Cx ) - r > 0:

        zZAxis1 = Cz

        DCA = ( fabs( Cx ) - r ) * cos( coneAngle )
        xCone1 = DCA * cos( coneAngle ) * np.sign( Cx )
        yCone1 = -DCA * sin( coneAngle )
        zCone1 = Cz

        if ( ( ( xZAxis1 > boundaryX1 and xZAxis1 < boundaryX2 )
               and
               ( yZAxis1 > boundaryY1 and yZAxis1 < boundaryY2 )
               and
               ( zZAxis1 > boundaryZ1 and zZAxis1 < boundaryZ2 )
             )
             and
             ( ( xCone1 > boundaryX1 and xCone1 < boundaryX2 )
               and
               ( yCone1 > boundaryY1 and yCone1 < boundaryY2 )
               and
               ( zCone1 > boundaryZ1 and zCone1 < boundaryZ2 )
             )
           ):
            coordA = np.array( [[xCone1, yCone1, zCone1], [ xZAxis1, yZAxis1, zZAxis1 ]] )
            return DCA, coordA
        else:
            return OUTOFBOUNDARY, np.array( [] )


    # the circle touches z axis:
    else:
        DCA = 0.0
        zZAxis1 = Cz
        zCone1 = Cz
        if ( ( xCone1 > boundaryX1 ) and ( xCone1 < boundaryX2 )
             and
             ( yCone1 > boundaryY1 ) and ( yCone1 < boundaryY2 )
             and
             ( zCone1 > boundaryZ1 ) and ( zCone1 < boundaryZ2 )
           ):
            return DCA, np.array( [[xCone1, yCone1, zCone1], [xZAxis1, yZAxis1, zZAxis1]] )
        else:
            return OUTOFBOUNDARY, np.array( [] )


cpdef tuple zAxisEllipseDCA(    double Ax, double Ay, double Az,
                                double Bx, double By, double Bz,
                                double Cx, double Cy, double Cz,
                                double Dx, double Dy, double Dz,
                                double coneAngle, double phi,
                                double boundaryX1=-100.0, double boundaryX2=100.0,
                                double boundaryY1=-150.0, double boundaryY2=150.0,
                                double boundaryZ1=-200.0, double boundaryZ2=200.0 ):
    '''
    DCA, coord = zAxisEllipseDCA(   double Ax, double Ay, double Az,
                                    double Bx, double By, double Bz,
                                    double Cx, double Cy, double Cz,
                                    double Dx, double Dy, double Dz,
                                    double coneAngle, double phi,
                                    double boundaryX1=-100.0, double boundaryX2=100.0,
                                    double boundaryY1=-150.0, double boundaryY2=150.0,
                                    double boundaryZ1=-200.0, double boundaryZ2=200.0

    compute minimal DCA to z-axis when the intersection is an ellipse.
    note that, the Lagrange multiplier may return the maximal DCA, when the angle
    between the z-axis and the closest cone surface is > pi/2, in which the minimum
    is need to be determined numerically subjected to within the volume. so, if the
    resulted yCone1, is above Ay, return OUTOFBOUNDARY
    returns DCA and the coordinates of the two/four points which gives rise to
    the returned DCA on the cone surface and z-axis


    input:
        Ax, Ay, Az, Bx, By, Bz:
            double, the coordinates of the first  and the second events;
        Cx, Cy, Cz:
            double, the coordiantes of point C, which is the where line AB
            intercepts with xz plane
            note the location of the xz plane is defined by Cy
        Dx, Dy, Dz:
            double, the coordiantes of point D, which is the projection of A
            onto the xz plane
        coneAngle:
            double, cone angle
        phi:
            double, the angle between cone surface and y axis.
    output:
        DCA:
            double, the DCA from the z-axis to cone surface;
        coord:
            numpy ndarray, in the formate of
            np.array( [ [ xc, yc, zc ], [ x0, y0, z0 ] ] )
            where ( x0, y0, z0 ) is the point on z-axis that has the shortest DCA
            and ( xc, yc, zc ) is the corresponding point on the cone surface.
    '''

    cdef:
        # E is the center of ellipse
        double  Ex=0.0, Ey=0.0, Ez=0.0
        # A1 and C1 are the coordinates of A and C in transformed coordinate system
        double  A1x=0.0, A1y=0.0, A1z=0.0, C1x=0.0, C1y=0.0, C1z=0.0
        double  sinBeta=0.0, cosBeta=0.0
        double  ADnorm=0.0, CDnorm=0.0, ACnorm=0.0, DGnorm=0.0, DFnorm=0.0, DEnorm=0.0
        double  A1F1norm=0.0, A1P1norm=0.0, A1H1norm=0.0
        double  focalLength=0.0, majorAxis=0.0, minorAxis=0.0
        double  F1x=0.0, F1y=0.0, F1z=0.0, H1x=0.0, H1y=0.0, H1z=0.0, P1x=0.0, P1y=0.0, P1z=0.0
        double  temp=0.0
        # k, c, l, m are parameters for the transformed z axis.
        double  k=0.0, c=0.0, l=0.0, m=0.0
        double  DELTA=0.0
        # int     nIntercept, nTouch, nAway
        double  x1=0.0, y1=0.0, z1=0.0, x2=0.0, y2=0.0, z2=0.0
        double  DCA=0.0

        double  xCone1=0.0, yCone1=0.0, zCone1=0.0,
        double  xZAxis1=0.0, yZAxis1=0.0, zZAxis1=0.0
        double  xCone2=0.0, yCone2=0.0, zCone2=0.0,
        double  xZAxis2=0.0, yZAxis2=0.0, zZAxis2=0.0
        np.ndarray coordA = np.array([])
        np.ndarray coordB = np.array([])

    # print( 'in elipse' )
    ADnorm = norm( Dx - Ax, Dy - Ay, Dz - Az )
    CDnorm = norm( Dx - Cx, Dy - Cy, Dz - Cz )

    # since CDnorm=0 has been taken care of by the zAxisCircleDCA:
    sinBeta = ( Dx - Cx ) / CDnorm
    cosBeta = ( Dz - Cz ) / CDnorm

    DGnorm = ADnorm * tan( 2 * coneAngle + phi )

    # DFnorm is negative when phi is negative--when D is inside the ellipse:
    DFnorm = ADnorm * tan( phi )
    majorAxis = ( DGnorm - DFnorm ) / 2.0
    DEnorm = DFnorm + majorAxis

    Ex = Dx + DEnorm * ( Cx - Dx ) / CDnorm
    Ey = Dy + DEnorm * ( Cy - Dy ) / CDnorm
    Ez = Dz + DEnorm * ( Cz - Dz ) / CDnorm

    # A1 and C1 are A and C in the transformed coordinates
    # pay attention to the order of x, z and sinBeta, -sinBeta
    A1x, A1y, A1z = matrixVectorDot( cosBeta, 0, -sinBeta, 0, 1, 0, sinBeta, 0, cosBeta, Ax - Ex, Ay - Ey, Az - Ez )
    C1x, C1y, C1z = matrixVectorDot( cosBeta, 0, -sinBeta, 0, 1, 0, sinBeta, 0, cosBeta, Cx - Ex, Cy - Ey, Cz - Ez )

    ACnorm = norm( A1x - C1x, A1y - C1y, A1z - C1z )

    # for unknown reason, there had been NA for minorAxis due to square root of
    # negative numbers:
    temp = ( A1y * A1y + A1z * ( A1z - C1z ) )**2 / ( ACnorm * cos( coneAngle ) )**2 - A1y * A1y - A1z * A1z
    if temp < 0:
        print( temp, coneAngle, ( A1y * A1y + A1z * ( A1z - C1z ) )**2 / ( ACnorm * cos( coneAngle ) )**2, A1y, A1z, ACnorm)
        print( 'ellipse minor axis negative' )
        return OUTOFBOUNDARY, np.array([])
    minorAxis = sqrt( temp )
    temp = majorAxis * majorAxis - minorAxis * minorAxis
    if  temp < 0:
        print( 'ellipse focal length negative' )
        return OUTOFBOUNDARY, np.array([])
    focalLength = sqrt( temp )

    # print( 'major axis: ', majorAxis )
    # print( 'minor axis: ', minorAxis )
    # print( 'focal: ', focalLength )
    # print( 'A1: ', A1x, A1y, A1z )
    # print( 'C1: ', C1x, C1y, C1z )
    # print( 'E: ', Ex, Ey, Ez )

    # to find the DCA to cone surface when the section is ellipse, there are a few
    # different situations to consider: when the major axis of the ellipse is perpendicular,
    # parallel to z axis, or neither.
    # 1. the major axis of the ellipse is perpendicular to z axis ( cosBeta is 0),
    #    the ellipse could intercept, touch, or away from z axis:
    if fabs( cosBeta ) < ZERO:
        # when the ellipse intercepts with z axis
        if majorAxis - fabs( Ex ) > 0:
            DCA = 0.0
            temp = minorAxis * sqrt( 1 - ( Ex / majorAxis ) **2 )
            # print( 'ellipse, cosBeta 0, intercept, z1 and z2: ', temp, -temp )
            if ( ( xCone1 > boundaryX1 ) and ( xCone1 < boundaryX2 )
                 and
                 ( yCone1 > boundaryY1 ) and ( yCone1 < boundaryY2 )
                 and
                 ( ( Ez - temp ) > boundaryZ1 ) and ( ( Ez - temp ) < boundaryZ2 )
               ):
                zCone1 = Ez - temp
                zZAxis1 = Ez - temp
                coordA = np.array( [[ xCone1, yCone1, zCone1 ], [ xZAxis1, yZAxis1, zZAxis1 ]] )
            if ( ( xCone2 > boundaryX1 ) and ( xCone2 < boundaryX2 )
                 and
                 ( yCone2 > boundaryY1 ) and ( yCone2 < boundaryY2 )
                 and
                 ( ( Ez + temp ) > boundaryZ1 ) and ( ( Ez + temp ) < boundaryZ2 )
               ):
                zCone2 = Ez + temp
                zZAxis2 = Ez + temp
                coordB = np.array( [[ xCone2, yCone2, zCone2 ], [ xZAxis2, yZAxis2, zZAxis2 ]] )

            if len( coordA ) > 0 and len( coordB ) == 0:
                return DCA, coordA
            elif len( coordA ) ==0 and len( coordB ) > 0:
                return DCA, coordB
            elif len( coordA ) > 0 and len( coordB ) > 0:
                return DCA, np.vstack( (coordA, coordB) )
            else:
                return OUTOFBOUNDARY, np.array([])

        # when the ellipse is away from z axis:
        elif fabs( Ex ) - majorAxis > 0:
            # print( 'ellipse, cosBeta 0, away')
            zZAxis1 = Ez

            P1x = 0.0
            P1y = 0.0
            P1z = -Ex
            A1P1norm = norm( A1x - P1x, A1y - P1y, A1z - P1z )

            F1x = 0.0
            F1y = 0.0
            if -Ex > majorAxis:
                F1z = majorAxis
            elif -Ex < -majorAxis:
                F1z = -majorAxis

            A1F1norm = norm( A1x - F1x, A1y - F1y, A1z - F1z )
            temp = vectorDot ( P1x - A1x, P1y - A1y, P1z - A1z, F1x - A1x, F1y - A1y, F1z - A1z ) / A1F1norm
            DCA = sqrt( A1P1norm**2 - temp**2 )
            xCone1 = A1x + temp * ( F1x - A1x ) / A1F1norm
            yCone1 = A1y + temp * ( F1y - A1y ) / A1F1norm
            zCone1 = A1z + temp * ( F1z - A1z ) / A1F1norm
            xCone1, yCone1, zCone1 = matrixVectorDot(  cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, xCone1, yCone1, zCone1 )
            xCone1 += Ex
            yCone1 += Ey
            zCone1 += Ez
            if yCone1 > Ay:
                return OUTOFBOUNDARY, np.array( [ ] )
            if ( ( ( xCone1 > boundaryX1 and xCone1 < boundaryX2 )
                   and
                   ( yCone1 > boundaryY1 and yCone1 < boundaryY2 )
                   and
                   ( zCone1 > boundaryZ1 and zCone1 < boundaryZ2 )
                 )
                 and
                 (
                   ( xZAxis1 > boundaryX1 and xZAxis1 < boundaryX2 )
                   and
                   ( yZAxis1 > boundaryY1 and yZAxis1 < boundaryY2 )
                   and
                   ( zZAxis1 > boundaryZ1 and zZAxis1 < boundaryZ2 )
                 )
               ):
                coordA = np.array( [[ xCone1, yCone1, zCone1 ], [ xZAxis1, yZAxis1, zZAxis1 ]] )

            if len( coordA ) > 0:
                return DCA, coordA
            else:
                return OUTOFBOUNDARY, np.array([])


        # when the ellipse touches the z axis
        else:
            # print( 'ellipse, cosBeta 0, touch')
            DCA = 0.0
            zZAxis1 = Ez
            zCone1 = Ez
            if ( ( xCone1 > boundaryX1 ) and ( xCone1 < boundaryX2 )
                 and
                 ( yCone1 > boundaryY1 ) and ( yCone1 < boundaryY2 )
                 and
                 ( Ez > boundaryZ1 ) and ( Ez < boundaryZ2 )
               ):
                return DCA, np.array( [ [ xCone1, yCone1, zCone1 ], [ xZAxis1, yZAxis1, zZAxis1 ] ] )
            else:
                return OUTOFBOUNDARY, np.array( [] )


    # 2. the major axis is parallel to the z axis when sinBeta is 0:
    elif fabs( sinBeta ) < ZERO:
        # the ellipse intercepts with z axis
        if minorAxis - fabs(Ex) > 0:

            DCA = 0.0
            temp = majorAxis * sqrt( 1 - ( fabs(Ex) / minorAxis )**2 )
            # print( 'ellipse, sinBeta 0, intercepts, z1, z2: ', temp, -temp)
            if ( ( xCone1 > boundaryX1 ) and ( xCone1 < boundaryX2 )
                 and
                 ( yCone1 > boundaryY1 ) and ( yCone1 < boundaryY2 )
                 and
                 ( (Ez - temp) > boundaryZ1 ) and ( (Ez - temp) < boundaryZ2 )
               ):
                zZAxis1 = Ez - temp
                zCone1 = Ez - temp
                coordA = np.array( [ [ xCone1, yCone1, zCone1 ], [ xZAxis1, yZAxis1, zZAxis1 ] ] )
            if ( ( xCone2 > boundaryX1 ) and ( xCone2 < boundaryX2 )
                 and
                 ( yCone2 > boundaryY1 ) and ( yCone2 < boundaryY2 )
                 and
                 ( (Ez + temp) > boundaryZ1 ) and ( (Ez + temp) < boundaryZ2 )
               ):
                zZAxis2 = Ez + temp
                zCone2 = Ez + temp
                coordB = np.array( [ [ xCone2, yCone2, zCone2 ], [ xZAxis2, yZAxis2, zZAxis2 ] ] )

            if len( coordA ) > 0 and len( coordB ) == 0:
                return DCA, coordA
            elif len( coordA ) ==0 and len( coordB ) > 0:
                return DCA, coordB
            elif len( coordA ) > 0 and len( coordB ) > 0:
                return DCA, np.vstack( (coordA, coordB) )
            else:
                return OUTOFBOUNDARY, np.array([])

        # ellipse is away from z axis
        elif fabs(Ex) -minorAxis > 0:
            # print( 'ellipse, sinBeta 0, away')
            zZAxis1 = Ez

            P1x = -Ex
            P1y = 0
            P1z = 0
            A1P1norm = norm( A1x - P1x, A1y - P1y, A1z - P1z )

            if -Ex > minorAxis:
                H1x = minorAxis
            elif -Ex < -minorAxis:
                H1x = -minorAxis

            H1y = 0
            H1z = 0
            A1H1norm = norm( A1x - H1x, A1y - H1y, A1z - H1z )
            temp = vectorDot ( P1x - A1x, P1y - A1y, P1z - A1z, H1x - A1x, H1y - A1y, H1z - A1z ) / A1H1norm
            DCA = sqrt( A1P1norm**2 - temp**2 )
            xCone1 = A1x + temp * ( H1x - A1x ) / A1H1norm
            yCone1 = A1y + temp * ( H1y - A1y ) / A1H1norm
            zCone1 = A1z + temp * ( H1z - A1z ) / A1H1norm
            xCone1, yCone1, zCone1 = matrixVectorDot(  cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, xCone1, yCone1, zCone1 )
            xCone1 += Ex
            yCone1 += Ey
            zCone1 += Ez
            if yCone1 > Ay:
                return OUTOFBOUNDARY, np.array( [ ] )
            if ( ( ( xCone1 > boundaryX1 and xCone1 < boundaryX2 )
                   and
                   ( yCone1 > boundaryY1 and yCone1 < boundaryY2 )
                   and
                   ( zCone1 > boundaryZ1 and zCone1 < boundaryZ2 )
                 )
                 and
                 (
                   ( xZAxis1 > boundaryX1 and xZAxis1 < boundaryX2 )
                   and
                   ( yZAxis1 > boundaryY1 and yZAxis1 < boundaryY2 )
                   and
                   ( zZAxis1 > boundaryZ1 and zZAxis1 < boundaryZ2 )
                 )
               ):
                coordA = np.array( [[ xCone1, yCone1, zCone1 ], [ xZAxis1, yZAxis1, zZAxis1 ]] )

            if len( coordA ) > 0:
                return DCA, coordA
            else:
                return OUTOFBOUNDARY, np.array([])

        # the ellipse touches the z axis
        else:
            # print( 'ellipse, sinBeta 0, touch')
            DCA = 0.0
            zZAxis1 = Ez
            zCone1 = Ez
            if ( ( xCone1 > boundaryX1 ) and ( xCone1 < boundaryX2 )
                 and
                 ( yCone1 > boundaryY1 ) and ( yCone1 < boundaryY2 )
                 and
                 ( zCone1 > boundaryZ1 ) and ( zCone1 < boundaryZ2 )
               ):
                return DCA, np.array( [ [ xCone1, yCone1, zCone1 ], [ xZAxis1, yZAxis1, zZAxis1 ] ] )
            else:
                return OUTOFBOUNDARY, np.array( [ ] )


    # 3. the major axis of the ellipse is neither perpendicular nor parallel to z axis:
    else:
        k = -sinBeta / cosBeta
        c = -Ex / cosBeta
        DELTA = 4 * majorAxis**2 * minorAxis**2 * ( ( majorAxis * k )**2 + minorAxis**2 - c**2 )

        # the ellipse intercepts z axis
        if DELTA > 0:
            DCA = 0.0
            z1 = ( -(2 * majorAxis**2 * c * k) + sqrt(DELTA) ) / 2 / ( minorAxis**2 + majorAxis**2 * k**2 )
            x1 = k * z1 + c
            y1 = 0.0
            z2 = ( -(2 * majorAxis**2 * c * k) - sqrt(DELTA) ) / 2 / ( minorAxis**2 + majorAxis**2 * k**2 )
            x2 = k * z2 + c
            y2 = 0.0

            # print( 'ellipse, general, intercept, z1, x1, z2 x2: ', z1, x1, z2, x2)

            xCone1, yCone1, zCone1 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, x1, y1, z1 )
            xCone1 += Ex
            yCone1 += Ey
            zCone1 += Ez
            xZAxis1 = xCone1
            yZAxis1 = yCone1
            zZAxis1 = zCone1

            xCone2, yCone2, zCone2 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, x2, y2, z2 )
            xCone2 += Ex
            yCone2 += Ey
            zCone2 += Ez
            xZAxis2 = xCone2
            yZAxis2 = yCone2
            zZAxis2 = zCone2

            if ( ( xCone1 > boundaryX1 and xCone1 < boundaryX2 )
                 and
                 ( yCone1 > boundaryY1 and yCone1 < boundaryY2 )
                 and
                 ( zCone1 > boundaryZ1 and zCone1 < boundaryZ2 )
               ):
                coordA = np.array( [ [ xCone1, yCone1, zCone1 ], [ xZAxis1, yZAxis1, zZAxis1 ] ] )

            if ( ( xCone2 > boundaryX1 and xCone2 < boundaryX2 )
                 and
                 ( yCone2 > boundaryY1 and yCone2 < boundaryY2 )
                 and
                 ( zCone2 > boundaryZ1 and zCone2 < boundaryZ2 )
               ):
                coordB = np.array( [ [ xCone2, yCone2, zCone2 ], [ xZAxis2, yZAxis2, zZAxis2 ] ] )

            if len( coordA ) > 0 and len( coordB ) == 0:
                return DCA, coordA
            elif len( coordA ) == 0 and len( coordB ) > 0:
                return DCA, coordB
            elif len( coordA ) > 0 and len( coordB ) > 0:
                return DCA, np.vstack( ( coordA, coordB ) )
            else:
                return OUTOFBOUNDARY, np.array ( [ ] )

        # the ellipse is away from z axis
        elif DELTA < 0:
            k = sinBeta
            l = cosBeta
            m = Ex
            z1 = sqrt( majorAxis**4 * k**2 / ( majorAxis**2  * k**2 + minorAxis**2 * l**2 ) )
            x1 = minorAxis**2 * l / majorAxis**2 / k * z1
            y1 = 0.0
            z2 = -z1
            x2 = -x1
            y2 = 0.0
            # print( 'ellipse, general, away, z1, x1, z2 x2: ', z1, x1, z2, x2)
            # print( 'm, l, k: ', m, l, k)
            # print(  -m/l, -m/k)

            d1, xCone1, yCone1, zCone1, xZAxis1, yZAxis1, zZAxis1 = lineDistanceAdvanced( A1x, A1y, A1z, x1, 0, z1, -m/l, 0, 0, 0, 0, -m/k  )
            xCone1, yCone1, zCone1 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, xCone1, yCone1, zCone1 )
            xZAxis1, yZAxis1, zZAxis1 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, xZAxis1, yZAxis1, zZAxis1 )
            xCone1 += Ex
            yCone1 += Ey
            zCone1 += Ez
            xZAxis1 += Ex
            yZAxis1 += Ey

            d2, xCone2, yCone2, zCone2, zZAxis2, yZAxis2, zZAxis2 = lineDistanceAdvanced( A1x, A1y, A1z, x2, 0, z2, -m/l, 0, 0, 0, 0, -m/k  )
            xCone2, yCone2, zCone2 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, xCone2, yCone2, zCone2 )
            xZAxis2, yZAxis2, zZAxis2 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, xZAxis2, yZAxis2, zZAxis2 )
            zZAxis1 += Ez
            xCone2 += Ex
            yCone2 += Ey
            zCone2 += Ez
            xZAxis2 += Ex
            yZAxis2 += Ey
            zZAxis2 += Ez

            if ( ( d1 < d2 )
                 and
                 yCone1 < Ay
                 and
                 ( xCone1 > boundaryX1 ) and ( xCone1 < boundaryX2 )
                 and
                 ( yCone1 > boundaryY1 ) and ( yCone1 < boundaryY2 )
                 and
                 ( zCone1 > boundaryZ1 ) and ( zCone1 < boundaryZ2 )
                 and
                 ( xZAxis1 > boundaryX1 ) and ( xZAxis1 < boundaryX2 )
                 and
                 ( yZAxis1 > boundaryY1 ) and ( yZAxis1 < boundaryY2 )
                 and
                 ( zZAxis1 > boundaryZ1 ) and ( zZAxis1 < boundaryZ2 )
               ):
                return d1, np.array( [ [xCone1, yCone1, zCone1 ], [ xZAxis1, yZAxis1, zZAxis1 ] ] )
            elif ( ( d2 < d1 )
                   and
                   yCone2 < Ay
                   and
                   ( xCone2 > boundaryX1 ) and ( xCone2 < boundaryX2 )
                   and
                   ( yCone2 > boundaryY1 ) and ( yCone2 < boundaryY2 )
                   and
                   ( zCone2 > boundaryZ1 ) and ( zCone2 < boundaryZ2 )
                   and
                   ( xZAxis2 > boundaryX1 ) and ( xZAxis2 < boundaryX2 )
                   and
                   ( yZAxis2 > boundaryY1 ) and ( yZAxis2 < boundaryY2 )
                   and
                   ( zZAxis2 > boundaryZ1 ) and ( zZAxis2 < boundaryZ2 )
                 ):
                return d2, np.array( [ [ xCone2, yCone2, zCone2 ], [xZAxis2, yZAxis2, zZAxis2 ] ] )
            else:
                return OUTOFBOUNDARY, np.array( [ ] )

        # the ellipse touches z axis
        else:
            z1 = -(2 * majorAxis**2 * c * k) / 2 / ( minorAxis**2 + majorAxis**2 * k**2 )
            x1 = k * z1 + c
            y1 = 0.0
            x1, y1, z1 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, x1, y1, z1 )
            x1 += Ex
            y1 += Ey
            z1 += Ez
            if ( ( x1 > boundaryX1 ) and ( x1 < boundaryX2 )
                 and
                 ( y1 > boundaryY1 ) and ( y1 < boundaryY2 )
                 and
                 ( z1 > boundaryZ1 ) and ( z1 < boundaryZ2 )
               ):
                return 0.0, np.array( [ [ x1, y1, z1 ], [ x1, y1, z1 ] ] )
            else:
                return OUTOFBOUNDARY, np.array([])



cpdef tuple zAxisHyperbolaDCA(  double Ax, double Ay, double Az,
                                double Bx, double By, double Bz,
                                double Cx, double Cy, double Cz,
                                double Dx, double Dy, double Dz,
                                double coneAngle, double phi,
                                double boundaryX1=-100.0, double boundaryX2=100.0,
                                double boundaryY1=-150.0, double boundaryY2=150.0,
                                double boundaryZ1=-200.0, double boundaryZ2=200.0,
                                bint  rightSide=False ):
    '''
    DCA, coord = zAxisHyperbolaDCA( double Ax, double Ay, double Az,
                                    double Bx, double By, double Bz,
                                    double Cx, double Cy, double Cz,
                                    double Dx, double Dy, double Dz,
                                    double coneAngle, double phi,
                                    double boundaryX1=-100.0, double boundaryX2=100.0,
                                    double boundaryY1=-150.0, double boundaryY2=150.0,
                                    double boundaryZ1=-200.0, double boundaryZ2=200.0,
                                    bint  rightSide=False )

    compute DCA from z axis to cone surface when the conic section is the hyperbola
    note that, always need to be clear which branch of the hyperbola is calculated.

    input:
        Ax, Ay, Az, Bx, By, Bz:
            double, the coordinates of the first  and the second events;
        Cx, Cy, Cz:
            double, the coordiantes of point C, which is the where line AB
            intercepts with xz plane
            note the location of the xz plane is defined by Cy
        Dx, Dy, Dz:
            double, the coordiantes of point D, which is the projection of A
            onto the xz plane
        coneAngle:
            double, cone angle
        phi:
            double, the angle between cone surface and y axis.
        rightSide:
            boolean, if this is calculating the right side branch of hyperbola
    output:
        DCA:
            double, the DCA from the z-axis to cone surface;
        coord:
            numpy ndarray, in the formate of
            np.array( [ [ xc, yc, zc ], [ x0, y0, z0 ] ] )
            where ( x0, y0, z0 ) is the point on z-axis that has the shortest DCA
            and ( xc, yc, zc ) is the corresponding point on the cone surface.
    '''
    cdef:
        double   sinBeta=0.0, cosBeta=0.0, r=0.0
        double   ADnorm=0.0, CDnorm=0.0, ACnorm=0.0, CM1norm=0.0, DF1norm=0.0, DF2norm=0.0
        double   A1F1norm=0.0, A1F2norm=0.0, A1P1norm=0.0
        # M's are the focal points; F's are the long axis points.
        double   M1x=0.0, M1y=0.0, M1z=0.0, M2x=0.0, M2y=0.0, M2z=0.0
        double   F1x=0.0, F1y=0.0, F1z=0.0, F2x=0.0, F2y=0.0, F2z=0.0
        # E is the center of the hyperbola
        double   Ex=0.0, Ey=0.0, Ez=0.0
        # A1 and C1 are the coordinates of A and C in transformed coordinate system
        double   A1x=0.0, A1y=0.0, A1z=0.0
        double   P1x=0.0, P1y=0.0, P1z=0.0
        double   focalLength=0.0, majorAxis=0.0, minorAxis=0.0
        double   temp=0.0
        # k, c, l, m are parameters for the transformed z axis.
        double   k=0.0, c=0.0, l=0.0, m=0.0
        double   DELTA=0.0
        # int     nIntercept, nTouch, nAway

        # (x1, y1, z1) and (x2, y2, z2) are two points on the hyperbola
        double   x1=0.0, y1=0.0, z1=0.0, d1=0.0, x2=0.0, y2=0.0, z2=0.0

        double  DCA=0.0

        # ( xCone1, yCone1, zCone1 ) and ( xCone2, yCone2, zCone2 ) are the points
        # on the cone surface that give rise to minimal DCA
        # ( xZAxis1, yZAxis1, zZAxis1 ) and ( xZAxis2, yZAxis2, zZAxis2 ) the
        # corresponding points on z-axis
        double  xCone1=0.0, yCone1=0.0, zCone1=0.0
        double  xZAxis1=0.0, yZAxis1=0.0, zZAxis1=0.0
        double  xCone2=0.0, yCone2=0.0, zCone2=0.0
        double  xZAxis2=0.0, yZAxis2=0.0, zZAxis2=0.0
        np.ndarray coordA = np.array([])
        np.ndarray coordB = np.array([])

    # print( 'in hyperbola' )
    ADnorm = norm( Dx - Ax, Dy - Ay, Dz - Az )
    CDnorm = norm( Dx - Cx, Dy - Cy, Dz - Cz )
    # now that it's sure CD norm is not zero:
    sinBeta = ( Dx - Cx ) / CDnorm
    cosBeta = ( Dz - Cz ) / CDnorm

    ACnorm = norm( Cx-Ax, Cy-Ay, Cz-Az )

    # when right size is true ( in case Ay > By and cone angle < PI / 2 )
    # if PI - 2 * coneAngle - phi > PI / 2, then the cone will not intercept
    # with xz plane:
    if rightSide is True and coneAngle < PI / 2 and PI - 2 * coneAngle - phi > PI / 2:
        return OUTOFBOUNDARY, np.array( [] )

    r = ADnorm * ACnorm / ( ACnorm + ADnorm/sin(coneAngle) )
    if r < 0:
        print( 'radius can not be negative' )
        return OUTOFBOUNDARY, np.array( [] )

    CM1norm = r / ADnorm * CDnorm
    DF1norm = ADnorm * tan( phi )
    M1x = Cx + CM1norm * ( Dx - Cx ) / CDnorm
    M1y = Cy + CM1norm * ( Dy - Cy ) / CDnorm
    M1z = Cz + CM1norm * ( Dz - Cz ) / CDnorm
    F1x = Dx + DF1norm * ( Cx - Dx ) / CDnorm
    F1y = Dy + DF1norm * ( Cy - Dy ) / CDnorm
    F1z = Dz + DF1norm * ( Cz - Dz ) / CDnorm

    DF2norm = ADnorm * tan( PI - 2 * coneAngle - phi )
    F2x = Dx + DF2norm * ( Dx - Cx ) / CDnorm
    F2y = Dy + DF2norm * ( Dy - Cy ) / CDnorm
    F2z = Dz + DF2norm * ( Dz - Cz ) / CDnorm
    Ex = ( F1x + F2x ) / 2.0
    Ey = ( F1y + F2y ) / 2.0
    Ez = ( F1z + F2z ) / 2.0

    focalLength = norm( Ex - M1x, Ey - M1y, Ez - M1z )
    # if focalLength < 0:
    #     print( 'hyperbola focal length can not be negative' )
    #     return OUTOFBOUNDARY
    majorAxis = norm( Ex - F1x, Ey - F1y, Ez - F1z )
    if focalLength - majorAxis < 0:
        print( 'hyperbola minorAxis can not be negative')
        return OUTOFBOUNDARY, np.array([])
    minorAxis = sqrt( focalLength * focalLength - majorAxis * majorAxis )
    if majorAxis == 0.0 or minorAxis == 0.0:
        print( 'can not have major/minor axis being 0.0' )
        print( 'r :', r, ' major ', majorAxis, ' minor ', minorAxis)
        print( 'C ', Cx, Cy, Cz )
        print( 'D ', Dx, Dy, Dz )
        print( 'F1 ', F1x, F1y, F1z )
        print( 'F2 ', F2x, F2y, F2z )
        print( 'cone angle ', coneAngle )
        print( 'phi ', phi )
        return OUTOFBOUNDARY, np.array([])
    M2x = Ex + focalLength * ( Dx - Cx ) / CDnorm
    M2y = Ey + focalLength * ( Dy - Cy ) / CDnorm
    M2z = Ez + focalLength * ( Dz - Cz ) / CDnorm

    A1x, A1y, A1z = matrixVectorDot( cosBeta, 0, -sinBeta, 0, 1, 0, sinBeta, 0, cosBeta, Ax - Ex, Ay - Ey, Az - Ez )
    # print( 'cone angle ', coneAngle )
    # print( 'phi ', phi )
    # print( 'C ', Cx, Cy, Cz )
    # print( 'D ', Dx, Dy, Dz )
    # print( 'E: ', Ex, Ey, Ez )
    # print( 'F1 ', F1x, F1y, F1z)
    # print( 'F2 ', F2x, F2y, F2z)
    # print( 'M1 ', M1x, M1y, M1z)
    # print( 'M2 ', M2x, M2y, M2z)
    # print( 'major axis ', majorAxis )
    # print( 'minor axis ', minorAxis )
    # print( 'focal length ', focalLength )
    # print( 'sinBeta ', sinBeta )
    # print( 'cosBeta ', cosBeta )

    # when the major axis of the hyperbola is perpendicular z axis (cosBeta is 0 ):
    if fabs( cosBeta ) < ZERO:
        # when the hyperbola intercepts with z axis:
        if ( not rightSide and -Ex < -majorAxis ) or ( rightSide and -Ex > majorAxis ):
            DCA = 0.0
            temp = minorAxis * sqrt( Ex**2 / majorAxis**2 - 1 )
            # print( 'hyperbola, cosBeta 0, intercept: ', temp)
            # x y for cone and zaxis are default to 0
            zCone1 = Ez - temp
            zZAxis1 = Ez - temp
            zCone2 = Ez + temp
            zZAxis2 = Ez + temp
            if ( ( xCone1 > boundaryX1 ) and ( xCone1 < boundaryX2 )
                 and
                 ( yCone1 > boundaryY1 ) and ( yCone1 < boundaryY2 )
                 and
                 ( zCone1 > boundaryZ1 ) and ( zCone1 < boundaryZ2 )
               ):
                coordA = np.array( [ [ xCone1, yCone1, zCone1 ], [xZAxis1, yZAxis1, zZAxis1 ] ] )

            if ( ( xCone2 > boundaryX1 ) and ( xCone2 < boundaryX2 )
                 and
                 ( yCone2 > boundaryY1 ) and ( yCone2 < boundaryY2 )
                 and
                 ( zCone2 > boundaryZ1 ) and ( zCone2 < boundaryZ2 )
               ):
                coordB = np.array( [ [ xCone2, yCone2, zCone2 ], [xZAxis2, yZAxis2, zZAxis2 ] ] )

            if len( coordA ) > 0 and len( coordB ) == 0:
                return DCA, coordA
            elif len( coordA ) == 0 and len( coordB ) > 0:
                return DCA, coordB
            elif len( coordA ) > 0 and len( coordB ) > 0:
                return DCA, np.vstack( ( coordA, coordB ) )
            else:
                return OUTOFBOUNDARY, np.array( [ ] )

        # when the hyperbola tangents to the z axis:
        elif ( not rightSide and fabs( -Ex + majorAxis ) < ZERO ) or ( rightSide and fabs( -Ex - majorAxis) < ZERO ):
            DCA = 0.0
            xCone1 = Ez
            xZAxis1 = Ez
            if ( ( xCone1 > boundaryX1 ) and ( xCone1 < boundaryX2 )
                 and
                 ( yCone1 < boundaryY1 ) and ( yCone1 < boundaryY2 )
                 and
                 ( zCone1 > boundaryZ1 ) and ( zCone1 < boundaryZ2 )
               ):
                return DCA, np.array( [ [ xCone1, yCone1, zCone1 ], [ xZAxis1, yZAxis1, zZAxis1 ] ] )
            else:
                return OUTOFBOUNDARY, np.array( [ ] )

        # when the hyperbola does not touch z axis:
        else:
            # print( 'hyperbola, cosBeta 0, away')
            P1x = 0
            P1y = 0
            P1z = -Ex
            A1P1norm = norm( A1x - P1x, A1y - P1y, A1z - P1z )
            F1x = 0.0
            F1y = 0.0
            if not rightSide:
                F1z = -majorAxis
            else:
                F1z = majorAxis
            # print( 'F1: ', F1x, F1y, F1z )
            # print( 'P1: ', P1x, P1y, P1z )
            # print( 'A1: ', A1x, A1y, A1z )
            A1F1norm = norm( A1x - F1x, A1y - F1y, A1z - F1z )
            temp = vectorDot ( P1x - A1x, P1y - A1y, P1z - A1z, F1x - A1x, F1y - A1y, F1z - A1z ) / A1F1norm

            DCA = sqrt( A1P1norm**2 - temp**2 )
            xCone1 = A1x + temp * ( P1x - A1x ) / A1P1norm
            yCone1 = A1y + temp * ( P1y - A1y ) / A1P1norm
            zCone1 = A1z + temp * ( P1z - A1z ) / A1P1norm
            xCone1, yCone1, zCone1 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, xCone1, yCone1, zCone1 )
            xCone1 += Ex
            yCone1 += Ey
            zCone1 += Ez

            # yCone1 must be less than A1y, otherwise, it's on the wrong side of the cone:
            if yCone1 > Ay:
                return OUTOFBOUNDARY, np.array( [ ] )

            # xZAxis1, yZAxis1 are defaulted to 0
            zZAxis1 = Ez
            if ( ( xCone1 > boundaryX1 ) and ( xCone1 < boundaryX2 )
                 and
                 ( yCone1 > boundaryY1 ) and ( yCone1 < boundaryY2 )
                 and
                 ( zCone1 > boundaryZ1 ) and ( zCone1 < boundaryZ2 )
                 and
                 ( xZAxis1 > boundaryX1 ) and ( xZAxis1 < boundaryX2 )
                 and
                 ( yZAxis1 > boundaryY1 ) and ( yZAxis1 < boundaryY2 )
                 and
                 ( zZAxis1 > boundaryZ1 ) and ( zZAxis1 < boundaryZ2)
               ):
               return DCA, np.array( [ [ xCone1, yCone1, zCone1 ], [ xZAxis1, yZAxis1, zZAxis1 ] ] )
            else:
                return OUTOFBOUNDARY, np.array( [] )

    # the major axis is parallel to the z axis ( sinBeta is 0 ),
    # in which case, the z axis always intercepts with the hyperbola
    elif fabs( sinBeta ) < ZERO:
        DCA = 0.0
        if not rightSide:
            temp = -majorAxis * sqrt( 1 + Ex**2 / minorAxis**2 )
        else:
            temp = majorAxis * sqrt( 1 + Ex**2 / minorAxis**2 )
        zCone1 = Ez + temp
        zZAxis1 = Ez + temp
        if ( ( xCone1 > boundaryX1 ) and ( xCone1 < boundaryX2 )
             and
             ( yCone1 > boundaryY1 ) and ( yCone1 < boundaryY2 )
             and
             ( zCone1 > boundaryZ1 ) and ( zCone1 < boundaryZ2 )
           ):
            return DCA, np.array( [ [ xCone1, yCone1, zCone1 ], [ xZAxis1, yZAxis1, zZAxis1 ] ] )
        else:
            return OUTOFBOUNDARY, np.array( [ ] )

    # when the major axis of the hyperbola is neither parallel nor perpendicular
    # to the z axis:
    else:
        # print( 'parallel to neither')
        k = -sinBeta / cosBeta
        c = -Ex / cosBeta
        DELTA = 4 * majorAxis**2 * minorAxis**2 * ( minorAxis**2 + c**2 - ( majorAxis * k )**2 )

        # when the  z axis intercepts with hyperbola (two solutions ):
        if DELTA > 0:
            # print( 'parallel to neither, two solutions')
            z1 = ( 2 * majorAxis**2 * c * k + sqrt( DELTA ) ) / 2 / ( minorAxis**2 - majorAxis**2 * k**2 )
            x1 = k * z1 + c
            y1 = 0.0
            z2 = ( 2 * majorAxis**2 * c * k - sqrt( DELTA ) ) / 2 / ( minorAxis**2 - majorAxis**2 * k**2 )
            x2 = k * z2 + c
            y2 = 0.0

            # order the two points such that z1 < z2 ( z1 on the left of z2 )
            if z1 > z2:
                x1, x2 = swap( x1, x2 )
                y1, y2 = swap( y1, y2 )
                z1, z2 = swap( z1, z2 )
            # print( 'solution 1: ', x1, y1, z1 )
            # print( 'solution 2: ', x2, y2, z2 )
            # the two solutions may both on the left branch, or both on the right
            # branch, or one on each branch;

            # if we are looking at the left branch:
            if not rightSide:
                # both on the left side:
                if z2 < 0:
                    # print( 'parallel to neither, two solutions, both on left')
                    DCA = 0.0
                    xCone1, yCone1, zCone1 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, x1, y1, z1 )
                    xCone1 += Ex
                    yCone1 += Ey
                    zCone1 += Ez
                    xZAxis1 = xCone1
                    yZAxis1 = yCone1
                    zZAxis1 = zCone1

                    xCone2, yCone2, zCone2 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, x2, y2, z2 )
                    xCone2 += Ex
                    yCone2 += Ey
                    zCone2 += Ez
                    xZAxis2 = xCone2
                    yZAxis2 = yCone2
                    zZAxis2 = zCone2
                    # print( 'cone 1 ', xCone1, yCone1, zCone1 )
                    # print( 'cone 2 ', xCone2, yCone2, zCone2 )

                    if ( ( xCone1 > boundaryX1 ) and ( xCone1 < boundaryX2 )
                         and
                         ( yCone1 > boundaryY1 ) and ( yCone1 < boundaryY2 )
                         and
                         ( zCone1 > boundaryZ1 ) and ( zCone1 < boundaryZ2 )
                       ):
                       coordA = np.array( [ [ xCone1, yCone1, zCone1 ], [ xZAxis1, yZAxis1, zZAxis1 ] ] )

                    if ( ( xCone2 > boundaryX1 ) and ( xCone2 < boundaryX2 )
                         and
                         ( yCone2 > boundaryY1 ) and ( yCone2 < boundaryY2 )
                         and
                         ( zCone2 > boundaryZ1 ) and ( zCone2 < boundaryZ2 )
                       ):
                       coordB = np.array( [ [ xCone2, yCone2, zCone2 ], [ xZAxis2, yZAxis2, zZAxis2 ] ] )

                    if len( coordA ) > 0 and len( coordB ) == 0:
                        return DCA, coordA
                    elif len( coordA ) == 0 and len( coordB ) > 0:
                        return DCA, coordB
                    elif len( coordA ) > 0 and len( coordB ) > 0:
                        return DCA, np.vstack ( ( coordA, coordB ) )
                    else:
                        return OUTOFBOUNDARY, np.array( [ ] )

                # just one point on the left branch
                elif z1 < 0 and z2 > 0:
                    # print( 'parallel to neither, two solutions, one on left')
                    DCA = 0.0
                    xCone1, yCone1, zCone1 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, x1, y1, z1 )
                    # print( 'point on cone ', xCone1, yCone1, zCone1)

                    xCone1 += Ex
                    yCone1 += Ey
                    zCone1 += Ez
                    # print( 'point on cone ', xCone1, yCone1, zCone1)
                    xZAxis1 = xCone1
                    yZAxis1 = yCone1
                    zZAxis1 = zCone1
                    if ( ( xCone1 > boundaryX1 ) and ( xCone1 < boundaryX2 )
                         and
                         ( yCone1 > boundaryY1 ) and ( yCone1 < boundaryY2 )
                         and
                         ( zCone1 > boundaryZ1 ) and ( zCone1 < boundaryZ2 )
                       ):
                        coordA = np.array( [ [ xCone1, yCone1, zCone1 ], [ xZAxis1, yZAxis1, zZAxis1 ] ] )
                        return DCA, coordA
                    else:
                        return OUTOFBOUNDARY, np.array ( [ ] )

                # if neither solution is on the left branch, then find the point
                # on the left branch that gives the extrema distance
                # the extrema can only be minimal or maximal, if the resulted yCone1
                # is larger than the apex of the cone, then it's the maximum, and
                # should be discarded:
                elif z1 > 0.0:
                    # print( 'parallel to neither, two solutions, neither on left')
                    k = sinBeta
                    l = cosBeta
                    m = Ex
                    # only interested in the left branch here, only take negative z
                    z1 = -sqrt( majorAxis**4 * k**2 / ( majorAxis**2  * k**2 - minorAxis**2 * l**2 ) )
                    x1 = -minorAxis**2 * l / majorAxis**2 / k * z1
                    y1 = 0.0
                    # print( 'x1, y1, z1: ', x1, y1, z1 )
                    DCA, xCone1, yCone1, zCone1, xZAxis1, yZAxis1, zZAxis1 = lineDistanceAdvanced( A1x, A1y, A1z, x1, y1, z1, -m/l, 0, 0, 0, 0, -m/k  )
                    xCone1, yCone1, zCone1 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, xCone1, yCone1, zCone1 )
                    xCone1 += Ex
                    yCone1 += Ey
                    zCone1 += Ez
                    if yCone1 > Ay:
                        return OUTOFBOUNDARY, np.array( [ ] )
                    xZAxis1, yZAxis1, zZAxis1 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, xZAxis1, yZAxis1, zZAxis1 )
                    xZAxis1 += Ex
                    yZAxis1 += Ey
                    zZAxis1 += Ez
                    # print( 'xCone1, yCone1, zCone1: ', xCone1, yCone1, zCone1 )
                    # print( 'xZAxis1, yZAxis1, zZAxis1: ', xZAxis1, yZAxis1, zZAxis1)

                    if ( ( xCone1 > boundaryX1 ) and ( xCone1 < boundaryX2 )
                         and
                         ( yCone1 > boundaryY1 ) and ( yCone1 < boundaryY2 )
                         and
                         ( zCone1 > boundaryZ1 ) and ( zCone1 < boundaryZ2 )
                         and
                         ( xZAxis1 > boundaryX1 ) and ( xZAxis1 < boundaryX2 )
                         and
                         ( yZAxis1 > boundaryY1 ) and ( yZAxis1 < boundaryY2 )
                         and
                         ( zZAxis1 > boundaryZ1 ) and ( zZAxis1 < boundaryZ2 )
                       ):
                        coordA = np.array( [ [ xCone1, yCone1, zCone1 ], [ xZAxis1, yZAxis1, zZAxis1 ] ] )
                        return  DCA, coordA
                    else:
                        # print( 'parallel to neither, two solutions, neither on left, return OUTOFBOUNDARY')
                        return OUTOFBOUNDARY, np.array( [ ]  )
                else:
                    return OUTOFBOUNDARY, np.array( [ ] )

            # if we are looking at the right side branch
            if rightSide:
                # both points on the right side:
                if z1 > 0.0:
                    # print( 'parallel to neither, two solutions, both on right')
                    DCA = 0.0
                    xCone1, yCone1, zCone1 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, x1, y1, z1 )
                    xCone1 += Ex
                    yCone1 += Ey
                    zCone1 += Ez
                    xZAxis1 = xCone1
                    yZAxis1 = yCone1
                    zZAxis1 = zCone1
                    xCone2, yCone2, zCone2 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, x2, y2, z2 )
                    xCone2 += Ex
                    yCone2 += Ey
                    zCone2 += Ez
                    xZAxis2 = xCone2
                    yZAxis2 = yCone2
                    zZAxis2 = zCone2

                    if ( ( xCone1 > boundaryX1 ) and ( xCone1 < boundaryX2 )
                         and
                         ( yCone1 > boundaryY1 ) and ( yCone1 < boundaryY2 )
                         and
                         ( zCone1 > boundaryZ1 ) and ( zCone1 < boundaryZ2 )
                       ):
                        coordA = np.array( [ [ xCone1, yCone1, zCone1 ], [ xZAxis1, yZAxis1, zZAxis1 ] ] )

                    if ( ( xCone2 > boundaryX1 ) and ( xCone2 < boundaryX2 )
                         and
                         ( yCone2 > boundaryY1 ) and ( yCone2 < boundaryY2 )
                         and
                         ( zCone2 > boundaryZ1 ) and ( zCone2 < boundaryZ2 )
                       ):
                        coordB = np.array( [ [ xCone2, yCone2, zCone2 ], [ xZAxis2, yZAxis2, zZAxis2 ] ] )

                    if len( coordA ) > 0 and len( coordB ) == 0:
                        return DCA, coordA
                    elif len( coordA ) == 0 and len( coordB ) > 0:
                        return DCA, coordB
                    elif len( coordA ) > 0 and len( coordB ) > 0:
                        return DCA, np.vstack ( ( coordA ,coordB ) )
                    else:
                        return OUTOFBOUNDARY, np.array( [ ]  )

                # only one point on the right branch
                elif z2 > 0.0 and z1 < 0:
                    # print( 'parallel to neither, two solutions, one on right')
                    xCone1, yCone1, zCone1 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, x2, y2, z2 )
                    xCone1 += Ex
                    yCone1 += Ey
                    zCone1 += Ez
                    xZAxis1 = xCone1
                    yZAxis1 = yCone1
                    zZAxis1 = zCone1

                    if ( ( xCone1 > boundaryX1 ) and ( xCone1 < boundaryX2 )
                         and
                         ( yCone1 > boundaryY1 ) and ( yCone1 < boundaryY2 )
                         and
                         ( zCone1 > boundaryZ1 ) and ( zCone1 < boundaryZ2 )
                       ):
                        coordA = np.array( [ [ xCone1, yCone1, zCone1 ], [ xZAxis1, yZAxis1, zZAxis1 ] ] )
                        return DCA, coordA
                    else:
                        return OUTOFBOUNDARY, np.array( [ ] )

                # both solutins are on the left branch:
                elif z2 < 0.0:
                    # print( 'parallel to neither, two solutions, neither on right')
                    k = sinBeta
                    l = cosBeta
                    m = Ex
                    # only interested in the right branch here, only take positive z
                    z1 = sqrt( majorAxis**4 * k**2 / ( majorAxis**2  * k**2 - minorAxis**2 * l**2 ) )
                    x1 = -minorAxis**2 * l / majorAxis**2 / k * z1
                    y1 = 0.0
                    DCA, xCone1, yCone1, zCone1, xZAxis1, yZAxis1, zZAxis1 = lineDistanceAdvanced( A1x, A1y, A1z, x1, y1, z1, -m/l, 0, 0, 0, 0, -m/k  )
                    xCone1, yCone1, zCone1 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, xCone1, yCone1, zCone1 )
                    xCone1 += Ex
                    yCone1 += Ey
                    zCone1 += Ez
                    if yCone1 > Ay:
                        return OUTOFBOUNDARY, np.array( [ ] )
                    xZAxis1, yZAxis1, zZAxis1 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, xZAxis1, yZAxis1, zZAxis1 )
                    xZAxis1 += Ex
                    yZAxis1 += Ey
                    zZAxis1 += Ez
                    if ( ( xCone1 > boundaryX1 ) and ( xCone1 < boundaryX2 )
                         and
                         ( yCone1 > boundaryY1 ) and ( yCone1 < boundaryY2 )
                         and
                         ( zCone1 > boundaryZ1 ) and ( zCone1 < boundaryZ2 )
                         and
                         ( xZAxis1 > boundaryX1 ) and ( xZAxis1 < boundaryX2 )
                         and
                         ( yZAxis1 > boundaryY1 ) and ( yZAxis1 < boundaryY2 )
                         and
                         ( zZAxis1 > boundaryZ1 ) and ( zZAxis1 < boundaryZ2 )
                       ):
                        coordA = np.array( [ [ xCone1, yCone1, zCone1 ], [ xZAxis1, yZAxis1, zZAxis1 ] ] )
                        return  DCA, coordA
                    else:
                        return OUTOFBOUNDARY, np.array( [ ]  )
                return OUTOFBOUNDARY, np.array( [ ]  )

        # there is no solution, the hyperbola and the z axis do not intercept:
        elif DELTA < 0:
            # print( 'parallel to neither, no solutions')
            k = sinBeta
            l = cosBeta
            m = Ex
            if not rightSide:
                # print( 'parallel to neither, no solutions, working on left')
                z1 = -sqrt( majorAxis**4 * k**2 / ( majorAxis**2  * k**2 - minorAxis**2 * l**2 ) )
                x1 = -minorAxis**2 * l / majorAxis**2 / k * z1
                y1 = 0.0
                DCA, xCone1, yCone1, zCone1, xZAxis1, yZAxis1, zZAxis1 = lineDistanceAdvanced( A1x, A1y, A1z, x1, y1, z1, -m/l, 0, 0, 0, 0, -m/k  )
                xCone1, yCone1, zCone1 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, xCone1, yCone1, zCone1 )
                xCone1 += Ex
                yCone1 += Ey
                zCone1 += Ez
                if yCone1 > Ay:
                    return OUTOFBOUNDARY, np.array( [ ] )
                xZAxis1, yZAxis1, zZAxis1 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, xZAxis1, yZAxis1, zZAxis1 )
                xZAxis1 += Ex
                yZAxis1 += Ey
                zZAxis1 += Ez
                if ( ( xCone1 > boundaryX1 ) and ( xCone1 < boundaryX2 )
                     and
                     ( yCone1 > boundaryY1 ) and ( yCone1 < boundaryY2 )
                     and
                     ( zCone1 > boundaryZ1 ) and ( zCone1 < boundaryZ2 )
                     and
                     ( xZAxis1 > boundaryX1 ) and ( xZAxis1 < boundaryX2 )
                     and
                     ( yZAxis1 > boundaryY1 ) and ( yZAxis1 < boundaryY2 )
                     and
                     ( zZAxis1 > boundaryZ1 ) and ( zZAxis1 < boundaryZ2 )
                   ):
                    coordA = np.array( [ [ xCone1, yCone1, zCone1 ], [ xZAxis1, yZAxis1, zZAxis1 ] ] )
                    return  DCA, coordA
                else:
                    return OUTOFBOUNDARY, np.array( [ ]  )

            else:
                # print( 'parallel to neither, no solutions, working on right')
                z1 = sqrt( majorAxis**4 * k**2 / ( majorAxis**2  * k**2 - minorAxis**2 * l**2 ) )
                x1 = -minorAxis**2 * l / majorAxis**2 / k * z1
                y1 = 0.0
                DCA, xCone1, yCone1, zCone1, xZAxis1, yZAxis1, zZAxis1 = lineDistanceAdvanced( A1x, A1y, A1z, x1, y1, z1, -m/l, 0, 0, 0, 0, -m/k  )
                xCone1, yCone1, zCone1 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, xCone1, yCone1, zCone1 )
                xCone1 += Ex
                yCone1 += Ey
                zCone1 += Ez
                if yCone1 > Ay:
                    return OUTOFBOUNDARY, np.array( [ ] )
                xZAxis1, yZAxis1, zZAxis1 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, xZAxis1, yZAxis1, zZAxis1 )
                xZAxis1 += Ex
                yZAxis1 += Ey
                zZAxis1 += Ez
                if ( ( xCone1 > boundaryX1 ) and ( xCone1 < boundaryX2 )
                     and
                     ( yCone1 > boundaryY1 ) and ( yCone1 < boundaryY2 )
                     and
                     ( zCone1 > boundaryZ1 ) and ( zCone1 < boundaryZ2 )
                     and
                     ( xZAxis1 > boundaryX1 ) and ( xZAxis1 < boundaryX2 )
                     and
                     ( yZAxis1 > boundaryY1 ) and ( yZAxis1 < boundaryY2 )
                     and
                     ( zZAxis1 > boundaryZ1 ) and ( zZAxis1 < boundaryZ2 )
                   ):
                    coordA = np.array( [ [ xCone1, yCone1, zCone1 ], [ xZAxis1, yZAxis1, zZAxis1 ] ] )
                    return  DCA, coordA
                else:
                    return OUTOFBOUNDARY, np.array( [ ]  )


        # if the hyperbola is tangents to z axis, there is only one solution:
        else:
            z1 = ( majorAxis**2 * c * k  ) / ( minorAxis**2 - majorAxis**2 * k**2 )
            x1 = k * z1 + c
            y1 = 0.0
            if not rightSide:
                # left branch and the solution is on the left side:
                if z1 < 0:
                    DCA = 0.0
                    xCone1, yCone1, zCone1 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, xCone1, yCone1, zCone1 )
                    xCone1 += Ex
                    yCone1 += Ey
                    zCone1 += Ez
                    xZAxis1 = xCone1
                    yZAxis1 = yCone1
                    zZAxis1 = zCone1
                    if ( ( xCone1 > boundaryX1 ) and ( xCone1 < boundaryX2 )
                         and
                         ( yCone1 > boundaryY1 ) and ( yCone1 < boundaryY2 )
                         and
                         ( zCone1 > boundaryZ1 ) and ( zCone1 < boundaryZ2 )
                       ):
                        coordA = np.array( [ [ xCone1, yCone1, zCone1 ], [ xZAxis1, yZAxis1, zZAxis1 ] ] )
                        return  DCA, coordA
                    else:
                        return OUTOFBOUNDARY, np.array( [ ]  )

                # left branch but the solution is on the right side:
                else:
                    k = sinBeta
                    l = cosBeta
                    m = Ex
                    # only interested in the left branch here, only take negative z
                    z1 = -sqrt( majorAxis**4 * k**2 / ( majorAxis**2  * k**2 - minorAxis**2 * l**2 ) )
                    x1 = -minorAxis**2 * l / majorAxis**2 / k * z1
                    y1 = 0.0

                    DCA, xCone1, yCone1, zCone1, xZAxis1, yZAxis1, zZAxis1 = lineDistanceAdvanced( A1x, A1y, A1z, x1, y1, z1, -m/l, 0, 0, 0, 0, -m/k  )
                    xCone1, yCone1, zCone1 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, xCone1, yCone1, zCone1 )
                    xCone1 += Ex
                    yCone1 += Ey
                    zCone1 += Ez
                    if yCone1 > Ay:
                        return OUTOFBOUNDARY, np.array( [ ] )
                    xZAxis1, yZAxis1, zZAxis1 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, xZAxis1, yZAxis1, zZAxis1 )
                    xZAxis1 += Ex
                    yZAxis1 += Ey
                    zZAxis1 += Ez
                    if ( ( xCone1 > boundaryX1 ) and ( xCone1 < boundaryX2 )
                         and
                         ( yCone1 > boundaryY1 ) and ( yCone1 < boundaryY2 )
                         and
                         ( zCone1 > boundaryZ1 ) and ( zCone1 < boundaryZ2 )
                         and
                         ( xZAxis1 > boundaryX1 ) and ( xZAxis1 < boundaryX2 )
                         and
                         ( yZAxis1 > boundaryY1 ) and ( yZAxis1 < boundaryY2 )
                         and
                         ( zZAxis1 > boundaryZ1 ) and ( zZAxis1 < boundaryZ2 )
                       ):
                        coordA = np.array( [ [ xCone1, yCone1, zCone1 ], [ xZAxis1, yZAxis1, zZAxis1 ] ] )
                        return  DCA, coordA
                    else:
                        return OUTOFBOUNDARY, np.array( [ ]  )


            elif rightSide:
                # right branch and the solution is also on the right side:
                if z1 > 0:
                    DCA = 0.0
                    xCone1, yCone1, zCone1 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, xCone1, yCone1, zCone1 )
                    xCone1 += Ex
                    yCone1 += Ey
                    zCone1 += Ez
                    xZAxis1 = xCone1
                    yZAxis1 = yCone1
                    zZAxis1 = zCone1
                    if ( ( xCone1 > boundaryX1 ) and ( xCone1 < boundaryX2 )
                         and
                         ( yCone1 > boundaryY1 ) and ( yCone1 < boundaryY2 )
                         and
                         ( zCone1 > boundaryZ1 ) and ( zCone1 < boundaryZ2 )
                       ):
                        coordA = np.array( [ [ xCone1, yCone1, zCone1 ], [ xZAxis1, yZAxis1, zZAxis1 ] ] )
                        return  DCA, coordA
                    else:
                        return OUTOFBOUNDARY, np.array( [ ]  )

                # right branch but the solution is on the left side:
                else:
                    k = sinBeta
                    l = cosBeta
                    m = Ex
                    # only interested in the right branch here, only take positive z
                    z1 = sqrt( majorAxis**4 * k**2 / ( majorAxis**2  * k**2 - minorAxis**2 * l**2 ) )
                    x1 = -minorAxis**2 * l / majorAxis**2 / k * z1
                    y1 = 0.0

                    DCA, xCone1, yCone1, zCone1, xZAxis1, yZAxis1, zZAxis1 = lineDistanceAdvanced( A1x, A1y, A1z, x1, y1, z1, -m/l, 0, 0, 0, 0, -m/k  )
                    xCone1, yCone1, zCone1 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, xCone1, yCone1, zCone1 )
                    xCone1 += Ex
                    yCone1 += Ey
                    zCone1 += Ez
                    if yCone1 > Ay:
                        return OUTOFBOUNDARY, np.array( [ ] )
                    xZAxis1, yZAxis1, zZAxis1 = matrixVectorDot( cosBeta, 0, sinBeta, 0, 1, 0, -sinBeta, 0, cosBeta, xZAxis1, yZAxis1, zZAxis1 )
                    xZAxis1 += Ex
                    yZAxis1 += Ey
                    zZAxis1 += Ez
                    if ( ( xCone1 > boundaryX1 ) and ( xCone1 < boundaryX2 )
                         and
                         ( yCone1 > boundaryY1 ) and ( yCone1 < boundaryY2 )
                         and
                         ( zCone1 > boundaryZ1 ) and ( zCone1 < boundaryZ2 )
                         and
                         ( xZAxis1 > boundaryX1 ) and ( xZAxis1 < boundaryX2 )
                         and
                         ( yZAxis1 > boundaryY1 ) and ( yZAxis1 < boundaryY2 )
                         and
                         ( zZAxis1 > boundaryZ1 ) and ( zZAxis1 < boundaryZ2 )
                       ):
                        coordA = np.array( [ [ xCone1, yCone1, zCone1 ], [ xZAxis1, yZAxis1, zZAxis1 ] ] )
                        return  DCA, coordA
                    else:
                        return OUTOFBOUNDARY, np.array( [ ]  )
            return OUTOFBOUNDARY, np.array( [ ] )




cpdef np.ndarray[np.float_t, ndim=1] doubleProbabilityPointSource(    double E0,
                                                                    double sx, double sy, double sz,
                                                                    double eDepA, double Ax, double Ay, double Az,
                                                                    double eDepB, double Bx, double By, double Bz,
                                                                    double cameraX, double cameraY, double cameraZ,
                                                                    int nC=8,
                                                                    bint fullAbsorption=True ):
    '''
    p = dblEventsProbKnownSource( double E0,
                                    double sx, double sy, double sz,
                                    double eDepA, double Ax, double Ay, double Az,
                                    double eDepB, double Bx, double By, double Bz,
                                    double cameraX, double cameraY, double cameraZ,
                                    int nC=8,
                                    bint fullAbsorption=True )

    compute the probability of a double events, given the event order and source
    position.

    since the source position is known, the entrace OD is accurate: if the photon
    pass through more than one crystal before the first interaction, they are
    counted.

    the exit OD is approximated over 8 ( 8 is the default ) possilbe exiting
    directions.

    the probability is calculated based on a few pieces:
        pEnter, entrace probability, which is the probability the photon travel to the
        first interaction location--attenuation
        pA, probability of the first interaction, which is the KN given E0 and ( E0- eDepA )
        in terms of d_sigma/d_eDep
        pA2B, the probability of the photon travel from A to B and ends at B
        pB, the probability if the second interaction, use KN if B is Compton(d_sigma/d_eDep),
        otherwise, photoelectric
        pExit, the probability of the photon exit the crystal withou further interaction

    note, acctually, it's the natural log of the praboility is calcuated, and the
    constant, r_0 / 2, where r_0 is the classic radius of electron is dropped


    note while unites for most of the calculation is in mm, in this particular
    function, becasue the attenuation coeff's are in cm, the the unites are
    converted

    input:
        E0, sx, sy, sz:
            doulbe, E0 is the initial energy of the Gamma, sx, sy,and sz is source position
        eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz
            double, the deposited energy and coordinates of the two events,
            order is important--A is the first interaction, B is the second
        cameraX, cameraY, cameraZ:
            double, it's the first 3 paramters in par['det'], cameraX and cameraZ
            is coordinates of isocenter projected on camera surface, cameraY is
            the location of the camera front surface( close to the source)
        nC:
            int, number of directions exit optical depth is averaged over;
        fullAbsorption:
            boolean, if this is a full absorption
    return:
        p:
            numpy ndarray (dim=1), [ pTotal, pEnter, pA, pA2B, pB, pExit ]

    '''

    cdef:
        double      theta=0.0, d=0.0, E1=0.0, u=0.0, E2=0.0, d_eDepA=0.0, d_eDepB=0.0, de=0.0
        double      entranceDepth=0.0, exitDepth=0.0
        # double      alpha=0.0, nf=0.0
        double      pEnter=0.0, pA=0.0, pA2B=0.0, pB=0.0, pExit=0.0, pTotal=0.0
        np.ndarray[np.float64_t, ndim=1]  p=np.zeros( 6, dtype=np.float64 )

    d_eDepA = sqrt( 25e-6 + 2.35**2 * eDepA * 5e-6 )
    d_eDepB = sqrt( 25e-6 + 2.35**2 * eDepB * 5e-6 )
    de = sqrt( d_eDepA**2 + d_eDepB**2 )

    entranceDepth = opticalDepth( sx, sy, sz, Ax, Ay, Az, cameraX, cameraY, cameraZ ) / 10.0

    u = interp1d( eCZT, attenCZT, E0 )
    pEnter =   -u * rhoCZT * entranceDepth
    # p = 0.0

    E1 = E0 - eDepA
    if E1 < ZERO:
        pTotal = IMPOSSIBLE
        p = np.array( [ pTotal, pEnter, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE])
        return p
    theta = scatterAngle( E0, eDepA )
    if theta < 0.0:
        pTotal = IMPOSSIBLE
        p = np.array( [ pTotal, pEnter, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE])
        return p

    pA = log( ( E1 / E0 )**2 * ( E1 / E0 + E0 / E1 - sin(theta)**2 ) * 2 * PI * 0.511 / E1**2 )
    # pA = pA - log( sin(theta) )

    # d = norm( Ax - Bx, Ay - By, Az - Bz ) / 10.0
    d = opticalDepth( Ax, Ay, Az, Bx, By, Bz, cameraX, cameraY, cameraZ ) / 10.0
    u = interp1d( eCZT, attenCZT, E1 )
    pA2B = - u * rhoCZT * d - log( sin(theta) )

    # if full absoption is assumed:
    if fullAbsorption:
        u = interp1d( eCZT, peCZT, eDepB )
        pB = log( u ) - ( E0 - eDepA - eDepB ) ** 2 / 2 / de**2 - 0.5 * log( 2 * PI * de**2 )
        pTotal = pEnter + pA + pA2B + pB + pExit
        p = np.array( [ pTotal, pEnter, pA, pA2B, pB, pExit ] )
        return p

    # compton-compton is assumed
    else:
        E2 = E1 - eDepB
        theta = scatterAngle( E1, eDepB )
        if theta  < 0.0:
            pTotal = IMPOSSIBLE
            p = np.array( [ pTotal, pEnter, pA, pA2B, IMPOSSIBLE, IMPOSSIBLE])
            return p

        pB = log( ( E2 / E1 )**2 * ( E2 / E1 + E1 / E2 - sin(theta)**2 ) * 2 * PI * 0.511 / E2**2 )

        exitDepth = opticalDepthApprox( Ax, Ay, Az, Bx, By, Bz, cameraX, cameraY, cameraZ, theta, nC, mode='exit' ) / 10.0
        u = interp1d( eCZT, attenCZT, E2 )
        pExit = - u * rhoCZT * exitDepth

        pTotal = pEnter + pA + pA2B + pB + pExit
        p = np.array( [ pTotal, pEnter, pA, pA2B, pB, pExit ] )
        return p


cpdef np.ndarray[np.float_t, ndim=1] doubleProbabilityLineSource( double E0,
                                                            double eDepA, double Ax, double Ay, double Az,
                                                            double eDepB, double Bx, double By, double Bz,
                                                            double cameraX, double cameraY, double cameraZ,
                                                            int nC=8,
                                                            bint fullAbsorption=True ):
    '''
    p = doubleProbabilityLineSource(  double E0,
                                double eDepA, double Ax, double Ay, double Az,
                                double eDepB, double Bx, double By, double Bz,
                                double cameraX, double cameraY, double cameraZ,
                                int nC=8,
                                bint fullAbsorption=True )

    compute the probability of a double events, given the event order, without knowing
    where the source is.

    both entrance and exit optical depth are approximated.

    the probability is calculated based on a few pieces:
        pEnter, entrace probability, which is the probability the photon travel to the
        first interaction location--attenuation
        pA, probability of the first interaction, which is the KN given E0 and ( E0- eDepA )
        in terms of d_sigma/d_eDep
        pA2B, the probability of the photon travel from A to B and ends at B
        pB, the probability if the second interaction, use KN if B is Compton(d_sigma/d_eDep),
        otherwise, photoelectric
        pExit, the probability of the photon exit the crystal withou further interaction

    note, acctually, it's the natural log of the praboility is calcuated, and the
    constant, r_0 / 2, where r_0 is the classic radius of electron is dropped


    note while unites for most of the calculation is in mm, in this particular
    function, becasue the attenuation coeff's are in cm, the the unites are
    converted

    input:
        E0, sx, sy, sz:
            doulbe, E0 is the initial energy of the Gamma, sx, sy,and sz is source position
        eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz
            double, the deposited energy and coordinates of the two events,
            order is important--A is the first interaction, B is the second
        cameraX, cameraY, cameraZ:
            double, it's the first 3 paramters in par['det'], cameraX and cameraZ
            is coordinates of isocenter projected on camera surface, cameraY is
            the location of the camera front surface( close to the source)
        nC:
            int, number of directions exit optical depth is averaged over;
        fullAbsorption:
            boolean, if this is a full absorption
    return:
        p:
            numpy ndarray (dim=1), [ pTotal, pEnter, pA, pA2B, pB, pExit ]

    '''

    cdef:
        double      theta=0.0, d=0.0, E1=0.0, u=0.0, E2=0.0, d_eDepA=0.0, d_eDepB=0.0, de=0.0
        double      entranceDepth=0.0, exitDepth=0.0
        # double      alpha=0.0, nf=0.0
        double      pEnter=0.0, pA=0.0, pA2B=0.0, pB=0.0, pExit=0.0, pTotal=0.0
        np.ndarray[np.float64_t, ndim=1]  p=np.zeros( 6, dtype=np.float64 )

    d_eDepA = sqrt( 25e-6 + 2.35**2 * eDepA * 5e-6 )
    d_eDepB = sqrt( 25e-6 + 2.35**2 * eDepB * 5e-6 )
    de = sqrt( d_eDepA**2 + d_eDepB**2 )

    theta = scatterAngle( E0, eDepA )
    if theta < 0.0:
        pTotal = IMPOSSIBLE
        p = np.array( [ pTotal, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE ] )
        return p

    entranceDepth = opticalDepthApprox( Ax, Ay, Az, Bx, By, Bz, cameraX, cameraY, cameraZ, theta, nC, mode='entrance' ) / 10.0

    u = interp1d( eCZT, attenCZT, E0 )
    pEnter =   -u * rhoCZT * entranceDepth
    # p = 0.0

    E1 = E0 - eDepA
    if E1 < ZERO:
        pTotal = IMPOSSIBLE
        p = np.array( [ pTotal, pEnter, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE])
        return p


    pA = log( ( E1 / E0 )**2 * ( E1 / E0 + E0 / E1 - sin(theta)**2 ) * 2 * PI * 0.511 / E1**2 )
    # pA = pA - log( sin(theta) )

    # d = norm( Ax - Bx, Ay - By, Az - Bz ) / 10.0
    d = opticalDepth( Ax, Ay, Az, Bx, By, Bz, cameraX, cameraY, cameraZ ) / 10.0
    u = interp1d( eCZT, attenCZT, E1 )
    pA2B = - u * rhoCZT * d - log( sin(theta) )

    # if full absoption is assumed:
    if fullAbsorption:
        u = interp1d( eCZT, peCZT, eDepB )
        pB = log( u ) - ( E0 - eDepA - eDepB ) ** 2 / 2 / de**2 - 0.5 * log( 2 * PI * de**2 )
        pTotal = pEnter + pA + pA2B + pB + pExit
        p = np.array( [ pTotal, pEnter, pA, pA2B, pB, pExit ] )
        return p

    # compton-compton is assumed
    else:
        E2 = E1 - eDepB
        theta = scatterAngle( E1, eDepB )
        if theta  < 0.0:
            pTotal = IMPOSSIBLE
            p = np.array( [ pTotal, pEnter, pA, pA2B, IMPOSSIBLE, IMPOSSIBLE])
            return p

        pB = log( ( E2 / E1 )**2 * ( E2 / E1 + E1 / E2 - sin(theta)**2 ) * 2 * PI * 0.511 / E2**2 )

        exitDepth = opticalDepthApprox( Ax, Ay, Az, Bx, By, Bz, cameraX, cameraY, cameraZ, theta, nC, mode='exit' ) / 10.0
        u = interp1d( eCZT, attenCZT, E2 )
        pExit = - u * rhoCZT * exitDepth

        pTotal = pEnter + pA + pA2B + pB + pExit
        p = np.array( [ pTotal, pEnter, pA, pA2B, pB, pExit ] )
        return p



cpdef np.ndarray[np.float_t, ndim=1] tripleProbabilityPointSource(    double E0,
                                                                    double sx, double sy, double sz,
                                                                    double eDepA, double Ax, double Ay, double Az,
                                                                    double eDepB, double Bx, double By, double Bz,
                                                                    double eDepC, double Cx, double Cy, double Cz,
                                                                    double cameraX, double cameraY, double cameraZ,
                                                                    int nC=8,
                                                                    bint fullAbsorption=True ):
    '''
    p = dblEventsProbKnownSource( double E0,
                                    double sx, double sy, double sz,
                                    double eDepA, double Ax, double Ay, double Az,
                                    double eDepB, double Bx, double By, double Bz,
                                    double eDepC, double Cx, double Cy, double Cz,
                                    double cameraX, double cameraY, double cameraZ,
                                    int nC=8,
                                    bint fullAbsorption=True )

    compute the probability of a triple events, given the event order and source
    position.

    since the source position is known, the entrace OD is accurate: if the photon
    pass through more than one crystal before the first interaction, they are
    counted.

    the exit OD is approximated over 8 ( 8 is the default ) possilbe exiting
    directions.

    the probability is calculated based on a few pieces:
        pEnter, entrace probability, which is the probability the photon travel to the
        first interaction location--attenuation
        pA, probability of the first interaction, which is the KN given E0 and ( E0- eDepA )
        in terms of d_sigma/d_eDep
        pA2ffB, the probability of the photon travel from A to B and ends at B
        pB, the probability if the second interaction, use KN if B is Compton(d_sigma/d_eDep),
        pB2C, the probability of the photon travel from B to C and ends at C
        pC, the probability of the third interaction, use KN if B is Compton(d_sigma/d_eDep)
        otherwise, photoelectric
        pExit, the probability of the photon exit the crystal withou further interaction

    note, acctually, it's the natural log of the praboility is calcuated, and the
    constant, r_0 / 2, where r_0 is the classic radius of electron is dropped


    note while unites for most of the calculation is in mm, in this particular
    function, becasue the attenuation coeff's are in cm, the the unites are
    converted

    input:
        E0, sx, sy, sz:
            doulbe, E0 is the initial energy of the Gamma, sx, sy,and sz is source position
        eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz, eDepC, Cx, Cy, Cz
            double, the deposited energy and coordinates of the three interactions,
            order is important--A is the first interaction, B is the second, C is the last
        cameraX, cameraY, cameraZ:
            double, it's the first 3 paramters in par['det'], cameraX and cameraZ
            is coordinates of isocenter projected on camera surface, cameraY is
            the location of the camera front surface( close to the source)
        nC:
            int, number of directions exit optical depth is averaged over;
        fullAbsorption:
            boolean, if this is a full absorption
    return:
        p:
            numpy ndarray (dim=1), [ pTotal, pEnter, pA, pA2B, pB, pB2C, pC, pExit ]

    '''

    cdef:
        double      theta=0.0, d=0.0, E1=0.0, u=0.0, E2=0.0, E3=0.0, d_eDepA=0.0, d_eDepB=0.0, d_eDepC=0.0, de=0.0
        double      entranceDepth=0.0, exitDepth=0.0
        # double      alpha=0.0, nf=0.0
        double      pEnter=0.0, pA=0.0, pA2B=0.0, pB=0.0, pB2C=0.0, pC=0.0, pExit=0.0, pTotal=0.0
        np.ndarray[np.float64_t, ndim=1]  p=np.zeros( 8, dtype=np.float64 )

    d_eDepA = sqrt( 25e-6 + 2.35**2 * eDepA * 5e-6 )
    d_eDepB = sqrt( 25e-6 + 2.35**2 * eDepB * 5e-6 )
    d_eDepC = sqrt( 25e-6 + 2.35**2 * eDepC * 5e-6 )
    de = sqrt( d_eDepA**2 + d_eDepB**2 + d_eDepC**2 )

    entranceDepth = opticalDepth( sx, sy, sz, Ax, Ay, Az, cameraX, cameraY, cameraZ ) / 10.0

    u = interp1d( eCZT, attenCZT, E0 )
    pEnter =   -u * rhoCZT * entranceDepth

    E1 = E0 - eDepA
    if E1 < ZERO:
        pTotal = IMPOSSIBLE
        p = np.array( [ pTotal, pEnter, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE])
        return p
    theta = scatterAngle( E0, eDepA )
    if theta < 0.0:
        pTotal = IMPOSSIBLE
        p = np.array( [ pTotal, pEnter, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE])
        return p
    pA = log( ( E1 / E0 )**2 * ( E1 / E0 + E0 / E1 - sin(theta)**2 ) * 2 * PI * 0.511 / E1**2 )

    d = opticalDepth( Ax, Ay, Az, Bx, By, Bz, cameraX, cameraY, cameraZ ) / 10.0
    u = interp1d( eCZT, attenCZT, E1 )
    pA2B = - u * rhoCZT * d - log( sin(theta) * d )

    E2 = E0 - eDepA - eDepB
    if E2 < ZERO:
        pTotal = IMPOSSIBLE
        p = np.array( [ pTotal, pEnter, pA, pA2B, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE])
        return p
    theta = scatterAngle( E1, eDepB )
    if theta < 0.0:
        pTotal = IMPOSSIBLE
        p = np.array( [ pTotal, pEnter, pA, pA2B, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE])
        return p
    pB = log( ( E2 / E1 )**2 * ( E2 / E1 + E1 / E2 - sin(theta)**2 ) * 2 * PI * 0.511 / E2**2 )

    d = opticalDepth( Bx, By, Bz, Cx, Cy, Cz, cameraX, cameraY, cameraZ ) / 10.0
    u = interp1d( eCZT, attenCZT, E2 )
    pB2C = - u * rhoCZT * d - log( sin(theta) * d )

    # if full absoption is assumed:
    if fullAbsorption:
        u = interp1d( eCZT, peCZT, eDepC )
        pB = log( u ) - ( E0 - eDepA - eDepB - eDepC ) ** 2 / 2 / de**2 - 0.5 * log( 2 * PI * de**2 )
        pTotal = pEnter + pA + pA2B + pB + pB2C + pC + pExit
        p = np.array( [ pTotal, pEnter, pA, pA2B, pB, pB2C, pC, pExit ] )
        return p

    # compton-compton is assumed
    else:
        E3 = E2 - eDepC
        if E3 < 0.0:
            pTotal = IMPOSSIBLE
            p = np.array( [ pTotal, pEnter, pA, pA2B, pB, pB2C, IMPOSSIBLE, IMPOSSIBLE])
            return p
        theta = scatterAngle( E2, eDepC )
        if theta  < 0.0:
            pTotal = IMPOSSIBLE
            p = np.array( [ pTotal, pEnter, pA, pA2B, pB, pB2C, IMPOSSIBLE, IMPOSSIBLE])
            return p
        pC = log( ( E3 / E2 )**2 * ( E3 / E2 + E2 / E3 - sin(theta)**2 ) * 2 * PI * 0.511 / E3**2 )

        exitDepth = opticalDepthApprox( Bx, By, Bz, Cx, Cy, Cz, cameraX, cameraY, cameraZ, theta, nC, mode='exit' ) / 10.0
        u = interp1d( eCZT, attenCZT, E3 )
        pExit = - u * rhoCZT * exitDepth

        pTotal = pEnter + pA + pA2B + pB + pB2C + pC + pExit
        p = np.array( [ pTotal, pEnter, pA, pA2B, pB, pB2C, pC, pExit ] )
        return p


cpdef np.ndarray[np.float_t, ndim=1] tripleProbabilityLineSource( double E0,
                                                            double eDepA, double Ax, double Ay, double Az,
                                                            double eDepB, double Bx, double By, double Bz,
                                                            double eDepC, double Cx, double Cy, double Cz,
                                                            double cameraX, double cameraY, double cameraZ,
                                                            int nC=8,
                                                            bint fullAbsorption=True ):
    '''
    p = tripleProbabilityLineSource(  double E0,
                                double eDepA, double Ax, double Ay, double Az,
                                double eDepB, double Bx, double By, double Bz,
                                double eDepC, double Cx, double Cy, double Cz,
                                double cameraX, double cameraY, double cameraZ,
                                int nC=8,
                                bint fullAbsorption=True )

    compute the probability of a double events, given the event order, without knowing
    where the source is.

    both entrance and exit optical depth are approximated.

    the probability is calculated based on a few pieces:
        pEnter, entrace probability, which is the probability the photon travel to the
        first interaction location--attenuation
        pA, probability of the first interaction, which is the KN given E0 and ( E0- eDepA )
        in terms of d_sigma/d_eDep
        pA2B, the probability of the photon travel from A to B and ends at B
        pB, the probability if the second interaction, use KN if B is Compton(d_sigma/d_eDep),
        pB2C,the probability of the photon travel from B to C and ends at C
        pC,the probability if the second interaction, use KN if B is Compton(d_sigma/d_eDep),
        otherwise, photoelectric
        pExit, the probability of the photon exit the crystal withou further interaction

    note, acctually, it's the natural log of the praboility is calcuated, and the
    constant, r_0 / 2, where r_0 is the classic radius of electron is dropped


    note while unites for most of the calculation is in mm, in this particular
    function, becasue the attenuation coeff's are in cm, the the unites are
    converted

    input:
        E0, sx, sy, sz:
            doulbe, E0 is the initial energy of the Gamma, sx, sy,and sz is source position
        eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz, eDepC, Cx, Cy, Cz
            double, the deposited energy and coordinates of the two events,
            order is important--A is the first interaction, B is the second, C is the last
        cameraX, cameraY, cameraZ:
            double, it's the first 3 paramters in par['det'], cameraX and cameraZ
            is coordinates of isocenter projected on camera surface, cameraY is
            the location of the camera front surface( close to the source)
        nC:
            int, number of directions exit optical depth is averaged over;
        fullAbsorption:
            boolean, if this is a full absorption
    return:
        p:
            numpy ndarray (dim=1), [ pTotal, pEnter, pA, pA2B, pB, pB2C, pC pExit ]

    '''

    cdef:
        double      theta=0.0, d=0.0, E1=0.0, u=0.0, E2=0.0, E3=0.0, d_eDepA=0.0, d_eDepB=0.0, d_eDepC=0.0, de=0.0
        double      entranceDepth=0.0, exitDepth=0.0
        # double      alpha=0.0, nf=0.0
        double      pEnter=0.0, pA=0.0, pA2B=0.0, pB=0.0, pExit=0.0, pTotal=0.0
        np.ndarray[np.float64_t, ndim=1]  p=np.zeros( 8, dtype=np.float64 )

    d_eDepA = sqrt( 25e-6 + 2.35**2 * eDepA * 5e-6 )
    d_eDepB = sqrt( 25e-6 + 2.35**2 * eDepB * 5e-6 )
    d_eDepC = sqrt( 25e-6 + 2.35**2 * eDepC * 5e-6 )
    de = sqrt( d_eDepA**2 + d_eDepB**2 + d_eDepC**2 )

    theta = scatterAngle( E0, eDepA )
    if theta < 0.0:
        pTotal = IMPOSSIBLE
        p = np.array( [ pTotal, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE ] )
        return p

    entranceDepth = opticalDepthApprox( Ax, Ay, Az, Bx, By, Bz, cameraX, cameraY, cameraZ, theta, nC, mode='entrance' ) / 10.0
    u = interp1d( eCZT, attenCZT, E0 )
    pEnter =   -u * rhoCZT * entranceDepth

    E1 = E0 - eDepA
    if E1 < ZERO:
        pTotal = IMPOSSIBLE
        p = np.array( [ pTotal, pEnter, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE])
        return p
    pA = log( ( E1 / E0 )**2 * ( E1 / E0 + E0 / E1 - sin(theta)**2 ) * 2 * PI * 0.511 / E1**2 )

    d = opticalDepth( Ax, Ay, Az, Bx, By, Bz, cameraX, cameraY, cameraZ ) / 10.0
    u = interp1d( eCZT, attenCZT, E1 )
    pA2B = - u * rhoCZT * d - log( sin(theta) * d )

    E2 = E1 - eDepB
    if E2 < ZERO:
        pTotal = IMPOSSIBLE
        p = np.array( [ pTotal, pEnter, pA, pA2B, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE])
        return p
    theta = scatterAngle( E1, eDepB )
    if theta < 0.0:
        pTotal = IMPOSSIBLE
        p = np.array( [ pTotal, pEnter, pA, pA2B, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE, IMPOSSIBLE])
        return p
    pB = log( ( E2 / E1 )**2 * ( E2 / E1 + E1 / E2 - sin(theta)**2 ) * 2 * PI * 0.511 / E2**2 )

    d = opticalDepth( Bx, By, Bz, Cx, Cy, Cz, cameraX, cameraY, cameraZ ) / 10.0
    u = interp1d( eCZT, attenCZT, E2 )
    pB2C = - u * rhoCZT * d - log( sin(theta) * d )

    # if full absoption is assumed:
    if fullAbsorption:
        u = interp1d( eCZT, peCZT, eDepC )
        pC = log( u ) - ( E0 - eDepA - eDepB - eDepC ) ** 2 / 2 / de**2 - 0.5 * log( 2 * PI * de**2 )
        pTotal = pEnter + pA + pA2B + pB + pB2C + pC + pExit
        p = np.array( [ pTotal, pEnter, pA, pA2B, pB, pB2C, pC, pExit ] )
        return p

    # compton-compton is assumed
    else:
        E3 = E2 - eDepC
        if E3 < 0.0:
            pTotal = IMPOSSIBLE
            p = np.array( [ pTotal, pEnter, pA, pA2B, pB, pB2C, IMPOSSIBLE, IMPOSSIBLE])
            return p
        theta = scatterAngle( E2, eDepC )
        if theta  < 0.0:
            pTotal = IMPOSSIBLE
            p = np.array( [ pTotal, pEnter, pA, pA2B, pB, pB2C, IMPOSSIBLE, IMPOSSIBLE])
            return p
        pC = log( ( E3 / E2 )**2 * ( E3 / E2 + E2 / E3 - sin(theta)**2 ) * 2 * PI * 0.511 / E3**2 )

        exitDepth = opticalDepthApprox( Bx, By, Bz, Cx, Cy, Cz, cameraX, cameraY, cameraZ, theta, nC, mode='exit' ) / 10.0
        u = interp1d( eCZT, attenCZT, E3 )
        pExit = - u * rhoCZT * exitDepth

        pTotal = pEnter + pA + pA2B + pB + pB2C + pC + pExit
        p = np.array( [ pTotal, pEnter, pA, pA2B, pB, pB2C, pC, pExit ] )
        return p



cpdef float imageEntropy( np.ndarray[np.float_t, ndim=2] h ):
    '''
    H = imageEntropy( h )

    compute the entropy of the image h
    '''
    cdef:
        int     M=h.shape[0]-1, N=h.shape[1]-1, i=0, j=0, m=0, n=0
        float   H=0.0
        np.ndarray[np.int_t, ndim=2]  hdiffx=np.zeros( [M, N-1], dtype=np.int )
        np.ndarray[np.int_t, ndim=2]  hidffy=np.zeros( [M-1, N], dtype=np.int )
        np.ndarray[np.float_t, ndim=2]  p=np.zeros( [511, 511], dtype=np.float )

    start_time = time.time()
    h = np.round( ( h - h.min() ) / ( h - h.min() ).max() * 255  )
    hdiffx = np.diff( h, axis=1 ).astype( np.int )
    hdiffy = np.diff( h, axis=0 ).astype( np.int )
    print( 'rounding and differential took ', time.time() - start_time )

    for m in range( M ):
        for n in range( N ):
            p[hdiffx[m,n]+255, hdiffy[m,n]+255] += 1

    print( 'sum of p ', p.sum() )
    p = p / p.sum()
    print( 'rounding, differential and binning took ', time.time() - start_time )
    H = -( p[p>0] * np.log( p[p>0] ) ).sum()
    end_time = time.time()
    print( 'time to compute entropy ', end_time - start_time)

    return H


    # def getDoubles(    np.ndarray[np.int64_t, ndim=1] npx,
cpdef np.ndarray[np.float_t, ndim=2] getDoubles(    np.ndarray[np.float_t, ndim=1] npx,
                                                    np.ndarray[np.float_t, ndim=1] mdl,
                                                    np.ndarray[np.float_t, ndim=1] edep,
                                                    np.ndarray[np.float_t, ndim=1] x,
                                                    np.ndarray[np.float_t, ndim=1] y,
                                                    np.ndarray[np.float_t, ndim=1] z,
                                                    np.ndarray[np.float_t, ndim=1] t,
                                                    par ):
    '''
    events = getDoubles( npx,mdl, edep, x, y, z, t, par )

    look though those MARKED AS DOUBLES in the data read in from events file, according to settings in par
    and find good events:
        first, check if dca is less than the dcaThreshold,

    input:
        npx,mdl, edep, x, y, z, t:
            each of them is a 1-D numpy array, these are the output from readInC
        par:
            python dictionay, this contains information read from the setting file,
            it's the output readPar
    output:
        event:
            numpay array, following the format for CORE reconstruction
            the first 8 collumns are eDepA, xA, yA, zA, eDepB, xB, yB, zB, in the order of determined events order
            9th collumn:  the E0 that gives the shortest DCA AND highest probabiltiy
                          (when more than one emission lines given )
            10th colummn: the shortest DCA
            11th collumn: the highest probability
            12th collumn: the determined events order: 0--(i, i+1); 1--(i+1, i),
                          where i and i+1 is the line index of these two interactions
                          in AllEventsCombined.txt file
            13 and 14th:  module number of i and i+1 (order in AllEventsCombined,
                          NOT the determined events orer )
            15 and 16th:  time stamp of i and i+1
            17th colummn: line index of i in AllEventsCombined


    '''

    cdef:
        int     i=0, N=0
        float   cameraX=0.0, cameraY=0.0, cameraZ=0.0, cameraAngle=0.0
        float   eDepA=0.0, Ax=0.0, Ay=0.0, Az=0.0
        float   eDepB=0.0, Bx=0.0, By=0.0, Bz=0.0
        float   sx=0.0, sy=0.0, sz=0.0
        float   E0=0.0, dcaThreshold=0.0
        float   thisDca=0.0, betterDca=1000.0, betterProb=-1000.0, betterE0=0.0
        # float   thisDca12=0.0, thisDca21=0.0
        int     betterOrder=-1
        float   coneAngle=0.0

        float   boundaryX1=0.0, boundaryX2=0.0, boundaryY1=0.0, boundaryY2=0.0, boundaryZ1=0.0, boundaryZ2=0.0
        bint    singleModule=False, singlePlane=False
        int     moduleNumber=0, planeNumber=0
        int     nODApproximalPoints=0
        float   slicePosition=0.0, gantryAngle=90.0, couchAngle=0.0
        bint    pointSource=False, beamMeasurement=False, rotateGantry=False, rotateCouch=False, timeStampOnly=False
        bint    useEmissionLines=False, useEdepSum=False
        bint    dopplerCorrection=False
        bint    fullAbsorption=False, fullAbsorptionOnly=False
        float   fullAbsorptionThreshold=0.0, timeWindow=0.0
        int     nInteractions=0
        bint    proceed = False, boundaryExceeded = False
        float   tempY=.0, tempZ=0.0
        np.ndarray[np.float_t, ndim=1] gammaE=np.zeros( par['nGamma'], dtype=np.float )
        np.ndarray[np.float_t, ndim=1] thisProb=np.zeros( 6, dtype=np.float)
        # np.ndarray[np.float_t, ndim=1] thisProb12=np.zeros( 6, dtype=np.float)
        # np.ndarray[np.float_t, ndim=1] thisProb21=np.zeros( 6, dtype=np.float)
        np.ndarray coord=np.array([])

    events = []

    start_time = time.time()
    # totalDouble = 0
    # possibleDouble = 0
    # goodDouble = 0

    if not par['doubles']:
        print( 'the parameter file is not set to use doubles' )
        return np.zeros( ( 1, 17 ) )

    N = len( npx )

    # unpacking
    cameraX, cameraY, cameraZ, cameraAngle = par['det']
    gantryAngle = par['gantryAngle']
    couchAngle = par['couchAngle']
    dcaThreshold = par['dcaThreshold']
    boundaryX1, boundaryX2 = par['boundaryX']
    boundaryY1, boundaryY2 = par['boundaryY']
    boundaryZ1, boundaryZ2 = par['boundaryZ']

    if par['pointSource']:
        pointSource = True
        sx, sy, sz = par['pointSourcePosition']
    if par['beamMeasurement']:
        beamMeasurement = True
        slicePosition = par['slicePosition']
    if pointSource and beamMeasurement:
        print( 'must be eigher point source or beam measurements, can not be both')
        return np.zeros( ( 1, 17 ) )
    nODApproximalPoints = par['nODApproximalPoints']

    if par['useEmissionLines']:
        useEmissionLines = True
        gammaE = par['gammaE']
    if par['useEdepSum']:
        useEdepSum = True
    if useEmissionLines and useEdepSum:
        print( 'can not use both emission lines and eDep sum')
        return np.zeros( ( 1, 17 ) )

    if par['fullAbsorptionOnly']:
        fullAbsorptionOnly = True
    fullAbsorptionThreshold = par['fullAbsorptionThreshold']

    if par['singlePlane']:
        singlePlane = True
        planeNumber = par['planeNumber']
    if par['singleModule']:
        singleModule = True
        moduleNumber = par['moduleNumber']

    if par['dopplerCorrection']:
        dopplerCorrection = True

    if par['timeStampOnly']:
        timeStampOnly = True
        timeWindow = par['timeWindow'] / 10.0

    if fabs( couchAngle ) > ZERO:
        rotateCouch = True
        couchAngle = couchAngle / 180 * PI

    if fabs( gantryAngle - 270.0 ) > ZERO:
        rotateGantry = True
        gantryAngle = gantryAngle / 180.0 * PI
    # print( 'total ', N )
    while i < N-1:
        proceed = False
        if timeStampOnly:
            nInteractions=1
            while t[i+nInteractions] - t[i] < timeWindow:
                nInteractions += 1
                if i + nInteractions >= N:
                    # print( 'next pointer ', i + nInteractions )
                    boundaryExceeded = True
                    break
            if boundaryExceeded:
                # print( 'should exit while now ')
                break
            if nInteractions != 2:
                i = i + nInteractions
                continue
            else:
                proceed = True
        else:
            if npx[i] == 2:
                if npx[i+1] == 2:
                    if t[i+1] == t[i]:
                        proceed = True
                    else:
                        i += 1
                        continue
                else:
                    i += 1
                    continue
            else:
                i += 1
                continue

        if proceed:
            # two pixel events found, process this 2-pixel events
            # clean house
            thisDca = 1000.0
            thisProb = np.ones( 6 ) * ( -1000.0 )
            betterDca = 1000.0
            betterProb = -1000.0
            betterOrder = -1
            betterE0 = 0.0

            # in the following cases, skip this 2-pixle events
            if singleModule and ( ( mdl[i] != moduleNumber ) or ( mdl[i+1] != moduleNumber ) ):
                i = i + 2
                # print( 'skipped due to single module')
                continue
            if singlePlane:
                if planeNumber == 0 and ( ( mdl[i] > 7 ) or ( mdl[i+1] > 7 ) ):
                    i = i + 2
                    # print( 'skipped due to single plane')
                    continue
                elif planeNumber == 1 and ( ( mdl[i] < 8 ) or ( mdl[i+1] < 8 ) ):
                    i = i + 2
                    # print( 'skipped due to single plane')
                    continue
            if edep[i] == 0.0 or edep[i+1] == 0.0:
                i = i + 2
                # print( 'skipped due to zero energy deposit ')
                continue

            # process this 2-pixel events
            if pointSource:
                eDepA = edep[i]
                Ax = x[i]
                Ay = y[i]
                Az = z[i]
                eDepB = edep[i+1]
                Bx = x[i+1]
                By = y[i+1]
                Bz = z[i+1]
                if useEmissionLines:
                    # print( 'pointSource, use useEmissionLines')
                    for E0 in gammaE:
                        # print( 'pointSource, use useEmissionLines, looping E0, current E0 is ', E0)
                        fullAbsorption = fabs(  ( eDepA + eDepB ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            continue
                        coneAngle = scatterAngle( E0, eDepA )
                        if coneAngle > 0.0 and ( fullAbsorption or ( (not fullAbsorption) and  ( scatterAngle( E0 - eDepA, eDepB ) > 0.0 ) ) ):
                            thisDca = pointDCAVector( Ax, Ay, Az, Bx, By, Bz, sx, sy, sz, coneAngle )
                            if fabs( thisDca ) < dcaThreshold:
                                thisProb =  doubleProbabilityPointSource(   E0,
                                                                            sx, sy, sz,
                                                                            eDepA, Ax, Ay, Az,
                                                                            eDepB, Bx, By, Bz,
                                                                            cameraX, cameraY, cameraZ,
                                                                            nODApproximalPoints, fullAbsorption )
                                if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                    betterDca = thisDca
                                    betterProb = thisProb[0]
                                    betterE0 = E0
                                    betterOrder = 0

                        coneAngle = scatterAngle( E0, eDepB )
                        if coneAngle > 0.0  and ( fullAbsorption or ( (not fullAbsorption) and  ( scatterAngle( E0 - eDepB, eDepA ) > 0.0 ) ) ):
                            thisDca = pointDCAVector( Bx, By, Bz, Ax, Ay, Az, sx, sy, sz, coneAngle )
                            if fabs( thisDca ) < dcaThreshold:
                                thisProb =  doubleProbabilityPointSource(   E0,
                                                                            sx, sy, sz,
                                                                            eDepB, Bx, By, Bz,
                                                                            eDepA, Ax, Ay, Az,
                                                                            cameraX, cameraY, cameraZ,
                                                                            nODApproximalPoints, fullAbsorption )
                                if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                    betterDca = thisDca
                                    betterProb = thisProb[0]
                                    betterE0 = E0
                                    betterOrder = 1

                    if betterProb > -800:
                        if betterOrder == 0:
                            events.append( np.array( [  eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], t[i], t[i+1], i ] ) )
                        elif betterOrder == 1:
                            events.append( np.array( [  eDepB, Bx, By, Bz, eDepA, Ax, Ay, Az,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], t[i], t[i+1], i ] ) )

                elif useEdepSum:
                    E0 = eDepA + eDepB
                    fullAbsorption = True
                    coneAngle = scatterAngle( E0, eDepA )
                    if coneAngle > 0:
                        thisDca = pointDCAVector( Ax, Ay, Az, Bx, By, Bz, sx, sy, sz, coneAngle )
                        if fabs( thisDca ) < dcaThreshold:
                            thisProb =  doubleProbabilityPointSource(   E0,
                                                                        sx, sy, sz,
                                                                        eDepA, Ax, Ay, Az,
                                                                        eDepB, Bx, By, Bz,
                                                                        cameraX, cameraY, cameraZ,
                                                                        nODApproximalPoints, fullAbsorption )
                            if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                betterDca = thisDca
                                betterProb = thisProb[0]
                                betterE0 = E0
                                betterOrder = 0

                    coneAngle = scatterAngle( E0, eDepB )
                    if coneAngle > 0:
                        thisDca = pointDCAVector( Bx, By, Bz, Ax, Ay, Az, sx, sy, sz, coneAngle )
                        if fabs( thisDca ) < dcaThreshold:
                            thisProb =  doubleProbabilityPointSource(   E0,
                                                                        sx, sy, sz,
                                                                        eDepB, Bx, By, Bz,
                                                                        eDepA, Ax, Ay, Az,
                                                                        cameraX, cameraY, cameraZ,
                                                                        nODApproximalPoints, fullAbsorption )
                            if ( fabs( thisDca )  <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                betterProb = thisProb[0]
                                betterE0 = E0
                                betterOrder = 1

                    if betterProb > -800:
                        if betterOrder == 0:
                            events.append( np.array( [  eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], t[i], t[i+1], i ] ) )
                        elif betterOrder == 1:
                            events.append( np.array( [  eDepB, Bx, By, Bz, eDepA, Ax, Ay, Az,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], t[i], t[i+1], i ] ) )

            elif beamMeasurement:
                # for beam measurements, need to compute the dca to z axis,
                # the z axis is the beam axis, so, when computing dca, need
                # to use rotated events coordinates:
                # print( 'in beamMeasurement')
                if rotateCouch:
                    eDepA = edep[i]
                    Ax = x[i] * cos( couchAngle ) - z[i] * sin( couchAngle )
                    Ay = y[i]
                    Az = x[i] * sin( couchAngle ) + z[i] * cos( couchAngle )
                    eDepB = edep[i+1]
                    Bx = x[i+1] * cos( couchAngle ) - z[i+1] * sin( couchAngle )
                    By = y[i+1]
                    Bz = x[i+1] * sin( couchAngle ) + z[i+1] * cos( couchAngle )
                    if rotateGantry:
                        tempY = Ay
                        tempZ = Az
                        Ay = -tempY * sin( gantryAngle ) - tempZ * cos( gantryAngle )
                        Az =  tempY * cos( gantryAngle ) - tempZ * sin( gantryAngle )
                        tempY = By
                        tempZ = Bz
                        By = -tempY * sin( gantryAngle ) - tempZ * cos( gantryAngle )
                        Bz =  tempY * cos( gantryAngle ) - tempZ * sin( gantryAngle )
                else:
                    eDepA = edep[i]
                    Ax = x[i]
                    Ay = y[i]
                    Az = z[i]
                    eDepB = edep[i+1]
                    Bx = x[i+1]
                    By = y[i+1]
                    Bz = z[i+1]
                    if rotateGantry:
                        tempY = Ay
                        tempZ = Az
                        Ay = -tempY * sin( gantryAngle ) - tempZ * cos( gantryAngle )
                        Az =  tempY * cos( gantryAngle ) - tempZ * sin( gantryAngle )
                        tempY = By
                        tempZ = Bz
                        By = -tempY * sin( gantryAngle ) - tempZ * cos( gantryAngle )
                        Bz =  tempY * cos( gantryAngle ) - tempZ * sin( gantryAngle )

                if useEmissionLines:
                    # print( 'in beamMeasurement, useEmissionLines')
                    for E0 in gammaE:
                        # print( ' in beamMeasurement, useEmissionLines, looping E0, current E0 is ', E0)
                        fullAbsorption = fabs(  ( eDepA + eDepB ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            continue

                        coneAngle = scatterAngle( E0, eDepA )
                        if coneAngle > 0.0 and ( fullAbsorption or ( ( not fullAbsorption ) and ( scatterAngle( E0 - eDepA, eDepB ) > 0.0 ) ) ):
                            thisDca,coord = zAxisDCA(   Ax, Ay, Az, Bx, By, Bz,
                                                        coneAngle, slicePosition,
                                                        boundaryX1, boundaryX2,
                                                        boundaryY1, boundaryY2,
                                                        boundaryZ1, boundaryZ2 )
                            if fabs( thisDca ) < dcaThreshold:
                                thisProb =  doubleProbabilityLineSource(    E0,
                                                                            eDepA, Ax, Ay, Az,
                                                                            eDepB, Bx, By, Bz,
                                                                            cameraX, cameraY, cameraZ,
                                                                            nODApproximalPoints, fullAbsorption )
                                if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                    betterDca = thisDca
                                    betterProb = thisProb[0]
                                    betterE0 = E0
                                    betterOrder = 0

                        coneAngle = scatterAngle( E0, eDepB )
                        if coneAngle > 0.0 and ( fullAbsorption or ( ( not fullAbsorption ) and ( scatterAngle( E0 - eDepB, eDepA ) > 0 ) ) ):
                            thisDca, coord = zAxisDCA(  Bx, By, Bz, Ax, Ay, Az,
                                                        coneAngle, slicePosition,
                                                        boundaryX1, boundaryX2,
                                                        boundaryY1, boundaryY2,
                                                        boundaryZ1, boundaryZ2 )
                            if fabs( thisDca ) < dcaThreshold:
                                thisProb =  doubleProbabilityLineSource(    E0,
                                                                            eDepB, Bx, By, Bz,
                                                                            eDepA, Ax, Ay, Az,
                                                                            cameraX, cameraY, cameraZ,
                                                                            nODApproximalPoints, fullAbsorption )
                                if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                    betterDca = thisDca
                                    betterProb = thisProb[0]
                                    betterE0 = E0
                                    betterOrder = 1

                    if betterProb > -800:
                        if betterOrder == 0:
                            events.append( np.array( [  eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], t[i], t[i+1], i ] ) )
                        elif betterOrder == 1:
                            events.append( np.array( [  eDepB, Bx, By, Bz, eDepA, Ax, Ay, Az,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], t[i], t[i+1], i ] ) )

                elif useEdepSum:
                    E0 = eDepA + eDepB
                    fullAbsorption = True
                    coneAngle = scatterAngle( E0, eDepA )
                    if coneAngle > 0:
                        thisDca, coord = zAxisDCA(  Ax, Ay, Az, Bx, By, Bz,
                                                    coneAngle, slicePosition,
                                                    boundaryX1, boundaryX2,
                                                    boundaryY1, boundaryY2,
                                                    boundaryZ1, boundaryZ2 )
                        if fabs( thisDca ) < dcaThreshold:
                            thisProb =  doubleProbabilityLineSource(    E0,
                                                                        eDepA, Ax, Ay, Az,
                                                                        eDepB, Bx, By, Bz,
                                                                        cameraX, cameraY, cameraZ,
                                                                        nODApproximalPoints, fullAbsorption )
                            if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                betterDca = thisDca
                                betterProb = thisProb[0]
                                betterE0 = E0
                                betterOrder = 0

                    coneAngle = scatterAngle( E0, eDepB )
                    if coneAngle > 0.0:
                        thisDca, coord = zAxisDCA(  Bx, By, Bz, Ax, Ay, Az,
                                                    coneAngle, slicePosition,
                                                    boundaryX1, boundaryX2,
                                                    boundaryY1, boundaryY2,
                                                    boundaryZ1, boundaryZ2 )
                        if fabs( thisDca ) < dcaThreshold:
                            thisProb =  doubleProbabilityLineSource(    E0,
                                                                        eDepB, Bx, By, Bz,
                                                                        eDepA, Ax, Ay, Az,
                                                                        cameraX, cameraY, cameraZ,
                                                                        nODApproximalPoints, fullAbsorption )
                            if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                betterDca = thisDca
                                betterProb = thisProb[0]
                                betterE0 = E0
                                betterOrder = 1

                    if betterProb > -800:
                        if betterOrder == 0:
                            events.append( np.array( [  eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], t[i], t[i+1], i ] ) )
                        elif betterOrder == 1:
                            events.append( np.array( [  eDepB, Bx, By, Bz, eDepA, Ax, Ay, Az,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], t[i], t[i+1], i ] ) )

        # done processing this 2-pixel events, moving index down by 2
        i = i + 2

    events = np.array( events )
    end_time = time.time()
    print( 'time used ', end_time - start_time )
    return events


cpdef np.ndarray[np.float_t, ndim=2] getTriples(    np.ndarray[np.int64_t, ndim=1] npx,
                                                    np.ndarray[np.int64_t, ndim=1] mdl,
                                                    np.ndarray[np.float_t, ndim=1] edep,
                                                    np.ndarray[np.float_t, ndim=1] x,
                                                    np.ndarray[np.float_t, ndim=1] y,
                                                    np.ndarray[np.float_t, ndim=1] z,
                                                    np.ndarray[np.float_t, ndim=1] t,
                                                    par ):
    '''
    events = getTriples( npx,mdl, edep, x, y, z, t, par )

    look through those MARKED AS TRIPLE and find good events, according to settings in par.

    input:
        npx,mdl, edep, x, y, z, t:
            each of them is a 1-D numpy array, these are the output from readInC
        par:
            python dictionay, this contains information read from the setting file,
            it's the output readPar
    output:
        event:
            numpay array, following the format for CORE reconstruction
            the first 12 collumns are eDepA, xA, yA, zA, eDepB, xB, yB, zB, eDepC, xC, yC, zC
            in the order of determined events order
            13th collumn:  the E0 that gives the shortest DCA AND highest probabiltiy
                           (when more than one emission lines given or if useTripleE0, the
                            computed E0 for the determined order )
            14th colummn: the shortest DCA
            15th collumn: the highest probability
            16th collumn: the determined events order: 0--(i, i+1, i+2); 1--(i, i+2, i+1),
                          2--(i+1, i, i+2 ), 3--(i+1, i+2, i), 4--(i+2, i, i+1)
                          5--(i+2, i+1, i)
                          where i, i+1 and i+2 are the line index of these three interactions
                          in AllEventsCombined.txt file
            17 and 19th:  module number of i to i+2 (order in AllEventsCombined,
                          NOT the determined events orer )
            20 and 22th:  time stamp of i to i+2
            13th colummn: line index of i in AllEventsCombined

    '''

    # NOTE: there are three interactions now, so there are 6 possible ordering:
    # 0--ABC;   1--ACB;     2--BAC;     3--BCA;     4--CAB;     5--CBA

    cdef:
        int     i=0, N=0
        float   cameraX=0.0, cameraY=0.0, cameraZ=0.0, cameraAngle=0.0
        float   eDepA=0.0, Ax=0.0, Ay=0.0, Az=0.0
        float   eDepB=0.0, Bx=0.0, By=0.0, Bz=0.0
        float   eDepC=0.0, Cx=0.0, Cy=0.0, Cz=0.0
        float   sx=0.0, sy=0.0, sz=0.0
        float   E0=0.0, dcaThreshold=0.0
        float   thisDca=0.0, betterDca=1000.0, betterProb=-1000.0, betterE0=0.0
        int     betterOrder=-1
        float   coneAngle=0.0
        float   boundaryX1=0.0, boundaryX2=0.0, boundaryY1=0.0, boundaryY2=0.0, boundaryZ1=0.0, boundaryZ2=0.0
        bint    singleModule=False, singlePlane=False
        int     moduleNumber=0, planeNumber=0
        int     nODApproximalPoints=0
        float   slicePosition=0.0, gantryAngle=90.0, couchAngle=0.0
        bint    pointSource=False, beamMeasurement=False, rotateGantry=False, rotateCouch=False
        bint    useEmissionLines=False, useEdepSum=False, useTripleE0=False
        bint    dopplerCorrection=False
        bint    fullAbsorption=False, fullAbsorptionOnly=False, timeStampOnly=False
        float   fullAbsorptionThreshold=0.0, timeWindow=0.0
        int     nInteractions=0
        bint    proceed=False, boundaryExceeded = False
        float   tempY=0.0, tempZ=0.0
        np.ndarray[np.float_t, ndim=1] gammaE=np.zeros( par['nGamma'], dtype=np.float )
        np.ndarray[np.float_t, ndim=1] thisProb=np.zeros( 8, dtype=np.float)
        np.ndarray coord=np.array([])

    start_time = time.time()
    if not par['triples']:
        print( 'the parameter file is not set to use triples' )
        return np.zeros( ( 1, 8 ) )

    events = []
    N = len( npx )

    # unpacking
    # the following are always needed:
    cameraX, cameraY, cameraZ, cameraAngle = par['det']
    gantryAngle = par['gantryAngle']
    couchAngle = par['couchAngle']
    dcaThreshold = par['dcaThreshold']
    boundaryX1, boundaryX2 = par['boundaryX']
    boundaryY1, boundaryY2 = par['boundaryY']
    boundaryZ1, boundaryZ2 = par['boundaryZ']
    nODApproximalPoints = par['nODApproximalPoints']
    fullAbsorptionThreshold = par['fullAbsorptionThreshold']

    # the following are conditional:
    if par['pointSource']:
        pointSource = True
        sx, sy, sz = par['pointSourcePosition']
    if par['beamMeasurement']:
        beamMeasurement = True
        slicePosition = par['slicePosition']
    if pointSource and beamMeasurement:
        print( 'must be eigher point source or beam measurements, can not be both')
        return np.zeros( (1, 8 ) )

    if par['useTripleE0']:
        useTripleE0 = True
    if par['useEmissionLines']:
        useEmissionLines = True
        gammaE = par['gammaE']
    if par['useEdepSum']:
        useEdepSum = True

    if useEmissionLines and useEdepSum:
        print( 'can not use both emission lines and eDep sum')
        return np.zeros( (1, 8 ) )
    if useEmissionLines and useTripleE0:
        print( 'can not use both emission lines and triple E0')
        return np.zeros( (1, 8 ) )
    if useEdepSum and useTripleE0:
        print( 'can not use both eDep sum and triple E0')
        return np.zeros( (1, 8 ) )

    # if useTripleE0 and ( useEmissionLines or useEdepSum ):
    #     print( 'use E0 computed from triple events is set to be True, it will override using emission lines or using sum of edeps' )

    if par['fullAbsorptionOnly']:
        fullAbsorptionOnly = True

    if par['timeStampOnly']:
        timeStampOnly = True
        timeWindow = par['timeWindow'] / 10.0

    if par['singlePlane']:
        singlePlane = True
        planeNumber = par['planeNumber']
    if par['singleModule']:
        singleModule = True
        moduleNumber = par['moduleNumber']

    if par['dopplerCorrection']:
        dopplerCorrection = True

    if fabs( couchAngle ) > ZERO:
        rotateCouch = True
        couchAngle = couchAngle / 180.0 * PI

    if fabs( gantryAngle -270.0 ) > ZERO:
        rotateGantry = True
        gantryAngle = gantryAngle / 180.0 * PI

    events =[]
    N = len( npx )

    while i < N-1:
        proceed = False
        if timeStampOnly:
            nInteractions = 1
            while t[i+nInteractions] - t[i] < timeWindow:
                nInteractions += 1
                if i + nInteractions >= N:
                    boundaryExceeded = True
                    break
            if boundaryExceeded:
                break
            if nInteractions != 3:
                i += nInteractions
                continue
            else:
                proceed = True
        else:
            if npx[i] == 3:
                if npx[i+1] == 3 and t[i+1] == t[i]:
                    if npx[i+2] == 3 and t[i+2] == t[i]:
                        proceed = True
                    else:
                        i += 2
                        continue
                else:
                    i += 1
                    continue
            else:
                i += 1
                continue

        if proceed:
            # three interactions all marked as 3 pixel events and they actually have
            # the same time stamp.
            thisDca = 1000.0
            thisProb = np.ones( 8 ) * ( -1000.0 )
            betterDca = 1000.0
            betterProb = -1000.0
            betterOrder = -1
            betterE0 = 1000.0

            # in the following cases, where the module number does not
            # meet the requirements, skip this 3-pixle events
            if singleModule and ( ( mdl[i] != moduleNumber ) or ( mdl[i+1] != moduleNumber ) or ( mdl[i+2] != moduleNumber ) ):
                i = i + 3
                continue
            if singlePlane:
                if planeNumber == 0 and ( ( mdl[i] > 7 ) or ( mdl[i+1] > 7 ) or ( mdl[i+2] > 7 ) ):
                    i = i + 3
                    continue
                elif planeNumber == 1 and ( ( mdl[i] < 8 ) or ( mdl[i+1] < 8 ) or ( mdl[i+2] < 8 ) ):
                    i = i + 3
                    continue

            if pointSource:
                eDepA = edep[i]
                Ax = x[i]
                Ay = y[i]
                Az = z[i]
                eDepB = edep[i+1]
                Bx = x[i+1]
                By = y[i+1]
                Bz = z[i+1]
                eDepC = edep[i+2]
                Cx = x[i+2]
                Cy = y[i+2]
                Cz = z[i+2]

                if useTripleE0:
                    E0 = E0Triple( Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, eDepA, eDepB )
                    # if E0 < 0, then this order is impossible:
                    if E0 > 0 :
                        # check if this is a full absoption:
                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            # required full absoption only, but this is not, do nothing
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepA )
                            # if this is not a full absoption, then the last interaction is Compton, check if
                            # the last interaction is a valid Compton ( positive cone angle )
                            if coneAngle > 0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepA - eDepB, eDepC ) > 0.0 ) ) ):
                                thisDca = pointDCAVector( Ax, Ay, Az, Bx, By, Bz, sx, sy, sz, coneAngle )
                                # compute probability is expensive, computing only if DCA is less than the threshold
                                if fabs( thisDca ) < dcaThreshold:
                                    thisProb =  tripleProbabilityPointSource(   E0,
                                                                                sx, sy, sz,
                                                                                eDepA, Ax, Ay, Az,
                                                                                eDepB, Bx, By, Bz,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    # update to record this permutation if it is better ( shortter DCA and higher probability )
                                    if  ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 0

                    E0 = E0Triple( Ax, Ay, Az, Cx, Cy, Cz, Bx, By, Bz, eDepA, eDepC )
                    if E0 > 0:
                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepA )
                            if coneAngle > 0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepA - eDepC, eDepB ) > 0.0 ) ) ):
                                thisDca = pointDCAVector( Ax, Ay, Az, Cx, Cy, Cz, sx, sy, sz, coneAngle )
                                if fabs( thisDca ) < dcaThreshold:
                                    thisProb =  tripleProbabilityPointSource(   E0,
                                                                                sx, sy, sz,
                                                                                eDepA, Ax, Ay, Az,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                eDepB, Bx, By, Bz,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    if  ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 1

                    E0 = E0Triple( Bx, By, Bz, Ax, Ay, Az, Cx, Cy, Cz, eDepB, eDepA )
                    if E0 > 0:
                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepB )
                            if coneAngle >0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepB - eDepA, eDepC ) > 0.0 ) ) ):
                                thisDca = pointDCAVector( Bx, By, Bz, Ax, Ay, Az, sx, sy, sz, coneAngle )
                                if fabs( thisDca ) < dcaThreshold:
                                    thisProb =  tripleProbabilityPointSource(   E0,
                                                                                sx, sy, sz,
                                                                                eDepB, Bx, By, Bz,
                                                                                eDepA, Ax, Ay, Az,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 2

                    E0 = E0Triple( Bx, By, Bz, Cx, Cy, Cz, Ax, Ay, Az, eDepB, eDepC )
                    if E0 > 0:
                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepB )
                            if coneAngle >0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepB - eDepC, eDepA ) > 0.0 ) ) ):
                                thisDca = pointDCAVector( Bx, By, Bz, Cx, Cy, Cz, sx, sy, sz, coneAngle )
                                if fabs( thisDca ) < dcaThreshold:
                                    thisProb =  tripleProbabilityPointSource(   E0,
                                                                                sx, sy, sz,
                                                                                eDepB, Bx, By, Bz,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                eDepA, Ax, Ay, Az,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 3

                    E0 = E0Triple( Cx, Cy, Cz, Ax, Ay, Az, Bx, By, Bz, eDepC, eDepA )
                    if E0 > 0:
                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepC )
                            if coneAngle >0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepC - eDepA, eDepB ) > 0.0 ) ) ):
                                thisDca = pointDCAVector( Cx, Cy, Cz, Ax, Ay, Az, sx, sy, sz, coneAngle )
                                if fabs( thisDca ) < dcaThreshold:
                                    thisProb =  tripleProbabilityPointSource(   E0,
                                                                                sx, sy, sz,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                eDepA, Ax, Ay, Az,
                                                                                eDepB, Bx, By, Bz,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 4

                    E0 = E0Triple( Cx, Cy, Cz, Bx, By, Bz, Ax, Ay, Az, eDepC, eDepB )
                    if E0 > 0:
                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepC )
                            if coneAngle >0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepC - eDepB, eDepA ) > 0.0 ) ) ):
                                thisDca = pointDCAVector( Cx, Cy, Cz, Bx, By, Bz, sx, sy, sz, coneAngle )
                                if fabs( thisDca ) < dcaThreshold:
                                    thisProb =  tripleProbabilityPointSource(   E0,
                                                                                sx, sy, sz,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                eDepB, Bx, By, Bz,
                                                                                eDepA, Ax, Ay, Az,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 5

                    if betterProb > -800:
                        if betterOrder == 0:
                            events.append( np.array( [  eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz, eDepC, Cx, Cy, Cz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2]. t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 1:
                            events.append( np.array( [  eDepA, Ax, Ay, Az, eDepC, Cx, Cy, Cz, eDepB, Bx, By, Bz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2]. t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 2:
                            events.append( np.array( [  eDepB, Bx, By, Bz, eDepA, Ax, Ay, Az, eDepC, Cx, Cy, Cz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2]. t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 3:
                            events.append( np.array( [  eDepB, Bx, By, Bz, eDepC, Cx, Cy, Cz, eDepA, Ax, Ay, Az,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2]. t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 4:
                            events.append( np.array( [  eDepC, Cx, Cy, Cz, eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2]. t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 5:
                            events.append( np.array( [  eDepC, Cx, Cy, Cz, eDepB, Bx, By, Bz, eDepA, Ax, Ay, Az,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2]. t[i], t[i+1], t[i+2], i ] ) )

                elif useEmissionLines:
                    for E0 in gammaE:
                        # skip to next E0 if the sum of total deposited energy is greater than E0
                        if ( eDepA + eDepB + eDepC ) / E0 - 1 > fullAbsorptionThreshold:
                            continue

                        # check if this is a full absoption:
                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            # required full absoption only, but this is not, do nothing
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepA )
                            # if this is not a full absoption, then the last interaction is Compton, check if
                            # the last interaction is a valid Compton ( positive cone angle )
                            if coneAngle > 0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepA - eDepB, eDepC ) > 0.0 ) ) ):
                                thisDca = pointDCAVector( Ax, Ay, Az, Bx, By, Bz, sx, sy, sz, coneAngle )
                                # compute probability is expensive, computing only if DCA is less than the threshold
                                if fabs( thisDca ) < dcaThreshold:
                                    thisProb =  tripleProbabilityPointSource(   E0,
                                                                                sx, sy, sz,
                                                                                eDepA, Ax, Ay, Az,
                                                                                eDepB, Bx, By, Bz,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    # update to record this permutation if it is better ( shortter DCA and higher probability )
                                    if  ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 0

                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepA )
                            if coneAngle > 0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepA - eDepC, eDepB ) > 0.0 ) ) ):
                                thisDca = pointDCAVector( Ax, Ay, Az, Cx, Cy, Cz, sx, sy, sz, coneAngle )
                                if fabs( thisDca ) < dcaThreshold:
                                    thisProb =  tripleProbabilityPointSource(   E0,
                                                                                sx, sy, sz,
                                                                                eDepA, Ax, Ay, Az,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                eDepB, Bx, By, Bz,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    if  ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 1

                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepB )
                            if coneAngle >0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepB - eDepA, eDepC ) > 0.0 ) ) ):
                                thisDca = pointDCAVector( Bx, By, Bz, Ax, Ay, Az, sx, sy, sz, coneAngle )
                                if fabs( thisDca ) < dcaThreshold:
                                    thisProb =  tripleProbabilityPointSource(   E0,
                                                                                sx, sy, sz,
                                                                                eDepB, Bx, By, Bz,
                                                                                eDepA, Ax, Ay, Az,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 2

                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepB )
                            if coneAngle >0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepB - eDepC, eDepA ) > 0.0 ) ) ):
                                thisDca = pointDCAVector( Bx, By, Bz, Cx, Cy, Cz, sx, sy, sz, coneAngle )
                                if fabs( thisDca ) < dcaThreshold:
                                    thisProb =  tripleProbabilityPointSource(   E0,
                                                                                sx, sy, sz,
                                                                                eDepB, Bx, By, Bz,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                eDepA, Ax, Ay, Az,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 3

                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepC )
                            if coneAngle >0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepC - eDepA, eDepB ) > 0.0 ) ) ):
                                thisDca = pointDCAVector( Cx, Cy, Cz, Ax, Ay, Az, sx, sy, sz, coneAngle )
                                if fabs( thisDca ) < dcaThreshold:
                                    thisProb =  tripleProbabilityPointSource(   E0,
                                                                                sx, sy, sz,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                eDepA, Ax, Ay, Az,
                                                                                eDepB, Bx, By, Bz,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 4

                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepC )
                            if coneAngle >0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepC - eDepB, eDepA ) > 0.0 ) ) ):
                                thisDca = pointDCAVector( Cx, Cy, Cz, Bx, By, Bz, sx, sy, sz, coneAngle )
                                if fabs( thisDca ) < dcaThreshold:
                                    thisProb =  tripleProbabilityPointSource(   E0,
                                                                                sx, sy, sz,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                eDepB, Bx, By, Bz,
                                                                                eDepA, Ax, Ay, Az,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 5

                    if betterProb > -800:
                        if betterOrder == 0:
                            events.append( np.array( [  eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz, eDepC, Cx, Cy, Cz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2]. t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 1:
                            events.append( np.array( [  eDepA, Ax, Ay, Az, eDepC, Cx, Cy, Cz, eDepB, Bx, By, Bz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2]. t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 2:
                            events.append( np.array( [  eDepB, Bx, By, Bz, eDepA, Ax, Ay, Az, eDepC, Cx, Cy, Cz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2]. t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 3:
                            events.append( np.array( [  eDepB, Bx, By, Bz, eDepC, Cx, Cy, Cz, eDepA, Ax, Ay, Az,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2]. t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 4:
                            events.append( np.array( [  eDepC, Cx, Cy, Cz, eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2]. t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 5:
                            events.append( np.array( [  eDepC, Cx, Cy, Cz, eDepB, Bx, By, Bz, eDepA, Ax, Ay, Az,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2]. t[i], t[i+1], t[i+2], i ] ) )

                elif useEdepSum:
                    E0 = eDepA + eDepB + eDepC
                    fullAbsorption = True

                    coneAngle = scatterAngle( E0, eDepA )
                    if coneAngle > 0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepA - eDepB, eDepC ) > 0.0 ) ) ):
                        thisDca = pointDCAVector( Ax, Ay, Az, Bx, By, Bz, sx, sy, sz, coneAngle )
                        # compute probability is expensive, computing only if DCA is less than the threshold
                        if fabs( thisDca ) < dcaThreshold:
                            thisProb =  tripleProbabilityPointSource(   E0,
                                                                        sx, sy, sz,
                                                                        eDepA, Ax, Ay, Az,
                                                                        eDepB, Bx, By, Bz,
                                                                        eDepC, Cx, Cy, Cz,
                                                                        cameraX, cameraY, cameraZ,
                                                                        nODApproximalPoints, fullAbsorption )
                            # update to record this permutation if it is better ( shortter DCA and higher probability )
                            if  ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                betterDca = thisDca
                                betterProb = thisProb[0]
                                betterE0 = E0
                                betterOrder = 0

                    coneAngle = scatterAngle( E0, eDepA )
                    if coneAngle > 0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepA - eDepC, eDepB ) > 0.0 ) ) ):
                        thisDca = pointDCAVector( Ax, Ay, Az, Cx, Cy, Cz, sx, sy, sz, coneAngle )
                        if fabs( thisDca ) < dcaThreshold:
                            thisProb =  tripleProbabilityPointSource(   E0,
                                                                        sx, sy, sz,
                                                                        eDepA, Ax, Ay, Az,
                                                                        eDepC, Cx, Cy, Cz,
                                                                        eDepB, Bx, By, Bz,
                                                                        cameraX, cameraY, cameraZ,
                                                                        nODApproximalPoints, fullAbsorption )
                            if  ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                betterDca = thisDca
                                betterProb = thisProb[0]
                                betterE0 = E0
                                betterOrder = 1

                    coneAngle = scatterAngle( E0, eDepB )
                    if coneAngle >0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepB - eDepA, eDepC ) > 0.0 ) ) ):
                        thisDca = pointDCAVector( Bx, By, Bz, Ax, Ay, Az, sx, sy, sz, coneAngle )
                        if fabs( thisDca ) < dcaThreshold:
                            thisProb =  tripleProbabilityPointSource(   E0,
                                                                        sx, sy, sz,
                                                                        eDepB, Bx, By, Bz,
                                                                        eDepA, Ax, Ay, Az,
                                                                        eDepC, Cx, Cy, Cz,
                                                                        cameraX, cameraY, cameraZ,
                                                                        nODApproximalPoints, fullAbsorption )
                            if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                betterDca = thisDca
                                betterProb = thisProb[0]
                                betterE0 = E0
                                betterOrder = 2

                    coneAngle = scatterAngle( E0, eDepB )
                    if coneAngle >0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepB - eDepC, eDepA ) > 0.0 ) ) ):
                        thisDca = pointDCAVector( Bx, By, Bz, Cx, Cy, Cz, sx, sy, sz, coneAngle )
                        if fabs( thisDca ) < dcaThreshold:
                            thisProb =  tripleProbabilityPointSource(   E0,
                                                                        sx, sy, sz,
                                                                        eDepB, Bx, By, Bz,
                                                                        eDepC, Cx, Cy, Cz,
                                                                        eDepA, Ax, Ay, Az,
                                                                        cameraX, cameraY, cameraZ,
                                                                        nODApproximalPoints, fullAbsorption )
                            if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                betterDca = thisDca
                                betterProb = thisProb[0]
                                betterE0 = E0
                                betterOrder = 3

                    coneAngle = scatterAngle( E0, eDepC )
                    if coneAngle >0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepC - eDepA, eDepB ) > 0.0 ) ) ):
                        thisDca = pointDCAVector( Cx, Cy, Cz, Ax, Ay, Az, sx, sy, sz, coneAngle )
                        if fabs( thisDca ) < dcaThreshold:
                            thisProb =  tripleProbabilityPointSource(   E0,
                                                                        sx, sy, sz,
                                                                        eDepC, Cx, Cy, Cz,
                                                                        eDepA, Ax, Ay, Az,
                                                                        eDepB, Bx, By, Bz,
                                                                        cameraX, cameraY, cameraZ,
                                                                        nODApproximalPoints, fullAbsorption )
                            if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                betterDca = thisDca
                                betterProb = thisProb[0]
                                betterE0 = E0
                                betterOrder = 4

                    coneAngle = scatterAngle( E0, eDepC )
                    if coneAngle >0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepC - eDepB, eDepA ) > 0.0 ) ) ):
                        thisDca = pointDCAVector( Cx, Cy, Cz, Bx, By, Bz, sx, sy, sz, coneAngle )
                        if fabs( thisDca ) < dcaThreshold:
                            thisProb =  tripleProbabilityPointSource(   E0,
                                                                        sx, sy, sz,
                                                                        eDepC, Cx, Cy, Cz,
                                                                        eDepB, Bx, By, Bz,
                                                                        eDepA, Ax, Ay, Az,
                                                                        cameraX, cameraY, cameraZ,
                                                                        nODApproximalPoints, fullAbsorption )
                            if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                betterDca = thisDca
                                betterProb = thisProb[0]
                                betterE0 = E0
                                betterOrder = 5

                    if betterProb > -800:
                        if betterOrder == 0:
                            events.append( np.array( [  eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz, eDepC, Cx, Cy, Cz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2]. t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 1:
                            events.append( np.array( [  eDepA, Ax, Ay, Az, eDepC, Cx, Cy, Cz, eDepB, Bx, By, Bz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2]. t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 2:
                            events.append( np.array( [  eDepB, Bx, By, Bz, eDepA, Ax, Ay, Az, eDepC, Cx, Cy, Cz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2]. t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 3:
                            events.append( np.array( [  eDepB, Bx, By, Bz, eDepC, Cx, Cy, Cz, eDepA, Ax, Ay, Az,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2]. t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 4:
                            events.append( np.array( [  eDepC, Cx, Cy, Cz, eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2]. t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 5:
                            events.append( np.array( [  eDepC, Cx, Cy, Cz, eDepB, Bx, By, Bz, eDepA, Ax, Ay, Az,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2]. t[i], t[i+1], t[i+2], i ] ) )


            elif beamMeasurement:
                if rotateCouch:
                    eDepA = edep[i]
                    Ax = x[i] * cos( couchAngle ) - z[i] * sin( couchAngle )
                    Ay = y[i]
                    Az = x[i] * sin( couchAngle ) + z[i] * cos( couchAngle )
                    eDepB = edep[i+1]
                    Bx = x[i+1] * cos( couchAngle ) - z[i+1] * sin( couchAngle )
                    By = y[i+1]
                    Bz = x[i+1] * sin( couchAngle ) + z[i+1] * cos( couchAngle )
                    eDepC = edep[i+2]
                    Cx = x[i+2] * cos( couchAngle ) - z[i+2] * sin( couchAngle )
                    Cy = y[i+2]
                    Cz = x[i+2] * sin( couchAngle ) + z[i+2] * cos( couchAngle )
                    if rotateGantry:
                        tempY = Ay
                        tempZ = Az
                        Ay = -tempY * sin( gantryAngle ) - tempZ * cos( gantryAngle )
                        Az =  tempY * cos( gantryAngle ) - tempZ * sin( gantryAngle )
                        tempY = By
                        tempZ = Bz
                        By = -tempY * sin( gantryAngle ) - tempZ * cos( gantryAngle )
                        Bz =  tempY * cos( gantryAngle ) - tempZ * sin( gantryAngle )
                        tempY = Cy
                        tempZ = Cz
                        Cy = -tempY * sin( gantryAngle ) - tempZ * cos( gantryAngle )
                        Cz =  tempY * cos( gantryAngle ) - tempZ * sin( gantryAngle )
                else:
                    eDepA = edep[i]
                    Ax = x[i]
                    Ay = y[i]
                    Az = z[i]
                    eDepB = edep[i+1]
                    Bx = x[i+1]
                    By = y[i+1]
                    Bz = z[i+1]
                    eDepC = edep[i+2]
                    Cx = x[i+2]
                    Cy = y[i+2]
                    Cz = z[i+2]
                    if rotateGantry:
                        tempY = Ay
                        tempZ = Az
                        Ay = -tempY * sin( gantryAngle ) - tempZ * cos( gantryAngle )
                        Az =  tempY * cos( gantryAngle ) - tempZ * sin( gantryAngle )
                        tempY = By
                        tempZ = Bz
                        By = -tempY * sin( gantryAngle ) - tempZ * cos( gantryAngle )
                        Bz =  tempY * cos( gantryAngle ) - tempZ * sin( gantryAngle )
                        tempY = Cy
                        tempZ = Cz
                        Cy = -tempY * sin( gantryAngle ) - tempZ * cos( gantryAngle )
                        Cz =  tempY * cos( gantryAngle ) - tempZ * sin( gantryAngle )


                if useTripleE0:
                    E0 = E0Triple( Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, eDepA, eDepB )
                    if E0 > 0:
                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepA )
                            if coneAngle >0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepA - eDepB, eDepC ) > 0.0 ) ) ):
                                thisDca, coord = zAxisDCA(  Ax, Ay, Az,
                                                            Bx, By, Bz,
                                                            coneAngle, slicePosition,
                                                            boundaryX1, boundaryX2,
                                                            boundaryY1, boundaryY2,
                                                            boundaryZ1, boundaryZ2)
                                if fabs( thisDca ) < dcaThreshold:
                                    thisProb =  tripleProbabilityLineSource(   E0,
                                                                                eDepA, Ax, Ay, Az,
                                                                                eDepB, Bx, By, Bz,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 0

                    E0 = E0Triple( Ax, Ay, Az, Cx, Cy, Cz, Bx, By, Bz, eDepA, eDepC )
                    if E0 > 0:
                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepA )
                            if coneAngle >0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepA - eDepC, eDepB ) > 0.0 ) ) ):
                                thisDca, coord = zAxisDCA(  Ax, Ay, Az,
                                                            Cx, Cy, Cz,
                                                            coneAngle, slicePosition,
                                                            boundaryX1, boundaryX2,
                                                            boundaryY1, boundaryY2,
                                                            boundaryZ1, boundaryZ2)

                                if fabs( thisDca ) < dcaThreshold:
                                    thisProb =  tripleProbabilityLineSource(   E0,
                                                                                eDepA, Ax, Ay, Az,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                eDepB, Bx, By, Bz,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 1

                    E0 = E0Triple( Bx, By, Bz, Ax, Ay, Az, Cx, Cy, Cz, eDepB, eDepA )
                    if E0 > 0:
                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepB )
                            if coneAngle > 0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepB - eDepA, eDepC ) > 0.0 ) ) ):
                                thisDca, coord = zAxisDCA(  Bx, By, Bz,
                                                            Ax, Ay, Az,
                                                            coneAngle, slicePosition,
                                                            boundaryX1, boundaryX2,
                                                            boundaryY1, boundaryY2,
                                                            boundaryZ1, boundaryZ2)

                                if fabs( thisDca ) < dcaThreshold:
                                    thisProb =  tripleProbabilityLineSource(    E0,
                                                                                eDepB, Bx, By, Bz,
                                                                                eDepA, Ax, Ay, Az,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 2

                    E0 = E0Triple( Bx, By, Bz, Cx, Cy, Cz, Ax, Ay, Az, eDepB, eDepC )
                    if E0 > 0:
                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepB )
                            if coneAngle > 0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepB - eDepC, eDepA ) > 0.0 ) ) ):
                                thisDca, coord = zAxisDCA(  Bx, By, Bz,
                                                            Cx, Cy, Cz,
                                                            coneAngle, slicePosition,
                                                            boundaryX1, boundaryX2,
                                                            boundaryY1, boundaryY2,
                                                            boundaryZ1, boundaryZ2)

                                if fabs( thisDca ) < dcaThreshold:
                                    thisProb =  tripleProbabilityLineSource(    E0,
                                                                                eDepB, Bx, By, Bz,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                eDepA, Ax, Ay, Az,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 3

                    E0 = E0Triple( Cx, Cy, Cz, Ax, Ay, Az, Bx, By, Bz, eDepC, eDepA )
                    if E0 > 0:
                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepC )
                            if coneAngle > 0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepC - eDepA, eDepB ) > 0.0 ) ) ):
                                thisDca, coord = zAxisDCA(  Cx, Cy, Cz,
                                                            Ax, Ay, Az,
                                                            coneAngle, slicePosition,
                                                            boundaryX1, boundaryX2,
                                                            boundaryY1, boundaryY2,
                                                            boundaryZ1, boundaryZ2)

                                if fabs( thisDca ) < fabs( dcaThreshold ):
                                    thisProb =  tripleProbabilityLineSource(    E0,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                eDepA, Ax, Ay, Az,
                                                                                eDepB, Bx, By, Bz,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 4

                    E0 = E0Triple( Cx, Cy, Cz, Bx, By, Bz, Ax, Ay, Az, eDepC, eDepB )
                    if E0 > 0:
                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepC )
                            if coneAngle > 0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepC - eDepB, eDepA ) > 0.0 ) ) ):
                                thisDca, coord = zAxisDCA(  Cx, Cy, Cz,
                                                            Bx, By, Bz,
                                                            coneAngle, slicePosition,
                                                            boundaryX1, boundaryX2,
                                                            boundaryY1, boundaryY2,
                                                            boundaryZ1, boundaryZ2)

                                if fabs( thisDca ) < dcaThreshold:
                                    thisProb =  tripleProbabilityLineSource(    E0,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                eDepB, Bx, By, Bz,
                                                                                eDepA, Ax, Ay, Az,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 5

                    # if betterDca < dcaThreshold:
                    if betterProb > -800:
                        if betterOrder == 0:
                            events.append( np.array( [  eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz, eDepC, Cx, Cy, Cz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2], t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 1:
                            events.append( np.array( [  eDepA, Ax, Ay, Az, eDepC, Cx, Cy, Cz, eDepB, Bx, By, Bz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2], t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 2:
                            events.append( np.array( [  eDepB, Bx, By, Bz, eDepA, Ax, Ay, Az, eDepC, Cx, Cy, Cz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2], t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 3:
                            events.append( np.array( [  eDepB, Bx, By, Bz, eDepC, Cx, Cy, Cz, eDepA, Ax, Ay, Az,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2], t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 4:
                            events.append( np.array( [  eDepC, Cx, Cy, Cz, eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2], t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 5:
                            events.append( np.array( [  eDepC, Cx, Cy, Cz, eDepB, Bx, By, Bz, eDepA, Ax, Ay, Az,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2], t[i], t[i+1], t[i+2], i ] ) )


                elif useEmissionLines:
                    for E0 in gammaE:
                        if ( eDepA + eDepB + eDepC ) / E0 - 1.0 > fullAbsorptionThreshold:
                            continue

                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepA )
                            if coneAngle >0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepA - eDepB, eDepC ) > 0.0 ) ) ):
                                thisDca, coord = zAxisDCA(  Ax, Ay, Az,
                                                            Bx, By, Bz,
                                                            coneAngle, slicePosition,
                                                            boundaryX1, boundaryX2,
                                                            boundaryY1, boundaryY2,
                                                            boundaryZ1, boundaryZ2)
                                if fabs( thisDca ) < dcaThreshold:
                                    thisProb =  tripleProbabilityLineSource(   E0,
                                                                                eDepA, Ax, Ay, Az,
                                                                                eDepB, Bx, By, Bz,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 0

                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepA )
                            if coneAngle >0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepA - eDepC, eDepB ) > 0.0 ) ) ):
                                thisDca, coord = zAxisDCA(  Ax, Ay, Az,
                                                            Cx, Cy, Cz,
                                                            coneAngle, slicePosition,
                                                            boundaryX1, boundaryX2,
                                                            boundaryY1, boundaryY2,
                                                            boundaryZ1, boundaryZ2)

                                if fabs( thisDca ) < dcaThreshold:
                                    thisProb =  tripleProbabilityLineSource(   E0,
                                                                                eDepA, Ax, Ay, Az,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                eDepB, Bx, By, Bz,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 1

                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepB )
                            if coneAngle > 0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepB - eDepA, eDepC ) > 0.0 ) ) ):
                                thisDca, coord = zAxisDCA(  Bx, By, Bz,
                                                            Ax, Ay, Az,
                                                            coneAngle, slicePosition,
                                                            boundaryX1, boundaryX2,
                                                            boundaryY1, boundaryY2,
                                                            boundaryZ1, boundaryZ2)

                                if fabs( thisDca ) < dcaThreshold:
                                    thisProb =  tripleProbabilityLineSource(    E0,
                                                                                eDepB, Bx, By, Bz,
                                                                                eDepA, Ax, Ay, Az,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 2

                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepB )
                            if coneAngle > 0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepB - eDepC, eDepA ) > 0.0 ) ) ):
                                thisDca, coord = zAxisDCA(  Bx, By, Bz,
                                                            Cx, Cy, Cz,
                                                            coneAngle, slicePosition,
                                                            boundaryX1, boundaryX2,
                                                            boundaryY1, boundaryY2,
                                                            boundaryZ1, boundaryZ2)

                                if fabs( thisDca ) < dcaThreshold:
                                    thisProb =  tripleProbabilityLineSource(    E0,
                                                                                eDepB, Bx, By, Bz,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                eDepA, Ax, Ay, Az,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 3

                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepC )
                            if coneAngle > 0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepC - eDepA, eDepB ) > 0.0 ) ) ):
                                thisDca, coord = zAxisDCA(  Cx, Cy, Cz,
                                                            Ax, Ay, Az,
                                                            coneAngle, slicePosition,
                                                            boundaryX1, boundaryX2,
                                                            boundaryY1, boundaryY2,
                                                            boundaryZ1, boundaryZ2)

                                if fabs( thisDca ) < fabs( dcaThreshold ):
                                    thisProb =  tripleProbabilityLineSource(    E0,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                eDepA, Ax, Ay, Az,
                                                                                eDepB, Bx, By, Bz,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 4

                        fullAbsorption = fabs( ( eDepA + eDepB + eDepC ) / E0 - 1.0 ) < fullAbsorptionThreshold
                        if fullAbsorptionOnly and ( not fullAbsorption ):
                            pass
                        else:
                            coneAngle = scatterAngle( E0, eDepC )
                            if coneAngle > 0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepC - eDepB, eDepA ) > 0.0 ) ) ):
                                thisDca, coord = zAxisDCA(  Cx, Cy, Cz,
                                                            Bx, By, Bz,
                                                            coneAngle, slicePosition,
                                                            boundaryX1, boundaryX2,
                                                            boundaryY1, boundaryY2,
                                                            boundaryZ1, boundaryZ2)

                                if fabs( thisDca ) < dcaThreshold:
                                    thisProb =  tripleProbabilityLineSource(    E0,
                                                                                eDepC, Cx, Cy, Cz,
                                                                                eDepB, Bx, By, Bz,
                                                                                eDepA, Ax, Ay, Az,
                                                                                cameraX, cameraY, cameraZ,
                                                                                nODApproximalPoints, fullAbsorption )
                                    if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                        betterDca = thisDca
                                        betterProb = thisProb[0]
                                        betterE0 = E0
                                        betterOrder = 5

                    # if betterDca < dcaThreshold:
                    if betterProb > -800:
                        if betterOrder == 0:
                            events.append( np.array( [  eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz, eDepC, Cx, Cy, Cz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2], t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 1:
                            events.append( np.array( [  eDepA, Ax, Ay, Az, eDepC, Cx, Cy, Cz, eDepB, Bx, By, Bz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2], t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 2:
                            events.append( np.array( [  eDepB, Bx, By, Bz, eDepA, Ax, Ay, Az, eDepC, Cx, Cy, Cz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2], t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 3:
                            events.append( np.array( [  eDepB, Bx, By, Bz, eDepC, Cx, Cy, Cz, eDepA, Ax, Ay, Az,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2], t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 4:
                            events.append( np.array( [  eDepC, Cx, Cy, Cz, eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2], t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 5:
                            events.append( np.array( [  eDepC, Cx, Cy, Cz, eDepB, Bx, By, Bz, eDepA, Ax, Ay, Az,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2], t[i], t[i+1], t[i+2], i ] ) )

                elif useEdepSum:
                    E0 = eDepA + eDepB + eDepC
                    fullAbsorption = True

                    coneAngle = scatterAngle( E0, eDepA )
                    if coneAngle >0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepA - eDepB, eDepC ) > 0.0 ) ) ):
                        thisDca, coord = zAxisDCA(  Ax, Ay, Az,
                                                    Bx, By, Bz,
                                                    coneAngle, slicePosition,
                                                    boundaryX1, boundaryX2,
                                                    boundaryY1, boundaryY2,
                                                    boundaryZ1, boundaryZ2)
                        if fabs( thisDca ) < dcaThreshold:
                            thisProb =  tripleProbabilityLineSource(   E0,
                                                                        eDepA, Ax, Ay, Az,
                                                                        eDepB, Bx, By, Bz,
                                                                        eDepC, Cx, Cy, Cz,
                                                                        cameraX, cameraY, cameraZ,
                                                                        nODApproximalPoints, fullAbsorption )
                            if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                betterDca = thisDca
                                betterProb = thisProb[0]
                                betterE0 = E0
                                betterOrder = 0

                    coneAngle = scatterAngle( E0, eDepA )
                    if coneAngle >0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepA - eDepC, eDepB ) > 0.0 ) ) ):
                        thisDca, coord = zAxisDCA(  Ax, Ay, Az,
                                                    Cx, Cy, Cz,
                                                    coneAngle, slicePosition,
                                                    boundaryX1, boundaryX2,
                                                    boundaryY1, boundaryY2,
                                                    boundaryZ1, boundaryZ2)

                        if fabs( thisDca ) < dcaThreshold:
                            thisProb =  tripleProbabilityLineSource(   E0,
                                                                        eDepA, Ax, Ay, Az,
                                                                        eDepC, Cx, Cy, Cz,
                                                                        eDepB, Bx, By, Bz,
                                                                        cameraX, cameraY, cameraZ,
                                                                        nODApproximalPoints, fullAbsorption )
                            if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                betterDca = thisDca
                                betterProb = thisProb[0]
                                betterE0 = E0
                                betterOrder = 1

                    coneAngle = scatterAngle( E0, eDepB )
                    if coneAngle > 0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepB - eDepA, eDepC ) > 0.0 ) ) ):
                        thisDca, coord = zAxisDCA(  Bx, By, Bz,
                                                    Ax, Ay, Az,
                                                    coneAngle, slicePosition,
                                                    boundaryX1, boundaryX2,
                                                    boundaryY1, boundaryY2,
                                                    boundaryZ1, boundaryZ2)

                        if fabs( thisDca ) < dcaThreshold:
                            thisProb =  tripleProbabilityLineSource(    E0,
                                                                        eDepB, Bx, By, Bz,
                                                                        eDepA, Ax, Ay, Az,
                                                                        eDepC, Cx, Cy, Cz,
                                                                        cameraX, cameraY, cameraZ,
                                                                        nODApproximalPoints, fullAbsorption )
                            if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                betterDca = thisDca
                                betterProb = thisProb[0]
                                betterE0 = E0
                                betterOrder = 2

                    coneAngle = scatterAngle( E0, eDepB )
                    if coneAngle > 0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepB - eDepC, eDepA ) > 0.0 ) ) ):
                        thisDca, coord = zAxisDCA(  Bx, By, Bz,
                                                    Cx, Cy, Cz,
                                                    coneAngle, slicePosition,
                                                    boundaryX1, boundaryX2,
                                                    boundaryY1, boundaryY2,
                                                    boundaryZ1, boundaryZ2)

                        if fabs( thisDca ) < dcaThreshold:
                            thisProb =  tripleProbabilityLineSource(    E0,
                                                                        eDepB, Bx, By, Bz,
                                                                        eDepC, Cx, Cy, Cz,
                                                                        eDepA, Ax, Ay, Az,
                                                                        cameraX, cameraY, cameraZ,
                                                                        nODApproximalPoints, fullAbsorption )
                            if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                betterDca = thisDca
                                betterProb = thisProb[0]
                                betterE0 = E0
                                betterOrder = 3

                    coneAngle = scatterAngle( E0, eDepC )
                    if coneAngle > 0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepC - eDepA, eDepB ) > 0.0 ) ) ):
                        thisDca, coord = zAxisDCA(  Cx, Cy, Cz,
                                                    Ax, Ay, Az,
                                                    coneAngle, slicePosition,
                                                    boundaryX1, boundaryX2,
                                                    boundaryY1, boundaryY2,
                                                    boundaryZ1, boundaryZ2)

                        if fabs( thisDca ) < fabs( dcaThreshold ):
                            thisProb =  tripleProbabilityLineSource(    E0,
                                                                        eDepC, Cx, Cy, Cz,
                                                                        eDepA, Ax, Ay, Az,
                                                                        eDepB, Bx, By, Bz,
                                                                        cameraX, cameraY, cameraZ,
                                                                        nODApproximalPoints, fullAbsorption )
                            if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                betterDca = thisDca
                                betterProb = thisProb[0]
                                betterE0 = E0
                                betterOrder = 4

                    coneAngle = scatterAngle( E0, eDepC )
                    if coneAngle > 0 and ( fullAbsorption or ( ( not fullAbsorption) and ( scatterAngle( E0 - eDepC - eDepB, eDepA ) > 0.0 ) ) ):
                        thisDca, coord = zAxisDCA(  Cx, Cy, Cz,
                                                    Bx, By, Bz,
                                                    coneAngle, slicePosition,
                                                    boundaryX1, boundaryX2,
                                                    boundaryY1, boundaryY2,
                                                    boundaryZ1, boundaryZ2)

                        if fabs( thisDca ) < dcaThreshold:
                            thisProb =  tripleProbabilityLineSource(    E0,
                                                                        eDepC, Cx, Cy, Cz,
                                                                        eDepB, Bx, By, Bz,
                                                                        eDepA, Ax, Ay, Az,
                                                                        cameraX, cameraY, cameraZ,
                                                                        nODApproximalPoints, fullAbsorption )
                            if ( fabs( thisDca ) <= fabs( betterDca ) ) and ( thisProb[0] >= betterProb ):
                                betterDca = thisDca
                                betterProb = thisProb[0]
                                betterE0 = E0
                                betterOrder = 5

                    # if betterDca < dcaThreshold:
                    if betterProb > -800:
                        if betterOrder == 0:
                            events.append( np.array( [  eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz, eDepC, Cx, Cy, Cz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2], t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 1:
                            events.append( np.array( [  eDepA, Ax, Ay, Az, eDepC, Cx, Cy, Cz, eDepB, Bx, By, Bz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2], t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 2:
                            events.append( np.array( [  eDepB, Bx, By, Bz, eDepA, Ax, Ay, Az, eDepC, Cx, Cy, Cz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2], t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 3:
                            events.append( np.array( [  eDepB, Bx, By, Bz, eDepC, Cx, Cy, Cz, eDepA, Ax, Ay, Az,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2], t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 4:
                            events.append( np.array( [  eDepC, Cx, Cy, Cz, eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2], t[i], t[i+1], t[i+2], i ] ) )
                        elif betterOrder == 5:
                            events.append( np.array( [  eDepC, Cx, Cy, Cz, eDepB, Bx, By, Bz, eDepA, Ax, Ay, Az,
                                                        betterE0, betterDca, betterProb, betterOrder,
                                                        mdl[i], mdl[i+1], mdl[i+2], t[i], t[i+1], t[i+2], i ] ) )

        # processed the 3-pixel events, in the case either useEmissionLines
        # or the useEdepSum is true, move index down by 3
        i = i + 3


    events = np.array( events )
    end_time = time.time()
    print( 'time used ', end_time - start_time )
    return events




cpdef np.ndarray[np.float_t, ndim=2] slitCamera(  np.ndarray[np.float_t, ndim=2] events,
                                                float slitY,
                                                float slitX1, float slitX2, float slitZ1, float slitZ2,
                                                float dcaThreshold=10 ):
    '''
    slitEvents = slitCamera( events, slitY, slitX1, slitX2, slitZ1, slitZ2, dcaThreshold=10 )

    digital slit camera.
    For each event, find where on the cone is the closest to the z-axis
    (or where the cone intercepts with z-axis),
    connect that point with the first interaction, and check if the line passes the slit
    or blocked by the slit.
    This may be better calculated while finding good events since the coordinates was calculated
    in the process.

    input:
        events:
            np.ndarray, the events coordiantes. MAKE SURE THE SUM OF THE TWO DEPOSITED ENERGY IS
            THE INITIAL ENERGY
        slitY:
            float/double, position of the slit in y direction
        slitX1, slitX2, slitZ1, slitZ2:
            float/double, openning position of the slit, MUST HAVE slitX1 < slitX2, slitZ1 < slitZ2
        dcaThreshold:
            threshold for DCA
    output:
        slitEvents:
            events whose connected lines can pass through the slit.
    '''
    cdef:
        int     n = len( events )
        int     i=0
        float   eDepA=0.0, Ax=0.0, Ay=0.0, Az=0.0, eDepB=0.0, Bx=0.0, By=0.0, Bz=0.0
        float   coneAngle=0.0, dca=0.
        float   x0=0.0, y0=0.0, z0=0.0, x1=0.0, y1=0.0, z1=0.0, x=0.0, z=0.0, a=0.0

        # some cone intercept with z-axis twice, this is to mark if the first intercept
        # can pass through the slit
        bint    isSlitEvent=False
        np.ndarray coord = np.array( [] )

    slitEvents = []

    for i in range( n ):
        isSlitEvent = False
        eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz = events[i]

        # it is assumed that the sum of eDepA and eDepB is the total initial energy
        coneAngle = scatterAngle( eDepA + eDepB, eDepA )
        if coneAngle > 0:
            dca, coord = zAxisDCA( Ax,  Ay,  Az, Bx, By, Bz, coneAngle,  Cy=0 )

            # if the events matrix is the results from event selection, then this
            # if statement is not needed, just in case it is not.
            if dca < dcaThreshold:
                x1, y1, z1 = events[i, 1:4 ]
                x0, y0, z0 = coord[0]
                a = ( slitY - y0 ) / ( y1 - y0 )

                x = x0 + a * ( x1 - x0 )
                z = z0 + a * ( z1 - z0 )
                if (x > slitX1) and (x < slitX2) and (z > slitZ1) and (z < slitZ2):
                    slitEvents.append( events[i] )
                    isSlitEvent = True

                # if a cone intercepts with z-axis twice, check the second interception
                if len( coord ) == 4:
                    x0, y0, z0 = coord[2]
                    a = ( slitY - y0 ) / ( y1 - y0 )
                    x = x0 + a * ( x1 - x0 )
                    z = z0 + a * ( z1 - z0 )
                    if (not isSlitEvent) and (x > slitX1) and (x < slitX2) and (z > slitZ1) and (z < slitZ2):
                        slitEvents.append( events[i] )
    slitEvents = np.array( slitEvents )
    return np.array( slitEvents )

def DCAPointSourceSelectEvents( np.ndarray events, np.ndarray S, double dcaThreshold=10.0 ):
    '''
    if I already have events array, and source position, use dca method and a specified
    cutoff to select a subset of events.
    '''
    cdef:
        int     nEvents, iEvent
        double   eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz
        double   sx, sy, sz
        double   E0, coneAngle, d

    nEvents = len( events )

    ev = []
    ca = []
    sx, sy, sz = S
    for iEvent in range( nEvents  ):
        eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz  = events[iEvent]
        E0 = eDepA + eDepB
        coneAngle = scatterAngle( E0, eDepA )
        d = pointDCAVector( Ax, Ay, Az, Bx, By, Bz, sx, sy, sz, coneAngle )
        if d < dcaThreshold:
            ca.append( coneAngle )
            ev.append( events[iEvent] )
    return np.array( ev ), np.array( ca )


def DCALineSourceSelectEvents( np.ndarray events, double Cy, double dcaThreshold, np.ndarray boundaryX, np.ndarray boundaryZ ):
    cdef:
        int     nEvents, iEvent
        double   eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz
        double   E0, coneAngle, d
        double   bx1, bx2, bz1, bz2

    nEvents = len( events )
    bx1, bx2 = boundaryX
    bz1, bz2 = boundaryZ

    ev = []
    ca = []
    for iEvent in range( nEvents ):
        eDepA, Ax, Ay, Az, eDepB, Bx, By, Bz  = events[iEvent]
        E0 = eDepA + eDepB
        coneAngle = scatterAngle( E0, eDepA )

        d = zAxisDCA( Ax, Ay, Az, Bx, By, Bz, coneAngle, Cy, boundaryX1=bx1, boundaryX2=bx2, boundaryZ1=bz1, boundaryZ2=bz2 )
        if d < dcaThreshold:
            ca.append( coneAngle )
            ev.append( events[iEvent] )
    return np.array( ev ), np.array( ca )



cpdef double kn_theta( double E0, double E1 ):
    cdef double theta, p, r0=2.82e-15
    if E1 > ( E0 * 0.511 ) / ( 0.511 + 2 * E0 ):
        theta = scatterAngle( E0, E0 - E1 )
        # p = r0 / 2 * ( E1 / E0 )**2 * ( E1 / E0 + E0 / E1 - sin(theta)**2 ) * 2 * PI * sin(theta)
        p = 1.0 / 2 * ( E1 / E0 )**2 * ( E1 / E0 + E0 / E1 - sin(theta)**2 ) * 2 * PI * sin(theta)
        # p = 1.0 / 2 * ( E1 / E0 )**2 * ( E1 / E0 + E0 / E1 - sin(theta)**2 )
        return p
    else:
        return -1.0



cpdef tuple foo():
    # print(PI)
    # print( log(10) )
    # print( sin(PI/2), cos(PI/3) )
    print( 'reloaded again')
    print ( tan( PI / 4 ))
    print(  acos( 0.5 ) )
    print( 'size of float ', sizeof( float ), ' size of int ', sizeof(int), ' size of long ', sizeof(long) )
    # cdef:
    #     int     a[10]
    #     float   b[10]
    #
    # for i in range( 10 ):
    #     a[i] = i
    #     b[i] = i+ 12
    # return a, b
