import numpy as np;
import os

import time
from shutil import copyfile

#we will generate the _kernel-opencl.c file when importing
from distutils.sysconfig import get_python_inc


def StaggeredFDTD_3D_OPENCL(arg):
   
    IncludeDir=get_python_inc()+os.sep+'BabelViscoFDTD'+os.sep
    print("Copying opencl files from "+IncludeDir +" to " +os.getcwd())
    copyfile(IncludeDir+'_gpu_kernel.c', os.getcwd()+os.sep+'_gpu_kernel.c')
    copyfile(IncludeDir+'_indexing.h', os.getcwd()+os.sep+'_indexing.h')
    print(os.getcwd()+os.sep+'_gpu_kernel.c')

    if (type(arg)!=dict):
        raise TypeError( "The input parameter must be a dictionary")

    for key in arg.keys():
        if type(arg[key])==np.ndarray:
            if np.isfortran(arg[key])==False:
                #print "StaggeredFDTD_3D: Converting ndarray " + key + " to Fortran convention";
                arg[key] = np.asfortranarray(arg[key]);
        elif type(arg[key])!=str:
            arg[key]=np.array((arg[key]))
    t0 = time.time()
    if arg['DT'].dtype==np.dtype('float32'):
        Results= _FDTDStaggered_3D(arg)
    else:
        raise TypeError('Operations only supported for float (32bits)')
    t0=time.time()-t0
    print ('Time to run low level FDTDStaggered_3D =', t0)
    return Results



def _FDTDStaggered_3D(arg):
    FIELDS_MEX_TYPE=['InvDXDTplus',
                    'DXDTminus',
                    'InvDXDTplushp',
                    'DXDTminushp',
                    'LambdaMiuMatOverH',
                    'LambdaMatOverH',
                    'MiuMatOverH',
                    'TauLong',
                    'OneOverTauSigma',
                    'TauShear',
                    'InvRhoMatH',
                    'DT',
                    'SourceFunctions',
                    'Ox',
                    'Oy',
                    'Oz']

    for k in FIELDS_MEX_TYPE:
        if arg[k].dtype != np.float:
            raise TypeError('The argument ' + k 'should be float')

    FIELDS_UINT32_TYPE=['IndexSensorMap',
                        'MaterialMap',
                        'PMLThickness',
                        'N1',
                        'N2',
                        'N3',
                        'SILENT',
                        'TimeSteps',
                        'LengthSource',
                        'SnapshotsPos',
                        'SourceMap',
                        'TypeSource',
                        'SelRMSorPeak',
                        'SelMapsRMSPeak',
                        'SelMapsSensors',
                        'SensorSubSampling',
                        'SensorStart',
                        'DefaultGPUDeviceNumber',
                        'SelRMSorPeak',
                        'SelMapsRMSPeak']

    for k in FIELDS_UINT32_TYPE:
        if arg[k].dtype != np.uint32:
            raise TypeError('The argument ' + k 'should be uint32')

    if type(arg['DefaultGPUDeviceName']) is not str:
        raise TypeError('The argument DefaultGPUDeviceName should be str')

    NumberSensors=arg['IndexSensorMap'].size
    NumberSnapshots=arg['SnapshotsPos'].size
    
    ZoneCount=arg['SPP_ZONES']
    
    NumberSources=arg['SourceFunctions'].shape[0]
    TimeStepsSource=arg['SourceFunctions'].shape[1]
    
    if TimeStepsSource != arg['LengthSource']:
        raise ValueError("The limit for time steps in source is different from N-dimension in SourceFunctions")
    if arg['N1']+1 != arg['MaterialMap'].shape[0]:
		ERROR_STRING("Material map dim 0 must be N1+1");
	if arg['N2']+1 != arg['MaterialMap))
			ERROR_STRING("Material map dim 1 must be N2+1");
	if arg['N3']+1 != arg['MaterialMap))
			ERROR_STRING("Material map dim 3 must be N3+1 ");
	if arg['ZoneCount'] != arg['MaterialMap))
			ERROR_STRING("Material map dim 4 must be ZoneCount ");

	if ( !((INHOST(SelRMSorPeak))& SEL_PEAK) && !((INHOST(SelRMSorPeak))&SEL_RMS))
			ERROR_STRING("SelRMSorPeak must be either 1 (RMS), 2 (Peak) or 3 (Both RMS and Peak)");


    

