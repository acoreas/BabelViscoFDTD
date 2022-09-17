from calendar import c
from multiprocessing.dummy import Array
from unicodedata import name
import numpy as np
import os
from pathlib import Path
import platform
import time
import tempfile
from shutil import copyfile

from .StaggeredFDTD_3D_With_Relaxation_BASE import StaggeredFDTD_3D_With_Relaxation_BASE

from distutils.sysconfig import get_python_inc
from math import ceil
import ctypes
from ctypes import c_byte, c_int, c_int64, c_uint32, c_float, c_wchar_p, c_uint64

mc = None

#this is just a copy the same structure used in the kernel, but it helps to identify which entry in the Argument Buffer as in Metal we can't reuse the same thing in Swift
StructMetal =\
'''     unsigned int  N1         ;
        unsigned int  N2                 ;
        unsigned int  N3                 ;
        unsigned int  Limit_I_low_PML    ;
        unsigned int  Limit_J_low_PML    ;
        unsigned int  Limit_K_low_PML    ;
        unsigned int  Limit_I_up_PML     ;
        unsigned int  Limit_J_up_PML     ;
        unsigned int  Limit_K_up_PML     ;
        unsigned int  SizeCorrI          ;
        unsigned int  SizeCorrJ          ;
        unsigned int  SizeCorrK          ;
        unsigned int  PML_Thickness      ;
        unsigned int  NumberSources      ;
        unsigned int  LengthSource       ;
        unsigned int  NumberSensors      ;
        unsigned int  TimeSteps          ;
        unsigned int  SizePML            ;
        unsigned int  SizePMLxp1         ;
        unsigned int  SizePMLyp1         ;
        unsigned int  SizePMLzp1         ;
        unsigned int  SizePMLxp1yp1zp1   ;
        unsigned int  ZoneCount          ;
        unsigned int  SensorSubSampling;
        unsigned int  SensorStart      ;
        unsigned int  nStep            ;
        unsigned int  CurrSnap         ;
        unsigned int  TypeSource       ;
        unsigned int  SelK             ;
        unsigned int  SelRMSorPeak;
        unsigned int  SelMapsRMSPeak;
        unsigned int  NumberSelRMSPeakMaps;
        unsigned int SelMapsSensors;
        unsigned int IndexRMSPeak_ALLV ;
        unsigned int IndexRMSPeak_Vx ;
        unsigned int IndexRMSPeak_Vy ;
        unsigned int IndexRMSPeak_Vz ;
        unsigned int IndexRMSPeak_Sigmaxx ;
        unsigned int IndexRMSPeak_Sigmayy ;
        unsigned int IndexRMSPeak_Sigmazz ;
        unsigned int IndexRMSPeak_Sigmaxy ;
        unsigned int IndexRMSPeak_Sigmaxz ;
        unsigned int IndexRMSPeak_Sigmayz ;
        unsigned int IndexRMSPeak_Pressure ;  
         unsigned int  IndexSensor_ALLV ;
         unsigned int  IndexSensor_Vx ;
         unsigned int  IndexSensor_Vy ;
         unsigned int  IndexSensor_Vz ;
         unsigned int  IndexSensor_Sigmaxx;
         unsigned int  IndexSensor_Sigmayy;
         unsigned int  IndexSensor_Sigmazz;
         unsigned int  IndexSensor_Sigmaxy;
         unsigned int  IndexSensor_Sigmaxz;
         unsigned int  IndexSensor_Sigmayz;
         unsigned int  IndexSensor_Pressure;
        float DT               ;
        const device float * InvDXDTplus_pr   ;
        const device float * DXDTminus_pr     ;
        const device float * InvDXDTplushp_pr ;
        const device float * DXDTminushp_pr   ;
        const device unsigned int *  IndexSensorMap_pr;   
        const device unsigned int *  SourceMap_pr;		  
        const device unsigned int *  MaterialMap_pr;
        const device float *  SourceFunctions_pr;  
        const device float *  LambdaMiuMatOverH_pr;
        const device float *  LambdaMatOverH_pr;   
        const device float *  MiuMatOverH_pr;      
        const device float *  TauLong_pr;          
        const device float *  OneOverTauSigma_pr;  
        const device float *  TauShear_pr;         
        const device float *  InvRhoMatH_pr;       
        const device float *  Ox_pr;               
        const device float *  Oy_pr;               
        const device float *  Oz_pr;	
        device float *  V_x_x_pr;
        device float *  V_y_x_pr;
        device float *  V_z_x_pr;
        device float *  V_x_y_pr;
        device float *  V_y_y_pr;
        device float *  V_z_y_pr;
        device float *  V_x_z_pr;
        device float *  V_y_z_pr;
        device float *  V_z_z_pr;
        device float *  Vx_pr; 
        device float *  Vy_pr; 
        device float *  Vz_pr; 
        device float *  Rxx_pr;
        device float *  Ryy_pr;
        device float *  Rzz_pr;
        device float *  Rxy_pr;
        device float *  Rxz_pr;
        device float *  Ryz_pr;
        device float *  Sigma_x_xx_pr;
        device float *  Sigma_y_xx_pr;
        device float *  Sigma_z_xx_pr;
        device float *  Sigma_x_yy_pr;
        device float *  Sigma_y_yy_pr;
        device float *  Sigma_z_yy_pr;
        device float *  Sigma_x_zz_pr;
        device float *  Sigma_y_zz_pr;
        device float *  Sigma_z_zz_pr;
        device float *  Sigma_x_xy_pr;
        device float *  Sigma_y_xy_pr;
        device float *  Sigma_x_xz_pr;
        device float *  Sigma_z_xz_pr;
        device float *  Sigma_y_yz_pr;
        device float *  Sigma_z_yz_pr;
        device float *  Sigma_xy_pr;  
        device float *  Sigma_xz_pr;  
        device float *  Sigma_yz_pr;  
        device float *  Sigma_xx_pr;  
        device float *  Sigma_yy_pr;  
        device float *  Sigma_zz_pr;               
        device float *  Pressure_pr;         
        device float *  SqrAcc_pr;           
        device float *  SensorOutput_pr '''
StructMetal=StructMetal.split(';')

class StaggeredFDTD_3D_With_Relaxation_METAL_own_swift(StaggeredFDTD_3D_With_Relaxation_BASE):
    '''
    This version is mainly for X64 and AMD processors using our initial implementation, which runs faster with that old version
    '''
    def __init__(self, arguments):
        #Begin with initializing Swift Functions, etc.
        os.environ['__BabelMetal'] =(os.path.dirname(os.path.abspath(__file__))+os.sep+'tools')
        print(os.environ['__BabelMetal'])
        os.environ['__BabelMetalDevice'] = arguments['DefaultGPUDeviceName']
        print('loading',os.path.dirname(os.path.abspath(__file__))+"/tools/libFDTDSwift.dylib") 
        self.swift_fun = ctypes.CDLL(os.path.dirname(os.path.abspath(__file__))+"/tools/libFDTDSwift.dylib")
        # Definition of some constants, etc
        self.MAX_SIZE_PML = 101
        self._c_mex_type = np.zeros(12, np.uint64)
        self._c_uint_type = np.uint64(0)
        self.HOST_INDEX_MEX = np.zeros((53, 2), np.uint64)
        self.HOST_INDEX_UINT = np.zeros((3, 2), np.uint64)
        self.LENGTH_INDEX_MEX = 53
        self.LENGTH_INDEX_UINT = 3
        self.ZoneCount = arguments['SPP_ZONES']

        self.C_IND = {
            "N1":0, "N2":1, "N3":2, "Limit_I_low_PML":3, "Limit_J_low_PML":4, "Limit_K_low_PML":5, "Limit_I_up_PML":6, "Limit_J_up_PML":7, "Limit_K_up_PML":8, 
            "SizeCorrI":9, "SizeCorrJ":10, "SizeCorrK":11, "PML_Thickness":12, "NumberSources":13, "NumberSensors":14, "TimeSteps":15, 
            "SizePML":16, "SizePMLxp1":17, "SizePMLyp1":18, "SizePMLzp1":19, "SizePMLxp1yp1zp1":20, "ZoneCount":21, "SelRMSorPeak":22, "SelMapsRMSPeak":23, "IndexRMSPeak_ALLV":24, 
            "IndexRMSPeak_Vx":25, "IndexRMSPeak_Vy":26, "IndexRMSPeak_Vz":27, "IndexRMSPeak_Sigmaxx":28, "IndexRMSPeak_Sigmayy":29, "IndexRMSPeak_Sigmazz":30,
            "IndexRMSPeak_Sigmaxy":31, "IndexRMSPeak_Sigmaxz":32, "IndexRMSPeak_Sigmayz":33, "NumberSelRMSPeakMaps":34, "SelMapsSensors":35, "IndexSensor_ALLV":36,
            "IndexSensor_Vx":37, "IndexSensor_Vy":38, "IndexSensor_Vz":39, "IndexSensor_Sigmaxx":40, "IndexSensor_Sigmayy":41, "IndexSensor_Sigmazz":42, "IndexSensor_Sigmaxy":43,
            "IndexSensor_Sigmaxz":44, "IndexSensor_Sigmayz":45, "NumberSelSensorMaps":46, "SensorSubSampling":47, "nStep":48, "TypeSource":49, "CurrSnap":50, "LengthSource":51, "SelK":52,
            "IndexRMSPeak_Pressure":53, "IndexSensor_Pressure":54, "SensorStart":55,
            "IndexSensorMap":0, "SourceMap":1, "MaterialMap": 2,
            # MEX
            "DT":0, "InvDXDTplus":1, "DXDTminus":1+self.MAX_SIZE_PML, "InvDXDTplushp":1+self.MAX_SIZE_PML*2, "DXDTminushp":1+self.MAX_SIZE_PML*3,
            "V_x_x":0, "V_y_x":1, "V_z_x":2, "V_x_y":3, "V_y_y":4, "V_z_y":5, "V_x_z":6, "V_y_z":7, "V_z_z":8, "Vx":9, "Vy":10, "Vz":11, "Rxx":12, "Ryy":13, "Rzz":14, "Rxy":15, "Rxz":16, "Ryz":17,
            "Sigma_x_xx":18, "Sigma_y_xx":19, "Sigma_z_xx":20, "Sigma_x_yy":21, "Sigma_y_yy":22, "Sigma_z_yy":23, "Sigma_x_zz":24, "Sigma_y_zz":25, "Sigma_z_zz":26, 
            "Sigma_x_xy":27, "Sigma_y_xy":28, "Sigma_x_xz":29, "Sigma_z_xz":30, "Sigma_y_yz":31, "Sigma_z_yz":32, "Sigma_xy":33, "Sigma_xz":34, "Sigma_yz":35, "Sigma_xx":36, "Sigma_yy":37, "Sigma_zz": 38,
            "SourceFunctions":39, "LambdaMiuMatOverH":40, "LambdaMatOverH":41, "MiuMatOverH":42, "TauLong":43, "OneOverTauSigma":44, "TauShear":45, "InvRhoMatH":46, "Ox":47, "Oy":48, "Oz":49,
            "Pressure":50, "SqrAcc":51, "SensorOutput":52, 
            }
        #we udpate the indexes to match the Arguments Buffer
        for k in self.C_IND:
            for n,l in enumerate(StructMetal):
                if 'InvDXDTplus_pr' in l:
                    self._FirstArrayEntry=n
                if k in l:
                    if k in ['Vx','Vy','Vz','Pressure'] and 'Index' in l:
                        continue
                    self.C_IND[k]=n
                    break
            assert(n<len(StructMetal))
        
        print('self.C_IND\n',self.C_IND)


        self.FUNCTION_LOCALS = {}
        for i in ['MAIN_1', "PML_1", "PML_2", "PML_3", "PML_4", "PML_5", "PML_6"]:
            self.FUNCTION_LOCALS[i] = {'STRESS':[0, 0, 0], 'PARTICLE':[0, 0, 0]}
        self.FUNCTION_GLOBALS = {}
        for i in ['MAIN_1', "PML_1", "PML_2", "PML_3", "PML_4", "PML_5", "PML_6"]:
            self.FUNCTION_GLOBALS[i] = {'STRESS':[0, 0, 0], 'PARTICLE':[0, 0, 0]}
                
        self.LENGTH_CONST_UINT = 56
        self.LENGTH_CONST_MEX = 1+self.MAX_SIZE_PML*4

        # Defines functions sent to Swift
        self.swift_fun.InitializeMetalDevices.argtypes = []
        self.swift_fun.SymbolInitiation_uint.argtypes = [
            c_uint32,
            c_uint32]
        self.swift_fun.SymbolInitiation_mex.argtypes = [
            c_uint32,
            c_float]
        self.swift_fun.ownGpuCalloc.argtypes = [
            c_uint32,
            c_int64]
        self.swift_fun.CreateAndCopyFromMXVarOnGPUMex.argtypes = [
            c_uint32,
            c_int64,
            ctypes.POINTER(c_float)]
        self.swift_fun.CreateAndCopyFromMXVarOnGPUUINT.argtypes = [
            c_uint32,
            c_int64,
            ctypes.POINTER(c_uint32)]
        self.swift_fun.SyncInput.argtypes=[]

        self.swift_fun.GetMaxTotalThreadsPerThreadgroup.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.swift_fun.GetMaxTotalThreadsPerThreadgroup.restype = ctypes.c_uint32
        self.swift_fun.GetThreadExecutionWidth.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.swift_fun.GetThreadExecutionWidth.restype = ctypes.c_uint32 
        extra_params = {"BACKEND":"METAL"}
        super().__init__(arguments, extra_params)
        
    def _PostInitScript(self, arguments, extra_params):
        print("Attempting Metal Initiation...")
        if self.swift_fun.InitializeMetalDevices() == -1:
            raise ValueError("Something has gone horribly wrong.")
        # if self.swift_fun.ConstantBuffers(c_int(self.LENGTH_CONST_UINT), c_int(self.LENGTH_CONST_MEX)) == -1:
        #     raise ValueError("Something has gone horribly wrong")
    

    def _InitSymbol(self, IP,_NameVar,td, SCode):
        if td == "float":
            self.swift_fun.SymbolInitiation_mex(c_uint32(self.C_IND[_NameVar]), c_float(IP[_NameVar]))
        elif td == "unsigned int": 
            self.swift_fun.SymbolInitiation_uint(c_uint32(self.C_IND[_NameVar]), c_uint32(IP[_NameVar]))
        else:
            raise ValueError("Something was passed incorrectly in symbol initiation.")
        
    
    def _InitSymbolArray(self, IP,_NameVar,td, SCode):
        self._CreateAndCopyFromMXVarOnGPU(_NameVar,None,IP)
      
    def _ownGpuCalloc(self, Name,td,dims,ArraysGPUOp):
        if Name == "Snapshots":
            pass
        self.swift_fun.ownGpuCalloc(self.C_IND[Name],np.int64(dims*4))
       
    def _CreateAndCopyFromMXVarOnGPU(self, Name,ArraysGPUOp,ArrayResCPU,flags=[]):
        SizeCopy = ArrayResCPU[Name].size
        if Name in ['LambdaMiuMatOverH','LambdaMatOverH','MiuMatOverH',
                    'TauLong','OneOverTauSigma','TauShear',
                    'InvRhoMatH', 'Ox','Oy','Oz', 'SourceFunctions',
                     'SensorOutput','SqrAcc',
                     'InvDXDTplus','DXDTminus','InvDXDTplushp','DXDTminushp']: # float
            self.swift_fun.CreateAndCopyFromMXVarOnGPUMex(c_uint32(self.C_IND[Name]),
                                                         c_int64(ArrayResCPU[Name].size),
                                                          ArrayResCPU[Name].ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
                
        elif Name in ['IndexSensorMap','SourceMap','MaterialMap',]: # unsigned int
            self.swift_fun.CreateAndCopyFromMXVarOnGPUUINT(c_uint32(self.C_IND[Name]),c_int64(ArrayResCPU[Name].size),
                    ArrayResCPU[Name].ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)))
    
    def _PreExecuteScript(self, arguments, ArraysGPUOp, outparams):
        if arguments['ManualLocalSize'][0]!=-1:
            self._SET_USER_LOCAL(arguments['ManualLocalSize'])
        else:
            self._CALC_USER_LOCAL("MAIN_1", "STRESS")
            self._CALC_USER_LOCAL("MAIN_1", "PARTICLE")
        
        for j in ["STRESS", "PARTICLE"]:
            for i in ["PML_1", "PML_2", "PML_3", "PML_4", "PML_5", "PML_6"]:
                self._CALC_USER_LOCAL(i, j)
        
        if arguments['ManualGroupSize'][0] != -1:
            self._SET_USER_GLOBAL(arguments['ManualGroupSize'])
        else:
            self._CALC_USER_GROUP_MAIN(arguments, outparams)

        self. _CALC_USER_GROUP_PML(outparams)

        self.swift_fun.maxThreadSensor.argtypes = []
        self.swift_fun.maxThreadSensor.restype = c_int
        
        self.localSensor = [self.swift_fun.maxThreadSensor(), 1, 1]
        self.globalSensor = [ceil(arguments['IndexSensorMap'].size / self.localSensor[0]), 1, 1]

    def _CALC_USER_LOCAL(self, Name, Type):
        if Type == "STRESS":
            Swift = 0
        elif Type == "PARTICLE":
            Swift = 1
        print(Name, Type)
        w = self.swift_fun.GetThreadExecutionWidth(ctypes.c_char_p(bytes(Name, 'utf-8')), c_int(Swift)) 
        h = self.swift_fun.GetMaxTotalThreadsPerThreadgroup(ctypes.c_char_p(bytes(Name, 'utf-8')), c_int(Swift)) / w
        z = 1
        if h % 2 == 0:
            h = h / 2
            z = 2
        self.FUNCTION_LOCALS[Name][Type][0] = w
        self.FUNCTION_LOCALS[Name][Type][1] = int(h)
        self.FUNCTION_LOCALS[Name][Type][2] = z
        print(Name, "local", Type + " = [" + str(w) + ", " + str(h) + ", " + str(z) + "]")
    
    def _SET_USER_LOCAL(self, ManualLocalSize):
        for Type in ['STRESS', 'PARTICLE']:
            for index in range(3):
                self.FUNCTION_LOCALS['MAIN_1'][Type][index] = ManualLocalSize[index] # Can probably change this
    
    def _SET_USER_GLOBAL(self, ManualGlobalSize):
        for Type in ['STRESS', 'PARTICLE']:
            for index in range(3):
               self.FUNCTION_GLOBALS['MAIN_1'][Type][index] = ManualGlobalSize[index] 

    def _CALC_USER_GROUP_MAIN(self, arguments, outparams):
        self._outparams = outparams
        for Type in ['STRESS', 'PARTICLE']:
            for index in range(3):
                self.FUNCTION_GLOBALS['MAIN_1'][Type][index] = ceil((arguments[('N'+str(index + 1))]-outparams['PML_Thickness']*2) / self.FUNCTION_LOCALS['MAIN_1'][Type][index])
            print("MAIN_1_global_" + Type, "=", str(self.FUNCTION_GLOBALS['MAIN_1'][Type]))
    
    def _CALC_USER_GROUP_PML(self, outparams):
        for Type in ['STRESS', 'PARTICLE']:
            self.FUNCTION_GLOBALS['PML_1'][Type][0] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_1'][Type][0])
            self.FUNCTION_GLOBALS['PML_1'][Type][1] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_1'][Type][1])
            self.FUNCTION_GLOBALS['PML_1'][Type][2] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_1'][Type][2])

            self.FUNCTION_GLOBALS['PML_2'][Type][0] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_2'][Type][0])
            self.FUNCTION_GLOBALS['PML_2'][Type][1] = ceil(outparams['SizeCorrJ'] / self.FUNCTION_LOCALS['PML_2'][Type][1])
            self.FUNCTION_GLOBALS['PML_2'][Type][2] = ceil(outparams['SizeCorrK'] / self.FUNCTION_LOCALS['PML_2'][Type][2])

            self.FUNCTION_GLOBALS['PML_3'][Type][0] = ceil(outparams['SizeCorrI'] / self.FUNCTION_LOCALS['PML_3'][Type][0])
            self.FUNCTION_GLOBALS['PML_3'][Type][1] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_3'][Type][1])
            self.FUNCTION_GLOBALS['PML_3'][Type][2] = ceil(outparams['SizeCorrK'] / self.FUNCTION_LOCALS['PML_3'][Type][2])

            self.FUNCTION_GLOBALS['PML_4'][Type][0] = ceil(outparams['SizeCorrI'] / self.FUNCTION_LOCALS['PML_4'][Type][0])
            self.FUNCTION_GLOBALS['PML_4'][Type][1] = ceil(outparams['SizeCorrJ'] / self.FUNCTION_LOCALS['PML_4'][Type][1])
            self.FUNCTION_GLOBALS['PML_4'][Type][2] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_4'][Type][2])

            self.FUNCTION_GLOBALS['PML_5'][Type][0] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_5'][Type][0])
            self.FUNCTION_GLOBALS['PML_5'][Type][1] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_5'][Type][1])
            self.FUNCTION_GLOBALS['PML_5'][Type][2] = ceil(outparams['SizeCorrK'] / self.FUNCTION_LOCALS['PML_5'][Type][2])

            self.FUNCTION_GLOBALS['PML_6'][Type][0] = ceil(outparams['SizeCorrI'] / self.FUNCTION_LOCALS['PML_6'][Type][0])
            self.FUNCTION_GLOBALS['PML_6'][Type][1] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_6'][Type][1])
            self.FUNCTION_GLOBALS['PML_6'][Type][2] = ceil(outparams['PML_Thickness']*2 / self.FUNCTION_LOCALS['PML_6'][Type][2])
            for i in ["PML_1", "PML_2", "PML_3", "PML_4", "PML_5", "PML_6"]:
                print(i + "_global_" + Type + "=", str(self.FUNCTION_GLOBALS[i][Type]))

    def _InitiateCommands(self, AllC):
        pass

    def _Execution(self, arguments, ArrayResCPU, ArrayResOP):
        TimeSteps = arguments['TimeSteps']
        self.swift_fun.EncoderInit.argtypes = [] # Not sure if this is necessary
        self.swift_fun.EncodeCommit.argtypes = []
        self.swift_fun.EncodeSensors.argtypes = []
        self.swift_fun.SyncChange.argtypes = []
        self.swift_fun.EncodeStress.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        self.swift_fun.EncodeParticle.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        self.swift_fun.CopyFromGPUMEX.argtypes = [
            ctypes.c_uint64 # Can we do this instead of sending pointers of everything?
        ]
        self.swift_fun.CopyFromGPUMEX.restype = ctypes.POINTER(ctypes.c_float)
        self.swift_fun.EncodeSensors.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
    
        InitDict = {'nStep':0, 'TypeSource':int(arguments['TypeSource'])}
        outparams=self._outparams
        DimsKernel={}
        DimsKernel['PML_1']=[outparams['PML_Thickness']*2,outparams['PML_Thickness']*2,outparams['PML_Thickness']*2]
        DimsKernel['PML_2']=[outparams['PML_Thickness']*2,outparams['N2']-outparams['PML_Thickness']*2,outparams['N3']-outparams['PML_Thickness']*2]
        DimsKernel['PML_3']=[outparams['N1']-outparams['PML_Thickness']*2,outparams['PML_Thickness']*2,outparams['N3']-outparams['PML_Thickness']*2]
        DimsKernel['PML_4']=[outparams['N1']-outparams['PML_Thickness']*2,outparams['N2']-outparams['PML_Thickness']*2,outparams['PML_Thickness']*2]
        DimsKernel['PML_5']=[outparams['PML_Thickness']*2,outparams['PML_Thickness']*2,outparams['N3']-outparams['PML_Thickness']*2]
        DimsKernel['PML_6']=[outparams['N1']-outparams['PML_Thickness']*2,outparams['PML_Thickness']*2,outparams['PML_Thickness']*2]
        DimsKernel['MAIN_1']=[outparams['N1']-outparams['PML_Thickness']*2,outparams['N2']-outparams['PML_Thickness']*2,outparams['N3']-outparams['PML_Thickness']*2]
        for k in DimsKernel:
            DimsKernel[k]=[c_uint32(DimsKernel[k][0]),c_uint32(DimsKernel[k][1]),c_uint32(DimsKernel[k][2])]

        for nStep in range(TimeSteps//10):
            InitDict["nStep"] = nStep
            for i in ['nStep', 'TypeSource']:
                self._InitSymbol(InitDict, i, 'unsigned int', [])
            self.swift_fun.SyncInput()
            self.swift_fun.EncoderInit()
            
            for i in ["PML_1", "PML_2", "PML_3", "PML_4", "PML_5", "PML_6", "MAIN_1"]:
                str_ptr = ctypes.c_char_p(bytes(i, 'utf-8'))
                glox_ptr = c_uint32(self.FUNCTION_GLOBALS[i]["STRESS"][0])
                gloy_ptr = c_uint32(self.FUNCTION_GLOBALS[i]["STRESS"][1])
                gloz_ptr = c_uint32(self.FUNCTION_GLOBALS[i]["STRESS"][2])
                locx_ptr = c_uint32(self.FUNCTION_LOCALS[i]["STRESS"][0])
                locy_ptr = c_uint32(self.FUNCTION_LOCALS[i]["STRESS"][1])
                locz_ptr = c_uint32(self.FUNCTION_LOCALS[i]["STRESS"][2])
                dk=DimsKernel[i]
                self.swift_fun.EncodeStress(str_ptr, glox_ptr, gloy_ptr, gloz_ptr, locx_ptr, locy_ptr, locz_ptr,dk[0],dk[1],dk[2])
            
            for i in ["PML_1", "PML_2", "PML_3", "PML_4", "PML_5", "PML_6", "MAIN_1"]:
                str_ptr = ctypes.c_char_p(bytes(i, 'utf-8'))
                glox_ptr = c_uint32(self.FUNCTION_GLOBALS[i]["PARTICLE"][0])
                gloy_ptr = c_uint32(self.FUNCTION_GLOBALS[i]["PARTICLE"][1])
                gloz_ptr = c_uint32(self.FUNCTION_GLOBALS[i]["PARTICLE"][2])
                locx_ptr = c_uint32(self.FUNCTION_LOCALS[i]["PARTICLE"][0])
                locy_ptr = c_uint32(self.FUNCTION_LOCALS[i]["PARTICLE"][1])
                locz_ptr = c_uint32(self.FUNCTION_LOCALS[i]["PARTICLE"][2])
                dk=DimsKernel[i]
                self.swift_fun.EncodeParticle(str_ptr, glox_ptr, gloy_ptr, gloz_ptr, locx_ptr, locy_ptr, locz_ptr,dk[0],dk[1],dk[2])
            self.swift_fun.EncodeCommit()
            if (nStep % arguments['SensorSubSampling'])==0  and (int(nStep/arguments['SensorSubSampling'])>=arguments['SensorStart']):

                glox_ptr = c_uint32(self.globalSensor[0])
                gloy_ptr = c_uint32(self.globalSensor[1])
                gloz_ptr = c_uint32(self.globalSensor[2])
                locx_ptr = c_uint32(self.localSensor[0])
                locy_ptr = c_uint32(self.localSensor[1])
                locz_ptr = c_uint32(self.localSensor[2])
            
                self.swift_fun.EncodeSensors(glox_ptr, gloy_ptr, gloz_ptr, locx_ptr, locy_ptr, locz_ptr)

        self.swift_fun.SyncChange()

        for i in ['SqrAcc', 'SensorOutput']:
            print('readout',i)
            SizeCopy = ArrayResCPU[i].size
            Shape = ArrayResCPU[i].shape
            tempArray = (ctypes.c_float * SizeCopy)()
            Buffer = self.swift_fun.CopyFromGPUMEX(c_uint64(self.C_IND[i]))
            ctypes.memmove(tempArray, Buffer, SizeCopy * 4) # Metal only supports single precision
            tempArray = np.ctypeslib.as_array(tempArray)
            ArrayResCPU[i][:,:,:] = np.reshape(tempArray, Shape,order='F')

        for i in ['Vx', 'Vy', 'Vz', 'Sigma_xx', 'Sigma_yy', 'Sigma_zz', 'Sigma_xy', 'Sigma_xz', 'Sigma_yz', 'Pressure']:
            print('readout',i,self.C_IND[i]-self._FirstArrayEntry)
            SizeCopy = ArrayResCPU[i].size * self.ZoneCount
            sz=ArrayResCPU[i].shape
            Shape = (sz[0],sz[1],sz[2],self.ZoneCount)
            tempArray = (ctypes.c_float *SizeCopy)()
            Buffer = self.swift_fun.CopyFromGPUMEX(c_uint64(self.C_IND[i]))
            ctypes.memmove(tempArray, Buffer, SizeCopy * 4)
            tempArray = np.ctypeslib.as_array(tempArray)
            tempArray=np.reshape(tempArray,Shape,order='F')
            ArrayResCPU[i][:,:,:] = np.sum(tempArray,axis=3)/self.ZoneCount
          

        self.swift_fun.freeGPUextern.argtypes = []
        self.swift_fun.freeGPUextern.restype = None

        self.swift_fun.freeGPUextern()

def StaggeredFDTD_3D_METAL(arguments):
    # if 'arm64' in platform.platform():
    #     Instance = StaggeredFDTD_3D_With_Relaxation_METAL_MetalCompute(arguments)
    # else:
    Instance = StaggeredFDTD_3D_With_Relaxation_METAL_own_swift(arguments)
    Results = Instance.Results
    return Results
