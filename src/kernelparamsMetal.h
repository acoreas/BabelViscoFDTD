#ifndef __METAL_VERSION__

#define DEVICE

#else

#define DEVICE device

#endif


struct ParamsBabel { unsigned int  N1         ;
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
        device float *  SensorOutput_pr;         
};

#ifdef INKERNEL
#define N1                         pp->N1
#define N2                          pp->N2
#define N3                          pp->N3
#define Limit_I_low_PML             pp->Limit_I_low_PML
#define Limit_J_low_PML             pp->Limit_J_low_PML
#define Limit_K_low_PML             pp->Limit_K_low_PML
#define Limit_I_up_PML              pp->Limit_I_up_PML
#define Limit_J_up_PML              pp->Limit_J_up_PML
#define Limit_K_up_PML              pp->Limit_K_up_PML
#define SizeCorrI                   pp->SizeCorrI
#define SizeCorrJ                   pp->SizeCorrJ
#define SizeCorrK                   pp->SizeCorrK
#define PML_Thickness               pp->PML_Thickness
#define NumberSources               pp->NumberSources
#define LengthSource                pp->LengthSource
#define NumberSensors                pp->NumberSensors
#define TimeSteps                   pp->TimeSteps

#define SizePML                     pp->SizePML
#define SizePMLxp1                   pp->SizePMLxp1
#define SizePMLyp1                  pp->SizePMLyp1
#define SizePMLzp1                  pp->SizePMLzp1
#define SizePMLxp1yp1zp1             pp->SizePMLxp1yp1zp1
#define ZoneCount                   pp->ZoneCount

#define SelRMSorPeak                pp->SelRMSorPeak
#define SelMapsRMSPeak              pp->SelMapsRMSPeak
#define NumberSelRMSPeakMaps        pp->NumberSelRMSPeakMaps

#define SelMapsSensors              pp->SelMapsSensors
#define NumberSelSensorMaps         pp->NumberSelSensorMaps
#define SensorSubSampling           pp->SensorSubSampling
#define SensorStart                 pp->SensorStart
#define nStep                       pp->nStep
#define CurrSnap                    pp->CurrSnap
#define TypeSource                  pp->TypeSource
#define SelK                        pp->SelK

#define DT                          pp->DT
#define InvDXDTplus_pr                 pp->InvDXDTplus_pr
#define DXDTminus_pr                   pp->DXDTminus_pr
#define InvDXDTplushp_pr               pp->InvDXDTplushp_pr
#define DXDTminushp_pr                 pp->DXDTminushp_pr

#define IndexRMSPeak_ALLV pp->IndexRMSPeak_ALLV
#define IndexRMSPeak_Vx pp->IndexRMSPeak_Vx
#define IndexRMSPeak_Vy pp->IndexRMSPeak_Vy
#define IndexRMSPeak_Vz pp->IndexRMSPeak_Vz
#define IndexRMSPeak_Sigmaxx pp->IndexRMSPeak_Sigmaxx
#define IndexRMSPeak_Sigmayy pp->IndexRMSPeak_Sigmayy
#define IndexRMSPeak_Sigmazz pp->IndexRMSPeak_Sigmazz
#define IndexRMSPeak_Sigmaxy pp->IndexRMSPeak_Sigmaxy
#define IndexRMSPeak_Sigmaxz pp->IndexRMSPeak_Sigmaxz
#define IndexRMSPeak_Sigmayz pp->IndexRMSPeak_Sigmayz
#define IndexRMSPeak_Pressure pp->IndexRMSPeak_Pressure 

#define IndexSensor_ALLV pp->IndexSensor_ALLV
#define IndexSensor_Vx pp->IndexSensor_Vx
#define IndexSensor_Vy pp->IndexSensor_Vy
#define IndexSensor_Vz pp->IndexSensor_Vz
#define IndexSensor_Sigmaxx pp->IndexSensor_Sigmaxx
#define IndexSensor_Sigmayy pp->IndexSensor_Sigmayy
#define IndexSensor_Sigmazz pp->IndexSensor_Sigmazz
#define IndexSensor_Sigmaxy pp->IndexSensor_Sigmaxy
#define IndexSensor_Sigmaxz pp->IndexSensor_Sigmaxz
#define IndexSensor_Sigmayz pp->IndexSensor_Sigmayz
#define IndexSensor_Pressure pp->IndexSensor_Pressure
#endif