import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pytest

# Ensure that we don't use the local version of BabelViscoFDTD as this will fail through a missing library error
working_dir = os.getcwd()
if working_dir in sys.path:
    sys.path.remove(working_dir)

# Grab BabelViscoFDTD from environment
from BabelViscoFDTD.tools.RayleighAndBHTE import BHTE, InitCuda,InitOpenCL,InitMetal, InitMLX

def test_BHTE_no_source(spatial_step,bioheat_exact,set_up_domain,compare_data,request,get_mpl_plot,computing_backend,get_gpu_device,tolerance):
    
    # =========================================================================
    # DOMAIN SETUP
    # =========================================================================

    # Create grid
    create_grid = set_up_domain['grid']
    X, Y, Z = create_grid(grid_dims = [128, 128, 128], grid_steps = 3*[spatial_step])
    
    # Set homogeneous brain medium
    set_medium = set_up_domain['medium']
    medium, medium_index = set_medium(medium_type='water')
    
    # Set Gaussian initial temperature distribution [degC]
    width = X.max()//6
    source = {}
    source['T0'] = (37 + 5 * np.exp( -(X / width)**2 - (Y / width)**2 - (Z / width)**2))
    
    # Time parameters
    Nt = 300 # number of time steps
    dt = 0.5 # time step
    
    # =========================================================================
    # SIMULATION USING BIOHEATEXACT
    # =========================================================================
    
    # Calculate diffusivity coefficient for medium
    D = bioheat_exact['diffusivity'](medium)
    
    # Compute Green's function solution using bioheatExact
    t0  = time.perf_counter()
    temp_exact = bioheat_exact['bioheat'](source['T0'], 0, [D, 0, 0], spatial_step, Nt * dt)
    t1 = time.perf_counter()
    logging.info(f"Truth method took {t1 - t0} s")
    
    # =========================================================================
    # SIMULATION USING BABELVISCOFDTD'S BHTE
    # =========================================================================
    
    # BHTE parameters
    nFactorMonitoring=int(2.5/dt) # Monitor every 2.5s
    pressure = np.zeros_like(source['T0'])
    MaterialMap = medium_index * np.ones_like(source['T0'],dtype=np.uint32)
    MaterialList = set_up_domain['material_list']()
    
    # Initialize GPU
    gpu_device = get_gpu_device()
    if computing_backend['type'] == "CUDA":
        InitCuda(gpu_device)
    elif computing_backend['type'] == "OpenCL":
        InitOpenCL(gpu_device)
    elif computing_backend['type'] == "Metal":
        InitMetal(gpu_device)
    elif computing_backend['type'] == "MLX":
        InitMLX(gpu_device)
    else:
        raise ValueError("Not sure what computing backend was chosen")
    
    # Run BHTE 
    t0  = time.perf_counter()
    temp_babelvisco,_,_,_ = BHTE(pressure,
                                MaterialMap,
                                MaterialList,
                                dx = spatial_step,
                                TotalDurationSteps=Nt,
                                nStepsOn=0,
                                LocationMonitoring=64,
                                nFactorMonitoring=nFactorMonitoring,
                                dt=dt,
                                DutyCycle=0.0,
                                Backend=computing_backend['type'],
                                stableTemp=37.0,
                                initT0=source['T0'].astype(np.float32))
    t1 = time.perf_counter()
    logging.info(f"BabelViscoFDTD BHTE method took {t1 - t0} s")

    # =========================================================================
    # VISUALISATION
    # =========================================================================
            
    # Save plot screenshot to be added to html report later
    request.node.screenshots = []
    plots = [source['T0'].T,temp_exact.T,temp_babelvisco.T,abs(temp_babelvisco.T-temp_exact.T)]
    plot_names = ['Initial Temp','bioheat_exact\noutput', 'babelvisco\nBHTE output','Output\nDifferences']
    screenshot = get_mpl_plot(plots, axes_num=3,titles=plot_names,color_map=plt.cm.jet,colorbar=True)
    request.node.screenshots.append(screenshot)
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    calc_dice_coeff = compare_data['dice_coefficient']
    dice_coeff = calc_dice_coeff(temp_exact,temp_babelvisco,rel_tolerance=tolerance)
    
    assert dice_coeff == pytest.approx(1.0, rel=1e-9), f"DICE score is not 1 ({dice_coeff})"

def test_BHTE_no_source_with_perfusion(spatial_step,set_up_domain,bioheat_exact,compare_data,request,get_mpl_plot,computing_backend,get_gpu_device,tolerance):
    
    # =========================================================================
    # DOMAIN SETUP
    # =========================================================================

    # Create grid
    create_grid = set_up_domain['grid']
    X, Y, Z = create_grid(grid_dims = [128, 128, 128], grid_steps = 3*[spatial_step])
    
    # Set homogeneous brain medium
    set_medium = set_up_domain['medium']
    medium, medium_index = set_medium(medium_type='brain')
    
    # Set Gaussian initial temperature distribution [degC]
    width = X.max()//6
    source = {}
    source['T0'] = (37 + 5 * np.exp( -(X / width)**2 - (Y / width)**2 - (Z / width)**2))
    
    # Time parameters
    Nt = 300 # number of time steps
    dt = 0.5 # time step
    
    # =========================================================================
    # SIMULATION USING BIOHEATEXACT
    # =========================================================================
    
    # Calculate diffusivity coefficient for medium
    D = bioheat_exact['diffusivity'](medium)
    
    # Calculate perfusion coefficient for medium
    P = bioheat_exact['perfusion'](medium)
    
    # Compute Green's function solution using bioheatExact
    t0  = time.perf_counter()
    temp_exact = bioheat_exact['bioheat'](source['T0'], 0, [D, P, 37.0], spatial_step, Nt * dt)
    t1 = time.perf_counter()
    logging.info(f"Truth method took {t1 - t0} s")
    
    # =========================================================================
    # SIMULATION USING BABELVISCOFDTD'S BHTE
    # =========================================================================
    
    # BHTE parameters
    nFactorMonitoring=int(2.5/dt) # Monitor every 2.5s
    pressure = np.zeros_like(source['T0'])
    MaterialMap = medium_index * np.ones_like(source['T0'],dtype=np.uint32)
    MaterialList = set_up_domain['material_list']()
    
    # Initialize GPU
    gpu_device = get_gpu_device()
    if computing_backend['type'] == "CUDA":
        InitCuda(gpu_device)
    elif computing_backend['type'] == "OpenCL":
        InitOpenCL(gpu_device)
    elif computing_backend['type'] == "Metal":
        InitMetal(gpu_device)
    elif computing_backend['type'] == "MLX":
        InitMLX(gpu_device)
    else:
        raise ValueError("Not sure what computing backend was chosen")
    
    # Run BHTE 
    t0  = time.perf_counter()
    temp_babelvisco,_,_,_ = BHTE(pressure,
                                MaterialMap,
                                MaterialList,
                                dx = spatial_step,
                                TotalDurationSteps=Nt,
                                nStepsOn=0,
                                LocationMonitoring=64,
                                nFactorMonitoring=nFactorMonitoring,
                                dt=dt,
                                DutyCycle=0.0,
                                Backend=computing_backend['type'],
                                stableTemp=37.0,
                                initT0=source['T0'].astype(np.float32))
    t1 = time.perf_counter()
    logging.info(f"BabelViscoFDTD BHTE method took {t1 - t0} s")

    # =========================================================================
    # VISUALISATION
    # =========================================================================
            
    # Save plot screenshot to be added to html report later
    request.node.screenshots = []
    plots = [source['T0'].T,temp_exact.T,temp_babelvisco.T,abs(temp_babelvisco.T-temp_exact.T)]
    plot_names = ['Initial Temp','bioheat_exact\noutput', 'babelvisco\nBHTE output','Output\nDifferences']
    screenshot = get_mpl_plot(plots, axes_num=3,titles=plot_names,color_map=plt.cm.jet,colorbar=True)
    request.node.screenshots.append(screenshot)
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    calc_dice_coeff = compare_data['dice_coefficient']
    dice_coeff = calc_dice_coeff(temp_exact,temp_babelvisco,rel_tolerance=tolerance)
    
    assert dice_coeff == pytest.approx(1.0, rel=1e-9), f"DICE score is not 1 ({dice_coeff})"
    
def test_BHTE_source_with_perfusion(spatial_step,set_up_domain,bioheat_exact,compare_data,request,get_mpl_plot,computing_backend,get_gpu_device,tolerance):
    
    # =========================================================================
    # DOMAIN SETUP
    # =========================================================================

    # Create grid
    create_grid = set_up_domain['grid']
    X, Y, Z = create_grid(grid_dims = [128, 128, 128], grid_steps = 3*[spatial_step])
    
    # Set homogeneous brain medium
    set_medium = set_up_domain['medium']
    medium, medium_index = set_medium(medium_type='brain')
    
    # Set initial temperature distribution to be constant [degC]
    source = {}
    source['T0'] = 37.0 * np.ones_like(X)
    
    # Set ultrasound parameters
    duty_cycle = 1.0
    
    # Set Gaussian volume rate of heat deposition [W/m^3]
    width = X.max()//6
    source['Q'] = 2e6 * np.exp( -(X / width)**2 - (Y / width)**2 - (Z / width)**2 )
    pressure = np.sqrt(source['Q'] * (2*medium['density']*medium['sos']*spatial_step) / duty_cycle) # pressure values needed for Babelvisco BHTE to produce same heat disposition
    
    # Time parameters
    Nt = 300 # number of time steps
    dt = 0.5 # time step
    
    # =========================================================================
    # SIMULATION USING BIOHEATEXACT
    # =========================================================================
    
    # Calculate diffusivity coefficient for medium
    D = bioheat_exact['diffusivity'](medium)
    
    # Calculate perfusion coefficient for medium
    P = bioheat_exact['perfusion'](medium)
    
    # Calculate normalized heat source
    S = bioheat_exact['heat_source'](medium,source)
    
    # Compute Green's function solution using bioheatExact
    t0  = time.perf_counter()
    temp_exact = bioheat_exact['bioheat'](source['T0'], S, [D, P, 37.0], spatial_step, Nt * dt)
    t1 = time.perf_counter()
    logging.info(f"Truth method took {t1 - t0} s")
    
    # =========================================================================
    # SIMULATION USING BABELVISCOFDTD'S BHTE
    # =========================================================================
    
    # BHTE parameters
    nFactorMonitoring=int(2.5/dt) # Monitor every 2.5s
    MaterialMap = medium_index * np.ones_like(source['T0'],dtype=np.uint32)
    MaterialList = set_up_domain['material_list']()
    
    # Initialize GPU
    gpu_device = get_gpu_device()
    if computing_backend['type'] == "CUDA":
        InitCuda(gpu_device)
    elif computing_backend['type'] == "OpenCL":
        InitOpenCL(gpu_device)
    elif computing_backend['type'] == "Metal":
        InitMetal(gpu_device)
    elif computing_backend['type'] == "MLX":
        InitMLX(gpu_device)
    else:
        raise ValueError("Not sure what computing backend was chosen")
    
    # Run BHTE 
    t0  = time.perf_counter()
    temp_babelvisco,_,_,_ = BHTE(pressure,
                                MaterialMap,
                                MaterialList,
                                dx = spatial_step,
                                TotalDurationSteps=Nt,
                                nStepsOn=Nt,
                                LocationMonitoring=64,
                                nFactorMonitoring=nFactorMonitoring,
                                dt=dt,
                                DutyCycle=duty_cycle,
                                Backend=computing_backend['type'],
                                stableTemp=37.0,
                                initT0=source['T0'].astype(np.float32))
    t1 = time.perf_counter()
    logging.info(f"BabelViscoFDTD BHTE method took {t1 - t0} s")

    # =========================================================================
    # VISUALISATION
    # =========================================================================
            
    # Save plot screenshot to be added to html report later
    request.node.screenshots = []
    plots = [source['T0'].T,temp_exact.T,temp_babelvisco.T,abs(temp_babelvisco.T-temp_exact.T)]
    plot_names = ['Initial Temp','bioheat_exact\noutput', 'babelvisco\nBHTE output','Output\nDifferences']
    screenshot = get_mpl_plot(plots, axes_num=3,titles=plot_names,color_map=plt.cm.jet,colorbar=True)
    request.node.screenshots.append(screenshot)
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    calc_dice_coeff = compare_data['dice_coefficient']
    dice_coeff = calc_dice_coeff(temp_exact,temp_babelvisco,rel_tolerance=tolerance)
    
    assert dice_coeff == pytest.approx(1.0, rel=1e-9), f"DICE score is not 1 ({dice_coeff})"
    
pytest.mark.skip("Placeholder test")
def test_BHTEMultiplePressureFields():
    pass