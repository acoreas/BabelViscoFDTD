import logging
import matplotlib.pyplot as plt
import numpy as np
import pytest

@pytest.fixture()
def set_up_domain():
    def _get_material_list_bhte():
        
        material_list = {}                 #Water    #Water      #Blood      #Brain      #Skull      #Skin
        material_list['Density']         = [1000.0,  1000.0,     1050.0,     1041.0,     1041.0,     1041.0,    ]   # (kg/m3)
        material_list['SoS']             = [1500.0,  1500.0,     1570.0,     1562.0,     1562.0,     1562.0,    ]   # (m/s)
        material_list['Attenuation']     = [0.0,     0.0,        0.0,        3.45,       3.45,       3.45,      ]   # (Np/m)
        material_list['SpecificHeat']    = [4178.0,  4178.0,     3617.0,     3630.0,     3630.0,     3630.0,    ]   # (J/kg/°C)
        material_list['Conductivity']    = [0.6,     0.6,        0.52,       0.51,       0.51,       0.51,      ]   # (W/m/°C)
        material_list['Perfusion']       = [0.0,     0.0,        10000.0,    559.0,      559.0,      559.0,     ]   # (ml/min/kg)
        material_list['Absorption']      = [0.0,     0.0,        0.0,        0.85,       0.85,       0.85,      ]   # Unitless
        material_list['InitTemperature'] = [37.0,    37.0,       37.0,       37.0,       37.0,       37.0,      ]   # (°C)
        # Water material is duplicated since metal compute sims run into issues when
        # trying to run in homogenous material (like water) and its properties are stored in index 0

        material_indices = {"water": 1, "blood": 2, "brain":3, "skull": 4, "skin": 5}
        
        return material_list, material_indices
    
    def _get_material_list_vwe(freq):
        
        def FitSpeedCorticalLong(frequency):
            #from Phys Med Biol. 2011 Jan 7; 56(1): 219–250. doi :10.1088/0031-9155/56/1/014 
            FRef=np.array([270e3,836e3])
            ClRef=np.array([2448.0,2516])
            p=np.polyfit(FRef, ClRef, 1)
            return(np.round(np.poly1d(p)(frequency)))
        
        def FitSpeedCorticalShear(frequency):
            #from Phys Med Biol. 2017 Aug 7; 62(17): 6938–6962. doi: 10.1088/1361-6560/aa7ccc 
            FRef=np.array([270e3,836e3])
            Cs270=np.array([1577.0,1498.0,1313.0]).mean()
            Cs836=np.array([1758.0,1674.0,1545.0]).mean()
            CsRef=np.array([Cs270,Cs836])
            p=np.polyfit(FRef, CsRef, 1)
            return(np.round(np.poly1d(p)(frequency)))
        
        def FitAttCorticalLong_Multiple(frequency,bcoeff=1,reductionFactor=0.8):
            # fitting from data obtained from
            #J. Acoust. Soc. Am., Vol. 64, No. 2,  doi: 10.1121/1.382016
            # Phys Med Biol. 2011 Jan 7; 56(1): 219–250. doi :10.1088/0031-9155/56/1/014
            # IEEE transactions on ultrasonics, ferroelectrics, and frequency control 68, no. 5 (2020): 1532-1545. doi: 10.1109/TUFFC.2020.3039743
            
            return np.round(203.25090263*((frequency/1e6)**bcoeff)*reductionFactor)

        def FitAttBoneShear(frequency,reductionFactor=1.0):
            #from Phys Med Biol. 2017 Aug 7; 62(17): 6938–6962. doi: 10.1088/1361-6560/aa7ccc 
            PichardoData=(57.0/.27 +373/0.836)/2
            return np.round(PichardoData*(frequency/1e6)*reductionFactor) 
        
        def FitSpeedTrabecularLong(frequency):
            #from Phys Med Biol. 2011 Jan 7; 56(1): 219–250. doi :10.1088/0031-9155/56/1/014
            FRef=np.array([270e3,836e3])
            ClRef=np.array([2140.0,2300])
            p=np.polyfit(FRef, ClRef, 1)
            return(np.round(np.poly1d(p)(frequency)))

        def FitSpeedTrabecularShear(frequency):
            #from Phys Med Biol. 2017 Aug 7; 62(17): 6938–6962. doi: 10.1088/1361-6560/aa7ccc 
            FRef=np.array([270e3,836e3])
            Cs270=np.array([1227.0,1365.0,1200.0]).mean()
            Cs836=np.array([1574.0,1252.0,1327.0]).mean()
            CsRef=np.array([Cs270,Cs836])
            p=np.polyfit(FRef, CsRef, 1)
            return(np.round(np.poly1d(p)(frequency)))

        def FitAttTrabecularLong_Multiple(frequency,bcoeff=1,reductionFactor=0.8):
            #reduction factor 
            # fitting from data obtained from
            #J. Acoust. Soc. Am., Vol. 64, No. 2,  doi: 10.1121/1.382016
            # Phys Med Biol. 2011 Jan 7; 56(1): 219–250. doi :10.1088/0031-9155/56/1/014
            # IEEE transactions on ultrasonics, ferroelectrics, and frequency control 68, no. 5 (2020): 1532-1545. doi: 10.1109/TUFFC.2020.3039743
            return np.round(202.76362433*((frequency/1e6)**bcoeff)*reductionFactor) 
        
        material_list={}                        #Density (kg/m3)    LongSoS (m/s),                  ShearSoS (m/s),                 Long Att (Np/m),                        Shear Att (Np/m)
        material_list['water']       = np.array([1000.0,            1500.0,                         0.0,                            0.0,                                    0.0] )
        # material_list['cortical']    = np.array([1896.5,            FitSpeedCorticalLong(freq),     FitSpeedCorticalShear(freq),    FitAttCorticalLong_Multiple(freq),      FitAttBoneShear(freq)])
        material_list['trabecular']  = np.array([1738.0,            FitSpeedTrabecularLong(freq),   FitSpeedTrabecularShear(freq),  FitAttTrabecularLong_Multiple(freq),    FitAttBoneShear(freq)])
        material_list['skin']        = np.array([1116.0,            1537.0,                         0.0,                            2.3*freq/500e3 ,                        0])
        material_list['brain']       = np.array([1041.0,            1562.0,                         0.0,                            3.45*freq/500e3 ,                       0])
    
        return material_list

    def _set_medium_bhte(medium_type='brain'):
        
        # Indices for materials
        MaterialList,material_index_dict = _get_material_list_bhte()
        medium_index = material_index_dict[medium_type]
        blood_index = material_index_dict['blood']
        
        # Define medium properties
        medium = {}
        # Medium properties specific to diffusion
        medium['density']               = MaterialList['Density'][medium_index]
        medium['thermal_conductivity']  = MaterialList['Conductivity'][medium_index]
        medium['specific_heat']         = MaterialList['SpecificHeat'][medium_index]
        # Blood properties specific to perfusion
        medium['blood_density']             = MaterialList['Density'][blood_index]
        medium['blood_specific_heat']       = MaterialList['SpecificHeat'][blood_index]
        medium['blood_perfusion_rate']      = MaterialList['Perfusion'][medium_index]*(1/60)*(1e-6)*medium['density'] # Need units in 1/s for truth method
        medium['blood_ambient_temperature'] = MaterialList['InitTemperature'][blood_index]
        # Medium properties specific to heat disposition
        medium['sos'] = MaterialList['SoS'][medium_index]
        medium['attenuation'] = MaterialList['Attenuation'][medium_index]
        medium['absorption'] = MaterialList['Absorption'][medium_index]
        logging.info(medium)
        
        return medium, medium_index
        
    def _create_grid(grid_limits=[], grid_steps=[], pml_thickness=0):
        """
        Create a 2D or 3D computational grid depending on input dimensions.
        
        Parameters
        ----------
        grid_limits : list
            For 3D: [xmin, xmax, ymin, ymax, zmin, zmax] or [xmax, ymax, zmax]
            For 2D: [xmin, xmax, ymin, ymax] or [xmax, ymax]
        grid_steps : list
            For 3D: [dx, dy, dz]
            For 2D: [dx, dy]
        pml_thickness : int, optional
            Number of grid points to extend on each side for PML (absorbing boundary)
        """
        
        # Determine dimensionality from grid_steps
        dim = len(grid_steps)
        if dim not in (2, 3):
            raise ValueError("grid_steps must have 2 (2D) or 3 (3D) elements.")
    
        # Expand grid_limits depending on input format
        if dim == 3:
            if len(grid_limits) == 6:
                xmin, xmax, ymin, ymax, zmin, zmax = grid_limits
            elif len(grid_limits) == 3:
                xmin = ymin = zmin = 0
                xmax, ymax, zmax = grid_limits
            else:
                raise ValueError("Invalid number of grid limit arguments for 3D grid.")
            dx, dy, dz = grid_steps
        else:  # 2D
            if len(grid_limits) == 4:
                xmin, xmax, ymin, ymax = grid_limits
            elif len(grid_limits) == 2:
                xmin = ymin = 0
                xmax, ymax = grid_limits
            else:
                raise ValueError("Invalid number of grid limit arguments for 2D grid.")
            dx, dy = grid_steps
        
       # Compute grid sizes
        def calc_N(xmin, xmax, d, pml):
            base = np.ceil((xmax - xmin) / d) + 1
            return int(base + 2 * pml) if pml else int(base)

        Nx = calc_N(xmin, xmax, grid_steps[0], pml_thickness)
        Ny = calc_N(ymin, ymax, grid_steps[1], pml_thickness)
        if dim == 3:
            Nz = calc_N(zmin, zmax, grid_steps[2], pml_thickness)
        else:
            Nz = None
        
        # Create 1D coordinate arrays
        x = np.linspace(xmin, xmax, Nx, np.float64)
        y = np.linspace(ymin, ymax, Ny, np.float64)
        if dim == 3:
            z = np.linspace(zmin, zmax, Nz, np.float64)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            return X, Y, Z
        else:
            X, Y = np.meshgrid(x, y, indexing='ij')
            return X, Y
        
    return {'grid': _create_grid,
            'medium_bhte': _set_medium_bhte,
            'material_list_bhte': _get_material_list_bhte,
            'material_list_vwe': _get_material_list_vwe}
    
@pytest.fixture()
def setup_propagation_model(set_up_domain,get_mpl_plot,get_line_plot,request):
    def _setup_propagation_model(us_frequency, points_per_wavelength, axes = 3):
        
        # =============================================================================
        # SIMULATION PARAMETERS
        # =============================================================================
        
        dt = 5e-8                       # time step
        medium_SOS = 1500               # m/s - water
        medium_density = 1000           # kg/m3
        pml_thickness = 12              # grid points for perfect matching layer
        reflection_limit = 1.0000e-05   # reflection parameter for PML
        tx_diameter = 0.03              # m - circular piston
        tx_plane_loc = 0.01             # m - in XY plane at Z = 0.01 m
        us_amplitude = 100e3            # Pa
        
        if axes == 3:
            x_dim = 0.05                # m
            y_dim = 0.05                # m
            z_dim = 0.10                # m
        else:
            x_dim = 0.20                # m
            y_dim = 0.40                # m

        # =============================================================================
        # SIMULATION DOMAIN SETUP
        # =============================================================================
        
        # Domain Properties
        shortest_wavelength = medium_SOS/us_frequency
        spatial_step = shortest_wavelength/ points_per_wavelength

        # Domain Dimensions
        if axes == 3:
            domain_dims =  np.array([x_dim,y_dim,z_dim])  # in m, x,y,z
            
            X,Y,Z = set_up_domain['grid'](grid_limits = [-x_dim/2,x_dim/2,-y_dim/2,y_dim/2,0,z_dim],
                                          grid_steps=3*[spatial_step],
                                          pml_thickness=pml_thickness)
        else:
            domain_dims =  np.array([x_dim,y_dim])  # in m, x and y
            
            X,Y = set_up_domain['grid'](grid_limits = [-x_dim/2,x_dim/2,0,y_dim],
                                        grid_steps=2*[spatial_step],
                                        pml_thickness=pml_thickness)
            
        
        logging.info(f'Domain size: {X.shape}')

        # Time Dimensions
        if axes == 3:
            sim_time = np.sqrt(domain_dims[0]**2+domain_dims[1]**2+domain_dims[2]**2)/medium_SOS #time to cross one corner to another
        else:
            sim_time = np.sqrt(domain_dims[0]**2+domain_dims[1]**2)/medium_SOS #time to cross one corner to another
        sensor_steps = int((1/us_frequency/8)/dt) # for the sensors, we do not need so much high temporal resolution, so we are keeping 8 time points per perioid

        # =============================================================================
        # MATERIAL MAP SETUP
        # =============================================================================
        
        # Retrieve list of different materials (e.g. brain, skull, etc.)
        material_list = set_up_domain['material_list_vwe'](us_frequency)
        material_list = np.array(list(material_list.values()))
        water_index = 0
        skull_index = 1
        skin_index  = 2
        brain_index = 3
        
        # Initialize material map as all water
        material_map = water_index * np.ones_like(X,np.uint32) # Initialize as all water
        
        # Add spheres of different materials at different locations
        def add_material_sphere(material_index,material_radius,material_center,center_offsets):
            material_center[0] += center_offsets[0]
            material_center[1] += center_offsets[1]
            if axes == 3:
                material_center[2] += center_offsets[2]
            
            if axes == 3:
                r = np.sqrt((X - material_center[0])**2 + (Y - material_center[1])**2 + (Z - material_center[2])**2)
            else:
                r = np.sqrt((X - material_center[0])**2 + (Y - material_center[1])**2)
            material_mask = r <= material_radius
            material_map[material_mask] = material_index
        
        if axes == 3:
            mat_radius = tx_diameter/4
            add_material_sphere(skull_index,mat_radius,[0, 0, z_dim/2],center_offsets=[0,-1*mat_radius,0])
            add_material_sphere(brain_index, mat_radius,[0, 0, z_dim/2],center_offsets=[0,mat_radius,2*mat_radius])
            add_material_sphere(skin_index,mat_radius,[0, 0, z_dim/2],center_offsets=[0,mat_radius,-2*mat_radius])
        else:
            mat_radius = tx_diameter
            add_material_sphere(skull_index,mat_radius,[0, y_dim/2],center_offsets=[-1*mat_radius,0])
            add_material_sphere(brain_index, mat_radius,[0, y_dim/2],center_offsets=[mat_radius,2*mat_radius])
            add_material_sphere(skin_index,mat_radius,[0, y_dim/2],center_offsets=[mat_radius,-2*mat_radius])
        
        # =============================================================================
        # GENERATE SOURCE MAP + SIGNAL
        # =============================================================================
        
        # Create source map
        if axes == 3:
            source_mask = (X[:,:,X.shape[2]//2]**2+Y[:,:,Y.shape[2]//2]**2) <= (tx_diameter/2.0)**2
        else:
            source_mask = (X[:,X.shape[1]//2]**2) <= (tx_diameter/2.0)**2
        source_mask = (source_mask*1.0).astype(np.uint32)

        source_map = np.zeros_like(X,np.uint32)
        tx_loc = int(np.round(tx_plane_loc/spatial_step)) + pml_thickness
        
        if axes == 3:
            source_map[:,:,tx_loc] = source_mask
        else:
            source_map[:,tx_loc] = source_mask
            
        # Create particle displacement maps
        amp_displacement = us_amplitude/medium_density/medium_SOS
        Ox = np.zeros_like(X)
        Oy = np.zeros_like(X)
        if axes == 3:
            Oz = np.zeros_like(X)
            Oz[source_map > 0] = 1 #only Z has a value of 1
        else:
            Oy[source_map > 0] = 1 #only Z has a value of 1
            
        Ox *= amp_displacement
        Oy *= amp_displacement
        if axes == 3:
            Oz *= amp_displacement

        # Generate source time signal
        source_length = 4.0/us_frequency # we will use 4 pulses
        source_time_vector = np.arange(0,source_length+dt,dt)

        # Plot source time signal
        pulse_source_tmp = np.sin(2*np.pi*us_frequency*source_time_vector)

        # note we need expressively to arrange the data in a 2D array
        pulse_source = np.reshape(pulse_source_tmp,(1,len(source_time_vector))) 
        print("Number of time points in source signal:",len(source_time_vector))
        
        # =============================================================================
        # GENERATE SENSOR MAP
        # =============================================================================
        
        # Create sensor map
        sensor_map=np.zeros_like(X,np.uint32)
        if axes == 3:
            sensor_map[pml_thickness:-pml_thickness,X.shape[1]//2,pml_thickness:-pml_thickness] = 1
        else:
            sensor_map[pml_thickness:-pml_thickness,pml_thickness:-pml_thickness] = 1

        # =============================================================================
        # SAVE PLOTS
        # =============================================================================

        screenshot = get_mpl_plot([material_map,sensor_map,source_map], axes_num=axes,titles=['Material Map','Sensor Map','Source Map'])
        request.node.screenshots.append(screenshot)
        screenshot = get_line_plot(source_time_vector*1e6,[pulse_source_tmp],title="Source Signal")
        request.node.screenshots.append(screenshot)
        
        # =============================================================================
        # SAVE PARAMETERS TO DICTIONARY
        # =============================================================================
        
        propagation_model_params = {}
        propagation_model_params['material_map'] = material_map
        propagation_model_params['material_list'] = material_list
        propagation_model_params['source_map'] = source_map
        propagation_model_params['pulse_source'] = pulse_source
        propagation_model_params['spatial_step'] = spatial_step
        propagation_model_params['sim_time'] = sim_time
        propagation_model_params['sensor_map'] = sensor_map
        propagation_model_params['Ox'] = Ox
        propagation_model_params['Oy'] = Oy
        if axes == 3:
            propagation_model_params['Oz'] = Oz
        propagation_model_params['pml_thickness'] = pml_thickness
        propagation_model_params['reflection_limit'] = reflection_limit
        propagation_model_params['dt'] = dt
        propagation_model_params['sensor_steps'] = sensor_steps
        
        return propagation_model_params
        
    return _setup_propagation_model