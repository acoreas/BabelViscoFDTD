import logging
import numpy as np
import pytest

@pytest.fixture()
def bioheat_exact():
    def _calc_diffusivity_coeff(medium):
        # Calculate diffusivity from medium parameters
        D = medium['thermal_conductivity'] / (medium['density'] * medium['specific_heat'])
            
        return D
      
    def _calc_perfusion_coeff(medium):
        # calculate perfusion coefficient from the medium parameters
        P = medium['blood_density'] *                                   \
            medium['blood_perfusion_rate'] *                            \
            medium['blood_specific_heat'] /                             \
            (medium['density'] * medium['specific_heat'])
            
        return P
    
    def _calc_heat_source(medium,source):
        # calculate normalised heat source
        S = source['Q'] / (medium['density'] * medium['specific_heat'])
        
        return S
        
    def _bioheat_exact(T0, S, material, dx, t):
        """
        Compute exact solution to Pennes' bioheat equation in homogeneous media.
        This function is modified from the bioheatExact function in the k-Wave Toolbox (https://github.com/ucl-bug/k-wave/tree/main)
        
        DESCRIPTION:
            bioheatExact calculates the exact solution to Pennes' bioheat
            equation in a homogeneous medium on a uniform Cartesian grid using a
            Fourier-based Green's function solution assuming a periodic boundary
            condition [1]. The function supports inputs in 1D, 2D, and 3D. The
            exact equation solved is given by  
        
                dT/dt = D * d^2T/dx^2 - P * (T - Ta) + S
        
            where the coefficients are defined below. Pennes' bioheat equation is
            often given in the alternative form 
        
                P0 * C0 * dT/dt =  Kt * d^2T/dx^2 - Pb * Wb * Cb * (T - Ta) + Q
        
                T:  temperature                     [degC]
                C0: tissue specific heat capacity   [J/(kg.K)]
                P0: tissue density                  [kg/m^3] 
                Kt: tissue thermal conductivity     [W/(m.K)]
                Pb: blood density                   [kg/m^3] 
                Wb: blood perfusion rate            [1/s]
                Ta: blood arterial temperature      [degC]
                Cb: blood specific heat capacity    [J/(kg.K)]
                Q:  volume rate of heat deposition  [W/m^3]
        
            In this case, the function inputs are calculated by
        
                D = Kt / (P0 * C0);
                P = Pb * Wb * Cb / (P0 * C0);
                S = Q / (P0 * C0);
        
            If the perfusion coefficient P is set to zero, bioheatExact
            calculates the exact solution to the heat equation in a homogeneous
            medium.
        
            [1] Gao, B., Langer, S., & Corry, P. M. (1995). Application of the
            time-dependent Green's function and Fourier transforms to the
            solution of the bioheat equation. International Journal of
            Hyperthermia, 11(2), 267-285.
        
        USAGE:
            T = bioheatExact(T0, S, material, dx, t)
            T = bioheatExact(T0, S, [D, P, Ta], dx, t)
        
        INPUTS:
            T0          - matrix of the initial temperature distribution at each
                          grid point [degC] 
            S           - matrix of the heat source at each grid point [K/s]
            material    - material coefficients given as a three element vector
                          in the form: material = [D, P, Ta], where
        
                              D:  diffusion coefficient [m^2/s]
                              P:  perfusion coefficient [1/s]
                              Ta: arterial temperature  [degC]
        
            dx          - grid point spacing [m]
            t           - time at which to calculate the temperature field [s]
        
        OUTPUTS:
            T           - temperature field at time t [degC]
        """
        
        # Check that T0 is a matrix
        ''' 
        Original:
        if numel(T0) == 1
            error('T0 must be defined as a matrix.');
        end
        '''
        if np.isscalar(T0):
            raise ValueError("T0 must be defined as an array, not scalar.")
        
        # If S is not 0, check that S and T0 are the same size
        '''
        Original:
        if ~((numel(S) == 1) && (S == 0)) && ~all(size(S) == size(T0))
            error('T0 and S must be the same size.')
        end
        '''
        if not (np.isscalar(S) and S == 0):
            if np.shape(S) != np.shape(T0):
                raise ValueError("T0 and S must be the same shape.")

        # Extract material properties
        '''
        Original:
        D  = material(1);   % diffusion coefficient
        P  = material(2);   % perfusion coefficient
        Ta = material(3);   % blood arterial temperature
        '''
        D,P,Ta = material
        
        # Check the medium properties are homogeneous
        '''
        Original:
        if numel(P) > 1 || numel(D) > 1 || numel(Ta) > 1
            error('Medium properties must be homogeneous.');
        end
        '''
        if np.ndim(P) > 0 or np.ndim(D) > 0 or np.ndim(Ta) > 0:
            raise ValueError("Material properties must be homogeneous scalars.")

        # Create the k-space grid (like kWaveGrid)
        '''
        Original:
        kgrid = kWaveGrid(size(T0, 1), dx, size(T0, 2), dx, size(T0, 3), dx);
        '''
        shape = T0.shape
        k_axes = []
        for n in shape:
            dk = 2 * np.pi / (n * dx)
            if n % 2 == 0:
                k = np.concatenate([np.arange(0, n//2), np.arange(-n//2, 0)]) * dk
            else:
                k = np.concatenate([np.arange(0, (n-1)//2+1), np.arange(-(n-1)//2, 0)]) * dk
            k_axes.append(k)
        mesh = np.meshgrid(*k_axes, indexing='ij')
        k_squared = sum(m**2 for m in mesh)

        # Define Green's function propagators
        '''
        Original:
        T0_propagator = exp(-(D .* ifftshift(kgrid.k).^2 + P) .* t);
        Q_propagator  = (1 - T0_propagator) ./ (D .* ifftshift(kgrid.k).^2 + P);
        '''
        T0_propagator = np.exp(-(D * k_squared + P) * t)
        Q_propagator = (1 - T0_propagator) / (D * k_squared + P)

        # replace Q propagator with limits for k == 0
        '''
        Original:
        if (numel(P) == 1) && (P == 0)
            Q_propagator(isnan(Q_propagator)) = t;
        else
            Q_propagator(isnan(Q_propagator)) = (1 - exp(-P .* t)) ./ P;
        end
        '''
        if P == 0:
            Q_propagator[np.isnan(Q_propagator)] = t
        else:
            Q_propagator[np.isnan(Q_propagator)] = (1 - np.exp(-P * t)) / P

        # Calculate exact Green's function solution (Eq. 12 [1])
        '''
        Original:
        if (numel(S) == 1) && (S == 0)
            T_change = real(ifftn( T0_propagator .* fftn(T0 - Ta) ));    
        else
            T_change = real(ifftn( T0_propagator .* fftn(T0 - Ta) + Q_propagator .* fftn(S) ));
        end
        '''
        if np.isscalar(S) and S == 0:
            T_change = np.fft.ifftn(T0_propagator * np.fft.fftn(T0 - Ta)).real
        else:
            T_change = np.fft.ifftn(T0_propagator * np.fft.fftn(T0 - Ta) + Q_propagator * np.fft.fftn(S)).real

        return T_change + Ta
        
    return {'bioheat': _bioheat_exact,
            'diffusivity': _calc_diffusivity_coeff,
            'perfusion': _calc_perfusion_coeff,
            'heat_source': _calc_heat_source}

@pytest.fixture()
def set_up_domain():
    def _get_material_list():
        MaterialList = {}                 #Water    #Water      #Blood      #Brain
        MaterialList['Density']         = [1000.0,  1000.0,     1050.0,     1041.0]     # (kg/m3)
        MaterialList['SoS']             = [1500.0,  1500.0,     1570.0,     1562.0]     # (m/s)
        MaterialList['Attenuation']     = [0.0,     0.0,        0.0,        np.inf]     # Normally use 3.45 (@500kHz) for Brain
        MaterialList['SpecificHeat']    = [4178.0,  4178.0,     3617.0,     3630.0]     # (J/kg/°C)
        MaterialList['Conductivity']    = [0.6,     0.6,        0.52,       0.51]       # (W/m/°C)
        MaterialList['Perfusion']       = [0.0,     0.0,        10000.0,    559.0]      # (ml/min/kg)
        MaterialList['Absorption']      = [0.0,     0.0,        0.0,        1.0]        # Normally use 0.85 for Brain
        MaterialList['InitTemperature'] = [37.0,    37.0,       37.0,       37.0]       # (°C)
        # Water material is duplicated since metal compute sims run into issues when
        # trying to run in homogenous material (like water) and its properties are stored in index 0
        
        # We chose new values for attenuation and absorption values for brain to essentially remove them from
        # babelvisco's BHTE heat disposition calculation since they aren't included in the truth method's calculation
    
        return MaterialList

    def _set_medium(medium_type='brain'):
        
        # Indices for materials
        blood_index = 2
        if medium_type == "water":
            medium_index = 1
        elif medium_type == "blood":
            medium_index = 2
        elif medium_type == "brain":
            medium_index = 3
        else:
            raise ValueError("Invalid medium_type provided")
        
        # Define medium properties
        medium = {}
        MaterialList = _get_material_list()
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
        
    def _create_grid(grid_dims = [],grid_steps = []):
        # Domain Dimensions
        Lx, Ly, Lz = grid_dims
        
        # Create computational grid
        dx, dy, dz = grid_steps
        Nx = int(Lx//dx);       # number of grid points in the x (row) direction
        Ny = int(Ly//dy);       # number of grid points in the y (column) direction
        Nz = int(Lz//dz);       # number of grid points in the z direction
        
        # Create the 1D coordinates for each axis, centered around zero
        x = np.linspace(-Lx/2, Lx/2 - dx, Nx, np.float64)
        y = np.linspace(-Ly/2, Ly/2 - dy, Ny, np.float64)
        z = np.linspace(-Lz/2, Lz/2 - dz, Nz, np.float64)
        
        # Create a meshgrid
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # 'ij' ensures (x,y,z) ordering
        
        return X,Y,Z
        
    return {'grid': _create_grid,
            'medium': _set_medium,
            'material_list': _get_material_list}