# Python imports
import math

# Third party libraries imports
import torch.nn as nn
import numpy as np
from waveblocks.blocks.microlens_arrays import MLAType


class PSFConfig:
    pass

class scattering:
    pass

class MLAConfig:
    pass

class CameraConfig:
    pass

class PolConfig:
    pass

class VolumeConfig:
    pass

class OpticConfig(nn.Module):
    """Class containing the global parameters of an optical system:
    Keyword args:
    wavelenght: wavelenght of optical system (in um)
    samplingRate: sampling used when propagating wavefronts through space. Or when using a camera.
    k: wave number
    """

    @staticmethod
    def get_default_PSF_config():
        psf_config = PSFConfig()
        psf_config.M = 40  # Magnification
        psf_config.NA = 0.9  # Numerical Aperture
        psf_config.Ftl = 165000  # Tube lens focal length
        psf_config.ns = 1.33  # Specimen refractive index (RI)
        psf_config.ng0 = 1.515  # Coverslip RI design value
        psf_config.ng = 1.515  # Coverslip RI experimental value
        psf_config.ni0 = 1  # Immersion medium RI design value
        psf_config.ni = 1  # Immersion medium RI experimental value
        psf_config.ti0 = (
            150  # Microns, working distance (immersion medium thickness) design value
        )
        psf_config.tg0 = 170  # Microns, coverslip thickness design value
        psf_config.tg = 170  # Microns, coverslip thickness experimental value
        psf_config.zv = (
            0  # Offset of focal plane to coverslip, negative is closer to objective
        )
        psf_config.wvl = 0.63  # Wavelength of emission

        # Sample space information
        psf_config.voxel_size = [6.5/60, 6.5/60, 0.43] # Axial step size in sample space
        psf_config.depths = list(np.arange(-10*psf_config.voxel_size[-1], 10*psf_config.voxel_size[-1], psf_config.voxel_size[-1])) # Axial depths where we compute the PSF
            
        return psf_config
    
    @staticmethod
    def get_default_MLA_config():
        mla_config = MLAConfig()
        mla_config.use_mla = True
        mla_config.pitch = 100
        mla_config.camera_distance = 2500
        mla_config.focal_length = 2500
        mla_config.arrangement_type = MLAType.periodic
        mla_config.n_voxels_per_ml = 1 # How many voxels per micro-lens
        mla_config.n_micro_lenses = 1

        return mla_config

    def setup_parameters(self):
        # Setup last PSF parameters
        self.PSF_config.fobj = self.PSF_config.Ftl / self.PSF_config.M
        # Calculate MLA number of pixels behind a lenslet
        self.mla_config.n_pixels_per_mla = 2 * [self.mla_config.pitch // self.camera_config.sensor_pitch]
        self.mla_config.n_pixels_per_mla = [int(n + (1 if (n % 2 == 0) else 0)) for n in self.mla_config.n_pixels_per_mla]
        # Calculate voxel size
        voxel_size_xy = self.camera_config.sensor_pitch / self.PSF_config.M
        self.PSF_config.voxel_size = [voxel_size_xy, voxel_size_xy, self.PSF_config.voxel_size[-1]]

        return
    @staticmethod
    def get_default_camera_config():
        camera_config = CameraConfig()
        camera_config.sensor_pitch = 6.5
        return camera_config

    @staticmethod
    def get_polarizers():
        pol_config = PolConfig()
        pol_config.polarizer = np.array([[1, 0], [0, 1]])
        pol_config.analyzer = np.array([[1, 0], [0, 1]])
        return pol_config
    
    def __init__(self, PSF_config=None):
        super(OpticConfig, self).__init__()
        if PSF_config is None:
            self.PSF_config = self.get_default_PSF_config()
        else:
            self.PSF_config = PSF_config
        self.set_k()
        self.mla_config = self.get_default_MLA_config()
        self.camera_config = self.get_default_camera_config()
        self.pol_config = self.get_polarizers()
        self.volume_config = VolumeConfig()


    def get_wavelenght(self):
        return self.PSF_config.wvl

    def get_medium_refractive_index(self):
        return self.PSF_config.ni

    def set_k(self):
        # Wave Number
        self.k = 2 * math.pi * self.PSF_config.ni / self.PSF_config.wvl  # wave number
        
    def get_k(self):
        return self.k
