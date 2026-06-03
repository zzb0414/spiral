"""
Spiral option class.
Initial creation. Zhibo. Started: 03/26/2026. Ended: 03/27/2026.
Adding the spiral mode option. Started: 03/30/2026. Ended: 03/30/2026.
"""
import numpy as np

class spi_opt:
    def __init__(self):
        self.Gmax = 10 # [G/cm]
        self.Smax = 15000 # [G/cm/s]
        self.grad_raster_time = 10 # [us]
        self.calc_time_factor = 1 # Further shortening of durations
        self.FOV = 0.2 # [m]
        self.Nx = 128 # Number of Nx voxels
        self.Ns = 24 # Number of spiral arms
        self.N_acc_in = 1 # Inner kspace acceleration factor
        self.N_acc_out = 1 # Outer kspace accerleration factor
        self.dwell = 10 # [us]
        self.BW = float(1 / self.dwell / self.Nx)
        self.NADC = 512
        self.mode = 0b00 # Spiral mode (2-bit binary): bit 1, spiral (0) or cones (1). bit 2, 2D (0) or 3D (1).
        
    def set_value(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def disp_param(self, keys):
        param = vars(self)
        if keys == "all":
            for key, value in param.items():
                print(f"{key}: {value}")
            return
        else:
            for key in keys:
                value = param[key]
                print(f"{key}: {value}")
            return
            
        