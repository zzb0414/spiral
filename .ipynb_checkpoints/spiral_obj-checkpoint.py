"""
Spiral option class.
Initial creation. Zhibo. Started: 03/26/2026. Ended: 03/27/2026.
Adding set_opt(self, my_opt), run(self) and calc_dcf(self, approach="analytical"). Zhibo. Started: 03/30/2026. Ended: 03/30/2026.
Adding set_manual(self, **kwarg). Zhibo, Started: 03/31/2026. Ended: 03/31/2026.
Optimizng cal_dcf(approach="analytical2"). "iterative" and "voronoi" approach unrecomended. Zhibo, started: 04/03/2026. Ended: 04/03/2026.
Adding apply_GIRF(self, girf_name). Zhibo. Started: 04/03/2026. Ended: 04/6/2026.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from spiral_opt import spi_opt
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from scipy.spatial import Voronoi
from shapely.geometry import Polygon


class spi_obj:
    def __init__(self):
        # Define glabal variables.
        global pi, Gamma
        pi = np.pi
        Gamma = float(42.5756 * 1e2) # [Hz/G]

        # When initilizing, make a set of spiral gradients/trajecotries using the default spiral options.
        default_opt = spi_opt()
        self.set_opt(default_opt)
        self.manual_delay = 0 # Unit in number of ADC samples
        self.run()

    # set_opt(self, my_opt): Set my_opt as the spiral option object.
    #                my_opt: The target spiral option object.
    def set_opt(self, my_opt):
        # Load spiral options to global parameters
        global Gmax, Smax, grad_raster_time, FOV, Nx, Ns, N_acc_in, N_acc_out, dwell, BW, NADC, mode
        Gmax = my_opt.Gmax # [G/cm]
        Smax = my_opt.Smax * 100 # [G/cm/s]
        grad_raster_time = my_opt.grad_raster_time / 1e6 # [sec]
        FOV = 1e3 * my_opt.FOV # [mm]
        Nx = my_opt.Nx # Number of Nx voxels
        Ns = my_opt.Ns # Number of spiral arms
        N_acc_in = my_opt.N_acc_in # Inner kspace acceleration factor
        N_acc_out = my_opt.N_acc_out # Outer kspace accerleration factor
        dwell = my_opt.dwell / 1e6 # [sec]
        BW = my_opt.BW
        NADC = my_opt.NADC
        mode = my_opt.mode

    # set_manual(self, **kwarg): Set attributes evaluated with the ADC dwell time manually based on **kwarg.
    def set_manual(self, **kwargs):
        # Make sure to set attributes in pairs and do not missing the dwell_time
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    # run(self): Run gradient waveform generation.
    def run(self):
        match mode:
            case 0b00:
                self.vds()
                self.calc_dcf("analytical")
                print("2D spiral generated")
            case 0b01:
                print("2D cones place holder")
            case 0b10:
                self.vds()
                self.calc_dcf("analytical2")
                print("2D spiral generated. Please use it in each kz location in 3D SoS.")
            case 0b11:
                print("3D cones place holder")
                
    # vds(self): Generate a set of 2D spiral gradient waveforms.
    def vds(self):
        # Calculate gradient sampling time based on gradient raster time.
        global grad_sample_time
        grad_sample_time = grad_raster_time / 8
        
        # Convert accerleration factor to spiral designe parameters.
        global f_coeff_in, f_coeff_out
        f_coeff_in = float(FOV / 10 / N_acc_in)
        f_coeff_out = float(FOV / 10 / N_acc_out) - f_coeff_in

        # Debug
        # print(f"f_coeff_in: {f_coeff_in}")
        # print(f"f_coeff_out: {f_coeff_out}")

        # Generate gradient waveforms evaluated at gradient raster time interval.
        global Rmax
        Rmax = float(Nx / FOV * 5) # [1/cm], kspace max radius
        Gx0, Gy0 = self.make_vds()
        Gx0 = np.insert(Gx0, 0, 0)
        Gy0 = np.insert(Gy0, 0, 0)
        grad_time = grad_raster_time * np.arange(0, len(Gx0))
        # NADC = np.floor(grad_raster_time * len(Gx0) / dwell)
        global dwell_time
        dwell_time = dwell * (np.arange(-10, NADC)) + (0.5 + self.manual_delay) * dwell # Manually adjust grad delay here.
        
        # Interpolate gradient waveforms onto ADC dwell time.
        Gro0 = np.interp(dwell_time, grad_time, Gx0, left=0, right=0)
        Gpe0 = np.interp(dwell_time, grad_time, Gy0, left=0, right=0)
        Gro0 = Gro0
        Gpe0 = Gpe0
        # dwell_time = dwell_time

        # Apply 2D in-plane rotations.
        spi_rot_angle = 2 * pi * np.arange(Ns) / Ns
        Gx = np.zeros((Ns, len(Gx0)))
        Gy = np.zeros((Ns, len(Gy0)))
        Gro = np.zeros((Ns, len(Gro0)))
        Gpe = np.zeros((Ns, len(Gpe0)))
        kx = np.zeros((Ns, len(Gro0)))
        ky = np.zeros((Ns, len(Gpe0)))
        for ii in range(Ns):
            rot_mtx = np.array([[np.cos(spi_rot_angle[ii]), -np.sin(spi_rot_angle[ii])], [np.sin(spi_rot_angle[ii]), np.cos(spi_rot_angle[ii])]])
            Gx[ii, ...], Gy[ii, ...] = self.rot_2D(Gx0, Gy0, rot_mtx)
            Gro[ii, ...], Gpe[ii, ...] = self.rot_2D(Gro0, Gpe0, rot_mtx)
            kx[ii, ...] = self.GK_convert("G2K", Gro[ii, ...], dwell_time)
            ky[ii, ...] = self.GK_convert("G2K", Gpe[ii, ...], dwell_time)

        # Store the theoretical waveforms, logical kspace trajectories and time interval values.
        self.Gx = -Gx
        self.Gy = Gy
        self.Gro = -Gro
        self.Gpe = Gpe
        self.kx = -kx
        self.ky = ky
        self.B0ro = np.zeros(Gro.shape)
        self.B0pe = np.zeros(Gpe.shape)
        self.grad_time = grad_time
        self.dwell_time = dwell_time

    # make_vds(): Generate the baseline gradient waveforms evaluated at gradient raster time interval for 2D spiral or 3D stack-of-spiral.
    def make_vds(self):
        Q0 = 0.0
        Q1 = 0.0
        R0 = 0.0
        R1 = 0.0
        T = 0.0
        cnt = 0

        theta = np.array(0)
        radius = np.array(0)
        
        # Output waveforms
        Gx = np.array([])
        Gy = np.array([])

        # grad_sample_time
        while R0 < Rmax:
            Q2_temp, R2_temp = self.find_Q2_R2(R0, R1, cnt);
            Q1 += Q2_temp * grad_sample_time
            Q0 += Q1 * grad_sample_time
            T += grad_sample_time
            R1 += R2_temp * grad_sample_time
            R0 += R1 * grad_sample_time
            cnt += 1
            theta = np.append(theta, Q0)
            radius = np.append(radius, R0)

            # Debug
            # if N_acc_out != 1:
            #     print(f"count, Q2, R2, Q1, Q0, T, R1, R0: ({cnt}, {Q2_temp:.8f}, {R2_temp:.8f}, {Q1:.13f}, {Q0:.19f}, {T:.8f}, {R1:.14f}, {R0:.20f})")

        # print(cnt)
        
        kx = np.multiply(radius[3::8], np.cos(theta[3::8]))
        ky = np.multiply(radius[3::8], np.sin(theta[3::8]))
        kx_fwd = kx
        ky_fwd = ky

        kx = np.append(kx, 0)
        ky = np.append(ky, 0)
        kx_fwd = np.insert(kx_fwd, 0, 0)
        ky_fwd = np.insert(ky_fwd, 0, 0)

        for ii in range(math.floor((cnt + 5) / 8)):
            kx_step = kx[ii] - kx_fwd[ii]
            ky_step = ky[ii] - ky_fwd[ii]
            Gx = np.append(Gx, float(kx_step * 10 / Gamma / grad_raster_time)) # [mT/m]
            Gy = np.append(Gy, float(ky_step * 10 / Gamma / grad_raster_time))

        # print(f"Gx end: {Gx[-1]}")
        # print(f"Gy end: {Gy[-1]}")
        return Gx, Gy

    # find_Q2_R2(self, R0, R1): A helper function to calculate intermediate parameters for spiral trajectories generation.    
    def find_Q2_R2(self, R0, R1, cnt):
        F = 0.0
        dFdr = 0.0
        F = F + f_coeff_in + f_coeff_out * (R0 / Rmax)
        dFdr = dFdr + f_coeff_out / Rmax;
        Gmax_FOV = (1 / Gamma / F / grad_raster_time)
        if Gmax_FOV > Gmax:
            G = Gmax
        else:
            G = Gmax_FOV

        temp_const1 = float(2 * pi * F / Ns)
        temp_const2 = float(2 * pi * dFdr / Ns)
        max_r1 = float(((Gamma * G) ** 2 / (1 + (R0 * temp_const1) ** 2)) ** 0.5)
        
        # if cnt == 1911:
        #     print([Gmax_FOV, Gmax])
        #     print(f"Gmax, temp_const1, temp_const2, max_r1, R0, R1: {G}, {F}, {dFdr}, {temp_const1}, {temp_const2}, {max_r1}, {R0}, {R1}")
        
        if R1 > max_r1:
            R2 = float((max_r1 - R1) / grad_sample_time)
        else:
            A = float(1 + temp_const1 ** 2 * R0 ** 2)
            B = float(2 * temp_const1 ** 2 * R0 * R1 ** 2 + 2 * temp_const1 **2 / F * dFdr * (R0 * R1) ** 2)
            C = float(temp_const1 ** 4 * R0 ** 2 * R1 **4 \
                     + 4 * temp_const1 ** 2 * R1 ** 4 \
                     + temp_const2 ** 2 * R0 ** 2 * R1 **4 \
                     + 4 * temp_const1 ** 2 / F * dFdr * R0 * R1 ** 4 \
                     - Gamma ** 2 * Smax ** 2)
            R2 = self.qdf(A, B, C)
        
        Q2 = temp_const2 * R1 ** 2 + temp_const1 * R2
        # Debug
        # if N_acc_out != 1:
            # print(f"Q2, R2: {Q2, R2}")

        return Q2, R2

    # qdf(self, A, B, C): A helper function to calculate intermediate parameters fro spiral trajectories generation.
    def qdf(self, A, B, C):
        temp = float(B ** 2 - 4 * A * C)
        if temp < 0:
            return float(-B / (2 * A))
        else:
            return float((-B + temp ** 0.5) / (2 * A))

    # rot_2d(self, x, y, rot_mtx): A helper function to perform one-time in-plane 2D rotation using rot_mtx.
    #                           x: x coordinates, Np x 1 numpy array
    #                           y: y coordinates, Np x 1 numpy array
    #                     rot_mtx: 2 x 2 numpy array
    def rot_2D(self, x, y, rot_mtx):
        vec = np.column_stack((x, y))
        vec = vec @ rot_mtx
        return vec[..., 0], vec[..., 1]

    # GK_convert(self, flag, inp, dwell_time): A helper function to perform gradient waveforms to kspace trajectories conversion (or vice versa).
    #                                    flag: String values indicating the operation type, e.g., "G2K" and "K2G"
    #                                     inp: Input variable, Ns x Np numpy array
    #                              dwell_time: ADC dwell time
    def GK_convert(self, flag, inp, dwell_time):
        if flag == "G2K":
            grad_time_new = np.arange(grad_raster_time * np.floor(dwell_time[0] / grad_raster_time), grad_raster_time * (np.ceil(dwell_time[-1] / grad_raster_time + 1)), grad_raster_time)
            grad_new = np.interp(grad_time_new, dwell_time, inp, left=0, right=0)
            # out = np.cumsum(grad_new, axis = 0)
            out = np.zeros(grad_new.shape)
            for ii in range(len(grad_new)):
                out[ii] = grad_new[ii] if ii == 0 else out[ii-1] + (grad_new[ii] + grad_new[ii-1]) / 2
            out = 10 * Gamma * out * grad_raster_time
            out = np.interp(dwell_time, grad_time_new, out, left=0, right=0)
        elif flag == "K2G":
            inp = np.insert(inp, 0, 0, axis = 1)
            out = np.diff(inp, axis = 1)
            out = out / (10 * Gamma * dwell)
        else:
            raise ValueError("Approach must be 'G2K' or 'K2G'")

        return out

    # calc_dcf(self): Calculate the density compensation function.
    def calc_dcf(self, approach = "analytical"):
        if approach.lower() == "analytical":
            print("Calculating analytical DCF ...")
            # 1. Convert Gradients from mT/m to G/cm to match Gamma (Hz/G)
            # 1 mT/m = 0.01 G/cm
            gro_g_cm = self.Gro * 0.01
            gpe_g_cm = self.Gpe * 0.01
            
            # 2. Calculate k-space velocity (magnitude)
            # |dk/dt| = gamma * |G| 
            # Result units: Hz/cm (which is cm^-1 * s^-1)
            g_mag = np.sqrt(gro_g_cm ** 2 + gpe_g_cm ** 2)
            velocity = Gamma * g_mag
            
            # 3. Calculate k-space radius
            # Convert kx, ky from 1/m to 1/cm to stay consistent
            kx_cm = self.kx * 0.01
            ky_cm = self.ky * 0.01
            k_radius = np.sqrt(kx_cm ** 2 + ky_cm ** 2)

            # 4. Calculate the Sine of the angle between k and G
            # Using the cross product formula: |k x G| = |k||G|sin(theta)
            # sin_theta = |kx*Gy - ky*Gx| / (|k|*|G|)
            # We use a small epsilon to avoid division by zero at the k-space center
            epsilon = 1e-10
            cross_product_mag = np.abs(kx_cm * gpe_g_cm - ky_cm * gro_g_cm)
            sin_theta = cross_product_mag / (k_radius * g_mag + epsilon)

            # 5. DCF = Radius * Velocity
            self.dcf = k_radius * velocity * sin_theta
            
            # Normalize to 1.0
            self.dcf /= np.max(self.dcf)
            print("Calculating analytical DCF done.")
            
        elif approach.lower() == "analytical2":
            print("Calculating analytical DCF, approach No.2 ...")
            # Another analytical approach (which works better).
            k_radius = np.sqrt(self.kx ** 2 + self.ky ** 2)
            k_mag1 = np.concatenate((np.expand_dims(-k_radius[:, 0], axis=1), k_radius[:, :-1]), axis=1)
            k_mag2 = np.concatenate((k_radius[:, 1:], 2 * np.expand_dims(k_radius[:, -1], axis=1) - np.expand_dims(k_radius[:, -2], axis=1)), axis=1)
            self.dcf = (k_radius + (k_mag1 + k_mag2) / 2) * ((k_mag2-k_mag1) / 2) 
            self.dcf /= np.max(self.dcf)
            print("Calculating analytical DCF, approach No.2, done.")

        elif approach.lower() == "voronoi":
            # An approach based on the Voronoi diagram.
            print("Calculating DCF using Voronoi diagram ...")
            # 1. Prepare original k-space points
            kx_f = self.kx.flatten()
            ky_f = self.ky.flatten()
            points = np.column_stack((kx_f, ky_f))
            num_original = len(points)
            
            # 2. Create a Circular "Fence" (Spatial Support)
            # Find the maximum radius of your spiral
            k_radius = np.sqrt(kx_f**2 + ky_f**2)
            k_max = np.max(k_radius)
            
            # Place support points at 1.1x the max radius to "clip" the edges
            # Using ~200 points usually provides a smooth circular boundary
            padding_radius = k_max * 1.005 
            theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
            fence_points = np.column_stack([
                padding_radius * np.cos(theta),
                padding_radius * np.sin(theta)
            ])
            
            # 3. Augment points and Compute Voronoi
            augmented_points = np.vstack([points, fence_points])
            vor = Voronoi(augmented_points)
            
            # 4. Extract Areas for Original Points Only
            areas_f = np.zeros(num_original)
            
            # vor.point_region maps the index of the input point to the vor.regions index
            for i in range(num_original):
                region_index = vor.point_region[i]
                region = vor.regions[region_index]
                
                # Because of the fence, even the outermost points should now be finite
                if -1 not in region and len(region) > 0:
                    poly_vertices = vor.vertices[region]
                    areas_f[i] = Polygon(poly_vertices).area
                else:
                    # Fallback: if a point is still infinite, we use the max finite area
                    areas_f[i] = 0.0
            
            # 5. Handle any remaining zeros (safety check)
            mask = (areas_f == 0)
            if np.any(mask):
                # Use max finite area to ensure the edge isn't zeroed out
                areas_f[mask] = np.max(areas_f)
            
            # 6. Normalize and Reshape
            max_val = np.max(areas_f)
            if max_val > 0:
                areas_f /= max_val
            
            self.dcf = areas_f.reshape(self.kx.shape)
            print("Calculating DCF using Voronoi diagram done.")
            
        elif approach.lower() == "iterative":
            # Does not look good. Prefer not to use it.
            # 1. Setup Parameters
            kx_f = self.kx.flatten()
            ky_f = self.ky.flatten()
            num_points = len(kx_f)
            
            # High grid resolution is key to reducing jitter
            grid_res = 512
            k_max = np.max(np.sqrt(kx_f**2 + ky_f**2))
            grid_limit = k_max * 1.1
            
            # Initial weights
            w = np.ones(num_points)
            
            # 2. Pipe & Menon Iterative Loop
            for i in range(12):
                # Step A: Gridding (Weight-aware Histogram)
                density_grid, _, _ = np.histogram2d(
                    kx_f, ky_f, bins=grid_res, 
                    range=[[-grid_limit, grid_limit], [-grid_limit, grid_limit]],
                    weights=w
                )
                
                # Step B: Smoothing (The Convolution Kernel)
                # A larger sigma (3.0-5.0) ensures a smooth density transition
                density_grid = gaussian_filter(density_grid, sigma=5.0)
                
                # plt.figure(i)
                # plt.imshow(density_grid)
                # plt.show()
                
                # Step C: Bi-linear Interpolation back to k-space points
                # Map coordinates to grid indices [0, grid_res-1]
                x_idx = (kx_f + grid_limit) / (2 * grid_limit) * (grid_res - 1)
                y_idx = (ky_f + grid_limit) / (2 * grid_limit) * (grid_res - 1)
                
                x0 = np.clip(x_idx.astype(int), 0, grid_res - 2)
                y0 = np.clip(y_idx.astype(int), 0, grid_res - 2)
                dx = x_idx - x0
                dy = y_idx - y0
                
                # Interpolate from the 4 nearest grid cells
                interp_density = (
                    density_grid[x0, y0] * (1 - dx) * (1 - dy) +
                    density_grid[x0 + 1, y0] * dx * (1 - dy) +
                    density_grid[x0, y0 + 1] * (1 - dx) * dy +
                    density_grid[x0 + 1, y0 + 1] * dx * dy
                )
                
                # Step D: Update Weights
                w /= (interp_density + 1e-11)
                w /= np.mean(w) # Stabilize energy

            # 3. Post-Processing for Smoothness
            self.dcf = w.reshape(self.kx.shape)
            
            # Apply Savitzky-Golay filter along the readout (axis=1) 
            # to remove residual discretization noise.
            # window_length should be odd and smaller than Np.
            window_len = min(31, self.kx.shape[1] // 4)
            if window_len % 2 == 0: window_len += 1
            
            # self.dcf = savgol_filter(self.dcf, window_length=window_len, polyorder=2, axis=1)
            
            # Final normalization and center-correction
            self.dcf[self.dcf < 0] = 0 # Remove filter underflow
            self.dcf /= np.max(self.dcf)
        else:
            raise ValueError("Approach must be 'analytical' or 'iterative'")
            
        return self.dcf

    # apply_GIRF(self, freq_axis_name, Hfun_name): Apply GIRF correction on current gradient waveforms.
    def apply_GIRF(self, freq_axis_name, Hfun_name):
        # Helper functions.
        def get_itp_grad_response(freq_axis, Hfun, freq_axis_double):
            sup = np.int_(np.arange(math.floor(len(freq_axis) - 0.5) / 2))
            sup_double = np.int_(np.arange(math.floor(len(freq_axis_double) - 0.5) / 2))

            # Debug
            # print(len(sup))
            # print(len(sup_double))
            
            temp = np.interp(freq_axis_double[sup_double], freq_axis[sup], Hfun[sup], left=0, right=0)

            if len(freq_axis_double) % 2 == 1:
                return np.concatenate((temp, np.conj(temp[:0:-1])))
            else:
                return np.concatenate((temp, np.array([0]), np.conj(temp[:0:-1])))

        # TBD.
        def RP2xyz(orient, Gro, Gpe):
            Gxyz = np.array([Gro, Gpe, np.zeros(Gro.shape)])
            return Gxyz

        def xyz2RP(orient, Gxyz):
            Grps = Gxyz
            return Grps

        # Gradient waveforms corrections.
        # Make a deepcopy of Gro, Gpe, kx, ky as new attributes.
        import copy as cp
        self.Gx_orig = cp.deepcopy(self.Gx)
        self.Gy_orig = cp.deepcopy(self.Gy)
        self.Gro_orig = cp.deepcopy(self.Gro)
        self.Gpe_orig = cp.deepcopy(self.Gpe)
        self.kx_orig = cp.deepcopy(self.kx)
        self.ky_orig = cp.deepcopy(self.ky)
        
        # Load pre-saved data.
        freq_axis = np.load(freq_axis_name)
        Hfun = np.load(Hfun_name)
        sup = freq_axis > 2e4
        Hfun[:, sup] = 0

        # Debug
        # plt.figure()
        # plt.plot(Hfun[0].real, Hfun[1].imag)
        # plt.show()
        
        # Extend gradient waveforms and time axis.
        Gx = self.Gx
        Gy = self.Gy
        grad_time = self.grad_time + 10 * grad_raster_time
        grad_time_ext = np.pad(grad_time, (10, 10), mode='linear_ramp', end_values=(0, grad_time[-1] + 10 * grad_raster_time))

        # This is what has been implemented in MATLAB.
        Gx_ext = np.pad(Gx, ((0, 0), (10, 10)), mode='edge')
        Gy_ext = np.pad(Gy, ((0, 0), (10, 10)), mode='edge')
        Gx_ext_double = np.pad(Gx_ext, ((0, 0), (0, len(Gx_ext[0]))), mode='reflect')
        Gy_ext_double = np.pad(Gy_ext, ((0, 0), (0, len(Gy_ext[0]))), mode='reflect')

        # This is what has been implemented in C++.
        flat_dur = math.floor(0.2 * (10 + len(Gx[0])) + 0.5)
        ramp_down_dur = math.floor(0.5 * (10 + len(Gx[0]) + 0.5))
        Gx_ext = np.pad(Gx, ((0, 0), (10, flat_dur)), mode='edge')
        Gy_ext = np.pad(Gy, ((0, 0), (10, flat_dur)), mode='edge')
        Gx_ext_double = np.pad(Gx_ext, ((0, 0), (0, ramp_down_dur)), mode='linear_ramp', end_values=(0, 0))
        Gy_ext_double = np.pad(Gy_ext, ((0, 0), (0, ramp_down_dur)), mode='linear_ramp', end_values=(0, 0))
        end_dur = 2 * (10 + len(Gx[0])) - len(Gx_ext_double[0])
        Gx_ext_double = np.pad(Gx_ext_double, ((0, 0), (0, end_dur)), mode='edge')
        Gy_ext_double = np.pad(Gy_ext_double, ((0, 0), (0, end_dur)), mode='edge')
        
        Ngrad = np.size(Gx, axis=1) + 20

        # Initilize temporary gradient waveforms.
        Gx_double = np.zeros(Gx_ext_double.shape)
        Gy_double = np.zeros(Gx_ext_double.shape)
        Gall_double = np.zeros((3, np.size(Gx_double, axis=1)))

        # Debug
        # print(f"Gx shape: {Gx.shape}")
        # print(f"Gx ext shape: {Gx_ext.shape}")
        # print(f"Gx ext double shape: {Gx_ext_double.shape}")
        # print(f"Ngrad: {Ngrad}")

        freq_axis_double = np.arange(2 * Ngrad, dtype="float32") / (2 * Ngrad) * 1e5 # For MATLAB implementation.
        freq_axis_double = np.arange(np.size(Gx_ext_double, axis=1), dtype="float32") / (np.size(Gx_ext_double, axis=1)) * 1e5 # For C++ implementation.

        # Debug
        # print(f"Freq length: {len(freq_axis)}")
        # print(f"Freq double length: {len(freq_axis_double)}")
        # print(f"Hfun shape: {Hfun.shape}")

        # Get augmented Hfun's.
        Hx_double = get_itp_grad_response(freq_axis, Hfun[3], freq_axis_double)
        Hy_double = get_itp_grad_response(freq_axis, Hfun[4], freq_axis_double)
        Hz_double = get_itp_grad_response(freq_axis, Hfun[5], freq_axis_double)

        # Debug

        # Store corrected gradient waveforms.
        for ii in range(len(Gx)):
            Gall_double = RP2xyz(np.array([0, 0, -1, 0]), Gx_ext_double[ii], Gy_ext_double[ii])
            Gall_double[0] = np.fft.ifft(np.fft.fft(Gall_double[0]) * Hx_double)
            Gall_double[1] = np.fft.ifft(np.fft.fft(Gall_double[1]) * Hy_double)
            Gall_double[2] = np.fft.ifft(np.fft.fft(Gall_double[2]) * Hz_double)
            Gall_double = Gall_double.real

            G = xyz2RP(np.array([0, 0, -1, 0]), Gall_double) 
            Gx_double[ii] = G[0]
            Gy_double[ii] = G[1]

            # Debug
            # if ii == 0:
            #     print(Hx_double)
            #     print(1e5 * Gx_ext_double[ii, 10:15])
            #     print(Gall_double[ii])
            #     print(G[ii])
                
            self.Gx[ii] = Gx_double[ii, 10:Ngrad-10]
            self.Gy[ii] = Gy_double[ii, 10:Ngrad-10]
            
            self.Gro[ii] = np.interp(dwell_time + 10 * grad_raster_time, grad_time_ext, Gx_double[ii, :Ngrad])
            self.Gpe[ii] = np.interp(dwell_time + 10 * grad_raster_time, grad_time_ext, Gy_double[ii, :Ngrad])

            self.kx[ii] = self.GK_convert("G2K", self.Gro[ii], dwell_time)
            self.ky[ii] = self.GK_convert("G2K", self.Gpe[ii], dwell_time)

        # Debug
        # print(f"Gx_double shape: {Gx_double.shape}")
        # print(f"Gx_double row shape: {Gx_double[0, :Ngrad].shape}")
        # print(f"grad_time_ext shape: {grad_time_ext.shape}")
        # print(f"dwell_time: {dwell_time.shape}")
        # print(f"Gro shape: {self.Gro.shape}")
        
        # Off-resonance corrections.
        # Initialize temporary B0 terms.
        B0x_double = np.zeros(Gx_ext_double.shape)
        B0y_double = np.zeros(Gx_ext_double.shape)
        B0all_double = np.zeros((3, np.size(B0x_double, axis=1)))

        # Get augmented Hfun's.
        Hx_B0_double = get_itp_grad_response(freq_axis, Hfun[0], freq_axis_double)
        Hy_B0_double = get_itp_grad_response(freq_axis, Hfun[1], freq_axis_double)
        Hz_B0_double = get_itp_grad_response(freq_axis, Hfun[2], freq_axis_double)

        # Store phase correcting terms.
        for ii in range(len(Gx)):
            Gall_double = RP2xyz(np.array([0, 0, 1, 0]), Gx_ext_double[ii], Gy_ext_double[ii])
            B0all_double[0] = np.fft.ifft(np.fft.fft(Gall_double[0]) * Hx_B0_double)
            B0all_double[1] = np.fft.ifft(np.fft.fft(Gall_double[1]) * Hy_B0_double)
            B0all_double[2] = np.fft.ifft(np.fft.fft(Gall_double[2]) * Hz_B0_double)
            B0all_double = B0all_double.real

            B0 = xyz2RP(np.array([0, 0, 1, 0]), B0all_double) 
            B0x_double[ii] = B0[0]
            B0y_double[ii] = B0[1]

            self.B0ro[ii] = np.interp(dwell_time + 10 * grad_raster_time, grad_time_ext, B0x_double[ii, :Ngrad])
            self.B0pe[ii] = np.interp(dwell_time + 10 * grad_raster_time, grad_time_ext, B0y_double[ii, :Ngrad])
        return