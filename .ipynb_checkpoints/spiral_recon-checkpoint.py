"""
Spiral recon class.
Initial creation. Zhibo. Started: 03/31/2026. Ended: 04/03/2026.
Adding apply_GIRF(). Zhibo. Started: 04/03/2026. Ended: 04/09/2026.
Adding CSM estimation. Zhibo. Started: 04/17/2026. Ended: 04/20/2026.
Adding iterative recon for off-resonance correction. Zhibo. Started: 04/29/2026. Ended: 05/01/2026.
"""
import numpy as np
import torch
import copy
from nufft_operators import prepare_tensors, NUFFT_forward_torch, NUFFT_adjoint_torch
from spiral_opt import spi_opt
from spiral_obj import spi_obj
from Py_RawReadRecon_v1 import UIHRawRead
from coils import calculate_csm_inati_iter, smooth
# from espirit import espirit, espirit_proj, fft, ifft

class spi_recon:
    def __init__(self):
        self.spi_opt = spi_opt()
        self.spi_obj = spi_obj()
        self.freq = 0 # Amount of reconstruction frequency offsets
        self.CSM = None
        self.fname = ""

    def run(self, ksp, freq):
        # Load .npy after self.read_raw(self, raw_name) has been performed.
        # ksp = np.load(self.fname) # Nc-Nro-Npe-Ns-1
        ksp_finufft = copy.deepcopy(ksp) # Make a deepcopy.

        if freq != 0:
            print(f"MFI recon using frequency offset = {freq} Hz.")
            dwell_time = self.spi_obj.dwell_time
            dwell_time -= dwell_time[0]
            MFI_phase = np.transpose(np.array([np.exp(1j * 2 * np.pi * freq * dwell_time)] * np.size(ksp_finufft, 2))) # (NADC, Npe)
        else:
            MFI_phase = 1.0

        # Apply IFFT along the slice dimension is kspace data is stack-of-spiral.
        mode = self.spi_opt.mode
        match bin(mode):
            case "0b00":
                ksp_finufft = ksp_finufft
                print("k-space remains unmodified (coil-kro-kpe-slice-1).")
            case "0b01":
                raise ValueError("2D cones not yet supported.")
            case "0b10":
                print("IFFT applied along slice dimension. k-space is now in (coil-kro-kpe-slice-1).")
                ksp_finufft = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(ksp_finufft, axes=3), axis=3), axes=3)
            case "0b11":
                raise ValueError("3D cones not yet supported.")

        # Prepare kspace trajectories and density compensation functions.
        crdsx = np.transpose(self.get_obj().kx).flatten() # (NADC*Npe,)
        crdsy = np.transpose(self.get_obj().ky).flatten()
        crdsx = np.squeeze(crdsx) / max(np.abs(crdsx)) * np.pi # From -pi to pi
        crdsy = np.squeeze(crdsy) / max(np.abs(crdsy)) * np.pi
        dcf = np.transpose(self.get_obj().dcf) # (NADC, Npe)
        dcf = np.squeeze(dcf)
        
        img_shape = [self.spi_opt.Nx, self.spi_opt.Nx, ksp_finufft.shape[3]]
        
        import finufft
        import time
        tic = time.perf_counter()
        
        img = np.zeros(img_shape).astype(np.complex64)
        img_c = np.zeros(np.append(img_shape, ksp_finufft.shape[0])).astype(np.complex64)
        for ii in range(img_shape[-1]):
            # print("current process slice: ", ii)
            for jj in range(ksp_finufft.shape[0]):
                # print("current process coil: ", jj)
                cur_ksp = np.squeeze(ksp_finufft[jj, :, :, ii]) * dcf # (NADC, Npe)
                cur_ksp *= MFI_phase
                cur_ksp = cur_ksp.flatten()
                img_t = finufft.nufft2d1(crdsx, crdsy, cur_ksp, (img_shape[0], img_shape[1]))
                img_c[..., ii, jj] = img_t
                # img[:, :, ii] += np.abs(img_t)**2

        print("Performing CSM calculation and coil combinations ...")
        # ESPIRIT, so slow.
        # ksp = fft(img_c, (0, 1))
        # for ii in range(img_shape[-1]):
        #     esp = espirit(np.expand_dims(ksp[..., ii, :], axis=2), 6, 20, 0.01, 0.9925)
        #     ip, proj, null = espirit_proj(np.expand_dims(img_c[..., ii, :], axis=2), esp)
        #     img[..., ii] = ip[..., 0, 0]

        # Inati.
        if self.CSM is None:
            print("Caulating CSM ...")
            csm, img_cc = calculate_csm_inati_iter(np.permute_dims(img_c, (3, 2, 1, 0)), smoothing=5, niter=5, thresh=1e-3, verbose=False)
            # csm_energy = np.sqrt(np.sum(np.abs(csm)**2, axis=0, keepdims=True))
            # csm = csm / (csm_energy + 1e-12)
            self.CSM = np.permute_dims(csm, (3, 2, 1, 0))
            print("CSM calculation done.")
            img = np.permute_dims(img_cc, (2, 1, 0))
        else:
            print("Using existing CSM.")
            img = (img_c * np.conj(self.CSM)).sum(-1)
        print("Coil combinations Done.")
        # img = img**0.5
            
        toc = time.perf_counter()
        print(f"3D stack-of-spiral recon used {toc - tic:0.4f} seconds") 
        return img

    def run_iterative_recon(self, ksp, B0_map, TE, iters=5, L=15, B=8, fmax=1000):
        # Estimate temporal coefficients for time segmented phase modulations.
        dwell_time = self.spi_obj.dwell_time
        dwell_time += TE - dwell_time[0]
        freqs = np.arange(-fmax, fmax+1, 20)
        
        p_basis = np.array(-1j * 2 * np.pi * freqs)
        b_total = np.exp(p_basis[:, np.newaxis] * dwell_time[np.newaxis, :])

        NADC = len(dwell_time)
        tau_L = np.linspace(dwell_time[0], dwell_time[-1], L)
        print(f"Time segments centers tau_L: {tau_L}")
        P = np.exp(p_basis[:, np.newaxis] * tau_L[np.newaxis, :])
        print("Sanity checks:")
        print(f"P shape: {P.shape}")
        print(f"b_total shape: {b_total.shape}")
        print(f"P row {P.shape[0]//2}: {P[P.shape[0]//2, :]}")
        
        a_total = np.linalg.pinv(P) @ b_total
        print(f"a_total shape: {a_total.shape}")
        
        weight_sums = np.sum(a_total, axis=0)
        print(f"Mean sum of weights: {np.mean(np.abs(weight_sums))}")
        print(f"Max weight value: {np.max(np.abs(a_total))}")

        # Prepare PyTorch tensors.
        CSM = self.CSM
        DCF = self.spi_obj.dcf
        kx = self.spi_obj.kx
        ky = self.spi_obj.ky
        csm_torch, ktraj, weights_torch, dcf_torch, b0_torch = prepare_tensors(CSM, np.transpose(kx) / np.max(kx) * np.pi, np.transpose(ky) / np.max(ky) * np.pi, a_total, np.transpose(DCF), B0_map)
        csm_energy = torch.sqrt(torch.sum(torch.abs(csm_torch)**2, dim=1, keepdim=True))
        csm_torch = csm_torch / (csm_energy + 1e-12)
        ksp_torch = torch.tensor(np.squeeze(ksp.reshape(ksp.shape[0], -1, ksp.shape[3], 1))).permute(2, 0, 1)

        # Perform CG SENSE.
        Nx = self.spi_opt.Nx
        Ny = Nx
        Ns = ksp.shape[3]
        x = torch.zeros(Nx, Ny, Ns, dtype=torch.complex64)
        
        ktraj = ktraj.to('cuda')
        r = NUFFT_adjoint_torch(ksp_torch, csm_torch, ktraj, weights_torch, torch.square(dcf_torch), b0_torch, tau_L, batch_size=8)
        p = copy.deepcopy(r)
        
        for i in range(iters):
            print(f"Iter {i}, loss: {torch.norm(r)}")
            # Apply A^H A
            q = NUFFT_adjoint_torch(NUFFT_forward_torch(p, csm_torch, ktraj, weights_torch, torch.ones_like(dcf_torch), b0_torch, tau_L, batch_size=8), csm_torch, ktraj, weights_torch, torch.square(dcf_torch), b0_torch, tau_L, batch_size=8)
            
            # Standard CG Update
            alpha = torch.sum(r*r) / torch.sum(p*q)
            x = x + alpha * p
            r_new = r - alpha * q
            beta = torch.sum(r_new*r_new) / torch.sum(r*r)
            p = r_new + beta * p
            r = r_new

        return x

            
    def read_raw(self, raw_name):
        mode = self.spi_opt.mode
        print(bin(mode))
        match bin(mode):
            case "0b00":
                fname = UIHRawRead.Run(raw_name, ScanMode='2D')
                self.fname = "".join(char for char in fname[0])
            case "0b01":
                raise ValueError("2D cones not yet supported.")
            case "0b10":
                fname = UIHRawRead.Run(raw_name, ScanMode='3D')
                self.fname = "".join(char for char in fname[0])
            case "0b11":
                self.fname = "".join(char for char in UIHRawRead.Run(raw_name, mode='3DCones'))

    def set_opt(self, my_opt):
        self.spi_opt = my_opt

    def set_obj(self, my_obj):
        if my_obj:
            print("Setting a spiral object manually.")
            self.spi_obj = my_obj
        else:
            print("Generating a spiral object automatically.")
            self.spi_obj.set_opt(self.spi_opt)
            self.spi_obj.run()

    def get_opt(self):
        return self.spi_opt

    def get_obj(self):
        return self.spi_obj