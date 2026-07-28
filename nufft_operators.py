"""
NUFFFT forward and adjoint operators with optional B0 corrections.
"""
import finufft
import numpy as np
import torch
import torchkbnufft as tkbn

# NUFFT_forward(): CPU based NUFFT forward using numpy and finufft.
def NUFFT_forward(img, CSM, crdsx, crdsy, B0_phasors, temporal_coeff):
    """
    Forward Operator (A): Image Space -> K-space
    img: (Nx, Ny, Ns), complex 3D images, numpy array.
    CSM: (Nx, Ny, Ns, Nc), complex coil sensitivity maps, numpy array.
    crdsx, crdsy: (NADC*Npe, ), kspace trajectories coordinates, range from -pi to pi, numpy array.
    B0_phasors: (Nx, Ny, Nz, L), time segmented complex phase terms in image domain, numpy array.
    temporal_coeff: (L, NADC), temporal coefficients for time segmented kspace data, numpy array.
    """
    L, NADC = temporal_coeff.shape
    Nx, Ny, Ns, Nc = CSM.shape
    Npe = int(len(crdsx) / NADC)
    
    # Pre-allocate output: [Nc, NADC, Npe, Ns]
    ksp_finufft = np.zeros((Nc, NADC, Npe, Ns), dtype=np.complex64)
    
    for ns in range(Ns):
        # Apply SENSE for this slice: [Nx, Ny, Nc]
        img_sense = img[:, :, ns, np.newaxis] * CSM[:, :, ns, :]
        
        for l in range(L):
            # Apply B0 phase for this segment and slice
            # B0_phasors shape should be [Nx, Ny, Ns, L]
            img_mod = img_sense * B0_phasors[:, :, ns, l, np.newaxis]
            
            # temporal_coeff[l] is shape [NADC,] -> needs to match k_l shape [NADC*Npe]
            # We reshape weights to match the flattened k-space trajectory
            weights = np.tile(temporal_coeff[l], Npe) 
            
            for nc in range(Nc):
                # 2D NUFFT for each coil
                k_l = finufft.nufft2d2(crdsx, crdsy, img_mod[:, :, nc])
                
                # Apply weights and reshape to [NADC, Npe]
                ksp_finufft[nc, :, :, ns] += (k_l * weights).reshape(NADC, Npe)

    # 1D FFT along the slice dimension (z)
    ksp_finufft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(ksp_finufft, axes=3), axis=3), axes=3)
    
    return ksp_finufft


# NUFFT_adjoint(): CPU based NUFFT adjoint using numpy and finufft.
def NUFFT_adjoint(ksp, CSM, crdsx, crdsy, B0_phasors, temporal_coeff, DCF):
    """
    Adjoint Operator (A^H): K-space -> Image Space
    ksp: (Nc, NADC, Npe, Ns), complex kspace data, numpy array.
    CSM: (Nx, Ny, Ns, Nc), complex coil sensitivity maps, numpy array.
    crdsx, crdsy: (NADC*Npe, ), kspace trajectories coordinates, range from -pi to pi, numpy array.
    B0_phasors: (Nx, Ny, Nz, L), time segmented complex phase terms in image domain, numpy array.
    temporal_coeff: (L, NADC), temporal coefficients for time segmented kspace data, numpy array.
    DCF: (NADC, Npe), density compensation function, numpy array.
    """
    L, NADC = temporal_coeff.shape
    Nx, Ny, Ns, Nc = CSM.shape
    Npe = int(len(crdsx) / NADC)

    # 1. Undo the FFT along slice dimension (z)
    # Applying ifftshift -> ifft -> fftshift
    ksp_adj = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(ksp, axes=3), axis=3), axes=3)
    
    # Pre-allocate output image: [Nx, Ny, Ns]
    img_out = np.zeros((Nx, Ny, Ns), dtype=np.complex64)
    
    # Flatten DCF to match the flattened trajectory [NADC*Npe]
    # Assuming DCF is provided as [NADC, Npe]
    dcf_flat = DCF.flatten()

    for ns in range(Ns):
        # Accumulate corrected segments for this slice
        slice_buffer = np.zeros((Nx, Ny, Nc), dtype=np.complex64)

        for l in range(L):
            # Weights for this segment: [NADC*Npe]
            # Conjugate the temporal coefficients for the adjoint
            weights = np.tile(np.conj(temporal_coeff[l]), Npe)

            for nc in range(Nc):
                # Apply DCF and weights in k-space
                k_segment = ksp_adj[nc, :, :, ns].flatten() * dcf_flat * weights
                
                # 2. 2D Adjoint NUFFT (Type 1)
                # Maps k-space data back to [Nx, Ny] grid
                img_l = finufft.nufft2d1(crdsx, crdsy, k_segment, (Nx, Ny))
                
                # 3. Phase De-modulation (Undo B0)
                # Multiply by conjugate of B0_phasor for this segment
                slice_buffer[:, :, nc] += img_l * np.conj(B0_phasors[:, :, ns, l])
        
        # 4. Coil Combination (SENSE Adjoint)
        # Multiply by conjugate CSM and sum across coils
        img_out[:, :, ns] = np.sum(slice_buffer * np.conj(CSM[:, :, ns, :]), axis=2)

    return img_out


# NUFFT_forward_torch(): GPU based NUFFT forward using PyTorch.
def NUFFT_forward_torch(img, csm_torch, ktraj, weights_torch, DCF, B0_map, t_l, omega, batch_size=8):
    """
    Forward Operator (A): Image Space -> K-space
    img: (Nx, Ny, Ns), complex 3D images, PyTorch tensor.
    csm_torch: (Ns, Nc, Nx, Ny), complex coil sensitivity maps, PyTorch tensor.
    ktraj: (2, Npe*NADC), kspace trajectories coordinates, range from -pi to pi, PyTorch tensor.
    weights_torch: (L, NADC), temporal coefficients for time segmented kspace data, PyTorch tensor.
    DCF: (Npe*NADC), density compensation function, PyTorch tensor.
    B0_map: (Nx, Ny, Ns), 3D static off-resonance maps in Hz, PyTorch tensor.
    t_l: (L, ), central time points for each readout segments, PyTorch tensor.
    omega: (), sampling mask, PyTorch tensor.
    """
    Nx, Ny, Ns = img.shape
    Nc = csm_torch.shape[1]
    L = len(t_l) 
    Np = ktraj.shape[1]
    device = ktraj.device
    
    nufft_obj = tkbn.KbNufft(im_size=(Nx, Ny)).to(device)
    dcf_gpu = DCF.to(device)

    # 1. SENSE Encoding (Full 3D Spatial Volume)
    # [Nc, Nx, Ny, Ns]
    csm_spatial = csm_torch.permute(1, 2, 3, 0)
    img_spatial_coils = img.unsqueeze(0) * csm_spatial

    # 2. Prepare Output
    ksp_full = torch.zeros((Ns, Nc, Np), device='cpu', dtype=torch.complex64)

    # 3. Time Segment Loop
    for l in range(L):
        # A. Apply Phase Accrual to the FULL 3D Volume
        # This MUST be done on the full volume for the Z-FFT to be valid later
        phase_term = torch.exp(-1j * 2 * torch.pi * B0_map * t_l[l]) # [Nx, Ny, Ns]
        img_l_spatial = img_spatial_coils * phase_term.unsqueeze(0) # [Nc, Nx, Ny, Ns]
        
        # B. Full Z-FFT (Spatial -> Hybrid Kz)
        # Now we transform the entire z-dimension correctly
        img_l_hybrid = torch.fft.fftshift(
            torch.fft.fft(torch.fft.ifftshift(img_l_spatial, dim=3), dim=3, norm='ortho'), 
            dim=3
        )
        
        w_l = weights_torch[l] # Temporal weight for this segment

        # C. Batch Loop for 2D NUFFT (Plane-by-Plane)
        for i in range(0, Ns, batch_size):
            end = min(i + batch_size, Ns)
            
            # Slice out the current Kz planes from the hybrid volume
            # [batch, Nc, Nx, Ny]
            img_batch_hz = img_l_hybrid[:, :, :, i:end].permute(3, 0, 1, 2).to(device)
            
            # Perform 2D NUFFT and add to the full k-space
            ksp_l = nufft_obj(img_batch_hz.contiguous(), ktraj, norm='ortho') * torch.sqrt(dcf_gpu)
            
            # Accumulate result (moved back to CPU if ksp_full is large)
            ksp_full[i:end] += (ksp_l * w_l.to(device)).cpu()
            
            del img_batch_hz, ksp_l
            
        del img_l_spatial, img_l_hybrid
        torch.cuda.empty_cache()

        if omega is not None:
            ksp_full = ksp_full.masked_fill(~omega, 0)
        
    return ksp_full


# NUFFT_adjoint_torch(): GPU based NUFFT adjoint using PyTorch.
def NUFFT_adjoint_torch(ksp, csm_torch, ktraj, weights_torch, DCF, B0_map, t_l, batch_size=8):
    """
    Adjoint Operator (A^H): K-space -> Image Space
    ksp: (Ns, Nc, Npe*NADC), complex kspace data, PyTorch tensor.
    csm_torch: (Ns, Nc, Nx, Ny), complex coil sensitivity maps, PyTorch tensor.
    ktraj: (2, Npe*NADC), kspace trajectories coordinates, range from -pi to pi, PyTorch tensor.
    weights_torch: (L, NADC), temporal coefficients for time segmented kspace data, PyTorch tensor.
    DCF: (Npe*NADC), density compensation function, PyTorch tensor.
    B0_map: (Nx, Ny, Ns), 3D static off-resonance maps in Hz, PyTorch tensor.
    t_l: (L, ), central time points for each readout segments, PyTorch tensor.
    """
    Ns, Nc, Np = ksp.shape
    Nx, Ny = csm_torch.shape[2], csm_torch.shape[3]
    L = len(t_l) # Number of time segments
    device = ktraj.device
    
    nufft_adj_obj = tkbn.KbNufftAdjoint(im_size=(Nx, Ny)).to(device)
    dcf_gpu = DCF.to(device)
    
    # Final accumulator in 3D Spatial Domain
    img_final = torch.zeros((Nx, Ny, Ns), device='cpu', dtype=torch.complex64)
    # Prepare SENSE maps
    csm_spatial = csm_torch.permute(1, 2, 3, 0) # [Nc, Nx, Ny, Ns]

    # Strictly reverse the forward operator's outer loop (Time Segments)
    for l in range(L):
        # 1. Initialize Hybrid Space Volume for this segment [Nc, Nx, Ny, Ns]
        img_l_hybrid = torch.zeros((Nc, Nx, Ny, Ns), device='cpu', dtype=torch.complex64)
        
        # Temporal weight for this segment (Conjugate)
        w_l_conj = weights_torch[l].conj().to(device)

        # 2. Batch Loop for 2D Adjoint NUFFT (K-space -> Hybrid Space)
        for i in range(0, Ns, batch_size):
            end = min(i + batch_size, Ns)
            
            ksp_batch = ksp[i:end].to(device)
            # Apply SQRT(DCF) and Temporal Weight
            ksp_weighted = (ksp_batch * torch.sqrt(dcf_gpu) * w_l_conj).contiguous()
            
            # NUFFT: [batch, Nc, Np] -> [batch, Nc, Nx, Ny]
            img_batch_hz = nufft_adj_obj(ksp_weighted, ktraj, norm='ortho')
            
            # Store in Hybrid Volume [Nc, Nx, Ny, batch]
            img_l_hybrid[:, :, :, i:end] = img_batch_hz.permute(1, 2, 3, 0).cpu()
            
            del ksp_batch, img_batch_hz

        # 3. Full Inverse Z-FFT (Hybrid Kz -> Spatial Z)
        # Must see all Ns slices to correctly transform the axial dimension
        img_l_spatial_coils = torch.fft.fftshift(
            torch.fft.ifft(torch.fft.ifftshift(img_l_hybrid, dim=3), dim=3, norm='ortho'), 
            dim=3
        )

        # 4. Conjugate Phase Modulation in pure 3D Spatial Domain
        # phase_term_conj: e^(+1j * 2 * pi * delta_f * t)
        # B0_map and t_l[l] must be absolute as previously discussed
        phase_term_conj = torch.exp(1j * 2 * torch.pi * B0_map * t_l[l]) # [Nx, Ny, Ns]
        
        # 5. SENSE Combine and Accumulate
        # Multiply by conjugate sensitivities and the conjugate phase term
        img_l_final = torch.sum(img_l_spatial_coils * torch.conj(csm_spatial), dim=0) * phase_term_conj
        
        img_final += img_l_final.cpu()
        
        del img_l_hybrid, img_l_spatial_coils, img_l_final
        torch.cuda.empty_cache()
    
    return img_final


# prepare_tensors(): Convert numpy arrays to tensors with reshapes.
def prepare_tensors(CSM, crdsx, crdsy, a_total, DCF, B0_map):
    """
    CSM: (Nx, Ny, Ns, Nc), complex coil sensitivity maps, numpy array.
    crdsx, crdsy: (NADC*Npe, ), kspace trajectories coordinates, range from -pi to pi, numpy array.
    a_total: (L, NADC), temporal coefficients for time segmented kspace data, numpy array.
    DCF: (NADC, Npe), density compensation function, numpy array.
    B0_map: (Nx, Ny, Ns), 3D static off-resonance maps in Hz, numpy array.
    """
    # 1. Coil Sensitivity: [Nx, Ny, Ns, Nc] -> [Ns, Nc, Nx, Ny]
    csm_torch = torch.tensor(CSM).permute(2, 3, 0, 1).to(torch.complex64)
    
    # 2. Trajectory: [2, Nadc, Narm] -> [2, Np]
    ktraj = torch.tensor(np.stack([crdsx, crdsy]), dtype=torch.float32)
    if ktraj.ndim == 3:
        ktraj = ktraj.reshape(2, -1)
    
    # 3. Temporal Weights: [L, Nadc] -> [L, Np]
    # Repeat across spiral arms to match flattened ktraj
    n_arms = crdsx.shape[1] if crdsx.ndim == 2 else 1
    weights_raw = torch.tensor(a_total, dtype=torch.complex64)
    
    if weights_raw.ndim == 2:
        # Expand across the new Arm dimension at the end, then flatten
        weights_torch = weights_raw.unsqueeze(-1).expand(-1, -1, n_arms).reshape(weights_raw.shape[0], -1)
    else:
        weights_torch = weights_raw
    
    # Check if a_total is only for one arm and repeat if necessary
    if weights_torch.ndim == 2 and weights_torch.shape[1] != ktraj.shape[1]:
        weights_torch = weights_torch.repeat(1, n_arms)
    elif weights_torch.ndim == 3:
        # If it has a batch dim [1, L, Nadc], squeeze and repeat
        weights_torch = weights_torch.squeeze(0).repeat(1, n_arms)

    # 4. DCF: Flatten [Nadc, Narm] -> [Np]
    dcf_torch = torch.tensor(DCF, dtype=torch.complex64).reshape(-1)
    
    # 5. Field Map: [Nx, Ny, Ns] -> No permutation needed if already [Nx, Ny, Ns]
    b0_torch = torch.tensor(B0_map, dtype=torch.float32)

    return csm_torch, ktraj, weights_torch, dcf_torch, b0_torch