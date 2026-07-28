"""
Demonstration of spiral reconstruction.
Author: Zhibo Zhu. Date: 04/22/2026.
"""
from pathlib import Path
import copy as cp
import numpy as np
import matplotlib.pyplot as plt
from spiral_opt import spi_opt
from spiral_obj import spi_obj
from spiral_recon import spi_recon
from xml.etree import ElementTree as et
import warnings
warnings.filterwarnings('ignore')
    
def run_pipeline(file_name, prot_name, npy_name, b0map_name, grad_resp_folder, fmax_MFI, L, B, P):
    # Initialize a default spi_recon object. It has default spi_opt and corresponding spi_obj.
    spi_recon1 = spi_recon()
    my_opt = spi_recon1.get_opt()
    
    # Prepare the raw data file name and the protocol name.
    # At the first run, a spi_recon object has to (1) read the raw data to save the data in .npy format and (2) read the protocol to prepare its spi_opt and spi_obj objects.
    # One the .npy format data have been saved, k-space data can be loaded into the memory map for distributed and parallel computation usage.
    # file_name = "//10.31.10.207/uiha-mr/UIHA_MRScanner/zhibozhu/rawdata/jupiter/spiral/20260417/17764379860000/UID_7629462974129938742_gre_spiral__sos_3d.raw"
    # prot_name = "//10.31.10.207/uiha-mr/UIHA_MRScanner/zhibozhu/rawdata/jupiter/spiral/20260417/17764379860000/UID_7629462974129938742_gre_spiral__sos_3d.prot"
    # npy_name = "//10.31.10.207/uiha-mr/UIHA_MRScanner/zhibozhu/rawdata/jupiter/spiral/20260417/17764379860000/UID_7629462974129938742_K/k_1690x48x80x1_CHA34_Ech0_Set0_Rep0_Ave0_UD_0_0_0_0_0.npy"
    
    # Write spiral design related parameters into a spi_param dictionary.
    prot_tree = et.parse(prot_name)
    spi_param = {}
    spi_param['Gmax'] = float(prot_tree.find('.//SpiralGMax_gs_cm/Value').text) # [G/cm]
    spi_param['Smax'] = float(prot_tree.find('.//SpiralSMax_T_per_m_s/Value').text) # [mT/m/ms]
    spi_param['grad_raster_time'] = 10 # [us]
    # spi_param['calc_time_factor'] = float(prot_tree.find('.//CalcTimeFactor/Value').text) if prot_tree.find('.//CalcTimeFactor/Value').text else 1
    spi_param['calc_time_factor'] = 1
    spi_param['FOV'] = float(prot_tree.find('.//FOVro/Value').text) * 1e-3 # [m]
    spi_param['Nx'] = int(prot_tree.find('.//MatrixRO/Value').text) # Number of Nx voxels
    spi_param['Ns'] = int(prot_tree.find('.//SpiralShotNum/Value').text) # Number of spiral arms
    spi_param['N_acc_in'] = float(prot_tree.find('.//SpiralNAccIn/Value').text) # Inner kspace acceleration factor
    spi_param['N_acc_out'] = float(prot_tree.find('.//SpiralNAccOut/Value').text) # Outer kspace accerleration factor
    spi_param['dwell'] = float(prot_tree.find('.//Dwelltime/Value').text) * 1e-3 / 2 # [us]
    spi_param['mode'] = 0b10 # Binary 2
    
    # Set parameters for my_opt, set the updated my_opt as the spi_opt in spi_recon object, and set the spi_obj automatically by the keyword None.
    my_opt.set_value(**spi_param)
    # my_opt.disp_param('all')
    spi_recon1.set_opt(my_opt)

    # spi_recon1 does read_raw(file_name) and does the baseline recon: No GIRF, no MFI.
    if not Path(npy_name).is_file():
        spi_recon1.read_raw(file_name)
        print('haha')
    spi_recon1.fname = npy_name
    ksp = np.load(npy_name, mmap_mode='r') # Avoid repeating loading raw data.
    Nc, NADC, Npe, NS, _ = ksp.shape
    spi_param['NADC'] = NADC - 10
    my_opt.set_value(**{'NADC': NADC - 10})
    my_opt.disp_param('all')
    spi_recon1.set_opt(my_opt)
    
    spi_recon1.set_obj(None)
    img_no_girf = spi_recon1.run(ksp, freq=0)
    
    # spi_recon2 adds GIRF corrections which is the true baseline (which is expected to be seen on a scanner).
    spi_recon2 = cp.deepcopy(spi_recon1)
    spi_recon2.CSM = None # Re-do CSM when GIRF is applied.
    # %run load_Hfun_Jupiter.py
    freq_axis_name = grad_resp_folder + '\\freq_axis.npy'
    Hfun_name = grad_resp_folder + '\\Hfun.npy'
    spi_recon2.get_obj().apply_GIRF(freq_axis_name, Hfun_name)
    spi_recon2.get_obj().calc_dcf("analytical2")
    img_girf = spi_recon2.run(ksp, freq=0)
    
    ns = 40
    plt.figure()
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    axes[0][0].imshow(np.abs(np.transpose(img_no_girf[..., ns])), cmap='grey')
    axes[0][1].imshow(np.angle(np.transpose(img_no_girf[..., ns])), cmap='jet')
    axes[1][0].imshow(np.abs(np.transpose(img_girf[..., ns])), cmap='grey')
    axes[1][1].imshow(np.angle(np.transpose(img_girf[..., ns])), cmap='jet')
    
    axes[0][0].set_title("No GIRF, magnitude")
    axes[0][1].set_title("No GIRF, phase (-$\pi$, +$\pi$]")
    axes[1][0].set_title("GIRF, magnitude")
    axes[1][1].set_title("GIRF, phase (-$\pi$, +$\pi$]")
    
    plt.show()

    # Process the B0 map.
    import pydicom
    from skimage.restoration import unwrap_phase
    from scipy.ndimage import median_filter
    
    b0_name = Path(b0map_name)
    b0map = []
    for file in b0_name.glob("*.dcm"):
        ds = pydicom.dcmread(file)
        b0map.append(ds.pixel_array)
    b0map = np.array(b0map) * ds.WindowWidth / (ds.LargestImagePixelValue - ds.SmallestImagePixelValue) # Convert from Integer to Hertz.
    b0map = np.transpose(b0map) # Manually tranpose the DICOM to match with reconstruction orientation.
    
    # In case of phase wrapping occurs, map the frequency values into radians, unwrap and map back.
    b0max = np.max(b0map[...])
    unwrapped_map = unwrap_phase(b0map * np.pi / b0max) * b0max / np.pi
    cleaned_map = median_filter(unwrapped_map, size=3)
    
    # spi_recon2 tests iterative reconstruction.
    TE = float(prot_tree.find('.//TE/Value').text) / 1e6
    print(f"TE: {TE} sec.")
    img_IR = spi_recon2.run_iterative_recon(ksp, cleaned_map, TE=TE, acc=False, iters=5, L=L, B=B, fmax=fmax_MFI)

    plt.figure()
    plt.imshow(np.transpose(np.abs(img_IR[:, :, ns])), cmap='grey')
    plt.show()
    
    # spi_recon3 tests multiple different ADC delay samples.
    # This is purely experimental. Still looking for the reason when this is needed and which value is the most proper.
    # spi_recon3 = cp.deepcopy(spi_recon1) # Deepcopy spi_recon1 (no GIRF yet)
    # spi_recon3.CSM = None
    # manual_delays = [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]
    # img_ADC_delay = np.zeros(img_girf.shape + (len(manual_delays), ))
    # for ii, manual_delay in enumerate(manual_delays):
    #     spi_recon3.get_obj().manual_delay = manual_delay # Set manual delay
    #     spi_recon3.set_obj(None) # Re-run spiral object
    #     spi_recon3.get_obj().apply_GIRF(freq_axis_name, Hfun_name) # Now do GIRF corrections
    #     spi_recon3.get_obj().calc_dcf("analytical2") # Now calculate new DCF
    #     img_ADC_delay[..., ii] = spi_recon3.run(ksp, freq=0) # Now re-run recon and save the result
    
    # plt.figure();
    # fig, axes = plt.subplots(2, 5, figsize=(18, 7.2))
    # axes[0, 0].imshow(np.transpose(np.abs(img_no_girf[:, :, 17])) / 1e6, cmap='grey');
    # axes[1, 0].imshow(np.transpose(np.abs(img_girf[:, :, 17])) / 1e6, cmap='grey');
    # axes[0][0].set_title("No GIRF, magnitude")
    # axes[1][0].set_title("GIRF, magnitude")
    # # for ii in range(2):
    #     for jj in range(4):
    #         axes[ii, jj+1].imshow(np.transpose(np.abs(img_ADC_delay[:, :, 17, ii * 4 + jj])) / 1e6, cmap='grey');
    #         axes[ii, jj+1].set_title(f"GIRF, magnitude, delay={manual_delays[ii * 4 + jj]}")
    # plt.show()
    
    # Multiple frequency interpolation parts.
    # For accleration purpose, prepare for parallel computation.
    from multiprocessing import Pool
    
    # spi_recon4 tests MFI.
    spi_recon4 = cp.deepcopy(spi_recon2) # Deepcopy spi_recon2 (GIRF applied). What is crucial here is do not erase spi_recon2()'s CSM. Doing so is both fast and proper.
    freq_bins = np.arange(-fmax_MFI, fmax_MFI+1, 20)
    # print(freq_bins)
    img_MFI = np.zeros(img_girf.shape + (len(freq_bins), )).astype(np.complex64)
    # print(img_MFI.shape)
    args = [(ksp, freq) for freq in freq_bins]
    with Pool(processes=P) as pool:
        img_f = pool.starmap(spi_recon4.run, args) # Now re-run recon and save the result
    
    img_MFI = np.stack(img_f, axis=-1)
    
    # Calculate the theoretical MFI coefficients.
    t = spi_recon4.get_obj().dwell_time - spi_recon4.get_obj().dwell_time[0]
    j_vec = np.column_stack([np.exp(1j * 2 * np.pi * freq_bins[m] * t) for m in range(len(freq_bins))]) / np.sqrt(len(t))
    A = np.column_stack([np.exp(1j * 2 * np.pi * freq_bins[m] * t) for m in range(len(freq_bins))]) / np.sqrt(len(t))
    c = np.linalg.pinv(A) @ j_vec
    
    plt.figure()
    plt.imshow(np.abs(c))
    plt.title("MFI coefficients")
    plt.show()
    
    # plt.figure()
    # plt.imshow(np.transpose(cleaned_map[:, :, ns]), cmap='hot', vmin=-b0max, vmax=b0max)
    # plt.title("B0 map")
    # plt.colorbar()
    # plt.show()
    
    # Combining MFI.
    Nx, Ny, Ns, Nf = img_MFI.shape
    img_final = np.zeros((Nx, Ny, Ns)).astype(np.complex64)
    diffs = np.abs(cleaned_map[..., np.newaxis] - freq_bins)
    indices = np.argmin(diffs, axis=-1)
    coeffs_per_pixel = c[:, indices].transpose(1, 2, 3, 0)
    img_final = np.sum(img_MFI * coeffs_per_pixel, axis=-1)
    
    # Final display.
    plt.figure()
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes[0][0].imshow(np.transpose(b0map[:, :, ns]), cmap='bwr', vmin=-b0max, vmax=b0max)
    axes[0][1].imshow(np.transpose(cleaned_map[:, :, ns]), cmap='bwr', vmin=-b0max, vmax=b0max)
    axes[1][0].imshow(np.transpose(np.abs(img_MFI[:, :, ns, Nf//2])), cmap='grey')
    axes[1][1].imshow(np.transpose(np.abs(img_final[:, :, ns])), cmap='grey')

    axes[0][0].set_title("Original static off-resonance")
    axes[0][1].set_title("Smoothed static off-resonance")
    axes[1][0].set_title("GIRF, magnitude")
    axes[1][1].set_title("MFI, magnitude")
    plt.show()

    plt.figure()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(np.transpose(np.abs(img_girf[:, :, ns])), cmap='grey')
    axes[1].imshow(np.transpose(np.abs(img_final[:, :, ns])), cmap='grey')
    axes[2].imshow(np.transpose(np.abs(img_IR[:, :, ns].cpu())), cmap='grey')

    axes[0].set_title("GIRF, magnitude")
    axes[1].set_title("MFI, magnitude")
    axes[2].set_title("Iterateive, magnitude")

    return img_no_girf, img_girf, img_final, img_MFI, img_IR, cleaned_map, spi_recon2