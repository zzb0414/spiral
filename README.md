### Deep learning spiral MRI for off-resonance induced spatial blurring mitigations.
Author: Zhibo Zhu, zhibozhu.one@gmail.com

This is the code repository for deep learning based spiral MRI deblurring.

Notes: Some of the core codes and raw data have been removed due to commercial confidentiality. Therefore no demonstrations nor explanations would be given regarding these removals, and the current codes will **NOT** run as an entirety without having them.

- Repository hierarachy:
  - _spiral_: Root folder containing all major codes.
  - _Grad_Response_Spi_: Original spiral gradient response functions.
  - _Hfun_: Spiral gradient response functions in .npy format.
  - _models_: Trained neural network models.
  - _spiral_base_lines_: Non-AI results as baselines.

- Core scripts and functions:
  - _demo.ipynb_: Main demonstration scripts for the full deblurring pipeline from non-AI approaches to AI-based approaches.
  - _spiral_opt.py_: Spiral option class defining spiral trajectory design parameters.
  - _spiral_obj_py_: Spiral object class defining spiral trajectory objects.
  - _spiral_recon.py_: Spiral reconstruction class defining spiral reconstruction objects.
  - _demo_spi_recon.py_: Entry to non-AI spiral reconstructions including naive (no corrections at all), GIRF only, GIRF + MFI and GIRF + iterative approaches.
  - _nufft_operators.py_: PyTorch based NUFFT operators for iterative reconstruction.
  - _spi_train_dataset.py_: Naive spiral MRI training dataset script (single orientation/view angle) for dataloader.
  - _AugmentedMRIDataset.py_: Augments spiral MRI training dataset script (random rotaion-flip left right-flip up down) for dataloader.
  - _loss_func.py_: Customized weighted l1 loss and gradient loss function classes for encouraging edges and boundaries sharpness.
  - _spiral_deblur_ResNet.py_: Residual neural network classes for spiral MRI deblurring tasks and train loop entry.
  - _spiral_deblur_UNet.py_: U-Net classes for spiral MRI deblurring tasks.
