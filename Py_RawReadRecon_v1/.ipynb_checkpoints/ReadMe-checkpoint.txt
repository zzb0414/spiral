
This program is written for the sole purpose for internal testing exclusively on UIH data. 

This program is under constant changes, and the functions and outcomes are by no means guaranteed, please use at your own risks!

If there are any questions and suggestions, please kindly contact yongquan.ye@united-imaging.com. 

#===================================================================
02/22/2022
Requirements:
1. python >= 3.8
2. numpy >= 1.20.3
3. pydicom (latest)
4. matplotlib >= 3.5.0

Usage:
1. Execute 'ReconBatch.py' in python command 
2. ReadReconDataTemplate.py shows example codes to read the k-space and image results

Limitations:
-- Only 2x GRAPPA with integrated ACS lines is supported (as of 02/22/2022)
-- One can always use K space data for customized recon for undersampling schemes other than 2x GRAPPA