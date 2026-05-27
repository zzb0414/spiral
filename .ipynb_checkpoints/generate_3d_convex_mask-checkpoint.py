import numpy as np
from scipy import ndimage
from skimage import morphology, filters

def generate_3d_convex_mask(volume, sf=0.5, dilation_voxels=5):
    """
    Generates a 3D convex hull mask that is slightly dilated.
    volume: 3D numpy array (Magnitude image)
    dilation_voxels: How many voxels to expand the mask
    """
    # 1. Preliminary Thresholding
    # Using Otsu to find a global threshold, or a small fraction of max
    thresh = filters.threshold_otsu(volume)
    binary_map = volume > (thresh * sf) # Be conservative to include dim artifacts
    
    # 2. 3D Convex Hull
    # skimage's convex_hull_image works slice-by-slice, but we want 3D
    # For a phantom, usually slice-by-slice + 3D dilation is robust
    convex_mask = np.zeros_like(binary_map)
    for i in range(volume.shape[0]):
        if np.any(binary_map[i]):
            convex_mask[i] = morphology.convex_hull_image(binary_map[i])
            
    # 3. 3D Dilation (The "Safety Margin")
    # This ensures we cover "leaking" artifacts just outside the boundary
    struct = ndimage.generate_binary_structure(3, 1) # 6-connectivity
    dilated_mask = ndimage.binary_dilation(convex_mask, structure=struct, iterations=dilation_voxels)
    
    # 4. Fill any remaining 3D holes (just in case)
    final_mask = ndimage.binary_fill_holes(dilated_mask)
    
    return final_mask.astype(np.float32)

# Usage Example:
# phantom_mask = generate_3d_convex_mask(target_volume_3d, dilation_voxels=4)