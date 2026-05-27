'''
description: the imdisp function can coveniently display a multislice image array
author: zheng
date: 07/19/2022
email: zzhong21@stanford.edu
'''
import matplotlib.pyplot as plt
import numpy as np
import math

def get_layout(slice_num):
    raw_row = math.floor(np.sqrt(slice_num))
    real_col = math.ceil(slice_num / raw_row)
    real_row = math.ceil(slice_num / real_col)

    layout = [real_row, real_col]
    return layout


def imdisp(im_in, layout=[]):
    print ("display images: ", im_in.shape)
    ndim = im_in.ndim
    if ndim > 3 or ndim < 2 :
        print ("imdisp cannot handle dimention bigger than 3 or smaller than 2!!!")
        return

    if ndim == 2:
        fig = plt.figure(figsize=(10, 7))
        plt.rcParams['image.cmap'] = 'gray'
        plt.imshow(im_in)
        plt.axis('off')
    
    if ndim == 3:
        #display for loop
        slice_num = im_in.shape[2];
        if len(layout) == 0:
            layout = get_layout(slice_num);
        if len(layout) != 2:
            print ("the layout should be row by col!!!")
            return
        elif (layout[0] * layout[1]) != slice_num:
            print ("the layout row by col is not equal to image shape!!! will resize the layout")
            layout = get_layout(slice_num);

        print("layout (row, col) is : ", layout)
        fig = plt.figure(figsize=(10, 7))
        plt.rcParams['image.cmap'] = 'gray'
        for i in range(slice_num):
            fig.add_subplot(layout[0], layout[1], i+1)
            plt.imshow(im_in[:,:,i])
            plt.axis('off')

    plt.show()
    return


if __name__ == "__main__":
    '''
    #this is only for testing
import os
import os.path
import cfl

    root_path = 'D:\work_stanford\github\WSSFSE\wess-recon\wess_python\data'
    file_path = os.path.join(root_path, "img_all_g")
    img = cfl.readcfl(file_path)
    #im_in = np.squeeze(img[:,:,15,:])
    im_in = np.squeeze(img[:,:,1,1])
    imdisp(np.abs(im_in), [])
    '''
