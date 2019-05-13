import sys, os
import numpy as np
import SimpleITK as sitk
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image


def show_images(images, cols = 1, titles = None):
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

    
def load_itk(filename):
    """Reading a '.mhd' file using SimpleITK. Returns the image array, origin and spacing of the image"""
    itkimage = sitk.ReadImage(filename)
    ct_scan = sitk.GetArrayFromImage(itkimage)  # axis in the order z,y,x
    
    # origin is used to convert the coordinates from world to voxel 
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    for i, el in enumerate(voxel_coordinates):
        voxel_coordinates[i] = int(el)
    return voxel_coordinates


def save_to_npy(input_path, folder_output, candidates, subs_id, width_size=32):
    all_files = [f for f in listdir(input_path) if isfile(join(input_path, f)) and '.mhd' in f]
    cands = pd.read_csv(candidates, header=None)

    y_train = []
    counter = 0
    for name in tqdm(all_files):
        image_samples = cands[cands[0] == name[:-4]]
        for index, row in image_samples.iterrows():
            image_name = join(input_path, name)
            # GET COORDINATES
            lung_img = sitk.GetArrayFromImage(sitk.ReadImage(image_name))
            _, orig, spac = load_itk(image_name)
            vox_coords = world_2_voxel([float(row[3]), float(row[2]), float(row[1])], orig, spac)

            # CLASS: 0 - nonnodule, 1 - nodule
            y_class = int(row[4])

            # CROP IMAGE
            w = width_size / 2
            patch = lung_img[int(vox_coords[0] - w): int(vox_coords[0] + w),
                    int(vox_coords[1] - w): int(vox_coords[1] + w),
                    int(vox_coords[2] - w): int(vox_coords[2] + w)]

            # print([row[0], patch.shape, y_class])
            try:
                if y_class == 1:
                    # FOR NODULES WE ADD ALL IMAGES
                    if width_size < 60:
                        for p in range(width_size):
                            np.save(join(folder_output, 'X_train_{}_{}.npy'.format(str(folder_output[-1]), counter)),
                                        np.resize(patch[p, :, :], (1, 32, 32)))
                            y_train.append(y_class)
                            counter += 1

                    else:
                        for p in range(16, 48):
                            np.save(join(folder_output, 'X_train_{}_{}.npy'.format(str(folder_output[-1]), counter)),
                                    np.resize(patch[p, :, :], (1, 64, 64)))
                            y_train.append(y_class)
                            counter += 1

                elif y_class == 0:
                    rand_img = randint(0, width_size - 1)  # choosing randomimage out of all nonnodules
                    np.save(join(folder_output, 'X_train_{}_{}.npy'.format(str(folder_output[-1]), counter)),
                                np.resize(patch[rand_img, :, :], (1, 32, 32)))
                    y_train.append(y_class)
                    counter += 1

            except:
                pass
    np.save(join(folder_output, 'y_train_{}.npy'.format(str(folder_output[-1]))), y_train)

    return 'Done!'