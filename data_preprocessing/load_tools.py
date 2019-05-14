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
       
    
    
def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]    



def load_itkfilewithtrucation(filename, upper=200, lower=-200):
    """"Code for this function is based on code from this repository: https://github.com/junqiangchen/LUNA16-Lung-Nodule-Analysis-2016-Challenge"""
    
    # 1,tructed outside of liver value
    srcitkimage = sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32)
    srcitkimagearray = sitk.GetArrayFromImage(srcitkimage)
    srcitkimagearray[srcitkimagearray > upper] = upper
    srcitkimagearray[srcitkimagearray < lower] = lower
    # 2,get tructed outside of liver value image
    sitktructedimage = sitk.GetImageFromArray(srcitkimagearray)
    origin = np.array(srcitkimage.GetOrigin())
    spacing = np.array(srcitkimage.GetSpacing())
    sitktructedimage.SetSpacing(spacing)
    sitktructedimage.SetOrigin(origin)
    # 3 normalization value to 0-255
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    itkimage = rescalFilt.Execute(sitk.Cast(sitktructedimage, sitk.sitkFloat32))
    return itkimage


def get_cube_from_img(img3d, center, block_size):
    """"Code for this function is based on code from this repository: https://github.com/junqiangchen/LUNA16-Lung-Nodule-Analysis-2016-Challenge"""
    # get roi(z,y,z) image and in order the out of img3d(z,y,x)range
    center_z = center[0]
    center_y = center[1]
    center_x = center[2]
    start_x = max(center_x - block_size / 2, 0)
    if start_x + block_size > img3d.shape[2]:
        start_x = img3d.shape[2] - block_size
    start_y = max(center_y - block_size / 2, 0)
    if start_y + block_size > img3d.shape[1]:
        start_y = img3d.shape[1] - block_size
    start_z = max(center_z - block_size / 2, 0)
    if start_z + block_size > img3d.shape[0]:
        start_z = img3d.shape[0] - block_size
    start_z = int(start_z)
    start_y = int(start_y)
    start_x = int(start_x)
    roi_img3d = img3d[start_z:start_z + block_size, start_y:start_y + block_size, start_x:start_x + block_size]
    return roi_img3d


def load_itk_version2(img_file):
    itk_img = load_itkfilewithtrucation(img_file, 600, -1000)
    img_array = sitk.GetArrayFromImage(itk_img)

    origin = np.array(itk_img.GetOrigin())
    spacing = np.array(itk_img.GetSpacing())
    
    return img_array, origin, spacing


def train_test_split_D1():
   
    # SPLIT NONNODULES
    negs = '/datagrid/temporary/dobkomar/output_path_32/0/'
    nonnodules =[os.path.join(negs,f) for f in os.listdir(negs) if os.path.isfile(os.path.join(negs,f))]
    seriesuid = list(set([ n.split('/0/')[1].split('_')[3] for n in nonnodules]))
    nonnodules_train, nonnodules_val = [], []

    dict_ser_train, dict_ser_val = {key: 0 for key in seriesuid}, {key: 0 for key in seriesuid}
    for el in nonnodules:
        cur_ser =  el.split('/0/')[1].split('_')[3]

        if  dict_ser_train[cur_ser] <= 17:  
            nonnodules_train.append(el)
            dict_ser_train[cur_ser] += 1

        if  dict_ser_val[cur_ser] <= 9:  
            if el not in nonnodules_train:
                nonnodules_val.append(el)
                dict_ser_val[cur_ser] += 1

    print(['Nonnodules:',len(nonnodules_train), len(nonnodules_val)])
    assert len(np.intersect1d(nonnodules_train, nonnodules_val)) == 0
    
    train_non = pd.DataFrame(nonnodules_train, columns=['filename'])
    train_non['class'] = [0 for i in range(len(nonnodules_train))]
    val_non = pd.DataFrame(nonnodules_val, columns=['filename'])
    val_non['class'] = [0 for i in range(len(nonnodules_val))]

    train_non.to_csv('/home.stud/dobkomar/data/train_data_0_D1_.csv', index=False)
    val_non.to_csv('/home.stud/dobkomar/data/val_data_0_D1_.csv', index=False)
    
    
    # SPLIT NODULES
    poss = '/datagrid/temporary/dobkomar/output_path_32/augmented/' 
    nodules = [os.path.join(poss,f) for f in os.listdir(poss) if os.path.isfile(os.path.join(poss,f))]
    all_nodules = list(set([x.split('aug_10/')[1].split('_')[0] for x in nodules]))
    nod_train, nod_val = all_nodules[:1200] , all_nodules[1200:]
    nodules_train = [x for x in nodules if x.split('augmented/')[1].split('_')[0]  in nod_train]
    nodules_val = [x for x in nodules if x.split('augmented/')[1].split('_')[0]  in nod_val]
    
    train = pd.DataFrame(nonnodules_train+nodules_train, columns=['filename'])
    train['index'] = [0 for i in range(len(nonnodules_train))]+[1 for i in range(len(nodules_train))]

    val = pd.DataFrame(nonnodules_val+nodules_val, columns=['filename'])
    val['index'] = [0 for i in range(len(nonnodules_val))]+[1 for i in range(len(nodules_val))]
    
    train.to_csv('/home.stud/dobkomar/data/train_data.csv', index=False)
    val.to_csv('/home.stud/dobkomar/data/val_data.csv', index=False)
    
    return "saved!"
