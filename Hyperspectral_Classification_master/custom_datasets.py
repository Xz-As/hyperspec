from .utils import open_file
import numpy as np

CUSTOM_DATASETS_CONFIG = {
         'DFC2018_HSI': {
            'img': '2018_IEEE_GRSS_DFC_HSI_TR.HDR',
            'gt': '2018_IEEE_GRSS_DFC_GT_TR.tif',
            'download': False,
            'loader': lambda folder: dfc2018_loader(folder)
            },
         'sh1':{
            'img': 'sh1.mat',
            'gt': 'sh1_gt.mat',
            'download': False,
            'loader': lambda folder: sh1_loader(folder)
         },
         'sh2':{
            'img': 'sh2.mat',
            'gt': 'sh2_gt.mat',
            'download': False,
            'loader': lambda folder: sh2_loader(folder)
         }
}

def dfc2018_loader(folder):
        img = open_file(folder + '2018_IEEE_GRSS_DFC_HSI_TR.HDR')[:,:,:-2]
        gt = open_file(folder + '2018_IEEE_GRSS_DFC_GT_TR.tif')
        gt = gt.astype('uint8')

        rgb_bands = (47, 31, 15)

        label_values = ["Unclassified",
                        "Healthy grass",
                        "Stressed grass",
                        "Artificial turf",
                        "Evergreen trees",
                        "Deciduous trees",
                        "Bare earth",
                        "Water",
                        "Residential buildings",
                        "Non-residential buildings",
                        "Roads",
                        "Sidewalks",
                        "Crosswalks",
                        "Major thoroughfares",
                        "Highways",
                        "Railways",
                        "Paved parking lots",
                        "Unpaved parking lots",
                        "Cars",
                        "Trains",
                        "Stadium seats"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values#, palette

def sh1_loader(folder):
        img = open_file(folder + 'sh1.mat')['sh1']
        gt = open_file(folder + 'sh1_gt.mat')['sh1_gt']
        #gt = np.array(gt['sh1'])
        gt = gt.astype('uint8')
        rgb_bands = (165, 99, 33)

        label_values = ["Water",
                        "Ground/Building",
                        "Plant"]
        ignored_labels = []
        return img, gt, rgb_bands, ignored_labels, label_values#, palette




def sh2_loader(folder):
        img = open_file(folder + 'sh2.mat')['sh2']
        gt = open_file(folder + 'sh2_gt.mat')['sh2_gt']

        gt = gt.astype('uint8')
        rgb_bands = (165, 99, 33)

        label_values = ["Water",
                        "Ground/Building",
                        "Plant"]
        ignored_labels = []
        return img, gt, rgb_bands, ignored_labels, label_values#, palette
