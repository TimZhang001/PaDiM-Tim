import imp
import os
import torch
import numpy as np
import datasets.mvtec as mvtec
import argparse
import cv2

from  utils.funcs       import rot_img, translation_img, hflip_img, rot90_img, grey_img
from  utils.util        import denormalization
from  torch.utils.data  import DataLoader
from  tqdm              import tqdm


# 图像的扩充
def augment_image(support_img):
    augment_support_img = support_img
    
    # hflip img
    flipped_img         = hflip_img(support_img)
    show_img            = flipped_img[0].permute(1, 2, 0).cpu().numpy()
    augment_support_img = torch.cat([augment_support_img, flipped_img], dim=0)
    
    # rgb to grey img
    greyed_img          = grey_img(support_img)
    show_img            = greyed_img[0].permute(1, 2, 0).cpu().numpy()
    augment_support_img = torch.cat([augment_support_img, greyed_img], dim=0)
    
    # rotate img in 90 degree
    for angle in [1,2,3]:
        rotate90_img        = rot90_img(support_img, angle)
        show_img            = rotate90_img[0].permute(1, 2, 0).cpu().numpy()
        augment_support_img = torch.cat([augment_support_img, rotate90_img], dim=0)

    # rotate img with small angle
    for angle in [-np.pi/4, -3 * np.pi/16, -np.pi/8, -np.pi/16, np.pi/16, np.pi/8, 3 * np.pi/16, np.pi/4]:
        rotate_img          = rot_img(support_img, angle)
        show_img            = rotate_img[0].permute(1, 2, 0).cpu().numpy()
        augment_support_img = torch.cat([augment_support_img, rotate_img], dim=0)
    
    # translate img
    for a,b in [(0.2,0.2), (-0.2,0.2), (-0.2,-0.2), (0.2,-0.2), (0.1,0.1), (-0.1,0.1), (-0.1,-0.1), (0.1,-0.1)]:
        trans_img           = translation_img(support_img, a, b)
        show_img            = trans_img[0].permute(1, 2, 0).cpu().numpy()
        augment_support_img = torch.cat([augment_support_img, trans_img], dim=0)

    return augment_support_img

def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--data_path',  type=str, default='./MVTec/MVTec_AD')
    parser.add_argument('--model_path', type=str, default='./save_checkpoints')
    parser.add_argument('--arch',       type=str, choices=['resnet18', 'wide_resnet50_2'], default='resnet18')
    parser.add_argument('--good_num',   type=int, default=10000)
    return parser.parse_args()

def main():

    args  = parse_args()
    for class_name in ['metal_nut']:
        train_dataset    = mvtec.MVTecDatasetAugment(args.data_path, class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=1, pin_memory=True)

        for (x, y, mask, path) in tqdm(train_dataloader, '| feature extraction | test | %s |' % class_name):
            augment_img = augment_image(x)
            for i in range(len(path)):
                for j in range(22):
                    cur_path  = path[i]
                    save_path = os.path.join(args.data_path, class_name, 'train_augment', 'good')
                    save_name = os.path.join(save_path, cur_path.split('/')[-1].split('.')[0] + '_' + str(j) + '.png')
                    os.makedirs(save_path, exist_ok=True)
                    img_index = i * 22 + j
                    cur_image = augment_img[img_index].cpu().numpy()
                    cur_image = denormalization(cur_image)
                    cv2.imwrite(save_name, cur_image)

if __name__ == '__main__':
    main()