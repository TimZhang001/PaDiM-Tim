'''
说    明：MvTec数据集中，PaDiM的测试程序
作    者：Zhangshengsen
创建时间：2022-10-18
'''

import argparse
import torch
import random
import os
import pickle
import scipy
import time

import numpy as np
import datasets.mvtec as mvtec
import torch.nn.functional as F

from scipy.spatial.distance import mahalanobis
from torchvision.models import wide_resnet50_2, resnet18
from random             import sample
from torch.utils.data   import DataLoader
from collections        import OrderedDict
from tqdm               import tqdm
from utils.util         import embedding_concat, get_aupr_curve, iou_curve, visualize_CovMatrix
from utils.util         import print_log, get_layers_feature_map, visualize_results, get_save_path, visualize_results_list
from scipy.ndimage      import gaussian_filter

from sklearn.metrics    import roc_auc_score

# device setup
device   = torch.device('cuda')

def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--data_path',  type=str, default='./MVTec/MVTec_AD')
    parser.add_argument('--model_path', type=str, default='./save_checkpoints')
    parser.add_argument('--log_path',   type=str, default='./logs_mvtec')
    parser.add_argument('--vis',        type=int, default=1)
    parser.add_argument('--arch',       type=str, choices=['resnet18', 'wide_resnet50_2'], default='resnet18')
    parser.add_argument('--kmeans',     type=int, default=1)
    parser.add_argument('--idx_num',    type=int, default=100) #100 448 550 1792
    return parser.parse_args()

def prepare_models(arch, idx_num):
    # load model
    if arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d   = 448
        d     = 100
        d     = idx_num
    elif arch == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d   = 1792
        d     = 550
        d     = idx_num

    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
    torch.cuda.manual_seed_all(1024)
        
    idx = torch.tensor(sample(range(0, t_d), d))

    return model, idx

def prepare_data(args, class_name):
    test_dataset    = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, pin_memory=True)
    test_outputs    = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    return test_dataloader, test_outputs

def prepare_save_path(args, class_name):
    train_model_path = os.path.join(args.model_path, '{}'.format(args.arch), 'kmeans_{}_idx_{}'.format(args.kmeans, args.idx_num))
    train_model_path = os.path.join(train_model_path, '%s_train.pkl' % class_name)
    return train_model_path

def load_padim_model(model_path):
    print('load train set feature from: %s' % model_path)
    with open(model_path, 'rb') as f:
        train_outputs = pickle.load(f)
    return train_outputs

def cal_mahalanobis_distance(test_outputs, train_outputs, idx):
    # Embedding concat
    embedding_vectors = test_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

    # randomly select d dimension
    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
    
    # calculate distance matrix
    B, C, H, W        = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()

    dist_list         = []
    model_num         = int(len(train_outputs[0]))
    for j in range(model_num):
        cur_dist_list = []
        for i in tqdm(range(H * W)):
            mean     = train_outputs[0][j, :, i]
            cov      = train_outputs[1][j, :, :, i]
            conv_inv = np.linalg.inv(cov)
            dist     = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            cur_dist_list.append(dist)

        cur_dist_list = np.array(cur_dist_list).transpose(1, 0).reshape(B, H, W)
        dist_list.append(cur_dist_list)

    # min of the dist_list
    dist_list = np.array(dist_list)
    dist_map  = np.min(dist_list, axis=0)

    return dist_map, dist_list

def get_score_map(dist_map, x):
    # upsample
    dist_map  = torch.tensor(dist_map)
    scores    = []
    if len(dist_map.shape) == 4:
        for idx in range(0, dist_map.shape[0]):
            cur_dist_map  = dist_map[idx, :, :, :]
            cur_score_map = F.interpolate(cur_dist_map.unsqueeze(1), size=x.size(2), mode='bilinear', align_corners=False).squeeze().numpy()
            # apply gaussian smoothing on the score map
            for i in range(cur_score_map.shape[0]):
                cur_score_map[i] = gaussian_filter(cur_score_map[i], sigma=4)
            
            # Normalization
            max_score  = cur_score_map.max()
            min_score  = cur_score_map.min()
            cur_scores = (cur_score_map - min_score) / (max_score - min_score)
            scores.append(cur_scores)
    else:
        score_map = F.interpolate(dist_map.unsqueeze(1), size=x.size(2), mode='bilinear', align_corners=False).squeeze().numpy()
        
        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores    = (score_map - min_score) / (max_score - min_score)

    return scores


def main():

    args      = parse_args()
    localtime = time.asctime( time.localtime(time.time()) )
    log       = open(os.path.join(args.log_path, 'log_{}_kmean{}_idx{}.txt'.format(args.arch, args.kmeans, args.idx_num)), 'a+')

    print_log('\n\n\n--------------------------------------------------------------', log)
    print_log('-------start test time is {}----------------'.format(localtime), log)

    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)

    torch.cuda.set_device(2)

    #
    image_auc_list  = []
    pixel_auc_list  = []
    image_aupr_list = []
    pixel_aupr_list = []
    iou_score_list  = []
    
    # 
    #for class_name in mvtec.CLASS_NAMES:
    for class_name in ['grid', 'cable','capsule','pill','screw','toothbrush','transistor']: #
        # prepare model
        model, idx = prepare_models(args.arch, args.idx_num)
        
        # set model's intermediate outputs
        outputs = []
        def hook(module, input, output):
            outputs.append(output)

        model.layer1[-1].register_forward_hook(hook)
        model.layer2[-1].register_forward_hook(hook)
        model.layer3[-1].register_forward_hook(hook)

        # prepare data
        test_dataloader, test_outputs = prepare_data(args, class_name)
        
        # extract train set features
        train_model_path = prepare_save_path(args, class_name)

        if not os.path.exists(train_model_path):
            print('can not find the the padim model %s' % class_name)
            continue
        
        # load train set features
        train_outputs = load_padim_model(train_model_path)
        model_num     = int(len(train_outputs[0]))
        #assert model_num == args.kmeans + 1

        gt_list      = []
        gt_mask_list = []
        test_imgs    = []
        
        # extract test set features
        for (x, y, mask, path) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            
            # model prediction
            with torch.no_grad():
                _ = model(x.to(device))
            
            # get intermediate layer outputs
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
            
            # initialize hook outputs
            outputs = []
        
        # 得到特征图
        query_features = get_layers_feature_map(test_outputs)

        # 通道拼接
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)
        
        # calculate mahalanobis distance
        dist_map, map_list = cal_mahalanobis_distance(test_outputs, train_outputs, idx)

        # upsample
        scores      = get_score_map(dist_map, x)
        scores_list = get_score_map(map_list, x)


        # ground truth
        gt_list   = np.asarray(gt_list)
        gt_mask   = np.asarray(gt_mask_list)
        gt_mask   = (gt_mask > 0.5).astype(np.int_)
        
        # AUC--------------------------------------------------------------------
        # calculate image-level ROC AUC score
        img_scores  = scores.reshape(scores.shape[0], -1).max(axis=1)
        img_auc     = roc_auc_score(gt_list, img_scores)
        image_auc_list.append(img_auc)

        # calculate perpixel level AUC
        pixel_auc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        pixel_auc_list.append(pixel_auc)

        # AUPR--------------------------------------------------------------------
        # calculate image-level AUPR
        img_aupr, cls_th, _, _   = get_aupr_curve(gt_list.flatten(), img_scores.flatten())
        image_aupr_list.append(img_aupr)

        # calculate perpixel level AUPR
        pixel_aupr, seg_th, _, _ = get_aupr_curve(gt_mask.flatten(), scores.flatten())
        pixel_aupr_list.append(pixel_aupr)

        # iou---------------------------------------------------------------------
        fpr, iou, thresh_ = iou_curve(gt_mask.flatten(), scores.flatten())
        iou_score         = iou.max()
        seg_thresh        = float(scipy.interpolate.interp1d(iou, thresh_)(iou_score))
        iou_score_list.append(iou_score)

        # print results
        print_log('finish %s' % class_name, log)
        print_log('Image-level AUC/AUPR: {:.4f} {:.4f}, Pixel-level AUC/AUPR/IOU: {:.4f} {:.4f} {:.4f}'.format(img_auc, img_aupr, pixel_auc, pixel_aupr, iou_score), log)

        if (args.vis):
            # save image path
            image_dir = get_save_path(class_name, args.arch, add_str=args.kmeans)
            #visualize_results(test_imgs, scores, img_scores, gt_mask_list, query_features, seg_th, cls_th, image_dir, class_name)
            visualize_results_list(test_imgs, scores, img_scores, gt_mask_list, query_features, seg_th, cls_th, image_dir, class_name, scores_list)
        
    mean_img_auc    = np.mean(np.array(image_auc_list),  axis = 0)
    mean_pixel_auc  = np.mean(np.array(pixel_auc_list),  axis = 0)
    mean_img_aupr   = np.mean(np.array(image_aupr_list), axis = 0)
    mean_pixel_aupr = np.mean(np.array(pixel_aupr_list), axis = 0)
    mean_iou_score  = np.mean(np.array(iou_score_list),  axis = 0)

    localtime = time.asctime( time.localtime(time.time()) )
    print_log('\n finish all object, time={}'.format(localtime), log)
    print_log('Image-level AUC/AUPR: {:.4f} {:.4f}, Pixel-level AUC/AUPR: {:.4f} {:.4f} {:.4f}'.format(mean_img_auc, mean_img_aupr, mean_pixel_auc, mean_pixel_aupr, mean_iou_score), log)


if __name__ == '__main__':
    main()
