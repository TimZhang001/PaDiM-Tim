'''
说    明：MvTec数据集中，PaDiM的训练程序
作    者：Zhangshengsen
创建时间：2022-10-18
'''

import argparse
import torch
import random
import datasets.mvtec as mvtec
import numpy as np
import os
import pickle

from torchvision.models import wide_resnet50_2, resnet18
from random             import sample
from torch.utils.data   import DataLoader
from collections        import OrderedDict
from tqdm               import tqdm
from utils.util         import embedding_concat, denormalization, get_save_path, visualize_featue_results

# device setup
device   = torch.device('cuda')

def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--data_path',  type=str, default='./MVTec/MVTec_AD')
    parser.add_argument('--model_path', type=str, default='./save_checkpoints')
    parser.add_argument('--arch',       type=str, choices=['resnet18', 'wide_resnet50_2'], default='resnet18')
    parser.add_argument('--good_num',   type=int, default=10000)
    return parser.parse_args()

def prepare_models(arch):
    # load model
    if arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d   = 448
        d     = 100
    elif arch == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d   = 1792
        d     = 550

    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
    torch.cuda.manual_seed_all(1024)
        
    idx = torch.tensor(sample(range(0, t_d), d))

    return model, idx

def prepare_data(args, class_name):
    train_dataset    = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=True, good_num=args.good_num)
    train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
    train_outputs    = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    return train_dataloader, train_outputs

def prepare_save_path(args, class_name):
    train_model_path = os.path.join(args.model_path, '%s' % args.arch)
    os.makedirs(train_model_path, exist_ok=True)
    train_model_path = os.path.join(train_model_path, '%s_train.pkl' % class_name)
    return train_model_path

def get_mean_features(train_outputs, embedding_vectors, train_images):
    train_outputs_mean = {k1: v1.mean(0) for k1, v1 in train_outputs.items()}
    train_outputs_mean = {k1: v1.mean(0) for k1, v1 in train_outputs_mean.items()}
    feature_layer1     = train_outputs_mean['layer1'].numpy()
    feature_layer2     = train_outputs_mean['layer2'].numpy()
    feature_layer3     = train_outputs_mean['layer3'].numpy()
    feature_layer      = torch.mean(torch.mean(embedding_vectors, dim=0), dim=0).cpu()
    feature_layer      = feature_layer.numpy()

    # get image_mean data
    train_images_mean = np.vstack(train_images)
    train_images_mean = np.mean(train_images_mean, axis=(0))
    train_images_mean = denormalization(train_images_mean)

    return feature_layer1, feature_layer2, feature_layer3, feature_layer, train_images_mean

def main():

    args  = parse_args()
    
    #for class_name in mvtec.CLASS_NAMES:
    for class_name in ['grid']:
        # prepare model
        model, idx = prepare_models(args.arch)
        
        # set model's intermediate outputs
        outputs = []

        def hook(module, input, output):
            outputs.append(output)

        model.layer1[-1].register_forward_hook(hook)
        model.layer2[-1].register_forward_hook(hook)
        model.layer3[-1].register_forward_hook(hook)

        # prepare data
        train_dataloader, train_outputs = prepare_data(args, class_name)
        
        # extract train set features
        train_model_path = prepare_save_path(args, class_name)

        # train
        train_images = []
        for (x, _, _,_) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
            
            # get all image datas
            train_images.append(x.numpy())

            # model prediction
            with torch.no_grad():
                _ = model(x.to(device))
            
            # get intermediate layer outputs
            for k, v in zip(train_outputs.keys(), outputs):
                train_outputs[k].append(v.cpu().detach())
            
            # initialize hook outputs
            outputs = []
        for k, v in train_outputs.items():
            train_outputs[k] = torch.cat(v, 0)

        # Embedding concat
        embedding_vectors = train_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        
        # get image_mean data
        feature_show = get_mean_features(train_outputs, embedding_vectors, train_images)
        image_dir    = get_save_path(class_name, args.arch, "feature_show")
        visualize_featue_results(feature_show, image_dir, class_name, args.good_num)

        # calculate multivariate Gaussian distribution
        B, C, H, W        = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W)
        mean              = torch.mean(embedding_vectors, dim=0).numpy()
        cov               = torch.zeros(C, C, H * W).numpy()
        I                 = np.identity(C)
        for i in range(H * W):
            # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
            cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
                
        # save learned distribution
        train_outputs = [mean, cov]
        with open(train_model_path, 'wb') as f:
            pickle.dump(train_outputs, f)


if __name__ == '__main__':
    main()
