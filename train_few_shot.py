'''
动机说明：
1. 利用FewShot=2,进行数据的扩充可以得到相对显著的性能提升，但是和使用所有的数据相比还是存一定的Gap。这个说明：
   a. 数据的扩充是有必要的
   b. 数据扩充的方式还有待改进
   c. 是否可以利用元学习来学习这种扩充方式呢

'''


import os
import sys
import time
import random
import argparse
import numpy as np
import torch
from utils.util import time_file_str, time_string, convert_secs2time, AverageMeter, print_log
from torchvision.models import wide_resnet50_2, resnet18

device   = torch.device('cuda')

def usr_parser():
    parser = argparse.ArgumentParser(description='Registration based Few-Shot Anomaly Detection')
    parser.add_argument('--obj',        type=str, default='bottle')
    parser.add_argument('--gpu',        type=int, default=0)
    parser.add_argument('--data_type',  type=str, default='mvtec')
    parser.add_argument('--data_path',  type=str, default='./MVTec/MVTec_AD')
    parser.add_argument('--epochs',     type=int, default=15, help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size',   type=int, default=224)
    parser.add_argument('--lr',         type=float, default=0.0001, help='learning rate of others in SGD')
    parser.add_argument('--momentum',   type=float, default=0.9, help='momentum of SGD')
    parser.add_argument('--seed',       type=int, default=668, help='manual seed')
    parser.add_argument('--shot',       type=int, default=2, help='shot count')
    parser.add_argument('--inferences', type=int, default=10, help='number of rounds per inference')
    args = parser.parse_args()
    args.input_channel = 3

    return args

def load_model():
    # feature extraction model
    backbone_model = resnet18(pretrained=True, progress=True)
    backbone_model = backbone_model.to(device)

    # image transform model

    return backbone_model




def main(args):
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed_all(args.seed)
        

    args.prefix = time_file_str()
    args.save_dir = './logs_mvtec/'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.save_model_dir = './logs_mvtec/' + args.stn_mode + '/' + str(args.shot) + '/' + args.obj + '/'
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    log = open(os.path.join(args.save_dir, 'log_{}_{}.txt'.format(str(args.shot),args.obj)), 'a+')
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)

    # load model and dataset
    FE_Model, ENC_Model = load_model()

    optimizers = torch.optim.SGD(ENC_Model.parameters(), lr=args.lr, momentum=args.momentum)
    models     = [FE_Model, ENC_Model]

    print('Loading Datasets')
    kwargs = {'num_workers': 8, 'pin_memory': True}
    train_dataset = FSAD_Dataset_train(args.data_path, class_name=args.obj, is_train=True, resize=args.img_size, shot=args.shot, batch=args.batch_size)
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
    test_dataset  = FSAD_Dataset_test(args.data_path, class_name=args.obj, is_train=False, resize=args.img_size, shot=args.shot)
    test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    # start training
    save_name = os.path.join(args.save_model_dir, '{}_{}_{}_model.pt'.format(args.obj, args.shot, args.stn_mode))
    start_time = time.time()
    epoch_time = AverageMeter()
    img_roc_auc_old = 0.0
    per_pixel_rocauc_old = 0.0
    print('Loading Fixed Support Set')
    fixed_fewshot_list = torch.load(f'./support_set/{args.obj}/{args.shot}_{args.inferences}.pt')
    print_log((f'---------{args.stn_mode}--------'), log)

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizers, init_lrs, epoch, args)
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' {:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, args.epochs, time_string(), need_time), log)

        if epoch <= args.epochs:
            image_auc_list = []
            pixel_auc_list = []
            for inference_round in tqdm(range(args.inferences)):
            #for inference_round in range(1):
                scores_list, test_imgs, gt_list, gt_mask_list = test(models, inference_round, fixed_fewshot_list,
                                                                     test_loader, **kwargs)
                scores = np.asarray(scores_list)
                # Normalization
                max_anomaly_score = scores.max()
                min_anomaly_score = scores.min()
                scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

                # calculate image-level ROC AUC score
                img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
                gt_list = np.asarray(gt_list)
                img_roc_auc = roc_auc_score(gt_list, img_scores)
                image_auc_list.append(img_roc_auc)

                # calculate per-pixel level ROCAUC
                gt_mask = np.asarray(gt_mask_list)
                gt_mask = (gt_mask > 0.5).astype(np.int_)
                per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
                pixel_auc_list.append(per_pixel_rocauc)

            image_auc_list = np.array(image_auc_list)
            pixel_auc_list = np.array(pixel_auc_list)
            mean_img_auc = np.mean(image_auc_list, axis = 0)
            mean_pixel_auc = np.mean(pixel_auc_list, axis = 0)

            if mean_img_auc + mean_pixel_auc > per_pixel_rocauc_old + img_roc_auc_old:
                state = {'STN': STN.state_dict(), 'ENC': ENC.state_dict(), 'PRED':PRED.state_dict()}
                print('Best Model Saving model...')
                print_log('Best Model Saving model to {}'.format(save_name), log)
                torch.save(state, save_name)
                per_pixel_rocauc_old = mean_pixel_auc
                img_roc_auc_old = mean_img_auc
            print('Img-level AUC:',   img_roc_auc_old)
            print('Pixel-level AUC:', per_pixel_rocauc_old)

            print_log(('Test Epoch(img, pixel): {} ({:.6f}, {:.6f}) best: ({:.3f}, {:.3f})'
            .format(epoch-1, mean_img_auc, mean_pixel_auc, img_roc_auc_old, per_pixel_rocauc_old)), log)

        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        train(models, epoch, train_loader, optimizers, log)
        train_dataset.shuffle_dataset()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
        
    log.close()