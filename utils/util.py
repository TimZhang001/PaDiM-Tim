import warnings
import torch
import matplotlib
import os
import cv2
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics._ranking import _binary_clf_curve
from sklearn.exceptions   import UndefinedMetricWarning
from skimage.segmentation import mark_boundaries


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


def get_aupr_curve(gt_list, scores):
    """
    by dlluo
    :param gt_list: ground truth
    :param scores: predicted anomaly score map
    :return: precision, recall, thresholds
    """
    assert gt_list.shape == scores.shape
    precision, recall, thresholds = precision_recall_curve(gt_list.flatten(), scores.flatten())
    a       = 2 * precision * recall
    b       = precision + recall
    f1      = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    best_th = thresholds[np.argmax(f1)]
    aupr    = auc(recall, precision)
    return aupr, best_th, precision, recall


def iou_curve(gt, scores, pos_label=None, sample_weight=None,
              drop_intermediate=True):
    '''
    Compute per region overlap-false positive rate pairs for different probability thresholds
    exceeded from sklearn
    ---------
    :param gt: ground truth of clf, 1-d numpy array binary mask [0,1]
    :param scores: clf scores
    :return: fpr, pro, thresholds
    '''

    assert gt.shape == scores.shape

    fps, tps, thresholds = _binary_clf_curve(
        gt, scores, pos_label=pos_label, sample_weight=sample_weight)

    # Attempt to drop thresholds corresponding to points in between and
    # collinear with other points. These are always suboptimal and do not
    # appear on a plotted ROC curve (and thus do not affect the AUC).
    # Here np.diff(_, 2) is used as a "second derivative" to tell if there
    # is a corner at the point. Both fps and tps must be tested to handle
    # thresholds with multiple data points (which are combined in
    # _binary_clf_curve). This keeps all cases where the point should be kept,
    # but does not drop more complicated cases like fps = [1, 3, 7],
    # tps = [1, 2, 4]; there is no harm in keeping too many thresholds.
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(fps, 2),
                                                    np.diff(tps, 2)),
                                      True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    if tps.size == 0 or fps[0] != 0:
        # Add an extra threshold position if necessary
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless",
                      UndefinedMetricWarning)
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "true positive value should be meaningless",
                      UndefinedMetricWarning)
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]


    # tns = fps[-1] - fps
    fns = tps[-1] - tps
    # compute per-region overlap from confusion-matrix
    iou = tps / (tps + fps + fns)

    # PRO curve up to an average false-positive rate of 30%
    index_range = np.where(fpr <= 0.3)
    index_range = index_range[0]
    fpr = fpr[index_range]
    iou = iou[index_range]
    thresholds = thresholds[index_range]

    return fpr, iou, thresholds

def print_log(print_string, log):
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()

def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x

def visualize_results(test_img, scores, img_scores, gts, query_features, threshold, cls_threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    vmax = vmax * 0.5 + vmin * 0.5
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        #kernel = morphology.disk(4)
        #mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 7, figsize=(15, 3), gridspec_kw={'width_ratios': [4, 4, 4, 4, 4, 4, 3]})

        fig_img.subplots_adjust(wspace=0.05, hspace=0)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Input image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax_img[2].imshow(heat_map, cmap='jet', norm=norm, interpolation='none')
        ax_img[2].imshow(vis_img, cmap='gray', alpha=0.7, interpolation='none')
        ax_img[2].title.set_text('Segmentation')
        
        for j in range(3, 6):
            featuremap = query_features[j-3][i]
            ax_img[j].imshow(featuremap, cmap='jet')
            ax_img[j].title.set_text('featuremap {}'.format(j-3))
        

        black_mask = np.zeros((int(mask.shape[0]), int(3 * mask.shape[1] / 4)))
        ax_img[6].imshow(black_mask, cmap='gray')
        ax = plt.gca()
        if img_scores[i] > cls_threshold:
            cls_result = 'nok'
        else:
            cls_result = 'ok'

        ax.text(0.05,
                0.89,
                'Detected anomalies',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.79,
                '------------------------',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.72,
                'Results',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.67,
                '------------------------',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.59,
                '\'{}\''.format(cls_result),
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='r',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.47,
                'Anomaly scores: {:.2f}'.format(img_scores[i]),
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.37,
                '------------------------',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.30,
                'Thresholds',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.25,
                '------------------------',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.17,
                'Classification: {:.2f}'.format(cls_threshold),
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.07,
                'Segementation: {:.2f}'.format(threshold),
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax_img[6].title.set_text('Classification')
        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=300, bbox_inches='tight')
        plt.close()


def visualize_results_list(test_img, scores, img_scores, gts, query_features, threshold, cls_threshold, save_dir, class_name, scores_list):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    vmax = vmax * 0.5 + vmin * 0.5
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        #kernel = morphology.disk(4)
        #mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        #fig_img, ax_img = plt.subplots(1, len(scores_list), figsize=(15, 3), gridspec_kw={'width_ratios': [4, 4, 4, 4, 4, 4, 3]})
        fig_img, ax_img = plt.subplots(1, 7+len(scores_list), figsize=(15+3*len(scores_list), 3))

        fig_img.subplots_adjust(wspace=0.05, hspace=0)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Input image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax_img[2].imshow(heat_map, cmap='jet', norm=norm, interpolation='none')
        ax_img[2].imshow(vis_img, cmap='gray', alpha=0.7, interpolation='none')
        ax_img[2].title.set_text('Segmentation')
        
        for j in range(3, 6):
            featuremap = query_features[j-3][i]
            ax_img[j].imshow(featuremap, cmap='jet')
            ax_img[j].title.set_text('featuremap {}'.format(j-3))
        
        for j in range(6, 6+len(scores_list)):
            score = scores_list[j-6][i]
            ax_img[j].imshow(score, cmap='jet')
            ax_img[j].title.set_text('score {}'.format(j-6))

        black_mask = np.zeros((int(mask.shape[0]), int(3 * mask.shape[1] / 4)))
        ax_img[6+len(scores_list)].imshow(black_mask, cmap='gray')
        ax = plt.gca()
        if img_scores[i] > cls_threshold:
            cls_result = 'nok'
        else:
            cls_result = 'ok'

        ax.text(0.05,
                0.89,
                'Detected anomalies',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.79,
                '------------------------',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.72,
                'Results',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.67,
                '------------------------',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.59,
                '\'{}\''.format(cls_result),
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='r',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.47,
                'Anomaly scores: {:.2f}'.format(img_scores[i]),
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.37,
                '------------------------',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.30,
                'Thresholds',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.25,
                '------------------------',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.17,
                'Classification: {:.2f}'.format(cls_threshold),
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.07,
                'Segementation: {:.2f}'.format(threshold),
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax_img[6+len(scores_list)].title.set_text('Classification')
        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=500, bbox_inches='tight')
        plt.close()

def visualize_featue_map(features_result, save_dir, class_name, good_num):
    
    fig_img, ax_img = plt.subplots(1, 5, figsize=(15, 3), gridspec_kw={'width_ratios': [4, 4, 4, 4, 4]})

    fig_img.subplots_adjust(wspace=0.05, hspace=0)
    for ax_i in ax_img:
        ax_i.axes.xaxis.set_visible(False)
        ax_i.axes.yaxis.set_visible(False)

    ax_img[0].imshow(features_result[0], cmap='jet')
    ax_img[0].title.set_text('feature_layer1')
    ax_img[1].imshow(features_result[1], cmap='jet')
    ax_img[1].title.set_text('feature_layer2')
    ax_img[2].imshow(features_result[2], cmap='jet')
    ax_img[2].title.set_text('feature_layer3')
    ax_img[3].imshow(features_result[3], cmap='jet')
    ax_img[3].title.set_text('feature_layer')

    ax_img[4].imshow(features_result[4])
    ax_img[4].title.set_text('image_mean')

    fig_img.savefig(os.path.join(save_dir, class_name + '_features_{}'.format(str(good_num))), dpi=300, bbox_inches='tight')
    plt.close()    


def visualize_CovMatrix(meanMatix, covMatix, save_dir, class_name):
    
    for i in range(0, len(meanMatix)):
        meanMatix1 = np.squeeze(meanMatix[i, :, :])
        imagesize  = np.int(np.sqrt(meanMatix1.shape[1]))
        meanMatix1 = np.reshape(meanMatix1, (meanMatix1.shape[0], imagesize, imagesize))
        meanMatix1 = np.mean(meanMatix1, axis=0)

        covMatix1  = np.squeeze(covMatix[i, :, :])
        covMatix1  = np.reshape(covMatix1, (covMatix1.shape[0], covMatix1.shape[1], imagesize, imagesize))
        covMatix1  = np.mean(covMatix1, axis=0)
        covMatix1  = np.mean(covMatix1, axis=0)
        
        fig_img, ax_img = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={'width_ratios': [4, 4]})

        fig_img.subplots_adjust(wspace=0.05, hspace=0)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(meanMatix1, cmap='jet')
        ax_img[0].title.set_text('mean')
        fig_img.colorbar(ax_img[0].images[0], ax=ax_img[0])
        
        max_val = np.percentile(covMatix1, 99)
        ax_img[1].imshow(covMatix1, cmap='jet', vmin=0, vmax=max_val)
        ax_img[1].title.set_text('cov')
        fig_img.colorbar(ax_img[1].images[0], ax=ax_img[1])

        fig_img.savefig(os.path.join(save_dir, class_name + '_covMatix_{}'.format(i)), dpi=300, bbox_inches='tight')
        plt.close()    

'''
1. 将提取的features进行平均，得到不同layer下的平均feature map
'''
def get_layers_feature_map(test_outputs):

    # 得到layer1, layer2, layer3的输出
    layer_feature1 = test_outputs['layer1']
    layer_feature2 = test_outputs['layer2']
    layer_feature3 = test_outputs['layer3']

    output_layers1 = []
    output_layers2 = []
    output_layers3 = []

    for index in range(len(layer_feature1)):
        output_layers = torch.mean(layer_feature1[index], dim=1, keepdim=False)
        output_layers = output_layers.squeeze().cpu().numpy()
        output_layers1.append(output_layers)

    for index in range(len(layer_feature2)):
        output_layers = torch.mean(layer_feature2[index], dim=1, keepdim=False)
        output_layers = output_layers.squeeze().cpu().numpy()
        output_layers2.append(output_layers)
    
    for index in range(len(layer_feature3)):
        output_layers = torch.mean(layer_feature3[index], dim=1, keepdim=False)
        output_layers = output_layers.squeeze().cpu().numpy()
        output_layers3.append(output_layers)
    
    return output_layers1, output_layers2, output_layers3


def get_save_path(obj, arch, add_str=''):
    # save image path
    image_dir = f'./vis_result/{arch}/{obj}/'

    if add_str != '':
        image_dir = image_dir + str(add_str) + '/'

    if not os.path.exists(image_dir):
        os.makedirs(image_dir, exist_ok=True)

    return image_dir



def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def time_file_str():
    ISOTIMEFORMAT = '%Y-%m-%d'
    string = '{}'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string + '-{}'.format(random.randint(1, 10000))


def print_log(print_string, log):
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()
