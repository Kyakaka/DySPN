import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
from PIL import Image
import cv2

cm = plt.get_cmap('plasma')
cm2 = plt.get_cmap('jet')

def summary(sample, output, path_output, setting):
    with torch.no_grad():
        if setting.spn_enable == True:
            _, _, H, W = output['pred_init'].shape
            feat_init = output['pred_init']
            list_feat = output['list_feat']
            offset = output['offset']
            aff = output['aff']
            rgb = sample['rgb'].detach()
            dep = sample['dep'].detach()
            dep = torch.max_pool2d(dep,3,1,1)
            pred = output['pred'].detach()
            gt = sample['gt'].detach()
            pred = torch.clamp(pred, min=0)

            rgb = rgb[0, :, :, :].data.cpu().numpy()
            dep = dep[0, 0, :, :].data.cpu().numpy()
            pred = pred[0, 0, :, :].data.cpu().numpy()
            gt = gt[0, 0, :, :].data.cpu().numpy()

            rgb = 255.0 * np.transpose(rgb, (1, 2, 0))
            dep_max = dep.max()
            dep = dep / dep_max
            pred = pred / dep_max
            pred_gray = pred
            gt = gt / dep_max

            rgb = np.clip(rgb, 0, 256).astype('uint8')
            dep = (255.0 * cm(dep)).astype('uint8')
            pred = (255.0 * cm(pred)).astype('uint8')
            pred_gray = (255.0 * pred_gray).astype('uint8')
            gt = (255.0 * cm(gt)).astype('uint8')

            rgb = Image.fromarray(rgb, 'RGB')
            dep = Image.fromarray(dep[:, :, :3], 'RGB')
            pred = Image.fromarray(pred[:, :, :3], 'RGB')
            pred_gray = Image.fromarray(pred_gray)
            gt = Image.fromarray(gt[:, :, :3], 'RGB')

            feat_init = feat_init[0, 0, :, :].data.cpu().numpy()
            feat_init = feat_init / dep_max
            feat_init = (255.0 * cm(feat_init)).astype('uint8')
            feat_init = Image.fromarray(feat_init[:, :, :3], 'RGB')

            for k in range(0, len(list_feat)):
                feat_inter = list_feat[k]
                feat_inter = feat_inter[0, 0, :, :].data.cpu().numpy()
                feat_inter = feat_inter / dep_max
                feat_inter = (255.0 * cm(feat_inter)).astype('uint8')
                feat_inter = Image.fromarray(feat_inter[:, :, :3], 'RGB')

                list_feat[k] = feat_inter

            path_save_rgb = '{}/01_rgb.png'.format(path_output)
            path_save_dep = '{}/02_dep.png'.format(path_output)
            path_save_init = '{}/03_pred_init.png'.format(path_output)
            path_save_pred = '{}/05_pred_final.png'.format(path_output)
            path_save_pred_gray = '{}/05_pred_final_gray.png'.format(path_output)
            path_save_gt = '{}/06_gt.png'.format(path_output)

            rgb.save(path_save_rgb)
            dep.save(path_save_dep)
            pred.save(path_save_pred)
            pred_gray.save(path_save_pred_gray)
            feat_init.save(path_save_init)
            gt.save(path_save_gt)
            for k in range(0, len(list_feat)):
                path_save_inter = '{}/04_pred_prop_{:02d}.png'.format(path_output, k)
                list_feat[k].save(path_save_inter)

            if setting.spn_module == "deform_dyspn":
                offset2 = output['offset2']
                x = np.array([])
                y = np.array([])
                offset_t = offset.cpu().numpy()
                for k in range(3):
                    for j in range(3):
                        if k != 1 & j != 1:
                            x = np.hstack((x, (offset_t[0, 2 * (3 * k + j), :, :] + j - 1).reshape((-1))))
                            y = np.hstack((y, (offset_t[0, 2 * (3 * k + j) + 1, :, :] + k - 1).reshape((-1))))
                x2 = np.array([])
                y2 = np.array([])
                offset_t2 = offset2.cpu().numpy()
                for k in range(3):
                    for j in range(3):
                        if k != 1 & j != 1:
                            x2 = np.hstack((x2, (offset_t2[0, 2 * (3 * k + j), :, :] + j - 1).reshape((-1))))
                            y2 = np.hstack((y2, (offset_t2[0, 2 * (3 * k + j) + 1, :, :] + k - 1).reshape((-1))))
                min_x = np.min((np.min(x), np.min(x2)) )
                max_x = np.max((np.max(x), np.max(x2)) )
                min_y = np.min((np.min(y), np.min(y2)) )
                max_y = np.max((np.max(y), np.max(y2)) )
                h = plt.hist2d(x, y, bins=300, cmap='jet', norm=colors.LogNorm(),
                               range=[[min_x, max_x], [min_y, max_y]]
                               # ,weights=list_w[i]
                               )
                cbar = plt.colorbar(h[3])
                density = cbar.get_ticks()
                plt.clim(density.min(), density.max())
                plt.savefig('{}/07_offset.png'.format(path_output))
                plt.clf()
                plt.cla()

                h = plt.hist2d(x2, y2, bins=300, cmap='jet', norm=colors.LogNorm(),
                               range=[[min_x, max_x], [min_y, max_y]]
                               # ,weights=list_w[i]
                               )
                cbar = plt.colorbar(h[3])
                density = cbar.get_ticks()
                plt.clim(density.min(), density.max())
                plt.savefig('{}/07_offset2.png'.format(path_output))
                plt.clf()
                plt.cla()
                dynamic = output['dynamic'].cpu().numpy()[0,...]
                dynamic_list = np.array_split(dynamic, 24, axis=0)
                image_list = []
                for i in range(6):
                    image = np.concatenate(dynamic_list[i*4:(i+1)*4][::-1], axis=1)
                    image_list.append(image)
                image = np.concatenate(image_list,axis=2)[0,:,:]
                image = (255.0 * cm2(image)).astype('uint8')
                path_save_dynamic = '{}/04_dynamic.png'.format(path_output)
                image = Image.fromarray(image[:, :, :3], 'RGB')
                image.save(path_save_dynamic)
            elif setting.spn_module=="dyspn":
                list_x = []
                list_y = []
                # list_w = []
                for i in range(len(offset)):
                    x = np.array([])
                    y = np.array([])
                    w = np.array([])
                    offset_t = offset[i].cpu().numpy()
                    # aff_t = aff[i].cpu().numpy()
                    # dysamplev6
                    # x = np.hstack((x, (offset_t[:,0,:,:]).reshape((-1))))
                    # y = np.hstack((y, (offset_t[:,1,:,:]).reshape((-1))))
                    # dysamplev7
                    # x = np.hstack((x, (offset_t[..., 0]).reshape((-1))))
                    # y = np.hstack((y, (offset_t[..., 1]).reshape((-1))))
                    # dysamplev8
                    x = np.hstack((x, (offset_t[0, :, :, :, 0]).reshape((-1))))
                    y = np.hstack((y, (offset_t[0, :, :, :, 1]).reshape((-1))))
                    x = np.hstack((x, (offset_t[..., 0]).reshape((-1)))) * W / 2
                    y = np.hstack((y, (offset_t[..., 1]).reshape((-1)))) * H / 2
                    # for k in range(3):
                    #     for j in range(3):
                    #         x = np.hstack((x, (offset_t[0, 2 * (3 * k + j), :, :] + j - 1).reshape((-1))))
                    #         y = np.hstack((y, (offset_t[0, 2 * (3 * k + j) + 1, :, :] + k - 1).reshape((-1))))
                    # w = np.hstack((w, (aff_t[0, 3 * k + j, :, :]).reshape((-1))))
                    list_x.append(x)
                    list_y.append(y)
                    # list_w.append(w)
                if len(offset) > 0:
                    min_x = np.min(np.concatenate(list_x))
                    max_x = np.max(np.concatenate(list_x))
                    min_y = np.min(np.concatenate(list_y))
                    max_y = np.max(np.concatenate(list_y))
                for i in range(len(offset)):
                    h = plt.hist2d(list_x[i], list_y[i], bins=300, cmap='jet', norm=colors.LogNorm(),
                                   range=[[min_x, max_x], [min_y, max_y]]
                                   # ,weights=list_w[i]
                                   )
                    cbar = plt.colorbar(h[3])
                    density = cbar.get_ticks()
                    plt.clim(density.min(), density.max())
                    plt.savefig('{}/07_offset_{:02d}.png'.format(path_output, i))
                    plt.clf()
                    plt.cla()
                list_x = []
                list_y = []
                # list_w = []
                ref_y = torch.linspace(-H + 1, H - 1, H, device=torch.device("cpu"))
                ref_x = torch.linspace(-W + 1, W - 1, W, device=torch.device("cpu"))
                for i in range(len(offset)):
                    x = np.array([])
                    y = np.array([])
                    w = np.array([])

                    offset_t = offset[i].cpu()
                    offset_t[..., 0] = (offset_t[..., 0] * W - ref_x.view(1, 1, 1, W)) / 2
                    offset_t[..., 1] = (offset_t[..., 1] * H - ref_y.view(1, 1, H, 1)) / 2

                    # aff_t = aff[i].cpu().numpy()
                    # dysamplev7
                    # x = np.hstack((x, (offset_t[0, :, :, 0]).reshape((-1))))
                    # y = np.hstack((y, (offset_t[0, :, :, 1] ).reshape((-1))))
                    # dysamplev8
                    # x = np.hstack((x, (offset_t[0, :, :, :, 0]).reshape((-1))))
                    # y = np.hstack((y, (offset_t[0, :, :, :, 1]).reshape((-1))))
                    x = np.hstack((x, (offset_t[..., 0].numpy()).reshape((-1))))
                    y = np.hstack((y, (offset_t[..., 1].numpy()).reshape((-1))))
                    # for k in range(3):
                    #     for j in range(3):
                    #         x = np.hstack((x, (offset_t[0, 2 * (3 * k + j), :, :] + j - 1).reshape((-1))))
                    #         y = np.hstack((y, (offset_t[0, 2 * (3 * k + j) + 1, :, :] + k - 1).reshape((-1))))
                    # w = np.hstack((w, (aff_t[0, 3 * k + j, :, :]).reshape((-1))))
                    list_x.append(x)
                    list_y.append(y)
                    # list_w.append(w)
                if len(offset) > 0:
                    min_x = np.min(np.concatenate(list_x))
                    max_x = np.max(np.concatenate(list_x))
                    min_y = np.min(np.concatenate(list_y))
                    max_y = np.max(np.concatenate(list_y))
                for i in range(len(offset)):
                    h = plt.hist2d(list_x[i], list_y[i], bins=300, cmap='jet', norm=colors.LogNorm(),
                                   range=[[min_x, max_x], [min_y, max_y]]
                                   # ,weights=list_w[i]
                                   )
                    cbar = plt.colorbar(h[3])
                    density = cbar.get_ticks()
                    plt.clim(density.min(), density.max())
                    plt.savefig('{}/07_offset2_{:02d}.png'.format(path_output, i))
                    plt.clf()
                    plt.cla()
        else:
            rgb = sample['rgb'].detach()
            dep = sample['dep'].detach()
            dep = torch.max_pool2d(dep,3,1,1)
            pred = output['pred'].detach()
            gt = sample['gt'].detach()
            pred = torch.clamp(pred, min=0)

            rgb = rgb[0, :, :, :].data.cpu().numpy()
            dep = dep[0, 0, :, :].data.cpu().numpy()
            pred = pred[0, 0, :, :].data.cpu().numpy()
            gt = gt[0, 0, :, :].data.cpu().numpy()

            rgb = 255.0 * np.transpose(rgb, (1, 2, 0))
            dep_max = dep.max()
            dep = dep / dep_max
            pred = pred / dep_max
            pred_gray = pred
            gt = gt / dep_max

            rgb = np.clip(rgb, 0, 256).astype('uint8')
            dep = (255.0 * cm(dep)).astype('uint8')
            pred = (255.0 * cm(pred)).astype('uint8')
            pred_gray = (255.0 * pred_gray).astype('uint8')
            gt = (255.0 * cm(gt)).astype('uint8')

            rgb = Image.fromarray(rgb, 'RGB')
            dep = Image.fromarray(dep[:, :, :3], 'RGB')
            pred = Image.fromarray(pred[:, :, :3], 'RGB')
            pred_gray = Image.fromarray(pred_gray)
            gt = Image.fromarray(gt[:, :, :3], 'RGB')



            path_save_rgb = '{}/01_rgb.png'.format(path_output)
            path_save_dep = '{}/02_dep.png'.format(path_output)
            path_save_pred = '{}/05_pred_final.png'.format(path_output)
            path_save_pred_gray = '{}/05_pred_final_gray.png'.format(path_output)
            path_save_gt = '{}/06_gt.png'.format(path_output)

            rgb.save(path_save_rgb)
            dep.save(path_save_dep)
            pred.save(path_save_pred)
            pred_gray.save(path_save_pred_gray)
            gt.save(path_save_gt)


