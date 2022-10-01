# Copyright (c) Megvii Inc. All rights reserved.
import torch
import torch.nn as nn
from mmdet.models.backbones.resnet import BasicBlock

from ops.voxel_pooling import voxel_pooling

from .base_lss_fpn import ASPP, BaseLSSFPN, Mlp, SELayer

__all__ = ['FusionLSSFPN']


class DepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels,
                 depth_channels):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.mlp = Mlp(1, mid_channels, mid_channels)
        self.se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_gt_conv = nn.Sequential(
            nn.Conv2d(1, mid_channels, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1),
        )
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        )
        self.aspp = ASPP(mid_channels, mid_channels)
        self.depth_pred = nn.Conv2d(mid_channels,
                                    depth_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)

    def forward(self, x, mats_dict, lidar_depth, scale_depth_factor=1000.0):
        x = self.reduce_conv(x)
        context = self.context_conv(x)
        inv_intrinsics = torch.inverse(mats_dict['intrin_mats'][:, 0:1, ...])
        pixel_size = torch.norm(torch.stack(
            [inv_intrinsics[..., 0, 0], inv_intrinsics[..., 1, 1]], dim=-1),
                                dim=-1).reshape(-1, 1)
        aug_scale = torch.sqrt(mats_dict['ida_mats'][:, 0, :, 0, 0]**2 +
                               mats_dict['ida_mats'][:, 0, :, 0,
                                                     0]**2).reshape(-1, 1)
        scaled_pixel_size = pixel_size * scale_depth_factor / aug_scale
        x_se = self.mlp(scaled_pixel_size)[..., None, None]
        x = self.se(x, x_se)
        depth = self.depth_gt_conv(lidar_depth)
        depth = self.depth_conv(x + depth)
        depth = self.aspp(depth)
        depth = self.depth_pred(depth)
        return torch.cat([depth, context], dim=1)


class FusionLSSFPN(BaseLSSFPN):
    def _configure_depth_net(self, depth_net_conf):
        return DepthNet(
            depth_net_conf['in_channels'],
            depth_net_conf['mid_channels'],
            self.output_channels,
            self.depth_channels,
        )

    def _forward_depth_net(self, feat, mats_dict, lidar_depth):
        return self.depth_net(feat, mats_dict, lidar_depth)

    def _forward_single_sweep(self,
                              sweep_index,
                              sweep_imgs,
                              mats_dict,
                              lidar_depth,
                              is_return_depth=False):
        """Forward function for single sweep.

        Args:
            sweep_index (int): Index of sweeps.
            sweep_imgs (Tensor): Input images.
            mats_dict (dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            is_return_depth (bool, optional): Whether to return depth.
                Default: False.

        Returns:
            Tensor: BEV feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape
        img_feats = self.get_cam_feats(sweep_imgs)
        lidar_depth = lidar_depth.reshape(batch_size * num_cams,
                                          *lidar_depth.shape[2:])
        source_features = img_feats[:, 0, ...]
        depth_feature = self._forward_depth_net(
            source_features.reshape(batch_size * num_cams,
                                    source_features.shape[2],
                                    source_features.shape[3],
                                    source_features.shape[4]), mats_dict,
            lidar_depth)
        depth = depth_feature[:, :self.depth_channels].softmax(1)
        img_feat_with_depth = depth.unsqueeze(
            1) * depth_feature[:, self.depth_channels:(
                self.depth_channels + self.output_channels)].unsqueeze(2)

        img_feat_with_depth = self._forward_voxel_net(img_feat_with_depth)

        img_feat_with_depth = img_feat_with_depth.reshape(
            batch_size,
            num_cams,
            img_feat_with_depth.shape[1],
            img_feat_with_depth.shape[2],
            img_feat_with_depth.shape[3],
            img_feat_with_depth.shape[4],
        )
        geom_xyz = self.get_geometry(
            mats_dict['sensor2ego_mats'][:, sweep_index, ...],
            mats_dict['intrin_mats'][:, sweep_index, ...],
            mats_dict['ida_mats'][:, sweep_index, ...],
            mats_dict.get('bda_mat', None),
        )
        img_feat_with_depth = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2)
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) /
                    self.voxel_size).int()
        feature_map = voxel_pooling(geom_xyz, img_feat_with_depth.contiguous(),
                                    self.voxel_num.cuda())
        if is_return_depth:
            return feature_map.contiguous(), depth
        return feature_map.contiguous()

    def forward(self,
                sweep_imgs,
                mats_dict,
                lidar_depth,
                timestamps=None,
                is_return_depth=False):
        """Forward function.

        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps,
                num_cameras, 3, H, W).
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps(Tensor): Timestamp for all images with the shape of(B,
                num_sweeps, num_cameras).

        Return:
            Tensor: bev feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape
        lidar_depth = self.get_downsampled_lidar_depth(lidar_depth)
        key_frame_res = self._forward_single_sweep(
            0,
            sweep_imgs[:, 0:1, ...],
            mats_dict,
            lidar_depth[:, 0, ...],
            is_return_depth=is_return_depth)
        if num_sweeps == 1:
            return key_frame_res

        key_frame_feature = key_frame_res[
            0] if is_return_depth else key_frame_res

        ret_feature_list = [key_frame_feature]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep(
                    sweep_index,
                    sweep_imgs[:, sweep_index:sweep_index + 1, ...],
                    mats_dict,
                    lidar_depth[:, sweep_index, ...],
                    is_return_depth=False)
                ret_feature_list.append(feature_map)

        if is_return_depth:
            return torch.cat(ret_feature_list, 1), key_frame_res[1]
        else:
            return torch.cat(ret_feature_list, 1)

    def get_downsampled_lidar_depth(self, lidar_depth):
        batch_size, num_sweeps, num_cams, height, width = lidar_depth.shape
        lidar_depth = lidar_depth.view(
            batch_size * num_sweeps * num_cams,
            height // self.downsample_factor,
            self.downsample_factor,
            width // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        lidar_depth = lidar_depth.permute(0, 1, 3, 5, 2, 4).contiguous()
        lidar_depth = lidar_depth.view(
            -1, self.downsample_factor * self.downsample_factor)
        gt_depths_tmp = torch.where(lidar_depth == 0.0, lidar_depth.max(),
                                    lidar_depth)
        lidar_depth = torch.min(gt_depths_tmp, dim=-1).values
        lidar_depth = lidar_depth.view(batch_size, num_sweeps, num_cams, 1,
                                       height // self.downsample_factor,
                                       width // self.downsample_factor)
        lidar_depth = lidar_depth / self.d_bound[1]
        return lidar_depth
