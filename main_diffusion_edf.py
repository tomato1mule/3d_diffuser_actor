"""Main script for trajectory optimization."""

import io
import os
from pathlib import Path
import random
from typing import Tuple, Optional

import cv2
from matplotlib import pyplot as plt
import numpy as np
import tap
import torch
import torch.distributed as dist
from torch.nn import functional as F

from utils.common_utils import (
    load_instructions, count_parameters, get_gripper_loc_bounds
)

# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False


from tqdm import trange
from main_trajectory import TrainTester

from typing import List
from diffedf_v2.legacy.gnn_data import FeaturedPoints
from diffedf_v2.legacy.train_utils import diffuse_T_target, random_time, LazyLogger
from diffedf_v2.model import DiffusionEDFv1
from voxel.voxel_grid import VoxelGrid
from edf_interface.data import PointCloud, SE3, TargetPoseDemo



IMAGE_SIZE =  128
VOXEL_SIZES = [100] # 100x100x100 voxels
NUM_LATENTS = 512 # PerceiverIO latents
CAMERAS = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
SCENE_BOUNDS = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
BATCH_SIZE = 1
# NUM_DEMOS = 8 # total number of training demonstrations to use while training PerAct
# NUM_TEST = 2 # episodes to evaluate on

RESCALE_FACTOR = 100.
N_XREF = 10
save_checkpoint = True
diffusion_schedules = [[1.0, 0.15], [0.15, 0.03], [0.03, 0.003]]
dtype = torch.float32




class Arguments(tap.Tap):
    cameras: Tuple[str, ...] = ("wrist", "left_shoulder", "right_shoulder")
    image_size: str = "256,256"
    max_episodes_per_task: int = 100
    instructions: Optional[Path] = "instructions.pkl"
    seed: int = 0
    tasks: Tuple[str, ...]
    variations: Tuple[int, ...] = (0,)
    checkpoint: Optional[Path] = None
    accumulate_grad_batches: int = 1
    val_freq: int = 500
    gripper_loc_bounds: Optional[str] = None
    gripper_loc_bounds_buffer: float = 0.04
    eval_only: int = 0

    # Training and validation datasets
    dataset: Path
    valset: Path
    dense_interpolation: int = 0
    interpolation_length: int = 100

    # Logging to base_log_dir/exp_log_dir/run_log_dir
    base_log_dir: Path = Path(__file__).parent / "train_logs"
    exp_log_dir: str = "exp"
    run_log_dir: str = "run"

    # Main training parameters
    num_workers: int = 1
    batch_size: int = 16
    batch_size_val: int = 4
    cache_size: int = 100
    cache_size_val: int = 100
    lr: float = 1e-4
    wd: float = 5e-3  # used only for CALVIN
    train_iters: int = 200_000
    val_iters: int = -1  # -1 means heuristically-defined
    max_episode_length: int = 5  # -1 for no limit

    # Data augmentations
    image_rescale: str = "0.75,1.25"  # (min, max), "1.0,1.0" for no rescaling

    # Model
    backbone: str = "clip"  # one of "resnet", "clip"
    embedding_dim: int = 120
    num_vis_ins_attn_layers: int = 2
    use_instruction: int = 0
    rotation_parametrization: str = 'quat'
    quaternion_format: str = 'wxyz'
    diffusion_timesteps: int = 100
    keypose_only: int = 0
    num_history: int = 0
    relative_action: int = 0
    lang_enhanced: int = 0
    fps_subsampling_factor: int = 5


class DiffusionEDFTrainTester(TrainTester):
    """Train/test a trajectory optimization algorithm."""

    def __init__(self, args):
        """Initialize."""
        super().__init__(args)

        self.grasp_clr = torch.load('grasp_pcd/colors.pt').cuda()
        self.grasp_pts = torch.load('grasp_pcd/points.pt').cuda()
        self.grasp_pts[:,2] -= self.grasp_pts[:,2].max()
        subsample_indices = [i for i in range(0, len(self.grasp_pts), 10)]
        self.grasp_clr = self.grasp_clr[subsample_indices]
        self.grasp_pts = self.grasp_pts[subsample_indices]
        self._voxelizer = VoxelGrid(
            coord_bounds=SCENE_BOUNDS,
            voxel_size=VOXEL_SIZES[0],
            batch_size=BATCH_SIZE,
            device='cuda',
            feature_size=3,
            max_num_coords=np.prod([IMAGE_SIZE, IMAGE_SIZE]) * len(CAMERAS),
        ).cuda()

    def get_model(self):
        """Initialize the model."""
        # Initialize model with arguments
        _model = DiffusionEDFv1.from_yaml('test_model.yaml').cuda()
        print("Model parameters:", count_parameters(_model))

        return _model

    @staticmethod
    def get_criterion():
        return TrajectoryCriterion()

    def train_one_step(self, model, criterion, optimizer, step_id, sample):
        # """Run a single training step."""
        # if step_id % self.args.accumulate_grad_batches == 0:
        #     optimizer.zero_grad()
        #
        # if self.args.keypose_only:
        #     sample["trajectory"] = sample["trajectory"][:, [-1]]
        #     sample["trajectory_mask"] = sample["trajectory_mask"][:, [-1]]
        # else:
        #     sample["trajectory"] = sample["trajectory"][:, 1:]
        #     sample["trajectory_mask"] = sample["trajectory_mask"][:, 1:]
        # print(sample['trajectory'].shape)
        # print(sample['trajectory_mask'].shape)
        #
        # # Forward pass
        # curr_gripper = (
        #     sample["curr_gripper"] if self.args.num_history < 1
        #     else sample["curr_gripper_history"][:, -self.args.num_history:]
        # )
        # print(curr_gripper.shape)
        # out = model(
        #     sample["trajectory"],
        #     sample["trajectory_mask"],
        #     sample["rgbs"],
        #     sample["pcds"],
        #     sample["instr"],
        #     curr_gripper
        # )
        # print(out)
        # print()
        # # exit(0)
        #
        # # Backward pass
        # loss = criterion.compute_loss(out)
        # loss.backward()
        #
        # # Update
        # if step_id % self.args.accumulate_grad_batches == self.args.accumulate_grad_batches - 1:
        #     optimizer.step()
        #
        # # Log
        # if dist.get_rank() == 0 and (step_id + 1) % self.args.val_freq == 0:
        #     self.writer.add_scalar("lr", self.args.lr, step_id)
        #     self.writer.add_scalar("train-loss/noise_mse", loss, step_id)

        # ###############################################################################################################
        # # Visualize
        # for k in sample:
        #     if type(sample[k]) is list:
        #         print(k, sample[k])
        #     else:
        #         print(k, sample[k].dtype, sample[k].shape, sample[k].min(), sample[k].max())
        # print(sample['index'])
        # #trajectory torch.float32 torch.Size([36, 2, 8])
        # #trajectory_mask torch.bool torch.Size([36, 2])
        # #rgbs torch.float32 torch.Size([36, 4, 3, 256, 256])
        # #pcds torch.float32 torch.Size([36, 4, 3, 256, 256])
        # #curr_gripper torch.float32 torch.Size([36, 8])
        # #curr_gripper_history torch.float32 torch.Size([36, 3, 8])
        # #action torch.float32 torch.Size([36, 8])
        # #instr torch.float32 torch.Size([36, 53, 512])
        # #task ['close_jar', 'close_jar', 'close_jar', 'close_jar', 'close_jar', 'sweep_to_dustpan_of_size', 'sweep_to_dustpan_of_size', 'sweep_to_dustpan_of_size', 'sweep_to_dustpan_of_size', 'sweep_to_dustpan_of_size', 'place_wine_at_rack_location', 'place_wine_at_rack_location', 'place_wine_at_rack_location', 'place_wine_at_rack_location', 'place_wine_at_rack_location', 'put_groceries_in_cupboard', 'put_groceries_in_cupboard', 'put_groceries_in_cupboard', 'put_groceries_in_cupboard', 'put_groceries_in_cupboard', 'open_drawer', 'open_drawer', 'open_drawer', 'open_drawer', 'open_drawer', 'open_drawer', 'push_buttons', 'push_buttons', 'push_buttons', 'push_buttons', 'push_buttons', 'place_wine_at_rack_location', 'place_wine_at_rack_location', 'place_wine_at_rack_location', 'place_wine_at_rack_location', 'place_wine_at_rack_location']
        #
        # import open3d as o3d
        # for i in range(len(sample['rgbs'])):
        #     rgbs = sample['rgbs'][i][:,:,torch.arange(128)*2,:][:,:,:,torch.arange(128)*2].permute(0, 2, 3, 1).reshape(-1, 3).detach().cpu().numpy() # 4*256*256, 3
        #     pcds = sample['pcds'][i][:,:,torch.arange(128)*2,:][:,:,:,torch.arange(128)*2].permute(0, 2, 3, 1).reshape(-1, 3).detach().cpu().numpy() # 4*256*256, 3
        #     trajectory = sample['trajectory'][i].detach().cpu().numpy()[-1,[0,1,2,6,3,4,5,7]] # 2, 8
        #     trajectory_mask = sample['trajectory_mask'][i][-1].detach().cpu().numpy() # 2
        #     curr_gripper = sample['curr_gripper'][i].detach().cpu().numpy()[[0,1,2,6,3,4,5,7]] # 8
        #     curr_gripper_history = sample['curr_gripper_history'][i].detach().cpu().numpy()[:,[0,1,2,6,3,4,5,7]] # 3, 8
        #     action = sample['action'][i].detach().cpu().numpy()[[0,1,2,6,3,4,5,7]] # 8
        #
        #     pcd_scene = o3d.geometry.PointCloud()
        #     pcd_scene.points = o3d.utility.Vector3dVector(pcds)
        #     pcd_scene.colors = o3d.utility.Vector3dVector(rgbs)
        #
        #     coord_gripper = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2)
        #     coord_gripper.rotate(o3d.geometry.get_rotation_matrix_from_quaternion(curr_gripper[3:3+4]))
        #     coord_gripper.translate(curr_gripper[:3])
        #     sphere_gripper = o3d.geometry.TriangleMesh.create_sphere(0.05)
        #     sphere_gripper.translate(curr_gripper[:3])
        #     sphere_gripper.paint_uniform_color([0, 0, 1] if curr_gripper[7] > 0.5 else [1, 0, 0])
        #
        #     coord_action = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2)
        #     coord_action.rotate(o3d.geometry.get_rotation_matrix_from_quaternion(trajectory[3:3+4]))
        #     coord_action.translate(trajectory[:3])
        #     sphere_action = o3d.geometry.TriangleMesh.create_sphere(0.05)
        #     sphere_action.translate(trajectory[:3])
        #     sphere_action.paint_uniform_color([0, 0, 1] if trajectory[7] > 0.5 else [1, 0, 0])
        #
        #     coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame()
        #     o3d.visualization.draw_geometries([
        #         coordinate,
        #         pcd_scene,
        #         coord_gripper, sphere_gripper,
        #         coord_action, sphere_action,
        #     ])
        # exit(0)
        # ###############################################################################################################

        ###############################################################################################################
        # Diffusion-EDF train_one_step

        # assert len(sample['rgbs']) == 1, \
        #     f"Batch size is {len(sample['rgbs'])} > 1. Batch inference is currently not supported."

        if step_id % self.args.accumulate_grad_batches == 0:
            optimizer.zero_grad()

        total_losses = []
        for bn in range(len(sample['rgbs'])):
            # batch = replay_sample
            # for k in batch:
            #     print(k, batch[k])
            # print()

            # # --------------------------------------------------------------------------------------------- #
            # # preprocess observations
            # # --------------------------------------------------------------------------------------------- #
            # obs, pcds = self._preprocess_inputs(batch, CAMERAS)
            #
            # # --------------------------------------------------------------------------------------------- #
            # # flatten observations
            # # --------------------------------------------------------------------------------------------- #
            # bs = obs[0][0].shape[0]
            # pcd_flat = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, 3) for p in pcds], 1)
            # # print(pcd_flat.reshape(-1, 3).min(dim=0).values, pcd_flat.reshape(-1, 3).max(dim=0).values)
            #
            # image_features = [o[0] for o in obs]
            # # print('image_features[0]', image_features[0].shape)
            # feat_size = image_features[0].shape[1]
            # flat_imag_features = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, feat_size) for p in image_features], 1)
            # # for p in image_features:
            # #     print('p', p.shape)

            pcd_flat = sample['pcds'][bn][:,:,torch.arange(128)*2,:][:,:,:,torch.arange(128)*2].permute(0,2,3,1).reshape(1,-1,3).cuda()
            flat_imag_features = sample['rgbs'][bn][:,:,torch.arange(128)*2,:][:,:,:,torch.arange(128)*2].permute(0,2,3,1).reshape(1,-1,3).cuda()

            # --------------------------------------------------------------------------------------------- #
            # tensorize scene bounds
            # --------------------------------------------------------------------------------------------- #
            bounds = torch.tensor(SCENE_BOUNDS).unsqueeze(0).cuda()

            # --------------------------------------------------------------------------------------------- #
            # voxelize!
            # --------------------------------------------------------------------------------------------- #
            # print('pcd_flat', pcd_flat.shape, pcd_flat[0].min(dim=0).values, pcd_flat[0].max(dim=0).values)
            # print('flat_imag_features', flat_imag_features.shape, flat_imag_features[0].min(dim=0).values, flat_imag_features[0].max(dim=0).values)
            # print('bounds', bounds.shape)
            voxel_grid = self._voxelizer.coords_to_bounding_voxel_grid(
                pcd_flat,
                coord_features=flat_imag_features,
                coord_bounds=bounds
            ).flatten(start_dim=-4, end_dim=-2)
            # shape: (batch, voxel_idx, 10-dim feature: global_x,global_y,global_z; r,g,b; vox_x,vox_y,vox_z; occupancy)

            # --------------------------------------------------------------------------------------------- #
            # parse peract data format into diffusion-edf data format
            # --------------------------------------------------------------------------------------------- #
            assert len(voxel_grid.shape) == 3
            if voxel_grid.shape[0] != 1:
                raise NotImplementedError("Batch training not supported yet")
            voxel_grid = voxel_grid[0]  # (voxel_id, 10-dim feature), remove batch dim
            target_pose = sample['trajectory'][bn][-1:,:7].cuda() #batch['gripper_pose']#[0]  # remove batch dim
            action_gripper_openness = sample['trajectory'][bn][-1:,7].cuda() #batch['action'][:, -1]  # (1,)
            action_check_collision = sample['trajectory'][bn][-1:,7].cuda() #batch['ignore_collisions'][:, -1].float()  # (1,) # Don't know why, but seems like ignore_collisions=1 means not ignore collision and ignore_collisions=0 means check collision.
            # context_vec = sample['instr'][bn][-1:].cuda() #batch['lang_goal_emb']#[0]  # remove batch dim
            # context_vec = torch.stack([sample['instr'][bn][-2], sample['instr'][bn][-1]], dim=-1).view(1,-1).cuda()
            context_vec = sample['instr'][bn][sample['index'][bn]].cuda()
            # print('voxel_grid', voxel_grid.shape)
            # print('target_pose', target_pose.shape)
            # print('action_gripper_openness', action_gripper_openness.shape)
            # print('action_check_collision', action_check_collision.shape)
            # print('context_vec', context_vec.shape)
            # print('voxel_grid', voxel_grid.shape) # [1000000, 10]
            # print('current_pose', current_pose.shape) # [1, 7]
            # print('target_pose', target_pose.shape) # [1, 7]
            # print('action_gripper_openness', action_gripper_openness.shape) # [1]
            # print('action_check_collision', action_check_collision.shape) # [1]

            scene_pcd = voxel_grid[voxel_grid[..., -1].nonzero().squeeze(-1)]
            scene_pcd = PointCloud(points=scene_pcd[..., 0:3], colors=scene_pcd[..., 3:6])
            target_pose = SE3.from_orn_and_pos(
                positions=target_pose[..., :3], orns=target_pose[..., 3:], versor_last_input=True
            )

            # --------------------------------------------------------------------------------------------- #
            # crop tool pointcloud
            # --------------------------------------------------------------------------------------------- #
            tool_pcd = PointCloud(points=self.grasp_pts, colors=self.grasp_clr)

            # --------------------------------------------------------------------------------------------- #
            # Rescale from meters unit to centimeter unit for numerical precision (planned to be removed in the future)
            # --------------------------------------------------------------------------------------------- #
            scene_pcd_rescaled = FeaturedPoints(
                x=scene_pcd.points.cuda() * RESCALE_FACTOR,
                f=scene_pcd.colors.cuda(),
                b=torch.zeros_like(scene_pcd.points[..., 0], dtype=torch.long).cuda()
            )
            tool_pcd_rescaled = FeaturedPoints(
                x=tool_pcd.points.cuda() * RESCALE_FACTOR,
                f=tool_pcd.colors.cuda(),
                b=torch.zeros_like(tool_pcd.points[..., 0], dtype=torch.long).cuda()
            )
            T_target = target_pose.poses.cuda() * torch.tensor(
                [1., 1., 1., 1., RESCALE_FACTOR, RESCALE_FACTOR, RESCALE_FACTOR], dtype=dtype
            ).cuda()

            # --------------------------------------------------------------------------------------------- #\n",
            # Augment input pose for binary action classification\n",
            # --------------------------------------------------------------------------------------------- # \n",
            T_aug, _, __, ___, ____ = diffuse_T_target(
                T_target=T_target,
                x_ref=torch.randn(N_XREF, 3).cuda() * 0.005 * RESCALE_FACTOR,
                time=torch.tensor([0.003],dtype=dtype).cuda(),
                lin_mult=model.module.lin_mult,
                ang_mult=model.module.ang_mult,
            )

            # --------------------------------------------------------------------------------------------- #
            # Diffuse!
            # --------------------------------------------------------------------------------------------- #
            time_in = torch.empty(0).cuda()
            T_diffused = torch.empty(0, 7).cuda()
            gt_ang_score, gt_lin_score = torch.empty(0, 3).cuda(), torch.empty(0, 3).cuda()
            gt_ang_score_ref, gt_lin_score_ref = torch.empty(0, 3).cuda(), torch.empty(0, 3).cuda()

            for time_schedule in diffusion_schedules:
                time = random_time(
                    min_time=time_schedule[1],
                    max_time=time_schedule[0],
                    device=T_target.device
                )  # Shape: (1,)

                x_ref = torch.randn(N_XREF, 3).cuda() * 0.005 * RESCALE_FACTOR
                T_diffused_, delta_T_, time_in_, gt_score_, gt_score_ref_ \
                    = diffuse_T_target(
                    T_target=T_target,
                    x_ref=x_ref,
                    time=time,
                    lin_mult=model.module.lin_mult,
                    ang_mult=model.module.ang_mult
                )

                (gt_ang_score_, gt_lin_score_), (gt_ang_score_ref_, gt_lin_score_ref_) = gt_score_, gt_score_ref_
                T_diffused = torch.cat([T_diffused, T_diffused_], dim=0)
                time_in = torch.cat([time_in, time_in_], dim=0)
                gt_ang_score = torch.cat([gt_ang_score, gt_ang_score_], dim=0)
                gt_lin_score = torch.cat([gt_lin_score, gt_lin_score_], dim=0)
                gt_ang_score_ref = torch.cat([gt_ang_score_ref, gt_ang_score_ref_], dim=0)
                gt_lin_score_ref = torch.cat([gt_lin_score_ref, gt_lin_score_ref_], dim=0)

            key_pcd_multiscale: List[FeaturedPoints] = model.module.encode_scene(scene_pcd_rescaled, context_vec=context_vec)
            query_pcd: FeaturedPoints = model.module.encode_tool(tool_pcd_rescaled)

            loss, fp_info, tensor_info, statistics = model.module.get_train_loss(
                Ts=T_diffused,
                time=time_in,
                key_pcd_multiscale=key_pcd_multiscale,
                query_pcd=query_pcd,
                target_ang_score=gt_ang_score,
                target_lin_score=gt_lin_score,
                # context_vec=context_vec
            )

            bce_loss = model.module.get_bce_loss(
                Ts=T_aug,
                time=torch.ones_like(T_aug[...,0]), # dummy variable that should be removed in future\n",
                key_pcd_multiscale=key_pcd_multiscale,
                query_pcd=query_pcd,
                target_binary=torch.cat([
                    action_gripper_openness, action_check_collision
                ], dim=-1).repeat(len(T_aug),1),
                # context_vec=context_vec\n",
            )
            total_loss = loss + (bce_loss*0.2)
            total_loss.backward()
            # total_losses.append(total_loss)

        # sum(total_losses).backward()




        if step_id % self.args.accumulate_grad_batches == self.args.accumulate_grad_batches - 1:
            optimizer.step()

        # Log
        if dist.get_rank() == 0 and (step_id + 1) % self.args.val_freq == 0:
            self.writer.add_scalar("lr", self.args.lr, step_id)
            self.writer.add_scalar("train-loss/noise_mse", loss, step_id)



    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, step_id, val_iters,
                        split='val'):
        """Run a given number of evaluation steps."""
        if self.args.val_iters != -1:
            val_iters = self.args.val_iters
        values = {}
        device = next(model.parameters()).device
        model.eval()

        # for i, sample in enumerate(loader):
        #     if i == val_iters:
        #         break
        #
        #     if self.args.keypose_only:
        #         sample["trajectory"] = sample["trajectory"][:, [-1]]
        #         sample["trajectory_mask"] = sample["trajectory_mask"][:, [-1]]
        #     else:
        #         sample["trajectory"] = sample["trajectory"][:, 1:]
        #         sample["trajectory_mask"] = sample["trajectory_mask"][:, 1:]
        #
        #     curr_gripper = (
        #         sample["curr_gripper"] if self.args.num_history < 1
        #         else sample["curr_gripper_history"][:, -self.args.num_history:]
        #     )
        #     action = model(
        #         sample["trajectory"].to(device),
        #         sample["trajectory_mask"].to(device),
        #         sample["rgbs"].to(device),
        #         sample["pcds"].to(device),
        #         sample["instr"].to(device),
        #         curr_gripper.to(device),
        #         run_inference=True
        #     )
        #     losses, losses_B = criterion.compute_metrics(
        #         action,
        #         sample["trajectory"].to(device),
        #         sample["trajectory_mask"].to(device)
        #     )
        #
        #     # Gather global statistics
        #     for n, l in losses.items():
        #         key = f"{split}-losses/mean/{n}"
        #         if key not in values:
        #             values[key] = torch.Tensor([]).to(device)
        #         values[key] = torch.cat([values[key], l.unsqueeze(0)])
        #
        #     # Gather per-task statistics
        #     tasks = np.array(sample["task"])
        #     for n, l in losses_B.items():
        #         for task in np.unique(tasks):
        #             key = f"{split}-loss/{task}/{n}"
        #             l_task = l[tasks == task].mean()
        #             if key not in values:
        #                 values[key] = torch.Tensor([]).to(device)
        #             values[key] = torch.cat([values[key], l_task.unsqueeze(0)])
        #
        #     # Generate visualizations
        #     if i == 0 and dist.get_rank() == 0 and step_id > -1:
        #         viz_key = f'{split}-viz/viz'
        #         viz = generate_visualizations(
        #             action,
        #             sample["trajectory"].to(device),
        #             sample["trajectory_mask"].to(device)
        #         )
        #         self.writer.add_image(viz_key, viz, step_id)

        for i, sample in enumerate(loader):
            if i == val_iters:
                break

            # print(sample['rgbs'].shape) # T, 4, 3, 256, 256
            # print(sample['pcds'].shape) # T, 4, 3, 256, 256
            # print(sample['trajectory'].shape) # T, 1, 8
            # print(sample['trajectory_mask'].shape) # T, 1
            # print(sample['curr_gripper'].shape) # T, 8
            # print(sample['curr_gripper_history'].shape) # T, 3, 8
            # print(action.shape) # T, 1, 8
            # print(sample['instr'].shape) # T, 53, 512
            # exit(0)

            # assert len(sample['rgbs']) == 1, \
            #     f"Batch size is {len(sample['rgbs'])} > 1. Batch inference is currently not supported."
            for bn in range(len(sample['rgbs'])):
                pcd_flat = sample['pcds'][bn][:,:,torch.arange(128)*2,:][:,:,:,torch.arange(128)*2].permute(0,2,3,1).reshape(1,-1,3).cuda()
                flat_imag_features = sample['rgbs'][bn][:,:,torch.arange(128)*2,:][:,:,:,torch.arange(128)*2].permute(0,2,3,1).reshape(1,-1,3).cuda()

                # --------------------------------------------------------------------------------------------- #
                # tensorize scene bounds
                # --------------------------------------------------------------------------------------------- #
                bounds = torch.tensor(SCENE_BOUNDS).unsqueeze(0).cuda()

                # --------------------------------------------------------------------------------------------- #
                # voxelize!
                # --------------------------------------------------------------------------------------------- #
                # print('pcd_flat', pcd_flat.shape, pcd_flat[0].min(dim=0).values, pcd_flat[0].max(dim=0).values)
                # print('flat_imag_features', flat_imag_features.shape, flat_imag_features[0].min(dim=0).values, flat_imag_features[0].max(dim=0).values)
                # print('bounds', bounds.shape)
                voxel_grid = self._voxelizer.coords_to_bounding_voxel_grid(
                    pcd_flat,
                    coord_features=flat_imag_features,
                    coord_bounds=bounds
                ).flatten(start_dim=-4, end_dim=-2)
                # shape: (batch, voxel_idx, 10-dim feature: global_x,global_y,global_z; r,g,b; vox_x,vox_y,vox_z; occupancy)

                # --------------------------------------------------------------------------------------------- #
                # parse peract data format into diffusion-edf data format
                # --------------------------------------------------------------------------------------------- #
                assert len(voxel_grid.shape) == 3
                if voxel_grid.shape[0] != 1:
                    raise NotImplementedError("Batch training not supported yet")
                voxel_grid = voxel_grid[0]  # (voxel_id, 10-dim feature), remove batch dim
                current_pose = sample['trajectory'][bn][-1:,:7].cuda()  # batch['gripper_pose']#[0]  # remove batch dim
                action_gripper_openness = sample['trajectory'][bn][-1:,7].cuda()  # batch['action'][:, -1]  # (1,)
                action_check_collision = sample['trajectory'][bn][-1:,7].cuda()  # batch['ignore_collisions'][:, -1].float()  # (1,) # Don't know why, but seems like ignore_collisions=1 means not ignore collision and ignore_collisions=0 means check collision.
                # context_vec = sample['instr'][bn][-1:].cuda() #batch['lang_goal_emb']#[0]  # remove batch dim
                # context_vec = torch.stack([sample['instr'][bn][-2], sample['instr'][bn][-1]], dim=-1).view(1,-1).cuda()
                context_vec = sample['instr'][bn][-1].cuda()
                # print('voxel_grid', voxel_grid.shape)
                # print('target_pose', target_pose.shape)
                # print('action_gripper_openness', action_gripper_openness.shape)
                # print('action_check_collision', action_check_collision.shape)
                # print('context_vec', context_vec.shape)
                # print('voxel_grid', voxel_grid.shape) # [1000000, 10]
                # print('current_pose', current_pose.shape) # [1, 7]
                # print('target_pose', target_pose.shape) # [1, 7]
                # print('action_gripper_openness', action_gripper_openness.shape) # [1]
                # print('action_check_collision', action_check_collision.shape) # [1]

                scene_pcd = voxel_grid[voxel_grid[..., -1].nonzero().squeeze(-1)]
                scene_pcd = PointCloud(points=scene_pcd[..., 0:3], colors=scene_pcd[..., 3:6])
                current_pose = SE3.from_orn_and_pos(
                    positions=current_pose[..., :3], orns=current_pose[..., 3:], versor_last_input=True
                )

                # --------------------------------------------------------------------------------------------- #
                # crop tool pointcloud
                # --------------------------------------------------------------------------------------------- #
                tool_pcd = PointCloud(points=self.grasp_pts, colors=self.grasp_clr)

                # --------------------------------------------------------------------------------------------- #
                # Rescale from meters unit to centimeter unit for numerical precision (planned to be removed in the future)
                # --------------------------------------------------------------------------------------------- #
                scene_pcd_rescaled = FeaturedPoints(
                    x=scene_pcd.points.cuda() * RESCALE_FACTOR,
                    f=scene_pcd.colors.cuda(),
                    b=torch.zeros_like(scene_pcd.points[..., 0], dtype=torch.long).cuda()
                )
                tool_pcd_rescaled = FeaturedPoints(
                    x=tool_pcd.points.cuda() * RESCALE_FACTOR,
                    f=tool_pcd.colors.cuda(),
                    b=torch.zeros_like(tool_pcd.points[..., 0], dtype=torch.long).cuda()
                )
                T_target = current_pose.poses.cuda() * torch.tensor(
                    [1., 1., 1., 1., RESCALE_FACTOR, RESCALE_FACTOR, RESCALE_FACTOR], dtype=dtype
                ).cuda()

                # --------------------------------------------------------------------------------------------- #\n",
                # Augment input pose for binary action classification\n",
                # --------------------------------------------------------------------------------------------- # \n",
                T_aug, _, __, ___, ____ = diffuse_T_target(
                    T_target=T_target,
                    x_ref=torch.randn(N_XREF, 3).cuda() * 0.005 * RESCALE_FACTOR,
                    time=torch.tensor([0.003], dtype=dtype).cuda(),
                    lin_mult=model.module.lin_mult,
                    ang_mult=model.module.ang_mult,
                )

                # --------------------------------------------------------------------------------------------- #
                # Diffuse!
                # --------------------------------------------------------------------------------------------- #
                time_in = torch.empty(0).cuda()
                T_diffused = torch.empty(0, 7).cuda()
                gt_ang_score, gt_lin_score = torch.empty(0, 3).cuda(), torch.empty(0, 3).cuda()
                gt_ang_score_ref, gt_lin_score_ref = torch.empty(0, 3).cuda(), torch.empty(0, 3).cuda()

                for time_schedule in diffusion_schedules:
                    time = random_time(
                        min_time=time_schedule[1],
                        max_time=time_schedule[0],
                        device=T_target.device
                    )  # Shape: (1,)

                    x_ref = torch.randn(N_XREF, 3).cuda() * 0.005 * RESCALE_FACTOR
                    T_diffused_, delta_T_, time_in_, gt_score_, gt_score_ref_ \
                        = diffuse_T_target(
                        T_target=T_target,
                        x_ref=x_ref,
                        time=time,
                        lin_mult=model.module.lin_mult,
                        ang_mult=model.module.ang_mult
                    )

                    (gt_ang_score_, gt_lin_score_), (gt_ang_score_ref_, gt_lin_score_ref_) = gt_score_, gt_score_ref_
                    T_diffused = torch.cat([T_diffused, T_diffused_], dim=0)
                    time_in = torch.cat([time_in, time_in_], dim=0)
                    gt_ang_score = torch.cat([gt_ang_score, gt_ang_score_], dim=0)
                    gt_lin_score = torch.cat([gt_lin_score, gt_lin_score_], dim=0)
                    gt_ang_score_ref = torch.cat([gt_ang_score_ref, gt_ang_score_ref_], dim=0)
                    gt_lin_score_ref = torch.cat([gt_lin_score_ref, gt_lin_score_ref_], dim=0)

                key_pcd_multiscale: List[FeaturedPoints] = model.module.encode_scene(scene_pcd_rescaled, context_vec=context_vec)
                query_pcd: FeaturedPoints = model.module.encode_tool(tool_pcd_rescaled)

                loss, fp_info, tensor_info, statistics = model.module.get_train_loss(
                    Ts=T_diffused,
                    time=time_in,
                    key_pcd_multiscale=key_pcd_multiscale,
                    query_pcd=query_pcd,
                    target_ang_score=gt_ang_score,
                    target_lin_score=gt_lin_score,
                    # context_vec=context_vec
                )

                bce_loss = model.module.get_bce_loss(
                    Ts=T_aug,
                    time=torch.ones_like(T_aug[..., 0]),  # dummy variable that should be removed in future\n",
                    key_pcd_multiscale=key_pcd_multiscale,
                    query_pcd=query_pcd,
                    target_binary=torch.cat([
                        action_gripper_openness, action_check_collision
                    ], dim=-1).repeat(len(T_aug), 1),
                    # context_vec=context_vec\n",
                )


                # Gather global statistics
                for n, l in statistics.items():
                    key = f"{split}-losses/mean/{n}"
                    if key not in values:
                        values[key] = torch.Tensor([]).to(device)
                    values[key] = torch.cat([values[key], torch.tensor([[l]]).to(device)])#l.unsqueeze(0)])
                key = f"{split}-losses/mean/bce-loss"
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], torch.tensor([[bce_loss.item()]]).to(device)])#l.unsqueeze(0)])


                # # Gather per-task statistics
                # tasks = np.array(sample["task"])
                # for n, l in losses_B.items():
                #     for task in np.unique(tasks):
                #         key = f"{split}-loss/{task}/{n}"
                #         l_task = l[tasks == task].mean()
                #         if key not in values:
                #             values[key] = torch.Tensor([]).to(device)
                #         values[key] = torch.cat([values[key], l_task.unsqueeze(0)])

                # # Generate visualizations
                # if i == 0 and dist.get_rank() == 0 and step_id > -1:
                #     viz_key = f'{split}-viz/viz'
                #     viz = generate_visualizations(
                #         action,
                #         sample["trajectory"].to(device),
                #         sample["trajectory_mask"].to(device)
                #     )
                #     self.writer.add_image(viz_key, viz, step_id)

        # Log all statistics
        values = self.synchronize_between_processes(values)
        values = {k: v.mean().item() for k, v in values.items()}
        if dist.get_rank() == 0:
            if step_id > -1:
                for key, val in values.items():
                    self.writer.add_scalar(key, val, step_id)

            # Also log to terminal
            print(f"Step {step_id}:")
            for key, value in values.items():
                print(f"{key}: {value:.03f}")

        return values.get('val-losses/traj_pos_acc_001', None)


def traj_collate_fn(batch):
    keys = [
        "trajectory", "trajectory_mask",
        "rgbs", "pcds",
        "curr_gripper", "curr_gripper_history", "action", "instr"
    ]
    ret_dict = {
        key: torch.cat([
            item[key].float() if key != 'trajectory_mask' else item[key]
            for item in batch
        ]) for key in keys
    }

    ret_dict["task"] = []
    for item in batch:
        ret_dict["task"] += item['task']
    return ret_dict


class TrajectoryCriterion:

    def __init__(self):
        pass

    def compute_loss(self, pred, gt=None, mask=None, is_loss=True):
        if not is_loss:
            assert gt is not None and mask is not None
            return self.compute_metrics(pred, gt, mask)[0]['action_mse']
        return pred

    @staticmethod
    def compute_metrics(pred, gt, mask):
        # pred/gt are (B, L, 7), mask (B, L)
        pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()
        # symmetric quaternion eval
        quat_l1 = (pred[..., 3:7] - gt[..., 3:7]).abs().sum(-1)
        quat_l1_ = (pred[..., 3:7] + gt[..., 3:7]).abs().sum(-1)
        select_mask = (quat_l1 < quat_l1_).float()
        quat_l1 = (select_mask * quat_l1 + (1 - select_mask) * quat_l1_)
        # gripper openess
        openess = ((pred[..., 7:] >= 0.5) == (gt[..., 7:] > 0.0)).bool()
        tr = 'traj_'

        # Trajectory metrics
        ret_1, ret_2 = {
            tr + 'action_mse': F.mse_loss(pred, gt),
            tr + 'pos_l2': pos_l2.mean(),
            tr + 'pos_acc_001': (pos_l2 < 0.01).float().mean(),
            tr + 'rot_l1': quat_l1.mean(),
            tr + 'rot_acc_0025': (quat_l1 < 0.025).float().mean(),
            tr + 'gripper': openess.flatten().float().mean()
        }, {
            tr + 'pos_l2': pos_l2.mean(-1),
            tr + 'pos_acc_001': (pos_l2 < 0.01).float().mean(-1),
            tr + 'rot_l1': quat_l1.mean(-1),
            tr + 'rot_acc_0025': (quat_l1 < 0.025).float().mean(-1)
        }

        # Keypose metrics
        pos_l2 = ((pred[:, -1, :3] - gt[:, -1, :3]) ** 2).sum(-1).sqrt()
        quat_l1 = (pred[:, -1, 3:7] - gt[:, -1, 3:7]).abs().sum(-1)
        quat_l1_ = (pred[:, -1, 3:7] + gt[:, -1, 3:7]).abs().sum(-1)
        select_mask = (quat_l1 < quat_l1_).float()
        quat_l1 = (select_mask * quat_l1 + (1 - select_mask) * quat_l1_)
        ret_1.update({
            'pos_l2_final': pos_l2.mean(),
            'pos_l2_final<0.01': (pos_l2 < 0.01).float().mean(),
            'rot_l1': quat_l1.mean(),
            'rot_l1<0025': (quat_l1 < 0.025).float().mean()
        })
        ret_2.update({
            'pos_l2_final': pos_l2,
            'pos_l2_final<0.01': (pos_l2 < 0.01).float(),
            'rot_l1': quat_l1,
            'rot_l1<0.025': (quat_l1 < 0.025).float(),
        })

        return ret_1, ret_2


def fig_to_numpy(fig, dpi=60):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    return img


def generate_visualizations(pred, gt, mask, box_size=0.3):
    batch_idx = 0
    pred = pred[batch_idx].detach().cpu().numpy()
    gt = gt[batch_idx].detach().cpu().numpy()
    mask = mask[batch_idx].detach().cpu().numpy()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.scatter3D(
        pred[~mask][:, 0], pred[~mask][:, 1], pred[~mask][:, 2],
        color='red', label='pred'
    )
    ax.scatter3D(
        gt[~mask][:, 0], gt[~mask][:, 1], gt[~mask][:, 2],
        color='blue', label='gt'
    )

    center = gt[~mask].mean(0)
    ax.set_xlim(center[0] - box_size, center[0] + box_size)
    ax.set_ylim(center[1] - box_size, center[1] + box_size)
    ax.set_zlim(center[2] - box_size, center[2] + box_size)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plt.legend()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    img = fig_to_numpy(fig, dpi=120)
    plt.close()
    return img.transpose(2, 0, 1)


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Arguments
    args = Arguments().parse_args()
    print("Arguments:")
    print(args)
    print("-" * 100)
    if args.gripper_loc_bounds is None:
        args.gripper_loc_bounds = np.array([[-2, -2, -2], [2, 2, 2]]) * 1.0
    else:
        args.gripper_loc_bounds = get_gripper_loc_bounds(
            args.gripper_loc_bounds,
            task=args.tasks[0] if len(args.tasks) == 1 else None,
            buffer=args.gripper_loc_bounds_buffer,
        )
    log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Logging:", log_dir)
    print(
        "Available devices (CUDA_VISIBLE_DEVICES):",
        os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    print("Device count", torch.cuda.device_count())
    args.local_rank = int(os.environ["LOCAL_RANK"])

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # DDP initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Run
    train_tester = DiffusionEDFTrainTester(args)
    train_tester.main(collate_fn=traj_collate_fn)
