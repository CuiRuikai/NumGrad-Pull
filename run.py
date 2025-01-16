# -*- coding: utf-8 -*-

import os
import time
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import mcubes
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.utils import get_root_logger, print_log
from models.dataset import DatasetNP
from models.fields import NGPullNetwork, Triplane

import warnings

warnings.filterwarnings("ignore")


def create_optimizer(net, triplane, lr_net=1e-3, lr_tri=1e-2):
    params_to_train = []
    if net is not None:
        params_to_train += [{"name": "net", "params": net.parameters(), "lr": lr_net}]
    if triplane is not None:
        params_to_train += [
            {"name": "tri", "params": triplane.parameters(), "lr": lr_tri}
        ]
    return torch.optim.Adam(params_to_train)


class Runner:
    def __init__(self, args, conf_path, mode="train"):
        self.device = torch.device("cuda")

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf["dataset.np_data_name"] = self.conf["dataset.np_data_name"]
        self.base_exp_dir = self.conf["general.base_exp_dir"] + args.dir
        os.makedirs(self.base_exp_dir, exist_ok=True)

        self.dataset_np = DatasetNP(self.conf["dataset"], args.dataname)
        self.dataname = args.dataname
        self.iter_step = 0

        # Training parameters
        self.maxiter = self.conf.get_int("train.maxiter")
        self.save_freq = self.conf.get_int("train.save_freq")
        self.report_freq = self.conf.get_int("train.report_freq")
        self.val_freq = self.conf.get_int("train.val_freq")
        self.batch_size = self.conf.get_int("train.batch_size")
        self.learning_rate = self.conf.get_float("train.learning_rate")
        self.warm_up_end = self.conf.get_float("train.warm_up_end", default=0.0)
        self.eval_num_points = self.conf.get_int("train.eval_num_points")

        self.lr_net = self.conf.get_float("train.lr_net")
        self.lr_tri = self.conf.get_float("train.lr_tri")

        self.eps = self.conf.get_float("train.grad_eps")
        self.resolution = self.conf.get_int("train.resolution")
        self.c2f_scale = self.conf.get_list("train.c2f_scale")

        self.mode = mode

        # Networks
        self.sdf_network = NGPullNetwork(**self.conf["model.sdf_network"]).to(
            self.device
        )
        self.triplane = Triplane(
            reso=self.resolution // (2 ** len(self.c2f_scale)),
            channel=self.conf["model.sdf_network.d_in"],
            init_type=self.conf.get_string("model.triplane.init_type"),
        ).to(self.device)
        self.optimizer = create_optimizer(
            self.sdf_network, self.triplane, self.lr_net, self.lr_tri
        )

        # Backup codes and configs for debug
        if self.mode[:5] == "train":
            self.file_backup()

    def train(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        log_file = os.path.join(os.path.join(self.base_exp_dir), f"{timestamp}.log")
        logger = get_root_logger(log_file=log_file, name="outs")
        self.logger = logger
        batch_size = self.batch_size

        res_step = self.maxiter - self.iter_step

        eps_tensor = (
            torch.cat(
                [
                    torch.as_tensor([[self.eps, 0.0, 0.0]]),
                    torch.as_tensor([[-self.eps, 0.0, 0.0]]),
                    torch.as_tensor([[0.0, self.eps, 0.0]]),
                    torch.as_tensor([[0.0, -self.eps, 0.0]]),
                    torch.as_tensor([[0.0, 0.0, self.eps]]),
                    torch.as_tensor([[0.0, 0.0, -self.eps]]),
                ]
            )
            .unsqueeze(1)
            .to(self.device)
        )

        for iter_i in tqdm(range(res_step)):
            if iter_i in self.c2f_scale:
                new_reso = int(
                    self.resolution
                    / (2 ** (len(self.c2f_scale) - self.c2f_scale.index(iter_i) - 1))
                )
                print_log("Update resolution to {}".format(new_reso), logger=logger)
                self.triplane.update_resolution(new_reso)
                self.optimizer = create_optimizer(self.sdf_network, self.triplane)
                torch.cuda.empty_cache()

            self.update_learning_rate_np(iter_i)

            points, samples, point_gt = self.dataset_np.np_train_data(batch_size)
            samples_all = torch.cat(
                [(samples.unsqueeze(0) + eps_tensor).reshape(-1, 3), samples], dim=0
            )
            sdfs_all = self.sdf_network(self.triplane(samples_all)).reshape(
                7, -1, 1
            )  # [7, N, 1]
            gradients_sample = torch.cat(
                [
                    0.5 * (sdfs_all[0, :] - sdfs_all[1, :]) / self.eps,
                    0.5 * (sdfs_all[2, :] - sdfs_all[3, :]) / self.eps,
                    0.5 * (sdfs_all[4, :] - sdfs_all[5, :]) / self.eps,
                ],
                dim=-1,
            )

            sdf_sample = sdfs_all[-1, :]  # 5000x1
            grad_norm = F.normalize(gradients_sample, dim=1)  # 5000x3
            sample_moved = samples - grad_norm * sdf_sample  # 5000x3

            loss = torch.linalg.norm((points - sample_moved), ord=2, dim=-1).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1
            if self.iter_step % self.report_freq == 0:
                print_log(
                    "iter:{:8>d} cd_l1 = {} lr={}".format(
                        self.iter_step, loss, self.optimizer.param_groups[0]["lr"]
                    ),
                    logger=logger,
                )

            if self.iter_step % self.val_freq == 0 and self.iter_step != 0:
                self.validate_mesh(
                    resolution=256,
                    threshold=args.mcubes_threshold,
                    point_gt=point_gt,
                    iter_step=self.iter_step,
                    logger=logger,
                )

            if self.iter_step % self.save_freq == 0 and self.iter_step != 0:
                self.save_checkpoint()

    def validate_mesh(
        self, resolution=64, threshold=0.0, point_gt=None, iter_step=0, logger=None
    ):
        bound_min = torch.tensor(self.dataset_np.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset_np.object_bbox_max, dtype=torch.float32)
        os.makedirs(os.path.join(self.base_exp_dir, "outputs"), exist_ok=True)
        mesh = self.extract_geometry(
            bound_min,
            bound_max,
            resolution=resolution,
            threshold=threshold,
            query_func=lambda pts: -self.sdf_network.sdf(self.triplane(pts)),
        )

        mesh.export(
            os.path.join(
                self.base_exp_dir,
                "outputs",
                "{:0>8d}_{}.ply".format(self.iter_step, str(threshold)),
            )
        )

    def update_learning_rate_np(self, iter_step):
        warn_up = self.warm_up_end
        max_iter = self.maxiter
        init_lr = self.learning_rate
        lr = (
            (iter_step / warn_up)
            if iter_step < warn_up
            else 0.5
            * (math.cos((iter_step - warn_up) / (max_iter - warn_up) * math.pi) + 1)
        )
        lr = lr * init_lr
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def extract_fields(self, bound_min, bound_max, resolution, query_func):
        N = 32
        X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs)
                        pts = torch.cat(
                            [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                            dim=-1,
                        )
                        val = (
                            query_func(pts)
                            .reshape(len(xs), len(ys), len(zs))
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        u[
                            xi * N : xi * N + len(xs),
                            yi * N : yi * N + len(ys),
                            zi * N : zi * N + len(zs),
                        ] = val
        return u

    def extract_geometry(self, bound_min, bound_max, resolution, threshold, query_func):
        print("Creating mesh with threshold: {}".format(threshold))
        u = self.extract_fields(bound_min, bound_max, resolution, query_func)
        vertices, triangles = mcubes.marching_cubes(u, threshold)
        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()

        vertices = (
            vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :]
            + b_min_np[None, :]
        )
        mesh = trimesh.Trimesh(vertices, triangles)

        return mesh

    def file_backup(self):
        dir_lis = self.conf["general.recording"]
        os.makedirs(os.path.join(self.base_exp_dir, "recording"), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, "recording", dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == ".py":
                    copyfile(
                        os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name)
                    )

        copyfile(
            self.conf_path, os.path.join(self.base_exp_dir, "recording", "config.conf")
        )

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(
            os.path.join(self.base_exp_dir, "checkpoints", checkpoint_name),
            map_location=self.device,
        )
        print(os.path.join(self.base_exp_dir, "checkpoints", checkpoint_name))
        self.sdf_network.load_state_dict(checkpoint["sdf_network_fine"])
        self.triplane.load_state_dict(checkpoint["triplane"])

        self.iter_step = checkpoint["iter_step"]

    def save_checkpoint(self):
        checkpoint = {
            "sdf_network_fine": self.sdf_network.state_dict(),
            "triplane": self.triplane.state_dict(),
            "iter_step": self.iter_step,
        }
        os.makedirs(os.path.join(self.base_exp_dir, "checkpoints"), exist_ok=True)
        torch.save(
            checkpoint,
            os.path.join(
                self.base_exp_dir,
                "checkpoints",
                "ckpt_{:0>6d}.pth".format(self.iter_step),
            ),
        )


if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="./confs/np_srb.conf")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--mcubes_threshold", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dir", type=str, default="gargoyle")
    parser.add_argument("--dataname", type=str, default="gargoyle")
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args, args.conf, args.mode)

    if args.mode == "train":
        runner.train()
    elif args.mode == "validate_mesh":
        threshs = [
            -0.001,
            -0.0025,
            -0.005,
            -0.01,
            -0.02,
            0.0,
            0.001,
            0.0025,
            0.005,
            0.01,
            0.02,
        ]
        for thresh in threshs:
            runner.validate_mesh(resolution=256, threshold=thresh)
