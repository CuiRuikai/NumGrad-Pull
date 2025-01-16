import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh


class Triplane(nn.Module):
    def __init__(
        self,
        reso=256,
        channel=32,
        init_type="geo_init",
    ):
        super().__init__()
        if init_type == "geo_init":
            sdf_proxy = nn.Sequential(
                nn.Linear(3, channel),
                nn.Softplus(beta=100),
                nn.Linear(channel, channel),
            )
            torch.nn.init.constant_(sdf_proxy[0].bias, 0.0)
            torch.nn.init.normal_(
                sdf_proxy[0].weight, 0.0, np.sqrt(2) / np.sqrt(channel)
            )
            torch.nn.init.constant_(sdf_proxy[2].bias, 0.0)
            torch.nn.init.normal_(
                sdf_proxy[2].weight, 0.0, np.sqrt(2) / np.sqrt(channel)
            )

            ini_sdf = torch.zeros([3, channel, reso, reso])
            X = torch.linspace(-1.0, 1.0, reso)
            (U, V) = torch.meshgrid(X, X, indexing="ij")
            Z = torch.zeros(reso, reso)
            inputx = torch.stack([Z, U, V], -1).reshape(-1, 3)
            inputy = torch.stack([U, Z, V], -1).reshape(-1, 3)
            inputz = torch.stack([U, V, Z], -1).reshape(-1, 3)
            ini_sdf[0] = sdf_proxy(inputx).permute(1, 0).reshape(channel, reso, reso)
            ini_sdf[1] = sdf_proxy(inputy).permute(1, 0).reshape(channel, reso, reso)
            ini_sdf[2] = sdf_proxy(inputz).permute(1, 0).reshape(channel, reso, reso)
            self.triplane = torch.nn.Parameter(ini_sdf / 3, requires_grad=True)
        elif init_type == "rand_init":
            self.triplane = torch.nn.Parameter(
                torch.randn([3, channel, reso, reso]) * 0.001, requires_grad=True
            )
        else:
            raise ValueError("Unknown init_type")

        self.R = reso
        self.C = channel
        self.register_buffer(
            "plane_axes",
            torch.tensor(
                [
                    [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                    [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                ],
                dtype=torch.float32,
            ),
        )

    def project_onto_planes(self, xyz):
        M, _ = xyz.shape
        xyz = xyz.unsqueeze(0).expand(3, -1, -1).reshape(3, M, 3)
        inv_planes = torch.linalg.inv(self.plane_axes).reshape(3, 3, 3)
        projections = torch.bmm(xyz, inv_planes)
        return projections[..., :2]  # [3, M, 2]

    def forward(self, xyz):
        # pts: [M,3]
        M, _ = xyz.shape
        projected_coordinates = self.project_onto_planes(xyz).unsqueeze(1)
        feats = F.grid_sample(
            self.triplane,  # [3,C,R,R]
            projected_coordinates.float(),  # [3,1,M,2]
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # [3,C,1,M]
        feats = feats.permute(0, 3, 2, 1).reshape(3, M, self.C).sum(0)
        return feats  # [M,C]

    def update_resolution(self, new_reso):
        new_tri = F.interpolate(
            self.triplane.data,
            size=(new_reso, new_reso),
            mode="bilinear",
            align_corners=True,
        )
        self.R = new_reso
        self.triplane = torch.nn.Parameter(new_tri, requires_grad=True)


class NGPullNetwork(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        skip_in=(4,),
        multires=0,
        bias=0.5,
        scale=1,
        geometric_init=True,
        weight_norm=True,
        inside_outside=False,
    ):
        super(NGPullNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=-np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(
                        lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )

            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)

        self.activation = nn.ReLU()

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)

        return x / self.scale

    def sdf(self, x):
        return self.forward(x)


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    Suggested by https://github.com/mikedh/trimesh/issues/507
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()
                )
            )
    else:
        print("is_mesh")
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh
