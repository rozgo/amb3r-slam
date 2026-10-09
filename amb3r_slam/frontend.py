"""Camera front-end: a pose for every frame as it arrives, from a few-view forward pass."""

import numpy as np
import torch

from amb3r_slam.model import SLAMModel
from amb3r_slam.tools.geometry import (average_transforms, sim3_from_matrix, sim3_inv,
                                      sim3_mul, sim3_to_se3_matrix)


def _umeyama_from_cameras(P_local, P_world, weights=None, s_prior=None, min_sep=1e-6,
                          scale=None):
    """Sim(3) taking a forward's local frame into the world, from cameras known in both.
    ``scale``, when given, is used instead of the ratio of the cameras' distances."""
    F = P_local.shape[0]
    if scale is not None and np.isfinite(scale) and scale > 0:
        s = float(scale)
    else:
        if F < 2:
            return None
        cw, cl = P_world[:, :3, 3], P_local[:, :3, 3]
        iu = torch.triu_indices(F, F, offset=1, device=cw.device)
        dw = (cw[iu[0]] - cw[iu[1]]).norm(dim=-1)
        dl = (cl[iu[0]] - cl[iu[1]]).norm(dim=-1)
        ok = (dl > min_sep) & (dw > min_sep)
        s = float(torch.median(dw[ok] / dl[ok])) if int(ok.sum()) else float('nan')
    if not np.isfinite(s) or s <= 0:
        if s_prior is None or not np.isfinite(s_prior) or s_prior <= 0:
            return None
        s = float(s_prior)

    Pl = P_local.clone()
    Pl[:, :3, 3] *= s
    rel = P_world @ torch.inverse(Pl)
    w = torch.ones(F, device=P_local.device) if weights is None else weights
    T = average_transforms(rel, w)
    g = sim3_from_matrix(T.double(), scale=1.0).float()
    g[7] = s
    return g


class FrontEnd:
    """Tracks frames against a keyframe and promotes keyframes by measured overlap."""

    def __init__(self, model, cfg, calib_K=None):
        self.net = SLAMModel(model, cfg.device, cfg.backend.pixel_target, patch=14)
        self.cfg = cfg
        self.device = cfg.device
        self.calib_K = calib_K
        self._last_K = None
        self.reset()

    def reset(self):
        self.kf_idx = None
        self.kf_pose = None
        self.recent = []
        self.poses = {}
        self.scale = None
        self.kf_depth = None    # the keyframe's depth at the subsampled pixels, in world units
        self.last_conf = 0.0


    @torch.no_grad()
    def _forward(self, images, ref):
        poses, pts, conf, self._last_K, _ = self.net(images, ref, self.calib_K)
        pts, conf = self.net.subsample(pts, conf)
        return poses, pts, conf

    @staticmethod
    def _depth(pts, pose):
        """Depth of one view's points (N, 3), given in a pass's frame, in its own camera."""
        return ((pts.reshape(-1, 3).float() - pose[:3, 3].float()) @ pose[:3, :3].float())[:, 2]

    def _depth_scale(self, pts, conf, poses_l, k):
        """World units per unit of this pass, from the keyframe alone: its depth in world units
        over the depth this pass gives it, on its more confident half. Unlike the cameras'
        distances, this does not depend on frames this front-end has placed since the
        keyframe, so an error in one frame's placement does not feed the next frame's scale."""
        if self.kf_depth is None:
            return None
        d = self._depth(pts[k], poses_l[k])
        if d.shape != self.kf_depth.shape:
            return None
        c = conf[k].reshape(-1).float()
        ok = torch.isfinite(d) & torch.isfinite(self.kf_depth) & (d > 1e-6) & (self.kf_depth > 1e-6)
        if int(ok.sum()) < 32:
            return None
        ok &= c >= torch.median(c[ok])
        s = float(torch.median(self.kf_depth[ok] / d[ok]))
        return s if np.isfinite(s) and s > 0 else None

    def reseat(self, frame, pose, depth, recent):
        """Restart tracking from the backend: keyframe ``frame`` at ``pose`` with its depth in
        world units, and ``recent`` (frame, pose) pairs as the context views."""
        self.kf_idx, self.kf_pose = frame, pose
        self.kf_depth = None if depth is None else depth.to(self.device).float()
        self.recent = [(f, p) for f, p in recent]
        for f, p in recent:
            self.poses[f] = p
        self.poses[frame] = pose

    def align_to(self, frames, P):
        """Sim(3) taking this front-end's world onto ``P``, the backend's poses of ``frames``,
        fitted on the frames already tracked here; None when fewer than two are shared."""
        rows = [i for i, fr in enumerate(frames) if fr in self.poses]
        if len(rows) < 2:
            return None
        P_mine = torch.stack([self.poses[frames[i]] for i in rows]).to(self.device).float()
        g = _umeyama_from_cameras(P_mine, P[rows].to(self.device).float())
        return g if g is not None and torch.isfinite(g).all() else None

    def reanchor(self, frames, P):
        """Replace the poses of ``frames`` with the backend's, so tracking restarts from them.
        ``P`` must be in the same world as the poses kept (see ``align_to``)."""
        for i, fr in enumerate(frames):
            self.poses[fr] = P[i]
        if self.kf_idx in self.poses:
            self.kf_pose = self.poses[self.kf_idx]
        self.recent = [(i, self.poses[i]) for i, _ in self.recent if i in self.poses]

    def transform(self, D):
        """Apply a Sim(3) correction ``D`` of the world to every stored pose."""
        if not self.poses:
            return
        keys = list(self.poses)
        P = torch.stack([self.poses[k] for k in keys]).to(self.device)
        P = sim3_to_se3_matrix(sim3_mul(D[None].expand(len(keys), 8),
                                        sim3_from_matrix(P.double()).float()))
        self.reanchor(keys, P)
        if self.scale is not None:
            self.scale = float(self.scale) * float(D[7])
        if self.kf_depth is not None:
            self.kf_depth = self.kf_depth * float(D[7])

    @torch.no_grad()
    def track(self, imgs, f):
        """Track frame ``f``; returns its world pose (4, 4) or None before the anchor."""
        fcfg = self.cfg.frontend
        n_ctx = max(0, int(fcfg.context))

        if self.kf_idx is None:
            self.kf_idx = f
            self.kf_pose = torch.eye(4, device=self.device)
            self.poses[f] = self.kf_pose
            return self.kf_pose

        ctx = [i for i, _ in self.recent[-n_ctx:]]
        views = sorted(dict.fromkeys([self.kf_idx] + ctx + [f]))
        poses_l, pts, conf = self._forward(imgs[views], ref=fcfg.ref)

        known = [(k, v) for k, v in enumerate(views) if v in self.poses]
        j = views.index(f)
        k = views.index(self.kf_idx)
        s_kf = self._depth_scale(pts, conf, poses_l, k)
        if len(known) >= 2 or s_kf is not None:
            rows = torch.tensor([k0 for k0, _ in known], device=poses_l.device)
            P_world = torch.stack([self.poses[v] for _, v in known]).to(poses_l.device)
            P_local = poses_l[rows].float()
            g = _umeyama_from_cameras(P_local, P_world.float(), s_prior=self.scale, scale=s_kf)
        else:
            k0 = known[0][0]
            g0 = sim3_from_matrix(self.poses[views[k0]].double(), scale=1.0).float()
            g = sim3_mul(g0, sim3_inv(sim3_from_matrix(poses_l[k0].double(),
                                                       scale=1.0).float()))
        if g is None or not torch.isfinite(g).all():
            return self.poses.get(f)

        self.scale = float(g[7])
        if self.kf_depth is None:
            self.kf_depth = self._depth(pts[k], poses_l[k]) * self.scale
        gl = sim3_from_matrix(poses_l[j].double(), scale=1.0).float()
        pose = sim3_to_se3_matrix(sim3_mul(g, gl))
        if not torch.isfinite(pose).all():
            return self.poses.get(f)
        self.poses[f] = pose

        ov = self._overlap(pts, poses_l, k, j, imgs.shape[-2:])
        self.last_conf = float(conf[j].float().mean())
        if ov < float(fcfg.min_overlap) or \
                (f - self.kf_idx) >= int(fcfg.max_gap):
            self.kf_idx, self.kf_pose = f, pose
            self.kf_depth = self._depth(pts[j], poses_l[j]) * self.scale
        self.recent.append((f, pose))
        self.recent = self.recent[-max(1, n_ctx):]
        return pose

    def _overlap(self, pts, poses_l, k, j, hw):
        """Fraction of the keyframe's points that still land inside the new frame."""
        K = self._last_K
        if K is None:
            return 1.0
        H, W = hw
        R, t = poses_l[j][:3, :3], poses_l[j][:3, 3]
        X = (pts[k].reshape(-1, 3) - t) @ R
        z = X[:, 2]
        Kj = K[j] if K.dim() == 3 else K
        uv = X @ Kj.T
        u, v = uv[:, 0] / z.clamp_min(1e-6), uv[:, 1] / z.clamp_min(1e-6)
        ok = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        return float(ok.float().mean())

