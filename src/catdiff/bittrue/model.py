"""位真 UNet 前向（docs/bittrue-spec.md §2-§5，结构镜像 src/catdiff/model/unet.py）。

全部数值为整数（torch int64 张量 / float64 容器精确累加）；输入 x INT16
@ s_x，输出 eps INT16 @ s16(g)。forward(x_q16, step_idx) 与 fake-quant 参考对齐；
trace/trace_in 记录各量化点输出/输入 codes（黄金向量用）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from catdiff.bittrue.graph import Graph, build_graph, is_fine
from catdiff.bittrue.loader import BittrueTables, load_export
from catdiff.bittrue.primitives import groupnorm_int, int_conv2d, \
    int_linear, requant, sat_int, silu_lut, softmax_uint8


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.int64)


def _tc(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).to(torch.float64)


class _StepCtx:
    """一步之内全部定点常量（组 g 的 LUT 与该步的 FiLM）。"""

    def __init__(self, t: BittrueTables, step_idx: int):
        num_steps = len(t.ddim_A)
        self.g = t.group(step_idx, num_steps)
        self.t = t
        self.film = {r: q[step_idx] for r, q in t.film_q32.items()}
        self.silu = {k: lut[self.g] for k, lut in t.silu_luts.items()}
        self.kexp = {a: k[self.g] for a, k in t.attn_Kexp.items()}


class BittrueUNet:
    def __init__(self, tables: BittrueTables, graph: Graph):
        self.t = tables
        self.g = graph
        cfg = tables.unet_cfg
        self.num_groups = cfg["norm_num_groups"]
        self.lpb = cfg["layers_per_block"]
        self.n_down = len(cfg["down_block_types"])
        self.n_up = len(cfg["up_block_types"])

    # ---------------------------------------------------------------- 入口
    def forward(self, x_q16: torch.Tensor, step_idx: int,
                trace: dict | None = None,
                trace_in: dict | None = None) -> torch.Tensor:
        ctx = _StepCtx(self.t, step_idx)
        x16 = _np(x_q16)
        x = self._conv_in(x16[0], ctx)  # 位真模型内为 (C,H,W)，batch=1
        if trace is not None:
            trace["conv_in"] = x
        if trace_in is not None:
            trace_in["conv_in"] = x16[0]
        skips = [x]
        for b in range(self.n_down):
            x, block_skips = self._down_block(b, x, ctx, trace, trace_in)
            skips.extend(block_skips)
        x = self._mid_block(x, ctx, trace, trace_in)
        for u in range(self.n_up):
            take = skips[-(self.lpb + 1):]
            del skips[-(self.lpb + 1):]
            x = self._up_block(u, x, list(reversed(take)), ctx, trace,
                               trace_in)
        # 末端：conv_norm_out(GN) → SiLU → conv_out（镜像 model/unet.py.forward）
        x = self._gn("conv_norm_out", x, ctx)
        x = self._silu("conv_norm_out.gn", "conv_norm_out.silu", x, ctx)
        eps = self._conv_out(x, ctx)
        if trace is not None:
            trace["conv_out"] = eps
        if trace_in is not None:
            trace_in["conv_out"] = x
        return torch.from_numpy(eps)

    # ---------------------------------------------------------------- 基元
    def _point_codes(self, codes: np.ndarray, src: str, dst: str,
                     ctx: _StepCtx, wide: bool = False) -> np.ndarray:
        """恒等 requant：src 点 scale → dst 点 scale（契约 §4）。

        wide=True：INT16 饱和（residual 操作数），否则 INT8。
        """
        if src == dst:
            return codes
        t = self.t
        s_from = t.point_scale(src, ctx.g)
        s_to = t.point_scale(dst, ctx.g)
        M, N = _ident_mn(s_from, s_to)
        bits = 16 if (wide or is_fine(self.g, dst)) else 8
        return requant(codes, M, N, bits=bits)

    def _conv_acc(self, name: str, x: np.ndarray, ctx: _StepCtx,
                  stride: int = 1, pad: int | tuple = 0,
                  acc_bits: int = 32) -> np.ndarray:
        """原始累加器（含 b32），不 requant（FiLM 注入域，契约 §2.1）。"""
        import torch.nn.functional as F
        from catdiff.bittrue.primitives import _pad_tuple
        lp = self.t.layers[name]
        xf = _tc(x[None])
        if pad:
            xf = F.pad(xf, _pad_tuple(pad))
        acc = F.conv2d(xf, _tc(lp.w.astype(np.int64)), stride=stride)             + torch.from_numpy(lp.b32[ctx.g]).to(torch.float64).view(1, -1, 1, 1)
        lim = (1 << (acc_bits - 1)) - 1
        return _np(torch.clamp(acc, -lim, lim))[0]

    def _conv_layer(self, name: str, x: np.ndarray, ctx: _StepCtx,
                    stride: int = 1, pad: int | tuple = 0,
                    out_bits: int = 8, acc_bits: int = 32) -> np.ndarray:
        """(C,H,W) codes → (C',H',W') codes（batch=1，位真模型内去 batch 维）。"""
        lp = self.t.layers[name]
        gi = ctx.g
        y = int_conv2d(_tc(x[None]), _tc(lp.w.astype(np.int64)),
                       lp.M[gi].tolist(), lp.N[gi].tolist(),
                       bias32=torch.from_numpy(lp.b32[gi]),
                       stride=stride, pad=pad, out_bits=out_bits,
                       acc_bits=acc_bits)
        return _np(y)[0]

    def _linear_layer(self, name: str, x: np.ndarray, ctx: _StepCtx,
                      out_bits: int = 8, acc_bits: int = 32) -> np.ndarray:
        lp = self.t.layers[name]
        gi = ctx.g
        y = int_linear(_tc(x), _tc(lp.w.astype(np.int64)),
                       lp.M[gi].tolist(), lp.N[gi].tolist(),
                       bias32=torch.from_numpy(lp.b32[gi]), out_bits=out_bits,
                       acc_bits=acc_bits)
        return _np(y)

    def _linear_spatial(self, name: str, x: np.ndarray,
                        ctx: _StepCtx) -> np.ndarray:
        """(C,H,W) 逐像素 linear：内部按 (HW,C)@(C,O)。"""
        c, h, w = x.shape
        y = self._linear_layer(name, x.reshape(c, h * w).T, ctx)  # (HW, O)
        return np.ascontiguousarray(y.T).reshape(y.shape[1], h, w)

    def _gn(self, gn_name: str, codes: np.ndarray, ctx: _StepCtx) -> np.ndarray:
        p = self.t.gns[gn_name]
        gi = ctx.g
        lut0, lut1 = _rsqrt_luts()
        fine = is_fine(self.g, p.in_point)  # 输入为细网格（hidden/concat）
        return groupnorm_int(codes, self.num_groups, p.eps_q[gi], lut0, lut1,
                             p.Gc[gi], p.Nc[gi], p.Bq[gi], out_bits=16,
                             fine_input=fine)

    def _silu(self, in_point: str, out_point: str, codes: np.ndarray,
              ctx: _StepCtx) -> np.ndarray:
        return silu_lut(ctx.silu[(in_point, out_point)], codes)

    # ---------------------------------------------------------------- 块
    def _conv_in(self, x_q16: np.ndarray, ctx: _StepCtx) -> np.ndarray:
        return self._conv_layer("conv_in", x_q16, ctx, pad=1)  # INT16×INT16 → INT8

    def _conv_out(self, x: np.ndarray, ctx: _StepCtx) -> np.ndarray:
        return self._conv_layer("conv_out", x, ctx, pad=1, out_bits=16,
                                acc_bits=48)

    def _resnet(self, r: str, x_in: np.ndarray, skip: np.ndarray | None,
                ctx: _StepCtx, trace: dict | None,
                trace_in: dict | None) -> np.ndarray:
        info = self.g.resnets[r]
        if skip is not None:
            cat_point = r + ".concat"
            xa = self._point_codes(x_in, info.concat[0], cat_point, ctx)
            xb = self._point_codes(skip, info.concat[1], cat_point, ctx)
            x_cat = np.concatenate([xa, xb], axis=0)
            if trace_in is not None:
                trace_in[cat_point] = x_cat
        else:
            x_cat = x_in
        n1 = self._gn(r + ".norm1", x_cat, ctx)
        h1 = self._silu(info.norm1, info.conv1_in, n1, ctx)
        acc = self._conv_acc(r + ".conv1", h1, ctx, pad=1, acc_bits=48)
        acc = acc + ctx.film[r][:, None, None]          # FiLM acc 域注入（§5.1）
        lp1 = self.t.layers[r + ".conv1"]
        hid = requant(acc, lp1.M[ctx.g], lp1.N[ctx.g],
                      bits=16)  # 一次 requant → hidden 细格（契约 §4 v1.1）
        n2 = self._gn(r + ".norm2", hid, ctx)
        h2 = self._silu(info.norm2, info.conv2_in, n2, ctx)
        # residual 操作数 requant 到块出格、INT16 饱和，相加后截 INT8（§4）
        main = self._conv_layer(r + ".conv2", h2, ctx, pad=1, out_bits=16,
                                acc_bits=48)
        sc_name = r + ".conv_shortcut"
        if sc_name in self.t.layers:
            sc = self._conv_layer(sc_name, x_cat, ctx, out_bits=16)
        else:
            sc = self._point_codes(x_cat, info.block_in, info.block_out, ctx,
                                   wide=True)
        out = sat_int(sc + main, 8)
        if trace is not None:
            trace[r] = out
        if trace_in is not None:
            trace_in[r] = x_cat
        return out

    def _attention(self, a: str, x_in: np.ndarray, ctx: _StepCtx,
                   trace: dict | None, trace_in: dict | None) -> np.ndarray:
        info = self.g.attentions[a]
        n = self._gn(a + ".group_norm", x_in, ctx)
        c, h, w = n.shape
        q = self._linear_layer(a + ".to_q", n.reshape(c, h * w).T, ctx)  # (HW,C)
        k = self._linear_layer(a + ".to_k", n.reshape(c, h * w).T, ctx)
        v = self._linear_layer(a + ".to_v", n.reshape(c, h * w).T, ctx)
        scores = q @ k.T                     # INT64 容器（INT32 语义）
        p = softmax_uint8(scores, ctx.kexp[a], self.t.attn_Qe, _exp_lut())
        av = requant(p.astype(np.int64) @ v,  # UINT8×INT8 → INT32，无偏置补偿
                     self.t.layers[a + ".av_requant"].M[ctx.g],
                     self.t.layers[a + ".av_requant"].N[ctx.g],
                     bits=16)  # 1/256 已折入；av 细格（§4 v1.1）
        o = self._linear_layer(a + ".to_out.0", av, ctx, out_bits=16,
                                   acc_bits=48)
        o_sp = np.ascontiguousarray(o.T).reshape(o.shape[1], h, w)
        res = self._point_codes(x_in, info.block_in, info.block_out, ctx,
                                wide=True)
        out = sat_int(o_sp + res, 8)
        if trace is not None:
            trace[a] = out
        if trace_in is not None:
            trace_in[a] = x_in
        return out

    def _down_block(self, b: int, x: np.ndarray, ctx: _StepCtx,
                    trace: dict | None, trace_in: dict | None):
        skips = []
        for i in range(self.lpb):
            x = self._resnet(f"down_blocks.{b}.resnets.{i}", x, None, ctx,
                             trace, trace_in)
            a = f"down_blocks.{b}.attentions.{i}"
            if a in self.g.attentions:
                x = self._attention(a, x, ctx, trace, trace_in)
            skips.append(x)  # 每层一个（点注意力后输出），镜像 DownBlock.forward
        if b < self.n_down - 1:
            name = f"down_blocks.{b}.downsamplers.0"
            if trace_in is not None:
                trace_in[name] = x
            x = self._conv_layer(name + ".conv", x, ctx, stride=2,
                                 pad=(0, 1, 0, 1))
            skips.append(x)
            if trace is not None:
                trace[name] = x
        return x, skips

    def _mid_block(self, x: np.ndarray, ctx: _StepCtx, trace: dict | None,
                   trace_in: dict | None) -> np.ndarray:
        x = self._resnet("mid_block.resnets.0", x, None, ctx, trace, trace_in)
        x = self._attention("mid_block.attentions.0", x, ctx, trace, trace_in)
        x = self._resnet("mid_block.resnets.1", x, None, ctx, trace, trace_in)
        if trace_in is not None:
            trace_in["mid_block"] = x
        # 容器级第二次量化（fakequant 对 mid_block 输出再 requant 一次）
        x = self._point_codes(x, "mid_block.resnets.1", "mid_block", ctx)
        if trace is not None:
            trace["mid_block"] = x
        return x

    def _up_block(self, u: int, x: np.ndarray, skips: list, ctx: _StepCtx,
                  trace: dict | None, trace_in: dict | None) -> np.ndarray:
        for i in range(self.lpb + 1):
            r = f"up_blocks.{u}.resnets.{i}"
            x = self._resnet(r, x, skips[i], ctx, trace, trace_in)
            a = f"up_blocks.{u}.attentions.{i}"
            if a in self.g.attentions:
                x = self._attention(a, x, ctx, trace, trace_in)
        if u < self.n_up - 1:
            name = f"up_blocks.{u}.upsamplers.0"
            if trace_in is not None:
                trace_in[name] = x
            x = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)  # nearest ×2
            x = self._conv_layer(name + ".conv", x, ctx, pad=1)
            if trace is not None:
                trace[name] = x
        return x


_RSQRT: tuple | None = None
_EXP: np.ndarray | None = None
_IDENT: dict = {}


def _rsqrt_luts():
    global _RSQRT
    if _RSQRT is None:
        from catdiff.bittrue.primitives import gen_rsqrt_lut
        _RSQRT = gen_rsqrt_lut()
    return _RSQRT


def _exp_lut():
    global _EXP
    if _EXP is None:
        from catdiff.bittrue.primitives import gen_exp_lut
        _EXP = gen_exp_lut()
    return _EXP


def _ident_mn(s_from: float, s_to: float):
    key = (s_from, s_to)
    if key not in _IDENT:
        from catdiff.bittrue.primitives import scales_to_MN
        _IDENT[key] = scales_to_MN(s_from / s_to)
    return _IDENT[key]


def load_bittrue_unet(export_dir: str | Path, tier: str,
                      bt_config: dict, unet_cfg: dict) -> BittrueUNet:
    tables = load_export(export_dir, tier, bt_config, unet_cfg)
    return BittrueUNet(tables, build_graph(unet_cfg))
