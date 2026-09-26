"""规范化接线图：量化点清单与每个 conv/linear/GN 的输入输出点。

导出器（export_requant.py，算 M/N 与内部点标定）与位真加载器（loader.py，
只消费导出包）共用本图——它是 docs/bittrue-spec.md §4/§7 的可执行形式。
结构规则镜像 src/catdiff/model/unet.py（冻结包，仅读取），由 unet config 参数化，
微网测试与小导出包同样适用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# requant_params 层记录种类（契约 §7）
KIND_CONV = 0   # conv/linear：逐通道 M/N
KIND_GN = 1     # GN 输出 requant：逐通道带符号 Gc/Nc + Bq
KIND_AV = 2     # 注意力 matmul 输出 requant：标量 M/N（ratio 含 1/256）

# 点种类
P_TABLED = 0    # v3 契约 51 处（act_scales_<steps>.json）
P_INTERNAL = 1  # F5 内部点（requant_params）
P_X = 2         # DDIM 状态边界（s_x = 2^-12 契约常量）


@dataclass
class LayerRef:
    name: str
    in_point: str
    out_point: str
    kind: int = KIND_CONV
    pre: float = 1.0


@dataclass
class GNRef:
    name: str            # GN 模块名（= norm_params.bin 记录名）
    in_point: str
    out_point: str       # 内部点（GN 输出 requant 目标）


@dataclass
class ResnetInfo:
    name: str
    block_in: str        # 块输入点（shortcut 支路输入）
    block_out: str       # 块输出表定点
    conv1_in: str        # silu1 点
    hidden: str
    conv2_in: str        # silu2 点
    norm1: str
    norm2: str
    concat: tuple[str, str] | None = None   # (x 来源点, skip 来源点)，up 路才有
    has_shortcut: bool = False
    film_idx: int = -1


@dataclass
class AttnInfo:
    name: str
    block_in: str
    block_out: str
    gn: str
    qkv: str
    av: str


@dataclass
class Graph:
    tabled_points: list[str] = field(default_factory=list)
    internal_points: list[str] = field(default_factory=list)
    layers: dict[str, LayerRef] = field(default_factory=dict)   # wired conv/linear
    gn_layers: dict[str, GNRef] = field(default_factory=dict)
    resnets: dict[str, ResnetInfo] = field(default_factory=dict)
    attentions: dict[str, AttnInfo] = field(default_factory=dict)
    # 标定观察 hook：(module_name, "fwd"|"pre", point_name)
    observers: list[tuple[str, str, str]] = field(default_factory=list)
    downsample_convs: dict[str, LayerRef] = field(default_factory=dict)
    upsample_convs: dict[str, LayerRef] = field(default_factory=dict)
    conv_out_layer: LayerRef | None = None

    def all_layers(self) -> dict[str, LayerRef]:
        """导出段全部层记录（wired conv/linear + av_requant），确定性排序。"""
        merged = dict(self.layers)
        merged.update(self.downsample_convs)
        merged.update(self.upsample_convs)
        if self.conv_out_layer is not None:
            merged[self.conv_out_layer.name] = self.conv_out_layer
        return dict(sorted(merged.items()))


def _with_attention(block_types: list[str], i: int) -> bool:
    return "Attn" in block_types[i]


def build_graph(config: dict) -> Graph:
    chs = config["block_out_channels"]
    lpb = config["layers_per_block"]
    n_down = len(config["down_block_types"])
    n_up = len(config["up_block_types"])
    down_types = list(config["down_block_types"])
    up_types = list(config["up_block_types"])

    g = Graph()

    # ---- 表定点（镜像 model/trace.py _DEFAULT_PATTERN 捕获面，共 51 处）----
    tabled = ["conv_in"]
    for b in range(n_down):
        for i in range(lpb):
            tabled.append(f"down_blocks.{b}.resnets.{i}")
            if _with_attention(down_types, b):
                tabled.append(f"down_blocks.{b}.attentions.{i}")
        if b < n_down - 1:
            tabled.append(f"down_blocks.{b}.downsamplers.0")
    tabled += ["mid_block", "mid_block.resnets.0", "mid_block.attentions.0",
               "mid_block.resnets.1"]
    for u in range(n_up):
        for i in range(lpb + 1):
            tabled.append(f"up_blocks.{u}.resnets.{i}")
            if _with_attention(up_types, u):
                tabled.append(f"up_blocks.{u}.attentions.{i}")
        if u < n_up - 1:
            tabled.append(f"up_blocks.{u}.upsamplers.0")
    tabled.append("conv_out")
    g.tabled_points = tabled
    tabled_set = set(tabled)

    def internal(name: str) -> str:
        if name in g.internal_points or name in tabled_set:
            raise ValueError(f"量化点重名: {name}")
        g.internal_points.append(name)
        return name

    # ---- resnet 内部点与 GN ----
    def down_block_in(b: int, i: int) -> str:
        if b == 0 and i == 0:
            return "conv_in"
        if i == 0:
            return f"down_blocks.{b - 1}.downsamplers.0"
        prev = f"down_blocks.{b}.resnets.{i - 1}"
        if _with_attention(down_types, b):
            prev = f"down_blocks.{b}.attentions.{i - 1}"
        return prev

    for b in range(n_down):
        for i in range(lpb):
            r = f"down_blocks.{b}.resnets.{i}"
            n1 = internal(f"{r}.norm1")
            s1 = internal(f"{r}.silu1")
            hid = internal(f"{r}.hidden")
            n2 = internal(f"{r}.norm2")
            s2 = internal(f"{r}.silu2")
            bi = down_block_in(b, i)
            g.resnets[r] = ResnetInfo(
                name=r, block_in=bi, block_out=r,
                conv1_in=s1, hidden=hid, conv2_in=s2, norm1=n1, norm2=n2,
                has_shortcut=True,  # 恒等支路也存在（scale 对齐 requant）
                )
            g.layers[f"{r}.conv1"] = LayerRef(f"{r}.conv1", s1, hid)
            g.layers[f"{r}.conv2"] = LayerRef(f"{r}.conv2", s2, r)
            if b > 0 and i == 0 and chs[b - 1] != chs[b]:
                g.layers[f"{r}.conv_shortcut"] = LayerRef(
                    f"{r}.conv_shortcut", bi, r)
            g.gn_layers[n1] = GNRef(n1, bi, n1)
            g.gn_layers[n2] = GNRef(n2, hid, n2)
            g.observers += [(f"{r}.norm1", "fwd", n1), (f"{r}.conv1", "pre", s1),
                            (f"{r}.norm2", "pre", hid), (f"{r}.norm2", "fwd", n2),
                            (f"{r}.conv2", "pre", s2)]
        if b < n_down - 1:
            last = (f"down_blocks.{b}.attentions.{lpb - 1}"
                    if _with_attention(down_types, b)
                    else f"down_blocks.{b}.resnets.{lpb - 1}")
            ds = f"down_blocks.{b}.downsamplers.0"
            g.downsample_convs[f"{ds}.conv"] = LayerRef(f"{ds}.conv", last, ds)

    # mid 输入 = 最后一个 down block 的末点（末块无 downsampler）
    last_b = n_down - 1
    mid_in = (f"down_blocks.{last_b}.attentions.{lpb - 1}"
              if _with_attention(down_types, last_b)
              else f"down_blocks.{last_b}.resnets.{lpb - 1}")
    for i in range(2):
        r = f"mid_block.resnets.{i}"
        n1 = internal(f"{r}.norm1")
        s1 = internal(f"{r}.silu1")
        hid = internal(f"{r}.hidden")
        n2 = internal(f"{r}.norm2")
        s2 = internal(f"{r}.silu2")
        bi = mid_in if i == 0 else "mid_block.attentions.0"
        g.resnets[r] = ResnetInfo(
            name=r, block_in=bi, block_out=r,
            conv1_in=s1, hidden=hid, conv2_in=s2, norm1=n1, norm2=n2,
            has_shortcut=True)
        g.layers[f"{r}.conv1"] = LayerRef(f"{r}.conv1", s1, hid)
        g.layers[f"{r}.conv2"] = LayerRef(f"{r}.conv2", s2, r)
        g.gn_layers[n1] = GNRef(n1, bi, n1)
        g.gn_layers[n2] = GNRef(n2, hid, n2)
        g.observers += [(f"{r}.norm1", "fwd", n1), (f"{r}.conv1", "pre", s1),
                        (f"{r}.norm2", "pre", hid), (f"{r}.norm2", "fwd", n2),
                        (f"{r}.conv2", "pre", s2)]
    mid_out = "mid_block.resnets.1"

    # ---- up 路：模拟 UNet2D.forward 的 skip 栈确定 concat 配对 ----
    # 注意（镜像 model/blocks.py DownBlock.forward）：每层只追加点注意力后的
    # 最终输出，resnet 与 attention 不各自入栈。
    rchs = list(reversed(chs))
    skips: list[str] = ["conv_in"]
    for b in range(n_down):
        for i in range(lpb):
            if _with_attention(down_types, b):
                skips.append(f"down_blocks.{b}.attentions.{i}")
            else:
                skips.append(f"down_blocks.{b}.resnets.{i}")
        if b < n_down - 1:
            skips.append(f"down_blocks.{b}.downsamplers.0")

    up_block_in = mid_out
    for u in range(n_up):
        take = skips[-(lpb + 1):]
        del skips[-(lpb + 1):]
        consumed = list(reversed(take))
        x_src = up_block_in
        for i in range(lpb + 1):
            r = f"up_blocks.{u}.resnets.{i}"
            cat = internal(f"{r}.concat")
            n1 = internal(f"{r}.norm1")
            s1 = internal(f"{r}.silu1")
            hid = internal(f"{r}.hidden")
            n2 = internal(f"{r}.norm2")
            s2 = internal(f"{r}.silu2")
            skip_src = consumed[i]
            prev_out = rchs[u - 1] if u > 0 else rchs[0]
            cin_skip = rchs[min(u + 1, len(rchs) - 1)]
            skip_ch = cin_skip if i == lpb else rchs[u]
            cin_total = (prev_out if i == 0 else rchs[u]) + skip_ch
            g.resnets[r] = ResnetInfo(
                name=r, block_in=cat, block_out=r,
                conv1_in=s1, hidden=hid, conv2_in=s2, norm1=n1, norm2=n2,
                concat=(x_src, skip_src), has_shortcut=True)
            g.layers[f"{r}.conv1"] = LayerRef(f"{r}.conv1", s1, hid)
            g.layers[f"{r}.conv2"] = LayerRef(f"{r}.conv2", s2, r)
            if cin_total != rchs[u]:
                g.layers[f"{r}.conv_shortcut"] = LayerRef(
                    f"{r}.conv_shortcut", cat, r)
            g.gn_layers[n1] = GNRef(n1, cat, n1)
            g.gn_layers[n2] = GNRef(n2, hid, n2)
            g.observers += [(f"{r}.norm1", "pre", cat), (f"{r}.norm1", "fwd", n1),
                            (f"{r}.conv1", "pre", s1),
                            (f"{r}.norm2", "pre", hid), (f"{r}.norm2", "fwd", n2),
                            (f"{r}.conv2", "pre", s2)]
            nxt = r
            if _with_attention(up_types, u):
                a = f"up_blocks.{u}.attentions.{i}"
                gn = internal(f"{a}.gn")
                qkv = internal(f"{a}.qkv")
                av = internal(f"{a}.av")
                g.attentions[a] = AttnInfo(a, block_in=nxt, block_out=a,
                                           gn=gn, qkv=qkv, av=av)
                g.layers[f"{a}.to_q"] = LayerRef(f"{a}.to_q", gn, qkv)
                g.layers[f"{a}.to_k"] = LayerRef(f"{a}.to_k", gn, qkv)
                g.layers[f"{a}.to_v"] = LayerRef(f"{a}.to_v", gn, qkv)
                g.layers[f"{a}.to_out.0"] = LayerRef(f"{a}.to_out.0", av, a)
                g.layers[f"{a}.av_requant"] = LayerRef(
                    f"{a}.av_requant", qkv, av, kind=KIND_AV, pre=1.0 / 256.0)
                g.gn_layers[f"{a}.group_norm"] = GNRef(f"{a}.group_norm", nxt, gn)
                g.observers += [(f"{a}.group_norm", "fwd", gn),
                                (f"{a}.to_q", "fwd", qkv),
                                (f"{a}.to_out.0", "pre", av)]
                nxt = a
            x_src = nxt
        if u < n_up - 1:
            us = f"up_blocks.{u}.upsamplers.0"
            g.upsample_convs[f"{us}.conv"] = LayerRef(f"{us}.conv", x_src, us)
            up_block_in = us
        else:
            up_block_in = x_src

    # ---- attention（down / mid）----
    for b in range(n_down):
        if not _with_attention(down_types, b):
            continue
        for i in range(lpb):
            a = f"down_blocks.{b}.attentions.{i}"
            rin = f"down_blocks.{b}.resnets.{i}"
            gn = internal(f"{a}.gn")
            qkv = internal(f"{a}.qkv")
            av = internal(f"{a}.av")
            g.attentions[a] = AttnInfo(a, block_in=rin, block_out=a,
                                       gn=gn, qkv=qkv, av=av)
            g.layers[f"{a}.to_q"] = LayerRef(f"{a}.to_q", gn, qkv)
            g.layers[f"{a}.to_k"] = LayerRef(f"{a}.to_k", gn, qkv)
            g.layers[f"{a}.to_v"] = LayerRef(f"{a}.to_v", gn, qkv)
            g.layers[f"{a}.to_out.0"] = LayerRef(f"{a}.to_out.0", av, a)
            g.layers[f"{a}.av_requant"] = LayerRef(
                f"{a}.av_requant", qkv, av, kind=KIND_AV, pre=1.0 / 256.0)
            g.gn_layers[f"{a}.group_norm"] = GNRef(f"{a}.group_norm", rin, gn)
            g.observers += [(f"{a}.group_norm", "fwd", gn),
                            (f"{a}.to_q", "fwd", qkv),
                            (f"{a}.to_out.0", "pre", av)]
    a = "mid_block.attentions.0"
    gn = internal(f"{a}.gn")
    qkv = internal(f"{a}.qkv")
    av = internal(f"{a}.av")
    g.attentions[a] = AttnInfo(a, block_in="mid_block.resnets.0", block_out=a,
                               gn=gn, qkv=qkv, av=av)
    g.layers[f"{a}.to_q"] = LayerRef(f"{a}.to_q", gn, qkv)
    g.layers[f"{a}.to_k"] = LayerRef(f"{a}.to_k", gn, qkv)
    g.layers[f"{a}.to_v"] = LayerRef(f"{a}.to_v", gn, qkv)
    g.layers[f"{a}.to_out.0"] = LayerRef(f"{a}.to_out.0", av, a)
    g.layers[f"{a}.av_requant"] = LayerRef(
        f"{a}.av_requant", qkv, av, kind=KIND_AV, pre=1.0 / 256.0)
    g.gn_layers[f"{a}.group_norm"] = GNRef(f"{a}.group_norm", "mid_block.resnets.0", gn)
    g.observers += [(f"{a}.group_norm", "fwd", gn), (f"{a}.to_q", "fwd", qkv),
                    (f"{a}.to_out.0", "pre", av)]

    # ---- 末端 ----
    last_up = f"up_blocks.{n_up - 1}.resnets.{lpb}"
    gn = internal("conv_norm_out.gn")
    silu_out = internal("conv_norm_out.silu")
    g.gn_layers["conv_norm_out"] = GNRef("conv_norm_out", last_up, gn)
    g.observers += [("conv_norm_out", "fwd", gn), ("conv_out", "pre", silu_out)]
    g.conv_out_layer = LayerRef("conv_out", silu_out, "conv_out")
    g.layers["conv_in"] = LayerRef("conv_in", "@x", "conv_in")
    return g


def is_fine(g: Graph, point: str) -> bool:
    """契约 §4 v1.1：内部点（除 qkv）为细网格（scale/256，INT16 存储）。"""
    return point in g.internal_points and not point.endswith(".qkv")


def silu_lut_specs(g: Graph) -> list[tuple[str, str]]:
    """(输入点, 输出点) 对：全部 SiLU LUT 的生成清单（契约 §5.2）。"""
    specs = set()
    for r in g.resnets.values():
        specs.add((r.norm1, r.conv1_in))
        specs.add((r.norm2, r.conv2_in))
    specs.add(("conv_norm_out.gn", "conv_norm_out.silu"))
    return sorted(specs)
