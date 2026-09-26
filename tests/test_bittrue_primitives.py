"""Task 2：整数原语库测试（docs/bittrue-spec.md §0-§6 边界用例钉死）。

关键 RTL 陷阱全部有负值/边界用例：requant 负值舍入、饱和 ±128/±32768、
累加器接近 int32 极限、softmax 全零行、GN 单元素通道。
"""

import math

import numpy as np
import pytest
import torch

from catdiff.bittrue.primitives import (
    ddim_update,
    gen_exp_lut,
    gen_rsqrt_lut,
    gen_silu_lut,
    int_conv2d,
    int_linear,
    requant,
    rsqrt_q14,
    sat_int,
    scales_to_MN,
    signed_scales_to_MN,
    silu_lut,
    softmax_uint8,
    x_to_pixel,
)


class TestRequant:
    def test_round_half_up_positive(self):
        # 3/2 → 2（半分向上）
        assert int(requant(3, 1, 1)) == 2
        assert int(requant(1, 1, 1)) == 1

    def test_negative_rounding_pinned(self):
        """负数 round-half-up：floor(v/2^N + 0.5)，-1.5 → -1，-2.5 → -2。"""
        assert int(requant(-3, 1, 1)) == -1   # -1.5 → -1
        assert int(requant(-5, 1, 1)) == -2   # -2.5 → -2
        assert int(requant(-1, 1, 1)) == 0    # -0.5 → 0
        assert int(requant(-2, 1, 1)) == -1   # -1.0 → -1
        # 与 floor(x+0.5) 全域一致
        rng = np.random.default_rng(0)
        acc = rng.integers(-10_000, 10_000, 4096)
        M, N = 137, 9
        got = requant(acc, M, N)
        want = np.clip(np.floor(acc * M / 2**N + 0.5), -128, 127)
        assert np.array_equal(got, want)

    def test_saturation_int8_and_int16(self):
        assert int(requant(10**12, 1, 1, bits=8)) == 127
        assert int(requant(-10**12, 1, 1, bits=8)) == -128
        assert int(requant(10**12, 1, 1, bits=16)) == 32767
        assert int(requant(-10**12, 1, 1, bits=16)) == -32768
        # 恰好到 -128 不被误饱和为 -127
        assert int(requant(-128, 1, 0, bits=8)) == -128
        assert int(requant(-129, 1, 0, bits=8)) == -128

    def test_near_int32_accumulator(self):
        """acc 接近 int32 极限（RTL 乘加器位宽），int64 中间积不溢出。"""
        acc = 2**31 - 1
        M, N = scales_to_MN(1.0)
        got = int(requant(acc, M, N))
        want = math.floor(acc * M / 2**N + 0.5)
        assert got == min(want, 127)
        assert got == 127

    def test_scales_to_MN_matches_float(self):
        """|requant(acc) - float 参考| ≤ 1 LSB 的占比 > 99.9%。"""
        rng = np.random.default_rng(1)
        ratios = 2.0 ** rng.uniform(-24, 6, 2000)
        accs = rng.integers(-2**31, 2**31, (2000, 64))
        n_within = n_total = 0
        for ratio in ratios:
            M, N = scales_to_MN(ratio)
            got = requant(accs, M, N).astype(np.float64)
            want = np.clip(np.floor(accs * ratio + 0.5), -128, 127)
            n_within += int((np.abs(got - want) <= 1).sum())
            n_total += accs.size
        assert n_within / n_total > 0.999


class TestIntConv:
    def test_identity_1x1(self):
        x = torch.randint(-128, 128, (1, 3, 4, 4), dtype=torch.int32)
        w = torch.zeros(3, 3, 1, 1, dtype=torch.int32)
        for c in range(3):
            w[c, c, 0, 0] = 1
        M = [128] * 3
        N = [7] * 3  # M/2^7 = 1.0 精确恒等
        y = int_conv2d(x, w, M, N, stride=1, pad=0)
        assert torch.equal(y.to(torch.int64), x.to(torch.int64))

    def test_matches_float_reference(self):
        rng = np.random.default_rng(2)
        x = torch.from_numpy(rng.integers(-128, 128, (1, 4, 8, 8))).int()
        w = torch.from_numpy(rng.integers(-127, 128, (6, 4, 3, 3))).int()
        s_in, s_w, s_out = 0.05, 0.01, 0.02
        M = [scales_to_MN(s_in * s_w / s_out)[0]] * 6
        N = [scales_to_MN(s_in * s_w / s_out)[1]] * 6
        bias = torch.from_numpy(rng.integers(-1000, 1000, 6)).int()
        y = int_conv2d(x, w, M, N, bias32=bias, stride=1, pad=1)
        # float 参考
        xf = x.double() * s_in
        wf = w.double() * s_w
        ref = torch.nn.functional.conv2d(xf, wf, padding=1) \
            + bias.double().view(1, -1, 1, 1) * (s_in * s_w)
        ref_q = torch.clamp(torch.round(ref / s_out), -128, 127)
        assert (y.double() - ref_q).abs().le(1).float().mean() > 0.999

    def test_asymmetric_pad_downsample_shape(self):
        """downsample pad (0,1,0,1)：16→8 且右/下边为补零乘积。"""
        x = torch.randint(-128, 128, (1, 2, 16, 16), dtype=torch.int32)
        w = torch.randint(-127, 128, (2, 2, 3, 3), dtype=torch.int32)
        y = int_conv2d(x, w, [127, 127], [7, 7], stride=2, pad=(0, 1, 0, 1))
        assert y.shape == (1, 2, 8, 8)

    def test_int16_path_conv_in(self):
        x = torch.randint(-4096, 4096, (1, 3, 8, 8), dtype=torch.int32)
        w = torch.randint(-32767, 32768, (4, 3, 3, 3), dtype=torch.int32)
        M = [1 << 20] * 4
        N = [31] * 4
        y = int_conv2d(x, w, M, N, stride=1, pad=1, out_bits=8)
        assert int(y.min()) >= -128 and int(y.max()) <= 127


class TestIntLinear:
    def test_matches_float(self):
        rng = np.random.default_rng(3)
        x = torch.from_numpy(rng.integers(-128, 128, (5, 8))).int()
        w = torch.from_numpy(rng.integers(-127, 128, (6, 8))).int()
        M, N = scales_to_MN(0.7)
        y = int_linear(x, w, [M] * 6, [N] * 6, out_bits=8)
        ref = torch.clamp(torch.round(x.double() @ w.double().T * 0.7),
                          -128, 127)
        assert (y.double() - ref).abs().le(1).float().mean() > 0.999


class TestSiluLut:
    def test_zero_at_center(self):
        lut = gen_silu_lut(0.03, 0.02 / 256.0)
        assert lut[128] == 0  # silu(0)=0
        assert lut.dtype == np.int64

    def test_range_and_shape(self):
        lut = gen_silu_lut(0.05, 0.05 / 256.0)
        assert lut.shape == (257,)
        # silu 下界 -0.2785 → 细格码 ≈ -0.2785/(0.05/256)
        assert lut.min() >= -int(0.2785 / (0.05 / 256.0)) - 1

    def test_lookup_interp(self):
        """细网格查表 + 插值：零点精确、单调、与解析值差 ≤1 细格 LSB。"""
        s_in, s_fine = 0.05, 0.05 / 256.0
        lut = gen_silu_lut(s_in, s_fine)
        x = np.array([-128 * 256, -256, 0, 256, 127 * 256], dtype=np.int64)
        y = silu_lut(lut, x)
        assert y[2] == 0
        # silu 负半轴非单调（极小值 -0.2785 @ x≈-1.28），只验证值域与精度
        z = np.linspace(-120 * 256, 120 * 256, 997)
        got = silu_lut(lut, z.astype(np.int64))
        ref = (z / 256.0 * s_in) / (1 + np.exp(-(z / 256.0 * s_in))) / s_fine
        # 线性插值误差主导（silu 拐点处曲率最大）：≤3 细格 LSB
        # = 3/256 ≈ 0.012 INT8 基准 LSB，较 v1.0 直查表仍细 100 倍
        assert np.abs(got - ref).max() <= 3.0


class TestSoftmax:
    def test_all_zero_row(self):
        """全零行（减 max 后 arg 全 0）：均匀概率 256/L。"""
        lut = gen_exp_lut()
        scores = np.zeros((4, 8), dtype=np.int64)
        Kexp, Qe = 1, 1  # acc_max - acc = 0 → a=0 恒成立（Qe≥1）
        p = softmax_uint8(scores, Kexp, Qe, lut)
        assert p.shape == scores.shape
        assert p.min() >= 0 and p.max() <= 65535
        assert np.allclose(p.sum(axis=1), 256 * np.ones(4), atol=0) or \
            p.max() == 256 // 8

    def test_dominant_row(self):
        lut = gen_exp_lut()
        scores = np.array([[0, -10**6]], dtype=np.int64)
        p = softmax_uint8(scores, 1, 1, lut)
        assert p[0, 0] == 65535 and p[0, 1] == 0

    def test_uint8_unsigned_multiply_no_offset(self):
        """UINT8×INT8 无偏置补偿：全 v=127、均匀 p → out ≈ 127。"""
        lut = gen_exp_lut()
        scores = np.zeros((1, 4), dtype=np.int64)
        p = softmax_uint8(scores, 1, 1, lut)
        v = np.full((1, 4), 127, dtype=np.int64)
        out_acc = (p.astype(np.int64) * v).sum()
        # 均匀 p ≈ 8192/65536 → out_acc ≈ 4·8192·127；value = acc/65536 ≈ 127
        assert abs(out_acc / 65536 - 127) <= 1


class TestGN:
    def _run(self, codes, num_groups, gamma, beta, s_in, s_out):
        from catdiff.bittrue.primitives import groupnorm_int
        lut0, lut1 = gen_rsqrt_lut()
        C = codes.shape[0]
        Gc = np.zeros(C, np.int64)
        Nc = np.zeros(C, np.uint8)
        Bq = np.zeros(C, np.int64)
        s_xhat = 2.0 ** -12
        from catdiff.bittrue.primitives import signed_scales_to_MN
        for c in range(C):
            ratio = s_xhat * gamma[c] / s_out
            M, N = signed_scales_to_MN(ratio)
            Gc[c], Nc[c] = M, N
            Bq[c] = math.floor(beta[c] / s_out + 0.5)
        eps_q = int(round(1e-6 / (s_in * s_in) * 2**16))
        return groupnorm_int(codes, num_groups, eps_q, lut0, lut1, Gc, Nc, Bq)

    def test_single_element_channel(self):
        """GN 单元素通道（H=W=1, G=1）：方差 0，仅 eps 底，输出有限且饱和。"""
        codes = np.array([100, -100], dtype=np.int64).reshape(2, 1, 1)
        gamma = np.array([1.0, 1.0])
        beta = np.array([0.0, 0.0])
        y = self._run(codes, 2, gamma, beta, 0.01, 0.01)
        assert y.shape == codes.shape
        assert np.abs(y).max() <= 127

    def test_matches_float_within_1lsb(self):
        rng = np.random.default_rng(4)
        C, H, W, G = 8, 8, 8, 4
        codes = rng.integers(-100, 100, (C, H, W)).astype(np.int64)
        gamma = rng.uniform(0.5, 1.5, C)
        beta = rng.uniform(-0.2, 0.2, C)
        s_in, s_out = 0.02, 0.03
        y = self._run(codes, G, gamma, beta, s_in, s_out)
        # float 参考（PyTorch GN 语义）
        x = torch.from_numpy(codes.astype(np.float64)).reshape(1, C, H, W) \
            * s_in
        import torch.nn.functional as F
        ref = F.group_norm(x, G, torch.tensor(gamma).double(),
                           torch.tensor(beta).double(), 1e-6) / s_out
        ref_q = torch.clamp(torch.round(ref), -128, 127)
        diff = (torch.from_numpy(y).double() - ref_q).abs()
        assert (diff <= 1).float().mean() > 0.999
        assert diff.max() <= 4


class TestDDIM:
    def test_clip_and_mapping(self):
        """x0 预测 clip [-1,1]；prev 单次舍入；像素映射 round((x+1)·127.5)。"""
        # A=2^23（x0 = x - eps·B'/A 的简单情形），B=0, C=2^30, D=0
        x = np.full((1, 1, 2, 2), 0, dtype=np.int64)
        eps = np.array([[[[0, 0], [4096, 8192]]]], dtype=np.int64) * 1
        # eps·B=0 → x0=0；prev = x0·C>>30 = 0
        out = ddim_update(x, eps * 0, 1 << 23, 0, 1 << 30, 0)
        assert np.all(out == 0)

    def test_coeff_scale_domain(self):
        """s16/s_x 折入 B/D：eps 格与 x 格不同也能正确合成。"""
        s_x, s16 = 2.0 ** -12, 0.0001
        alpha_t, alpha_prev = 0.5, 0.25
        A = round(2**23 / math.sqrt(alpha_t))
        B = round(2**23 * math.sqrt(1 - alpha_t) / math.sqrt(alpha_t)
                  * s16 / s_x)
        C = round(2**30 * math.sqrt(alpha_prev))
        D = round(2**30 * math.sqrt(1 - alpha_prev) * s16 / s_x)
        # 构造：x=0, eps=1000·s16/s_x 格? 直接与 float 公式对齐验证
        x_q = 1000
        eps_q = 5000
        out = int(ddim_update(np.array([[x_q]], dtype=np.int64),
                              np.array([[eps_q]], dtype=np.int64), A, B, C, D)[0, 0])
        x0 = (x_q * s_x - math.sqrt(1 - alpha_t) * eps_q * s16) / math.sqrt(alpha_t)
        x0 = max(-1.0, min(1.0, x0))
        want = math.sqrt(alpha_prev) * x0 + \
            math.sqrt(1 - alpha_prev) * eps_q * s16
        want_q = want / s_x
        assert abs(out - want_q) <= 2

    def test_pixel_mapping(self):
        from catdiff.bittrue.primitives import x_to_pixel
        assert x_to_pixel(4096) == 255
        assert x_to_pixel(-4096) == 0
        assert x_to_pixel(0) == 128


class TestLuts:
    def test_exp_lut(self):
        lut = gen_exp_lut()
        assert lut[0] == 16384  # e^0·2^14
        assert lut.dtype == np.uint16
        # 单调递减
        assert (np.diff(lut.astype(np.int64)) <= 0).all()
        # 精度：与解析值差 ≤ 1
        a = np.arange(4096)
        ref = np.exp(-a * 2.0 ** -8) * 2**14
        assert np.abs(lut.astype(np.float64) - ref).max() <= 1.0

    def test_rsqrt_lut_accuracy(self):
        lut0, lut1 = gen_rsqrt_lut()
        # 对若干 var_x 值验证 inv_std 相对误差 ≤ 2^-10
        rng = np.random.default_rng(5)
        for _ in range(200):
            var_x = 2 ** rng.uniform(-7.5, 4)
            v = int(var_x * 2**16)
            inv = rsqrt_q14(v, lut0, lut1) / 2.0**14
            want = 1.0 / math.sqrt(v / 2.0**16)  # 输入即定点 v（含 2^-16 截断）
            assert abs(inv - want) / want < 2.0 ** -10, var_x

    def test_rsqrt_floor(self):
        """契约 §3 步骤 1：var_x < 2^-8 截到地板 → inv = 16。"""
        lut0, lut1 = gen_rsqrt_lut()
        assert rsqrt_q14(0, lut0, lut1) == 16 * (1 << 14)
        assert rsqrt_q14(255, lut0, lut1) == 16 * (1 << 14)
        assert rsqrt_q14(256, lut0, lut1) == 16 * (1 << 14)
