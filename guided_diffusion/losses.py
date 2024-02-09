"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np

import torch as th


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        #  logvar2 - logvar1 表示第二项，是两个高斯分布的对数方差之差。
        + logvar2
        - logvar1
        # th.exp(logvar1 - logvar2) 是KL散度公式的第三项，表示对数方差之差的指数。
        + th.exp(logvar1 - logvar2)
        # ((mean1 - mean2) ** 2) * th.exp(-logvar2) 是KL散度公式的最后一项，表示均值之差的平方乘以对数方差之差的指数。
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


# approx_standard_normal_cdf的函数，用于近似计算标准正态分布的累积分布函数（CDF）。标准正态分布的CDF表示给定值在标准正态分布中小于或等于该值的概率。
def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    # 函数计算 np.sqrt(2.0 / np.pi)，这是一个常数，表示标准正态分布的标准差的倒数。
    # 函数计算 (x + 0.044715 * th.pow(x, 3))，这一部分对输入值 x 进行一些变换。
    # 函数通过 th.tanh() 函数计算变换后的值的双曲正切。
    # 函数计算 0.5 * (1.0 + ...)，将双曲正切的结果映射到 [0, 1] 的范围，以近似表示标准正态分布的CDF。
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


# 计算高斯分布离散到给定图像的对数似然。
def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    # centered_x，即目标图像 x 减去均值 means，表示图像与均值的中心化差异。
    centered_x = x - means
    # 计算 inv_stdv，表示对数标准差 log_scales 的指数形式，用于表示标准差的倒数。
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs
