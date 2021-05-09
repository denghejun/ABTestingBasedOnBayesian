import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import beta
from IPython.core.pylabtools import figsize

visitors_to_A = 80
visitors_to_B = 20
conversions_from_A = 60
conversions_from_B = 2

### 目的：找出Testing A 比 Testing B 更better的衡量标准
### MCMC后验采样法

# 要么转化要么不转化，每一个user转化与不转化的概率都是0.5，符合先验分布beta分布 = beat（1，1） = 1/(1+1)
alpha_prior = 1
beta_prior = 1

posterior_A = beta(alpha_prior + conversions_from_A,
                   beta_prior + visitors_to_A - conversions_from_A)

posterior_B = beta(alpha_prior + conversions_from_B,
                   beta_prior + visitors_to_B - conversions_from_B)

samples = 2000
samples_posterior_A = posterior_A.rvs(samples)
samples_posterior_B = posterior_B.rvs(samples)

print(' P(Rate A > Rate B) = ')
# 衡量1：Testing A 比 Testing B better的概率有多大（0到1之间）
print((samples_posterior_A > samples_posterior_B).mean())


# 衡量2：A和B分别的概率密度函数图对比
### pdf 概率密度图 possibility density function = pdf
### x轴代表概率
### Y轴代表某一个概率的密度
### 例如A的最高密度概率为0.8
### 例如B的最高密度概率为0.1，相差不大

figsize(12.5, 4)

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
x = np.linspace(0, 1, 500)
plt.plot(x, posterior_A.pdf(x), label='posterior of A')
plt.plot(x, posterior_B.pdf(x), label='posterior of B')
plt.xlim(0, 1)
plt.xlabel('value')
plt.ylabel('density')
plt.show()
