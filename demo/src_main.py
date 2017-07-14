import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("darkgrid",{"font.sans-serif":['simhei', 'Arial']})


from pyasset.estimator import Estimator
from pyasset.allocation import Allocation
from pyasset.backtest import Backtest
from pyasset.config import  TRADING_DAYS_A_YEAR
import pyasset.optimizer as opz

# 获取收益率数据
db = pd.HDFStore('DB.h5')
ret = db['asset_ret']
# ret = db['ret_index']
# ret = ret[['Bond', 'Stock']]
ret = ret.dropna()
db.close()

# 估计参数
m_estimator = Estimator(ret)
ret_xp = m_estimator.ewm(halflife=60).mean().shift(1)
ret_xp = (ret_xp+1)**TRADING_DAYS_A_YEAR - 1 # 年化

cov_xp = m_estimator.ewm(halflife=15).cov().shift(1, axis=0)
cov_xp = cov_xp * TRADING_DAYS_A_YEAR # 年化

m_allocation = Allocation(ret_xp, cov_xp, 'A','2004-12-31', '2016-01-09')
m_allocation.get_rebalance_info()

weight_m_fix = m_allocation.get_equal_weight()
weight_m_vol_parity = m_allocation.get_volatility_parity_weight()
weight_m_rp = m_allocation.get_risk_parity_weight()
weight_m_mv = m_allocation.get_mean_variance_weight(target_vol=0.2, bound=[(0,1)]*ret.shape[1])

# 权重画图
weight_m_fix.plot(title='fixed')
weight_m_vol_parity.plot(title='vp')
weight_m_rp.plot(title='rp')
weight_m_mv.plot(title='mv')


# penalty l2 to rp
weight_m_l2 = []
for day in m_allocation.ret_xp_trading.columns:
    xb = weight_m_rp.T[day]
    r = m_allocation.ret_xp_trading[day]
    C = m_allocation.cov_xp_trading[day]
    A = np.diag(np.diag(C))
    weight_m_l2.append(opz.Markovitz_l2_penalty(r, C, xb, A, target_vol=0.2, bound=[(0,1)]*ret.shape[1], lmbd=2))

weight_m_l2 = pd.DataFrame(weight_m_l2, index=m_allocation.ret_xp_trading.columns, columns=m_allocation.ret_xp_trading.index)

weight_m_l2.plot(title='l2')
plt.show()

# # A 为对角阵, 不设置benchnmark
# weight_m_l2_eye = []
# for day in m_allocation.ret_xp_trading.columns:
#     xb = np.array([0,0])
#     r = m_allocation.ret_xp_trading[day]
#     C = m_allocation.cov_xp_trading[day]
#     A = np.eye(2)
#     weight_m_l2_eye.append(opz.Markovitz_l2_penalty(r, C, xb, A, target_vol=0.1, bound=[(0,1)]*2, lmbd=1))
#
# weight_m_l2_eye = pd.DataFrame(weight_m_l2_eye, index=m_allocation.ret_xp_trading.columns, columns=m_allocation.ret_xp_trading.index)

# # 将权重画图
# def plot_weight(ax,weight: pd.DataFrame, title:str, width=20, legend=False, y_lim=1.1):
#     N = len(weight)
#     ind = weight.index
#     p1 = ax.bar(ind, weight.Cash, width=width, color='y')
#     p2 = ax.bar(ind, weight.Stock, width=width, color='r', bottom=weight.Cash)
#     p3 = ax.bar(ind, weight.Bond, width=width, color='b', bottom=weight.Cash + weight.Stock)
#     ax.set_title(title)
#     ax.set_ybound(0, y_lim)
#     if legend:
#         return (p1[0],p2[0],p3[0]), ( 'Money', 'Stock', 'Bond')
#
#
# fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, sharex=True)
# plot_weight(ax0,weight_m_fix, title='Fixed Weight')
# plot_weight(ax1,weight_m_vol_parity, title='Volatility Parity')
# plot_weight(ax2,weight_m_mv, title='Mean-Variance with Control')
# a, b = plot_weight(ax3,weight_m_rp, title='Risk Parity', legend=True)
# fig.legend(a, b, loc=5)
# fig.subplots_adjust(hspace=0.4, right=0.75)
# plt.show()

# 回测
b1 = Backtest(weight_m_fix, ret)
print('fix', b1.analyze())
b1.res_nv.plot(legend=True, label='fixed')

b2 = Backtest(weight_m_vol_parity, ret)
print('vp', b2.analyze())
b2.res_nv.plot(legend=True, label='vp')

b3 = Backtest(weight_m_mv, ret)
print('mv', b3.analyze())
b3.res_nv.plot(legend=True, label='mv')

b4 = Backtest(weight_m_rp, ret)
print('rp', b4.analyze())
b4.res_nv.plot(legend=True, label='rp')

b5 = Backtest(weight_m_l2, ret)
print('l2', b5.analyze())
b5.res_nv.plot(legend=True, label='l2')

# b6 = Backtest(weight_m_l2_eye, ret)
# print('l2_eye', b6.analyze())
# b6.res_nv.plot(legend=True, label='l2_eye')

# print(weight_m_l2_eye)

plt.show()