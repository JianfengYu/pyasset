"""
收益和风险的分解，收益特征的分析
"""
import numpy as np
import pandas as pd
import seaborn as sns
from pyasset.backtest import Backtest


class Xray:

    def __init__(self, bcaktest_result: Backtest, strategy_name='strategy'):

        self._s_name = strategy_name

        self._sdt = bcaktest_result._start
        self._edt = bcaktest_result._end

        self._quote = bcaktest_result._quote
        self._rebalance_weight = bcaktest_result._weight

        self._nv = bcaktest_result.res_nv
        self._daily_weight = bcaktest_result.res_weight
        self._backtest_res = bcaktest_result


    def return_analyser(self, start_date=None, end_date=None, plot=False):
        """收益分解"""
        if start_date is None and end_date is None:
            ret_daily = self._nv.pct_change().dropna()
        else:
            ret_daily = self._nv.pct_change().dropna()[start_date:end_date]

        asset_ret = self._quote.reindex(ret_daily.index)
        asset_weight = self._daily_weight.shift(1).reindex(ret_daily.index) # 每日开盘权重
        asset_contribution = ((asset_ret*asset_weight).T / ret_daily * np.log(ret_daily+1)).T
        asset_contribution_total = asset_contribution.sum() / np.log(ret_daily+1).sum()

        asset_contribution_total['residual'] = 1 - asset_contribution_total.sum()

        self.return_contribution = asset_contribution_total

        if plot is True:
            aaa = pd.DataFrame({'Attribution': asset_contribution_total}).reset_index()
            aaa.rename_axis({'index': 'Asset'}, inplace=True, axis=1)
            # sns.distplot(b4.res_nv.pct_change().dropna(), kde=False, bins=50)
            sns.barplot(x='Asset', y='Attribution', data=aaa)
            sns.plt.title(self._s_name)
            sns.plt.show()

        return pd.DataFrame({'Attribution': asset_contribution_total})


    def draw_down_plot(self, legend=True):
        self._backtest_res.daily_dd.plot(title = 'Draw Down', label=self._s_name, legend=legend)




if __name__ == '__main__':
    # 获取收益率数据
    db = pd.HDFStore('..\data\DB.h5')
    print(db.keys())
    ret = db['ret_index']
    # ret = ret[['Bond', 'Stock']]
    ret = ret.dropna()
    db.close()

    from pyasset.estimator import Estimator
    from pyasset.allocation import Allocation
    from pyasset.config import TRADING_DAYS_A_YEAR

    # 估计参数
    m_estimator = Estimator(ret)
    ret_xp = m_estimator.ewm(halflife=60).mean().shift(1)
    ret_xp = (ret_xp + 1) ** TRADING_DAYS_A_YEAR - 1  # 年化

    cov_xp = m_estimator.ewm(halflife=10).cov().shift(1, axis=0)
    cov_xp = cov_xp * TRADING_DAYS_A_YEAR  # 年化

    m_allocation = Allocation(ret_xp, cov_xp, 'M', '2005-12-31', '2016-01-09')
    m_allocation.get_rebalance_info()

    weight_m_rp = m_allocation.get_risk_parity_weight()

    b4 = Backtest(weight_m_rp, ret, start_date='2005-12-31', end_date='2016-01-09', fee_rate=0)
    b4.analyze()

    # ret_daily = b4.res_nv.pct_change()
    # asset_ret = b4._quote.reindex(ret_daily.index)
    # asset_weight = b4.res_weight.shift(1).reindex(ret_daily.index)
    # print(((asset_ret * asset_weight).sum(axis=1) - ret_daily).sum())
    #
    # asset_contribution = ((asset_ret * asset_weight).T / ret_daily * np.log(ret_daily + 1)).T
    # asset_contribution_total = asset_contribution.sum() / (np.log(ret_daily + 1).sum())
    # asset_contribution_total['residual'] = 1 - asset_contribution_total.sum()
    #
    # print(asset_contribution_total)

    x = Xray(b4, strategy_name='test')
    attri = x.return_analyser(start_date='2005-12-31', end_date='2016-01-09', plot=True)