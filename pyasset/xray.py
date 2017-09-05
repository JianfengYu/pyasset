"""
收益和风险的分解，收益特征的分析
"""
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import matplotlib.pyplot as plt
from pyasset.backtest import Backtest
import pyasset.analyser as als

# sns.set_style("darkgrid", {"axes.facecolor": ".9"})

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']#指定默认字体
mpl.rcParams['axes.unicode_minus'] =False # 解决保存图像是负号'-'显示为方块的问题

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

    def run(self):
        """进行所有的分析"""
        self.draw_rebalance_weight()
        print(self.return_analyser(plot=True))
        self.draw_return_distribution()
        self.draw_down_plot()
        self.draw_daily_weight()

    def draw_return_distribution(self, color='blue'):
        """每日收益分布作图"""
        daily_ret = self._backtest_res.res_nv.pct_change().dropna()
        sns.distplot(daily_ret, kde=False, fit=stats.norm, color=color)

        loc, _ = plt.yticks()
        plt.yticks(loc, ['{0:.2f}%'.format(a*100) for a in np.array(loc) / len(daily_ret)])
        plt.title(self._s_name + ' 每日收益分布')
        plt.show()

    def return_analyser(self, start_date=None, end_date=None, plot=False,
                        cmap=sns.color_palette("Paired", 12)): # sns.color_palette("Paired") "RdGy", 6
        """收益分解"""
        if start_date is None and end_date is None:
            ret_daily = self._nv.pct_change().dropna()
        else:
            ret_daily = self._nv.pct_change().dropna()[start_date:end_date]

        asset_ret = self._quote.reindex(ret_daily.index)                     # 资产收益

        # 归因
        # 计算每日开盘权重，需要在调仓日进行修正
        asset_weight = self._daily_weight.shift(1).reindex(ret_daily.index).T  # 前一日收盘权重
        rebalance_weight = self._rebalance_weight.T
        for rebalance_date in rebalance_weight.columns:
            asset_weight[rebalance_date] = rebalance_weight[rebalance_date]
        asset_weight = asset_weight.T

        # 每日的log return分解
        asset_contribution = ((asset_ret*asset_weight).T / ret_daily * np.log(ret_daily+1).T).T

        asset_contribution_total = asset_contribution.sum() / (np.log(ret_daily+1).sum())

        asset_contribution_total['交易损耗'] = 1 - asset_contribution_total.sum()

        self.return_contribution = asset_contribution_total

        # 业绩归因柱状图
        if plot is True:
            aaa = pd.DataFrame({'业绩归因': asset_contribution_total}).reset_index()
            aaa.rename_axis({'index': '资产'}, inplace=True, axis=1)
            # sns.distplot(b4.res_nv.pct_change().dropna(), kde=False, bins=50)
            sns.plt.figure(figsize=(10,8))
            g = sns.barplot(x='资产', y='业绩归因', data=aaa, palette=cmap)

            sns.plt.title(self._s_name + ' 业绩归因')

            loc, _ = sns.plt.yticks()
            sns.plt.yticks(loc, ['{0:.1f}%'.format(a*100) for a in loc])

            sns.plt.xticks(rotation=-30)

            # 增加annotate
            def annotateBars(row, ax=g):
                for p in ax.patches:
                    # print(p.get_x(), p.get_y(), p.get_width(), p.get_height())
                    sign = 1 if p.get_y() >= 0 else -1
                    large0 = p.get_height() if p.get_height() >=0 else 0
                    # print(p.get_height(), large0)
                    ax.annotate('{0:.1f}%'.format(100 * sign * p.get_height()),
                                (p.get_x() + p.get_width() / 2., large0),
                                ha='center', va='center', fontsize=11, color='gray', rotation=30, xytext=(0, 15),
                                textcoords='offset points')
            aaa.apply(annotateBars, ax=g, axis=1)

            sns.plt.ylabel('收益占比')
            # sns.plt.rcParams['image.cmap'] = 'Paired'

            sns.plt.show()

            # asset_contribution_total = pd.DataFrame({'业绩归因': asset_contribution_total})
            # asset_contribution_total.plot(title=self._s_name + ' 业绩归因', kind = 'bar', alpha=0.9,
            #                               colormap="Paired"
            #                               )
            # plt.show()

        return asset_contribution_total

    def draw_down_plot(self, legend=True, figure_size=(12, 6), color='Green'):  # color = Green
        """绘制回撤分析图"""

        # 回撤分析图
        self._backtest_res.daily_dd.plot(kind='area', title=self._s_name+' 回撤分析', label=self._s_name, legend=legend,
                                         alpha=0.6, color='Gray', ylim=(self._backtest_res.daily_dd.min()-0.1, 0),
                                         figsize=figure_size)
        mdd_sdt = self._backtest_res.analyze_result['Mdd_start']
        mdd_edt = self._backtest_res.analyze_result['Mdd_end']
        mdd_range = self._backtest_res.analyze_result['Mdd_range']

        # 最大回撤区间
        self._backtest_res.daily_dd[mdd_sdt:mdd_edt].plot(kind='area', label='最大回撤区间({0}日)'.format(mdd_range),
                                                          legend=legend, alpha=0.8, color=color)

        locs, labels = plt.yticks()
        plt.yticks(locs, ['{0:.0f}%'.format(a * 100) for a in locs])

        plt.legend(loc=3)
        plt.show()

    def draw_rebalance_weight(self, figure_size=(12, 6),
                              cmap=sns.color_palette("Paired", 12)):  # sns.color_palette("Paired") "RdGy", 6
        """绘制调仓的权重图"""

        weight = self._rebalance_weight.copy()
        weight.index = [pd.datetime.strftime(a, '%Y-%m') for a in weight.index]

        weight.plot(title=self._s_name + ' 调仓权重', kind='bar', stacked=True, figsize=figure_size, alpha=0.9,
                    color=cmap)

        step = len(weight.index) // 8
        plt.xticks(range(0, len(weight.index), step), [weight.index[i] for i in range(0, len(weight.index), step)])

        loc, _ = plt.yticks()
        plt.yticks(loc, ['{0:.0f}%'.format(a*100) for a in loc])

        plt.legend(bbox_to_anchor=(1.05, 0.5), loc=2, borderaxespad=0.) # put legend on the right
        plt.show()

    def draw_daily_weight(self, figure_size=(12, 6),
                          cmap=sns.color_palette("Paired", 12)): # ns.diverging_palette(255, 133, l=60, n=7, center="dark")
        """绘制每日权重图"""

        weight = self._daily_weight.copy()
        weight.index = [pd.datetime.strftime(a, '%Y-%m') for a in weight.index]
        weight.plot(title=self._s_name + ' 每日权重', kind='area', stacked=True, figsize=figure_size, alpha=0.9,
                    color=cmap)

        step = len(weight.index) // 8
        plt.xticks(range(0, len(weight.index), step), [weight.index[i] for i in range(0, len(weight.index), step)])
        plt.yticks(np.arange(0,1.1,0.2), ['{0:.0f}%'.format(a*100) for a in np.arange(0,1.1,0.2)])

        plt.legend(bbox_to_anchor=(1.05, 0.5), loc=2, borderaxespad=0.)
        plt.show()


def analyze(res_nv, rf=None, res_turnover=None, freq='D'):
    """对回测结果进行分析"""

    annal_ret = als.cal_annal_return(res_nv, freq=freq)
    annal_vol = als.cal_annal_volatility(res_nv, freq=freq)
    max_dd, mdd_sdt, mdd_edt, mdd_range, daily_dd = als.cal_max_drawdown_info(res_nv)
    sharpe = als.cal_sharpe(res_nv, rf=rf, freq=freq)
    if res_turnover is None:
        to_average = .0
    else:
        to_average = res_turnover.mean()
    max_wait_days = als.cal_max_wait_periods(res_nv)
    IR = als.cal_information_ratio(res_nv)
    
    analyze_result ={
            "Annal ret": annal_ret, "Annal vol": annal_vol, "Max Drawdown": max_dd, "Sharpe": sharpe,
            "Average turnover": to_average, "Mdd_start": mdd_sdt, "Mdd_end": mdd_edt, "Mdd_range": mdd_range,
            "Max_wait": max_wait_days, 'Information Ratio': IR
        }

    res_report = {
            "Annal ret" : '{0:.2%}'.format(annal_ret), "Annal vol": '{0:.2%}'.format(annal_vol),
            "Max Drawdown": '{0:.2%}'.format(max_dd), "Sharpe": '{0:.2}'.format(sharpe),
            "Average turnover": '{0:.2%}'.format(to_average), "Mdd_start": mdd_sdt, "Mdd_end": mdd_edt,
            "Mdd_range": '{0} Days'.format(mdd_range), "Max_wait": '{0} Days'.format(max_wait_days),
            "Information Ratio": '{0:.2}'.format(IR)
        }
    return pd.DataFrame(res_report, index=['value']).T.sort_index()



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

    m_allocation = Allocation(ret_xp, cov_xp, 'M', '2008-12-31', '2016-01-09')
    m_allocation.get_rebalance_info()

    weight_m = m_allocation.get_equal_weight()

    b4 = Backtest(weight_m, ret, start_date='2005-12-31', end_date='2016-01-09', fee_rate=0)
    b4.analyze()

    # ret_daily = b4.res_nv.pct_change()
    # asset_ret = b4._quote.reindex(ret_daily.index)
    # asset_weight = b4.res_weight.shift(1).reindex(ret_daily.index)
    # print((((1+asset_ret) * asset_weight).sum(axis=1) - ret_daily-1).sum())
    #
    # asset_contribution = ((asset_ret * asset_weight).T / ret_daily * np.log(ret_daily + 1)).T
    # asset_contribution_total = asset_contribution.sum() / (np.log(ret_daily + 1).sum())
    # asset_contribution_total['residual'] = 1 - asset_contribution_total.sum()
    #
    # print(asset_contribution_total)

    x = Xray(b4, strategy_name='测试')
    # attri = x.return_analyser(start_date='2005-12-31', end_date='2016-01-09', plot=True)
    # print(attri)
    # x.draw_down_plot()
    # x.draw_rebalance_weight()
    # x.draw_daily_weight()
    x.run()