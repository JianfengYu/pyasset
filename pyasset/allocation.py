import numpy as np
import pandas as pd

import pyasset.optimizer as opz


class Allocation:
    """权重类"""

    def __init__(self, ret_xp: pd.DataFrame, cov_xp: pd.Panel,
                 rebalance_freq: str, start_date: str, end_date: str):

        self.ret_xp = ret_xp[start_date:end_date]   # 日频预期收益
        self.cov_xp = cov_xp[start_date:end_date]   # 日频预期的方差协方差矩阵
        self.freq = rebalance_freq                  # 调仓频率
        self.start = start_date                     # 起始日
        self.end = end_date                         # 终止日
        self.ret_xp_trading = None                  # 调仓日观察到的期望收益
        self.cov_xp_trading = None                  # 调仓日观察到的方差协方差矩阵

    def get_rebalance_info(self) -> None:
        """得到可交易日的调仓日信息"""
        rebalance_days = get_trading_days(self.ret_xp, self.freq)

        self.ret_xp_trading = self.ret_xp.T[rebalance_days]  # 注意这里进行了转置，列为日期
        self.cov_xp_trading = self.cov_xp[rebalance_days]

    def get_equal_weight(self) -> pd.DataFrame:
        """
        等权配置

        Returns
        -------
        pd.DataFrame
            权重DataFrame，index为调仓时间

        """
        assert self.ret_xp_trading is not None, "U should get_rebalance_info first!"

        w = 1 / len(self.ret_xp_trading.index)
        weight = pd.DataFrame(np.ones_like(self.ret_xp_trading) * w, index=self.ret_xp_trading.index,
                              columns=self.ret_xp_trading.columns)

        return weight.T

    def get_fixed_weight(self, fixed_weight: list) -> pd.DataFrame:
        """
        固定比例配置

        Parameters
        ----------
        fixed_weight: list
            固定配置比例

        Returns
        -------
        pd.DataFrame
            权重DataFrame，index为调仓时间
        """
        assert self.ret_xp_trading is not None, "U should get_rebalance_info first!"

        f_w = np.array(fixed_weight) / sum(fixed_weight)
        w = pd.DataFrame(f_w, index=self.ret_xp_trading.index, columns=[self.ret_xp_trading.columns[0]])
        weight = w.reindex_like(self.ret_xp_trading)
        weight = weight.T.fillna(method='ffill')

        return weight

    def get_volatility_parity_weight(self) -> pd.DataFrame:
        """
        波动率平价

        Returns
        -------
        pd.DataFrame
            权重DataFrame，index为调仓时间
        """
        assert self.ret_xp_trading is not None, "U should get_rebalance_info first!"

        # r = self.ret_xp_trading
        cov_matrix = self.cov_xp_trading

        date = cov_matrix.items              # 调仓日
        columns = cov_matrix.major_axis      # 资产label
        weight = []                 # 记录权重

        for day in cov_matrix.items:

            # r_day = r[day]        # 预期收益
            cov_day = cov_matrix[day]          # 方差协方差矩阵

            tmp_inv_sigma = 1 / np.sqrt(np.diag(cov_day))
            tmp_weight = tmp_inv_sigma / np.sum(tmp_inv_sigma)
            weight.append(tmp_weight)

        weight = pd.DataFrame.from_records(weight, index=date, columns=columns)
        return weight

    def get_risk_parity_weight(self, tol=1e-6, max_loop=100000) -> pd.DataFrame:
        """
        风险平价权重

        Parameters
        ----------
        tol: float
            迭代误差精度
        max_loop: int
            最大迭代次数

        Returns
        -------
        pd.DataFrame
            权重DataFrame，index为调仓时间

        """
        assert self.ret_xp_trading is not None, "U should get_rebalance_info first!"

        # r = self.ret_xp_trading
        cov_matrix = self.cov_xp_trading

        date = cov_matrix.items             # 调仓日
        columns = cov_matrix.major_axis     # 资产label
        weight = []                         # 记录权重

        for day in cov_matrix.items:
            # r_day = r[day]                # 预期收益
            cov_day = cov_matrix[day]       # 方差协方差矩阵
            tmp_weight = opz.risk_parity_solver(cov_day, tol=tol, max_loop=max_loop)
            weight.append(tmp_weight)

        weight = pd.DataFrame.from_records(weight, index=date, columns=columns)

        return weight

    def get_mean_variance_weight(self,  tau=None, target_vol=0.10, bound=None,
                                 constraints=({'type': 'eq', 'fun': lambda w: sum(w) - 1.0}),
                                 x0=None, tol=1e-10) -> pd.DataFrame:
        """
        均值方差 需要自己给定边界条件

        Parameters
        ----------
        tau: float or None
            风险厌恶系数， 默认为None
        target_vol: float or None
            目标波动率，默认为0.1
        bound: list or None
            边界条件，默认为None，即无边界条件。
        constraints: tuple of dict
            限制条件，默认为
            ({'type': 'eq',
              'fun': lambda w: sum(w) - 1.0})
        x0: None or list
            初始值
        tol: float
            迭代精度

        Returns
        -------
        pd.DataFrame
            权重DataFrame，index为调仓时间
        """
        assert self.ret_xp_trading is not None, "U should get_rebalance_info first!"

        r = self.ret_xp_trading
        cov_matrix = self.cov_xp_trading

        date = cov_matrix.items              # 调仓日
        columns = cov_matrix.major_axis      # 资产label
        weight = []                          # 记录权重

        for day in cov_matrix.items:
            r_day = r[day]                     # 预期收益
            cov_day = cov_matrix[day]          # 方差协方差矩阵

            tmp_weight = opz.Markovitz_solver(r_day, cov_day, tau=tau, target_vol=target_vol,
                                              bound=bound, constrains=constraints, x0=x0, tol=tol)

            weight.append(tmp_weight)

        weight = pd.DataFrame.from_records(weight, index=date, columns=columns)
        return weight


def get_trading_days(df_ret: pd.DataFrame, freq='Q'):
    """
    根据调仓频率获取时期期末的交易日，即给定调仓频率的可交易日最后一天。

    Parameters
    ----------
    df_ret: pd.DataFrame
        行情信息，要求index是日期
    freq: str
        调仓频率，用于resample函数。默认为Q，即季度。

    Returns
    -------
    np.array
        由调仓日组成的np.array

    """
    trading_days = pd.Series(df_ret.index)
    trading_days.index = trading_days
    trading_days = trading_days.resample(freq, label='right', closed='right').last()
    trading_days = trading_days.drop(trading_days.index[-1], axis=0)  # 去除最后一天
    return trading_days.values


# # 老的决定权重的代码
# def decide_weight(re_trading_forcast: pd.DataFrame, cov_trading_forcast: pd.Panel, method: str,
#                   tolerance=1e-10, max_loop=100000, risk_budget=None, tau=None, constraints=None,
#                   target_vol=0.1, x0=None
#                   ):
#     """确定权重"""
#
#     if method == 'equal': # 等权
#         w = 1 / len(re_trading_forcast.index)
#         weight = pd.DataFrame(np.ones_like(re_trading_forcast) * w,
#                          index=re_trading_forcast.index, columns=re_trading_forcast.columns)
#         return weight.T
#
#     else:
#         date = cov_trading_forcast.items                        # 调仓日
#         columns = cov_trading_forcast.major_axis                # 资产label
#         weight = []                                             # 记录权重
#
#         for day in cov_trading_forcast.items:
#
#             r = re_trading_forcast[day]                         # 预期收益
#             C = cov_trading_forcast[day]                        # 方差协方差矩阵
#             bounds = [(0,1) for a in range(len(r))]             # 边界条件
#
#             if method == 'volatility_parity': # 波动率平价
#                 tmp_InvSigma = 1 / np.sqrt(np.diag( C ))
#                 tmp_weight = tmp_InvSigma / np.sum(tmp_InvSigma)
#
#             elif method =='risk_parity':  # 风险平价
#                 tmp_weight = opz.risk_parity_solver(C, tol=tolerance, max_loop=max_loop)
#
#             elif method == 'risk_budget':
#                 tmp_weight = opz.risk_budget_solver(C, risk_budget=risk_budget)
#
#             elif method == 'mean_var_averse':
#                 tmp_weight = opz.Markovitz_solver(r, C, tau=tau, target_vol=target_vol,
#                                                   bound=bounds, constrains=constraints, x0=x0)
#
#             elif method == 'max_sharpe':
#                 tmp_weight = opz.max_sharpe_solver(r, C, bound=bounds)
#
#             elif method == 'mean_var_vol':
#                 tmp_weight = opz.Markovitz_mu_solver(
#                     r, C, target_vol=target_vol, bound=bounds, constrains=constraints)
#             elif method == 'mv_old':
#                 tmp_weight = opz.Markovitz(r, C, target_vol=target_vol)
#
#             else:
#                 raise Exception('The method is not supported!')
#
#             weight.append(tmp_weight)
#
#         weight = pd.DataFrame.from_records(weight, index=date, columns=columns)
#         return weight


if __name__ == "__main__":
    db = pd.HDFStore('..\data\DB.h5')
    ret = db['ret_index']
    ret = ret.dropna()
    db.close()

    trade_days = get_trading_days(ret, freq='Q')
    ret_exp = ret.ewm(halflife=30).mean().shift(1)
    cov_exp = ret.ewm(halflife=30).cov().shift(1, axis=0)

    # print(np.sqrt(np.dot(np.dot(re_xp, np.linalg.inv(cov_xp)), re_xp)) / 0.01)

    # print(re_xp.shape)
    #
    a = Allocation(ret_exp, cov_exp, 'M', start_date='2005-12-31', end_date='2016-01-09')
    a.get_rebalance_info()

    # bnds = [(0, 1)] * 3
    # print(a.get_mean_variance_weight(target_vol=0.1, bound=bnds))

    print(a.get_fixed_weight([2, 0.4, 0.4]))

    # w = []
    # res_rp = a.get_risk_parity_weight()
    #
    # for day in res_rp.index:
    #     xb = res_rp.T[day]
    #     r = a.ret_xp_trading[day]
    #     C = a.cov_xp_trading[day]
    #     A = np.diag(np.diag(C))
    #     w.append(opz.Markovitz_l2_penalty(r, C, xb, A, target_vol=0.1, bound=bnds, lmbd=1))
    #
    # w = pd.DataFrame(w, index=res_rp.index, columns=res_rp.columns)
    # print(w)
