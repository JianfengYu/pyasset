import backtest as bt
import pandas as pd

from pyasset import allocation as ac


class Allocation:
    """根据收益率和频率确定权重、调仓日"""

    def __init__(self, df_ret: pd.DataFrame, freq: str, start_date: str, end_date: str):
        self.df_ret = df_ret
        self.freq = freq
        self.start = start_date
        self.end = end_date

    def _calculate_expect_mean_cov(self, cash_in=True):
        if cash_in:
            df_ret = self.df_ret
        else:
            df_ret = self.df_ret[['Bond', 'Stock']]

        # rolling计算每一期的方差协方差矩阵与期望收益
        re_forcast, cov_forcast = ac.calculate_mean_varience(df_ret, self.freq)

        re_forcast = re_forcast[self.start:self.end]
        cov_forcast = cov_forcast[self.start:self.end]

        # 确定调仓日
        trading_days = ac.get_trading_days(re_forcast, freq=self.freq)
        re_forcast = re_forcast.T[trading_days]
        cov_forcast = cov_forcast[trading_days]

        return re_forcast, cov_forcast


    def _get_mean_var_control_weight(self, target_vol=0.03):
        """带限制的均值方差，不用调整"""
        re_forcast, cov_forcast = self._calculate_expect_mean_cov(cash_in=True)
        w = ac.decide_weight(re_forcast, cov_forcast, method='mean_var_c', target_vol=target_vol)

        return w

    def _get_risk_parity_weight(self, cash_in=True):

        re_forcast, cov_forcast = self._calculate_expect_mean_cov(cash_in=cash_in)
        w = ac.decide_weight(re_forcast, cov_forcast, method='risk_parity')

        # 如果cash参与了配置
        if cash_in:
            return w
        else:
            w = w.T
            new_w = pd.DataFrame([])
            for date in w.columns:
                if date.year <= 2011:
                    tmp_w = w[date]
                    bond = tmp_w['Bond'] * 0.8
                    stock = tmp_w['Stock'] * 0.8
                    if stock > 0.3:
                        bond = bond * 0.3 / stock
                        stock = 0.3
                    if bond > 0.5:
                        stock = stock * 0.5 / bond
                        bond = 0.5
                    cash = 1 - bond - stock
                    new_w[date] = pd.Series([cash, bond, stock], index=['Cash', 'Bond', 'Stock'])
                elif date.year <=2013:
                    tmp_w = w[date]
                    bond = tmp_w['Bond'] * 0.95
                    stock = tmp_w['Stock'] * 0.95
                    if stock > 0.3:
                        bond = bond * 0.3 / stock
                        stock = 0.3
                    if bond > 0.95:
                        stock = stock * 0.95 / bond
                        bond = 0.95
                    cash = 1 - bond - stock
                    new_w[date] = pd.Series([cash, bond, stock], index=['Cash', 'Bond', 'Stock'])
                else:
                    tmp_w = w[date]
                    bond = tmp_w['Bond']
                    stock = tmp_w['Stock']
                    if bond < 1.35:
                        stock = stock * 1.35 / bond
                        bond = 1.35
                    if stock > 0.3:
                        bond = bond * 0.3 / stock
                        stock = 0.3
                    if bond + stock > 1.35:
                        bond = bond * 1.35 / (bond+stock)
                        stock = stock * 1.35 / (bond+stock)    
                    cash = 1 - bond - stock
                    new_w[date] = pd.Series([cash, bond, stock], index=['Cash', 'Bond', 'Stock'])
            return new_w.T


    def _get_vol_parity_weight(self, cash_in=True):
        re_forcast, cov_forcast = self._calculate_expect_mean_cov(cash_in=cash_in)
        w = ac.decide_weight(re_forcast, cov_forcast, method='volatility_parity')

        # 如果cash参与了配置
        if cash_in:
            return w
        else:
            w = w.T
            new_w = pd.DataFrame([])
            for date in w.columns:
                tmp_w = w[date]
                bond = tmp_w['Bond'] * 0.8
                stock = tmp_w['Stock'] * 0.8
                if stock > 0.3:
                    bond = bond * 0.3 / stock
                    stock = 0.3
                cash = 1 - bond - stock
                new_w[date] = pd.Series([cash, bond, stock], index=['Cash', 'Bond', 'Stock'])
            return new_w.T

    def _get_fix_weight(self):
        re_forcast, cov_forcast = self._calculate_expect_mean_cov(cash_in=True)
        w = re_forcast.T.copy()
        w['Cash'] = 0.2
        w['Bond'] = 0.5
        w['Stock'] = 0.3

        return w

class Backtest:

    def __init__(self, weight:pd.DataFrame, re_quote: pd.DataFrame):
        self._quote = re_quote
        self._weight = weight

    def run(self):
        res = bt.backtest_fund_index(weight=self._weight, quote=self._quote, start_date=None)
        return res


if __name__ == '__main__':
    db = pd.HDFStore('DB.h5')
    ret = db['ret_index']
    index = db['index']
    ret = ret.dropna()

    allocation_m = Allocation(ret, 'M', '2005-12-31', '2016-01-9')

    # weight = allocation_m._get_mean_var_control_weight()
    weight = allocation_m._get_risk_parity_weight(cash_in=False)
    print(weight)
    # backtest = Backtest(weight, ret)
    # print(backtest.run()['net_value'])
    # db.close()

    #