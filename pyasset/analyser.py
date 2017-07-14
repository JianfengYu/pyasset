import numpy as np
import pandas as pd
from scipy.stats import norm

from pyasset.config import TRADING_DAYS_A_YEAR, TRADING_MONTHS_A_YEAR, TRADING_WEEKS_A_YEAR

""""""""
"本模块计算默认输入数据无空值"
""""""""



def check_have_null(net_value: pd.Series) -> bool:
    """
    检查是否有空值
    
    Parameters
    ----------
    net_value: pd.Series
        净值序列

    Returns
    -------
    bool
        是否有空值

    """
    if net_value.isnull().sum() == 0:
        return True
    else:
        return False


def cal_annal_return(net_value: pd.Series, freq='M') -> float:
    """
    计算年化收益率

    Parameters
    ----------
    net_value: pd.Series
        净值序列
    freq: str
        净值序列的观测频率，'M'为月，'W'为周，'D'为天。默认为月。

    Returns
    -------
    float
        年化收益率

    """
    if freq is 'M':
        periods = TRADING_MONTHS_A_YEAR
    elif freq is 'W':
        periods = TRADING_WEEKS_A_YEAR
    elif freq is 'D':
        periods = TRADING_DAYS_A_YEAR
    else:
        raise Exception('The freq type is not suportted!')

    annal_re = pow(net_value.values[-1] / net_value.values[0],
                   periods / (len(net_value) - 1)) - 1

    return annal_re


def cal_annal_volatility(net_value: pd.Series, freq='M') -> float:
    """
    计算年化波动率

    Parameters
    ----------
    net_value: pd.Series
        净值序列
    freq: str
        净值序列观测频率，'M'为月，'W'为周，'D'为天。默认为月。

    Returns
    -------
    float
        年化波动率

    """
    if freq is 'M':
        periods = TRADING_MONTHS_A_YEAR
    elif freq is 'W':
        periods = TRADING_WEEKS_A_YEAR
    elif freq is 'D':
        periods = TRADING_DAYS_A_YEAR
    else:
        raise Exception('The freq type is not suportted!')

    rtns = net_value.pct_change().dropna()
    if len(rtns) <= 1:
        return .0
    vlt = np.sqrt(periods) * rtns.std(ddof=1)
    return vlt


def cal_max_drawdown(net_value: pd.Series) -> float:
    """
    计算最大回撤

    Parameters
    ----------
    net_value
        净值序列

    Returns
    -------
    float
        最大回撤

    """

    # 计算当日之最大的净值
    max_here = net_value.expanding(min_periods=1).max()
    drawdown_here = net_value / max_here - 1

    # 计算最大回撤开始和结束时间
    tmp = drawdown_here.sort_values().head(1)
    max_dd = float(tmp.values)

    return max_dd


def cal_sharpe(net_value: pd.Series, rf=None, freq='M') -> float:
    """
    计算年化Sharpe比率

    Parameters
    ----------
    net_value: pd.Series
        净值序列
    rf: None or pd.Series
        无风险收益序列，必须要和净值序列匹配。默认为None，即无风险收益为0。
    freq: str
        净值序列观测频率，'M'为月，'W'为周，'D'为天。默认为月。

    Returns
    -------
    float
        年化Sharpe比率

    """
    if freq is 'M':
        periods = TRADING_MONTHS_A_YEAR
    elif freq is 'W':
        periods = TRADING_WEEKS_A_YEAR
    elif freq is 'D':
        periods = TRADING_DAYS_A_YEAR
    else:
        raise Exception('The freq type is not suportted!')

    if rf is None:
        re_p = net_value.pct_change().dropna()
    else:
        re_p = (net_value.pct_change - rf).dropna()

    if re_p.std() != 0:
        sharpe = re_p.mean() / re_p.std() * np.sqrt(periods)
    else:
        sharpe = np.sign(re_p.mean()) * np.inf

    return sharpe


def cal_downside_risk(net_value: pd.Series, bench_nv=None, r_min=.0, freq='M') -> float:
    """
    计算年化的下行波动率

    Parameters
    ----------
    net_value: pd.Series
        净值序列
    bench_nv: None or pd.Series
        基准的净值序列，需要与net_value具有相同的index。默认为None，即基准为0。bench_nv和r_min必须制定一个。
    r_min: float
        最小收益值，默认为0。bench_nv和r_min必须制定一个。
    freq: str
        净值序列观测频率，'M'为月，'W'为周，'D'为天。默认为月。

    Returns
    -------
    float
        年化下行波动率

    """
    if freq is 'M':
        periods = TRADING_MONTHS_A_YEAR
    elif freq is 'W':
        periods = TRADING_WEEKS_A_YEAR
    elif freq is 'D':
        periods = TRADING_DAYS_A_YEAR
    else:
        raise Exception('The freq type is not suportted!')

    if bench_nv is not None and r_min != 0:
        raise ValueError("You can only assign one between bench_nv and r_min!")

    r_p = net_value.pct_change().dropna()

    if bench_nv is not None:
        r_b = bench_nv.pct_change().dropna()
        dummy = r_p < r_b
        diff = r_p[dummy] - r_b[dummy]
    else:
        diff = r_p[r_p < r_min]

    if len(diff) <= 1:
        return 0.
    else:
        return np.sqrt((diff*diff).sum() / len(r_p) * periods)


def cal_sortino(net_value: pd.Series, freq='M') -> float:
    """
    计算年化的sortino值
        sortino = 年化收益/年化下行波动率

    Parameters
    ----------
    net_value: pd.Series
        净值序列
    freq: str
        净值序列观测频率，'M'为月，'W'为周，'D'为天。默认为月。

    Returns
    -------
    float
        年化sortino值

    """

    annal_rp = cal_annal_return(net_value, freq=freq)
    downside_risk = cal_downside_risk(net_value, bench_nv=None, freq=freq)

    if downside_risk != 0:
        return annal_rp / downside_risk
    else:
        return np.sign(annal_rp) * np.inf


def cal_calmar(net_value: pd.Series, freq='M') -> float:
    """
    计算年化calmar
        calmar = 年化收益/最大回撤

    Parameters
    ----------
    net_value: pd.Series
        净值序列
    freq: str
        净值序列观测频率，'M'为月，'W'为周，'D'为天。默认为月。

    Returns
    -------
    float
        年化calmar

    """
    maxdd = cal_max_drawdown(net_value)
    annal_ret = cal_annal_return(net_value, freq=freq)

    if maxdd != 0:
        calmar = annal_ret / abs(maxdd)
    else:
        calmar = np.sign(annal_ret) * np.inf

    return calmar


def cal_max_drawdown_info(net_value: pd.Series) -> tuple:
    """
    计算最大回撤的相关信息

    Parameters
    ----------
    net_value: pd.Series
        净值序列

    Returns
    -------
    tuple
        (最大回撤， 起始日， 终止日， 持续时间)

    """

    # assert net_value.count() == len(net_value), 'There are NaN values!'
    # net_value = net_value.dropna()
    # net_value = net_value.sort_index()

    max_here = net_value.expanding(min_periods=1).max()  # 计算当日之前的账户最大价值
    drawdown_here = net_value / max_here - 1  # 计算当日的回撤

    # 计算最大回撤和结束时间
    tmp = drawdown_here.sort_values().head(1)
    max_dd = float(tmp.values)
    end_date = tmp.index.strftime('%Y-%m-%d')[0]

    # 计算开始时间
    tmp = net_value[:end_date]
    tmp = tmp.sort_values(ascending=False).head(1)
    start_date = tmp.index.strftime('%Y-%m-%d')[0]

    # 计算回撤持续时间
    dt_range = len(pd.period_range(start_date, end_date))

    return max_dd, start_date, end_date, dt_range, drawdown_here


def cal_max_wait_periods(net_value: pd.Series) -> int:
    """
    计算再创新高最长等待天数

    Parameters
    ----------
    net_value: pd.Series
        净值序列

    Returns
    -------
    int
        再创新高最长等待天数

    """

    max_here = net_value.expanding(min_periods=1).max()  # 计算当日之前的账户最大值
    new_high_dummy = net_value == max_here  # 得到是否达到最大值的日期dummy
    max_wait_days = np.max(np.bincount(new_high_dummy.cumsum()))

    return int(max_wait_days)


def portfolio_value_at_risk(re: pd.Series, cov: pd.DataFrame, weight: pd.Series, alpha=0.01) -> float:
    """
    计算组合的预期VaR，可认为是极端情况下的最大回撤

    Parameters
    ----------
    re: pd.Series
        收益序列
    cov: pd.DataFrame
        方差协方差矩阵
    weight: pd.Series
        权重序列
    alpha: float
        正态分布假定下的损失发生概率

    Returns
    -------
    float
        组合的预期Value at Risk

    """
    re_p = re.dot(weight)
    std_p = np.sqrt(weight.dot(cov).dot(weight))
    dist = norm.ppf(alpha)

    if re_p + dist * std_p > 0:
        return 0
    else:
        return re_p + dist * std_p


if __name__ == '__main__':
    nv = pd.Series([1, 1.2, 1.4, 1.2, 1.5, 1.2],
                   index=pd.to_datetime([
                       '2012-01-01', '2012-02-01', '2012-03-01', '2012-04-01', '2012-05-01', '2012-06-01'
                   ]))
    print(cal_annal_return(nv))
    print(cal_max_drawdown(nv))
    print(cal_downside_risk(nv))
    print(cal_annal_volatility(nv))
    print(cal_calmar(nv))
    print(cal_sharpe(nv))
    print(cal_sortino(nv))
    print(cal_max_wait_periods(nv))
