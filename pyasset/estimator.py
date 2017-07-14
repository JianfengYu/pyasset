"""""""""""“”“”
" 模块由于参数估计
"""""""""""""""

# TODO: 模块未完成，不一定使用

import numpy as np
import pandas as pd

class Estimator:

    def __init__(self, df_ret: pd.DataFrame):
        self.df_re = df_ret

    def ewm(self, com=None, span=None, halflife=None, alpha=None, min_periods=0, freq=None, adjust=True,
            ignore_na=False, axis=0):
        """pd.ewm方法"""
        return self.df_re.ewm(com=com, span=span, halflife=halflife, alpha=alpha, min_periods=min_periods, freq=freq,
                              adjust=adjust, ignore_na=ignore_na, axis=axis)


if __name__ == "__main__":

    db = pd.HDFStore('DB.h5')
    ret = db['ret_index']
    ret = ret.dropna()
    db.close()

    # print(ret)
    a = Estimator(ret)
    # 用pandas DataFrame的ewm方法计算ema
    print(
        a.ewm(halflife=10).mean()
    )
