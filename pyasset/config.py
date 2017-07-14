import pandas as pd
from sqlalchemy import create_engine

# 连接engine
zyyx_conn = create_engine("mysql+pymysql://simu_sywg:simu_sywg123@106.75.45.237:15077/CUS_FUND_DB?charset=gbk")
sws_conn = create_engine("mssql+pymssql://pu_zhulan:pu_zhulan@192.30.1.40:1433/StructuredFund?charset=utf8")

# 计算年化所用常数
TRADING_DAYS_A_YEAR = 252
TRADING_WEEKS_A_YEAR = 52
TRADING_MONTHS_A_YEAR = 12


if __name__ == '__main__':

    # 检查数据库中是否有该基金经理
    df = pd.read_excel('新财富中国最佳私募证券投资经理_匹配后.xlsx')
    df.columns = ['manager', 'strategy', 'year', 'company', 'city']

    for item in df.itertuples():
        sql = """
        select * from v_fund_manager where user_name = '{0}' and org_name = '{1}'
        """.format(item.manager, item.company)
        print(item)
        print(pd.read_sql(sql, con=sws_conn))