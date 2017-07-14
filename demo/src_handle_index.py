import pandas as pd

db = pd.HDFStore('DB.h5')

asset_index = pd.read_excel('asset_index.xlsx', sheetname='国内市场')
print(asset_index.head())

asset_name = asset_index.ix[0]
print(asset_name)

asset_index = asset_index.drop([0,1,2], axis=0)
print(asset_index.head())


# asset_index = asset_index.dropna()
asset_index.columns = asset_name.values
asset_index = asset_index.set_index(asset_name.values[0])

asset_ret = asset_index.pct_change()#.dropna()
print(asset_ret.head())

db['asset_ret'] = asset_ret


db.close()

print(asset_ret.corr())