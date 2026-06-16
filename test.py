import numpy as np
import pandas as pd

#行の表示数の上限を撤廃
pd.set_option('display.max_rows', None)

#列の表示数の上限を撤廃
pd.set_option('display.max_columns', None)

arr = np.load("expData/differentK_June10_kscale20_delay.npy")
df = pd.DataFrame(arr,
                  columns = ["delay", "k_scale","NRMSE", "median", "R1"],
                  index = [i for i in range(3220)])

print(df)

df.to_csv('csvData/to_csv_out_June10_4.csv')