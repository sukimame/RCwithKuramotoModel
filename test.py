import numpy as np
import pandas as pd

#行の表示数の上限を撤廃
pd.set_option('display.max_rows', None)

#列の表示数の上限を撤廃
pd.set_option('display.max_columns', None)

arr = np.load("expData/differentK_June9_delay_omega_sdlog.npy")
df = pd.DataFrame(arr,
                  columns = ["delay", "k_scale","sd","NRMSE", "median", "R1"],
                  index = [i for i in range(1760)])

print(df)

df.to_csv('to_csv_out_June9_1.csv')