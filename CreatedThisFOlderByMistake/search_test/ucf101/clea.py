from glob import glob
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
csvs = glob("*.csv")

csv = csvs[0]

df = pd.read_csv(csv, index_col=[0])
for column in ["bbox1", "bbox2", "bbox3", "bbox4"]:
    df[column] = df[column].progress_apply(
        lambda x: float(x.split(",")[0][len("tf.Tensor("):]  )
    )

# Getting the information array from the above
information = []
for fno in df.frame_nos:
    if len(df[df.frame_nos == fno]) > 1:
        print(fno)
    information.append(
        df[df.frame_nos == fno].loc[:, "category":"bbox4"].values
    )