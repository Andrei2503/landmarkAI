import pandas as pd
import os

stats = []
for class_dir in os.listdir("raw_dataset"):
    count = len(os.listdir(f"raw_dataset/{class_dir}"))
    stats.append({"Class": class_dir, "Images": count})

df = pd.DataFrame(stats)
print(df)