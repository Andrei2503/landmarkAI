import pandas as pd
import os

stats = []
for split in ["train", "val", "test"]:
    for class_name in os.listdir(f"learn/final_dataset/{split}"):
        count = len(os.listdir(f"learn/final_dataset/{split}/{class_name}"))
        stats.append({"Split": split, "Class": class_name, "Count": count})

df = pd.DataFrame(stats)
print(df.pivot_table(index="Class", columns="Split", values="Count", aggfunc="sum"))