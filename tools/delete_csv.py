import os
import glob

for f in glob.glob("data/processed/*.csv"):
    try:
        os.remove(f)
        print("✅ Удалён:", f)
    except Exception as e:
        print("❌ Не удалось удалить:", f, "|", e)
