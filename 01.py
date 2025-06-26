import requests
import pandas as pd

# 定义三个链接（替换为你的真实URL）
url1 = "https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt"  # 链接1的数据（4列数字）
url2 = "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data"  # 链接2的数据（8列数字 + 类别）
url3 = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna"  # 链接3的数据（8列数字 + 类别）


def download_and_save_csv(url, filename):
    try:
        # 1. 下载数据
        response = requests.get(url)
        response.raise_for_status()  # 检查请求是否成功

        # 2. 按行分割数据
        lines = response.text.strip().split('\n')

        # 3. 解析数据（假设每行用空格/制表符分隔）
        data = []
        for line in lines:
            # 移除多余空格并分割
            row = line.strip().split()
            data.append(row)

        # 4. 转换为 DataFrame 并保存为 CSV
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, header=False)  # 不保存索引和列名
        print(f"✅ 成功保存到 {filename}")

    except Exception as e:
        print(f"❌ 下载或保存失败: {e}")


# 下载并保存为 CSV
download_and_save_csv(url1, "Skin_NonSkin_data.csv")  # 保存为 CSV（4列数字）
download_and_save_csv(url2, "ecoli_data.csv")  # 保存为 CSV（8列数字 + 类别）
download_and_save_csv(url3, "cod-rna_data.csv")  # 保存为 CSV（8列数字 + 类别")