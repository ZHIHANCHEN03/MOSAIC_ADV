## Colab 一键运行（Upload adv_work -> Run Script）

### 0) Colab 环境准备
- **Runtime**: 选择 GPU (T4/L4/A100)
- **Secrets**: 如果需要 Gemini 生成 Layout，请在 Colab 左侧钥匙图标添加 `GOOGLE_API_KEY`。

---

### Cell 1: 初始化环境 (Clone & Install)
```bash
%%bash
set -euo pipefail

# 1. Clone Repo (如果不存在)
if [ ! -d "MOSAIC_ADV" ]; then
  git clone https://github.com/ZHIHANCHEN03/MOSAIC_ADV.git MOSAIC_ADV
fi

# 2. Install Dependencies
cd MOSAIC_ADV
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
python3 -m pip install google-generativeai python-dotenv
```

---

### Cell 2: 上传 `adv_work` 文件夹 (Critical Step)
**请手动操作**：
1. 在本地电脑找到你的 `adv_work` 文件夹。
2. 在 Colab 左侧文件栏，打开 `MOSAIC_ADV` 文件夹。
3. 将本地的 `adv_work` 文件夹直接拖拽到 Colab 的 `MOSAIC_ADV` 文件夹内。
   *(确保路径结构为 `MOSAIC_ADV/adv_work/final_work/...`)*

或者运行以下代码确认文件夹是否就位：
```python
import os
if os.path.exists("MOSAIC_ADV/adv_work/final_work/run_experiment.sh"):
    print("✅ adv_work folder detected!")
else:
    print("❌ adv_work folder NOT found. Please upload it to MOSAIC_ADV/ first.")
```

---

### Cell 3: 运行实验脚本 (Run Experiment)
直接调用你上传的脚本，无需重复编写代码。

```bash
%%bash
set -euo pipefail

# 进入项目根目录
if [ -d "MOSAIC_ADV" ]; then cd MOSAIC_ADV; fi

# 赋予脚本执行权限
chmod +x adv_work/final_work/run_experiment.sh

# 运行一键脚本
# 它会自动调用: generate_test_cases.py -> inference_baseline.py -> inference_masked.py -> eval.py
bash adv_work/final_work/run_experiment.sh
```

---

### Cell 4: 查看结果
```python
import glob, os, json
from PIL import Image
from IPython.display import display

def show_results(dirpath, title):
    print(f"\n== {title} ==")
    if not os.path.exists(dirpath): return
    files = sorted(glob.glob(os.path.join(dirpath, "*_cfg_*.jpg")))
    # Filter out comparison images
    files = [f for f in files if "compared" not in f]
    
    if not files:
        print("No images found.")
        return
        
    for f in files[:3]: 
        print(f)
        display(Image.open(f).resize((300, 300)))

# 显示图片
show_results("MOSAIC_ADV/outputs/baseline", "Baseline")
show_results("MOSAIC_ADV/outputs/ours", "Ours")

# 显示量化指标
print("\n== Metrics ==")
for p in ["MOSAIC_ADV/outputs/baseline/evaluation_results.json", "MOSAIC_ADV/outputs/ours/evaluation_results.json"]:
    if os.path.exists(p):
        with open(p) as f:
            print(f"{os.path.basename(os.path.dirname(p))}:", json.load(f).get("summary"))
```
