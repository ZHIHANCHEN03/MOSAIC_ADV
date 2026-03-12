## Colab 一键运行（git clone → 跑 `run_experiment.sh` → 直接看结果）

### 0) Colab 环境要求
- 在 Colab 右上角选择 **Runtime → Change runtime type → GPU**（建议 L4/A100；T4 可能会慢或显存不够）
- 需要能从 HuggingFace 下载模型：`black-forest-labs/FLUX.1-dev` 与 `ByteDance-FanQie/MOSAIC`

---

### Cell 1：Clone repo & Install Dependencies
把下面这一格贴到 Colab 里运行。

```bash
%%bash
set -euo pipefail

# 1. Clone Repo
if [ ! -d "MOSAIC_ADV" ]; then
  git clone https://github.com/ZHIHANCHEN03/MOSAIC_ADV.git MOSAIC_ADV
fi
cd MOSAIC_ADV

# 2. Install Dependencies
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
python3 -m pip install google-generativeai python-dotenv
```

---

### Cell 2（重要）：上传/创建 adv_work 文件夹
**注意**：因为 `adv_work` 是我们新加的文件夹，原始 repo 里没有。你需要把本地生成的 `adv_work` 文件夹上传到 Colab 的 `MOSAIC_ADV` 目录下。
或者，你可以运行下面的代码直接在 Colab 里生成这些文件（如果你不想手动上传）：

```python
import os
os.makedirs("MOSAIC_ADV/adv_work/final_work", exist_ok=True)
os.chdir("MOSAIC_ADV")

# --- 这里填入我们之前生成的 python 脚本内容 ---
# (为了简洁，建议直接把本地 adv_work 拖进 Colab 左侧文件栏的 MOSAIC_ADV 文件夹里)
```

---

### Cell 3（可选）：启用 Gemini 布局
如果你想让 Layout Generator 真调用 Gemini：

```python
import os
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
```

---

### Cell 4：运行实验脚本

```bash
%%bash
set -euo pipefail
cd MOSAIC_ADV

# 赋予执行权限
chmod +x adv_work/final_work/run_experiment.sh

# 运行一键脚本 (它会自动生成数据、跑 Baseline、跑 Ours、跑 Eval)
bash adv_work/final_work/run_experiment.sh
```

---

### Cell 5：查看结果

```python
import glob, os, json
from PIL import Image
from IPython.display import display

# 辅助函数：展示文件夹里的前几张图
def show_results(dirpath, title):
    print(f"\n== {title} ==")
    files = sorted(glob.glob(os.path.join(dirpath, "*_cfg_*.jpg")))
    if not files:
        print("No images found.")
        return
    for f in files[:3]: # 只看前3张
        print(f)
        display(Image.open(f).resize((512, 512)))

# 展示 Baseline vs Ours
show_results("outputs/baseline", "Baseline (Vanilla)")
show_results("outputs/ours", "Ours (Masked)")

# 打印量化指标
for p in ["outputs/baseline/evaluation_results.json", "outputs/ours/evaluation_results.json"]:
    if os.path.exists(p):
        with open(p) as f:
            print(f"\n{p}:", json.load(f).get("summary"))
```
