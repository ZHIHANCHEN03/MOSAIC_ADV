## Colab 一键运行（git clone → 跑 `run_experiment.sh` → 直接看结果）

### 0) Colab 环境要求
- 在 Colab 右上角选择 **Runtime → Change runtime type → GPU**（建议 L4/A100；T4 可能会慢或显存不够）
- 需要能从 HuggingFace 下载模型：`black-forest-labs/FLUX.1-dev` 与 `ByteDance-FanQie/MOSAIC`

---

### Cell 1：Clone repo
把下面这一格贴到 Colab 里运行（把 `REPO_URL` 换成你的仓库地址；私有仓库请自行处理鉴权）。

```bash
%%bash
set -euo pipefail

REPO_URL="YOUR_REPO_GIT_URL_HERE"
REPO_DIR="MOSAIC_ADV"

if [ ! -d "$REPO_DIR" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

git rev-parse --short HEAD
ls -lah
```

---

### Cell 2：安装依赖

```bash
%%bash
set -euo pipefail
cd MOSAIC_ADV

python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

---

### Cell 3（可选）：启用 Gemini 布局（否则自动 fallback 网格布局）
如果你想让 `layout_generator.py` 真调用 Gemini，在 Colab 运行：

```python
import os
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
```

不设置也能跑，只是 layout 会用网格分配 bbox。

---

### Cell 4：运行 Final Work 脚本（一键 baseline vs ours + eval）

```bash
%%bash
set -euo pipefail
cd MOSAIC_ADV

bash adv_work/final_work/run_experiment.sh
```

跑完会生成：
- `outputs/baseline/`：baseline 结果图 + `evaluation_results.json`
- `outputs/ours/`：ours 结果图 + `evaluation_results.json`

---

### Cell 5：在 Colab 里直接展示结果图
（每个目录最多展示前 10 张 `*_cfg_*.jpg`，并打印评估 JSON 路径）

```python
import glob, os, json
from PIL import Image
from IPython.display import display

def show_first_images(dirpath, title, limit=10, size=(512, 512)):
    print(f"\n== {title} ==")
    paths = sorted([p for p in glob.glob(os.path.join(dirpath, "*_cfg_*.jpg")) if "compared" not in p])
    if not paths:
        print("No images found:", dirpath)
        return
    for p in paths[:limit]:
        print(p)
        img = Image.open(p).convert("RGB")
        display(img.resize(size))

show_first_images("outputs/baseline", "Baseline")
show_first_images("outputs/ours", "Ours")

for p in ["outputs/baseline/evaluation_results.json", "outputs/ours/evaluation_results.json"]:
    if os.path.exists(p):
        print("\nFound:", p)
        with open(p, "r") as f:
            data = json.load(f)
        print("Summary:", data.get("summary", {}))
    else:
        print("\nNot found:", p)
```

---

### 常见问题排查
- **没开 GPU**：会特别慢甚至跑不动；请确认 Colab Runtime 是 GPU，并在 Cell 里跑 `!nvidia-smi`
- **显存不够**：换更大显存的 GPU（L4/A100），或减少分辨率/subjects（需要你改 test cases）
- **下载失败**：检查网络、HuggingFace 访问、或设置 `HF_TOKEN`（如果模型需要）

