## Colab 一键运行（Upload adv_work -> Run Script）

### 0) Colab 环境准备
- **Runtime**: 选择 GPU (T4/L4/A100)
- **Secrets**: 如果需要 Gemini 生成 Layout，请在 Colab 左侧钥匙图标添加 `GOOGLE_API_KEY`。

---

### Cell 1: 初始化环境（不安装 PyTorch/CUDA）
**注意**：仅安装 MOSAIC + 你的改动所需依赖，不安装/覆盖 PyTorch 与 CUDA。
**注意**：下面命令里的 URL 不要加反引号。

**运行完此 Cell 后，如果系统提示 "Restart Session"，请务必点击重启！**

```bash
%%bash
set -euo pipefail

# 1. Force fresh clone (delete old folder if exists)
rm -rf MOSAIC_ADV
git clone https://github.com/ZHIHANCHEN03/MOSAIC_ADV.git MOSAIC_ADV
cd MOSAIC_ADV

# 2. Modify requirements.txt for compatibility
sed -i '/^torch==/d' requirements.txt
sed -i '/^torchvision/d' requirements.txt
sed -i '/^torchaudio/d' requirements.txt
sed -i '/^deepspeed/d' requirements.txt

# 3. Install Dependencies
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
python3 -m pip install google-generativeai python-dotenv
```

**检查安装结果** (可单独运行确认)：
```python
import torch
print(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Capability: {torch.cuda.get_device_capability(0)}")
else:
    print("WARNING: CUDA not available!")
```

---

### Cell 2: 上传 `adv_work` 文件夹 (Critical Step)
**请手动操作**：
1. 在本地电脑找到你的 `adv_work` 文件夹。
2. 在 Colab 左侧文件栏，打开 `MOSAIC_ADV` 文件夹。
3. 将本地的 `adv_work` 文件夹直接拖拽到 Colab 的 `MOSAIC_ADV` 文件夹内。
   *(确保路径结构为 `MOSAIC_ADV/adv_work/final_work/...`)*

或者运行以下代码确认文件夹是否就位，并测试 LLM 是否可用：
```python
import os
if os.path.exists("MOSAIC_ADV/adv_work/final_work/run_experiment.sh"):
    print("✅ adv_work folder detected!")
else:
    print("❌ adv_work folder NOT found. Please upload it to MOSAIC_ADV/ first.")

# Optional: LLM connectivity check
try:
    from google import genai
    client = genai.Client()
    resp = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents="Return 'OK' if you can respond."
    )
    print("✅ LLM test OK:", resp.text.strip())
except Exception as e:
    print("❌ LLM test failed:", e)
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
