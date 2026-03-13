## Colab 快速展示（只跑 8/10/12 各 1 张）

### Cell 1: Clone + 安装依赖
```bash
%%bash
set -euo pipefail

rm -rf MOSAIC_ADV
git clone https://github.com/ZHIHANCHEN03/MOSAIC_ADV.git MOSAIC_ADV
cd MOSAIC_ADV

python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
python3 -m pip install google-generativeai python-dotenv lang_sam
```

### Cell 2: 上传 adv_work 文件夹 + 测试 LLM
```python
import os
if os.path.exists("MOSAIC_ADV/adv_work/final_work/quick_show_adv.sh"):
    print("✅ adv_work folder detected!")
else:
    print("❌ adv_work folder NOT found. Please upload it to MOSAIC_ADV/ first.")

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

### Cell 3: 运行 quick_show_adv.sh
```bash
%%bash
set -euo pipefail
if [ -d "MOSAIC_ADV" ]; then cd MOSAIC_ADV; fi
chmod +x adv_work/final_work/quick_show_adv.sh
bash adv_work/final_work/quick_show_adv.sh
```

### Cell 4: 展示三张图片
```python
import glob, os
from PIL import Image
from IPython.display import display

files = sorted(glob.glob("MOSAIC_ADV/outputs/ours_quick/*.jpg"))
print(files)
for f in files[:3]:
    display(Image.open(f).resize((512, 512)))
```
