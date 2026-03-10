# Workshop Paper: The Most Promising Path
**Target**: P13N / CVPR Generative Vision Workshop
**Selected Approach**: **Spatial-Aware Cross-Attention Masking** (`inference_attn_mask.py`)

---

## 为什么它是最 Promising 的？(Winning Factors)
1.  **Academic Value (学术价值)**: 它是对 Diffusion Model 底层机制（Attention）的改进，比单纯的 Pipeline 拼接（Iterative Generation）显得更 "Deep"、更有技术含量。
2.  **Visual Quality (视觉效果)**: 它保留了**全局光影的一致性**。猫和包是一起生成的，光线、阴影、透视会自然融合。而 Iterative 方法容易像“贴图”，缺乏整体感。
3.  **Efficiency (效率)**: 一次生成搞定，不需要跑 N 遍 Diffusion，速度快 N 倍。

---

## 具体 Pipeline (Execution Plan)

### Step 1: LLM Layout Generation (The "Brain")
*   **Input**: User Prompt (e.g., "A cat and a dog playing on the grass") + Canvas Size (512x512).
*   **Model**: GPT-4 / Llama 3 (Few-shot Prompting).
*   **Output**: JSON Bounding Boxes.
    ```json
    {"cat": [50, 100, 200, 400], "dog": [250, 100, 400, 400]}
    ```
*   **Action**: 写一个简单的 Python 脚本（模拟 LLM 或调用 API），把 Prompt 转成 Box 坐标。

### Step 2: Attention Mask Construction (The "Bridge")
*   **Input**: Bounding Boxes from Step 1.
*   **Process**:
    *   创建一个 `(H_latent, W_latent)` 的零矩阵。
    *   **Square/Box Masking**: 根据 Box 坐标，将对应区域设为 1.0 (Foreground)。
    *   **Overlap Handling**: 重叠区域允许同时关注多个 Subject (即 Mask 值叠加或取 Max)。
    *   **Soft Constraint (关键改进)**:
        *   对二值 Mask 应用 **Gaussian Blur** (Kernel Size ~15)，使边缘平滑过渡。
        *   背景区域不设为 $-\infty$，而是设为 **Soft Penalty** (e.g., -5.0 到 -10.0)，允许模型在边缘处进行微弱的 Context 交互，避免 Artifacts。
*   **Output**: `spatial_mask` Tensor，形状 `[Query_Tokens, Num_Refs]` (Float32)。

### Step 3: Inference with Masked Attention (The "Core")
*   **Input**: `spatial_mask` + Reference Images + Prompt.
*   **Process**:
    *   修改 `src/flux_omini_mosaic.py` 中的 `attn_forward`。
    *   在计算 `attn_score` 后，注入 Soft Mask：
        ```python
        # mask is [0, 1] after sigmoid or normalization
        # penalty is negative (e.g., -10)
        attn_bias = (mask - 1.0) * penalty_strength 
        attn_score = attn_score + attn_bias
        ```
    *   执行去噪循环。
*   **Output**: 一张结构清晰、特征不混淆且光影自然的高质量图片。

### Step 4: Evaluation (The "Proof")
*   **Run**: 使用 `eval.py`。
*   **Metrics**:
    *   **DINO Score**: 证明猫更像猫了（Identity）。
    *   **CLIP Score**: 证明图文对齐了（Interaction）。
    *   **Mixing Score**: 证明猫没长包的花纹（Disentanglement）。

---

## 下一步行动 (Action Items)

1.  **完善 `inference_attn_mask.py`**:
    *   目前它只是个 Demo，需要把它和 `src/flux_omini_mosaic.py` 真正打通。
    *   你需要我修改 `src/flux_omini_mosaic.py`，加入 Mask 接收逻辑吗？
2.  **实现 LLM 接口**:
    *   写一个 `layout_generator.py`，用简单的规则或 API 生成 Box。

准备好开始了吗？我们先从修改 `flux_omini_mosaic.py` 开始，这是最硬核的一步。
