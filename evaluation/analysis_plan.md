# MOSAIC 评估指标与改进方案

## 1. 原本 Repo 的指标
通过分析 `preprocess/dift_point_matching.py` 和 `train.py`，MOSAIC 原本主要关注训练过程中的 Loss，**缺乏直接的生成质量评估指标**。

*   **训练指标 (Losses)**:
    *   `loss_align` (对应性损失): 衡量特征对齐程度。
    *   `loss_sep` (解耦损失): 衡量不同主体的特征分离程度。
    *   `diffusion_loss`: 基础的扩散模型生成损失。
*   **预处理指标**:
    *   使用了 DINOv2 和 Stable Diffusion 特征来计算点匹配 (Point Matching)，但这仅用于生成训练数据（Ground Truth），并非用于评估生成结果。
    *   使用了简单的质量过滤（如 `compositeStructure`, `objectConsistency` >= 5），但这依赖于外部打分（可能来自 GPT-4V 或人工），代码中并未包含自动打分模型。

## 2. 我们要做什么 (Evaluation Plan)

针对多主体生成中“交互变差”和“特征混淆”的问题，我们需要建立一套**自动化、量化的评估体系**。

### 核心目标
量化评估生成图像在以下三个维度的表现：
1.  **身份一致性 (Identity Consistency)**: 生成的主体是否与参考图一致？
2.  **文本/交互一致性 (Prompt/Interaction Fidelity)**: 是否正确执行了 Prompt 描述的动作和交互？
3.  **特征解耦 (Disentanglement)**: 不同主体的特征是否混淆（如猫长了背包的花纹）？

### 实现逻辑
输入：
*   **JSON 配置文件 (`example_cases.json`)**: 定义了测试用例。
    *   `index`: 唯一标识符 (0, 1, ...)。
    *   `prompt`: 生成指令 (e.g., "A toy car on a wooden floor")。
    *   `image_paths`: 参考图片列表 (e.g., ["assets/rc_car.jpg"])。
*   **生成结果目录 (`outputs/`)**: 存放生成的图像文件。
    *   命名格式: `{index}_cfg_{guidance_scale}_{height}x{width}.jpg` (e.g., `0_cfg_3.5_512x512.jpg`)。

输出：
*   **Metrics Report**: 包含 CLIP Score, DINO Identity Score 等数值指标。
*   **Failure Analysis**: 自动判定是否存在 Identity Loss, Mixing, 或 Interaction Failure。
*   **Visualization**: 可视化得分和 Failure Mode。

### 详细指标设计

#### A. 身份一致性 (Identity) - DINOv2
*   **方法**:
    1.  使用 **Grounding DINO** 检测生成图中的主体 Bounding Box。
    2.  裁剪出主体区域。
    3.  提取 **DINOv2** 特征，计算其与参考图特征的 **Cosine Similarity**。
*   **指标**: `Identity Score (Subject A)`

#### B. 文本/交互一致性 (Interaction) - CLIP
*   **方法**:
    1.  计算生成图与完整 Prompt 的 **CLIP Text-Image Similarity**。
    2.  (可选) 计算生成图与去除交互词后的 Prompt 的相似度差值。
*   **指标**: `CLIP Score (Overall)`

#### C. 特征解耦 (Disentanglement) - Cross-Identity Score
*   **方法**:
    1.  计算生成主体 A 与参考主体 B 的相似度。
*   **指标**: `Mixing Score (A -> B)` (越低越好)

---

## 3. 下一步计划
1.  实现 `evaluation/eval.py` 脚本，集成 DINOv2, CLIP, Grounding DINO (如果环境允许，否则使用简单的 Crop 或整个图像近似)。
2.  定义 `evaluation/run_eval.sh` 批处理脚本。
3.  生成测试报告。
