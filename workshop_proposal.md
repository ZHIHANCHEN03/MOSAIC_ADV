# MOSAIC Workshop Proposal: 解决 Scaling Object 问题方案

## 背景与问题 (Problem Statement)
MOSAIC 在处理 1-2 个主体时表现良好，但当主体数量增加 (Scaling Object, e.g., >3) 时，生成质量显著下降。主要表现为：
1.  **特征混淆 (Identity Mixing)**: 不同主体的特征相互干扰（如猫长了背包的花纹）。
2.  **交互失效 (Interaction Failure)**: 物体并未按照 Prompt 的描述进行交互，而是简单并排或重叠。
3.  **主体丢失 (Missing Objects)**: 某些主体未能生成，或者被背景吞没。

针对 Workshop Paper (如 P13N, CVPR Generative Vision)，我们需要提出**快速有效且有理论依据**的解决方案。

---

## 方案一：修改 Layer / Attention 机制 (Model-Centric Approach)
**核心思想**：通过干预 Self-Attention 和 Cross-Attention 层，强制模型“各司其职”，从底层解决特征混淆。

### 1. Spatial-Aware Cross-Attention Masking (空间感知注意力掩码)
*   **原理**:
    *   目前的 Cross-Attention 是全局的，即图像的每个像素都在关注所有的 Reference Tokens。
    *   **改进**: 引入一个粗略的 Layout Mask (可以是用户提供，也可以是 LLM 预测)。
    *   在计算 Attention Score $A = Q K^T$ 时，加上一个 Mask $M$：
        $$A_{masked} = A + M$$
    *   其中 $M_{ij} = -\infty$ 如果像素 $i$ 不属于主体 $j$ 的区域。
*   **优势**: 从根本上切断了特征混淆的路径。生成的“猫”区域只能看到“猫”的 Reference，绝对看不到“背包”。
*   **实现难度**: 中等。需要修改 `src/flux_omini_mosaic.py` 中的 `attn_forward` 函数。

### 2. Frequency-based Feature Injection (基于频率的特征注入)
*   **原理**:
    *   主体特征主要体现在高频细节（纹理、边缘）上，而结构和布局体现在低频信息上。
    *   **改进**: 在去噪过程的早期（High Noise, Low Frequency），允许全局 Attention 以确定布局；在后期（Low Noise, High Frequency），严格限制 Attention 范围，只允许局部特征注入。
*   **优势**: 不需要严格的 Mask，能自然地保持图像整体的协调性，同时保留细节特征。

---

## 方案二：工程化 / Pipeline 优化 (System-Centric Approach)
**核心思想**：不修改模型权重，而是通过优化生成流程 (Inference Pipeline) 来规避单次生成的瓶颈。

### 1. Iterative "Divide-and-Conquer" Generation (分治法迭代生成)
*   **原理**:
    *   **Step 1 (Layout Planning)**: 使用 LLM 或简单的规则，将画布划分为 N 个区域（Grid 或 Bounding Boxes）。
    *   **Step 2 (Base Generation)**: 生成一张只有背景和模糊轮廓的底图。
    *   **Step 3 (Sequential Inpainting)**: 利用 MOSAIC 的 `latent_mask` 能力，**逐个**生成主体。
        *   Pass 1: 生成 Subject A (Mask A)。
        *   Pass 2: 生成 Subject B (Mask B)。
*   **优势**:
    *   **极度稳定**: 每次只处理一个主体，完全避免了多主体间的 Attention 竞争。
    *   **无需训练**: 纯推理端优化，即插即用。
    *   **可解释性强**: 如果某个物体生成坏了，可以单独重绘它，而不影响其他物体。

### 2. Multi-Pass Composition via Regional Prompting (区域提示词合成)
*   **原理**:
    *   利用 Diffusers 的 Multi-Diffusion 或 Regional Prompting 技术。
    *   在去噪的每一步，将 Latent 按照区域切分。
    *   区域 A 的 Latent 只受 "A cat" Prompt 和 Reference A 引导；区域 B 只受 "A backpack" 引导。
    *   最后在 Latent 空间通过加权平均合并（如 Gaussian Blending）以消除边界缝隙。
*   **优势**: 比简单的 Inpainting 更自然，光影融合更好，因为所有物体是**同时**但在**不同区域**去噪的。

---

## Workshop Paper 策略建议

### 推荐路线：方案二 (Iterative / Regional) + 方案一 (Attention Mask) 的简化版
为了在短时间内产出高质量 Paper：

1.  **Method**: 提出一种 **"Layout-Guided Multi-Subject Generation Pipeline"**。
    *   结合 **LLM Layout Planning** (用 GPT-4 预测 Bounding Box) 和 **Attention Masking** (在推理时强制约束)。
2.  **Experiments**:
    *   对比 **Vanilla MOSAIC** vs **Ours (Layout-Guided)**。
    *   指标：Identity Score (DINO), Interaction Score (CLIP), 加上人工评估 (User Study)。
    *   展示 Scaling 曲线：当主体数从 2 增加到 6 时，Vanilla 性能急剧下降，而 Ours 保持平稳。
3.  **Contribution**:
    *   揭示了 Multi-Subject 生成中的 "Attention Collapse" 现象。
    *   提出了一种无需训练 (Training-free) 的通用解决方案。

### 为什么适合 Workshop?
*   **CVPR Generative Vision**: 关注 "Visual Recognition" 和 "Generative Models" 的结合。你的方案利用了 Detection/Layout (Recognition) 来指导 Generation，非常切题。
*   **P13N**: 关注 "Personalization"。解决多主体个性化是该领域的核心痛点，且你的方案提升了可控性。
