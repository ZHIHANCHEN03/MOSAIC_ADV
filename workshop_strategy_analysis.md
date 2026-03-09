# Workshop Paper 投稿策略分析

针对两个 CVPR 2026 Workshop：
1.  **P13N: Personalization in Generative AI** (个性化生成)
2.  **Generative Models for Computer Vision** (生成式视觉)

以下是对方案一（修改 Layer）和方案二（工程化迭代）的深度对比及投稿建议。

---

## 结论先行 (Recommendation)

**首选：方案一 (Model-Centric / Attention Masking)**

*   **理由**：
    *   **学术味更浓**：修改模型底层机制（Attention）通常被认为比纯 Pipeline 拼接（方案二）更具有“Novelty”和“Technical Depth”。
    *   **通用性更强**：Attention Masking 是一种可以直接集成到任何 Diffusion 模型中的算法改进，不仅仅是针对 MOSAIC 的补丁。
    *   **符合 Workshop 调性**：
        *   **P13N** 明确提到 "Advanced optimization methods" 和 "Multi-subject composition"，你的方案直接切中 "Composition" 这一痛点，且方法优雅（Training-free）。
        *   **Gen Vision** 关注 "Advances in generative image models"，对底层机制的探索更受欢迎。

**备选：方案二 (System-Centric / Iterative)**
*   **适用场景**：如果你时间极度紧迫（比如只有 2 天），或者方案一实验效果不稳定。方案二是一个稳妥的 "Application Paper" 或 "System Paper"，强调实效性。

---

## 详细对比分析

| 维度 | 方案一：Attention Masking (Layer Mod) | 方案二：Iterative Generation (Pipeline) |
| :--- | :--- | :--- |
| **创新性 (Novelty)** | ⭐⭐⭐⭐<br>深入模型内部，通过干预 Cross-Attention 解决特征泄露，这是一种算法层面的创新。 | ⭐⭐<br>利用现有的 Inpainting 能力进行串行生成，更多是工程上的“组合拳”。 |
| **论文卖点 (Pitch)** | "We propose a **training-free attention control mechanism** that enforces strict identity separation in latent space."<br>(提出一种免训练的注意力控制机制，强制在潜在空间分离身份) | "We present a **robust multi-pass generation framework** that decomposes complex scenes into manageable sub-tasks."<br>(提出一种鲁棒的多遍生成框架，将复杂场景分解为子任务) |
| **P13N 匹配度** | **High**.<br>完美契合 "Multi-subject composition" 主题。你可以声称解决了多主体个性化中的 "Identity Leakage" 这一核心难题。 | **Medium-High**.<br>虽然也解决了问题，但看起来更像是一个 "Trick" 而非 "Methodology"。 |
| **Gen Vision 匹配度** | **High**.<br>符合 "Advances in generative models"。 | **Medium**.<br>可能被认为技术深度不足。 |
| **实验工作量** | **中等**。<br>需要修改 `flux_omini_mosaic.py`，并调节 Mask 的强度（可能需要 Soft Mask 而非 Hard Mask）。 | **低**。<br>代码几乎写完了，只需要跑大量 Case 并挑选好图。 |
| **潜在风险** | 可能会影响光影融合（不同区域光照不一致），或者需要精细调节 Mask 边缘。 | 物体之间没有交互（因为是分步生成的），看起来像“贴图”；速度慢 N 倍。 |

---

## 投稿故事线 (Storyline for Option 1)

如果你选方案一，你的 Paper 应该这样写：

1.  **Introduction**:
    *   个性化生成很火，但多主体 (Multi-Subject) 还是个坑。
    *   现有方法（如 MOSAIC）在 Scaling Object 时会失败。
    *   **Key Insight**: 我们发现失败的根本原因是 **Attention Collapse** —— 也就是该看猫的 Token 去看了包。
2.  **Method (The "Magic")**:
    *   我们不需要重新训练！
    *   我们引入了 **"Spatial-Aware Cross-Attention (SACA)"**。
    *   在推理时，根据粗略的 Layout（可以是用户给的，也可以是简单的 Grid），动态修改 Attention Map。
    *   公式：$A_{new} = A \odot M + (1-M) \times (-\infty)$。
3.  **Experiments**:
    *   **Qualitative**: 放一张 6 个物体的图，MOSAIC 糊成一团，你的 SACA 泾渭分明。
    *   **Quantitative**: Identity Score (DINO) 提升了 X%，Disentanglement Score (Mixing) 降低了 Y%。

## 总结

**去投 P13N Workshop，用方案一。**
它足够硬核，能够支撑起一篇 4 页的 Short Paper，甚至有机会被选为 Oral/Spotlight。方案二可以作为 Baseline 对比，用来证明方案一“既保留了多主体的整体性（光影自然），又实现了类似分步生成的隔离效果”。
