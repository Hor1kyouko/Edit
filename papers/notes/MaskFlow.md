# MaskFlow: Precise, Consistent and Seamless Regional Image Editing

> 中文阅读笔记（术语与公式保留英文）

## 0. 论文信息

- 标题：**MaskFlow: Precise, Consistent and Seamless Regional Image Editing**
- 作者：Rui Xu, Yang Yong, Shunzi Yang, Ruihao Gong, Chengtao Lv
- 版本：arXiv:2608.06929v2，2026-08-11
- 论文：[arXiv](https://arxiv.org/abs/2608.06929) / [HTML](https://arxiv.org/html/2608.06929)
- 项目页：[MaskFlow Project](https://reychiaro.github.io/MaskFlow/)
- 官方代码：[ReyChiaro/MaskFlow](https://github.com/ReyChiaro/MaskFlow)
- 状态：截至笔记整理时是 arXiv preprint，未在论文主页看到正式会议/期刊接收信息。
- Backbone：QwenImage-2511 / Qwen-Image-Edit-2511，训练 attention LoRA。

## 1. 一句话总结

MaskFlow 的核心不是“把 mask 编码后喂给编辑模型”，而是让 mask 直接决定 **Flow Matching probability path**：mask 内沿 target image 的生成轨迹运动，mask 外沿 source image 的重建轨迹运动；随后在每个采样步通过 **Soft-Poisson de-seaming** 修正 clean-target estimate，并反推出新的 velocity field，使生成前景与保留背景在边界处逐步融合。

## 2. 论文要解决的三个问题

给定 source image、用户 mask 和 editing instruction，区域编辑同时需要：

1. **Precise localization**：变化应严格发生在 mask 指定的区域。
2. **Consistent preservation**：mask 外的背景、文字、身份和结构应保持不变。
3. **Seamless transition**：编辑区域与背景之间不能出现颜色、纹理或梯度断裂。

论文认为已有方法的主要问题是：

- instruction-only editor 缺少显式空间约束，复杂场景和 infographic 中仅靠文字难以指定准确位置；
- 很多 mask-based editor 只是把 mask 当作 additional condition，模型仍然在整张 latent 上预测全局 velocity/noise；
- foreground generation 与 background preservation 使用不同过程，二者 trajectory 不一致，容易在 mask boundary 产生 seam；
- 最终图像上的 Poisson blending 属于 post-processing，无法回头修正中间采样阶段已经产生的不兼容结构。

## 3. 论文故事线

论文把 regional editing 拆成三个相互对应的层次：

| 目标 | 问题来源 | 方法设计 |
|---|---|---|
| precise | 语言定位含糊，mask 只是弱 condition | mask-guided probability path + 去掉训练 prompt 中冗余位置描述 |
| consistent | 全局生成会改动背景 | mask 外显式采用 source probability path |
| seamless | target/source 两条区域轨迹在边界不连续 | 每一步执行 Soft-Poisson clean-estimate refinement |

完整逻辑为：

$$
\text{mask-conditioned path}
\rightarrow
\text{regional flow supervision}
\rightarrow
\text{clean target estimate}
\rightarrow
\text{Soft-Poisson refinement}
\rightarrow
\text{corrected velocity}
\rightarrow
\text{next latent state}.
$$

## 4. Flow Matching 预备知识

论文使用 noise-to-data 的时间方向：

- $t=0$：Gaussian noise；
- $t=1$：clean data。

这和部分 Rectified Flow 编辑论文采用的 data-to-noise 记号相反，阅读公式时不要直接比较 timestep 大小。

设：

- $\epsilon\sim\mathcal N(0,I)$：Gaussian noise；
- $x_1\sim p_{data}$：clean target latent；
- $\alpha(t),\beta(t)$：插值 schedule；
- 常用写法 $\alpha(t)=1-\sigma(t),\beta(t)=\sigma(t)$，其中 $\sigma(0)=1,\sigma(1)=0$。

普通 probability path 为：

$$
x(t)=\alpha(t)x_1+\beta(t)\epsilon.
$$

其 target velocity 是：

$$
u(t)=\dot\alpha(t)x_1+\dot\beta(t)\epsilon
=\dot\sigma(t)(\epsilon-x_1).
$$

Conditional Flow Matching objective：

$$
\mathcal L_{CFM}
=
\mathbb E\left[
\left\|v_\theta(x(t),\sigma(t))-
\dot\sigma(t)(\epsilon-x_1)\right\|_2^2
\right].
$$

由当前 latent 和预测 velocity 可以估计 clean target：

$$
\widehat x_1(t)
=x(t)-\frac{\sigma(t)}{\dot\sigma(t)}
v_\theta(x(t),\sigma(t)).
$$

这一步是理解 Soft-Poisson 的关键：论文不是直接对 velocity 做空间平滑，而是先将 velocity 转换成 clean-target estimate，在 clean latent 上解决 Poisson 问题，再从 refined target 重新计算 velocity。

## 5. 核心方法一：Mask-guided probability path

### 5.1 张量定义

- $x_S\in\mathbb R^{CHW}$：source image latent；
- $x_1\in\mathbb R^{CHW}$：paired target image latent；
- $M\in\{0,1\}^{H\times W}$：binary edit mask，1 表示可编辑；
- $m\in\{0,1\}^{CHW}$：将 $M$ 沿 channel broadcast 后展平；
- $\odot$：element-wise multiplication。

### 5.2 拼接两条区域 probability path

MaskFlow 定义：

$$
x(t)
=m\odot\left(\alpha(t)x_1+\beta(t)\epsilon\right)
+(1-m)\odot\widetilde x(t),
$$

实际取：

$$
\widetilde x(t)=\alpha(t)x_S+\beta(t)\epsilon.
$$

代入后：

$$
x(t)
=m\odot\left(\alpha x_1+\beta\epsilon\right)
+(1-m)\odot\left(\alpha x_S+\beta\epsilon\right).
$$

由于两部分共享同一 $\epsilon$，可以合并为：

$$
x(t)=
\alpha(t)\left[m\odot x_1+(1-m)\odot x_S\right]
+\beta(t)\epsilon.
$$

因此它训练模型到达的 clean endpoint 实际是一个 composite latent：

$$
x_{endpoint}
=m\odot x_1+(1-m)\odot x_S.
$$

解释：

- mask 内：从 noise 走向 target latent；
- mask 外：从同一 noise 走向 source latent；
- mask 不再只是告诉模型“在哪里编辑”，而是直接决定每个空间位置所属的 endpoint 与 target velocity。

这就是论文声称“mask is incorporated into the probability path”的准确含义。

### 5.3 Mask-aware objective

定义 mask area ratio：

$$
a(M)=\frac{1}{HW}\sum_{i,j}M_{ij}.
$$

训练损失为：

$$
\mathcal L_{MF}
=\mathbb E\left[
\left\|
m\odot\frac{\omega_{mask}}{a(M)}
\left(v_\theta(x(t),\sigma(t)\mid x_S,M)-\dot x(t)\right)
\right\|_2^2
\right].
$$

其中 $1/a(M)$ 补偿 mask size：小 mask 的有效像素较少，如果不做归一化，其梯度贡献会系统性弱于大 mask。

### 5.4 一个需要注意的公式细节

按论文 Eq. (6) 的字面形式，residual 又乘了一次 $m$，因此 loss 主要监督 masked region；mask 外虽然在 probability path 中沿 source endpoint 构造，但没有在该公式中得到对称的显式 per-pixel regression loss。

所以“mask 外严格保持 source trajectory”应拆成两层理解：

1. **Evidence**：训练 input path 的 mask 外部分确实由 source latent 构造；官方推理配置也提供 noisy-source unmasking 和最终 source pixel blending。
2. **尚需验证的解释**：仅凭论文 Eq. (6)，不能直接推出网络在 mask 外学习到了严格等于 source velocity 的 vector field。

这也是复现时需要对照代码确认 loss reduction、unmasking scheduler 和 pixel blend 的原因。

## 6. 核心方法二：Soft-Poisson de-seaming

### 6.1 为什么普通拼接会产生 seam？

mask 内目标是生成 $x_1$，mask 外目标是保持 $x_S$。即使两个区域各自正确，边界两侧也可能具有不同的：

- color statistics；
- illumination；
- texture frequency；
- spatial gradient；
- semantic geometry。

硬拼接只保证位置归属，不保证边界处的一阶空间变化连续。

### 6.2 Soft mask 与扩展边界区域

论文从 binary mask $M$ 出发，经有限支撑 Gaussian kernel 得到 soft mask $\widetilde M$。原 mask 内仍保持 1，边界外形成逐渐衰减的 transition band：

$$
\widetilde\Omega=\{p\mid\widetilde M(p)>0\}.
$$

- $\widetilde m(p)\approx1$：更相信生成的 target estimate；
- $\widetilde m(p)\approx0$：更靠近 source latent；
- $\partial\widetilde\Omega$：使用 source 作为 Dirichlet boundary。

### 6.3 优化变量和表示空间

在每个 sampling step：

1. 当前 latent：$x(t)$；
2. 模型预测 velocity：$v_\theta(x(t),\sigma(t))$；
3. 根据 velocity 得到 clean-target estimate：$\widehat x_1(t)$；
4. reshape 成 $C\times H\times W$ latent feature map；
5. 在 latent spatial field 上求解 refined field $z^*$。

这里的 Poisson 运算发生在 latent feature map，而不是最终 RGB image；空间梯度/Laplacian 作用于 $(H,W)$，并对每个 channel 分别计算。

### 6.4 Soft-Poisson objective

$$
\begin{aligned}
z^*=\arg\min_z\quad
&\int_{\widetilde\Omega}
\|\nabla z(p)-\nabla\widehat x_1(p)\|_F^2dp\\
&+\lambda_e\int_{\widetilde\Omega}
\widetilde m(p)\|z(p)-\widehat x_1(p)\|_2^2dp\\
&+\lambda_s\int_{\widetilde\Omega}
(1-\widetilde m(p))\|z(p)-x_S(p)\|_2^2dp,
\end{aligned}
$$

约束：

$$
z|_{\partial\widetilde\Omega}
=x_S|_{\partial\widetilde\Omega}.
$$

三项分别是：

1. **Gradient transfer term**：保留模型所生成前景的局部结构和梯度；
2. **Edit anchor term**：soft mask 值大时靠近 predicted clean edit；
3. **Source anchor term**：接近边界和外侧时逐渐回到 source latent。

与 classical Poisson cloning 相比，多出的两个 soft anchor term 防止纯梯度匹配造成大范围颜色漂移，并允许在 transition band 中连续控制 source/edit 的相对影响。

### 6.5 从 refined target 反推 refined velocity

求得 $\widehat x_1^*(t)$ 后，重新定义：

$$
v_\theta^*(x(t),\sigma(t))
=\frac{\dot\sigma(t)}{\sigma(t)}
\left(x(t)-\widehat x_1^*(t)\right).
$$

然后用 $v_\theta^*$ 更新下一步 latent。

这使 Soft-Poisson 不只是一个视觉后处理器，而成为 trajectory controller：

$$
\widehat x_1
\rightarrow
\widehat x_1^*
\rightarrow
v_\theta^*
\rightarrow
x_{i+1}.
$$

在所有 timestep 重复后，边界一致性会持续影响后续生成，而不是在最终结果上一次性修补。

### 6.6 数值求解

论文使用 four-neighbor stencil 和 Jacobi iteration，默认每个采样步 50 次迭代。单点更新同时包含：

- 邻域 refined values；
- 扩展区域外的 source boundary values；
- predicted target 的局部 gradient difference；
- edit/source 两个 soft anchor。

优点是简单、可卷积并行；代价是若在 50 个 denoising steps 中每步做 50 次迭代，会增加明显的空间迭代开销。官方仓库另外提供 DMD 8-step/16-step distilled variant，但这属于部署加速扩展，不是核心论文公式的必要组成。

## 7. MEData / MaskEdit-10k 数据构造

论文构建约 10K 个 $<prompt,source,mask,target>$ 配对，覆盖 natural scenes 与 infographic images。

流程：

1. **Object detection / selection**：VLM 从含多个概念的 source image 中选择值得编辑的主要对象；
2. **Prompt generation**：为同一编辑生成两种 instruction：
   - complete instruction：包含 operation、target position 和 desired result；
   - mask-dependent instruction：去掉明确位置描述，以指示性表达代替；
3. **Target generation**：用包含精确位置的完整 instruction 生成 target；
4. **Mask construction**：SAM 给出初始 segmentation，再由人工修正并制作更接近真实用户输入的 arbitrary-shape masks。

训练时刻意不提供冗余的位置语言，使模型必须从 mask 获得 location。这个设计对 infographic 特别重要，因为页面中经常有重复文本框、相似 icon 和密集布局。

命名注意：论文称数据集为 **MEData**，当前官方仓库 README 称公开数据为 **MaskEdit-10k**；使用时应核对二者是否完全等价以及公开版本的 split 定义。

## 8. 训练与推理配置

- Base model：QwenImage-2511；当前代码默认 Qwen/Qwen-Image-Edit-2511。
- 参数更新：attention LoRA。
- LoRA rank：256。
- 训练：5K steps。
- Optimizer：Prodigy。
- 推理：50 sampling steps。
- CFG scale：4.0。
- Soft-Poisson：默认 50 Jacobi iterations。
- 输入：source RGB image、aligned binary/soft mask、editing prompt。
- 官方代码还支持 source pixel blend、noisy-source unmasking、mask dilation/blur 以及 DMD 加速 checkpoint；这些实现选项需要和论文主方法、消融设定区分。

## 9. 实验结果与证据强度

### 9.1 主结果

在 MEData 上，MaskFlow 报告：

| Metric | MaskFlow | 主要含义 |
|---|---:|---|
| CLIP ↑ | 0.9782 | 编辑结果与 reference 的语义/视觉相似度 |
| DINO ↑ | 0.9532 | 结构一致性 |
| FID ↓ | 19.90 | 数据分布级生成质量 |
| PSNR ↑ | 22.60 | 像素保真度 |
| SSIM ↑ | 0.7846 | 结构保真度 |
| Background MSE ↓ | 报告为 0.0000 | mask 外变化极小，但显示精度可能掩盖非零值 |
| Background LPIPS ↓ | 报告为 0.0000 | mask 外感知变化极小 |

与 commercial、general editing、regional editing 和 QwenImage+Inpaint baselines 相比，论文报告 MaskFlow 在 CLIP、FID、PSNR、SSIM 和 background LPIPS 上最佳。

### 9.2 Ablation

| MF | SPD | FID ↓ | CLIP ↑ | DINO ↑ | LPIPS ↓ | PSNR ↑ | SSIM ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| × | × | 29.85 | 0.9492 | 0.9106 | 0.2070 | 19.11 | 0.6412 |
| ✓ | × | 20.51 | 0.9761 | 0.9505 | 0.1074 | 22.38 | 0.7828 |
| ✓ | ✓ | 19.90 | 0.9782 | 0.9532 | 0.1047 | 22.60 | 0.7846 |

解释：

- 大部分定量提升来自 MaskFlow probability path / training framework；
- Soft-Poisson 的增益较小但方向一致，主要是边界视觉质量；
- 论文的全局 metric 只能间接衡量 seam，缺少专门的 boundary-gradient、boundary color discontinuity 或用户感知指标，因此“seamless”的定量证据弱于 qualitative evidence。

### 9.3 去掉文字位置描述

有额外 position description 时 FID 为 29.49；去掉后为 17.21，同时 CLIP、DINO、LPIPS、PSNR、SSIM、VGG 均改善。

合理解释是：当 mask 已提供位置时，额外位置文本构成冗余甚至冲突的 localization channel；去掉它迫使模型真正使用 mask。

但该结果仍可能同时受到 prompt distribution 简化的影响，不能仅凭这一消融证明 attention 一定以预期方式编码了 mask。

## 10. 与 Follow-Your-Shape 的关系

两者都把“区域”提升到 trajectory 层次，但方向相反：

| | Follow-Your-Shape | MaskFlow |
|---|---|---|
| 区域来源 | 从 source/target velocity discrepancy 自动估计 | 用户显式提供 mask |
| 是否训练 | training-free | 训练 attention LoRA，需要 paired regional data |
| mask 的角色 | 控制 source KV injection | 定义 probability path、loss 和 boundary refinement |
| target/source trajectory | 两条完整运行的 trajectory | 在一个空间 latent 内拼接 target path 与 source path |
| velocity 使用 | $\|v_{tgt}-v_{src}\|$ 用于发现区域 | clean estimate 经 Poisson 修正后重建 $v^*$ |
| shape freedom | 自动 mask 可随 trajectory discrepancy 反映扩张 | 由用户 mask/soft transition region 限定 |
| boundary | 主要依赖生成与 feature injection | 显式 gradient-domain Soft-Poisson |

一句话区分：

> Follow-Your-Shape 问“target/source trajectory 在哪里开始分歧”，MaskFlow 问“已知用户指定区域后，怎样让不同区域沿不同 endpoint 运动但仍在边界连续”。

## 11. 与 FlowEdit / trajectory-difference 方法的关系

FlowEdit 构造全图 source-to-target direct transport：

$$
v^{\Delta}=v_{tgt}-v_{src},
$$

并积分这个 relative velocity 推进 editing state。

MaskFlow 不计算 target/source model velocity difference。它预先通过 mask 定义 spatially hybrid endpoint：

$$
x_{endpoint}=m\odot x_{target}+(1-m)\odot x_{source},
$$

然后训练一个 conditional vector field 到达该 endpoint。

因此：

- FlowEdit：**运行时从两个查询结果中提取相对编辑方向**；
- MaskFlow：**训练时直接规定每个空间位置应属于哪条概率路径**；
- Soft-Poisson：进一步把边界 compatibility 写回 velocity field。

## 12. 论文最有价值的研究观点

### Observation

mask 内 generation 和 mask 外 reconstruction 即使各自正确，也会因局部 trajectory 不同在边界形成 discontinuity。

### Interpretation

regional editing 不应只被看作 condition control，而应被看作 **spatially heterogeneous transport problem**：不同空间位置具有不同 clean endpoint，但它们的空间梯度必须兼容。

### Evidence

MaskFlow component 带来主要 localization/background 提升；加入 Soft-Poisson 后定量指标进一步小幅改善，定性边界更平滑。

### Conclusion

论文支持“mask 应进入 probability path”这一设计；但“Soft-Poisson 在所有任务上解决 semantic incompatibility”仍未被充分证明。Poisson objective 本质上约束局部梯度与 source/edit anchor，不能保证物体几何、遮挡、光照因果关系或语义边界一定正确。

## 13. 局限与未解决问题

1. **依赖显式 mask**：精确但增加用户输入；mask 错误会直接写入 probability path。
2. **mask 是硬语义边界还是允许形变的参考？** 方法使用 transition band，但主体生成仍由原 mask 约束。大幅 shape expansion、new shadow、reflection、occlusion 可能需要超出原区域。
3. **训练数据约 10K**：规模有限且由合成 target、SAM 与人工 refinement 构造，可能存在 generator bias 和 mask style bias。
4. **评估主要在自建 MEData**：训练和评估分布的关联可能放大优势，需要外部真实用户 mask benchmark。
5. **Background MSE/LPIPS 为 0.0000**：这很可能与最终 pixel blending 或显示精度有关。若使用硬 source copy，背景 metric 接近零并不能证明模型自身的 vector field 学会了 background preservation。
6. **Soft-Poisson 的计算成本**：每个 timestep 50 次 Jacobi iteration；需要报告 wall-clock、显存、分辨率扩展性。
7. **latent gradient 不等于 RGB perceptual continuity**：VAE decoder 是非线性的，latent Laplacian 平滑不保证最终像素的颜色和纹理严格连续。
8. **边界 metric 不充分**：缺少专门针对 narrow boundary band 的 gradient、LPIPS、color difference 和 human preference evaluation。
9. **Eq. (6) 的 mask 外监督解释需要核对**：论文声称维持 source path，但公开公式只显式加权 masked residual。
10. **最新预印本风险**：论文和仓库在快速更新，MEData/MaskEdit-10k、backbone 名称和推理配置可能继续变化。

## 14. 对当前研究的启发

### 14.1 将 trajectory difference 与显式 mask path 结合

Follow-Your-Shape 可产生动态 discrepancy mask $M_t^{TDM}$，MaskFlow 使用用户 mask $M^{user}$。可以验证：

$$
M_t^{hybrid}
=\operatorname{Union}
(M^{user},M_t^{TDM})
$$

是否能同时保留用户精确控制与 shape expansion freedom。必须分别评估 mask leakage 与 under-editing，不能只看全局 CLIP。

### 14.2 对 complex multi-instruction 使用多区域 probability path

若有多个指令和 mask：

$$
x(t)=\sum_k m_k\odot
(\alpha x_1^{(k)}+\beta\epsilon)
+\left(1-\bigvee_km_k\right)\odot
(\alpha x_S+\beta\epsilon).
$$

真正困难的是 mask overlap 与跨区域 interaction。SplitFlow/FlowDC 处理的是 semantic velocity conflict，MaskFlow 处理的是 spatial endpoint assignment；二者可以形成互补研究方向。

### 14.3 把 boundary compatibility 作为 trajectory-level constraint

Soft-Poisson 的重要启发不是 Poisson 本身，而是：

> 不要等最终图像生成后才修 seam，而要把兼容性修正重新转换成 velocity，并让后续 trajectory 响应它。

可以进一步研究比 Laplacian 更语义化的 constraint，例如：

- boundary normal/tangential velocity consistency；
- source/target Jacobian matching；
- attention/feature gradient compatibility；
- uncertainty-aware transition width；
- 根据 trajectory discrepancy 自适应调整 $\lambda_e,\lambda_s$。

## 15. 推荐复现实验

为区分各组件真正贡献，建议最小控制变量实验：

1. Base QwenImage-Edit。
2. Standard LoRA fine-tuning，仅把 mask 作为 condition。
3. Mask-guided probability path，不启用 pixel blend/SPD。
4. 在 3 上加入 noisy-source unmasking。
5. 在 3 上仅加入 final pixel blend。
6. 在 3 上加入每步 Soft-Poisson。
7. 只在最后一步执行 Soft-Poisson。
8. 对比 latent Poisson 与 RGB post-Poisson。

固定：checkpoint、seed、prompt、mask、scheduler、CFG、分辨率和 sampling steps。

至少报告：

- edit-region CLIP/DINO；
- background LPIPS/MSE；
- 仅 boundary band 的 LPIPS、gradient discontinuity、color difference；
- mask dilation 后的 leakage curve；
- wall-clock 和显存；
- large shape change、shadow/reflection、text editing、occlusion 四类分组结果。

## 16. 最终评价

MaskFlow 最强的贡献不是提出了另一个 mask-conditioned editor，而是把 regional editing 重新表述成：

$$
\boxed{
\text{spatially heterogeneous probability path}
+
\text{trajectory-level boundary correction}
}
$$

它将 mask 从输入提示提升为 transport endpoint selector，并将 Poisson blending 从最终像素后处理提升为每步 clean-target/velocity refinement。这个故事在方法结构上是连贯的：MaskFlow 负责 precise 与 consistent，Soft-Poisson 负责 seamless。

不过从证据强度看，主要性能提升来自 mask-aware path/training；Soft-Poisson 的专门边界收益尚缺更有针对性的量化，背景近零误差也需要排除 hard source copying 的影响。因而这篇论文既提供了很有价值的 trajectory-level regional editing formulation，也留下了明确的验证空间。

