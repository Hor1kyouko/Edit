# Follow-Your-Shape：Shape-Aware Image Editing via Trajectory-Guided Region Control

> Zeqian Long, Mingzhe Zheng, Kunyu Feng, Xinhua Zhang, Hongyu Liu, Harry Yang, Linfeng Zhang, Qifeng Chen, Yue Ma. ICLR 2026.  
> 正式论文：[ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6b315c0b736711b56f33cbacfb6d5d67-Abstract-Conference.html) · [PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/6b315c0b736711b56f33cbacfb6d5d67-Paper-Conference.pdf) · [arXiv](https://arxiv.org/abs/2508.08134) · [官方代码](https://github.com/mayuelala/FollowYourShape)

## 一句话结论

Follow-Your-Shape 的核心不是“直接预测一个编辑 mask”，而是把 source reconstruction trajectory 与 target editing trajectory 在同一时刻的 token-wise velocity 差异当作“模型想在哪里改变”的内生信号，构造 Trajectory Divergence Map（TDM）；随后用三阶段 Scheduled KV Injection，在背景 token 上复用 inversion 的 Key/Value，在编辑 token 上保留 target Key/Value，从而缓解大尺度 shape replacement 中“改不动”与“背景被重绘”的冲突。

## [KEY FINDING]

论文最有研究价值的贡献是：把 velocity-field discrepancy 从一个全局轨迹误差，转化为空间可定位的 region-control signal。它说明 flow model 的预测速度不仅是求解 ODE 的数值量，也可能携带可用于编辑控制的 token-level semantic change information。

## [WHAT IT MEANS]

这提供了一条区别于 segmentation mask、cross-attention map 和固定时间/层启发式的新路线：从 source/target 条件导致的动力学响应差异中发现 editable region，再用该区域控制 feature reuse。这里的“shape-aware”主要通过允许目标语义在 TDM 前景区重生成、同时冻结背景信息来实现；TDM 本身并不显式编码几何约束。

## [IMPACT ON CURRENT HYPOTHESIS]

对基于 Flow Matching / Rectified Flow 的编辑研究而言，这篇文章支持“trajectory/velocity discrepancy 可作为编辑敏感性信号”的假设，但尚不足以证明该信号专门表示 shape，或它优于其他 feature-space / Jacobian-space discrepancy 的原因。论文给出了结果支持和少量 post-hoc 可视化，却没有完成充分的机制辨识。

## [NEXT EXPERIMENT]

优先做受控分解实验：固定 inversion、solver、prompt、注入层与时间窗，只替换 region signal，比较 velocity difference、latent difference、hidden-feature difference、attention difference 以及随机面积匹配 mask；同时报告 mask localization（与 source/target union mask 的 IoU、precision、recall）和最终 edit/background 指标。这样才能把“区域找得准”与“KV injection 本身有效”分开归因。

## [CONFIDENCE]

中等。方法的工程效果由正式论文、定量表格和公开代码共同支持；但“trajectory divergence 等价于语义差异/shape region”的解释，目前更接近有支持的机制假设，而不是已经严格验证的结论。

---

## 1. 论文要解决什么问题

一般 image editing 的目标是同时满足两个相互竞争的要求：

1. **Editability**：目标对象要真正发生语义和结构变化，例如 swan → boat、horse → dragon；
2. **Preservation**：未编辑区域的布局、纹理、光照和细节尽可能保持原样。

在大尺度 shape transformation 中，这个冲突尤其强。全局复用 source feature/KV 会把生成轨迹拉回 source reconstruction，导致目标形状“改不动”；完全依赖 target prompt 自由去噪，又容易重绘背景、产生 ghosting 或结构漂移。

作者认为已有 region control 有三类局限：

- 外部 binary segmentation mask 边界刚性，并且依赖额外模型和标注质量；
- cross-attention map 会随 token、head、layer、timestep 波动，定位噪声较大；
- unconditional/global KV injection 没有空间选择性，保背景的同时也压制编辑。

因此论文提出的问题可以写成：**能否不训练新模型、也不要求用户提供 mask，而从 source/target 两条 flow trajectory 自身推导 editable region，并据此选择性复用 source KV？**

## 2. 必要背景：Rectified Flow 中的 trajectory 与 velocity

FLUX.1-[dev] 是 flow-based 生成模型。抽象地说，模型学习一个时间相关 velocity field：

$$
\frac{d x_t}{dt}=v_\theta(x_t,t,c),
$$

其中 $x_t$ 是时刻 $t$ 的 latent，$c$ 是文本条件。数值求解这个 ODE，就得到 latent trajectory。

对真实图像编辑，流程包括两条相关但不同的路径：

- **Inversion / reconstruction path**：从 source image latent 出发，使用 source prompt $c_{src}$ 反向走到高噪 latent，记录轨迹、velocity 以及部分 attention KV；
- **Editing / denoising path**：从 inverted latent 出发，用 target prompt $c_{tgt}$ 去噪，得到编辑图像。

重要细节是：论文比较的不是在同一个 latent 上只更换 prompt 后的 velocity，而是分别在两条轨迹各自的 latent 上评估：

$$
v_\theta(z_t,t,c_{tgt})\quad\text{与}\quad v_\theta(x_t,t,c_{src}).
$$

因此 TDM 混合了至少两种因素：

1. condition change，即 $c_{src}\rightarrow c_{tgt}$；
2. state/trajectory drift，即 $x_t\rightarrow z_t$。

这一点很关键：它使 TDM 有机会放大已经发生的编辑偏离，但也意味着不能直接把 TDM 解释成“纯 prompt semantic difference”。

## 3. 核心方法一：Trajectory Divergence Map（TDM）

### 3.1 单步 token-wise divergence

对第 $i$ 个 image token，论文定义：

$$
\delta_t^{(i)}=\left\|v_\theta(z_t^{(i)},t,c_{tgt})-v_\theta(x_t^{(i)},t,c_{src})\right\|_2.
$$

这里每个 token 的 velocity 是一个 feature vector，沿 channel/feature dimension 做 $L_2$ norm，得到一个 scalar；把所有 spatial tokens 重新排列成二维网格，就得到当前 timestep 的 divergence map。

作者的直觉是：

- 目标区域需要重解释/重生成，source 与 target 的速度响应差异大；
- 背景不需要改变，两条轨迹的速度应较相似。

每个 timestep 内再做 min-max normalization：

$$
\tilde\delta_t^{(i)}=
\frac{\delta_t^{(i)}-\min_j\delta_t^{(j)}}
{\max_j\delta_t^{(j)}-\min_j\delta_t^{(j)}}.
$$

这消除了不同 timestep 的绝对量级，使地图落在 $[0,1]$，但代价是丢失了跨时间的绝对 divergence magnitude。即使某一步所有 token 的差异都很小，min-max 以后仍会产生相对“高响应区”。实现中也未显式加入分母 epsilon，极端情况下存在数值稳定性风险。

### 3.2 时间聚合

早期高噪 timestep 的空间结构不稳定，所以论文不直接使用单步 TDM，而在中间编辑窗口 $\mathcal N$ 内收集多张 map。对每个 spatial token，在时间维做 softmax-weighted fusion：

$$
\alpha_t^{(i)}=
\frac{\exp(\tilde\delta_t^{(i)})}
{\sum_{t'\in\mathcal N}\exp(\tilde\delta_{t'}^{(i)})},
\qquad
\hat\delta^{(i)}=
\sum_{t\in\mathcal N}\alpha_t^{(i)}\tilde\delta_t^{(i)}.
$$

公开实现实际上在 softmax 前乘以 scale=5，即 temperature/sharpness 超参数；因此它更偏向保留某个 token 在时间上最强的响应，而不是普通平均。

### 3.3 空间平滑与二值化

聚合 map 先经过 Gaussian smoothing：

$$
\tilde M_S=G_\sigma * \hat\delta,
$$

实现中 $\sigma=0.7$。之后用 Otsu thresholding 自动寻找阈值，得到 binary edit mask：

$$
M_S=\mathbf 1[\tilde M_S>\tau^*].
$$

论文称 TDM 的数值分布通常是“background dominant + long-tailed foreground”的偏斜单峰，因此适合 Otsu。这里需要谨慎：经典 Otsu 更自然地适用于可分的双类直方图；论文只展示有限案例，并没有系统报告阈值稳定性、mask 面积分布或 localization metric。

### 3.4 “mask-free”的准确含义

本文的 mask-free 是指**不需要用户或外部 segmentation model 在推理输入阶段提供 mask**，不是指系统内部完全没有 mask。方法最终明确地产生二值 $M_S$，并用它做 KV blending。因此更准确的表述是：**external-mask-free / self-localized region control**。

## 4. 核心方法二：三阶段 Scheduled KV Injection

方法把去噪过程分成三段。注意论文的 timestep 符号按 $T\rightarrow 0$ 描述，而代码用 loop index 从早期高噪到晚期低噪推进，阅读时不要混淆。

### Stage 1：Initial Trajectory Stabilization

前 $k_{front}$ 个高噪步骤令 $M_S=0$，在所有 image tokens 上使用 inversion path 的 source KV。作用是先把 target denoising trajectory 锚定在 source reconstruction manifold 附近，避免一开始就发生全局语义漂移。

直观上，这一阶段回答“先保住场景骨架”。但如果持续太久，source KV 会压制目标对象的结构变化。

### Stage 2：Editing and TDM Aggregation

中间窗口令 $M_S=1$，允许 target KV 全局参与，给目标语义足够自由度；同时计算并缓存每个 timestep 的 TDM。这个阶段既是“探索目标轨迹”，也是“观察哪些 token 持续偏离 source”。

它带有明显的 bootstrap/self-referential 特征：先允许 target 路径产生差异，再用这个已经产生的差异定义后续允许编辑的区域。优点是区域能适应大尺度形状变化；风险是早期错误漂移也可能被写入 mask。

### Stage 3：Structural and Semantic Conformance

在最后 $k_{tail}$ 个步骤使用聚合后的 TDM mask 混合 KV：

$$
\{K^{\ast},V^{\ast}\}
=M_S\odot\{K_{tgt},V_{tgt}\}
+(1-M_S)\odot\{K_{inv},V_{inv}\}.
$$

随后：

$$
F'_{out}
=\operatorname{Attention}\left(Q_{tgt},K^{\ast},V^{\ast}\right).
$$

也就是说：

- query 始终来自当前 target denoising state；
- edit region 使用 target K/V，让新语义收敛；
- non-edit region 使用 inversion K/V，恢复 source context。

这种做法不是直接替换 latent，也不是最终像素 blending，而是在 self-attention 的信息检索内容上做空间选择。

## 5. 公开代码与论文公式的对应关系

官方实现基于 FLUX.1-[dev]，使用 RF-Solver 风格的二阶更新。关键对应关系如下：

| 概念 | 代码中的实际含义 |
|---|---|
| source trajectory velocity | inversion 阶段保存每一步两次预测的平均 `(pred + pred_mid) / 2` |
| target velocity | 当前 editing latent 上两次 target prediction 的平均 |
| token-wise TDM | `(pred_src - pred_tar).pow(2).sum(dim=-1).sqrt()` |
| spatial grid | image token reshape 为约 $\lceil H/16\rceil\times\lceil W/16\rceil$ |
| temporal fusion | `softmax(delta_stack * 5, dim=0)` 后加权求和 |
| smoothing / threshold | `gaussian_filter(..., sigma=0.7)` + `threshold_otsu` |
| source KV cache | inversion 时将后部 single-stream blocks 的 image K/V offload 到 CPU |
| selective replacement | 先取 source image K/V，再把 `edit_indices` 对应位置替换回 target image K/V |
| attention scope | 主要注入 FLUX single-stream blocks；公开代码条件是 block id `> 19` |

一个容易忽略的实现事实是：FLUX single-stream sequence 由 512 个 text tokens 与 image tokens 拼接。代码只替换 image-token K/V，text K/V 始终来自当前 target pass。这使背景 preservation 不是“恢复整个 source attention state”，而是“保留 target textual condition，同时在背景空间位置恢复 source image memory”。

另外，论文 Table 3 写的是 injecting DiT block start index 19；代码条件为 `info['id'] > 19`，按 0-based index 实际从 block 20 开始。复现时应以代码行为为准，并记录 off-by-one 约定。

## 6. ControlNet 在方法中的角色

完整模型还可加入 depth 与 Canny ControlNet residual：

$$
z'_t=Block(z_t)+\beta\,ControlNetBlock(z_t,c_{cond}).
$$

作者把它描述为 auxiliary structural guidance。默认设置是 normalized denoising interval $[0.1,0.3]$，depth strength 2.5、Canny strength 3.5。不过 quantitative comparison 为隔离 TDM 效果，另外报告了关闭 ControlNet 的版本。

这里存在一个概念张力：若目标是大幅改变 object shape，source depth/edge condition 既可能稳定场景，也可能保留旧形状并妨碍编辑。论文针对部分案例手工选择 None、Depth 或 Depth+Canny，并改变 $k_{front}$ 和 strength，说明“full model”并非完全统一的自动策略。

## 7. ReShapeBench

作者认为 PIE-Bench 等通用编辑 benchmark 混合了 object replacement、style、background change 等任务，无法专门诊断 shape transformation，因此构建 ReShapeBench。

### 数据构成

- 120 张新图：70 个 single-object scenes，50 个 multi-object scenes；
- 每张新图有两种 shape transformation，共 240 个新编辑 case；
- 另有 50-case general set，混合上述样本与筛选后的 PIE-Bench case；
- 总计 290 个 shape-aware editing cases；
- 分辨率统一为 $512\times512$；
- source/target prompt 使用四句模板，主要只替换 foreground object 描述；
- prompt 由 Qwen-2.5-VL 生成并经过人工核验。

### shape transformation 的四个标准

1. **Cross-contour**：外轮廓发生超出局部 affine/warping 的显著改变；
2. **Cross-semantic**：对象跨到新的 semantic class，而不只是颜色/材质属性变化；
3. **Structural transition**：内部 part topology 发生重组；
4. **Subject continuity**：对象仍保留相似 spatial anchor 和场景角色。

作者明确排除单纯 posture/viewpoint change。这一定义把“shape editing”与“object replacement”高度绑定：许多测试同时改变 contour、topology 和 semantic class。因此 benchmark 能测试大幅结构替换，但难以区分性能来自几何编辑能力还是强大的 semantic replacement prior。

## 8. 实验结果应该怎样读

### 设置

- backbone：FLUX.1-[dev]；
- inference：A100 40 GB；
- 论文正文写 14 个 denoising steps，附录超参表写 inference step=15；结合 solver 跳过 final timestep，实际是 $(15-1)\times2=28$ NFE；
- guidance scale=2；
- 默认 $k_{front}=2$，$k_{tail}=3$；
- TDM softmax scale=5，Gaussian $\sigma=0.7$；
- 单图约 65.3 秒；GPU 约 25 GB，CPU 缓存约 12 GB。

### 主要定量结果

在 ReShapeBench 上：

| 方法 | AS↑ | PSNR↑ | LPIPS×1000↓ | CLIP Sim↑ |
|---|---:|---:|---:|---:|
| RF-Edit | 6.52 | 33.28 | 17.53 | 30.41 |
| KV-Edit | 6.51 | 34.73 | 16.42 | 26.97 |
| Ours w/o ControlNet | 6.52 | 34.85 | 9.04 | 32.97 |
| Ours Full | **6.57** | **35.79** | **8.23** | **33.71** |

结果支持以下观察：

- 相比全局 feature/KV reuse，TDM-selective injection 改善了 editability–preservation trade-off；
- 不用 ControlNet 的版本仍明显强，说明增益不全来自额外结构条件；
- Full model 最好，但相对 w/o ControlNet 的增益比相对 baseline 的增益小。

不能直接推出的结论：

- 这些指标不能证明编辑形状本身正确。CLIP 更偏语义对齐，AS 偏自然度；论文没有直接的 contour/topology accuracy metric；
- 背景 PSNR/LPIPS 使用“以主体为中心的固定大小 box”遮掉前景，而不是 ground-truth edit mask。若真实编辑区域超出 box，背景指标会被污染；若 box 过大，又可能忽略大量被重绘区域；
- benchmark、prompt 和方法共同由作者提出，存在 method–benchmark co-design 的有利偏置风险，需要外部数据验证。

### $k_{front}$ 消融

论文测试 $k_{front}=0\ldots4$，$k_{front}=2$ 最优。作者的解释是：太小导致 trajectory drift，太大则 source reconstruction constraint 过强，压制 shape change。

但表中并非简单“越大背景越好、编辑越差”：$k_{front}>2$ 时 PSNR 也下降。因此更合理的解释是，过强早期注入可能使 target trajectory 在后期被迫急剧转向，最终同时损害编辑和重建一致性；这需要 trajectory-distance 曲线或分阶段 error analysis 才能确认。

## 9. Observation / Interpretation / Evidence / Conclusion

### Observation

- source 与 target 的 token-wise velocity difference 在展示案例中对目标区域响应更强；
- 时间聚合、Gaussian smoothing、Otsu 后可形成空间连贯的二值区域；
- 基于该区域选择性注入 source KV，在作者 benchmark 上同时提高背景相似性和文本对齐；
- 早期全局 source KV 注入存在最优短窗口；
- TDM 在视频跨帧时不稳定。

### Interpretation

- velocity field 对 condition/state 改变的局部敏感性可作为“哪里需要编辑”的 proxy；
- 三阶段 schedule 将 early stabilization、middle exploration/localization、late conformance 分开，减少不同目标在同一 timestep 直接冲突；
- K/V 控制的是 attention 可检索的内容，Q 保留 target state，因此比完全替换 attention feature 更具编辑自由度。

### Evidence

- ICLR 2026 正式论文给出两个 benchmark 的 quantitative comparison；
- 有 $k_{front}$、ControlNet timing/strength 消融；
- 公开代码与主要公式总体对应；
- post-hoc 分析展示了一个 TDM 与 source/target SAM union pseudo-mask 的案例，并与 cross-attention 可视化比较。

### Conclusion

**有较强证据支持 Follow-Your-Shape 是有效的工程方法；只有有限证据支持 TDM 是一般性的、因果上更优的 structural localization signal。** 当前应把机制解释标为“supported but under-tested”，而不是 established。

## 10. 优点

- training-free，能直接建立在预训练 Rectified Flow 模型上；
- 不要求用户 mask，区域从编辑动力学中自动产生；
- 方法设计与问题对应清楚：用 region selection 化解 editability/preservation 冲突；
- source/target velocity、timestep、token 与 KV 操作都较明确，便于复现和扩展；
- 同时给出 benchmark、代码、runtime 与 memory 信息；
- 中间产物 TDM 可视化，具备一定 interpretability；
- 论文承认 prompt ambiguity 与视频 TDM instability，而不是把所有失败归为实现问题。

## 11. 局限与尚未解决的问题

### 11.1 TDM 的因果含义没有被拆开

公式同时改变 latent state 和 prompt condition。需要比较：

- $v(z_t,c_{tgt})-v(z_t,c_{src})$：同 state、只改 condition；
- $v(z_t,c_{src})-v(x_t,c_{src})$：同 condition、只改 state；
- 原始 TDM：两者同时改变。

这能判断区域信号主要来自 prompt semantic sensitivity，还是编辑轨迹已经偏离后的 state discrepancy。

### 11.2 TDM localization 证据不足

论文主要用单个 crocodile case 的 pseudo-mask 做 post-hoc 展示，没有在全 benchmark 上报告 IoU、boundary F-score、mask stability 或跨 timestep consistency。因此“准确定位”仍是欠测试的强表述。

### 11.3 TDM 组件缺少完整消融

尚需分别移除/替换：min-max normalization、softmax temporal fusion、scale=5、Gaussian smoothing、Otsu、binary mask，以及只在 late stage 使用 mask的策略。当前无法知道主要增益来自 trajectory signal，还是通用的时空平滑与自适应阈值。

### 11.4 对 prompt 质量敏感

TDM 完全由 source/target prompt 驱动。模糊、多对象指代、关系编辑或复合指令可能产生 diffuse divergence。论文已展示 failure case，但缺少系统的 prompt robustness evaluation。

### 11.5 “shape”与“semantic replacement”纠缠

ReShapeBench 要求 cross-semantic，导致很多任务本质上是 object replacement。若只做同类别非刚性 topology change、局部 part growth/removal、细粒度轮廓约束，TDM 是否仍有效尚不清楚。

### 11.6 计算与存储开销不低

28 NFE、约 65.3 秒、25 GB GPU 与 12 GB CPU 缓存限制了交互式应用。所谓 training-free 不等于 compute-free。

### 11.7 泛化边界

主要实现与超参围绕 FLUX.1-[dev] 和其 single-stream MM-DiT 结构。换到不同 tokenization、不同 flow parameterization、纯 DiT 或 video transformer 时，注入层与 TDM 分辨率可能不再成立；作者也观察到视频跨帧 TDM 波动。

## 12. 对复杂多指令编辑研究的启发

这篇论文对 multi-instruction editing 的潜在价值不只是“生成一个 mask”，而是提供 instruction-conditioned dynamical attribution：每条指令都可能诱导不同的 velocity divergence pattern。

可探索但尚未验证的方向：

1. **Per-instruction TDM**：分别计算每条 atomic instruction 相对 source 的 divergence，得到多个候选 region；
2. **Conflict map**：比较不同指令 TDM 的重叠和 velocity direction cosine，区分空间冲突与方向冲突；
3. **Signed/vector TDM**：不只取 $L_2$ magnitude，保留 velocity direction 或低秩子空间，避免“差异很大但方向不对应目标”的问题；
4. **Continuous gating**：用 soft mask 或置信度加权 KV，而不是 Otsu 二值化；
5. **Online mask revision**：Stage 3 继续监测 divergence，在编辑失败或区域扩张时动态更新 mask；
6. **Layer-time factorization**：研究哪些层的 KV preservation 对 background、identity、geometry、texture 分别负责；
7. **Trajectory causal intervention**：直接在 velocity/hidden representation 上做局部干预，与 KV injection 对比，以识别真正的控制变量。

这些是 exploration hypotheses，不应直接写成已有结论。

## 13. 建议的验证实验（最小改动、清晰归因）

### Experiment A：region signal 对照

固定 solver、seed、prompt、$k_{front}$、$k_{tail}$、注入层和 mask 面积，只替换：

- TDM；
- same-state prompt velocity difference；
- latent difference；
- hidden feature difference；
- cross-attention；
- random connected mask。

同时报告 foreground edit metric、background PSNR/LPIPS、mask IoU、boundary F-score。

### Experiment B：时序聚合机制

固定单步 TDM，比较 mean、max、softmax(scale=1/5/10)、EMA、median；检查 TDM temporal variance 和最终 mask stability。目的不是只找最好参数，而是验证“持续 divergence”是否真的比单次峰值更可靠。

### Experiment C：state 与 condition 分解

计算三种差分并可视化/定量：

$$
D_{cond}=\|v(z_t,c_{tgt})-v(z_t,c_{src})\|,
$$

$$
D_{state}=\|v(z_t,c_{src})-v(x_t,c_{src})\|,
$$

$$
D_{joint}=\|v(z_t,c_{tgt})-v(x_t,c_{src})\|.
$$

这是理解 TDM 机制最关键的实验之一。

### Experiment D：复杂指令冲突

设计两个对象、两条独立指令以及同一区域冲突指令，比较 joint-prompt TDM 与 per-instruction TDM。观察 joint prompt 是否产生 region merging、遗漏或错误归因。

## 14. 阅读时最容易产生的误解

- **误解：TDM 比较整条轨迹的几何距离。** 实际上它逐 timestep 比较 token-wise predicted velocity，再跨时间聚合。
- **误解：TDM 是纯 prompt 差分。** 实际 source/target velocity 在不同 trajectory latent 上计算。
- **误解：mask-free 表示不用 mask。** 实际是不需要外部 mask，内部仍生成 binary TDM mask。
- **误解：ControlNet 是核心贡献。** 核心是 TDM + Scheduled KV Injection；ControlNet 是可选辅助结构条件。
- **误解：KV blending 同时混合 Q/K/V。** 论文公式与代码保留 target Q，只混合 image K/V。
- **误解：定量指标已经直接验证 shape accuracy。** 当前指标主要验证自然度、背景相似度和文本语义，不直接测 contour/topology correctness。

## 15. 总体评价

Follow-Your-Shape 是一篇问题定义清晰、方法直观且可操作性强的论文。它最重要的思想是把 flow trajectory 的局部响应差异当作 region discovery signal，并通过 schedule 将“先稳定、再探索、后约束”组织成一个完整编辑过程。这个工程组合很有说服力，也很适合作为后续 trajectory-aware editing 的基线。

但从科学解释角度，论文仍留下一个核心空白：**TDM 为什么有效，以及它究竟测到的是 prompt-induced semantic sensitivity、trajectory drift、solver discrepancy，还是三者的混合？** 只要这个问题没有通过受控分解和大规模 localization evaluation 回答，就应避免把 TDM 直接等同于 shape semantics。对后续研究而言，这个空白反而是很有价值的切入点。

## 16. 文献整理 skill 检索记录

本笔记借鉴了以下公开 skill 的组织原则，但没有自动安装它们：

1. [literature-review-codex-plugin](https://github.com/eresuntomatito/literature-review-codex-plugin)：强调 source inventory、evidence extraction、claim→evidence→source 可追溯性，适合封闭 PDF 语料的正式综述；
2. [research-systematic-literature-review](https://github.com/yananlong/codex-skills/blob/main/research/research-systematic-literature-review/SKILL.md)：包含 paper-context evidence-map、publication version resolution、confidence grading 和 adversarial checks，适合在单篇论文之外验证 novelty/SOTA；
3. [Literature Review（thematic synthesis）](https://github.com/SkillMedev/skills/blob/main/skills/literature-review/SKILL.md)：强调按共识、争议、证据质量和研究空白组织，而不是逐篇摘要；
4. [literature-review-skill](https://github.com/pinshuai/literature-review-skill)：提供多学术数据库检索与结构化 DOI/citation metadata 工作流，适合后续扩展 related-work corpus。

当前任务是单篇技术论文精读，不需要完整 PRISMA 系统综述。因此最合适的实践是：正式论文为主证据、官方代码做实现交叉核对、外部相关工作只用于定位争议，并明确区分 observation / interpretation / evidence / conclusion。

## 17. 参考与复现入口

- 正式发表页：https://proceedings.iclr.cc/paper_files/paper/2026/hash/6b315c0b736711b56f33cbacfb6d5d67-Abstract-Conference.html
- 正式 PDF：https://proceedings.iclr.cc/paper_files/paper/2026/file/6b315c0b736711b56f33cbacfb6d5d67-Paper-Conference.pdf
- arXiv：https://arxiv.org/abs/2508.08134
- 官方代码：https://github.com/mayuelala/FollowYourShape
- 核心 TDM / solver 实现：https://github.com/mayuelala/FollowYourShape/blob/main/src/flux/sampling.py
- KV injection 实现：https://github.com/mayuelala/FollowYourShape/blob/main/src/flux/modules/layers.py

## 18. BibTeX

```bibtex
@inproceedings{long2026followyourshape,
  title     = {Follow-Your-Shape: Shape-Aware Image Editing via Trajectory-Guided Region Control},
  author    = {Long, Zeqian and Zheng, Mingzhe and Feng, Kunyu and Zhang, Xinhua and Liu, Hongyu and Yang, Harry and Zhang, Linfeng and Chen, Qifeng and Ma, Yue},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}
```
