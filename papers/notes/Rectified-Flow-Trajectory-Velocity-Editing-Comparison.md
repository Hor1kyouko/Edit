# Rectified Flow 编辑中的 Source/Target Trajectory 与 Velocity Difference

> 横向文献比较：不同工作如何解释并使用 source/target trajectory、velocity difference 与 flow residual。

## 1. 核心结论

很多论文都使用近似形式

\[
\Delta v_t=v_\theta(x_t^{tgt},t,c_{tgt})-v_\theta(x_t^{src},t,c_{src}),
\]

但实际解决的是四类不同问题：

1. **定位：哪里应该编辑？** DiffEdit、Follow-Your-Shape、VeloEdit、SteerFlow。
2. **动力学：编辑 trajectory 应朝哪里走？** FlowEdit、SplitFlow、FlowDC、FlowDirector。
3. **优化：当前候选结果偏离理想 flow/path 多少？** DDS、RFDS、DRFS。
4. **平衡：怎样在 source preservation 与 target editing 间切换？** DNAEdit、SteerFlow、VeloEdit、FlowSlider。

所以不能看到 \(v_{tgt}-v_{src}\) 就直接称其为“编辑方向”；必须检查 velocity 的 evaluation state、condition、noise coupling、最终用途和实际被积分的变量。

## 2. 基本概念与混淆来源

Rectified Flow 定义条件速度场：

\[
\frac{dx_t}{dt}=v_\theta(x_t,t,c).
\]

- \(v_\theta(\cdot,t,c)\)：latent space 上的 velocity field；
- \(x_t\)：某条 trajectory 在时间 \(t\) 的位置；
- \(v_\theta(x_t,t,c)\)：该状态上的瞬时 velocity；
- trajectory 是 velocity 随时间积分的结果，不等于单步 velocity。

source/target velocity 通常来自同一个模型：

\[
v_t^{src}=v_\theta(x_t^{src},t,c_{src}),\qquad
v_t^{tgt}=v_\theta(x_t^{tgt},t,c_{tgt}).
\]

它们同时改变 state 和 condition，因此一般的 \(\Delta v_t\) 混合了：

\[
\text{prompt effect}+\text{state drift}+\text{path mismatch}+\text{model error}.
\]

| 比较 | 控制变量 | 更合理的解释 |
|---|---|---|
| \(v(x_t,c_{tgt})-v(x_t,c_{src})\) | same state, cross prompt | semantic steering |
| \(v(x_t^{tgt},c_{src})-v(x_t^{src},c_{src})\) | same prompt, cross state | trajectory drift / source stabilization |
| \(v(x_t^{tgt},c_{tgt})-v(x_t^{src},c_{src})\) | state、prompt 都变 | mixed editing discrepancy |
| \([v_\theta-\dot x]_{tgt}-[v_\theta-\dot x]_{src}\) | 比较 flow residual | target/source path mismatch difference |

## 3. 方法总表

| 方法 | 核心比较 | 差异用途 | 是否直接推进编辑状态 | 论文故事 |
|---|---|---|---:|---|
| DiffEdit | target/source noise prediction | ROI mask | 否 | prediction difference 暴露编辑区域 |
| DDS | target/source denoising prediction | optimization gradient | 否 | source query 抵消 SDS 共享误差 |
| FlowEdit | target/source RF velocity | direct transport ODE | 是 | 绕过 inversion，缩短 transport path |
| Follow-Your-Shape | 两条 trajectory 的 token velocity | TDM mask | 否 | trajectory divergence 揭示 shape-changing region |
| DRFS | target/source flow residual | 优化 target image | 否 | 同时减 predicted velocity 与 prescribed path velocity |
| DNAEdit | linear/source；target/source velocity | noise alignment；moving reference | 间接 | 先纠正 source coupling，再平衡 fidelity/editability |
| SplitFlow | 多个 subtarget/source velocities | 多条 ODE、projection、aggregation | 是 | 复杂 prompt 拆成多个相对 flows |
| FlowDC | 多个 subtarget/source velocities | 编辑子空间、正交衰减 | 是 | 保留目标子空间，移除 irrelevant velocity |
| VeloEdit | predicted / analytic keep velocity | mask、replacement、strength control | 是 | 相对 source reconstruction velocity 分解编辑作用 |
| FlowSlider | cross-prompt 与 cross-state difference | 连续强度控制 | 是 | 分离 semantic steering 与 fidelity stabilization |
| SteerFlow | target/source absolute velocities | interpolation、mask refinement | 是 | 在 source reconstruction 与 target denoising 间插值 |
| MaskFlow | 无显式 target-source model subtraction | mask 分配 endpoint，Poisson 修正 velocity | 是 | spatially heterogeneous path + boundary compatibility |

## 4. Follow-Your-Shape：差异用于定位

对空间 token \(i\)：

\[
\delta_t^i=\left\|v_\theta(z_t^{tgt,i},t,c_{tgt})-v_\theta(x_t^{src,i},t,c_{src})\right\|_2.
\]

论文对 discrepancy 做时间聚合、平滑和阈值化，得到编辑 mask，再用 mask 控制 source KV injection。\(\Delta v\) 不作为 ODE update direction；它回答 **where to edit**，不是 **how to move**。

限制是 state 与 prompt 同时改变，因此 TDM 是综合 trajectory discrepancy，而非严格的纯语义 mask。

- [Follow-Your-Shape](https://proceedings.iclr.cc/paper_files/paper/2026/file/6b315c0b736711b56f33cbacfb6d5d67-Paper-Conference.pdf)

## 5. FlowEdit：差异就是 direct editing dynamics

FlowEdit 将 inversion-based editing 重写为：

\[
Z_t^{inv}=Z_0^{src}+Z_t^{tgt}-Z_t^{src},
\]

因此：

\[
\frac{dZ_t^{inv}}{dt}=v_\theta(Z_t^{tgt},t,c_{tgt})-v_\theta(Z_t^{src},t,c_{src}).
\]

实际以共享噪声构造 source state：

\[
\widehat Z_t^{src}=(1-t)Z_0^{src}+tN_t,
\]

并用 parallelogram relation 构造 target query point：

\[
\widehat Z_t^{tgt}=Z_t^{FE}+\widehat Z_t^{src}-Z_0^{src}.
\]

随后积分：

\[
Z_{t-\Delta t}^{FE}=Z_t^{FE}-\Delta t\left[v_\theta(\widehat Z_t^{tgt},c_{tgt})-v_\theta(\widehat Z_t^{src},c_{src})\right].
\]

这里 \(Z_t^{FE}\) 是 source-to-target direct-edit state，而不是普通 noise-to-image latent。共享噪声使共同 noise component 更容易相消。

- [FlowEdit](https://arxiv.org/html/2412.08629)

## 6. DDS、RFDS、DRFS：差异作为优化 residual

### DDS

\[
\Delta\epsilon_t=\epsilon_\theta(x_t^{tgt},t,c_{tgt})-\epsilon_\theta(x_t^{src},t,c_{src}).
\]

差异用于优化梯度，以 source reference query 抵消 SDS 中共享的、与编辑无关的误差。

- [Delta Denoising Score](https://openaccess.thecvf.com/content/ICCV2023/papers/Hertz_Delta_Denoising_Score_ICCV_2023_paper.pdf)

### RFDS 与 DRFS

RFDS 定义模型 velocity 相对规定路径 velocity 的 residual：

\[
r_t=v_\theta(x_t,t,c)-\dot x_t.
\]

DRFS 进一步比较 target/source 两侧完整 residual：

\[
\Delta r_t=(v_t^{tgt}-v_t^{src})-(\dot x_t^{tgt}-\dot x_t^{src}).
\]

并引入 shifted target state：

\[
\widehat x_t^{tgt}=a_tx_0^{tgt}+b_t\epsilon+c_t(x_0^{tgt}-x_0^{src}),
\]

使 target velocity 在更合理的 target forward posterior 附近查询。它说明裸 \(v_{tgt}-v_{src}\) 不一定是正确编辑 residual。

- [RFDS / iRFDS](https://arxiv.org/abs/2406.03293)
- [Delta Rectified Flow Sampling](https://arxiv.org/html/2509.05342)

## 7. FlowSlider：拆开 semantic effect 与 trajectory drift

在 FlowEdit 的 mixed difference 中加减 \(V(z_t^{tgt},t,c_{src})\)：

\[
V^\Delta=V_{steer}+V_{fid},
\]

\[
V_{steer}=V(z_t^{tgt},c_{tgt})-V(z_t^{tgt},c_{src}),
\]

\[
V_{fid}=V(z_t^{tgt},c_{src})-V(z_t^{src},c_{src}).
\]

前者是 same-state, cross-prompt semantic steering；后者是 same-prompt, cross-state source stabilization。连续控制只缩放前者：

\[
V_s^\Delta=V_{fid}+sV_{steer}.
\]

这提示 Follow-Your-Shape 可以分别构造 semantic mask 与 drift map，而不是只看二者混合后的 norm。

- [FlowSlider](https://arxiv.org/html/2604.02088)

## 8. VeloEdit、SteerFlow 与 DNAEdit：reference velocity 路线

### VeloEdit

定义解析 source-preservation velocity：

\[
v_t^{keep}=\frac{x_t-x_{orig}}{t},\qquad v_t^{diff}=v_t^{pred}-v_t^{keep}.
\]

高相似度区域以 \(v^{keep}\) 替换模型 velocity，低相似度区域使用

\[
v_t^{final}=(1-\alpha)v_t^{keep}+\alpha v_t^{pred}.
\]

差异同时用于 region detection、background preservation 与 continuous editing。其风险是 analytic keep velocity 依赖 linear-flow assumption。

- [VeloEdit](https://arxiv.org/html/2603.13388)

### SteerFlow

\[
V_t^{edit}=V_t^{src}+\alpha_t(V_t^{tar}-V_t^{src})=(1-\alpha_t)V_t^{src}+\alpha_tV_t^{tar}.
\]

它积分的是 source/target absolute denoising velocities 的混合，而 FlowEdit 积分 relative velocity。SteerFlow 还用 \(\|V_t^{tar}-V_t^{src}\|\) 扩展 SAM3 base mask。

- [SteerFlow](https://arxiv.org/html/2604.01715)

### DNAEdit

第一种差异用于 source/noise alignment：

\[
\Delta v_t^{DNA}=v_t^{linear}-v_t^{src}.
\]

第二种 \(\Delta v_t=v_t^{tgt}-v_t^{src}\) 用于移动 reference latent，再构造 guidance velocity。前者修 inversion/coupling，后者平衡 editability/fidelity，不能混为同一信号。

- [DNAEdit](https://arxiv.org/html/2506.01430)

## 9. SplitFlow 与 FlowDC：复杂多指令编辑

SplitFlow 对每个 sub-target prompt 构造：

\[
v_t^{\Delta(i)}=v_\theta(x_t^{tgt(i)},t,c_{tgt}^{(i)})-v_\theta(x_t^{src},t,c_{src}),
\]

再通过 Latent Trajectory Projection 和基于 cosine consistency 的 Velocity Field Aggregation 合并 sub-flows。它假设复杂 prompt 的主要问题是多个 semantic gradients 纠缠或冲突。

FlowDC 从多个 parallel editing velocities 构造正交 editing subspace：

\[
u_i=v_i-\sum_{j<i}\operatorname{proj}_{u_j}(v_i),
\]

再保留 in-subspace component、衰减 orthogonal component。它假设失败来自完整 velocity 中混入目标子空间外的 irrelevant direction。

- [SplitFlow](https://arxiv.org/html/2510.25970)
- [FlowDC](https://arxiv.org/html/2512.11395)

## 10. MaskFlow：spatial endpoint assignment

MaskFlow 不显式计算 target-source model velocity difference，而定义：

\[
x(t)=m\odot(\alpha x_{target}+\beta\epsilon)+(1-m)\odot(\alpha x_{source}+\beta\epsilon).
\]

它在训练时规定不同空间位置属于 target 或 source endpoint；Soft-Poisson 再修正 clean-target estimate，并反推出 boundary-compatible velocity。

- Follow-Your-Shape：从 trajectory difference 发现区域；
- FlowEdit：用 trajectory difference 推进 direct edit；
- MaskFlow：由显式 mask 规定区域 trajectory，并修正边界 vector field。

- [MaskFlow](https://arxiv.org/html/2608.06929)

## 11. 统一框架

\[
v_t^{final}=v_t^{base}+G_t\odot\Phi(v_t^{tgt},v_t^{src},\dot x_t^{tgt},\dot x_t^{src}).
\]

- \(G_t\)：空间/时间 gate；
- \(v_t^{base}\)：source reconstruction、target generation 或零基准；
- \(\Phi\)：raw difference、residual difference、projection、interpolation 或 replacement。

典型选择：

- Follow-Your-Shape：\(G_t=\operatorname{Mask}(\|v_{tgt}-v_{src}\|)\)，gate 作用于 KV injection；
- FlowEdit：\(v^{base}=0,\Phi=v_{tgt}-v_{src}\)；
- SteerFlow：\(v^{base}=v_{src},\Phi=\alpha_t(v_{tgt}-v_{src})\)；
- VeloEdit：\(v^{base}=v_{keep},\Phi=\alpha(v_{pred}-v_{keep})\)；
- DRFS：\(\Phi=(v_{tgt}-v_{src})-(\dot x_{tgt}-\dot x_{src})\)。

## 12. 阅读检查清单

1. 两个 velocity 是否在同一个 latent state 上查询？
2. source 和 target 是否共享 noise coupling 与 timestep convention？
3. difference 被当作 norm、direction、gradient、residual 还是 mask？
4. 最终积分的是 absolute velocity、relative velocity，还是另一个 optimization variable？
5. 是否校正 prescribed path velocity 与 off-trajectory target query？

## 13. 建议的验证实验

在同一 backbone、noise、scheduler 与 timestep 下分别计算：

\[
D_{semantic}=\|v(x_t^{tgt},c_{tgt})-v(x_t^{tgt},c_{src})\|,
\]

\[
D_{drift}=\|v(x_t^{tgt},c_{src})-v(x_t^{src},c_{src})\|,
\]

\[
D_{mixed}=\|v(x_t^{tgt},c_{tgt})-v(x_t^{src},c_{src})\|.
\]

分别与 ground-truth edit mask、实际 latent displacement、最终 pixel change 和 background leakage 做相关分析，从而验证 mixed TDM 的有效性究竟主要来自 semantic condition difference，还是已经发生的 trajectory drift。

## 14. 未解决问题

1. raw \(v_{tgt}-v_{src}\) 中 semantic effect、state drift 与 model error 的可识别性不足。
2. shared noise 能降低 variance，但不保证建立了最优语义 pairing。
3. token-wise velocity norm 与最终像素编辑区域之间缺少严格因果证明。
4. multi-instruction 方法对 conflict、orthogonality、independent semantic direction 的假设主要是经验支持。
5. preservation 常由 KV reuse、velocity anchoring、hard latent replacement、pixel copy 等机制共同产生，实验必须拆分因果贡献。

