# 1 Transformer 复盘

## 1.1 架构回顾

- **Encoder**：自注意力 → FFN。
- **Decoder**：因果自注意力 → Cross-Attention → FFN。
- **Decoder-only**：Embedding → 多层「因果自注意力 + FFN」→ 最终归一化 → 词表投影。
- Attention、FFN 子层配有残差连接和归一化。原始 Transformer 使用 Post-Norm，许多现代 Decoder-only 模型使用 Pre-Norm。
- 各 Block 输入、输出通常保持 $N\times D$，便于残差相加与堆叠。

以下省略 batch 维度：$N$ 为序列长度，$D$ 为模型维度，$H$ 为头数，每头维度 $d=D/H$。

## 1.2 Attention：公式与维度

### 1.2.1 $S=QK^\top$：两两匹配

对单个头：

$$
Q,K,V\in\mathbb R^{N\times d}
$$

$$
S=QK^\top:\quad(N\times d)(d\times N)=N\times N
$$

$S[i,j]$ 是第 i 个 Query 与第 j 个 Key 的点积。

$$
A=\operatorname{softmax}\left(\frac{S}{\sqrt d}+M\right)
$$

$M$ 为掩码；Softmax 按行计算，每行权重之和为 1。

### 1.2.2 $O=AV$：加权求和

$$
O=AV:\quad(N\times N)(N\times d)=N\times d
$$

$$
O[i]=\sum_j A[i,j]V[j]
$$

**理解提示：V 的每一行是一个向量。** $AV$ 就是把 $a_1v_1+a_2v_2+\cdots$ 对所有 Query 一起计算。

**O 的含义与用途：** 每一行是对应 token 按自身权重从上下文汇总的 d 维信息。各头输出经过拼接、输出投影和残差相加，更新 token 表示，再交给 FFN 和后续层处理。

### 1.2.3 Cross-Attention：分数矩阵不要求是方阵

Q 来自 Decoder，K、V 来自 Encoder。设目标、源序列长度分别为 $N_t,N_s$：

$$
Q\in\mathbb R^{N_t\times d},\qquad
K,V\in\mathbb R^{N_s\times d}
$$

$$
S=QK^\top\in\mathbb R^{N_t\times N_s}
$$

$$
O=AV\in\mathbb R^{N_t\times d}
$$

**行数对应 Query 数量，列数对应 Key 数量；两边长度不同时就是非方阵。输出长度跟随 Q。**

## 1.3 MHA：拆分与拼接

先进行可学习的投影：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

在常见标准 MHA 配置下：

$$
Q,K,V:\quad N\times D
\rightarrow H\text{ 个 }N\times d
$$

- **Q、K、V 都沿特征维度拆头**，每个头仍保留全部 token。
- 每头独立计算权重 $A_h$，得到 $O_h=A_hV_h$。
- 各头输出沿特征维度拼接，再做输出投影：

$$
O=\operatorname{Concat}(O_1,\ldots,O_H)W_O
$$

$$
H\text{ 个 }(N\times d)
\rightarrow N\times D
\rightarrow N\times D
$$

例如 $D=512,H=8$：每头输出 $N\times64$，拼接后为 $N\times512$。

## 1.4 正弦/余弦位置编码与 RoPE

### 1.4.1 正弦/余弦位置编码

$$
PE(p,2r)=\sin\left(\frac{p}{10000^{2r/D}}\right)
$$

$$
PE(p,2r+1)=\cos\left(\frac{p}{10000^{2r/D}}\right)
$$

p 表示位置，r 表示维度对的索引。将位置编码加到 Token Embedding 上，维度仍为 $N\times D$。

### 1.4.2 RoPE

将每头 Q、K 的维度两两配对，按 token 位置旋转，不同维度对使用不同频率。维度不变，标准 RoPE 不旋转 V。

对其中一对维度，以列向量记法：

$$
\tilde q_i=R(i\theta)q_i,\qquad
\tilde k_j=R(j\theta)k_j
$$

$$
\begin{aligned}
\tilde q_i^\top\tilde k_j
&=q_i^\top R(i\theta)^\top R(j\theta)k_j\\
&=q_i^\top R((j-i)\theta)k_j
\end{aligned}
$$

**推导提示：** $R(\alpha)^\top=R(-\alpha)$，连续旋转的角度相加，因此出现相对位置 $j-i$。

**只旋转 Q、K 的原因：** 位置通过 Q、K 影响注意力权重；V 提供被加权汇总的内容。

## 1.5 FFN：函数与实现

FFN 对每个 token 独立应用同一套参数，保持序列长度不变。

### 1.5.1 标准 FFN

$$
\operatorname{FFN}(X)=\sigma(XW_1+b_1)W_2+b_2
$$

$$
N\times D
\rightarrow N\times D_{\mathrm{ff}}
\rightarrow N\times D
$$

原始 Transformer 的激活函数 $\sigma$ 为 ReLU。

### 1.5.2 门控 FFN：以 SwiGLU 为例

```python
gate = silu(x @ W_gate)
up = x @ W_up
mid = gate * up
out = mid @ W_down
```

| 对象 | 维度 |
|---|---|
| `W_gate`、`W_up` | $D\times D_{\mathrm{ff}}$ |
| `gate`、`up`、`mid` | $N\times D_{\mathrm{ff}}$ |
| `W_down` | $D_{\mathrm{ff}}\times D$ |
| `out` | $N\times D$ |

**理解提示：** `*` 是逐元素乘法，`@` 是矩阵乘法。此处 `mid` 是中间结果，三个可训练投影通常称为 **gate、up、down**。