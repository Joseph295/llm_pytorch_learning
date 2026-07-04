# 第 0 章 · 环境与工具链：地基打对，后面不塌

> **本章目标**：搭好一套贯穿全教程的 Python/PyTorch 环境，并且——更重要的——理解这套工具链每一层的**因果逻辑**。学完本章你应该能回答：
> 1. Python 的依赖管理为什么比 Maven/Gradle 的世界混乱得多？uv 解决了什么？
> 2. `nvidia-smi` 显示 CUDA 12.4，装的 torch 是 cu121 编译的，能跑吗？为什么？
> 3. 你的 Mac 上 `torch.cuda.is_available()` 返回 `False`，是装错了吗？
> 4. 一台 24GB 内存的 M4 Mac，在大模型学习中能做什么、不能做什么？

**前置要求**：无。这是全教程第一章。
**硬件路径**：本地 M4 全程完成。
**预计用时**：2~3 小时（含跑通全部代码）。

---

## 0.1 来龙去脉：从 Maven 的秩序到 Python 的丛林

### 你熟悉的世界是什么样的

作为 JVM 工程师，你习惯的依赖管理是这样的：`pom.xml` / `build.gradle` 声明依赖 → 从中央仓库拉 jar → 打成 fat jar 或 assembly → 扔到任何装了对应 JDK 的机器上就能跑。这套秩序建立在两个前提上：

1. **字节码是平台无关的**——jar 包里没有本地二进制，不关心你是 x86 还是 ARM；
2. **classpath 是应用级隔离的**——两个应用的依赖天然互不干扰。

### Python 世界把这两个前提都打破了

**前提一被打破**：科学计算生态里的 Python 包（NumPy、PyTorch）本质上是 **C/C++/CUDA 代码的 Python 外壳**。PyTorch 的安装包里，Python 代码只占很小一部分，剩下的是编译好的本地二进制（libtorch、CUDA kernel、Metal shader）。这意味着包必须区分平台：Linux x86 + CUDA 是一个包，macOS ARM 是另一个包——这就是 wheel 文件名里 `macosx_11_0_arm64` 这类平台标签的由来。

**前提二被打破**：Python 的 `import` 默认从**解释器级别**的 `site-packages` 目录找包——相当于所有应用共享一个全局 classpath。项目 A 要 `numpy 1.x`、项目 B 要 `numpy 2.x`，装在同一个解释器里必然打架。

这两个问题催生了 Python 生态二十年的工具演化：`virtualenv`（应用级隔离）→ `pip` + `requirements.txt`（声明依赖但不锁版本、不管 Python 本身）→ `conda`（连 Python 和 C 库一起管，但慢且和 pip 混用有坑）→ `poetry/pdm`（引入 lockfile）→ **`uv`**（2024 年出现，Rust 编写，把"管理 Python 版本 + 虚拟环境 + 依赖解析 + lockfile"全部收编，速度比 pip 快 10~100 倍）。

**本教程选 uv**，理由用你的语言说：它相当于 Python 世界终于有了一个 "Gradle + SDKMAN 合体"——一个工具管完工具链和依赖，且 `uv.lock` 提供可复现构建。

### PyTorch 的特殊麻烦：三层版本耦合

PyTorch 不是普通的包。在 NVIDIA GPU 机器上，它的运行依赖三层东西的版本兼容：

```
NVIDIA 驱动 (随系统装，如 550.x)
    ↑ 决定了最高支持的 CUDA 版本
CUDA Runtime (torch 的 wheel 里自带！如 cu121 = CUDA 12.1)
    ↑ torch 针对某个 CUDA 版本编译
PyTorch 本体 (如 2.7.x)
```

这套东西你应该似曾相识——**Hadoop 生态的版本兼容矩阵**（Spark 3.x 配 Hadoop 3.x 配 Hive 3.x，错一格就 `NoSuchMethodError`）。区别在于 PyTorch 把中间层打进了自己的 wheel：**装 torch 不需要在系统里装 CUDA Toolkit**，wheel 自带 CUDA runtime。系统里唯一需要的是驱动，而驱动向后兼容——这是第 0.6 节那道高频面试题的答案来源。

### 你的 M4 Mac 在这个版图里的位置

Mac 没有 NVIDIA GPU，但 Apple Silicon 有强力 GPU 和一个杀手锏：**统一内存（Unified Memory）**。PyTorch 通过 **MPS 后端**（Metal Performance Shaders）驱动 Apple GPU，你的 24GB 内存是 CPU 和 GPU **共享**的——不存在独显那种"模型太大装不进显存"的硬墙（代价是带宽低于高端独显显存）。

能做什么 / 不能做什么，直接决定了本教程的硬件路径设计：

| | 本地 M4 (24GB 统一内存) | 云端 NVIDIA GPU |
|---|---|---|
| 张量/autograd/nn 全部基础实验 | ✅ 完全够 | 不需要 |
| 从零预训练 miniGPT（第 9 章） | ✅ 千万参数级模型无压力 | 不需要 |
| 7B 模型推理（量化后） | ✅ 借助统一内存可跑 | 更快 |
| 多卡分布式训练（第 12/13 章） | ⚠️ 只能用 gloo 后端模拟通信语义 | ✅ NCCL 实战必须上云 |
| FlashAttention / Triton / CUDA 算子 | ❌ CUDA 生态独占 | ✅ 必须上云 |
| 7B 全参数微调（第 15 章） | ❌ 算力和生态都不够 | ✅ 上云 |

> **一句话**：M4 学"原理与实现"绰绰有余，"工业级规模"的章节按讲义标注上云，单次实验费用会控制在几十元人民币量级。

---

## 0.2 核心原理：这套工具链的每一层在干什么

### venv 的本质：没有魔法，只有 PATH

Java 的隔离靠 classpath 指来指去，Python 的 venv 更简单粗暴——**它就是一个目录**：

```
.venv/
├── pyvenv.cfg          # 记录基础解释器是谁
├── bin/python          # 指向基础解释器的软链
└── lib/python3.12/site-packages/   # 本项目专属的"依赖仓库"
```

"激活"一个 venv 本质上只做一件事：把 `.venv/bin` 挂到 `PATH` 最前面，于是 `python` 命令解析到这个 venv 的解释器，它的 `sys.path`（Python 的 classpath）指向这个 venv 自己的 `site-packages`。仅此而已。

uv 在这之上给你三个保证：
1. **`pyproject.toml`** 声明直接依赖（角色 = `pom.xml`）；
2. **`uv.lock`** 锁定全量传递依赖的精确版本和哈希（角色 = 依赖锁定文件，**必须提交进 git**——这是环境可复现的根基）；
3. **`uv run xxx.py`** 免激活直接在正确环境里运行（角色 ≈ `./gradlew run`，永远不会"用错解释器"）。

### wheel：Python 的"预编译制品"

wheel（`.whl`）是预编译的二进制分发格式，类比平台相关的 native 制品包。看懂文件名就看懂了平台机制：

```
torch-2.7.1-cp312-cp312-macosx_11_0_arm64.whl
      └版本┘ └Python版本┘ └───平台标签────┘
```

pip/uv 安装时自动挑选匹配当前平台的 wheel。**在 macOS ARM 上，PyPI 默认的 torch wheel 就是带 MPS 支持的版本，无需任何额外操作**；在 Linux 上，默认 wheel 捆绑 CUDA runtime（所以有 2GB+ 这么大——里面塞着为多代 GPU 架构预编译的 kernel）。

### 设备抽象：一份代码，三个后端

PyTorch 用 `torch.device` 把计算后端抽象掉了，这是你要建立的第一个 PyTorch 心智模型：

```python
device = torch.device(
    "cuda" if torch.cuda.is_available()      # NVIDIA GPU
    else "mps" if torch.backends.mps.is_available()  # Apple GPU
    else "cpu"
)
x = torch.randn(1024, 1024, device=device)   # 张量创建在指定设备上
```

写代码时永远面向 `device` 变量编程，同一份训练代码本地 MPS 调试、上云 CUDA 跑大规模——这是全教程代码的统一约定。

需要提前知道的 MPS 限制（后面章节会反复遇到）：
- **不支持 float64**（深度学习用不到 fp64，但 NumPy 默认 fp64，交互时容易踩，见 0.4 节）；
- 少数长尾算子未实现，可用环境变量 `PYTORCH_ENABLE_MPS_FALLBACK=1` 回退 CPU 执行（有性能代价）；
- 没有 NCCL——多卡通信是 NVIDIA 生态的东西，Mac 上分布式实验用 gloo 后端走 CPU 内存。

### 异步执行：GPU 编程的第一个思维转变

有一件事必须在第 0 章就说清楚，因为它影响你怎么"计时"：**GPU 操作是异步的**。`torch.mm(a, b)` 在 GPU 上调用后立即返回，计算在后台排队执行——像你熟悉的异步提交任务，拿到的是"future"而不是结果。只有当你真正读取数据（如 `.item()`、打印、拷回 CPU）或显式调用 `torch.mps.synchronize()` 时才会等待完成。

所以**给 GPU 代码计时必须先同步**，否则你测到的是"提交任务的耗时"而不是"计算耗时"——本章的 benchmark 代码会演示，第 11 章性能优化会深挖这个话题。

---

## 0.3 动手实验

### 实验 1：初始化环境（一次性）

本仓库已用 uv 初始化（`pyproject.toml` + `uv.lock` 在仓库根目录）。克隆或拉取仓库后只需一条命令：

```bash
cd ~/llm_pytorch_learning
uv sync          # 读取 uv.lock，精确复现环境（含 Python 3.12 本身）
```

> 国内网络如果下载慢，用镜像：`UV_DEFAULT_INDEX="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple" uv sync`

### 实验 2：环境体检

```bash
uv run chapters/ch00_environment/code/check_env.py
```

这个脚本会检查：Python/torch 版本、可用设备、MPS 对各数据类型（fp32/fp16/bf16/fp64）的支持情况，并在最优设备上做一次真实计算。**逐行读它的代码**——它演示了设备无关代码的标准写法。你的 M4 预期输出要点：

- `cuda available: False`——**这不是错误**，Mac 没有 NVIDIA GPU；
- `mps available: True`——你的 GPU 后端；
- `float64 on mps: ✗`——亲眼确认 0.2 节说的限制。

### 实验 3：CPU vs MPS 性能基准

```bash
uv run chapters/ch00_environment/code/mps_benchmark.py
```

对不同尺寸的矩阵乘法分别在 CPU 和 MPS 上计时（注意代码里的 warmup 和 `synchronize`——0.2 节讲的异步执行在这里落地）。你会看到一个值得深思的现象：

- **大矩阵（2048+）**：MPS 明显更快，数倍加速；
- **小矩阵（256 以下）**：MPS 可能**反而更慢**。

为什么？GPU 每次启动 kernel 有固定开销（微秒级），小任务的计算量摊不平这个开销——**和你在大数据里见过的"小文件问题"是同一个道理**：任务调度开销超过了任务本身。记住这个直觉，第 11 章讲 kernel fusion、第 17 章讲 continuous batching，本质都是在"摊薄固定开销"。

### 实验 4：云 GPU 平台踩点（只注册，不花钱）

第 12/13/15/17 章需要云 GPU。现在先注册好账号、熟悉界面，到时候直接开工。选平台的参考（价格随行情波动，以平台实时为准，量级如下）：

| 平台 | 典型机型与量级价格 | 适合 |
|---|---|---|
| AutoDL（国内） | RTX 4090 约 ¥2/时；A100-80G 约 ¥6~10/时 | 性价比高，教程默认推荐 |
| 阿里云/腾讯云 | 同规格通常更贵，但企业环境常见 | 想顺便熟悉国内云厂生态 |
| RunPod / Lambda（海外） | A100 约 $1.5~2/时；H100 约 $2~3/时 | 需要海外支付方式 |

**费用心理预期**：第 12 章双卡通信实验约 2 小时 × 双 4090 ≈ ¥10~20；第 15 章 7B QLoRA 微调约 3~5 小时 × A100 ≈ ¥30~50。全教程云端总花费预估 **¥100~300**。

**三条纪律**（写下来，血泪教训都在这）：① 实验完**立即关机释放**，云 GPU 按小时计费不看你在不在用；② 选官方 PyTorch 镜像（如 `pytorch/pytorch:2.x-cuda12.x`），别从裸系统装环境；③ 数据和 checkpoint 存到平台的持久化盘/对象存储，实例盘随释放销毁——这条对你是常识，但云 GPU 平台的"系统盘/数据盘"分界比大数据集群更容易踩。

---

## 0.4 易错点清单

每条按 **错误做法 → 现象 → 原因 → 修正** 四元组给出。

**① 用系统 Python 直接 pip install**
```bash
pip install torch        # ✗
```
→ **现象**：macOS/新版 Linux 报 `error: externally-managed-environment`；老系统上则默默污染全局环境。
→ **原因**：PEP 668 禁止向系统解释器直接装包（保护系统工具依赖的 Python）。
→ **修正**：永远在项目里 `uv add <pkg>` / `uv sync`。你可以把"裸 pip"理解为"往 JDK 的 lib 目录里手动扔 jar"——能跑，但迟早出事。

**② 在 Mac 上检查 `torch.cuda.is_available()` 得到 False，以为装错了**
→ **现象**：网上教程全用 `cuda`，你机器上返回 `False`，重装三遍还是 False。
→ **原因**：CUDA 是 NVIDIA 专有生态，Mac 上正确的检查是 `torch.backends.mps.is_available()`。
→ **修正**：用 0.2 节的三级设备选择写法。网上教程 `device="cuda"` 硬编码的地方，你替换成 `device` 变量。

**③ MPS 上创建 float64 张量**
```python
torch.randn(3, 3, dtype=torch.float64, device="mps")   # ✗
```
→ **现象**：`TypeError: Cannot convert a MPS Tensor to float64 dtype ...`。
→ **原因**：Metal 后端没有 fp64 算力。最常见的触发路径不是显式写 fp64，而是 **NumPy 数组默认 fp64**，`torch.from_numpy(arr).to("mps")` 时爆炸。
→ **修正**：从 NumPy 来的数据先 `.float()`（转 fp32）再 `.to("mps")`。

**④ 直接 `python xxx.py` 而不是 `uv run xxx.py`**
→ **现象**：`ModuleNotFoundError: No module named 'torch'`，但你明明装了。
→ **原因**：`python` 解析到了系统解释器（另一个 `site-packages`），不是项目 venv。等价于 classpath 没配就 `java MyClass`。
→ **修正**：命令行统一 `uv run`；IDE（VS Code/PyCharm）把解释器指到 `.venv/bin/python`。

**⑤ MPS 报某算子不支持（NotImplementedError）**
→ **现象**：`NotImplementedError: The operator 'aten::xxx' is not currently implemented for the MPS device`。
→ **原因**：MPS 后端算子覆盖不全（长尾算子）。
→ **修正**：`PYTORCH_ENABLE_MPS_FALLBACK=1 uv run xxx.py` 让该算子回退 CPU。**注意**这是性能陷阱：数据要在 CPU↔GPU 间来回搬。偶尔一个算子无所谓，训练热路径上有 fallback 就要换思路（换算子实现或该实验上云）。

**⑥ 给 GPU 代码计时不同步**
```python
t0 = time.perf_counter()
c = a @ b                     # MPS 上异步执行
elapsed = time.perf_counter() - t0    # ✗ 只测到了"提交"耗时
```
→ **现象**：GPU "快得离谱"（比如 4096×4096 矩阵乘 0.1ms），benchmark 数字好看得不真实。
→ **原因**：0.2 节讲的异步执行——操作提交后立即返回。
→ **修正**：计时前 `torch.mps.synchronize()`（CUDA 上是 `torch.cuda.synchronize()`）。`mps_benchmark.py` 里有标准写法。

---

## 0.5 开源项目的最佳实践

**① HuggingFace transformers：可选依赖（extras）的设计**
`transformers` 的 `pip install transformers[torch]` 语法里，`[torch]` 是 extras——核心包不硬依赖任何深度学习框架，torch/jax 都是可选项。看它的 [`setup.py`](https://github.com/huggingface/transformers/blob/main/setup.py) 中 `extras["torch"]` 的定义。**设计动机**：同一个库要服务训练（要 torch）和纯 tokenizer 用户（不要 torch），硬依赖会强迫所有人下 2GB。你未来给团队写工具库时用得上这个模式（uv 里对应 `[project.optional-dependencies]`）。

**② vLLM：生产系统对版本的强 pin**
vLLM 的 [`pyproject.toml`](https://github.com/vllm-project/vllm/blob/main/pyproject.toml) 对 torch 是**精确版本锁定**（如 `torch==2.x.y`），因为它编译的自定义 CUDA 算子和 torch 的 C++ ABI 强绑定，版本差一个 patch 都可能 symbol 冲突。**对比 ①**：库（transformers）追求宽版本区间最大化兼容，应用/系统（vLLM）追求精确锁定保证确定性——你在 JVM 世界见过同样的分野（库的 `[1.0,2.0)` vs 应用的 dependencyManagement 锁死）。

**③ PyTorch 官方 Docker 镜像：生产训练环境的事实标准**
生产训练不会在裸机上 pip install，而是 `pytorch/pytorch:2.x.y-cuda12.x-cudnn9-runtime` 镜像 + 项目 lockfile 两层固化。云 GPU 平台（AutoDL 等）的"官方镜像"就是在这个基础上做的。这和大数据集群统一镜像发布是同一个运维哲学。

**④ 本仓库的实践**：`uv.lock` 已提交进 git。任何时候环境疑似坏了，`rm -rf .venv && uv sync` 即可精确重建——这是"cattle, not pets"在开发环境上的应用。

---

## 0.6 典型面试题

**Q1：`nvidia-smi` 显示 "CUDA Version: 12.4"，但装的 torch 是 cu121 构建的，能正常跑吗？反过来（驱动老、torch 的 CUDA 新）呢？**

> **参考答案**：能跑。`nvidia-smi` 显示的是**驱动支持的 CUDA 上限**，不是"已安装的 CUDA 版本"；torch 的 wheel 自带 CUDA 12.1 runtime，只要 12.1 ≤ 驱动上限 12.4 就兼容（驱动向后兼容）。反过来不行：驱动上限 11.8 跑不了 cu121 的 torch，会报 CUDA 初始化错误，只能升驱动或降 torch 构建。**加分点**：说清系统 CUDA Toolkit 只在需要自己编译扩展（如 flash-attn 从源码装）时才需要，且版本要和 torch 的 runtime 对齐。

**Q2：如何保证一次训练可复现？"环境可复现"分哪几层？**

> **参考答案**：分四层，从外到内：① **镜像层**——Docker 固化系统库/驱动配套/CUDA；② **依赖层**——lockfile（uv.lock/poetry.lock）锁全量传递依赖精确版本；③ **代码与数据层**——git commit + 数据版本化；④ **随机性层**——固定 seed（torch/numpy/python 三处），必要时 `torch.use_deterministic_algorithms(True)`（注意有性能代价，且个别 CUDA 算子本身非确定，如 atomicAdd 的浮点累加顺序）。**加分点**：指出即便四层全做，跨 GPU 型号仍可能有数值差异（不同架构的浮点运算顺序不同）。

**Q3：写一段设备无关的 PyTorch 代码，要求同一份代码在 CUDA/MPS/CPU 机器上都能跑。有哪些常见的"设备泄漏"点？**

> **参考答案**：核心是 `device` 变量化（见 0.2 节三级选择）+ 所有张量创建都显式传 `device=`。常见泄漏点：① 代码中间 `torch.zeros(...)` 没传 device（默认 CPU，后续和 GPU 张量运算报 device mismatch）；② 硬编码 `.cuda()`（应写 `.to(device)`）；③ `torch.cuda.synchronize()` 硬编码（应按后端分派）；④ 从 checkpoint 加载时不带 `map_location`（第 15 章细讲）。

**Q4：为什么 Linux 上的 PyTorch wheel 有 2GB+？既然这么大，为什么官方还是选择把 CUDA runtime 打进 wheel？**

> **参考答案**：大的原因：wheel 里捆绑了 CUDA runtime、cuDNN、cuBLAS 等库，且 kernel 为多代 GPU 架构（sm_70/80/90...）各编译一份（fat binary）。打进 wheel 的原因是**部署确定性**：如果依赖系统 CUDA，用户系统五花八门的 CUDA 安装会制造海量"在我机器上能跑"问题——用空间换掉整类环境问题。这个取舍和 fat jar / 静态链接的逻辑完全一致。

**Q5：conda、pip、uv 各自的定位是什么？混用有什么坑？**

> **参考答案**：pip 只管 Python 包；conda 是通用二进制包管理器（连 Python 解释器和 C 库一起管），历史上解决了科学计算包编译难的问题，但求解器慢、与 pip 混用时两边的元数据互不可见，容易造成同一个包两个版本共存的"薛定谔环境"；uv 用 Rust 重写了 pip+virtualenv+python 版本管理的全链路，速度快且有一等公民的 lockfile。现在 wheel 生态成熟（manylinux/universal2），conda 的历史使命大半已完成，新项目推荐 uv。**混用铁律**：一个环境只认一个管理器。

---

## 0.7 疑难杂症排查

**案例 1：`AssertionError: Torch not compiled with CUDA enabled`**

场景：代码里写了 `.cuda()` 或 `device="cuda"`，在 Mac（或装了 CPU 版 torch 的 Linux）上运行。
排查思路（从上到下）：
1. `python -c "import torch; print(torch.__version__)"` → 版本号带 `+cpu` 后缀？说明装的是 CPU 构建；
2. 在 Mac 上？→ 根本没有 CUDA，改用 MPS（易错点②）；
3. Linux 有 N 卡但版本号不带 `+cuXXX`？→ 从错误的 index 装了 CPU 版，用官方命令重装指定 CUDA 构建。
**本质**：这个报错说的是"你手里的 torch 二进制里没有 CUDA 代码"，是安装问题，不是运行时问题——重启、清缓存都没用。

**案例 2：Mac 上 `import torch` 报 `ImportError: ... incompatible architecture (have 'x86_64', need 'arm64')`**

场景：M 系列 Mac，import 直接崩。
原因：你的 Python 解释器是 x86_64 构建（跑在 Rosetta 转译下），pip 给它装了 x86 wheel，而某些依赖链上出现了 arm64 二进制（或反之），两种架构混载。常见来源：老的 Anaconda x86 安装包、或从 Intel Mac 迁移助理搬来的环境。
排查：`python -c "import platform; print(platform.machine())"` → 输出 `x86_64` 就实锤了。
修正：删掉 x86 的 Python，用 uv 重建（uv 管理的 Python 是原生 arm64）。**教训**：Apple Silicon 上环境问题先查架构，再查版本。

**案例 3：`uv sync` 下载 torch 反复超时/极慢**

排查顺序：① 换镜像源（0.3 节的清华镜像）；② `uv cache dir` 确认缓存盘有空间（wheel 解压后 ~5GB）；③ 公司网络有代理的话设置 `HTTPS_PROXY`。uv 的下载支持断点续传和全局缓存，同一个 wheel 第二个项目复用缓存不会重新下载——这点比 pip 好得多。

**通用方法论**（贯穿全教程的排查第一步）：环境类问题先运行本章的 `check_env.py`，它输出的"版本 + 设备 + dtype 支持"三件套能立刻定位 80% 的环境问题。后续每章的排查案例都假设你已经跑过这一步。

---

## 0.8 练习题

做完再看 `exercises/solutions/` 里的参考答案。**基础题必做**——它们验证你的环境真的可用；进阶和挑战题强烈建议做，它们是第 11 章性能优化的伏笔。

### 基础 1：环境体检报告解读
运行 `check_env.py`，回答：
a) 你的 torch 版本是多少？wheel 是为哪个平台构建的（从版本信息推断）？
b) MPS 对 bf16 的支持情况如何？这对后续章节用 bf16 混合精度训练意味着什么？
c) 把脚本里的设备选择逻辑改成"强制 CPU"，重跑，计算部分的耗时变化了多少？

### 基础 2：亲手触发并修复一个 dtype 错误
写一个 10 行以内的脚本：用 NumPy 创建一个数组，转成 torch 张量并搬到 MPS 上做一次矩阵乘。先写出会报错的版本，看清报错信息，再修复它。（考察易错点③的完整闭环）

### 进阶 1：寻找 CPU/MPS 的性能交叉点
扩展 `mps_benchmark.py`：对矩阵尺寸 64, 128, 256, 512, 1024 做 CPU vs MPS 对比，找出 MPS 开始反超 CPU 的临界尺寸。用"kernel 启动固定开销 vs 计算量"的框架解释你的数据，并回答：这个现象对"小模型该不该用 GPU 推理"有什么启示？

### 挑战 1：验证 MPS fallback 的真实代价
找一个 MPS 未实现的算子（提示：可尝试 `torch.linalg` 家族的长尾函数，或查 PyTorch GitHub 上的 MPS op coverage issue）。分别测量：a) 开启 `PYTORCH_ENABLE_MPS_FALLBACK=1` 时该算子在 MPS 张量上的耗时；b) 直接在 CPU 张量上的耗时。解释为什么 fallback 往往比"直接用 CPU"还慢（提示：数据搬运方向）。

---

## 本章小结与下一章预告

你现在有了一套**可复现**的 PyTorch 环境，更重要的是理解了它的分层逻辑：uv 管环境与依赖（lockfile 是复现的根基）→ wheel 是平台相关的预编译制品（torch 自带 CUDA runtime）→ `torch.device` 抽象计算后端（一份代码三后端）→ GPU 是异步执行的（计时必须同步）。

**下一章（第 1 章）**：写给 JVM 工程师的 Python 速成——不教语法，专讲 PyTorch 代码里高频出现、而 Java 心智模型会让你误判的语言特性：为什么 `model(x)` 一个"对象加括号"能调用前向传播（`__call__`）、`@torch.no_grad()` 装饰器和 `with torch.no_grad():` 为什么是同一个东西的两副面孔（上下文管理器）、以及 NumPy/torch 的广播规则——第 2 章张量的直接地基。
