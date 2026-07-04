# 疑难杂症总索引（跨章节）

全教程的排查案例按症状归类。遇到问题先查这里，再翻对应章节。

## 环境与加载

| 症状 | 章节 | 核心 |
|---|---|---|
| `Torch not compiled with CUDA` | 0.7 | 装了 CPU 版 / Mac 用 MPS |
| `import torch` 架构错误 | 0.7 | Apple Silicon 先查架构（x86 vs arm64） |
| `weights only`/pickle 安全 | 4.2 | `torch.load(weights_only=True)` |
| 权重"加载成功"效果像没训 | 4.7 | strict=False 静默丢 key，先对 key |
| DDP/compile 前缀不匹配 | 4.6 | `module.`/`_orig_mod.` 前缀 |

## Python 与张量

| 症状 | 章节 | 核心 |
|---|---|---|
| exit 137（OOM 无输出） | 1.7 | `__getitem__` 缺 IndexError / 无限循环 |
| `AttributeError module has no attribute` | 1.7 | 文件名遮蔽库名 / 版本不符 |
| `view size incompatible` | 2.7 | 上游 transpose，用 reshape 或 contiguous |
| 显存缓慢上涨 OOM | 2.7、3.7 | `total+=loss` 未 detach / hook 存张量 |
| dtype 静默提升吃显存 | 2.4 | fp32 带维度张量参与运算 |

## Autograd 与训练

| 症状 | 章节 | 核心 |
|---|---|---|
| `.grad` 是 None | 3.7 | 中间节点/requires_grad/图断连 三层查 |
| inplace 操作打断反向 | 3.7 | 版本计数器，改非原地 |
| 第二次 backward 报错 | 3.7 | 图已销毁，多半是跨迭代复用 |
| loss 不降 | 6.7 | 过拟合单 batch 黄金测试 |
| loss spike / 震荡 | 9.7、15.4 | **β₂=0.95 对小模型是毒药、best-checkpoint 保险** |
| loss 变 NaN | 6.7、8.7、15.4 | fp16→bf16、NaN 哨兵二分定位 |
| 周期性 loss 跳动 | 6.7 | 找同周期事件源（eval 忘 train() 等） |

## 模型与注意力

| 症状 | 章节 | 核心 |
|---|---|---|
| 训练指标好但生成差 | 7.7、8.7 | 因果泄漏（mask 在 softmax 后 / 数据错位） |
| 注意力输出 NaN | 7.7 | 全 -inf 行 / logits 上溢 |
| 深了训不动 | 8.7 | 残差投影初始化 /√(2L)、看逐层激活剖面 |
| 换长度就崩/变差 | 8.7、10.7 | learned pos 越界 / RoPE 外推 |
| 加载官方权重乱码 | 10.7 | QKV 排布/RoPE 配对，逐层对拍定位 |

## 数据管线

| 症状 | 章节 | 核心 |
|---|---|---|
| mac 上 num_workers>0 崩 | 5.4 | spawn 需 `if __name__=='__main__'` |
| 所有 worker 相同随机数 | 5.4 | worker_init_fn 重设 numpy 种子 |
| GPU 利用率 30% 慢 | 5.7、11.7 | 数据饥饿诊断（合成数据对照） |
| 训练卡住无报错 | 5.7 | worker 死/`/dev/shm` 满，py-spy dump |
| encode 卡死（O(n²)） | 9.4 | 分块 encode（GPT-2 按词切） |

## 性能与显存

| 症状 | 章节 | 核心 |
|---|---|---|
| GPU 利用率高但慢 | 11.7 | 利用率骗人，算 MFU + profiler |
| torch.compile 反而慢 | 11.7 | 没 warmup / 反复重编译 / graph break |
| OOM / 显存时涨时降 | 11.7、15.4 | 显存四类定位 + 菜单 |
| bf16 autocast 慢/无收益 | 6（AMP） | 硬件×负载依赖，M4 上常打平 |

## 分布式

| 症状 | 章节 | 核心 |
|---|---|---|
| 多卡 hang 无报错 | 12.7、15.4 | **找谁没到齐**（集合通信 rank 不齐） |
| 多卡效果比单卡差 | 12.7 | 忘 DistributedSampler / lr 没 scale |
| NCCL 报错/超时 | 12.7 | `NCCL_DEBUG=INFO` |
| FSDP 显存比预期高 | 13.7 | 包裹粒度 / prefetch / state_dict 类型 |
| 3D 并行结果错 | 13.7 | process group 分组错，逐维度定位 |

## 微调与推理

| 症状 | 章节 | 核心 |
|---|---|---|
| 微调后复读指令 | 14.7、15.4 | SFT loss 掩码（指令部分 -100） |
| LoRA loss 不降 | 14.7 | 加错层 / B 没初始化 0 / lr 太小 |
| QLoRA 显存仍爆 | 14.7、15.4 | 显存菜单依次开 |
| KV cache 后结果变了 | 16.7、17.7 | 位置索引错 / 块表映射错（正确应完全一致） |
| 量化后精度大降 | 16.7 | group size / outlier / 敏感层 |
| 推理吞吐低 | 16.7、18.7 | 先看 batching 是否充分 |

## 通用方法论

1. **环境问题**：先跑 `chapters/ch00_environment/code/check_env.py`（版本/设备/dtype 三件套）
2. **loss 问题**：先过拟合单 batch（第 6 章黄金测试）
3. **显存问题**：先分四类（模型状态/激活/缓冲/碎片，第 11 章）
4. **分布式 hang**：找"谁没到齐"，不是找"谁错了"（第 12 章）
5. **数值不一致**：二分定位第一个发散的层（第 10、16 章）
6. **性能问题**：先定位主导项（Roofline/profiler），再优化（第 11 章）
7. **静默 bug 最贵**：报错是慈悲；广播/掩码/掩膜/dtype 的静默错误要靠 shape 注释 + 对拍防御
