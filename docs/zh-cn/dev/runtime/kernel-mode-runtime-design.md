# PyPTO kernel mode：运行时与 simpler 接口

[总体方案与执行过程](kernel-mode-plan.md) · [运行时与 simpler 接口](kernel-mode-runtime-design.md) · [PR 依赖与实施计划](kernel-mode-pr-plan.md)

## 5. 内部 kernel 层与 PyTorch 层

### 5.1 目录与责任

| 拟新增/整理的目录 | 责任 |
| ----------------- | ---- |
| `runtime/_execution_mode.py` | 初始化时的进程 mode 准入、防重入及错误状态；不在 import/compile 时认领 |
| `runtime/kernel/context.py` | PyPTO 进程级管理器，唯一 Worker 的 lazy 初始化和强引用、共享 handle 表、配置一致性、并发等待与失败状态 |
| `runtime/kernel/callable.py` | 从编译产物取得加载记录、去重 prepare、构造每次调用的独立快照 |
| `runtime/kernel/abi.py` | simpler 正式接口适配、descriptor/参数长度类型方向与错误转换；不实现底层执行器 |
| `torch/__init__.py` | 必要公共接口导出；延迟加载 torch_npu 和 native 扩展，import 不初始化 Worker |
| `torch/interop.py` | Tensor dtype/shape/stride/format/device/pointer 校验与描述、本次 device/stream、Out/InOut 与 alias 校验 |
| `torch/launch.py` | 向 native adapter 提交 handle、参数快照、Tensor 与 Worker 保活引用；对接框架队列及内部退出协调 |
| `torch/registration.py` | 可选 torch.library/Fake/Meta/mutation/alias 集成，不作为 eager 前置要求 |
| 独立 `torch_npu_adapter` 扩展 | native callback、OpCommand/taskQueue、tensor 引用、recordStream，与通用编译器隔离 |

小文件可以合并；职责不回到一个巨大的 L1 demo。通用 runtime ABI 和编译代码不引用 torch_npu 私有头文件，也不为了框架分层新增 TVM-FFI 或通用插件机制。

#### python/pypto/torch/ 的具体功能与边界

以下为拟新增文件，不代表现有公共 API；首次实现按上述四个 Python 文件组织，native adapter 使用独立的可选 C++ 扩展构建目标。

- **`__init__.py`**：只导出最终确定的 torch 集成接口；按需加载 `torch_npu`/native 扩展。不在 import 时创建 Worker、注册设备资源或触发编译；无 torch_npu 环境仍可使用通用编译与 IR 功能。
- **`interop.py`**：读取并验证 Tensor 的 device、dtype、shape、stride、format 和逻辑起点地址；校验完整 Out/InOut 与 alias，构造借用 storage 的参数描述。每次真实调用取得当前设备和 caller stream，不固定首个 stream，不隐式搬运、转 contiguous 或分配缺失的业务输出。Scalar 类型及 ABI 位置沿用 main 已修复的签名；本层只转换本次值。
- **`launch.py`**：接收 `runtime/kernel/` 已准备好的 callable handle 和本次调用信息，交给 native adapter 提交。保证提交信息携带 Tensor 引用、参数快照及 Worker/native owner 引用。对接框架的提交与退出协调，但不创建 Worker、不执行 prepare、不维护另一份 handle 缓存，也不公开 close/shutdown。具体退出钩子遵循第 5.5 节的待验证契约。
- **`registration.py`**：提供可选的 `torch.ops` 注册支持，描述参数 schema、Out/InOut mutation、返回值和 alias；按目标 PyTorch 版本支持相应的 Fake/Meta 行为。Fake/Meta 只进行元数据推导和校验，不初始化 Worker、不 prepare、不提交设备工作。注册不是普通 JIT 直接调用的前置条件，也不默认提供 autograd；具体注册方式需验证对 mutation/alias 的支持。

**native adapter 的边界**：Python `launch.py` 负责调用衔接，可选 C++ 扩展负责 native 参数所有权、OpCommand/等价队列入口、callback 和 allocator 的 stream 使用记录。实际 simpler native launch 在 callback 中调用。Host 参数与 Tensor 引用必须覆盖延迟 callback 窗口，设备异步使用按 recordStream 或等价协议保护；具体要求见第 5.3 节。

```text
torch.ops 注册入口 / 普通 op(...)
  → torch.interop：参数校验与当前设备/stream
  → PyPTO JIT：隐式编译或缓存恢复
  → runtime/kernel：进程唯一 Worker、prepare、共享 handle 缓存
  → torch.launch + native adapter：框架队列、参数快照与异步保活
  → simpler native launch
```

公共入口的调度由 JIT/kernel executor 串接以上步骤；目录拆分不要求 `interop.py` 反向调用编译器。Worker 创建、持有、callable 注册与 close 的权威状态全部保留在 `runtime/kernel/`；torch 层只提供框架适配及生命周期协调。

对应验证包括：import 无 Worker 初始化；当前 stream 随调用变化；Tensor/alias 校验；`torch.ops` 与直接调用进入同一 kernel 执行路径；Fake/Meta 无设备副作用；延迟 callback 的参数与 storage 保活；退出时与唯一 Worker 管理器正确协调。编译、prepare 和生命周期验证复用第 10 章用例，不重复实现通用 Scalar 测试。

#### 进程级 Worker 与算子级 callable 的所有权

```text
进程 P：PyPTO 内部管理器（强引用持有）
  ├─ 唯一 kernel Worker W
  └─ 共享 callable 注册表
       ├─ op_A / specialization_0 → handle_A0，属于 W
       ├─ op_A / specialization_1 → handle_A1，属于 W
       └─ op_B / specialization_0 → handle_B0，属于 W
```

第一次真实 kernel 调用触发 Worker 懒创建与初始化；后续算子仅查询或新增自己的 callable 注册项。改变 Scalar 值或 stream 不新增 Worker；新算子或新 specialization 也不新增 Worker。管理器保持 Worker 的强引用，不依赖任何单个 JIT 算子对象的寿命；删除一个算子不关闭共享 Worker。

初始化锁覆盖进程内所有算子，handle 注册锁按 callable identity 去重，两者作用不同。已有 Worker 的 device/runtime/config 不兼容时明确报错，不通过另建实例绕过；本轮沿用每卡一个进程的部署方式，不引入每设备或每 runtime 一个 Worker 的注册表。不同 rank/进程分别由各自 PyPTO 管理器持有一个 Worker。

失败回滚和进程退出按统一生命周期协议处理；Worker 状态不确定时阻止继续提交，不由单个算子私自重建。fork 后禁止复用继承的 Worker/handle，子进程的可用性遵循底层 runtime 的进程初始化限制。program 的现有高级 Worker 接口不在此处重设计。

### 5.2 lazy 路径与热路径

从 torch 入口到设备完成的完整时序见第 2.4 节；下面仅展开 kernel executor 内部的简化职责。

```python
# 说明性伪代码，helper 名称不是已存在的公共 API。
def kernel_invoke(compiled, values):
    bound = compiled.signature.bind_all(values)  # 必须包含 Out/InOut
    frame = framework_adapter.describe_call(bound)  # 本次设备、stream、tensor 引用
    compiled.validate_target_and_configuration(frame)
    state = get_process_kernel_state()  # PyPTO 持有，所有算子取得同一状态
    worker = state.ensure_worker(compiled.runtime_config, frame.device)  # 进程唯一
    loaded = state.ensure_callable(worker, compiled)  # 按产物 identity 去重 prepare
    return framework_adapter.enqueue(loaded, bound, frame)
```

稳定签名和 dtype/方向 metadata 在创建 JIT/compiled 时预解析。热路径不反复解析签名、编译、准备相同 callable 或同步。runtime Scalar 值与 tensor 地址每次编码成独立快照，不能复用仍被延迟 callback 引用的可变 Host buffer。

init/load/prepare 的相同 key miss 由一个执行者处理，其他调用共享结果；成功后发布可用状态，失败回滚或保留明确失败状态。prepare 只是注册/资源准备，**绝不能调用一次业务算子做 warmup**，否则 InOut cache 会被额外修改。

可继续采用附件的 PyPTO 层 prepare 去重建议；但纯注册返回值、handle 布局等以冻结后的 simpler 正式 ABI 为准。PyPTO 保存映射不等于另造 device 注册器。

### 5.3 stream、外部 storage 与 native callback

每次真实 torch 调用获取当前设备/stream，compiled 不能固化第一次流；内部 init/prepare 无用户 stream，launch 才传本次流。获取 stream 的目标版本接口需证明不会隐式 drain 队列；不把 raw-stream 参数变成普通用户的必填项。

有序路径应为：torch A 入队 → PyPTO callback 入队 → torch B 入队。真正 native launch 在 PyPTO callback 中发生，不能裸 ACL launch 后补一个空命令。callback 不执行 Python 编译/prepare，不依赖重新取 GIL。

native adapter 接 simpler 正式可调用 native 接口，并在内部持有 owner/launcher 的有效引用；不固定上一版提出的 capsule/vtable 形态，更不自行解析 runtime DLL 私有符号。跨绑定 ABI、线程上下文、错误和所有权属于接入前必须冻结的细节。

外部 Tensor 至少覆盖以下窗口：Host 返回到 callback 结束由 native 参数副本和 Tensor 引用保活；callback 后的异步设备使用由正确 stream 上的 recordStream 或等价 allocator 协议保护。按唯一 storage 处理别名。它保护业务 Tensor，不是开放 runtime workspace allocator。

若部分 enqueue 失败且无法证明内部流已 join 回 caller，单靠 caller recordStream 不足以释放 storage。需按 simpler 的失败/完成协议持有在途引用、标记失效并传播错误；不能吞错、reset 外部设备或猜测已经安全。这个错误窗口的处理局限于 PyPTO 自己的调用，不扩展为管理所有外部 graph 的新系统。

### 5.4 输出、workspace 与所有权

业务输入、输出和 InOut cache 外部传入；正式入口不分配缺失业务输出。动态调度中的 heap、task window、中间 Tensor 与复用由 simpler 内部管理，没有 per-op workspace 查询、分配接口或外部 allocator 回调。

已有 program Worker 的 alloc_tensor 等管理业务 Tensor 的功能可以保留，不能与“不给算子暴露 workspace allocator”混淆。特定算子已有普通 scratch Tensor 参数是否属于业务接口，按其 DSL 签名处理；本方案不新增统一 workspace 参数要求。

不公开新的 close/shutdown。内部引用、缓存和 runtime 自有完成协议共同维持资源寿命，不承诺 Python GC 触发立即卸载，不销毁 caller stream、不 reset 借用设备。解释器退出时不通过已拆除的 Python 模块强制做全设备同步。

如果当前 runtime 尚不能可靠安全回收某类资源，应保守保留并记录容量/能力限制，继续完善 runtime 所有权协议；不把“用户手动保证所有图销毁再 close”追加为正式接口要求。Kernel task 每次必须有界完成，代码/资源驻留不等于 AICPU 常驻服务。

### 5.5 Worker close 的触发时机与退出顺序

**目标是在进程正常退出时，由 PyPTO 的进程级管理器统一执行一次安全收尾。** kernel Worker 从首次真实调用初始化后持续复用，不在单个算子返回、算子对象 GC、一次 capture/replay 结束或缓存项移除时关闭。普通用户不需要调用 close/shutdown。

安全退出顺序如下；它是需要在目标 torch_npu/simpler 版本上验证的生命周期契约，不代表当前依赖已具备可靠退出钩子：

```text
进程进入正常退出，PyPTO 管理器开始收尾
  → 停止接受新调用，协调正在进行的编译/init/prepare/提交
  → 框架停止新的 graph replay 提交
  → 排空已接受的 torch Host callback
  → 等待在途 eager 与 graph replay 的设备工作完成
  → 确认相关图不再使用 callable 资源，按底层协议完成图的释放
  → 调用唯一 Worker 的 close
  → close 成功后清空 handle 注册表及管理器持有的相关资源引用
  → torch/runtime 继续销毁队列、stream 和设备上下文
```

管理器内部采用幂等的收尾状态，重复通知不重复 close，也不能在收尾中重新创建 Worker。Worker 未初始化时无需关闭；部分初始化失败走对应回滚路径。close 失败保留失败状态及仍需存活的引用，不能发布“已安全关闭”的状态。

**不能直接将 `atexit(worker.close)` 当成完整实现。** 需要由 PyPTO 与框架的内部退出集成保证上述顺序，尤其是框架资源还有效、Host callback 已排空且不会再 replay。单纯阻止新的 PyPTO Python 调用并不能阻止已捕获图 replay；一次设备同步也不能保证之后没有新提交。具体退出钩子、执行线程及图资源释放顺序属于待验证的接入细节，不依赖 Python 模块的析构顺序碰巧正确。

若退出时框架上下文已拆除，或无法确认异步使用已经结束，不强行调用依赖这些资源的 close，也不提前释放仍在使用的 callable；保留到进程终止，由进程/驱动清理机制处理。异常终止不保证 Python 收尾执行。此回退不等于 close 已成功，驱动清理行为及资源释放结果需要在部署环境验证。

本方案不新增公开 close、要求用户配对 init/close，或建立追踪所有外部 graph 的管理系统。正常退出的安全收尾由内部生命周期集成负责；尚未完成该集成时，应明确记录限制，而不是将收尾条件转交普通算子用户。

## 6. capture 的正式处理方式

本版撤回“capture 内只读命中，冷调用提前拒绝”的产品规则，也不要求用户先调用 prepare 或业务 warmup 才能进入 capture。

允许普通接口在首次 capture 内进入 lazy init/load/prepare 尝试。实现仍须尊重实际 ACL/CANN 对分配、stream/event 和 capture 的限制：保留底层阶段与错误原因，控制失败回滚，不是假装所有操作天然 capture-safe。

建议以以下状态分别实测，而非用统一 `cache_only` 规则替代：

| 进入 capture 时状态 | 验证内容 |
| ------------------- | -------- |
| 尚未编译、未初始化 | 首次 JIT/装配/init/prepare 哪一步可完成，若失败记录实际 API/阶段及资源状态 |
| 内部缓存已有 generated 产物、binary 尚未就绪 | 验证 kernel 内部补编译与装配；测试以缓存夹具构造此状态，不要求用户调用 `.compile()` |
| binary 已就绪、尚未 prepare | 首次加载/注册与 capture 传播的真实限制 |
| callable 已加载 | 多节点、不同 scalar 快照、stream/event 传播与重复 replay |

如果需要 capture 状态查询，应放在 torch adapter/具体受限操作的诊断边界；不能据此自动退出 capture、偷偷执行业务、切 program、自动重新捕获或要求新的公开准备仪式。查出真实限制后修复对应内部操作；未通过的组合如实记为未支持/失败，而非改变范围宣称完成。

图 owner 仍是框架/使用者；改变 shape、地址、按值 scalar 或 Host task 结构后的图更新由其处理。PyPTO 不新增追踪所有外部 graph 引用、证明全进程设备静默的管理器。异步 replay 与其他调用共享设备资源的执行顺序由 simpler 保证，Python lock 无法覆盖 replay。

## 7. 进程级 mode 互斥

```text
UNCLAIMED → INITIALIZING(program) → PROGRAM
          → INITIALIZING(kernel)  → KERNEL
```

两个箭头竞争同一进程准入。定义、分析和 compile 不进入状态机。即使 program 在设备 0、kernel 在设备 1，也不允许同一进程同时初始化。普通 JIT、compiled、one-shot runner、显式 ChipWorker、分布式 Worker 初始化都须覆盖；simpler 对直接调用其入口的情况提供最终防护。

PyPTO 的状态机用于提早报错和协调初始化，不能取代 simpler 的权威状态。失败若完成回滚且 runtime 明确未认领，才可回到 UNCLAIMED；已经锁定 mode 或清理不确定时保留相应失效状态，不能伪装可改 mode。新的公开切换/close API 不在方案内。

PID/generation 与加载记录关联；fork 后不能继续使用继承的 native 状态。不同 rank 在各自进程初始化，不能通过继承 handle 复用注册资源。

两种 mode 的 runtime 测试用独立进程，不通过改 mode 复用一个测试 worker。program 沿用现有执行协议；本轮不设计 simpler pipeline，也不将其改造作为 kernel 接入的前置条件或验收项。

## 8. simpler 依赖与支持规格

### 8.1 simpler 向 PyPTO 提供的 kernel 接口

这里的“对外”指 **simpler → PyPTO**，不是要求 torch 算子用户直接调用 simpler。PyPTO 持有进程唯一 Worker；Python 侧负责 init/prepare，native adapter 的 callback 调用同一 Worker 对应的 native launch。下表采用附件中的 L2 Worker **目标签名**；当前 pin 的支持情况另见 8.2，不能把提案当成已经导出的 API。

| 目标接口 | 输入 | 输出 / 语义 | PyPTO 调用时机与约束 |
| -------- | ---- | ----------- | -------------------- |
| `Worker(level=2, execution_mode="kernel", device_id=..., platform=..., runtime=...)` | 设备、目标平台、runtime 身份；kernel 模式为 PyPTO 内部固定选择 | Worker 对象；构造与设备初始化分开 | 本进程第一次真实 kernel 调用，内部管理器只构造一个；后续算子共用，不向用户暴露 mode 参数 |
| `worker.init(config=...)` | 与编译产物一致的 context 固定配置；`config` 的正式名称/类型需冻结 | 成功返回 `None`，失败抛异常；创建 runtime 常驻资源及内部流 | 进程首次初始化一次；无 callable、无 caller stream，不执行业务、不 reset 外部设备 |
| `worker.kernel_mode_supported` | 无业务参数；能力查询的可用阶段由正式 ABI 定义 | `bool` 或正式能力描述；不能把单个 true 当成完整平台/runtime/capture 矩阵已通过 | 初始化/能力校验边界查询，避免热路径重复；能力获取本身不执行业务 |
| `worker.kernel_prepare_callable(chip_callable)` | PyPTO 编译并组装的完整 ChipCallable：代码、签名、目标/runtime/ABI 信息 | opaque callable handle；装载 Host 资源，上传所需 device code 并注册；不执行算子 | 进程共享 handle 表 miss 时调用；不接 caller stream，不要求 PyPTO 分配 callable ID；同一产物的去重由 PyPTO 完成 |
| `worker.kernel_launch(handle, args, caller_stream=...)` | 当前 Worker 的 handle、本次借用 Tensor 描述及按值 Scalar、当前 caller stream | 成功表示已提交，通常返回 `None`；不表示设备工作已完成 | 每次业务调用使用本次参数；torch 路径通过 callback 调用其 native 等价入口，不能在 Python 先 launch 再补队列 |
| `worker.close()` | 无业务参数；必须满足退出时序与线程约束 | 释放 Worker 自有资源，使 handle 失效；失败必须可诊断 | 由 PyPTO 按第 5.5 节统一收尾，普通用户不调用；不销毁借用的 caller stream，不 reset 外部设备 |

`kernel_prepare_callable` 的目标语义是纯注册：重复调用可以产生新的 handle，不要求 simpler 提供 lookup。PyPTO 的共享缓存保证同一产物不重复 prepare；失败不发布 handle。prepare 返回不等于设备上传已完成，simpler 必须建立注册到 launch 的就绪依赖。

`kernel_launch` 不接收 `eager/capture` 或 mode 参数，不重新编译、不重新选择 runtime。capture 诊断位于 PyPTO/框架适配边界；内部动态内存与资源回收由 simpler 管理，不增加 per-op workspace 查询、外部 allocator 或 pipeline 接口。

#### 参数对象与所有权

| 对象 | 必要内容 | 所有权与有效期 |
| ---- | -------- | -------------- |
| Worker 初始化配置 | platform/runtime/ABI 相容信息及固定资源配置 | PyPTO 从产物和部署配置建立；后续不兼容算子报错，不另建 Worker |
| ChipCallable | orchestration SO、AICore binaries、参数签名、目标/runtime、完整 identity 所需 metadata | PyPTO 生成并保持装载所需资源；simpler 拥有注册后的 runtime 资源。具体复制/借用边界必须由正式 ABI 说明 |
| Callable handle | opaque 注册标识及其 owner/generation 关联 | 仅属于创建它的 Worker；不解释为可跨 Worker 使用的裸整数，不落盘、不跨进程复用 |
| 每次 launch 参数（本文称 `KernelArgs`） | 每个 Tensor 的逻辑起点设备地址、dtype、shape、支持的 stride/layout、方向/别名信息，以及按声明 dtype 编码的 Scalar | PyPTO 在 callback 执行前持有独立快照；simpler 接收后按正式协议保存其所需参数。外部 Tensor storage 始终借用 |
| caller stream | 本次框架原生 stream 句柄 | 框架持有；init/prepare 不接收、不保存为默认 launch stream；每次 launch 显式传入 |

`KernelArgs`、配置字段与 native handle 包装是契约名称，尚不是可直接 import 的类型承诺。Tensor/Scalar 的分池与 ABI 位置由产物签名和 simpler 参数协议共同确定，不能将 Python 参数列表或浮点值随意转成整数数组。

#### native callback 可调用的边界

simpler 还需向 PyPTO 的可选 C++ adapter 提供 **Worker 对应的 native launch 入口及 owner 保活方式**。它与 Python launch 执行同一协议，不是一个接收 callback 的新调度器：

```text
PyPTO native callback
  → simpler 的受支持 native launcher(handle, args, caller_stream)
  → simpler 内部 runtime C ABI
  → CANN / 设备执行
```

正式 native 符号、类型和导出形式待双方冻结，本方案不杜撰一个已存在的 C++ 方法名。该入口必须允许在约定的 torch 队列 callback 线程执行、无需 Python/GIL，并明确错误返回、参数接收完成点、device 绑定要求及 owner 有效期。init/close 的线程要求与 launch 分开规定；PyPTO 不能因设置 DeviceGuard 就推定所有线程均可调用。

OpCommand/taskQueue 由 PyPTO adapter 和 torch_npu 管理。simpler 不接收 OpCommand、不替 PyPTO 构造 callback；PyPTO 不直接加载 runtime 私有符号，也不自行创建 DeviceContext 或维护 simpler 内部流。

#### 编译期 SDK 能力

为让 PyPTO 掌握编译生命周期，simpler 还需要提供稳定的 SDK 信息：公共头文件/库、目标及 runtime ABI 描述、必要的编译链接参数、callable/参数描述格式。这里列的是接入所需能力，具体查询 API 未冻结。PyPTO 使用这些信息生成代码与二进制；上传、注册和执行仍走上表。当前 `simpler_setup.KernelCompiler` 的继承关系及迁移见第 4.3 节，不假设已有一个新的 SDK 查询函数。

### 8.2 目标接口与当前 pin 的差异

本次核对仍以文首记录的 simpler gitlink 为基线，以下均为源码检查结果，未声称设备验证通过。

| 当前源码入口 | 已观察到的事实 | 与上述目标的差异 |
| ------------ | -------------- | ---------------- |
| [Worker 构造](https://github.com/hw-native-sys/simpler/blob/748c39fbe7825e8411f44e91b754a3b2449f633c/python/simpler/worker.py#L4626) | 当前为 `Worker(level=..., **config)` | 接收任意 config 不等于实现了 kernel execution_mode；当前 Worker 中未找到上述 kernel prepare/launch/能力属性 |
| [Worker.init](https://github.com/hw-native-sys/simpler/blob/748c39fbe7825e8411f44e91b754a3b2449f633c/python/simpler/worker.py#L7727) | 当前公开配置参数为 `prewarm_config` | 不能将目标签名 `init(config=...)` 当成现有可调用形式；kernel 的固定配置语义需正式接入 |
| [Worker.register](https://github.com/hw-native-sys/simpler/blob/748c39fbe7825e8411f44e91b754a3b2449f633c/python/simpler/worker.py#L6735)、[run](https://github.com/hw-native-sys/simpler/blob/748c39fbe7825e8411f44e91b754a3b2449f633c/python/simpler/worker.py#L11217)、[unregister](https://github.com/hw-native-sys/simpler/blob/748c39fbe7825e8411f44e91b754a3b2449f633c/python/simpler/worker.py#L7324) | 存在已有 program 注册、运行和注销接口 | 保留给 program；不能将其视为已支持借用 torch NPU Tensor 与 caller stream 的 kernel 接口 |
| [Worker.close](https://github.com/hw-native-sys/simpler/blob/748c39fbe7825e8411f44e91b754a3b2449f633c/python/simpler/worker.py#L11872) | 已有终止/失败清理协议；native teardown 要求 init-owner 线程 | kernel 生命周期及 torch 退出集成仍须核对；不能在任意 atexit 线程直接调用 |
| [runtime C API 声明](https://github.com/hw-native-sys/simpler/blob/748c39fbe7825e8411f44e91b754a3b2449f633c/src/common/worker/runtime_c_api.h#L600) | 已声明 `simpler_kernel_mode_supported/init/prepare_callable/launch`；旧 prepare 含外部 callable_id 与 caller_stream | 属于 simpler 内部底层边界；与“prepare 返回 opaque handle、无 stream”的 L2 目标不同，不要求 PyPTO 绕过 Worker 直接接旧接口 |
| [onboard C API 实现](https://github.com/hw-native-sys/simpler/blob/748c39fbe7825e8411f44e91b754a3b2449f633c/src/common/platform/onboard/host/c_api_shared.cpp#L1253) | `supported` 返回 0，init/prepare/launch 为未支持或无有效 kernel context 的错误路径 | 声明存在不等于 kernel 能力已接通；需要 simpler 正式实现后才能验收 |

因此实现前应冻结上述 Python/native 两侧契约并升级依赖，再执行对应集成测试。这里列出现有 program 接口仅用于区分复用边界，不展开 simpler 的全部多层 Worker、远程内存或调度 API。

### 8.3 支持规格与验收约束

| 契约 | PyPTO 对接方式与验收要求 |
| ---- | ------------------------ |
| init/prepare/launch | 两个准备入口无用户 stream，launch 带本次 stream；只准备、不额外执行业务 |
| 所有底层入口的进程 mode 防护 | PyPTO 与直接 simpler 初始化交叉测试；不同设备不能绕过 |
| native callback 支持 | 冻结 ABI、句柄所有权、callback 线程可用性及设备上下文要求；DeviceGuard 本身不是线程安全证明 |
| kernel 共享资源串行 | eager、多个 callable、不同 stream、graph replay 共享资源时由 runtime 建立实际设备依赖 |
| 动态中间内存 | runtime 管理申请/回收/复用与背压；PyPTO 不写分配器或向外导出 workspace API |
| 驻留代码容量 | 8192 条目，device code 总量不超过 2 GiB，按需 2 MiB 粒度申请；不固定预留 512 MiB 大 arena |
| 安全释放与错误 | 只处理自有资源；部分失败不能被吞掉，未证实完成的在途引用不能提前释放 |

代码容量与 task heap 分开统计；上限所属注册/执行域应写入 ABI。粒度涉及的取整计费、重复注册、失败回滚和第 8193 项需要边界测试，可先用计数/mock 验证，不要求每个 UT 真的分配 2 GiB。

当前 pin 的 `supported=0` 和 init/prepare/launch stub 是现状，不能通过忽略错误跳过。正式 ABI 发布和 runtime 实现是跨仓前置工作；新清单并未证明它们已经完成。之前文档中的旧容量数字、旧 prepare 参数和单平台 demo 不能替代这里的目标规格。
