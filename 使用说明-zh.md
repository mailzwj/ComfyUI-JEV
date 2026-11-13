# JEV-27B-VL ComfyUI 插件使用说明

把 **JEV-27B-VL**（Qwen3.8-27B-VL 基座 + 决策 LoRA + 决策头）作为 ComfyUI 自定义节点使用，
完整移植模型包 `serve_decide.py` 的 `jev27-bare-v1` 决策协议：

- **System 1**：单次前向读数，毫秒级直出决策（注入决策 LoRA，读完即卸载，不污染后续步骤）；
- **System 2**：基座模型先思考再读数（无 LoRA），可自适应触发或强制使用；
- 同时提供普通 **JEVChat** 对话节点（官方 chat 模板、思考通道单独输出）。

输入支持：文本（状态 / 问题 / 选项）+ 图片（任意张，可选）。
输出：概率分布（与选项对齐）、最终选择、完整协议 JSON（含用量与思考明细）。

## 1. 安装

1. 插件位于 `ComfyUI/custom_nodes/ComfyUI-JEV/`（已含在仓库中，无需 pip 安装任何新依赖）。
2. 模型放在 `ComfyUI/models/LLM/JEV-27B-VL/`，目录内容：

   ```
   models/LLM/JEV-27B-VL/
       JEV-27B-VL.safetensors            合并单文件（骨干+视觉+lm_head+MTP，int8 量化，26.5 GB）
       character_lora_bf16.safetensors   决策 LoRA（401 模块）
       decision_head.json                S1 词元表 + 每类 bias
       calibration.json                  每类 softmax 温度
       adapter_config.json
       tokenizer.json / tokenizer_config.json
       config.json / generation_config.json
       preprocessor_config.json / video_preprocessor_config.json
       chat_template.jinja
   ```

   插件加载时把 `models/LLM` 注册为 `llm` 模型目录，其中任意 qwen35 合并单文件都可选。

3. **重启 ComfyUI**（首次安装、改插件代码、或往 `models/LLM` 新增模型文件后都需要）。
   节点类别为 `JEV`，节点名（显示名）：`JEV Model Loader`、`JEV Decide`、`JEV Chat`。

环境要求：ComfyUI ≥ 0.45（需原生 `qwen35_27b` 文本编码器）；Python 依赖用标准 ComfyUI 环境
（torch / transformers / safetensors）即可，本插件不新增任何第三方依赖。

## 2. 节点一览

| 节点 | 作用 | 连线类型 |
|---|---|---|
| `JEVModelLoader` | 加载模型：合并权重 + JEV tokenizer + 401 个 LoRA hook（常驻、未注入） | 输出 `model`（JEVMODEL） |
| `JEVDecide` | 决策：S1 读数 + 可选自适应 S2 思考。**输出节点** | 输入 `model`；输出 `probabilities`（JEVPROB）、`choice`、`info` |
| `JEVChat` | 基座模型普通对话，思考通道单独输出。**输出节点** | 输入 `model`；输出 `thinking`、`text` |

`JEVMODEL` 是自定义连线类型：加载器的 `model` 输出接到任意多个 `JEVDecide` / `JEVChat` 的
`model` 输入，模型只加载一次、全程共享。

## 3. JEVModelLoader

| 参数 | 默认 | 说明 |
|---|---|---|
| `model_name` | `JEV-27B-VL/JEV-27B-VL.safetensors` | 下拉选择 `models/LLM` 下的合并单文件 |

加载期自动完成：qwen35_27b / int8 量化 / MTP 头检测、JEV tokenizer 原子性断言、
决策 LoRA 构造（401 个 bypass hook，此时不注入）、视觉塔 `pos_embed` 反量化
（约 5MB，保证图片路径在所有显存模式下可用）。模型此时尚未上 GPU，首次执行才流式加载。

## 4. JEVDecide（决策节点）

### 三种任务类型 `kind`

| kind | 选项 | 说明 |
|---|---|---|
| `noul` | 固定 `false` / `true` | 是/否判断。`options` 不用填。S2 显示为 `Yes (true)` / `No (false)`，读数后翻转回 `[P(false), P(true)]` |
| `score` | 固定 `0`–`5` | 六档打分。`options` 不用填。禁止 `thinking=auto/on`（协议规定） |
| `choice` | `options` 里填 **2–256** 个选项，**一行一个** | 多选一。>16 个选项时由 `strategy` 决定推理方式 |

### 参数表

| 参数 | 默认 | 说明 |
|---|---|---|
| `model` | — | 接 `JEVModelLoader.model` |
| `kind` | `noul` | 任务类型（见上表） |
| `state` | 空 | 决策状态/上下文文本（多行）。图片紧跟在该文本之后进入提示 |
| `images` | 无 | 可选，`Load Image` 节点输出；多图按顺序排在 `state` 文本之后 |
| `question` | 空 | 要决策的问题 |
| `options` | 空 | 仅 `choice` 用，一行一个选项（标签自动编 A、B、C…Aa…共 256 个） |
| `strategy` | `auto` | 仅 choice >16 选项时生效：`auto`=single（单次前向）；`permute`（乱序 4 次取平均）；`tournament`（分组淘汰赛） |
| `thinking` | `default` | `default`/`off`：只跑 S1；`auto`：S1 领先项概率 < `threshold` 时追加 S2 并折叠；`on`：总是先 S1 再 S2 折叠 |
| `threshold` | `0.0` | **0 = 用协议默认 0.8**（不是 0！）。S1 最大概率低于它才触发 S2 |
| `mix` | `0.0` | **0 = 用协议默认 0.5**（不是 0！）。折叠权重 `p = (1-mix)·p1 + mix·p2` |
| `think_budget` | `0` | S2 思考生成 token 上限；**0 = 协议默认 32768** |
| `reasoning_effort` | `default` | `xhigh` / `medium` / `low`，对应官方模板的思考强度指令 |
| `seed` | `-1` | S2 采样种子；**-1 = 协议种子** `crc32(question+options)`（与 serve 一致），其余值直接用作种子 |
| `use_mtp` | `False` | MTP 投机解码。serve 参考实现是普通解码，**对拍/复现 serve 结果请保持关**；纯本地提速可开 |
| `system2_only` | `False` | 跳过 S1，直接返回 S2 分布（score 类型禁用） |
| `want_reasoning` | `True` | `info` JSON 中是否包含 S2 思考文本 |

### 输出

- `probabilities`：与选项对齐的概率列表（noul 为 `[P(false), P(true)]`，score 为 6 项，
  choice 为 N 项）。可接任意接受列表/数值的下游节点，或仅用于查看。
- `choice`：最终选项文本（如 `true`、`5`、`Paris`）。
- `info`：完整协议 JSON 字符串（`jev27-bare-v1`，与托管 API 响应同构），示例：

  ```json
  {
    "kind": "choice",
    "effective_kind": "choice",
    "options": ["London", "Paris", "Berlin"],
    "probabilities": [0.99982, 0.000101, 0.000083],
    "choice_index": 0,
    "choice": "London",
    "adaptation": "native",
    "protocol": "jev27-bare-v1",
    "model": "JEV-27B-VL\\JEV-27B-VL.safetensors",
    "usage": { "prompt_tokens": 46, "completion_tokens": 0, "total_tokens": 46 },
    "elapsed_seconds": 1.86,
    "num_model_requests": 1
  }
  ```

  `thinking` 不是 `off`/`default` 时额外带（`auto` 未触发时 `used: false`；触发时含思考明细）：

  ```json
  "thinking": {
    "mode": "on", "threshold": 0.8, "budget": 512, "used": true,
    "think_tokens": 63, "think_seconds": 4.2, "finished_within_budget": true,
    "reasoning": "……思考文本（want_reasoning=False 时省略）……"
  }
  ```

  字段含义：

- `adaptation`：S1 读数路径——`native`（noul/score 与 ≤16 选项的 choice，单次前向），
  choice >16 选项时为其策略名（`permute` / `tournament`）。
- `system2_only=True` 的响应没有 `adaptation`/`effective_kind` 字段，
  改带 `"system": 2`，且 `probabilities` 即 S2 分布（不折叠）。

### 典型接法（GUI）

```
Load Image ──images──▶ JEV Decide ◀──model── JEV Model Loader
                            │ choice / probabilities / info（末端查看，如接预览/日志节点）
```

- 纯文本决策：不接 `images` 即可；`state` 可空。
- 带图判断（如"图中是否出现了 X"）：`kind=noul`，`question` 写问题，图接 `images`。
- 多轮/上下文场景：把上下文放进 `state`（协议帧：`[kind]…[state]…[question]…[options]…[decision]:`）。

## 5. JEVChat（对话节点）

| 参数 | 默认 | 说明 |
|---|---|---|
| `model` | — | 接 `JEVModelLoader.model` |
| `system` | 无 | 可选系统消息 |
| `prompt` | 空 | 用户消息 |
| `images` | 无 | 可选，排在 prompt 文本之后 |
| `enable_thinking` | `True` | 思考通道开关（官方模板 `enable_thinking`） |
| `reasoning_effort` | `default` | 思考强度指令（xhigh/medium/low） |
| `max_tokens` | `2048` | 生成上限（含思考部分） |
| `temperature` / `top_k` / `top_p` | `1.0` / `20` / `0.95` | 采样参数（与 generation_config 默认一致） |
| `seed` | `-1` | **-1 = `crc32(prompt)` 派生种子**，其余值直接用作种子 |
| `use_mtp` | `True` | 对话是自由生成，默认开 MTP 提速 |

输出：`thinking`（思考通道内容，无思考时为空串）、`text`（回答正文）。
拆分规则：输出在最后一个思考收尾 token 处切分，其后全部算正文。

## 6. 通过 API 调用（可选）

两个决策/对话节点都是**输出节点**，可直接经 `POST /prompt` 提交，不需要额外预览节点。
结果经 WebSocket 与 `GET /history/{prompt_id}` 的 `outputs` 通道返回。

ComfyUI V3 API 约定（与本项目实测一致）：

1. 请求体为 `{"prompt": {"1": {"class_type": …, "inputs": {…}}, "2": …}}`；
2. **V3 提示词不自动补默认值**：每个输入都必须在 `inputs` 里显式给出（可先查
   `GET /object_info` 取默认值与下拉选项）；
3. 下拉选项值以 `/object_info` 返回为准（Windows 下模型路径是反斜杠形式
   `JEV-27B-VL\JEV-27B-VL.safetensors`）；
4. 节点间连线写成 `["源节点id", 输出序号]`；
5. 图片：先 `POST /upload/image`（multipart，字段名 `image`），把返回的 `name`
   填进 `images` 输入；GUI 里 Load Image 的张量也可经 `input` 目录文件名传入。

最小示例（文本 choice，3 选项）：

```json
{
  "prompt": {
    "1": {
      "class_type": "JEVModelLoader",
      "inputs": { "model_name": "JEV-27B-VL\\JEV-27B-VL.safetensors" }
    },
    "2": {
      "class_type": "JEVDecide",
      "inputs": {
        "model": ["1", 0],
        "kind": "choice",
        "state": "",
        "question": "The capital of France is?",
        "options": "London\nParis\nBerlin",
        "strategy": "auto",
        "thinking": "default",
        "threshold": 0.0,
        "mix": 0.0,
        "think_budget": 0,
        "reasoning_effort": "default",
        "seed": -1,
        "use_mtp": false,
        "system2_only": false,
        "want_reasoning": true
      }
    }
  }
}
```

仓库自带完整示例：`tools/api_run.py`（7 个用例：noul / score / choice / s2 / adaptive /
chat / image，自动取默认值、传图、轮询 history 并打印结果），
`tools/compare_with_serve.py`（与 vLLM `serve.sh` 的 `/v1/decide` 逐选项对拍，需用户侧运行）。

## 7. 性能与内存（RTX 5090 32GB 实测）

- 权重 int8 共 ~26.5GB，由 ComfyUI 的 MinVram/异步卸载机制托管：权重从系统内存按前向流式
  进入显存。系统内存常驻约 28GB，显存峰值约 15–25GB（32GB 卡舒适；显存更小的卡能跑但变慢）。
- LoRA 适配权重（bf16，~0.4GB）只在 S1 hook 注入期间上卡，`finally` 必卸载。
- 实测耗时（同一会话）：首次 noul 7.7s（含首次权重流式）；预热后 S1 约 1.9–2s；
  单图 noul 约 10.5s（含首次图片路径加载，视觉塔一次前向 + 999 prompt tokens）；
  S2 另加思考生成时间（512 预算内数秒）。
- 数值说明：ComfyUI 侧骨干是 int8 convrot 量化，与 vLLM bf16 参考服务相比 S1 读数有
  微小浮点差异（属预期）；hook 注入/卸载本身是位级精确的（实测残差 0.0）。

## 8. 常见问题

- **改了插件或加了模型文件，节点列表没变？** 重启 ComfyUI（节点下拉列表在启动时构建）。
- **`threshold`/`mix` 填 0 是什么效果？** 不是"不思考/纯 S1"，而是"用协议默认值
  0.8 / 0.5"（与 serve 的 0 约定一致）。想关掉 S2 请把 `thinking` 设为 `default` 或 `off`。
- **`seed=-1` 和填具体数字的区别？** -1 表示按协议用 `crc32` 从问题/选项（或对话文本）
  派生种子，保证与参考服务可复现；具体数字则完全由你控制。
- **`use_mtp` 默认值为什么两个节点不一样？** JEVDecide 默认关（对齐 serve 的普通解码，
  保证决策结果与参考一致）；JEVChat 默认开（自由对话只要速度，思考/采样与 MTP 无对拍需求）。
- **图片路径报错 `NoCapableBackendError`？** 旧版本插件未做 `pos_embed` 反量化时，
  服务进程（NORMAL_VRAM）下会触发；当前版本加载期已自动处理，若遇到请确认用的是最新代码并重启。
- **显存/内存吃紧？** 这是 27B 文本编码器 + 视觉塔的固有体量；可关其他占卡程序，
  ComfyUI 设置中保持默认 MinVram 行为即可（勿改全局卸载策略为激进模式）。
- **和托管 API / vLLM serve 对拍？** 用 `tools/compare_with_serve.py`：先起模型包 `serve.sh`
  （vLLM，端口 8000），再跑脚本，输出每个选项的概率差。S1 应到浮点精度一致；
  S2 两个运行时的采样器逐 token 不同，脚本有意不比对 S2 正文。

## 9. 验证记录（交付前全部通过）

- `tools/verify_model_dir.py`：目录完整、`load_clip` 自动检测（qwen35_27b / int8 / MTP）、
  16 个特殊 token 原子性、图片三连 = ids [248053, 248056, 248054]、思考收尾 = 248069。
- `tools/smoke_load.py`：401 hooks / 256 选项标签 / A–P 决策头断言；S1 三型实跑正确、位级确定。
- `tools/validate.py`：LoRA 注入读数位移 0.848→0.986、卸载后位级还原（残差 0.0）；
  S2 协议种子确定性；system2_only；图片展开（36→998 prompt tokens）；chat 思考/正文拆分。
- `tools/api_run.py`（运行中的 ComfyUI，`POST /prompt` 全流程 7/7 通过）：
  noul `[0.015, 0.985]→true`、score `→5`、choice `[0.9998,…]→Paris`、S2 强制思考正确、
  adaptive（阈值 0.9 时 S1 0.856 触发 S2 折叠）、chat 通道拆分、单图 999 prompt tokens。
