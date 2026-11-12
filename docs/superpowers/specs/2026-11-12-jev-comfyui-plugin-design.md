# JEV-27B-VL ComfyUI 插件设计

日期：2026-11-12
状态：已确认（用户确认磁盘允许合并单文件；交付范围=文本+图片）

## 目标

把 JEV-27B-VL（Qwen3.8-27B-VL 基座 + 决策 LoRA + 决策头）接入 ComfyUI 0.45.0：
接收模型支持的任意输入（文本、图片），输出 JEV 模型的处理结果（System 1 三态决策 /
System 2 思考问答）。参数按模型能力设计。模型按 ComfyUI 规范重打包存放于
`models/LLM/JEV-27B-VL`。调试环境 `D:\UserApps\Conda\envs\cu130\python.exe`，禁止污染。

## 验证结论（架构依据，全部实测）

1. ComfyUI 0.45.0 原生 `qwen35_27b` 文本编码器配置与 JEV `config.json` 逐项一致
   （64 层 / hidden 5120 / inter 17408 / 24q·4kv 头 / linear 48 值头 / head_dim 256 /
   `lm_head=True` / vision 27 层·1152·4304 / MTP 1 层）。→ 复用原生加载路径，不重写架构。
2. 原生 qwen35 分支仅对单一 state-dict 生效（`comfy/sd.py` L1905 `if len(clip_data)==1`），
   前缀替换（`model.language_model.→model.`、`model.visual.→visual.`、`lm_head.→model.lm_head.`）
   只作用于 `clip_data[0]`（L2067）。→ 双文件（backbone+vision）无法直载，合并为单文件。
3. int8 convrot 量化自动检测（任意 `*.comfy_quant` 键 → `comfy.utils.detect_layer_quantization`
   → `{"mixed_ops": True}` → `comfy.ops.mixed_precision_ops`），MTP 由 `mtp.fc.weight` 存在自动启用
   （L2073）。→ 重打包产物与 ComfyUI 量化栈零改动对接。
4. 特殊 token 实测（JEV tokenizer；`added_tokens` 中 special=True 的才原子编码，decode/encode
   往返全部单 token）：
   - 248044（pad/bos）、im_start(248045)、im_end(248046)、begin(248047)、end(248048)、
     248049-248052（quad 工具族）、vision_start(248053)、vision_end(248054)、vision_pad(248055)、
     image_pad(248056)、video_pad(248057)、think(248068)、/think(248069) 均原子。
   - 旧式 `<|img|>`、`<|think|>` 等 **不是** 原子 token（BPE 切分为子词）；JEV 协议不使用它们。
   - ComfyUI 自带 `comfy/text_encoders/qwen35_tokenizer` 的 added_tokens 注册与 JEV 不同
     （im_start 等连它都不可靠）→ 插件一律用 JEV 模型目录内的 `Qwen2TokenizerFast`
     替换 `clip.tokenizer`（子类 `JEVTokenizer`，仅换 tokenizer 本体 + 原子性断言）。
   - **JEV 图片标记**（chat template 与 S1 协议共用，`serve_decide.py PROFILES["qwen"]["image"]`
     逐字节验证）= vision_start + image_pad + vision_end 三连（43 字符 → ids
     [248053, 248056, 248054]），其中 image_pad(248056) 是原生
     `tokenize_with_weights`（qwen35.py L1283）替换为视觉嵌入 dict 的位置；
     vision_start/vision_end 作为普通 token 通过。
5. LoRA `character_lora_bf16.safetensors`：802 键 / 401 模块 / r=32，键形如
   `base_model.model.model.language_model.layers.N.<mod>.lora_{A,B}.weight` 与
   `base_model.model.lm_head.lora_{A,B}.weight`（mod ∈ linear_attn.in_proj_qkv / in_proj_z /
   out_proj / mlp.{gate,up,down}_proj / self_attn.{q,k,v,o}_proj），文件内无 alpha；
   `adapter_config.json`：lora_alpha=64、r=32、use_dora=false → 每模块 scale=alpha/r=2.0。
   量化权重不能走稠密 weight-diff（会反量化约 52GB），必须走 bypass 注入
   （`comfy/weight_adapter/bypass.py` `BypassForwardHook`：`g(f(x)+h(x))`，不读原权重，
   h 用未旋转输入 x 与 LoRA ΔW 计算，数学等价 W+ΔW）。
   `hook.inject()/eject()` 无参、幂等（bypass.py L177-264）→ S1/S2 共用一份常驻权重，手工切换。
   官方用法参考 `comfy/sd.py` `load_bypass_lora_for_models`（L147-243）：
   `comfy.lora.load_lora` → `BypassInjectionManager().add_adapter(key, adapter, strength)`
   → `create_injections(cond_stage_model)`。
6. 属性链（实测）：`CLIP`（sd.py L246）→ `clip.cond_stage_model` = `Qwen35TEModel`
   （sd1_clip.SD1ClipModel，state_dict 键前缀 `qwen35_27b.`）→ 其属性 `qwen35_27b`
   = `Qwen35ClipModel`（SDClipModel，含 `.transformer` = Qwen35 模型实例、`.process_tokens`、
   `.execution_device`）→ `.transformer.model` = Qwen35Transformer。
   `CLIP.generate`（sd.py L481-490）的标准上下文：`reset_clip_options` → `load_model`
   → `set_clip_options({"layer": None})` + `{"execution_device": device}`
   → `with model_management.cuda_device_context(device), comfy.ops.use_quantized_matmul(cond, device)`。
7. 前向/生成形态（实测 qwen35.py / llama.py）：
   - 一次性前向：`x, _ = transformer.model.forward(None, embeds=embeds, attention_mask=None, position_ids=pos)`
     （无 KV 返回二元组，llama.py L1131-1134）；`BaseGenerate.logits(x)` 内部取 `x[:, -1:]`
     过 lm_head（量化安全 CastBiasWeightContext，llama.py L1199-1209）→ 得末位 [1,1,V] logits。
   - mrope 位置：`comfy.text_encoders.qwen_vl.qwen2vl_mrope_position_ids(embeds_info, len, device)`
     （由 `process_tokens` 的 embeds_info 算出，视觉 token 占多位置）。
   - 生成：`transformer.generate(embeds, do_sample, max_length, temperature, top_k, top_p,
     min_p, repetition_penalty, seed, stop_tokens=[...], position_ids=pos, mtp=True)`
     （Qwen35.generate，qwen35.py L908-919）；MTP 路径返回 **int 列表**，含触发停止的
     stop token（`del commit[stop+1:]`，L1169-1172）。
   - `process_tokens` 接受 int 与 embed dict 混合的行；image dict 经
     `Qwen35.preprocess_embed`（qwen35.py L893-902，qwen2vl smart resize + 视觉塔 fp32）
     展开为 N 个视觉向量，grid 记入 embeds_info。
8. vLLM 语义对拍（serve_decide.py 逐行核对 + vLLM v1 detokenizer 源码确认）：
   - vLLM stop-terminated 且 `include_stop_str_in_output=False`（默认）时，触发停止的 token
     **保留在 token_ids 但从解码文本中剔除** → serve 的 `thought` 不含尾部 /think(248069)。
   - S2 read prompt = 生成帧全文 + thought + /think(248069) + "\n\n" + "Answer: ("（/think 恰一次）。
   - S2 生成采样 = 模型 generation_config 默认：temperature 1.0 / top_k 20 / top_p 0.95，
     `seed = crc32((question + body).encode()) & 0x7FFFFFFF`，stop=[/think(248069)]。
   - S1 读数 = 候选词元位 logits 上 `softmax((logit + bias_i) / T_kind)`（bias 来自决策头，
     超出训练槽位的部分取 0.0）；vLLM 的 top_logprobs 分块读取与直接取全 logits 等价。
   - JEVChat 停止 = config `stop_tokens` [248044, 248046]（pad/im_end）。

## 模型目录（完成态）

`D:\Workspace\ComfyUI\models\LLM\JEV-27B-VL\`

| 文件 | 来源 | 说明 |
|---|---|---|
| `JEV-27B-VL.safetensors` | 新增：backbone+vision 合并（CPU，26.5GB） | 原生加载入口（单 state-dict，2241 键） |
| `backbone_int8_convrot.safetensors` | 已有（1686 键） | 保留 |
| `vision_projector_int8_convrot.safetensors` | 已有（555 键） | 保留 |
| `character_lora_bf16.safetensors` | 已有 | 决策 LoRA（bypass 外挂） |
| `decision_head.json` | 已有 | bias(24) / verbalizer_ids(24) / slots.ranges{noul:[0,2],score:[2,8],choice:[8,24]} / slots.verbalizers |
| `calibration.json` | 已有 | `per_kind` 温度：noul 1.0143 / choice 1.0161 / score 1.0036 |
| `convrot_manifest.json` | 已有 | 量化台账（保留） |
| `tokenizer.json` `tokenizer_config.json` | 复制自 `F:\Download\JEV-27B-VL` | JEV fast tokenizer（协议 token 全原子） |
| `config.json` `generation_config.json` | 复制 | 记录配置（采样默认 1.0/top_k20/top_p0.95；eos [248046,248044]） |
| `preprocessor_config.json` `video_preprocessor_config.json` | 复制 | 完整性 |
| `chat_template.jinja` | 复制 | 参考（运行时以 tokenizer_config 内嵌模板为准） |
| `adapter_config.json` | 复制自 `F:\Download\JEV-27B-VL\adapter_vllm` | alpha=64 来源 |

## 节点设计（V3 schema，3 个节点）

自定义类型：`JEVMODEL`（加载器输出/消费者输入）、`JEVPROB`（float 列表）。

### JEVModelLoader
- 输入：`model_name`（COMBO，`folder_paths.get_filename_list("llm")`，默认
  `JEV-27B-VL.safetensors`）。
- 输出：`model`（JEVMODEL）。
- 行为：`comfy.sd.load_clip([合并单文件])`（检测 qwen35_27b + int8 量化 + MTP 全自动）→
  替换 `clip.tokenizer` 为 JEVTokenizer（原子性断言）→ 读 decision_head.json /
  calibration.json / LoRA（构造 401 个 bypass hook，**不注入**）→ 构建 S1/S2 标签表
  （断言：S1 前 16 标签 == verbalizer_ids[8:24]）。模型保持未上 GPU（懒加载）。
- `folder_paths.add_model_folder_path("llm", models/LLM)` + 扩展名集补 `.safetensors`
  （新目录默认扩展名集为空）。

### JEVDecide
- 输入：`model`(JEVMODEL)、`kind`(COMBO noul/score/choice)、`state`(STRING 多行上下文)、
  `images`(IMAGE 可选)、`question`(STRING)、`options`(STRING 换行分隔，仅 choice)、
  `strategy`(COMBO auto/single/permute/tournament)、`thinking`(COMBO
  default/off/auto/on)、`threshold`(FLOAT，0=用 profile 0.8)、`mix`(FLOAT，0=用 profile 0.5)、
  `think_budget`(INT，0=32768)、`reasoning_effort`(COMBO default/xhigh/medium/low)、
  `seed`(INT，-1=按 serve 规则 crc32 派生)、`use_mtp`(BOOLEAN 默认 true)、
  `system2_only`(BOOLEAN)、`want_reasoning`(BOOLEAN)。
- 输出：`probabilities`(JEVPROB)、`choice`(STRING)、`info`(STRING=JSON，含 probabilities /
  choice_index / adaptation / thinking 明细 / usage / elapsed / num_model_requests)。
- 行为（对齐 serve_decide `/v1/decide`）：
  - 校验：choice 需 2-256 选项；score 禁 thinking auto/on；system2_only 禁 score。
  - S1：注入 LoRA hook → 按 strategy 跑 1-N 次裸 prompt 前向 + 末位 logits 读数 →
    必弹 hook（try/finally）。
  - S2（thinking=on，或 auto 且 max(p1)<threshold，或 system2_only）：基座（无 LoRA）
    思考生成（stop=/think(248069)）→ 平铺 read prompt 末位 logits 读数答案字母分布 →
    与 p1 按 mix 折叠。
  - noul 的 S2 显示选项为 `Yes (true)`/`No (false)`，读数结果翻转回 [P(false), P(true)]。

### JEVChat
- 输入：`model`(JEVMODEL)、`system`(STRING 可选)、`prompt`(STRING)、`images`(IMAGE 可选)、
  `enable_thinking`(BOOLEAN 默认 true)、`reasoning_effort`(COMBO)、`max_tokens`(INT 默认 2048)、
  `temperature`(FLOAT 1.0)、`top_k`(INT 20)、`top_p`(FLOAT 0.95)、`seed`(INT -1=crc32)、
  `use_mtp`(BOOLEAN)。
- 输出：`thinking`(STRING)、`text`(STRING)。
- 行为：官方 chat template（含 system / 思考开关 / effort 指令块）渲染 → 同 S2 生成路径，
  stop=[248044, 248046]；输出在最后一个 /think(248069) 处拆 thinking/text。

## S1 协议（jev27-bare-v1，逐字对齐 serve_decide）

裸 prompt（无 chat 包装，前缀为空）：

```
[kind] {kind}
[state] {state}{图片三连 × N}
[question] {question}
[options]
{选项行}
[decision]:
```

- 选项行：choice → `{A}) {o}`、`{B}) {o}`…（标签表 A-Z,AA-ZZ，共 256）；
  noul → 两行 `false` / `true`；score → 六行 `0`..`5`。
- 图片三连 = vision_start + image_pad + vision_end（由 tokenizer 按 id 解码得到，
  代码中不写字面量）；分词后 image_pad 位被替换为视觉嵌入 dict（每图一次）。
- 读数：`softmax((logit_t + bias_i) / T_kind)`，choice 用 S1 标签表前 n 个 id +
  决策头 bias（槽外补 0），noul/score 用 verbalizer_ids[lo:hi] + bias[lo:hi]。
- strategy：`single`（默认，≤16 选项的 choice 与 noul/score 恒为单 pass）；
  `permute`（1+3 个乱序，crc32(question) 种子，取平均）；`tournament`
  （≤16/组逐组 → 每组冠军+补位进 16 强决赛 → 组内概率×决赛份额归一）。
  >16 选项的 choice 按 strategy 选路（serve 原样移植）。

## S2 协议（基座思考，无 LoRA）

- 生成帧 = 官方 chat template 渲染单条 user 消息（内容 = state 文本 + 图片 × N + tail），
  `add_generation_prompt=True, enable_thinking=True[, reasoning_effort=...]`。effort 指令块
  由模板内嵌（xhigh/medium 注入指令，low 缩短指令，medium 无指令块）。
- tail（逐字）：
  `\n\nQuestion: {question}\n\nOptions:\n{(A) o\n(B) o...}\n\nThink it through carefully, then give your final answer on the last line in the form: Answer: (X)`
  （noul 显示选项 `Yes (true)` / `No (false)`）
- 生成：temperature 1.0 / top_k 20 / top_p 0.95 / min_p 0 / rep 1.0，
  `seed = crc32((question + body).encode()) & 0x7FFFFFFF`（body=选项行文本），
  stop=[/think(248069)]，max_tokens=think_budget。
- read prompt = 生成帧全文 + thought（解码 skip_special=False，尾部 /think 已剥离，
  与 vLLM 输出一致）+ /think(248069) + `\n\nAnswer: (`；末位 logits 在答案标签
  （"Answer: (" 后单 token 标签表，256 个）上 softmax；noul 翻转。
- 折叠：`p = (1-mix)·p1 + mix·p2`。

## JEVChat

官方 chat template（可带 system 消息）渲染 → 生成 stop=[248044, 248046] →
输出 id 流在最后一个 think(248068)…/think(248069) 段拆 thinking，其后为 text；
无思考段时 thinking 为空。采样参数用户可调。

## LoRA / 决策头加载

- LoRA 键按后缀匹配重映射到 `cond_stage_model.state_dict()` 键（`layers.N.<mod>` /
  `lm_head` 后缀定位，前缀不硬编码），补齐 `{key}.alpha`=64，
  `comfy.lora.load_lora` 得到 401 个 `LoRAAdapter` → `BypassInjectionManager`
  → `create_injections(cond_stage_model)`（断言 hook 数 == 401）。
- S1 前 `inject()` 全部 hook，结束后 `eject()`（try/finally 保证）；S2/chat 始终无 LoRA。
- 决策头 bias(24) 与 calibration `per_kind` 温度直接加载使用。

## 加载与内存

- 合并单文件 26.5GB（int8 convrot 骨干 + int8 视觉）；MinVram/fast_disk 由 ComfyUI 托管，
  32GB 显存下常驻可行（权重 ~27GB + 16K 上下文 KV ~2GB + 激活）。
- 所有推理在 `cuda_device_context + use_quantized_matmul` 上下文内完成（同
  `CLIP.generate`）。模型加载/卸载交给 ComfyUI patcher 生命周期；hook 权重随
  `inject()` 移到计算设备（bypass 内建）。

## 验证计划

1. 加载冒烟：load_clip 检测断言（qwen35_27b / 量化 / mtp）、tokenizer 原子性断言、
   标签表断言（S1 前 16 == 决策头 A-P、S2 答案表 256）、401 hook 断言。
2. S1 功能：noul/score/choice 文本用例（明显答案高置信、概率归一、重复运行位级确定）。
3. S2：off/auto/on 行为、seed 确定性（两次运行 thought+概率一致）、
   **基座洁净性**（同一 read prompt 在 inject 前与 eject 后 logits 位级一致，
   证明 S2 无 LoRA 泄漏）、思考文本在 /think(248069) 处干净收尾。
4. 图像：S1/S2 带 1-2 张图，视觉展开 token 数 == smart_resize 网格理论值，
   明显图像问题置信合理。
5. JEVChat：思考开/关、system、采样参数、拆分。
6. 用户侧对拍（可选）：`tools/compare_with_serve.py` 对比本插件 info JSON 与
   vLLM `serve.sh` `/v1/decide` 响应（本机无 vLLM，cu130 禁装；README 给出流程）。

## 风险与已知差异

- 本机 vLLM 缺失 → 无法本机对拍 vLLM 端点；以功能/结构性验证 + 用户侧脚本替代。
- GPU 其余占用（采样时仅 ~3.8GB 空闲）→ 首次加载需观察 OOM；MinVram 兜底，README 注明。
- 与 serve 的已知可接受差异：reasoning 展示文本按 skip_special 处理略异；
  state 与图片的交错顺序固定为「state 文本在前，图片按序在后」（serve 允许任意交错，
  节点输入形态决定）；LoRA 走 bypass hook 手工切换而非 patcher 注入生命周期
  （数学等价，官方模块原生支持）。
- ComfyUI 服务器需重启一次以注册新节点（执行时告知用户）。
