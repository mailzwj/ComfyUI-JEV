# JEV-27B-VL ComfyUI 插件 — 实施计划

规格：`docs/superpowers/specs/2026-11-12-jev-comfyui-plugin-design.md`
执行方式：本会话内联执行（目标 goal 已激活），每个任务验证后再进入下一个。
Python：`D:\UserApps\Conda\envs\cu130\python.exe`（禁装包）。

## 任务清单

### T1 模型目录补全（进行中）
- `tools/merge_jev_weights.py`：合并 backbone(1686)+vision(555) → `JEV-27B-VL.safetensors`（2241 键，~26.5GB，断言无键冲突）。
- 复制 8 个 sidecar：`F:\Download\JEV-27B-VL\{tokenizer.json,tokenizer_config.json,config.json,generation_config.json,preprocessor_config.json,video_preprocessor_config.json,chat_template.jinja}` + `F:\Download\JEV-27B-VL\adapter_vllm\adapter_config.json` → 模型目录。
- `tools/verify_model_dir.py`：`comfy.sd.load_clip` 检测断言（clip_name=qwen35_27b、量化键、mtp、lm_head），tokenizer 原子性断言。
- 验证：verify 输出 `ok: ...` 全行；失败即停。

### T2 插件骨架 + 加载器
- `__init__.py`：注册 `llm` 模型目录（`models/LLM`，扩展名 .safetensors）+ `comfy_entrypoint`（返回 3 节点）。
- `jev.py`：
  - `JEVTokenizer(Qwen35ImageTokenizer)`：加载后把内层 `tokenizer` 换成 JEV 目录的 `Qwen2TokenizerFast`，断言 16 个原子 id
    （248044-248052, 248053-248057, 248068, 248069）decode→encode 往返；由 id 解码出 `IMG_TRIPLE`
    （[248053,248056,248054]）与 `END_THINK`（248069），**代码中不写 think 族字面量**。
  - 标签表（serve L112-132 逐字）：S1 `single_token_labels("x\n{}) y")` 前 16 个 id == 决策头 A-P 断言；S2 答案表 256。
  - LoRA 重映射：按后缀匹配 `cond_stage_model.state_dict()` 键（401 stem：lm_head×1 + linear_attn 3×48 + mlp 3×16…
    分布已实测：in_proj_qkv/z/out_proj 各 48 层=144，q/k/v/o_proj 各 16 层=64，gate/up/down 各 16 层=48+48+48… 共 400 + lm_head=401），
    补 `{key}.alpha`=64 → `comfy.lora.load_lora` → `BypassInjectionManager`（断言 401 hook）。
  - `load_jev_model(name)`：`load_clip([full_path])` → 换 tokenizer → 读 head/calib → 标签表 → LoRA hooks（不注入）。
- `nodes.py`：`JEVModelLoader`（io.Combo 自 `get_filename_list("llm")`）、`JEVDecide`、`JEVChat`（V3，`io.Custom("JEVMODEL"/"JEVPROB")`）。
- `tools/smoke_load.py`：进程内加载 + 全断言，输出 `ok: ...`。
- 重启 ComfyUI（先征求用户同意）；`/object_info` 确认 3 节点。

### T3 S1 引擎 + JEVDecide（文本，thinking=off）
- 引擎（`jev.py`）：`_context`（reset_clip_options/load_model/set_clip_options×2/cuda_device_context+use_quantized_matmul）、
  `_embed_tokens`（tokenize_with_weights(llama_template="{}") → process_tokens → mrope pos）、
  `_last_logits`（transformer.model.forward 2 元组 → transformer.logits[:, -1]）。
- `s1_pass` / `s1_dist` / `_groups`（serve L242-293 逐字移植，asyncio→同步循环；logprobs 查表→末位 logits 直读）。
- `tools/api_run.py`：经 8188 API 提交 JEVDecide 图（noul/score/choice 三用例），轮询 /history 打印输出。
- 验证：三 kind 概率归一、明显答案高置信、重复运行位级确定。

### T4 S2 引擎 + 自适应
- `s2_pass`：生成帧 = `apply_chat_template([{user, [text(state), image×N, text(tail)]}], tokenize=False,
  add_generation_prompt=True, enable_thinking=True[, reasoning_effort])`；
  `transformer.generate(embeds, True, budget, 1.0, 20, 0.95, 0.0, 1.0, seed, stop_tokens=[248069], position_ids=pos, mtp=...)`
  （seed = crc32(question+body)&0x7FFFFFFF）；解码 thought（skip_special=False，剥一个尾部 END_THINK，与 vLLM 一致）；
  read = gen_text + thought + END_THINK + "\n\nAnswer: (" → 末位 logits 读答案字母（noul 翻转）。
- `decide()`：完整分支逻辑（score 禁 thinking、system2_only 禁 score、auto 阈值 0.8、fold w=0.5、usage 计数）。
- 验证：seed 确定性（两次运行位级一致）、**S2 洁净性**（同一 read prompt 在 inject 与 eject 后 logits `torch.equal`）、
  思考文本以 END_THINK 干净收尾。

### T5 图像
- S1/S2 带 `F:\Download\JEV-27B-VL\agent_judge.png`：视觉展开 token 数 == qwen2vl smart resize 网格理论值；
  明显图像问题置信合理。

### T6 JEVChat
- chat template（可带 system、enable_thinking、effort）→ 生成 stop=[248044, 248046] → 最后一个 248069 处拆 thinking/text。
- 验证：思考开/关、system 生效、参数可调。

### T7 验证电池 + 交付
- `tools/validate.py`：上述全部断言一次跑完（进程内）。
- `tools/compare_with_serve.py`：用户侧 vLLM `/v1/decide` 对拍脚本（本机无 vLLM，交付给用户）。
- README.md：安装/模型目录/节点用法/对拍流程/内存说明。
- 清理 `_dev_checks`、`_reltest`；插件目录 git init + 提交；`present` 交付。

## 关键设计决定（与 spec 一致）
1. 单文件合并权重 + 原生 `qwen35_27b` 加载（量化/MTP 自动检测），不动 `comfy/`。
2. LoRA 走 `BypassInjectionManager` 手工 inject/eject（仅包裹 S1 段），S2/chat 恒无 LoRA；
   不克隆 patcher、不用 set_injections（保持共享加载、S2 洁净）。
3. `JEVTokenizer` 子类只换内层 tokenizer 对象（保留 SD1Tokenizer 包装链与图片 dict 替换逻辑）。
4. 所有 think 族 token 字面量在代码里用 `tokenizer.decode([id], skip_special_tokens=False)` 生成。
5. 与 serve 的已知差异（README 注明）：state+图片交错固定为 state 在前；vLLM 本机缺失，对拍脚本交付给用户。
