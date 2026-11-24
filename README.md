# CARE · B 组歌词分支

B 组实现包含 Covers80 歌词特征、候选匹配、句对定位与翻唱判型，以及 JamendoLyrics 时间戳评测。音频、模型权重、A 组交接包和人工标签由运行者放在本地。

## 代码结构

| 目录或模块 | 作用 |
|---|---|
| `group_b/import_shared.py`、`cache.py`、`media.py` | 校验 A 组共享清单和录音，读写缓存，以采样点裁剪查询音频 |
| `group_b/features.py` | Echora Kiki 转写、Demucs 人声、Qwen 句向量和语种证据，以及 Jamendo 的 Qwen3-ASR＋ForcedAligner |
| `group_b/lyrics.py`、`pairs.py` | 歌词清洗、shingles、有效歌词门槛、词面与语义打分、有序匹配、句对展示和判型 |
| `group_b/position_fusion.py`、`joint_tune40.py` | 在同一参考位置上融合音频和歌词，导出整段候选，按官方内外折复现实验 |
| `group_b/covers_evidence.py`、`cover_alignment.py` | 输出默认设置下的折内拟合证据，接入 A 的正确版本对路径，比较语义、音乐和融合三种展示结果 |
| `group_b/annotations.py`、`gates.py`、`evaluate.py` | 整理 A/B 人工标签，并进行门槛验证及评分 |
| `group_b/jamendo.py`、`pool_handoff.py` | 对官方逐行标注评分，并按预测行汇聚 A 的 MERT 帧 |
| `group_b/handoff.py`、`review_pack.py` | 导出缓存索引、B 侧空白模板和 20 组试听包 |
| `scripts/`、`requirements/`、`tests/` | 批处理和试跑入口、依赖版本、回归测试 |

## 环境与输入

在仓库根目录运行。推理环境用 Python 3.12、可用的 CUDA 12.8 和 PATH 中的 FFmpeg、FFprobe。两套推理环境分别安装 Kiki 和 Qwen-ASR 依赖。

```bash
uv venv --python 3.12 .venv-kiki
uv venv --python 3.12 .venv-b
uv pip install --python .venv-kiki/bin/python --torch-backend cu128 -r requirements/kiki.txt
uv pip install --python .venv-b/bin/python --torch-backend cu128 -r requirements/asr.txt
export CARE_ROOT=/path/to/data_weights
export CARE_OUTPUT="$PWD/outputs/group_b"
export CARE_MANIFESTS="$CARE_OUTPUT/manifests"
B=.venv-b/bin/python
"$B" -m group_b.import_shared /path/to/CARE_shared_for_B.zip --out "$CARE_MANIFESTS"
```

`CARE_ROOT` 下使用 `datasets/` 和 `weights/`。导入会核对正式查询的采样点、各折、录音 SHA-256 和 Jamendo 官方逐行标注。需要已有运行环境时，设置 `CARE_ENV_DIR` 为包含 `.venv-kiki` 和 `.venv-b` 的目录。换用已生成的缓存和清单时，设置 `CARE_OUTPUT` 和 `CARE_MANIFESTS` 为对应路径。读取缓存会核对录音 ID、校验值、文件路径及源时间范围。

## 提取、拟合与试听

```bash
bash scripts/run_smoke.sh
bash scripts/run_features.sh covers80
"$B" -m group_b.covers_evidence
"$B" -m group_b.cover_alignment /path/to/CARE_cover_pair_alignment_for_B.zip
"$B" -m group_b.review_pack "$CARE_OUTPUT/Covers80_review_20"
"$B" -m group_b.handoff annotations --out "$CARE_OUTPUT/annotations_b_blank.jsonl"
"$B" -m group_b.handoff shingles --out "$CARE_OUTPUT/lyrics_shingles.jsonl"
```

对单条录音补跑可用 `python -m group_b.features STAGE DATASET --start-index N --limit 1`，并按阶段选择对应环境。`run_features.sh` 最后更新 `cache_index.json`。试听包每折抽取 4 个有融合输出的 B 侧查询，提供原始 30 秒输入和系统选中的参考歌词片段。试听判断用于复核输出；人工真值请用空白模板独立填写。

## A 组调用方式

```python
from group_b.pairs import Corpus, fold_sets

fit_ids, held_out_ids = fold_sets(outer=1, inner=None)
corpus = Corpus(config={})
background = corpus.calibrate(fit_ids)
score = corpus.score_candidate("covers80_001", "covers80_002", background)
```

`Corpus.score_candidate` 按官方作品 ID 索引查询和候选，返回原始分数、匹配行及双侧时间，并在歌词分支缺失时返回原因。查询时间从 30 秒片段起点算，参考时间从原录音起点算。`Corpus.calibrate` 只接收当前折的拟合作品。候选列表由 A 的检索流水线提供。

### 同位置整段融合

曲库检索确认参考录音后，可以用 `build_position_candidates` 根据音频、歌词缓存和 A 的路径生成整段候选，再交给原来的 `predict`。位置打分考察各候选前后两秒内的音频和歌词证据，并在歌词证据较弱时减小其权重。输出为参考录音上的 30 秒区间及选中的候选索引。

```python
from group_b.joint_tune40 import build_position_candidates
from group_b.pairs import predict

# corpus、background 使用上一步已建立的语料与本折拟合参数。
# paths 含 A 组各条路径的 reference_span_sec，blocks 仍使用句对接口的格式。
request = {
    "work_id": query_work_id, "reference_work_id": confirmed_reference_work_id,
    "confirmed": True, "context": "end_to_end", "blocks": reliable_music_blocks,
}
request["position_candidates"] = build_position_candidates(
    corpus.queries[query_work_id], corpus.refs[confirmed_reference_work_id], request, paths
)
request["position_config"] = fold_position_config
result = predict(corpus, request, background, mode="fusion")
```

`fold_position_config` 含 `weights` 和可选的 `start_bias_sec`。权重名为 `lyrics_mix`、`audio_temperature`、`agreement_bonus`、`far_penalty`。也可把候选直接交给 `group_b.position_fusion.select_reference_interval`。候选字段为 `start_sec`、`audio`、`a_path`、`anchors`、`timed_words`、`window_words` 和 `far_penalty`。路径和歌词分量取 0 到 1，缺失证据填 0。远距离低置信候选的惩罚分量填 −1，其余填 0。已有的 `reference_interval` 可以继续从 A 侧传入。歌词门槛未通过时仍可用音频证据选区间。

复现 B 侧后 40 对的五折实验需先生成正确版本对路径索引，并提供候选标注、上一轮实验结果和调参前的候选清单。

```bash
"$B" -m group_b.joint_tune40 /path/to/dataset_candidates.jsonl \
  /path/to/previous_validation.json "$CARE_OUTPUT/joint_review" \
  --prior /path/to/earlier_candidate_manifest.json
```

复现时每折只在开发集的内折选权重，在外折留出作品上报告一致性。原实验还使用了可用的独立转写缓存与音高候选缓存。缺省时使用本地 Kiki 歌词和独立音频搜索，强度可能不同。候选标注在拟合实验中作为指定参考，人工正式评分仍按下一节导入。

## 正式人工评测与 Jamendo

```bash
"$B" -m group_b.annotations /path/to/annotations_a.jsonl /path/to/annotations_b.jsonl \
  --out "$CARE_OUTPUT/human_labels.jsonl"
"$B" -m group_b.gates --outer 1 --labels "$CARE_OUTPUT/human_labels.jsonl" \
  --out "$CARE_OUTPUT/gate-fold1.json"
"$B" -m group_b.evaluate --predictions "$CARE_OUTPUT/cover_pair_alignment/fusion.json" \
  --labels "$CARE_OUTPUT/human_labels.jsonl" --out "$CARE_OUTPUT/covers80-fusion-metrics.json"
bash scripts/run_features.sh jamendolyrics
"$B" -m group_b.jamendo score
"$B" -m group_b.pool_handoff /path/to/four-mert-archives
```

自动标注草稿会被人工标签导入拒绝。2、3、5 秒门槛需要各外折开发作品的人工标签。现有正确版本对预测使用固定初值，若开发集选出的门槛发生变化，用 `python -m group_b.pairs --outer FOLD --requests FOLD.jsonl --config GATE.json --mode MODE --out PREDICTIONS.json` 逐折重跑，再合并评分。请求每行包含 `work_id`、`reference_work_id`、`confirmed` 和可靠路径 `blocks`，可附 `reference_interval`。端到端测试取 A 提供的实际首名、确认状态和定位结果。

Jamendo 使用同一份官方标注报告行召回率、起止时间 MAE 和两端误差均在 0.5 秒内的比例；MERT 帧汇聚使用各模型自己的预测区间。

## 测试

轻量测试只依赖 `requirements/core.txt`，可直接运行 `python -m unittest discover -s tests`。推理试跑需另行提供两套环境和数据。
