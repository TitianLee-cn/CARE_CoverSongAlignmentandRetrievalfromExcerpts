# CARE 组员 A 预处理

在 WSL 中使用已装好依赖的 `RL_HW4`。脚本读取 CARE 本地的 `data_weights` 和 `vendor`，默认写入 `CARE/precomputed_data`。各阶段支持用 `--output` 指定临时目录，特征脚本也可以用 `--manifest-root` 指向已有清单。输出的数据路径和索引可相对解包位置读取。

## 共用清单与小样本验收

```bash
conda activate RL_HW4

python preprocessing/prepare_manifests.py

for model in coverhunter mert unified; do
    python preprocessing/extract_cover_embeddings.py --model "$model" --limit 2 || break
done
python preprocessing/extract_beat_chroma.py --limit 2
python preprocessing/extract_jamendo_mert_frames.py --limit 2
python preprocessing/align_cover_pairs.py --limit 2
python preprocessing/check_caches.py --limit 2 --include-alignments
```

`prepare_manifests.py` 完整解码官方清单的 160 首 Covers80 和 79 首 JamendoLyrics 音频，记录原文件 SHA-256、实际样本数和时长。查询以 Python `random.Random(42)` 在原始采样点上均匀选起点，每条固定为 30 秒。划分使用独立的种子 42 随机序列。每轮外层有 16 部测试和 64 部开发作品，每轮内层有 48 部拟合、16 部验证，测试和验证曲库分别是 80 部和对应的 64 部作品。

将 `precomputed_data/shared/` 发给 B。B 把它放在相同相对位置即可读取两份清单与折划分。清单中的音频路径相对于 CARE 根目录。B 如需对照自己的原录音，可在自己的 CARE 目录执行 `python preprocessing/prepare_manifests.py --verify`。它核对本地音频与共享清单，并保留收到的文件。小样本验收跑的是 A 负责的模型，B 可用同一批作品 ID 对接自己的输出。

## 人工标注与合并

从 `precomputed_data/shared/annotation_template.jsonl` 分别复制本人负责的行到 `annotations_a.jsonl` 和 `annotations_b.jsonl`，两份文件都可放在 `precomputed_data/shared/`。分工表约定前 40 对由 A 主标，后 40 对由 B 主标，复核时更新对应作品的原行。

- `query_has_effective_lyrics` 填 `true` 或 `false`，并在 `query_lyrics_reason` 写判断理由。无法判断时留空，在 `notes` 说明。
- `query_language`、`reference_language` 记录两侧主要演唱语种。`cover_type` 记录原词翻唱、同语种重新填词或不同语种翻唱，证据不足或混语时在 `notes` 说明。
- `reference_intervals` 可填多处区间，每个区间包含 `start_sec`、`end_sec`。用 `reference_interval_status` 区分 `pending`、`confirmed`、`uncertain`、`no_match`。
- `line_pairs` 的每项包含 `query_lines` 和 `reference_lines`。每侧是一条或多条含 `text`、`start_sec`、`end_sec` 的歌词行。用 `line_pair_status` 记录处理状态，可为可接受的重复副歌位置分别留项。
- `annotation_status` 可填 `pending`、`partial`、`done`；`review_status` 可填 `pending`、`confirmed`、`uncertain`。查询时间从切片起点计，参考时间从整曲起点计。

```bash
python preprocessing/merge_annotations.py \
  precomputed_data/shared/annotations_a.jsonl \
  precomputed_data/shared/annotations_b.jsonl
```

合并脚本检查固定 ID、重复作品和时间边界，并输出 `precomputed_data/shared/annotations.jsonl`。两份输入不必同时具备逐行标注，待标项会保留。评分时使用相应阶段完成的人工答案。

## A 的全量特征

三个编码器按相同长度的窗口做批量推理。CoverHunter 在 GPU 上用官方 CQT 滤波核计算特征，默认每批 8 窗。MERT 默认 4 窗，Unified 默认 8 窗。可用 `--batch-size N` 调整，新的缓存元数据中记录 `inference_batch_size`。原有逐窗缓存可继续使用。

去掉小样本命令中的 `--limit 2` 即可处理剩余作品。已有且参数相符的缓存会复用。可重复添加 `--work-id` 筛选作品，或加 `--force` 重算筛选项。每个输出子目录包含 `index.jsonl`、本次运行的 `failures.jsonl` 和 `items/*.npz`。索引内的 `path` 相对该索引目录，每个 NPZ 的 `metadata` 是 JSON 字符串。

| 输出目录 | NPZ 主要字段 |
| --- | --- |
| `cover_embeddings/<model>/` | `embeddings`、`bounds_samples`、`bounds_sec`。第 i 个配置的向量行号为 `config_window_rows[config_offsets[i]:config_offsets[i+1]]`。9 组窗口的顺序写在元数据 `configs`。 |
| `jamendo_mert_frames/` | `frames`、`frame_sec`，后者按卷积感受野中心映射至整曲时间。每 15 秒左右分块，两侧各保留最多 2.5 秒上下文，输出每个帧位置一次。 |
| `beat_chroma/` | `beat_sec`、`beat_bounds_sec`、`chroma`、`valid`、`four_beat_mode`、`unique_modes`。四拍模式记在末拍，无效模式为四个 `-1`。分折 IDF 在评测中从本轮参考记录的去重模式统计。 |
| `cover_pair_alignment/` | `beat_indices`、`match_bounds_sec`、`cosine`、`match_valid`、`path_offsets`。每次匹配四个时间值依次为查询拍开始、结束和参考拍开始、结束。各路径的移调量和代价在元数据内。默认保留最多三处位置，参考区间 IoU 达 0.5 时去重。 |

CoverHunter 和 Unified 各用单独的 Python 进程调用。默认对官方正确 A/B 对做 DTW。正式召回后，可传入 `--pairs /path/to/pairs.jsonl`，每行给 `query_recording_id` 和 `reference_recording_id`，结果写到 `candidate_alignment/`。查询侧均用片内时间，原录音偏移保存在元数据。参考侧均用整曲时间。

重新编码前，从 CARE 根目录检出对应的上游源码版本。Unified 的上游目录含 Windows 保留文件名，需使用 Linux 文件系统检出。

```bash
git clone https://github.com/Liu-Feng-deeplearning/CoverHunter.git vendor/CoverHunter
git -C vendor/CoverHunter checkout --detach 8ab82d040f779195ff652afbc624aa3fce2ea6b0
git clone https://github.com/raraz15/unified-track-and-version-id.git vendor/unified-track-and-version-id
git -C vendor/unified-track-and-version-id checkout --detach 3081d416d696de567ad172ba1d7e69e4c440e1b3
```
