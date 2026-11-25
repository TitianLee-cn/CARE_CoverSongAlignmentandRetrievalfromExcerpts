# CARE · Cover Song Alignment and Retrieval from Excerpts

CARE 从 30 秒音乐片段检索同一作品的录音，在参考录音中定位对应时间区间，并在有可靠歌词时输出句对和翻唱类型。Covers80 用于检索和定位，JamendoLyrics 用于比较歌词时间戳。

| 目录 | 内容 |
| --- | --- |
| [preprocessing](preprocessing/README.md) | 固定切片与五折清单，提取音频向量、拍点和 chroma，生成版本对对齐缓存 |
| [b_lyrics](b_lyrics/README.md) | Kiki 转写、Demucs 人声、歌词证据、句对与翻唱判型，以及 JamendoLyrics 评分 |
| [inference](inference/README.md) | Top-10、重排、歌曲确认、E2 区间定位和 Covers80 实验 |
| [docs](docs/plan_completed.md) | 算法、数据、实验口径和结果 |

## 用现有缓存运行

在 WSL 中从仓库根目录运行，使用已有的 `RL_HW4` 环境或配有项目依赖的等效环境。预处理使用 PATH 中的 FFmpeg/FFprobe，也可用 `CARE_FFMPEG` 指定安装前缀。

| 本地路径 | 内容 |
| --- | --- |
| `data_weights/datasets/`、`data_weights/weights/` | 音频数据集、模型权重 |
| `precomputed_data/shared/` | 录音、查询与五折清单 |
| `precomputed_data/cover_embeddings/`、`precomputed_data/beat_chroma/` | 检索向量、时间窗和逐拍特征 |
| `precomputed_data/group_b/` | 歌词与人声缓存、B 组清单和校准证据 |
| `precomputed_data/reference_labels/covers80/` | 已归档的 Covers80 区间答案 |
| `precomputed_data/experiments/` | 已冻结的参数、模型与逐查询预测 |

```bash
conda activate RL_HW4
python inference/pipeline.py --help
python inference/confirmation_replay.py report
```

完整推理从 CoverHunter Top-10 开始，用音频、和声与歌词重排，选择参考录音，经 E2 计算区间，再调用句对与判型模块。缓存缺失时按两份预处理文档生成。重新编码前，按[预处理文档](preprocessing/README.md)准备第三方源码。

## 实验入口

| 实验 | 入口 |
| --- | --- |
| 纯音频基线和五折选窗 | `python inference/audio_baselines.py run` |
| CoverHunter 召回调参和接入 CARE 的对照 | `python inference/coverhunter_tuning.py`、`python inference/coverhunter_care.py --jobs 4` |
| 歌词、和声及 E2 歌词消融 | `python inference/lyrics_ablation.py --jobs 4` |
| 冻结配置复评和逐路径 E2 对照 | `python inference/frozen_reevaluation.py run --jobs 4` |
| 歌曲确认门槛调参 | `python inference/confirmation_coverage.py run --jobs 4` |
| 历史区间恢复、0% 和声门槛、无拒答和人工复核后重计分 | [确认回放与计分](inference/README.md#歌曲确认回放与人工复核后重计分) |
| Kiki 整句辅助定位 | `python inference/kiki_lyric_experiment.py run` |
| MERT、Unified 匹配区间基线 | `python inference/covers80_interval_comparison.py run` |
| JamendoLyrics 时间戳对比、B 侧有效歌词与对齐实验 | [B 组复现实验](b_lyrics/README.md) |

报告区分歌曲首名、实际报歌、区间输出，以及歌曲和区间同时命中。Covers80 有多个可接受区间时取最大 IoU，门槛为 0.50。人工无法确认的两条区间在共同比较中标为待确认，项目约定的补计另行汇报。

## Covers80 区间标注

| 文件 | 内容 |
| --- | --- |
| [当前参考区间](precomputed_data/reference_labels/covers80/reference_intervals.jsonl) | 80 条评分答案，含 20 条人工标注和 60 条自动参考 |
| [23 条复核之前的参考区间](precomputed_data/reference_labels/covers80/reference_intervals_before_23_review.jsonl) | 80 条历史评分答案，含 9 条人工标注和 71 条自动参考，用于复现原 Kiki 实验 |
| [最近一轮人工复核记录](precomputed_data/review/e2_no_abstention_23_listening/manual_review.jsonl) | 23 条试听结论，保留修订前的区间、复核结果和 2 条待确认记录 |

音频、权重、缓存、试听素材和预测文件保存在本地。各数据集和第三方权重按各自许可使用。
