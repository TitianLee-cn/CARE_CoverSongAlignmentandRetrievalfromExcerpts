# Covers80 缓存推理

从 CARE 根目录运行。已有的 `RL_HW4` 环境可用。这个入口在 CPU 上读取已有向量、逐拍特征、转写和人声缓存。fastText 包和 `data_weights/weights/fasttext/lid.176.bin` 用于识别查询里通过人声门槛的歌词语种。

```bash
conda activate RL_HW4
export CARE_ROOT="$PWD/data_weights"
export CARE_OUTPUT="$PWD/precomputed_data/group_b"
export CARE_MANIFESTS="$CARE_OUTPUT/manifests"

# 用第 3 折的开发作品拟合参数，并预测该折全部 16 条留出查询
python inference/pipeline.py run --fold 3 --window 15 --overlap 50
FOLD=precomputed_data/inference/fold3_w15_o50
python inference/verify.py "$FOLD/e2_path/predictions.jsonl" --score --compare "$FOLD/predictions.jsonl"

# 复用同一套参数单独预测两条，写入独立文件
python inference/pipeline.py predict --fold 3 --window 15 --overlap 50 \
  --query-id covers80_059_q --query-id covers80_022_q \
  --out precomputed_data/inference/fold3_w15_o50/e2_path/smoke.jsonl
python inference/verify.py precomputed_data/inference/fold3_w15_o50/e2_path/smoke.jsonl
```

`--fold` 取 1–5，`--window` 取 10、15、20 秒，`--overlap` 取 25、50、75（百分比）。`fit` 只拟合参数，`predict` 读取同折、同配置的参数，`run` 依次执行两步。可重复传入本折的 `--query-id`。新版定位（默认 `--e2 path`）写到 `precomputed_data/inference/fold{折号}_w{窗长}_o{重叠率}/e2_path/`。使用 `--e2 legacy` 可复现原版，使用父目录下的旧参数及默认输出路径。参数是 `parameters.json`，结果是 `predictions.jsonl`。运行查询子集时可用 `--out` 保留原有的整折结果。

## 数据与推理

- 启动时比对 A、B 清单的作品、录音、音频 SHA、切片位置和实验折，并核对音频、和声分片的元数据。查询侧的秒数均从 30 秒切片起点算，`query_source_start_sec` 单独记录该片在原录音的偏移。参考侧的秒数从整曲起点算。
- 测试查询在全部 80 条参考录音中按 CoverHunter 窗口最高余弦召回不同录音的前十名。对这十条分别计算可移调的子序列 DTW、和声稀有度分和 B 的候选歌词分。歌词门槛未通过、查询文本或人声分片缺失时，仍用音频和和声完成排序及歌曲确认。
- 每折的 64 部开发作品再分四轮，每轮用 48 部拟合分支单调校准，用余下 16 部验证。库外验证在召回前移除正确参考，用于选择首名与次名分差、首名绝对分门槛。最终在 64 部开发作品上重拟合分支校准，并使用 B 为同一组作品保存的歌词背景分位数。融合沿用方案中的固定权重，可靠拍最低余弦沿用 B 的 0.60，连续覆盖门槛为 0.50。
- 整段位置由首名的音频命中窗、可靠拍锚点或完整歌词边界提出。开发折的合并参考区间仅用于验证这些提案的 IoU 并估计各来源的置信度。预测入口不读取标注文件。多个可信位置冲突、或没有足够可信的位置时保留歌曲决定，区间记为待确认。现有合并区间标记为自动参考，评分时应保留这一限定。
- 将实际首名、歌曲确认状态、预测区间和选中路径的连续可靠块传入 B 的端到端句对与判型入口。B 可以只展示句对，类型仍为待确认。同语种重新填词还要求转写质量得到确认，本入口不会自行填入这个标志。

## 读取结果

JSONL 每行一条查询。`top10` 按重排顺序保存，每个候选含录音 ID、原始 `recall_rank`、最终 `rank`、两侧命中窗、原始与校准分数、DTW 路径和歌词匹配行。`confirmed` 是歌曲决定，`reference_interval` 和 `interval_status` 是整段定位结果，`interval_proposals` 保留各提案及置信度。`music_path_index` 指向首名的 `paths`。`prediction.display` 是 B 筛选后可展示的句对，`prediction.type_status` 区分已判型、待确认和跳过。

`verify.py` 核对每条查询的 ID、十个候选的唯一性、各侧时间边界及歌词关闭后的返回结果。加 `--score` 时才读取合并参考区间，均值 IoU 的分母只包含首名正确且成功输出区间的查询，应和定位数一起看。

## 第 3 折整段定位的配对验证

`fit --fold 3 --e2 path` 在四次内层划分中分别用 48 部拟合作品的召回正例训练两版定位置信度模型，在另外 16 部上记录预测。每版共 64 条折外预测。比较共用每条查询的 Top-10、重排首名、歌曲确认和定位提案。定位指标以歌曲正确确认且有评分参考的查询为分母。内层定位模型只见本轮拟合作品的区间。阈值用汇总的开发折外结果选择，因此开发指标含选阈值的收益；外层 16 条在全开发集重拟合和阈值冻结后评测。

- **原版**按来源拟合逻辑回归，和声输入使用首条路径的录音级分数。固定置信度门槛 0.60、冲突分差 0.15。外层对照直接读取原有预测。
- **新版**为各条和声路径记录 `path_index`、`match_quality`（1 减该路径 DTW 总代价除以查询拍数）、`longest_reliable_sec`（查询轴上最长连续可靠段秒数），以及查询和参考两侧的左右外推秒数。参考侧记录裁到录音边界后的实际外推。上下文特征包括其他来源对该位置的最大 IoU、按提案速度映射到邻近参考窗口的 CoverHunter 相似度及其与全曲最优值的差，以及时间一致的匹配歌词行比例。特征顺序在参数文件的 `feature_names` 中。按来源标准化后拟合逻辑回归，正例为提案与任一自动参考区间 IoU ≥ 0.5。
- **选择规则**沿用跨来源一致优先和冲突分差检查。在开发集比较最低置信度 0.30、0.40、0.50、0.60、0.70 与分差 0、0.10、0.15 的组合；最大化查询级 F1（2 × 最终命中数 ÷〔正确确认数 + 输出数〕），并列优先更多命中、更严格门槛。第 3 折冻结值为 0.30 和 0.15。
- **结果文件**均在新版目录。`development_oof.jsonl` 保存每个内层留出样本的两版置信度和提案。`development_metrics.json` 保存阈值对照与分组计数。`verify.py --compare` 核对两份外层结果的查询、候选顺序与歌曲决定，再生成 `outer_metrics.json`。两个阶段的 `*_comparison.jsonl` 按查询列出旧版、新版的输出、最大可达 IoU 和合格提案最高置信度。
- **分组与归因**。`no_lyrics` 表示查询的有效歌词门槛关闭。`any_conflict` 表示至少两个候选位置 IoU < 0.5；`conflicting` 专列不同来源的首选位置发生此类冲突，两组可重叠。另报旧模型套用新门槛及新模型套用旧门槛的计数。对于失败，先检查提案中是否有合格位置，再检查这些位置是否至少有一个过置信度门槛，最后检查选区间规则。归因以自动评分参考为准。

## 五折纯音频检索

```bash
cd /mnt/e/AI4Titi/0.0.2_DataMining/FinalProject/CARE
conda activate RL_HW4
python inference/audio_baselines.py run
```

`run` 在 CPU 上校验三个编码器的 480 个缓存，对开发查询做双精度小样本对照，再完成选窗和外层测试。`smoke` 只执行校验和小样本对照。`test` 读取已选定的配置重跑外层测试。`--output DIR` 指定实验目录，默认写到 `precomputed_data/experiments/audio_fivefold/`。

每折、每种配置、每个模型都使用四批各 16 条的内层留出查询，检索本折完整 64 首开发参考。三模型各有 64 次检索，所以最大化三者平均 Hit@10 等价于最大化总命中数（分母 192）。命中数并列时选择三个模型检索总耗时较短者。五折配置全部写入 `selected_windows.json` 后再读取它，用每折 16 条留出查询检索全部 80 首参考。相同录音内并列时按缓存顺序取最早查询窗、再取最早参考窗；不同录音同分时按录音 ID 排序。

`validation/fold*.jsonl` 保存所有配置的开发查询排名。`coverhunter.jsonl`、`mert.jsonl`、`unified.jsonl` 分别保存 80 条外层预测，每条含 Top-10 的余弦分、查询切片内窗口、参考整曲内窗口及耗时。`summary.csv` 和 `summary.json` 汇总 Hit@1、Hit@5、Hit@10、MRR@10 和检索时间；未进前十的查询对 MRR@10 贡献为零。计时使用 CPU 单 BLAS 线程，从内存中的向量开始，包括窗口点积、每首录音取最大值、排序和构造 Top-10。每条查询独立检索三次取墙钟耗时中位数，再对查询求和。缓存读取、校验、指标计算和写盘在计时之外。不同机器或负载的耗时可能影响同分配置的重新选择。

完整 CARE 入口接受 `--window-config precomputed_data/experiments/audio_fivefold/selected_windows.json`，按 `--fold` 读取窗长和重叠率，并核对共享折划分的 SHA-256。

## CoverHunter 召回探索性调参

```bash
conda activate RL_HW4
python inference/coverhunter_tuning.py
python inference/coverhunter_care.py --jobs 4
```

从 CARE 根目录运行。产物集中在 `precomputed_data/experiments/coverhunter_tuning/`。A.jsonl 逐条复用上次五折实验的原始排名和耗时，B.jsonl 使用 CoverHunter 单独选择的滑窗，C.jsonl 联合选择滑窗和 λ。B 和 C 共用相同的查询、参考录音与内层划分。每条录音得分为 (1 − λ) × 最高窗口对余弦 + λ × 各查询窗最佳余弦的均值；无论 λ 取何值，候选保存的两侧窗口都对应全矩阵最高余弦。λ > 0 时 `top10[].score` 为录音聚合分。查询时间从切片起点算，参考时间从整曲起点算。

B_config.json 与 C_config.json 分别保存五折冻结参数及该折所有备选配置的留出结果。`validation` 中每项依次为查询 ID、内层折号、正确参考在前十的排名（未进入时为空）、检索纳秒数。每折先最大化 64 条开发查询的 Hit@10，再最大化 MRR@10（用最小公倍数 2520 精确比较），再优先较小 λ、较短总检索耗时。计时口径与原实验相同，每种配置单独检索，单 BLAS 线程，每条查询取三次墙钟时间中位数，包含打分和排序、排除缓存 I/O。全部五折完成选择后才按新配置计算外层排名。summary.json 保存三组合计、分折指标和查询进出前十的名单。

第二条命令读取已冻结的结果。C 合计 Hit@10 严格高于 B 时，为两组分别运行五次完整 CARE。在每折的开发作品上重新拟合校准与确认参数，共用 path 版 E2 和其余算法设置。`full_care/` 下按组与折保存参数、内层验证和外层完整预测，汇总放在 `full_care/summary.json`。每条完整预测的召回顺序、音频分和两侧命中窗都会与对应的纯音频结果交叉核验。两条命令均支持 `--verify-only` 回查已落盘的结果。

## 歌词重排与音频／和声权重探索性实验

```bash
conda activate RL_HW4
python inference/lyrics_ablation.py --smoke
python inference/lyrics_ablation.py --jobs 4
python inference/lyrics_ablation.py --verify-only
```

四组统一读取 CoverHunter 冻结的 B 组召回配置。每折将每个开发查询的库内和库外候选各生成一次。库外先从曲库移除正确作品，再召回前十。校准器在内层 48 部拟合作品上估计，64 条内层留出预测用于选择确认规则与 C 的和声权重。确认规则沿用按证据分组的库外误报上限 5%。权重优先看正确确认数，再看 Hit@1、MRR@10，完全并列时选较小权重。权重为零时歌曲确认使用纯音频规则。外层按每折冻结参数分别检索完整 80 首参考录音及移除正确作品后的 79 首。

A 使用原重排，B 关闭歌词加分，C 继续调和声权重。A、B、C 共享 E2 模型和基于 A 的内层结果选出的 E2 运行门槛。D 复用 C 的排序和确认规则，重新拟合 E2 并选门槛。D 的定位提案和置信度特征来自音乐，歌曲确认后调用 B 的判型逻辑。同语种句对保留在判型内部，严格版预测只返回分类状态及证据汇总。

在完整推理入口可独立使用 `--no-lyric-rerank` 和 `--no-e2-lyrics`，也可用 `--harmony-weight` 指定固定权重。用实验冻结参数预测时，传入 `--parameters` 和 CoverHunter B 的 `--window-config`，输出路径可用 `--out` 指定。

默认实验目录为 `precomputed_data/experiments/lyrics_ablation/`。每折的 `weight_sweep.json` 记录内层选择依据，各组各有冻结参数、库内预测和库外预测。`summary.csv` 与 `summary.json` 汇总五折和分折指标，`changes.jsonl` 保存相邻组间发生变化的查询及前后状态。确认覆盖率以全部库内查询为分母，库外误报率以全部移除正确作品后的查询为分母。定位命中要求首名正确且与任一自动参考区间 IoU ≥ 0.5。报告同时给出全部查询、具备参考的正确确认查询、输出区间数和正确录音上输出区间数，便于查阅不同分母。

## 冻结配置复评

```bash
conda activate RL_HW4
python inference/frozen_reevaluation.py run --jobs 4
python inference/frozen_reevaluation.py report
```

复评沿用 `coverhunter_tuning/B_config.json` 中每折的 CoverHunter 滑窗和 λ=0。完整 CARE 使用 0.7 音频、0.3 和声、0.15 歌词加分，E2 使用逐路径置信度并保留歌词证据。已用于探索性分析的外层折继续沿用共享划分。默认输出目录为 `precomputed_data/experiments/frozen_reevaluation/`，可用 `--output DIR` 指定其他目录。也可分开运行 `audio`、`fit`、`predict` 和 `report`。

- `audio_summary.csv` 记录三条正式基线与三条同窗基线。正式基线复用已逐查询重算并核对的排名与原始计时，同窗基线重新计时。每条查询使用内存向量和单 BLAS 线程搜索三次，取耗时中位数。报告平均值及 80 条中位数累计时间。
- `care_summary.csv` 记录完整 CARE 排名、库内确认和库外误报。库外先移除该查询对应作品，再对其余 79 首召回。完整 CARE 串行计时，起点为内存向量召回，终点为 E2 与歌词判型完成。计时包括查询过程中发生的和声与歌词分片读取和 DTW。嵌入缓存载入及启动预热单独记录，拟合、库外对照、写盘和评分均在计时之外。五折各有 16 条库内和 16 条库外预测。
- `e2_summary.csv` 记录各折和总体的 E2 比较。旧版使用录音级和声分和原来的置信度特征，新版使用逐路径特征。两版用相同的校准正例和相同区间位置，分别用开发集内层留出预测选门槛，再在 64 部开发作品上重拟合模型。除各自门槛外，还交叉使用两版门槛进行同门槛比较。合格提案同时记录区间数及至少有一处合格提案的查询数。定位指标以正确确认且有自动参考的查询为分母，无有效歌词单列。`e2_comparison.jsonl` 包含全部 80 条查询，并记录可评分查询在不同门槛下选出的区间和 IoU。
- `care/fold*/parameters.json` 保存两版参数。`development_oof.jsonl` 和 `development_metrics.json` 保存内层选参依据。`prior_care_differences.jsonl` 记录与上轮歌词开关实验 A 组的逐查询差异，空文件表示比较字段全部一致。

## 和声覆盖门槛的探索性复评

```bash
conda activate RL_HW4
python inference/confirmation_coverage.py smoke
python inference/confirmation_coverage.py run --jobs 4
python inference/confirmation_coverage.py verify
```

歌曲确认参数 `confirmation.music.min_harmony_coverage` 表示最长连续可靠和声段占 30 秒查询的比例。旧参数文件缺失该字段时取 0.50。将门槛设为 0 时仍需通过首名分数和领先分差检查。音频确认分支不使用和声覆盖门槛。完整推理可通过 `pipeline.py predict --parameters` 读取实验冻结参数。

本实验沿用冻结复评的 CoverHunter B 组召回、分数融合和候选。冻结复评保存了内层库内排序，脚本使用同一版校准和曲库协议重建内层库外分数，并逐条核对库内排名和确认输入。对照组沿用 50% 门槛和原阈值搜索规则。调参组对 0%、20%、30%、40%、50% 逐一搜索最低首名分和最小分差。对满足内层库外误报率不超过 5% 的组合，依次比较正确确认数、库内错误确认数、库外误报数和覆盖门槛。仍并列时优先较大的分差门槛和首名分门槛。音频确认规则沿用对照组。

五折确认参数冻结后，用原冻结复评的外层库内及库外预测重放确认层。库外预测源于先移除正确作品再召回。两组均输出同样的候选和分数用于核对。歌词分组以有效歌词门槛为准。实验默认写到 `precomputed_data/experiments/confirmation_coverage/`，含内层逐查询输入、完整搜索网格、两组冻结参数、库内外确认预测、汇总与变化清单。

## Kiki 整句歌词定位的探索性实验

```bash
conda activate RL_HW4
python inference/kiki_lyric_experiment.py baseline
python inference/kiki_lyric_experiment.py smoke
python inference/kiki_lyric_experiment.py run
python inference/kiki_lyric_experiment.py verify
```

从 CARE 根目录运行，默认目录为 `precomputed_data/experiments/kiki_lyric_localization/`。基线读取五折 0% 和声覆盖门槛的确认参数和冻结复评的 E2 置信度。已保存的提案直接复用，缺失的已确认提案用冻结模型补算。基线记录参考区间、源预测和参数的 SHA-256。评分使用当前 9 条人工复核、71 条自动参考，对每个可接受区间单独算 IoU 并取最大值。

`kiki_lyric_localization.py` 提供 `read_evidence`、`generate` 和 `fuse`，依次读取逐行文本与人声门槛、生成字面定位候选、与 E2 区间融合。调用方提供系统实际首名，无需传入评分标注。文本匹配保存原始行号、文本、秒数、句对分及最终路径残差。库外确认也对实际误确认录音使用同一流程。

D 组在每折的 64 条内层留出预测上，按当前歌曲和 E2 门槛重放，然后在 12 组配置中选择命中增加且错误不增加的方案。回退 A 的折仍用 C 的固定设置生成诊断候选，最终输出采用 A。开发和外层预测在五折选参冻结后分别保存。B/C/D 共享一次缓存读取与 gate，分别计时句匹配和融合。匹配时间包含清洗、句对生成和路径搜索，缓存反序列化单独计时，无有效歌词的查询不计入匹配调用数。

- `baseline_audit.json` 保存基线核对和 E2 补算记录。
- `fold*/configuration.json` 和 `development_trials.jsonl` 保存冻结设置、完整搜索及逐查询开发评分。
- `fold*/A.jsonl` 至 `D.jsonl` 保存库内外冻结排名、原 E2、歌词提案、最终结果和评分。
- `summary.csv` 和 `summary.json` 保存汇总和分折指标。JSON 还包含候选覆盖和回退原因。`changes.jsonl` 保存相对 A 的变化及已采用句对。

## Covers80 匹配区间对比

```bash
conda activate RL_HW4
python inference/covers80_interval_comparison.py run
python inference/covers80_interval_comparison.py verify
```

从 CARE 根目录运行。CARE 复用 Kiki 实验 A 组的实际确认与原 E2 输出。MERT 和 Unified 用冻结同窗配置重新检索 80 首 B 版录音，以实际首名的最高余弦窗口对和单位速度推算 30 秒参考区间。预测落盘后才按官方配对及每个可接受参考区间评分。

默认结果位于 `precomputed_data/experiments/covers80_interval_comparison/`。每种方法的 `*_predictions.jsonl` 保存首名、Top-10、分数、命中窗、报歌状态和参考区间，`*_scores.jsonl` 保存逐查询最大 IoU 和命中状态。`end_to_end.csv` 区分错歌出区间与歌对位置错。`same_correct_song.csv` 和 `common_subsets.json` 记录共同首名正确子集上的定位表现、暂缓和查询清单。`ranking_audit.json` 记录与历史同窗基线的逐候选核对。命中率按全子集计算，平均 IoU 以各方法实际输出区间为分母。

## 歌曲确认回放与人工复核后重计分

在已有参考区间与 23 条人工复核记录的项目目录中执行。

```bash
original_labels=precomputed_data/reference_labels/covers80/reference_intervals_before_23_review.jsonl
python inference/confirmation_replay.py restore-labels --output "$original_labels"
python inference/confirmation_replay.py report
```

恢复出的文件与 Kiki 整句实验原始参考区间的 SHA-256 一致。Kiki 实验和历史区间基线自动读取该文件，也可设置 `CARE_KIKI_REFERENCE_INTERVALS` 指定历史版本。上面的 `report` 读取最新参考区间与人工复核记录，复用无拒答 CARE 的预测和重跑后的 MERT、Unified 预测，输出分折、共同正确首名子集和 CARE 两条待确认样本的补计。只读评分可在 Python 标准库环境运行。

需要从冻结排序重新生成预测时，先完成冻结配置复评与歌曲确认门槛调参，然后用以下命令写入新目录。

```bash
python inference/confirmation_replay.py zero-coverage --labels "$original_labels" \
  --output precomputed_data/experiments/confirmation_coverage_zero_replay
python inference/confirmation_replay.py run --labels "$original_labels" \
  --output precomputed_data/experiments/confirmation_no_abstention_replay
python inference/confirmation_replay.py report \
  --care-predictions precomputed_data/experiments/confirmation_no_abstention_replay
```

`zero-coverage` 仅将歌曲确认的和声覆盖门槛设为 0。`run` 同时将歌曲确认的首名分、分差与和声覆盖门槛，以及 E2 的置信度和分差设为 0。回放固定原排序和模型，对缓存缺失的 E2 提案按冻结配置补算。
