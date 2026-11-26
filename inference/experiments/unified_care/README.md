# Unified + CARE on Covers80

This experiment uses the shared CARE inference code from `main` at `5c687c3`. Keep that checkout separate from `JY`. The patch adds Unified as an audio model and preserves the CoverHunter default. It also allows the experiment to pass the interval labels used during the original fit.

Run in the CARE environment described in the [main README](https://github.com/TitianLee-cn/CARE_CoverSongAlignmentandRetrievalfromExcerpts/blob/5c687c3/README.md). Populate the main checkout with the existing `shared`, `cover_embeddings/{unified,coverhunter,mert}`, `beat_chroma`, `cover_pair_alignment` and `group_b` caches under `precomputed_data`. For audio export, use the matching Covers80 recordings under `data_weights/datasets` and make FFmpeg available. Keep the reference labels tracked by the main checkout.

From the main checkout, with `JY_ROOT` set to the absolute path of the JY checkout:

```bash
code="$JY_ROOT/inference/experiments/unified_care"
git apply "$code/pipeline-unified.patch"
python "$code/retune_roles.py"
python "$code/run_role_swap_full.py" fit
python "$code/run_role_swap_full.py" retune
python "$code/run_role_swap_full.py" predict
python "$code/rescore_human_review.py" .
```

The scorers write experiment outputs under `precomputed_data/experiments/unified_role_swap` or print the final review summary. The report and the reviewed summary are available in [Covers80_result.md](../../../docs/Covers80_result.md) and [review_summary.json](../../../precomputed_data/review/unified_care/review_summary.json).

## Review audio

The [audio ZIP](../../../assets/covers80/unified_care_review_audio.zip) is tracked with Git LFS. It contains 80 query clips, 80 predicted B clips, the three hand-marked B clips, the manifest, and 24 human annotation records. Four accepted intervals point to their predicted B clip. The remaining queries have `not_reviewed` in the manifest for this round.

```bash
git lfs pull --include="assets/covers80/unified_care_review_audio.zip"
```

To regenerate it from the main checkout:

```bash
python "$code/export_role_swap_clips.py" outputs/unified_care_e2_clips.zip
python "$code/export_review_audio.py" outputs/unified_care_e2_clips.zip \
  outputs/unified_care_review_audio.zip --main-checkout . \
  --manifest outputs/unified_care_audio_manifest.csv
```

Use `--review-clips` in place of `--main-checkout` if the prior review ZIP already contains the three manually cut WAV clips. See the [Covers80 dataset description](https://www.ee.columbia.edu/~dpwe/LabROSA/projects/coversongs/covers80/) for the original recordings and citation.
