"""Package held-out Covers80 inputs and the lyric clips emitted by the fusion branch."""
import argparse
import hashlib
import html
import json
import random
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .cache import MANIFESTS, OUTPUT, ROOT, load_stage, read, rows, write


def crop(row, left, right, target):
    rate = row["sample_rate"]
    if not 0 <= left < right <= row["num_samples"]:
        raise ValueError(f"Clip outside {row['recording_id']}: {left}:{right}")
    audio = ROOT/"datasets"/row["path"]
    with audio.open("rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest()!=row["sha256"]:
            raise ValueError(f"Audio checksum differs: {audio}")
    subprocess.run(["ffmpeg","-nostdin","-v","error","-i",str(audio),
                    "-af",f"atrim=start_sample={left}:end_sample={right},asetpts=PTS-STARTPTS",
                    "-c:a","pcm_s16le","-ar",str(rate),str(target)],check=True)


def produce(folder):
    q = {r["work_id"]:r for r in rows(MANIFESTS/"covers80_queries.jsonl")}
    refs = {r["work_id"]:r for r in rows(MANIFESTS/"covers80_references.jsonl")}
    folds = rows(MANIFESTS/"covers80_folds.jsonl")
    fitted = read(OUTPUT/"scores/covers80-lyric-evidence.json")
    fusion_path = OUTPUT/"cover_pair_alignment/fusion.json"
    fusion = read(fusion_path)
    predictions = {r["work_id"]:r for r in fusion["predictions"]}
    template = [r for r in rows(MANIFESTS/"annotation_template.jsonl") if r["annotator"]=="B"]
    b_works = {r["work_id"] for r in template}
    seed = 42
    rng = random.Random(seed)
    folder.mkdir(parents=True,exist_ok=False)
    (folder/"audio").mkdir()
    entries = []
    for fold in folds:
        fold_id = fold["outer_fold"]
        eligible = [w for w in sorted(fold["test_work_ids"]) if w in b_works and predictions[w]["display"]]
        for work in sorted(rng.sample(eligible,4)):
            query,ref,pred = q[work],refs[work],predictions[work]
            if (pred["outer_fold"],pred["query_id"],pred["reference_recording_id"],pred["reference_work_id"]) != (
                    fold_id,query["query_id"],ref["recording_id"],work):
                raise ValueError(f"Prediction disagrees with test fold: {work}")
            pair = max(pred["display"],key=lambda p:p["score"])
            rate = ref["sample_rate"]
            a,b = (round(t*rate) for t in pair["reference_sec"])
            qi = f"audio/{work}_input_30s.wav"
            ri = f"audio/{work}_reference_lyrics.wav"
            crop(query,query["start_sample"],query["end_sample"],folder/qi)
            crop(ref,a,b,folder/ri)
            q_lines = {l["line_id"]:l["text"] for l in load_stage("query","kiki",query)["lines"]}
            r_lines = {l["line_id"]:l["text"] for l in load_stage("reference","kiki",ref)["lines"]}
            entries.append(dict(work_id=work,title=query["work_name"],outer_fold=fold_id,query_id=query["query_id"],
                query_recording_id=query["recording_id"],reference_recording_id=ref["recording_id"],
                query_source=query["path"],reference_source=ref["path"],query_sha256=query["sha256"],
                reference_sha256=ref["sha256"],input_audio=qi,reference_audio=ri,
                input_source_sec=[query["start_sec"],query["end_sec"]],
                reference_audio_source_sec=[a/rate,b/rate],
                query_lyrics=" / ".join(q_lines[i] for i in pair["query_lines"]),
                reference_lyrics=" / ".join(r_lines[i] for i in pair["reference_lines"]),
                match=pair,system_type=pred.get("cover_type"),system_type_status=pred["type_status"]))
    write(folder/"review_manifest.json",dict(context="oracle_correct_pair",frozen_config=fusion["config"],
          music_cosine_min=fusion["cosine_min"],alignment_archive_sha256=fusion["alignment_archive_sha256"],
          prediction_sha256=hashlib.sha256(fusion_path.read_bytes()).hexdigest(),
          tuning_status="development-fit background percentiles and fixed provisional thresholds; no human labels used",
          selection=f"Seed {seed}, four B-side test works with displayed pairs per fold; highest scored pair per work",
          outer_calibration=[{k:f[k] for k in ("outer_fold","fit_work_ids","background")} for f in fitted["folds"]],
          entries=entries))
    (folder/"annotations_b_blank.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in template),encoding="utf-8")
    page(folder/"index.html",entries)
    with ZipFile(folder.with_suffix(".zip"),"x",compression=ZIP_DEFLATED) as z:
        for file in sorted(folder.rglob("*")):
            if file.is_file():
                z.write(file,file.relative_to(folder).as_posix())


def page(path, entries):
    cards=[]
    for row in entries:
        e=html.escape
        a,b=row["match"]["query_sec"]
        l,r=row["reference_audio_source_sec"]
        s,t=row["input_source_sec"]
        cards.append(f"""<article data-work="{row['work_id']}">
<h2>{e(row['title'])} <small>第 {row['outer_fold']} 折 · {row['work_id']}</small></h2>
<div class="sides"><section><h3>原始 30 秒输入</h3>
<p>原录音 {s:.2f}–{t:.2f} 秒 · 歌词在片段内 {a:.2f}–{b:.2f} 秒</p>
<p class="lyrics">{e(row['query_lyrics'])}</p>
<audio controls preload="none" src="{row['input_audio']}"></audio>
<button type="button" class="clip" data-start="{a}" data-end="{b}">只听输入侧歌词</button></section>
<section><h3>系统找到的参考歌词片段</h3>
<p>参考原录音 {l:.2f}–{r:.2f} 秒</p><p class="lyrics">{e(row['reference_lyrics'])}</p>
<audio controls preload="none" src="{row['reference_audio']}"></audio></section></div>
<div class="review"><label>试听判断 <select><option value="">未检查</option><option value="correct">对应</option>
<option value="incorrect">不对应</option><option value="uncertain">不确定</option></select></label>
<label>备注 <input placeholder="可记正确位置、切分或转写问题"></label></div></article>""")
    path.write_text("""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>CARE · Covers80 人工试听</title>
<style>
body{font:16px/1.55 "Microsoft YaHei",system-ui,sans-serif;background:#f3f5f7;color:#17232e;margin:0}
main{max-width:960px;margin:auto;padding:24px}h1{margin:0}h2{font-size:1.2rem}
small{font-size:.75rem;font-weight:normal;color:#475569}article{margin:18px 0;padding:18px;background:white;border:1px solid #d8e0e7;border-radius:10px}
.sides{display:grid;grid-template-columns:1fr 1fr;gap:14px}.sides section{background:#f3f7fa;padding:12px;border-radius:7px}
h3{margin:0}p{margin:6px 0}.lyrics{font-weight:600;overflow-wrap:anywhere}
audio{width:100%;margin-top:8px}button,select,input{font:inherit;padding:4px 8px}
button{cursor:pointer} .review{margin-top:12px;display:flex;gap:14px;flex-wrap:wrap}
@media(max-width:680px){.sides{grid-template-columns:1fr}}
</style><main><h1>Covers80 人工试听</h1>
<p>20 组 · 每个测试折 4 组。输入是官方清单中的原始 30 秒。参考录音来自 A 组给定的正确版本，所展示的歌词和片段由融合分支选择。</p>
<p>试听后可记录「对应」「不对应」或「不确定」，再导出 JSONL。双语种、类型和多处可接受区间可填入随包的空白标注模板。</p>
<button id="export" type="button">导出试听判断</button>"""+"".join(cards)+r"""
</main><script>
const cards = [...document.querySelectorAll('article')];
const key = 'care-covers80-review20-frozen-v1';
let prior = {};
try { prior = JSON.parse(localStorage.getItem(key) || '{}'); } catch {}
const state = () => cards.map(c => ({
  work_id:c.dataset.work, verdict:c.querySelector('select').value || null,
  notes:c.querySelector('input').value
}));
for (const c of cards) {
  const old = prior[c.dataset.work];
  if (old) {
    c.querySelector('select').value = old.verdict || '';
    c.querySelector('input').value = old.notes || '';
  }
  c.addEventListener('input', () => {
    try { localStorage.setItem(key, JSON.stringify(Object.fromEntries(state().map(r => [r.work_id,r])))); } catch {}
  });
  const audio = c.querySelector('audio');
  const button = c.querySelector('.clip');
  let stopAt = null;
  button.onclick = () => {
    const play = () => {
      audio.currentTime = Number(button.dataset.start);
      stopAt = Number(button.dataset.end);
      audio.play().catch(() => { stopAt = null; });
    };
    if (audio.readyState > 0) play();
    else { audio.addEventListener('loadedmetadata',play,{once:true}); audio.load(); }
  };
  audio.addEventListener('timeupdate', () => {
    if (stopAt !== null && audio.currentTime >= stopAt) { audio.pause(); stopAt = null; }
  });
}
document.querySelector('#export').onclick = () => {
  const url = URL.createObjectURL(new Blob([state().map(r => JSON.stringify(r)).join('\n')+'\n'],{type:'application/jsonl'}));
  const link = document.createElement('a');
  link.href=url;
  link.download='covers80_listening_review_20.jsonl';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url),1000);
};
</script></html>""",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("folder",type=Path)
    produce(p.parse_args().folder)
