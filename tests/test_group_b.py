import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from group_b.annotations import normalize
from group_b.cover_alignment import reliable_blocks
from group_b.evaluate import evaluate
from group_b.jamendo import evaluate as timestamps
from group_b.jamendo import pool_frames
from group_b.lyrics import clean_line, gate, local_alignment, ordered_lines
from group_b.pairs import DEFAULTS, fold_sets, music_pairs, predict


def line(i, start, end, text):
    return dict(line_id=i,start_sec=start,end_sec=end,text=text,clean=clean_line(text),vector=[1.,0.])


class BranchTests(unittest.TestCase):
    def test_blocks_skip_unreliable_gap_and_merge_repeated_boundaries(self):
        bounds = np.array([[0,1,5,6], [0,1,6,7], [1,2,7,8], [2,3,8,9], [3,4,9,10]])
        blocks = reliable_blocks(bounds, [True,True,False,True,True])
        self.assertEqual(blocks, [[[0.,5.5],[1.,6.5]], [[2.,8.],[3.,9.],[4.,10.]]])

    def test_acoustic_union_and_filler(self):
        vocals=dict(frame_sec=1,vocal_power=[1.,1.,1.,0.],vocal_to_mix_power=[1.]*4)
        q=[line(1,.5,2,"oh keep going"),line(2,1,4,"keep singing")]
        self.assertEqual(clean_line("Oh la-la 啊啊"),"")
        self.assertEqual(q[0]["clean"],"keep going")
        self.assertAlmostEqual(gate(q,vocals,2)["seconds"],2.5)
        self.assertFalse(gate(q,vocals,3)["enabled"])
        self.assertTrue(gate(q,vocals,2)["enabled"])

    def test_local_alignment(self):
        raw,path=local_alignment(np.array([[0,.9,.1],[0,0,1.]]),.5)
        self.assertEqual(path,[(0,1),(1,2)])
        self.assertAlmostEqual(raw,.45)

    def test_jamendo_denominator_and_no_time_leak(self):
        gold=[{"text":"first line","start":1.,"end":2.,"line_id":1,"valid":True},
              {"text":"another line","start":3.,"end":4.,"line_id":2,"valid":True}]
        pred=[line(1,1.1,2.2,"first line")]
        self.assertEqual(ordered_lines(gold,pred,.8),[(0,0)])
        with patch("group_b.jamendo.load_stage",return_value={"lines":pred}):
            out=timestamps([{"recording_id":"s"}],{"s":gold},"kiki",.8,None)
        self.assertEqual(out["line_recall"],.5)
        self.assertAlmostEqual(out["end_mae_sec"],.2)
        self.assertEqual(out["both_within_05_all_references"],.5)

    def test_no_music_extrapolation_and_split_lines(self):
        q=[line(1,0,2,"hello there"),line(2,3,5,"good bye")]
        r=[line(1,4,5,"hello"),line(2,5,6,"there")]
        display=music_pairs(q,r,[[[0,4],[2,6]]],None,DEFAULTS,"fusion")
        self.assertEqual([(p["query_lines"],p["reference_lines"]) for p in display],[([1],[1,2])])

    def test_no_type_for_insufficient_lyrics(self):
        from unittest.mock import Mock
        corpus=Mock()
        corpus.score_candidate.return_value={"gate":{"enabled":False}}
        row=predict(corpus,{"work_id":"x","reference_work_id":"x","confirmed":True},{}, "fusion")
        self.assertEqual(row["type_status"],"skipped")
        self.assertNotIn("cover_type",row)

    def test_abstentions_count_as_misses(self):
        p=dict(reference_work_id="x",type_status="pending",gate={"enabled":True})
        out=evaluate({"x":p},{"x":{"cover_type":"original","valid_lyrics":True,"pairs":[]}})
        self.assertEqual(out["classes"]["original"]["recall"],0)
        self.assertEqual(out["matrix"]["original"]["pending"],1)
        self.assertEqual(out["joint_accuracy"],0)

    def test_nested_splits(self):
        works=[f"w{i}" for i in range(80)]
        folds=[]
        for i in range(5):
            held=works[i*16:(i+1)*16]
            dev=[work for work in works if work not in held]
            inner=[]
            for j in range(4):
                val=dev[j*16:(j+1)*16]
                inner.append(dict(inner_fold=j+1,validation_work_ids=val,
                                  fit_work_ids=[work for work in dev if work not in val]))
            folds.append(dict(outer_fold=i+1,test_work_ids=held,development_work_ids=dev,inner=inner))
        test=[]
        with patch("group_b.pairs.rows",return_value=folds):
            for outer in range(1,6):
                dev,held=fold_sets(outer,None)
                test+=held
                validation=[]
                for inner in range(1,5):
                    fit,val=fold_sets(outer,inner)
                    self.assertEqual(set(fit+val),set(dev))
                    validation+=val
                self.assertCountEqual(dev,validation)
        self.assertCountEqual(test,works)

    def test_automatic_draft_is_rejected_as_a_human_label(self):
        identity=dict(work_id="w",query_id="q",query_recording_id="a",reference_recording_id="b",official_index=41)
        with TemporaryDirectory() as tmp, patch("group_b.annotations.MANIFESTS",Path(tmp)):
            root=Path(tmp)
            (root/"annotation_template.jsonl").write_text(json.dumps(identity)+"\n",encoding="utf-8")
            source=root/"draft.jsonl"
            source.write_text(json.dumps({**identity,"annotation_status":"auto_draft"})+"\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"Automatic draft"):
                normalize([source],root/"labels.jsonl")

    def test_mert_pool_uses_predicted_half_open_interval(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            np.savez(root/"a.npz",times=[0.,.5,1.],vectors=[[1.,2.],[3.,4.],[99.,99.]])
            idx=root/"frames.jsonl"
            idx.write_text(json.dumps(dict(recording_id="a",sha256="test",file="a.npz")),encoding="utf8")
            with patch("group_b.jamendo.load_stage",return_value={"lines":[line(1,0,1,"hello there")]}):
                pool_frames(idx,[dict(recording_id="a",sha256="test")],"kiki",root)
            pooled=json.loads((root/"jamendo/mert-kiki/a.json").read_text())["lines"][0]
            self.assertEqual(pooled["vector"],[2.,3.])
            self.assertEqual(pooled["frames"],2)


if __name__=="__main__":
    unittest.main()
