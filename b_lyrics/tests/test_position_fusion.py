import unittest
from unittest.mock import Mock

from group_b.pairs import predict
from group_b.position_fusion import score_positions, select_reference_interval

WEIGHTS=dict(lyrics_mix=.55,audio_temperature=.12,agreement_bonus=.36,far_penalty=0.)


def candidate(start,audio,lyric):
    return dict(start_sec=start,audio=audio,a_path=0.,anchors=lyric,
                timed_words=lyric,window_words=lyric,far_penalty=0.)


class PositionTests(unittest.TestCase):
    def test_local_agreement_and_bias(self):
        candidates=[candidate(0.,.9,.2),candidate(30.,.8,.85),candidate(60.,.2,.9)]
        result=select_reference_interval(candidates,100.,WEIGHTS,start_bias_sec=.5)
        self.assertEqual(result["candidate_index"],1)
        self.assertEqual(result["reference_interval"],[30.5,60.5])

    def test_empty_and_bounds(self):
        self.assertIsNone(select_reference_interval([],45.,WEIGHTS))
        result=select_reference_interval([candidate(0.,.8,0.)],12.,WEIGHTS)
        self.assertEqual(result["reference_interval"],[0.,12.])
        with self.assertRaisesRegex(ValueError,"extends past"):
            select_reference_interval([candidate(20.,.8,0.)],45.,WEIGHTS)
        with self.assertRaisesRegex(ValueError,"finite"):
            score_positions([candidate(0.,float("nan"),0.)],**WEIGHTS)

    def test_confirmed_query_without_lyrics_uses_music_for_interval(self):
        corpus=Mock()
        corpus.refs={"r":{"duration_sec":100.}}
        corpus.score_candidate.side_effect=lambda *args,**kwargs: {"gate":{"enabled":False}}
        request=dict(work_id="q",reference_work_id="r",confirmed=True,context="end_to_end",
                     position_candidates=[candidate(5.,.9,.0),candidate(60.,.2,.99)],
                     position_config={"weights":WEIGHTS})
        result=predict(corpus,request,{},"fusion")
        self.assertEqual(result["reference_interval"],[5.,35.])
        self.assertEqual(result["type_status"],"skipped")
        self.assertNotIn("position_fusion",predict(corpus,{**request,"confirmed":False},{},"fusion"))
        self.assertNotIn("position_fusion",predict(corpus,{**request,"reference_interval":[8,38]},{},"fusion"))
        self.assertNotIn("position_fusion",predict(corpus,request,{},"music"))


if __name__=="__main__":
    unittest.main()
