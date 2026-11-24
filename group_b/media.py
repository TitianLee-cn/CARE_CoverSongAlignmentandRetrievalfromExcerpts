import json
import subprocess

import numpy as np


def decode(path):
    info=json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","a:0",
        "-show_entries","stream=sample_rate,channels","-of","json",str(path)]))["streams"][0]
    proc=subprocess.run(["ffmpeg","-nostdin","-v","error","-i",str(path),"-map","0:a:0",
                         "-f","f32le","-acodec","pcm_f32le","pipe:1"],capture_output=True,check=True)
    samples=np.frombuffer(proc.stdout,dtype="<f4").reshape(-1,info["channels"])
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError(f"Empty or nonfinite audio: {path}")
    return samples,int(info["sample_rate"]),len(proc.stderr.splitlines())
