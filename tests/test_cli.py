import json
import subprocess
import sys


def test_train_checkpoint_can_be_evaluated(tmp_path):
    run = tmp_path / "training"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "hover",
            "train",
            "--steps",
            "256",
            "--episode-seconds",
            "0.1",
            "--eval-episodes",
            "1",
            "--output",
            str(run),
        ],
        check=True,
    )
    assert (run / "policy.zip").is_file()
    assert (run / "progress.csv").is_file()
    config = json.loads((run / "config.json").read_text())
    assert config["steps"] == 256
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hover",
            "evaluate",
            "--checkpoint",
            str(run / "policy.zip"),
            "--episode-seconds",
            "0.1",
            "--eval-episodes",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metrics = json.loads(result.stdout)
    assert metrics["episodes"] == 1
    assert metrics["mean_position_error_m"] >= 0
