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


def run_hover(*args):
    return subprocess.run(
        [sys.executable, "-m", "hover", *map(str, args)], check=True, capture_output=True, text=True
    )


def test_vector_training_checkpoints_and_resumes(tmp_path):
    run = tmp_path / "vector"
    common = ["--episode-seconds", "0.2", "--eval-episodes", "1"]
    # 4 worlds x 16 steps = 64-step rollouts; checkpoint every rollout.
    first = [
        "train",
        "--n-worlds",
        4,
        "--n-steps",
        16,
        "--batch-size",
        32,
        "--n-epochs",
        1,
        "--checkpoint-every",
        64,
        "--steps",
        128,
        "--output",
        run,
        *common,
    ]
    run_hover(*first)
    config = json.loads((run / "config.json").read_text())
    assert config["n_worlds"] == 4
    assert config["devices"]["sim_device"] == "cpu"
    assert config["devices"]["policy_device"] == "cpu"
    assert sorted(p.name for p in (run / "checkpoints").iterdir()) == [
        "hover_128_steps.zip",
        "hover_64_steps.zip",
    ]
    # Simulate an interrupted job: drop the final artifacts and extend the budget.
    (run / "evaluation.json").unlink()
    (run / "policy.zip").unlink()
    run_hover("train", "--resume", run, "--steps", 256)
    result = json.loads((run / "evaluation.json").read_text())
    assert result["actual_training_steps"] == 256
    assert result["throughput"]["steps"] == 128
    config = json.loads((run / "config.json").read_text())
    assert config["resumes"][0]["from_steps"] == 128
    assert (run / "progress.csv").is_file()
    assert (run / "progress.resume-128.csv").is_file()
    # A finished run is left alone.
    assert "already finished" in run_hover("train", "--resume", run).stdout


def test_baselines_and_benchmark_report_json():
    common = ["--episode-seconds", "0.2", "--eval-episodes", "1"]
    baselines = json.loads(run_hover("baselines", *common).stdout)
    assert set(baselines) == {"zero_action", "position_pd"}
    bench = json.loads(run_hover("benchmark", "--n-worlds", 8, "--steps", 80, *common).stdout)
    assert bench["world_steps"] == 80
    assert bench["world_steps_per_second"] > 0
