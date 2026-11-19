from __future__ import annotations

from pathlib import Path

import pytest

from pheragent.cli import _batch_base_request_from_args, _build_parser, _request_from_args


def test_deploy_is_a_direct_command(tmp_path: Path) -> None:
    args = _build_parser().parse_args(["deploy", str(tmp_path / "task.yaml"), "--execute"])

    assert args.command == "deploy"
    assert args.task == tmp_path / "task.yaml"
    assert args.execute is True


def test_build_resume_from_does_not_require_base_dockerfile(tmp_path: Path) -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "build",
            "--repo",
            str(tmp_path),
            "--resume-from",
            "pheragent:previous-checkpoint",
            "--start-at-block",
            "02-tests",
        ]
    )

    request = _request_from_args(args, require_dockerfile=True)

    assert request.base_dockerfile is None
    assert request.resume_from == "pheragent:previous-checkpoint"
    assert request.start_at_block == "02-tests"


def test_build_accepts_chat_completions_llm_api(tmp_path: Path) -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "build",
            "--repo",
            str(tmp_path),
            "--base-dockerfile",
            str(tmp_path / "Dockerfile"),
            "--llm-api",
            "chat-completions",
        ]
    )

    request = _request_from_args(args, require_dockerfile=True)

    assert request.llm_api == "chat-completions"


def test_build_accepts_ablation_mode(tmp_path: Path) -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "build",
            "--repo",
            str(tmp_path),
            "--base-dockerfile",
            str(tmp_path / "Dockerfile"),
            "--ablation",
            "full",
        ]
    )

    request = _request_from_args(args, require_dockerfile=True)

    assert request.ablation_mode == "full"


def test_build_accepts_single_command_forward_ablation(tmp_path: Path) -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "build",
            "--repo",
            str(tmp_path),
            "--base-dockerfile",
            str(tmp_path / "Dockerfile"),
            "--ablation",
            "single-command-forward",
        ]
    )

    request = _request_from_args(args, require_dockerfile=True)

    assert request.ablation_mode == "single-command-forward"


def test_build_accepts_single_command_recovery_ablation(tmp_path: Path) -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "build",
            "--repo",
            str(tmp_path),
            "--base-dockerfile",
            str(tmp_path / "Dockerfile"),
            "--ablation",
            "single-command-recovery",
        ]
    )

    request = _request_from_args(args, require_dockerfile=True)

    assert request.ablation_mode == "single-command-recovery"


def test_build_accepts_whole_script_forward_ablation(tmp_path: Path) -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "build",
            "--repo",
            str(tmp_path),
            "--base-dockerfile",
            str(tmp_path / "Dockerfile"),
            "--ablation",
            "whole-script-forward",
        ]
    )

    request = _request_from_args(args, require_dockerfile=True)

    assert request.ablation_mode == "whole-script-forward"


def test_build_accepts_whole_script_recovery_ablation(tmp_path: Path) -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "build",
            "--repo",
            str(tmp_path),
            "--base-dockerfile",
            str(tmp_path / "Dockerfile"),
            "--ablation",
            "whole-script-recovery",
        ]
    )

    request = _request_from_args(args, require_dockerfile=True)

    assert request.ablation_mode == "whole-script-recovery"


def test_build_accepts_task_description_and_task_file(tmp_path: Path) -> None:
    task_file = tmp_path / "task.txt"
    task_file.write_text("Install speech CLI support\n", encoding="utf-8")
    parser = _build_parser()
    args = parser.parse_args(
        [
            "build",
            "--repo",
            str(tmp_path),
            "--base-dockerfile",
            str(tmp_path / "Dockerfile"),
            "--task-file",
            str(task_file),
            "--task-description",
            "Validate python -m whisper --help.",
        ]
    )

    request = _request_from_args(args, require_dockerfile=True)

    assert request.task_description == (
        "Install speech CLI support\n\nValidate python -m whisper --help."
    )


def test_build_projects_accepts_task_description(tmp_path: Path) -> None:
    projects_file = tmp_path / "projects.txt"
    projects_file.write_text("owner/repo abc123\n", encoding="utf-8")
    parser = _build_parser()
    args = parser.parse_args(
        [
            "build-projects",
            "--projects-file",
            str(projects_file),
            "--base-dockerfile",
            str(tmp_path / "Dockerfile"),
            "--task-description",
            "Setup repo for its SetupBench validation.",
        ]
    )

    request = _batch_base_request_from_args(args)

    assert request.task_description == "Setup repo for its SetupBench validation."


def test_build_projects_accepts_jobs(tmp_path: Path) -> None:
    projects_file = tmp_path / "projects.txt"
    projects_file.write_text("owner/repo abc123\n", encoding="utf-8")
    parser = _build_parser()
    args = parser.parse_args(
        [
            "build-projects",
            "--projects-file",
            str(projects_file),
            "--base-dockerfile",
            str(tmp_path / "Dockerfile"),
            "--jobs",
            "3",
        ]
    )

    assert args.jobs == 3


def test_build_projects_accepts_ablation_mode(tmp_path: Path) -> None:
    projects_file = tmp_path / "projects.txt"
    projects_file.write_text("owner/repo abc123\n", encoding="utf-8")
    parser = _build_parser()
    args = parser.parse_args(
        [
            "build-projects",
            "--projects-file",
            str(projects_file),
            "--base-dockerfile",
            str(tmp_path / "Dockerfile"),
            "--ablation",
            "without-checkpoint-rollback",
        ]
    )

    request = _batch_base_request_from_args(args)

    assert request.ablation_mode == "without-checkpoint-rollback"


def test_build_projects_rejects_zero_jobs(tmp_path: Path) -> None:
    projects_file = tmp_path / "projects.txt"
    projects_file.write_text("owner/repo abc123\n", encoding="utf-8")
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "build-projects",
                "--projects-file",
                str(projects_file),
                "--base-dockerfile",
                str(tmp_path / "Dockerfile"),
                "--jobs",
                "0",
            ]
        )


def test_build_without_resume_requires_base_dockerfile(tmp_path: Path) -> None:
    parser = _build_parser()
    args = parser.parse_args(["build", "--repo", str(tmp_path)])

    with pytest.raises(SystemExit, match="--base-dockerfile is required"):
        _request_from_args(args, require_dockerfile=True)
