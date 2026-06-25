import pytest

from graft import cli


def test_cli_list(capsys):
    rc = cli.main(["list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "mlp" in out
    assert "perceiver" in out


def test_cli_specs(capsys):
    rc = cli.main(["specs"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "clip-vit-l-14" in out


def test_cli_plan(capsys):
    rc = cli.main(
        ["plan", "--connector", "mlp", "--encoder", "clip-vit-l-14", "--llm", "llama-2-7b"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "4096" in out


def test_cli_describe_unknown_returns_error(capsys):
    rc = cli.main(["describe", "nope"])
    assert rc == 1


def test_cli_requires_subcommand():
    with pytest.raises(SystemExit):
        cli.main([])


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "graft" in capsys.readouterr().out
