"""``bngsim run`` and ``bngsim check`` with forward sensitivities (issue #1925).

``--sensitivity-params`` and ``--sensitivity-ic`` are ``Simulator``'s keywords
of the same names, so each test holds the command's JSON to what the call
itself returns. With them, the digest is ``Result.digest(sensitivities=True)``,
and ``bngsim check`` reaches the sensitivity checks of #1893 from the real
command line, with no patching.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
from bngsim import (
    ArgumentTypeError,
    Model,
    ModelError,
    ParameterError,
    Simulator,
    _check,
    _cli,
    examples,
)

CHAIN = str(examples.path("chain", "sbml"))  # A -> B -> C; k1, k2
BIRTH_DEATH = str(examples.path("birth_death", "net"))
DECAY = str(examples.path("decay", "net"))  # A() -> 0; k, A0

#: A species whose BioNetGen name holds a comma, which a list split at every
#: comma would cut in two.
_COMMA_NET = """\
begin parameters
    1 k 0.5
end parameters
begin species
    1 R(l,d) 10
    2 X() 0
end species
begin reactions
    1 1 2 k #_R1
end reactions
begin groups
    1 Rtot 1
    2 Xtot 2
    3 Both 1,2
end groups
"""


def _bngsim(*args: str, script: str | None = None, cwd=None) -> subprocess.CompletedProcess:
    if script is None:
        cmd = [sys.executable, "-m", "bngsim", *args]
    else:
        code = script + f"\nfrom bngsim._cli import main\nraise SystemExit(main({list(args)!r}))\n"
        cmd = [sys.executable, "-c", code]
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", timeout=300, cwd=cwd
    )


def _json(run: subprocess.CompletedProcess, exit_code: int = 0) -> dict:
    assert run.returncode == exit_code, (run.returncode, run.stderr)
    return json.loads(run.stdout)


def _error(run: subprocess.CompletedProcess, exit_code: int, code: str) -> dict:
    assert run.returncode == exit_code, (run.returncode, run.stderr)
    assert run.stdout == ""
    error = json.loads(run.stderr)
    assert error["code"] == code, error
    return error


def _plain(value):
    return json.loads(json.dumps(value))


_WINDOW = ("--t-end", "10", "--n-points", "11")


def _python_run(path: str, **sim_kw) -> object:
    sim = Simulator(Model.load(path), method="ode", **sim_kw)
    return sim.run(t_span=(0, 10), n_points=11, timeout=600)


# ─── run ─────────────────────────────────────────────────────────────────────


def test_run_with_sensitivity_params_digests_them_as_the_python_call_does():
    d = _json(_bngsim("run", CHAIN, *_WINDOW, "--sensitivity-params", "k1,k2", "--json"))
    r = _python_run(CHAIN, sensitivity_params=["k1", "k2"])
    assert d["digest"] == _plain(r.digest(sensitivities=True))
    assert d["record"]["run"]["sensitivity_params"] == ["k1", "k2"]
    ranked = d["digest"]["observables"]["columns"][0]["sensitivity"]["parameters"]
    assert {p["name"] for p in ranked} == {"k1", "k2"}


def test_run_without_json_prints_the_sensitivity_digest_text():
    run = _bngsim("run", CHAIN, *_WINDOW, "--sensitivity-params", "k1")
    assert run.returncode == 0, run.stderr
    r = _python_run(CHAIN, sensitivity_params=["k1"])
    assert run.stdout == f"chain.sbml: {r.digest(sensitivities=True)}\n"
    assert "by dln y/dln p" in run.stdout


def test_sensitivity_ic_and_max_parameters_reach_the_call_and_the_digest():
    d = _json(
        _bngsim(
            "run", CHAIN, *_WINDOW, "--sensitivity-params", "k1,k2", "--sensitivity-ic", "A",
            "--max-parameters", "1", "--json",
        )
    )  # fmt: skip
    r = _python_run(CHAIN, sensitivity_params=["k1", "k2"], sensitivity_ic=["A"])
    assert d["digest"] == _plain(r.digest(sensitivities=True, max_parameters=1))
    assert d["record"]["run"]["sensitivity_ic"] == ["A"]
    assert d["digest"]["sensitivity"]["max_parameters"] == 1


def test_a_run_without_the_options_digests_no_sensitivities():
    d = _json(_bngsim("run", CHAIN, *_WINDOW, "--json"))
    assert "sensitivity" not in d["digest"]
    assert d["record"]["run"]["sensitivity_params"] == []


def test_the_options_repeat_and_split_only_at_commas_outside_parentheses(tmp_path):
    assert _cli._names("k1, k2") == ["k1", "k2"]
    assert _cli._names("R(l,d),X()") == ["R(l,d)", "X()"]
    assert _cli._names("A(b!1).B(a!1,c~P)") == ["A(b!1).B(a!1,c~P)"]
    d = _json(
        _bngsim(
            "run", CHAIN, *_WINDOW, "--sensitivity-params", "k1", "--sensitivity-params", "k2",
            "--json",
        )
    )  # fmt: skip
    assert d["record"]["run"]["sensitivity_params"] == ["k1", "k2"]
    net = tmp_path / "comma.net"
    net.write_text(_COMMA_NET)
    d = _json(_bngsim("run", str(net), *_WINDOW, "--sensitivity-ic", "R(l,d)", "--json"))
    assert d["record"]["run"]["sensitivity_ic"] == ["R(l,d)"]
    r = _python_run(str(net), sensitivity_ic=["R(l,d)"])
    assert d["digest"] == _plain(r.digest(sensitivities=True))


@pytest.mark.parametrize("suffix", [".h5", ".hdf5"])
def test_out_h5_holds_the_sensitivities(tmp_path, suffix):
    pytest.importorskip("h5py")
    from bngsim import Result

    out = tmp_path / f"sens{suffix}"
    d = _json(
        _bngsim(
            "run", CHAIN, *_WINDOW, "--sensitivity-params", "k1", "--out", str(out), "--json"
        )
    )  # fmt: skip
    assert d["out"] == str(out.resolve())
    back = Result.load(out)
    assert list(back.sensitivity_params) == ["k1"]
    np.testing.assert_array_equal(
        back.sensitivities, _python_run(CHAIN, sensitivity_params=["k1"]).sensitivities
    )


def test_out_npz_holds_the_sensitivities(tmp_path):
    out = tmp_path / "sens.npz"
    d = _json(
        _bngsim(
            "run", CHAIN, *_WINDOW, "--sensitivity-params", "k1", "--out", str(out), "--json"
        )
    )  # fmt: skip
    assert d["out"] == str(out.resolve())
    r = _python_run(CHAIN, sensitivity_params=["k1"])
    with np.load(out, allow_pickle=False) as z:
        assert list(z["sensitivity_params"]) == ["k1"]
        np.testing.assert_array_equal(z["sensitivities_species"], r.sensitivities)


# ─── What the command refuses ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "args, argument, says",
    [
        (["run", BIRTH_DEATH, "--method", "ssa", "--t-end", "1", "--sensitivity-params", "k"],
         "--sensitivity-params", "an ODE run's; a ssa run takes none"),
        (["check", BIRTH_DEATH, "--method", "ssa", "--t-end", "1", "--sensitivity-ic", "X()"],
         "--sensitivity-ic", "an ODE run's; a ssa run takes none"),
        (["run", CHAIN, "--t-end", "1", "--max-parameters", "3"], "--max-parameters",
         "for a run with --sensitivity-params or --sensitivity-ic"),
        (["run", CHAIN, "--t-end", "1", "--sensitivity-params", "k1", "--out", "x.csv"], "--out",
         "no sensitivities"),
        (["run", CHAIN, "--t-end", "1", "--sensitivity-ic", "A", "--out", "x.tsv",
          "--out-kind", "species"], "--out", "no sensitivities"),
        (["run", CHAIN, "--t-end", "1", "--sensitivity-params", "k1,,k2"],
         "--sensitivity-params", "with none empty"),
        (["run", CHAIN, "--t-end", "1", "--sensitivity-params", ""], "--sensitivity-params",
         "with none empty"),
        (["run", CHAIN, "--t-end", "1", "--sensitivity-params", "k1,k1"],
         "--sensitivity-params", "names k1 more than once"),
        (["run", CHAIN, "--t-end", "1", "--sensitivity-params", "k1", "--sensitivity-params",
          "k1"], "--sensitivity-params", "names k1 more than once"),
        (["run", CHAIN, "--t-end", "1", "--sensitivity-ic", "A(b,"], "--sensitivity-ic",
         "unbalanced parentheses"),
        (["run", CHAIN, "--t-end", "1", "--sensitivity-ic", "A)"], "--sensitivity-ic",
         "unbalanced parentheses"),
        (["run", CHAIN, "--t-end", "1", "--sensitivity-params", "k11"], "--sensitivity-params",
         "--sensitivity-params k11: Parameter 'k11' not found in model."),
        (["check", CHAIN, "--t-end", "1", "--sensitivity-ic", "a"], "--sensitivity-ic",
         "--sensitivity-ic a: Species 'a' not found in model."),
        (["run", CHAIN, "--t-end", "1", "--sensitivity-params", "A"], "--sensitivity-params",
         "'A' is a species, not a parameter: --sensitivity-ic A computes sensitivities to its "
         "initial amount."),
        (["run", CHAIN, "--t-end", "1", "--sensitivity-ic", "k1"], "--sensitivity-ic",
         "'k1' is a parameter, not a species: --sensitivity-params k1 computes sensitivities "
         "to it."),
    ],
)  # fmt: skip
def test_a_wrong_sensitivity_command_line_exits_2(args, argument, says, tmp_path):
    error = _error(
        _bngsim(*args, "--json", cwd=tmp_path), _cli.EXIT_USAGE, "usage.invalid_argument"
    )
    assert error["details"]["argument"] == argument, error
    assert says in error["message"], error
    assert not list(tmp_path.iterdir())  # refused before anything was written


def test_a_name_the_model_lacks_names_the_close_ones():
    error = _error(
        _bngsim("run", CHAIN, "--t-end", "1", "--sensitivity-params", "k11", "--json"),
        _cli.EXIT_USAGE,
        "usage.invalid_argument",
    )
    assert "k1" in error["details"]["close_matches"], error
    error = _error(
        _bngsim("run", CHAIN, "--t-end", "1", "--sensitivity-ic", "a", "--json"),
        _cli.EXIT_USAGE,
        "usage.invalid_argument",
    )
    assert error["details"]["close_matches"] == ["A"], error
    error = _error(
        _bngsim("run", CHAIN, "--t-end", "1", "--sensitivity-params", "A", "--json"),
        _cli.EXIT_USAGE,
        "usage.invalid_argument",
    )
    assert error["remedies"] == [
        {"kind": "pass_argument", "argument": "--sensitivity-ic", "value": "A"}
    ]


def test_the_other_run_commands_take_no_sensitivity_option():
    # Only run and check make a sensitivity run: elsewhere argparse refuses
    # the option, and names it, rather than dropping it.
    for command in ("run-until", "run-batch", "parameter-scan"):
        run = _bngsim(command, CHAIN, "--t-end", "1", "--sensitivity-params", "k1", "--json")
        error = _error(run, _cli.EXIT_USAGE, "usage.invalid_argument")
        assert "--sensitivity-params" in error["message"], (command, error)


# ─── check ───────────────────────────────────────────────────────────────────


def test_check_of_a_sensitivity_run_passes_with_the_sensitivity_check():
    d = _json(_bngsim("check", CHAIN, *_WINDOW, "--sensitivity-params", "k1,k2", "--json"))
    names = [c["name"] for c in d["check"]["checks"]]
    assert names == [*_check.DEFAULT_DETERMINISTIC, *_check.DEFAULT_SENSITIVITY]
    assert {c["status"] for c in d["check"]["checks"]} == {"pass"}
    assert d["digest"] == _plain(
        _python_run(CHAIN, sensitivity_params=["k1", "k2"]).digest(sensitivities=True)
    )


def test_check_exits_6_when_the_sensitivities_move_with_the_tolerance():
    # rtol 1e-2: dB/dk1 moves by about 4e-3 of its scale at rtol/10, past the
    # bound of 1e-3. A real run, nothing patched.
    run = _bngsim(
        "check", CHAIN, *_WINDOW, "--rtol", "1e-2", "--atol", "1e-2", "--sensitivity-params",
        "k1,k2", "--checks", "sensitivity", "--json",
    )  # fmt: skip
    d = _json(run, _cli.EXIT_CHECK_FAILED)
    (verdict,) = d["check"]["checks"]
    assert (verdict["name"], verdict["status"]) == ("sensitivity", "diff")
    assert d["check"]["ok"] is False


def test_check_of_sensitivities_on_a_run_with_none_is_unsupported_and_exits_6():
    d = _json(
        _bngsim("check", CHAIN, *_WINDOW, "--checks", "sensitivity", "--json"),
        _cli.EXIT_CHECK_FAILED,
    )
    (verdict,) = d["check"]["checks"]
    assert verdict["status"] == "unsupported"
    assert verdict["note"] == "the run computed no sensitivities"


def test_check_of_sensitivities_out_of_budget_is_a_timeout_and_exits_6():
    d = _json(
        _bngsim(
            "check", CHAIN, *_WINDOW, "--sensitivity-params", "k1", "--checks", "sensitivity",
            "--budget", "1e-9", "--json",
        ),
        _cli.EXIT_CHECK_FAILED,
    )  # fmt: skip
    (verdict,) = d["check"]["checks"]
    assert (verdict["name"], verdict["status"]) == ("sensitivity", "timeout")


def test_check_skips_the_difference_quotient_of_a_parameter_at_zero():
    # A quotient moves a value by a share of itself: none at 0. A skip is
    # not a finding, so the check is ok.
    d = _json(
        _bngsim(
            "check", CHAIN, *_WINDOW, "--set", "k1=0", "--sensitivity-params", "k1",
            "--checks", "sensitivity,sensitivity_fd", "--json",
        )
    )  # fmt: skip
    by_name = {c["name"]: c["status"] for c in d["check"]["checks"]}
    assert by_name == {"sensitivity": "pass", "sensitivity_fd": "skip"}, d["check"]


def test_check_all_of_a_sensitivity_run_adds_both_sensitivity_checks():
    d = _json(
        _bngsim(
            "check", DECAY, *_WINDOW, "--sensitivity-params", "k", "--checks", "all", "--json"
        )
    )  # fmt: skip
    names = [c["name"] for c in d["check"]["checks"]]
    assert names[-len(_check.SENSITIVITY) :] == list(_check.SENSITIVITY)
    by_name = {c["name"]: c["status"] for c in d["check"]["checks"]}
    assert by_name["sensitivity"] == by_name["sensitivity_fd"] == "pass"


def test_check_exits_6_on_a_difference_quotient_diff():
    # A sensitivity 2% off the truth, and the states right: the difference
    # quotient catches it. The run is patched; the command line is not.
    script = (
        "import copy, numpy as np, bngsim\n"
        "S = bngsim.Simulator\n"
        "run = S.run\n"
        "def wrong(self, *a, **kw):\n"
        "    out = run(self, *a, **kw)\n"
        "    if len(out._sensitivity_params):\n"
        "        out = copy.copy(out)\n"
        "        out._sensitivities = np.asarray(out._sensitivities) * 1.02\n"
        "    return out\n"
        "S.run = wrong\n"
    )
    args = ["check", DECAY, *_WINDOW, "--sensitivity-params", "k", "--json"]
    ok = _json(_bngsim(*args, script=script))
    assert [c["name"] for c in ok["check"]["checks"]][-1] == "sensitivity"
    bad = _json(
        _bngsim(*args, "--checks", "sensitivity_fd", script=script), _cli.EXIT_CHECK_FAILED
    )
    assert bad["check"]["checks"][0]["status"] == "diff"


# ─── Help ────────────────────────────────────────────────────────────────────


def _option_help(command: str, option: str) -> str:
    _cli._build_parser()
    action = _cli._SUBPARSERS[command]._option_string_actions[option]
    return " ".join(str(action.help).split())


def test_check_help_lists_every_check_by_its_name():
    text = _option_help("check", "--checks")
    listed = text.split("separated by commas (", 1)[1].split(")", 1)[0].split(", ")
    assert listed == [*_check.DETERMINISTIC, *_check.STOCHASTIC, *_check.SENSITIVITY]
    defaults = (
        f"({', '.join(_check.DEFAULT_DETERMINISTIC)} for an ODE run, "
        f"{', '.join(_check.DEFAULT_STOCHASTIC)} for a stochastic one), and "
        f"{', '.join(_check.DEFAULT_SENSITIVITY)} too for a run with sensitivities"
    )
    assert defaults in text, text


def test_check_help_is_made_from_the_check_tuples(monkeypatch):
    # So a check added to bngsim._check is in --help without an edit here.
    monkeypatch.setattr(_check, "SENSITIVITY", (*_check.SENSITIVITY, "sensitivity_new"))
    monkeypatch.setattr(_check, "DEFAULT_SENSITIVITY", ("sensitivity_new",))
    text = _option_help("check", "--checks")
    assert "sensitivity_fd, sensitivity_new); default" in text
    assert "and sensitivity_new too for a run with sensitivities" in text


def test_the_command_line_page_lists_every_check_by_its_name():
    # The page is not made from the tuples, as --help is: a check added to
    # bngsim._check fails here until the page names it.
    from pathlib import Path

    page = (Path(__file__).parents[2] / "docs" / "reference" / "command-line.md").read_text(
        encoding="utf-8"
    )
    (bullet,) = [line for line in page.splitlines() if line.startswith("- **The checks:**")]
    for name in (*_check.DETERMINISTIC, *_check.STOCHASTIC, *_check.SENSITIVITY):
        assert f"`{name}`" in bullet, name


def test_run_and_check_help_name_the_simulator_keywords():
    for command in ("run", "check"):
        assert "sensitivity_params" in _option_help(command, "--sensitivity-params")
        assert "sensitivity_ic" in _option_help(command, "--sensitivity-ic")


# ─── The call behind the command ─────────────────────────────────────────────


def test_simulator_refuses_a_parameter_the_model_lacks_when_it_is_made():
    # It was found only once the run started, as a failed run (run.failed).
    with pytest.raises(ParameterError) as e:
        Simulator(Model.load(CHAIN), sensitivity_params=["k1", "k11"])
    assert e.value.code == "param.not_found"
    assert e.value.details["name"] == "k11"
    assert "k1" in e.value.details["close_matches"]
    assert str(e.value).startswith("sensitivity_params: Parameter 'k11' not found in model.")


def test_simulator_refuses_a_species_the_model_lacks_when_it_is_made():
    with pytest.raises(ModelError) as e:
        Simulator(Model.load(CHAIN), sensitivity_ic=["a"])
    assert e.value.code == "species.not_found"
    assert e.value.details["close_matches"] == ["A"]
    assert str(e.value).startswith("sensitivity_ic: Species 'a' not found in model.")


@pytest.mark.parametrize("keyword", ["sensitivity_params", "sensitivity_ic"])
def test_simulator_refuses_one_name_given_as_a_string(keyword):
    # A string is a sequence of names to Python: "k1" was read as "k" and "1".
    name = "k1" if keyword == "sensitivity_params" else "A"
    with pytest.raises(ArgumentTypeError) as e:
        Simulator(Model.load(CHAIN), **{keyword: name})
    assert e.value.code == "usage.wrong_type"
    assert f"{keyword}=[{name!r}]" in str(e.value)
    assert e.value.details["argument"] == keyword


@pytest.mark.parametrize(
    "keyword, name, other, error",
    [
        ("sensitivity_params", "A", "sensitivity_ic", ParameterError),
        ("sensitivity_ic", "k1", "sensitivity_params", ModelError),
    ],
)
def test_simulator_points_a_name_of_the_other_kind_at_the_other_keyword(
    keyword, name, other, error
):
    with pytest.raises(error) as e:
        Simulator(Model.load(CHAIN), **{keyword: [name]})
    assert f"{other}=[{name!r}] computes sensitivities to" in str(e.value)
    assert e.value.remedies == [{"kind": "pass_argument", "argument": other, "value": [name]}]


def test_simulator_refuses_a_name_that_is_not_a_string():
    with pytest.raises(ArgumentTypeError) as e:
        Simulator(Model.load(CHAIN), sensitivity_params=["k1", 2])
    assert e.value.details["argument"] == "sensitivity_params"


def test_simulator_takes_the_names_it_took_before():
    # Every name the model has, and none: no change for a right call.
    sim = Simulator(Model.load(CHAIN), sensitivity_params=("k1", "k2"), sensitivity_ic=["A"])
    assert sim.run(t_span=(0, 1), n_points=2).sensitivities.shape == (2, 3, 2)
    none = Simulator(Model.load(CHAIN), sensitivity_params=[]).run(t_span=(0, 1), n_points=2)
    assert none.sensitivities.shape[-1] == 0


def _comma_model(tmp_path) -> str:
    net = tmp_path / "comma.net"
    net.write_text(_COMMA_NET)
    return str(net)


@pytest.mark.parametrize(
    "keyword, name, says, target",
    [
        ("sensitivity_params", "X", "'X' is the species 'X()', not a parameter", "X()"),
        ("sensitivity_params", "Xtot",
         "'Xtot' is an observable of the one species 'X()', not a parameter", "X()"),
        ("sensitivity_ic", "Xtot",
         "'Xtot' is an observable of the one species 'X()', not a species", "X()"),
    ],
)  # fmt: skip
def test_simulator_points_at_the_model_s_own_spelling(tmp_path, keyword, name, says, target):
    with pytest.raises((ParameterError, ModelError)) as e:
        Simulator(Model.load(_comma_model(tmp_path)), **{keyword: [name]})
    assert says in str(e.value)
    assert e.value.remedies == [
        {"kind": "pass_argument", "argument": "sensitivity_ic", "value": [target]}
    ]
    # The remedy, followed, runs.
    sim = Simulator(Model.load(_comma_model(tmp_path)), sensitivity_ic=[target])
    assert sim.run(t_span=(0, 1), n_points=2).sensitivities_ic.shape == (2, 2, 1)


def test_simulator_gives_no_setter_advice_for_a_sensitivity_name(tmp_path):
    # set_param's message for an observable says how to set a concentration;
    # a sensitivity name needs to be told which names it takes.
    with pytest.raises(ParameterError) as e:
        Simulator(Model.load(_comma_model(tmp_path)), sensitivity_params=["Both"])
    assert str(e.value) == (
        "sensitivity_params: 'Both' is an observable, not a parameter; Model.param_names "
        "lists the names sensitivity_params takes."
    )
    assert e.value.remedies == []
    with pytest.raises(ModelError) as e:
        Simulator(Model.load(_comma_model(tmp_path)), sensitivity_ic=["X"])
    assert str(e.value) == "sensitivity_ic: Species 'X' not found in model. Close matches: 'X()'."
    assert "set_" not in str(e.value)


def test_the_command_points_at_the_model_s_own_spelling_and_quotes_it(tmp_path):
    net = _comma_model(tmp_path)
    error = _error(
        _bngsim("run", net, "--t-end", "1", "--sensitivity-params", "X", "--json"),
        _cli.EXIT_USAGE,
        "usage.invalid_argument",
    )
    says = "--sensitivity-ic 'X()' computes sensitivities to its initial amount"
    assert says in error["message"], error
    (remedy,) = error["remedies"]
    assert remedy == {"kind": "pass_argument", "argument": "--sensitivity-ic", "value": "X()"}
    # The remedy, followed, runs.
    _json(_bngsim("run", net, "--t-end", "1", remedy["argument"], remedy["value"], "--json"))


@pytest.mark.parametrize("names", [{"k1"}, frozenset({"k1"}), {"k1": 1.0}, 3])
def test_simulator_refuses_names_that_are_not_a_list(names):
    with pytest.raises(ArgumentTypeError) as e:
        Simulator(Model.load(CHAIN), sensitivity_params=names)
    assert e.value.details["argument"] == "sensitivity_params"


def test_simulator_takes_names_from_any_ordered_iterable():
    sim = Simulator(Model.load(CHAIN), sensitivity_params=(n for n in ["k2", "k1"]))
    r = sim.run(t_span=(0, 1), n_points=2)
    assert list(r.sensitivity_params) == ["k2", "k1"]
    assert r.sensitivities.shape == (2, 3, 2)
