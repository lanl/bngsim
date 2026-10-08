"""bngsim._net_reader — a .net file as a dict, and a model built from one.

``parse_net_file`` returns the C++ ``.net`` loader's own reading of a file as a
dictionary of model components; ``build_model_from_parsed`` feeds such a
dictionary to ``ModelBuilder`` exactly as the loader does. So a file can be
loaded, inspected or modified, and built — or handed to another engine.

There is one reading of the ``.net`` format, ``src/net_file_loader.cpp``'s.
Until issue #803 this module parsed the text itself, a second reading that had
to be moved in step with the loader's by hand and was not every time (#554,
#597, #600, #606). Now ``Model.from_net``, ``parse_net_file`` and every backend
read a file the same way, and a format fix lands once.

Example
-------
>>> from bngsim import build_model_from_parsed, parse_net_file
>>> parsed = parse_net_file("model.net")
>>> parsed["parameters"][0]
('kf', 0.5, '0.5', False)
>>> model = build_model_from_parsed(parsed)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any


def parse_net_file(path: str | Path) -> dict[str, Any]:
    """Read a BNG .net file into a structured dictionary.

    The reading is the C++ loader's (``Model.from_net``'s), so it needs bngsim's
    compiled extension: the records it hands ModelBuilder, with the values the
    model it builds from them holds. The build runs here too, so a file
    ``Model.from_net`` refuses is refused here, and the ``Sat``/``Hill`` rewrite
    and the lifting of expression-valued initial concentrations happen as they do
    there.

    Parameters
    ----------
    path : str or Path
        Path to the .net file.

    Returns
    -------
    dict
        Structured contents of the ``.net`` file, with keys::

            parameters   : list of (name, value, expression, is_expression)
            species      : list of (name, init_conc, is_fixed)
            species_ic_params : list of (species_idx0, param_name) for each
                           species whose IC column names a parameter
            observables  : list of (name, entries), entries = [(sp_idx0, factor), ...]
            functions    : list of (name, expression)
            reactions    : list of dict with keys reactants, products (0-based
                           species indices), type, rate_law, legacy_constants,
                           stat_factor
            net_file_dir : the file's own directory, absolute, which is what a
                           relative ``tfun('...')`` path resolves against
            unit_conversions : list of (reaction, expression), one for each
                           ``unit_conversion=`` comment BNG2.pl left on a
                           reaction line of a compartmental model. The
                           reaction's rate holds that factor as a number, so
                           the parameters the expression reads are refused for
                           writing in the model built from this dictionary
                           (``Model.frozen_params``, issue #711)

        Values are evaluated: a parameter's is the number ``Model.from_net``
        puts in its slot, and a species whose initial concentration names a
        parameter carries that parameter's value. An expression-valued initial
        concentration is a synthetic ``_InitialConc<N>`` parameter, as BNG2.pl
        writes one, and the species names it.

        A reaction's ``type`` is one of:

        ``"elementary"``
            ``rate_law`` is a parameter name; ``stat_factor`` carries a
            ``<number>*`` prefix on it.
        ``"functional"``
            ``rate_law`` names a function in the ``functions`` block. A
            deprecated ``Sat`` or ``Hill`` rate law comes back as one of these,
            rewritten into an explicit function (and a single-species observable
            for each reactant it reads), with the loader's ``UserWarning``.
        ``"mm"``
            Michaelis-Menten (``MM kcat Km``); ``rate_law`` is
            ``"<kcat>,<Km>"``, the form ``ModelBuilder`` takes.

        ``legacy_constants`` holds an ``"mm"`` reaction's ``[kcat, Km]`` and is
        empty otherwise: a ``Sat`` or ``Hill`` rate law, whose operands it used to
        carry, now comes back rewritten (issue #803).

    Raises
    ------
    FileNotFoundError, IsADirectoryError
        If ``path`` is not a file.
    ValueError
        If ``Model.from_net`` would refuse the file. The reading builds the model,
        so that includes a ``tfun('...')`` data file ``Model.from_net`` would not
        find beside the ``.net``: put it there to read the file, then set
        ``net_file_dir`` in the dict to build against another location.
    ImportError
        If bngsim's compiled extension is not available.
    """
    try:
        from bngsim._bngsim_core import net_file_structure
    except ImportError as exc:
        raise ImportError(
            "parse_net_file reads a .net file with bngsim's compiled .net loader, and "
            f"bngsim._bngsim_core is not available ({exc}). Since issue #803 there is no "
            "pure-Python reading of the format to fall back on."
        ) from exc

    path = Path(path)
    # The errors a missing file raised when this module read the text itself, and
    # Model.from_net's for a missing one.
    if not path.exists():
        raise FileNotFoundError(f"Net file not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"{path} is a directory, not a .net file")
    parsed = net_file_structure(os.fspath(path))
    for message in parsed.pop("load_warnings"):
        warnings.warn(message, UserWarning, stacklevel=2)
    # Values, initial concentrations and each reaction's type are the built
    # model's; the binding returns the dict in this shape, so there is nothing
    # left to reshape here. Absolute, so a build after a change of working
    # directory still finds a relative tfun file.
    parsed["net_file_dir"] = os.path.abspath(parsed["net_file_dir"])
    from bngsim._model import _unit_conversions_in

    parsed["unit_conversions"] = _unit_conversions_in(path)
    return parsed


def build_model_from_parsed(parsed: dict[str, Any]):
    """Build a NetworkModel from parsed .net data via ModelBuilder.

    It makes the calls ``NetFileLoader::load`` makes, in the same order, so the
    dictionary ``parse_net_file`` returns builds the model ``Model.from_net``
    loads, and a modified dictionary builds the modified model. Four things to
    know when modifying one:

    * a species listed in ``species_ic_params`` takes its initial concentration
      from that parameter, so change the parameter, or drop the entry, rather than
      the number in ``species``;
    * a parameter with ``is_expression`` true is evaluated from its expression,
      so change the expression, or set ``is_expression`` false with a value;
    * a reaction's rate law is read by what it names: a function in
      ``functions`` makes it functional, anything else elementary (``"mm"``
      aside), whatever its ``type`` says;
    * the parameters a ``unit_conversions`` entry reads are refused for writing
      in the model built (issue #711): the rates hold that factor as a number.
      Drop an entry only with the number it stands for taken out of the rate.
      An entry names parameters by the names in ``parameters``: rename one in
      both, or the entry names nothing and the renamed parameter is written
      with no rate following it. A dictionary with no ``unit_conversions`` key
      builds a model with nothing refused.

    Parameters
    ----------
    parsed : dict
        Output of ``parse_net_file()``, or a dictionary of the same shape.

    Returns
    -------
    bngsim.Model
        The constructed model.

    Raises
    ------
    ValueError
        If a reaction's ``rate_law`` is empty, or its ``type`` is ``"legacy"``
        (a ``Sat``/``Hill`` token, which ``parse_net_file`` no longer returns —
        write the rate law as a function), or a ``species_ic_params`` entry names
        a parameter the dictionary does not declare or a species it does not have.
    RuntimeError
        If ``ModelBuilder`` refuses the model, as it refuses the same records
        from the loader: an elementary ``rate_law`` that names neither a
        parameter nor a function, for instance, or a parameter expression that
        reads an observable or a function (issue #844).
    """
    from bngsim._bngsim_core import (
        ModelBuilder,
        net_function_tables,
        net_refuse_parameters_that_read_state,
    )
    from bngsim._model import Model, _ar_report_map_from_net, _unit_conversion_folds

    # The loader's own refusal, which phase 2 makes before its first call: a
    # parameter that reads an observable or a function would build as 0 (#844).
    net_refuse_parameters_that_read_state(
        [tuple(p) for p in parsed["parameters"]],
        [tuple(f) for f in parsed["functions"]],
        [name for name, _ in parsed["observables"]],
    )

    builder = ModelBuilder()
    # Where a relative `tfun('...')` path resolves from, set before the tables
    # below are registered against it.
    builder.set_net_file_dir(parsed.get("net_file_dir", ""))

    for name, value, expr, is_expr in parsed["parameters"]:
        builder.add_parameter(name, value, expr, is_expr)

    for name, init_conc, is_fixed in parsed["species"]:
        builder.add_species(name, init_conc, is_fixed)

    # A species IC written as a parameter name is handed to the builder as a
    # reference rather than as a number, so build() re-resolves it from the
    # compiled parameter and records the (species, parameter) pair the
    # forward-sensitivity seeding reads (issue #554).
    declared = {name for name, _, _, _ in parsed["parameters"]}
    for species_idx0, param_name in parsed.get("species_ic_params", ()):
        # ModelBuilder refuses both of these as well (issue #863); checking here
        # first names the dict field the caller got wrong.
        if param_name not in declared:
            raise ValueError(
                f"species_ic_params names parameter {param_name!r}, which the parameters "
                "do not declare"
            )
        if not 0 <= species_idx0 < len(parsed["species"]):
            raise ValueError(f"species_ic_params names species index {species_idx0}")
        builder.add_species_param_ref(species_idx0, param_name)

    # A function whose expression calls `tfun(...)` needs its table registered
    # before build() compiles the expression; `net_function_tables` is the
    # loader's own reading of those calls (issue #597).
    for name, expression in parsed["functions"]:
        analyzed = net_function_tables(name, expression)
        builder.add_function(name, analyzed["expression"])
        for table in analyzed["tables"]:
            if table["is_inline"]:
                builder.add_inline_table_function_spec(
                    table["name"],
                    table["xs"],
                    table["ys"],
                    table["index_name"],
                    table["method"],
                )
            else:
                builder.add_table_function_spec(
                    table["name"],
                    table["filepath"],
                    table["index_name"],
                    table["method"],
                    table["header_name"],
                )

    for observable_name, entries in parsed["observables"]:
        builder.add_observable(observable_name, entries)

    for i, rxn in enumerate(parsed["reactions"]):
        rtype = rxn["type"]
        rate_law = rxn["rate_law"]
        if rtype == "legacy":
            raise ValueError(
                f"reaction {i + 1} has type 'legacy' ({rate_law!r}), which "
                "build_model_from_parsed does not build. parse_net_file rewrites a "
                "deprecated Sat or Hill rate law into an explicit function; in a "
                "dictionary built by hand, write the rate law as a function and give "
                "the reaction type 'functional'."
            )
        if not rate_law.strip():
            # ModelBuilder's validation passes an empty name over, so refuse it here.
            raise ValueError(f"reaction {i + 1} has an empty rate_law")
        # By what the rate law names, not by the label, and by ModelBuilder's
        # rule rather than a copy of it: an elementary rate that names a
        # function is resolved to functional in build(), exactly as the loader
        # relies on. Passing "functional" through would build a reaction that
        # names a parameter as one that never fires.
        if rtype != "mm":
            rtype = "elementary"
        builder.add_reaction(
            rxn["reactants"], rxn["products"], rtype, rate_law, rxn["stat_factor"]
        )

    core = builder.build()
    model = Model(_core=core)
    # The output columns of species an assignment rule defines (sbml_to_net's
    # networks), rebuilt as Model.from_net rebuilds them.
    model._ar_report_map = _ar_report_map_from_net(core)
    model._frozen_params = _unit_conversion_folds(parsed.get("unit_conversions"), core)
    return model
