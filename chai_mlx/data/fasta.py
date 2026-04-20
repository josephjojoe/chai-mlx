"""Lightweight chai-lab-format FASTA parser and validators.

chai-lab's own FASTA parser ``chai_lab.data.parsing.input_validation.read_inputs``
imports the heavy featurization stack (RDKit, torch, chai-lab's entity
tables). For tasks like CLI-side pre-flight validation or ESM pre-cache
population we only need the header metadata, not the entity types. This
module provides a minimal, dependency-free parser that matches chai-lab's
header grammar (``>kind|name=SHORT``) and a set of fast validators that
raise pretty errors up front instead of letting chai-lab fail deep in
``string_to_tensorcode`` or downstream featurizers.

Keeping this in ``chai_mlx.data`` means both ``scripts/inference.py`` and
``scripts/precompute_esm_mlx.py`` can import it without pulling chai-lab.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# chai-lab packs entity names into a fixed-length tensor when
# ``entity_name_as_subchain=True`` (see
# ``chai_lab.utils.tensor_utils.string_to_tensorcode`` and HANDOFF.md
# §5.4). That path is mandatory for our featurizer so constraint CSVs
# can reference chains by the same label the user wrote. Exceeding the
# limit fails loudly here rather than silently deep in the featurizer.
MAX_ENTITY_NAME_LENGTH: int = 4

# Kinds chai-lab's FASTA parser recognises. Mirrors
# ``chai_lab.data.parsing.fasta.Entity`` plus the glycan specialisation
# (see chai-lab/examples/covalent_bonds/*.fasta).
_KNOWN_KINDS: frozenset[str] = frozenset(
    {"protein", "ligand", "dna", "rna", "glycan"}
)


@dataclass(frozen=True)
class FastaRecord:
    """One parsed FASTA record.

    ``line_number`` is 1-indexed and points at the header line so error
    messages can lead users back to the offending line.
    """

    kind: str
    name: str
    sequence: str
    line_number: int


def parse_fasta_records(fasta_path: str | Path) -> list[FastaRecord]:
    """Return one :class:`FastaRecord` per ``>kind|name=...`` header.

    Records with a malformed header (missing ``|``, missing ``name=``,
    empty sequence) are still returned so callers can emit useful error
    messages against them — with ``name=""`` and/or ``sequence=""``. The
    downstream :func:`validate_fasta_or_raise` turns those into loud
    errors.
    """
    path = Path(fasta_path)
    text = path.read_text()

    records: list[FastaRecord] = []
    current_kind: str | None = None
    current_name: str | None = None
    current_seq: list[str] = []
    current_line: int | None = None

    def _flush() -> None:
        nonlocal current_kind, current_name, current_seq, current_line
        if current_line is None:
            return
        records.append(
            FastaRecord(
                kind=current_kind or "",
                name=current_name or "",
                sequence="".join(current_seq).strip(),
                line_number=current_line,
            )
        )
        current_kind = None
        current_name = None
        current_seq = []
        current_line = None

    for ln, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            _flush()
            header = line[1:]
            current_line = ln
            if "|" in header:
                kind, _, rest = header.partition("|")
                current_kind = kind.strip().lower()
                current_name = ""
                for kv in rest.split("|"):
                    k, _, v = kv.partition("=")
                    if k.strip() == "name":
                        current_name = v.strip()
                        break
            else:
                current_kind = header.strip().lower()
                current_name = ""
        else:
            current_seq.append(line)

    _flush()
    return records


@dataclass(frozen=True)
class FastaValidationIssue:
    """One problem found in a FASTA file."""

    line_number: int
    message: str


def find_fasta_issues(records: list[FastaRecord]) -> list[FastaValidationIssue]:
    """Return every problem with *records* (empty list = FASTA is good).

    Checks performed (all cheap, no chai-lab imports):

    * Header must be ``>kind|name=SHORT`` with a non-empty name.
    * ``kind`` must be a known entity kind.
    * ``name`` must be at most :data:`MAX_ENTITY_NAME_LENGTH` characters
      (chai-lab's ``entity_name_as_subchain`` constraint).
    * Names must be unique across all records (chai-lab errors on
      duplicates anyway, but we want to surface the offenders before the
      ~5 s model load).
    * Sequence must be non-empty.
    """
    issues: list[FastaValidationIssue] = []
    seen_names: dict[str, int] = {}
    for rec in records:
        if not rec.kind:
            issues.append(
                FastaValidationIssue(
                    line_number=rec.line_number,
                    message="header is missing a kind; expected '>kind|name=...' "
                            "where kind is one of "
                            + ", ".join(sorted(_KNOWN_KINDS)),
                )
            )
            continue
        if rec.kind not in _KNOWN_KINDS:
            issues.append(
                FastaValidationIssue(
                    line_number=rec.line_number,
                    message=f"unknown entity kind {rec.kind!r}; expected one of "
                            + ", ".join(sorted(_KNOWN_KINDS)),
                )
            )
            continue
        if not rec.name:
            issues.append(
                FastaValidationIssue(
                    line_number=rec.line_number,
                    message=f"record of kind {rec.kind!r} is missing 'name=SHORT' in "
                            "its header; add '|name=XXX' with a <=4 character label.",
                )
            )
            continue
        if len(rec.name) > MAX_ENTITY_NAME_LENGTH:
            issues.append(
                FastaValidationIssue(
                    line_number=rec.line_number,
                    message=(
                        f"entity name {rec.name!r} is {len(rec.name)} characters; "
                        f"chai-lab packs names into a fixed-length tensor when "
                        f"entity_name_as_subchain=True and cannot exceed "
                        f"{MAX_ENTITY_NAME_LENGTH} characters (see HANDOFF.md §5.4). "
                        f"Shorten it to <={MAX_ENTITY_NAME_LENGTH} chars."
                    ),
                )
            )
            continue
        if rec.name in seen_names:
            issues.append(
                FastaValidationIssue(
                    line_number=rec.line_number,
                    message=(
                        f"entity name {rec.name!r} is used more than once "
                        f"(first seen at line {seen_names[rec.name]}); "
                        "each entity must have a unique name."
                    ),
                )
            )
            continue
        seen_names[rec.name] = rec.line_number
        if not rec.sequence:
            issues.append(
                FastaValidationIssue(
                    line_number=rec.line_number,
                    message=f"record {rec.name!r} has no sequence; add a non-empty "
                            "sequence line below the header.",
                )
            )
    return issues


def _ligand_smiles_issues(records: list[FastaRecord]) -> list[FastaValidationIssue]:
    """Validate every ligand record's SMILES via RDKit.

    RDKit is a transitive dep of chai-lab; when the ``[featurize]``
    extra is installed it's always present. When it isn't, we skip the
    check silently (the featurizer will fail later with its own
    RDKit-parses-your-ligand error, which is still clearer than the
    alternative).
    """
    issues: list[FastaValidationIssue] = []
    try:
        from rdkit import Chem
    except ImportError:
        return issues
    for rec in records:
        if rec.kind != "ligand" or not rec.sequence:
            continue
        mol = Chem.MolFromSmiles(rec.sequence)
        if mol is None:
            issues.append(
                FastaValidationIssue(
                    line_number=rec.line_number,
                    message=(
                        f"ligand {rec.name!r} has SMILES that RDKit could not parse: "
                        f"{rec.sequence!r}. Double-check the SMILES string."
                    ),
                )
            )
    return issues


def _modified_residue_issues(records: list[FastaRecord]) -> list[FastaValidationIssue]:
    """Flag inline modified-residue / PTM tokens as unsupported here.

    Chai-lab's FASTA parser accepts modified residues written as
    ``[FOO]`` or ``(FOO)`` inline in polymer sequences (e.g.
    ``APNGL[HIP]TRP`` for a histidinol-phosphate substitution; see
    ``chai_lab.data.parsing.input_validation.constituents_of_modified_fasta``
    and ``chai_lab.data.parsing.structure.sequence``). Chai-mlx
    inherits that machinery but the path is not end-to-end validated
    in this repo (HANDOFF.md §8.1). Catching the bracketed token at
    validation time gives users a loud "yes, you can try this, but
    it's untested here" pointer rather than a cryptic failure later.

    The check scans for brackets ``[…]`` or parentheses ``(…)`` in
    protein / DNA / RNA polymer sequences; ligand SMILES (``C(=O)``
    style parens are legitimate SMILES tokens) and glycan strings are
    left alone. Users can bypass via
    ``CHAI_MLX_ALLOW_MODIFIED_RESIDUES=1``.
    """
    issues: list[FastaValidationIssue] = []
    polymer_kinds = {"protein", "dna", "rna"}
    for rec in records:
        if rec.kind not in polymer_kinds or not rec.sequence:
            continue
        # Bracketed ``[FOO]`` (or legacy ``(FOO)``) tokens are chai-1's
        # modified-residue syntax on polymer sequences. A plain
        # canonical FASTA (``MKWV...``) never contains these symbols,
        # so the presence of either is a reliable signal.
        if "[" in rec.sequence or "(" in rec.sequence:
            issues.append(
                FastaValidationIssue(
                    line_number=rec.line_number,
                    message=(
                        f"record {rec.name!r} contains an inline "
                        "modified-residue token ('[FOO]' or '(FOO)'); "
                        "chai-mlx inherits chai-lab's support for these "
                        "but the path is NOT validated end-to-end here "
                        "(see HANDOFF.md §8.1 'Modified residues / "
                        "PTMs'). If you want to try it anyway, re-run "
                        "with CHAI_MLX_ALLOW_MODIFIED_RESIDUES=1 set in "
                        "your environment to bypass this check."
                    ),
                )
            )
    return issues


def _glycan_issues(records: list[FastaRecord]) -> list[FastaValidationIssue]:
    """Validate every glycan record against chai-lab's glycan grammar.

    Chai-lab parses strings like ``NAG(4-1 NAG)`` via
    ``chai_lab.data.parsing.glycans._glycan_string_to_sugars_and_bonds``.
    When chai-lab is installed, run that pre-flight so glycan typos
    surface here instead of mid-featurization. When chai-lab is not
    installed, skip.
    """
    issues: list[FastaValidationIssue] = []
    try:
        from chai_lab.data.parsing.glycans import _glycan_string_to_sugars_and_bonds
    except ImportError:
        return issues
    for rec in records:
        if rec.kind != "glycan" or not rec.sequence:
            continue
        try:
            _glycan_string_to_sugars_and_bonds(rec.sequence)
        except Exception as exc:  # chai-lab raises ValueError on bad strings
            issues.append(
                FastaValidationIssue(
                    line_number=rec.line_number,
                    message=(
                        f"glycan {rec.name!r} failed chai-lab's glycan parser: "
                        f"{type(exc).__name__}: {exc}. Expected syntax looks like "
                        "'NAG(4-1 NAG)'; see chai-lab/examples/covalent_bonds "
                        "for worked examples."
                    ),
                )
            )
    return issues


def validate_fasta_or_raise(fasta_path: str | Path) -> list[FastaRecord]:
    """Parse and validate a FASTA. Raises :class:`SystemExit` on any issue.

    Successful validation returns the parsed records for reuse by
    callers that want to avoid re-parsing.

    The error message lists every offending record with its line number
    and a one-sentence fix, so users can edit once and re-run. Checks
    are layered cheapest-first (structural / name rules in
    :func:`find_fasta_issues`, then SMILES / glycan parse pre-flights
    that only run when optional deps are installed, then a modified-
    residue fast-fail that users can bypass via
    ``CHAI_MLX_ALLOW_MODIFIED_RESIDUES=1``).
    """
    import os

    records = parse_fasta_records(fasta_path)
    if not records:
        raise SystemExit(
            f"error: {fasta_path} contains no FASTA records. "
            "Expected one or more '>kind|name=SHORT' blocks."
        )
    issues = find_fasta_issues(records)
    issues.extend(_ligand_smiles_issues(records))
    issues.extend(_glycan_issues(records))
    if os.environ.get("CHAI_MLX_ALLOW_MODIFIED_RESIDUES", "").lower() not in (
        "1", "true", "yes"
    ):
        issues.extend(_modified_residue_issues(records))
    if issues:
        header = f"error: {fasta_path} has {len(issues)} problem(s):"
        lines = [header] + [f"  line {i.line_number}: {i.message}" for i in issues]
        raise SystemExit("\n".join(lines))
    return records
