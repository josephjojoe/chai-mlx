"""Shared Modal infrastructure for chai-mlx CUDA comparison harnesses.

This module defines:

* ``app`` — the shared Modal app (name: ``chai-mlx-cuda``).
* ``image`` — the container image with ``chai_lab``, ``torch``, ``gemmi``,
  ``biopython``, ``numpy`` pinned to match the local environment semantics.
* ``chai_model_volume`` / ``chai_outputs_volume`` — persistent Modal Volumes
  for weights and run outputs (mirrors Modal's official chai-1 example pattern
  so we don't re-download the ~10 GB of weights on every run).
* ``download_inference_dependencies`` — Modal Function that populates the
  weights volume from Chai Discovery's CDN. Runs once; subsequent harness runs
  reuse the cached weights.
* ``Target`` / ``DEFAULT_TARGETS`` — the target slate driving every harness.
  Each entry is a possibly-multi-entity FASTA (protein + ligand + nucleic-acid
  records), optionally bundled with a ``constraint_resource`` pointer and
  tagged with ``kinds`` (``"monomer"``, ``"multimer"``, ``"ligand"``,
  ``"dna"``, ``"rna"``, ``"long"``, ``"esm"``, ``"constraints"``) so callers
  can filter to the subsets they care about.

All of the per-harness files in this directory (``run_reference.py``,
``run_intermediates.py``, ``bench_throughput.py``) import from here so that
the image, weights, and outputs storage are shared across harnesses.

Usage
-----

Prime the weights cache (only needed once per Modal workspace)::

    modal run cuda_harness.modal_common::download_inference_dependencies

Then any harness in this directory can run via ``modal run``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import modal

MINUTES = 60

app = modal.App(name="chai-mlx-cuda")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # uv needs git on PATH to install a VCS spec; debian_slim doesn't ship it.
    # kalign>=3.3 is required by chai-lab whenever ``use_templates_server=True``
    # (see chai_lab/tools/kalign.py) because the online-templates path
    # aligns each hit against the query sequence. debian_slim ships
    # kalign>=3.3 in its default repos.
    .apt_install("git", "kalign")
    .uv_pip_install(
        # Install chai-lab from the same upstream commit our local checkout
        # is pinned at (post-v0.6.1 main, with PR #360's `_component_moved_to`
        # caching helper that our intermediates harness depends on, and
        # PR #415's gemmi-0.7 support). Modal's canonical example pins 0.5.0
        # but that predates both PRs and the published 0.6.1 wheel does not
        # include #360 either.
        "chai_lab @ git+https://github.com/chaidiscovery/chai-lab@61036259c98222160963cb780750e354876ce485",
        "huggingface-hub==0.36.0",
        "numpy>=1.26,<2",
    )
    .uv_pip_install(
        "torch==2.7.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
)

chai_model_volume = modal.Volume.from_name(
    "chai-mlx-weights",
    create_if_missing=True,
)
MODELS_DIR = Path("/models/chai1")

chai_outputs_volume = modal.Volume.from_name(
    "chai-mlx-cuda-outputs",
    create_if_missing=True,
)
OUTPUTS_DIR = Path("/outputs")

image = image.env(
    {
        "CHAI_DOWNLOADS_DIR": str(MODELS_DIR),
        "HF_XET_HIGH_PERFORMANCE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        # chai-lab's TorchScript trunk is memory-hungry in the masked_fill /
        # einsum paths around the MSA pair-weighted averaging op. Switching
        # to expandable segments keeps fragmentation from tipping us over
        # the 80 GB H100 limit on model_size=256 crops.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
)


INFERENCE_DEPENDENCIES = (
    "conformers_v1.apkl",
    "models_v2/trunk.pt",
    "models_v2/token_embedder.pt",
    "models_v2/feature_embedding.pt",
    "models_v2/diffusion_module.pt",
    "models_v2/confidence_head.pt",
    "models_v2/bond_loss_input_proj.pt",
)


@app.function(
    volumes={MODELS_DIR: chai_model_volume},
    image=image,
    timeout=30 * MINUTES,
)
async def download_inference_dependencies(force: bool = False) -> list[str]:
    """Populate the shared weights volume from Chai's CDN.

    Mirrors Modal's own example so we stay compatible with their caching.
    Runs concurrently to minimize wall clock.  Returns the list of files it
    touched so the caller can log what happened.
    """
    import asyncio

    import aiohttp

    base_url = "https://chaiassets.com/chai1-inference-depencencies/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/58.0.3029.110 Safari/537.3"
        )
    }

    downloaded: list[str] = []

    async def _download(session: aiohttp.ClientSession, dep: str) -> None:
        local_path = MODELS_DIR / dep
        if not force and local_path.exists():
            return
        local_path.parent.mkdir(parents=True, exist_ok=True)
        url = base_url + dep
        print(f"[download] {dep}")
        async with session.get(url) as response:
            response.raise_for_status()
            with open(local_path, "wb") as fh:
                while chunk := await response.content.read(1 << 20):
                    fh.write(chunk)
        downloaded.append(dep)

    async with aiohttp.ClientSession(headers=headers) as session:
        await asyncio.gather(*(_download(session, dep) for dep in INFERENCE_DEPENDENCIES))

    await chai_model_volume.commit.aio()
    return downloaded


# ---------------------------------------------------------------------------
# Target slate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FastaRecord:
    """One FASTA record inside a :class:`Target`.

    ``kind`` is one of ``"protein"``, ``"ligand"``, ``"dna"``, ``"rna"``,
    ``"glycan"`` (matching chai-lab's FASTA header parser). ``name`` is
    the per-record entity name; it doubles as the ``chainA`` / ``chainB``
    key in constraint CSVs. ``sequence`` is an amino-acid / nucleotide
    string for polymers, a SMILES for ligands, or a chai-lab glycan
    specification for glycans.
    """

    kind: str
    name: str
    sequence: str


@dataclass(frozen=True)
class Target:
    """One harness target: a name, a set of FASTA records, and metadata.

    ``records`` is a tuple of :class:`FastaRecord` values. A single
    target can carry any combination of entity kinds (protein, ligand,
    DNA, ...).

    ``kinds`` is a frozenset of coarse tags harnesses use for filtering
    (e.g. ``--target-kinds multimer,ligand``).  A single target can
    carry multiple kinds.

    ``constraint_resource`` names a CSV shipped under
    ``cuda_harness/constraints/``; when present, harnesses that support
    constraints will copy it next to the FASTA and pass
    ``constraint_path=...`` to chai-lab.
    """

    name: str
    records: tuple[FastaRecord, ...]
    description: str = ""
    kinds: frozenset[str] = field(default_factory=frozenset)
    constraint_resource: str | None = None

    def to_fasta(self) -> str:
        return "".join(
            f">{r.kind}|name={r.name}\n{r.sequence}\n" for r in self.records
        )

    @property
    def n_protein_residues(self) -> int:
        return sum(len(r.sequence) for r in self.records if r.kind == "protein")

    @property
    def n_nucleic_residues(self) -> int:
        return sum(
            len(r.sequence) for r in self.records if r.kind in ("dna", "rna")
        )

    @property
    def polymer_records(self) -> tuple[FastaRecord, ...]:
        return tuple(
            r for r in self.records if r.kind in ("protein", "dna", "rna")
        )

    @property
    def is_multimer(self) -> bool:
        return len(self.polymer_records) > 1


def _target(
    name: str,
    records: tuple[tuple[str, str, str], ...],
    *,
    description: str,
    kinds: frozenset[str],
    constraint_resource: str | None = None,
) -> Target:
    return Target(
        name=name,
        records=tuple(FastaRecord(kind, label, seq) for kind, label, seq in records),
        description=description,
        kinds=kinds,
        constraint_resource=constraint_resource,
    )


DEFAULT_TARGETS: dict[str, Target] = {
    # -- small-monomer baseline --------------------------------------------
    "1L2Y": _target(
        "1L2Y",
        (("protein", "1L2Y", "NLYIQWLKDGGPSSGRPPPS"),),
        description="20-residue miniprotein; local parity baseline",
        kinds=frozenset({"monomer"}),
    ),
    "1VII": _target(
        "1VII",
        (("protein", "1VII", "LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"),),
        description="35-residue villin headpiece",
        kinds=frozenset({"monomer"}),
    ),
    "1CRN": _target(
        "1CRN",
        (
            (
                "protein",
                "1CRN",
                "TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN",
            ),
        ),
        description="46-residue crambin",
        kinds=frozenset({"monomer"}),
    ),
    "1UBQ": _target(
        "1UBQ",
        (
            (
                "protein",
                "1UBQ",
                "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
            ),
        ),
        description="76-residue ubiquitin",
        kinds=frozenset({"monomer"}),
    ),

    # -- multimer (two-protein heterodimer) --------------------------------
    # Barnase (110 aa) + barstar (89 aa). Canonical tight heterodimer with
    # a 2.0 Å crystal structure; small enough to fit on an 80 GB H100 with
    # use_esm_embeddings=True.
    #
    # Entity names are limited to 4 characters because chai-lab packs
    # subchain IDs into a fixed-length tensor when
    # entity_name_as_subchain=True, which we rely on so constraint CSVs
    # can address chains by the same label the user wrote in the FASTA.
    "1BRS": _target(
        "1BRS",
        (
            (
                "protein",
                "BARN",
                "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREG"
                "KLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR",
            ),
            (
                "protein",
                "BARS",
                "KKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDCLTGWVEYPLVLEWRQFEQSK"
                "QLTENGAESVLQVFREAKAEGCDITIILS",
            ),
        ),
        description="Barnase-barstar heterodimer (PDB 1BRS, ~199 residues)",
        kinds=frozenset({"multimer"}),
    ),

    # -- ligand complex ----------------------------------------------------
    # FKBP-12 (107 residues) + FK506 (tacrolimus). Classic drug-target
    # complex; exercises EntityType.LIGAND, the atom-wise encoder path,
    # and ligand-polymer clash bookkeeping.
    "1FKB": _target(
        "1FKB",
        (
            (
                "protein",
                "FKBP",
                "GVQVETISPGDGRTFPKRGQTCVVHYTGMLEDGKKFDSSRDRNKPFKFMLGKQEVIRGWE"
                "EGVAQMSVGQRAKLTISPDYAYGATGHPGIIPPHATLVFDVELLKLE",
            ),
            (
                "ligand",
                "FK",
                "CC=CC1CC(C)CC2(O)OC3(C(CC(C3O)C)C)OC(=O)C(N4CCCCC4C(=O)O2)"
                "CC(=O)C(C)C=C(C)C(O)C(OC)CC(=O)C(C)CC(=C)C(C)C(O)C(C)C1OC",
            ),
        ),
        description="FKBP-12 + FK506 (PDB 1FKB)",
        kinds=frozenset({"ligand"}),
    ),

    # -- >200 residue monomer ---------------------------------------------
    # Triosephosphate isomerase, 248 residues. TIM barrel fold; a
    # canonical stress test for a longer monomer target.
    "7TIM": _target(
        "7TIM",
        (
            (
                "protein",
                "7TIM",
                "ASFVRKFFVGGNWKMNGDKKSLGELIHTLDGAKLSADTEVVCGAPSIYLDFARQKLDAKI"
                "GVAAQNCYKVPKGAFTGEISPAMIKDIGAAWVILGHSERRHVFGESDELIGQKVAHALAE"
                "GLGVIACIGEKLDEREAGITEKVVFQETKAIADNVKDWSKVVLAYEPVWAIGTGKTATPQ"
                "QAQEVHEKLRGWLKTHVSDAVAVQSRIIYGGSVTGGNCKELASQHDVDGFLVGGASLKPE"
                "FVDIINAKH",
            ),
        ),
        description="Triosephosphate isomerase, 248-residue TIM barrel",
        kinds=frozenset({"long"}),
    ),

    # -- nucleic acid (DNA duplex) ----------------------------------------
    # Dickerson dodecamer, 2 × 12 bp. Tests EntityType.DNA end-to-end.
    "1BNA": _target(
        "1BNA",
        (
            ("dna", "DNA1", "CGCGAATTCGCG"),
            ("dna", "DNA2", "CGCGAATTCGCG"),
        ),
        description="Dickerson DNA dodecamer (PDB 1BNA)",
        kinds=frozenset({"dna", "multimer"}),
    ),

    # -- ESM-on-MLX evaluation --------------------------------------------
    # Same sequence as 1UBQ; separate entry so sweeps can tell the two
    # configurations apart (MLX-ESM vs zero-ESM).  The short ``UESM``
    # entity name keeps us inside chai-lab's 4-char subchain-ID budget.
    "1UBQ_ESM": _target(
        "1UBQ_ESM",
        (
            (
                "protein",
                "UESM",
                "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
            ),
        ),
        description="76-residue ubiquitin; MLX-ESM vs CUDA-ESM comparison",
        kinds=frozenset({"esm"}),
    ),

    # -- constraints ------------------------------------------------------
    # Crambin + a small bridging ligand so the covalent-bond constraint
    # has a legal partner. Contact + pocket restraints act on crambin
    # alone; the covalent bond ties a Cys sulphur on crambin to the
    # ligand.
    "1CRN_CONSTR": _target(
        "1CRN_CONSTR",
        (
            (
                "protein",
                "1CRN",
                "TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN",
            ),
            # Methanethiol: smallest ligand with a heavy atom that can
            # form a plausible covalent bond to a Cys sulphur.
            ("ligand", "LIG1", "CS"),
        ),
        description="Crambin + bridging ligand + synthetic restraints",
        kinds=frozenset({"constraints", "ligand"}),
        constraint_resource="1CRN_all_three.csv",
    ),

    # -- RNA --------------------------------------------------------------
    # 2KOC: 14-nucleotide RNA hairpin (NMR ensemble). The sequence uses
    # only the canonical ACGU alphabet -- no post-transcriptionally
    # modified nucleotides -- so comparing our canonical-base prediction
    # against the experimental structure is a clean test.
    #
    # Why not 1EHZ (yeast tRNA-Phe): tRNA structures are heavily
    # modified (D, Psi, m2G, T, etc.) and those modifications shape the
    # TPsiC / D-loops. chai-lab's FASTA front-end only accepts the 4-
    # letter canonical alphabet for RNA (see
    # residue_constants.residue_types_with_nucleotides), so a tRNA
    # prediction is lying to the model about the sequence. 2KOC has no
    # such ambiguity.
    "2KOC": _target(
        "2KOC",
        (
            (
                "rna",
                "2KOC",
                "GGCACUUCGGUGCC",
            ),
        ),
        description="14-nt RNA hairpin (PDB 2KOC, NMR)",
        kinds=frozenset({"rna"}),
    ),

    # -- Glycan -----------------------------------------------------------
    # UBQG: synthetic small glycan target. Ubiquitin (76 aa, same
    # sequence as the 1UBQ monomer in this slate) with a single NAG
    # attached to Asn-25 via an N-linked covalent bond. The protein
    # chain is 76 aa and the whole complex fits in the 256-token crop,
    # so this is cheap to run on a 16 GB Mac.
    #
    # This is a pure compute target for the glycan + covalent-bond
    # code path. The full-scale 1AC5 glycan archetype (483-aa
    # carboxypeptidase-Y + two N-linked glycans, 768-token crop) is not
    # kept in the default slate. UBQG keeps the glycan + covalent-bond
    # feature path cheap enough to run on a 16 GB Mac; users with larger
    # machines can add 1AC5 back locally by appending a target definition
    # in their own code.
    "UBQG": _target(
        "UBQG",
        (
            (
                "protein",
                "UBQG",
                "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
            ),
            ("glycan", "NAG1", "NAG"),
        ),
        description="Ubiquitin + single NAG glycan (synthetic small target for the glycan code path)",
        kinds=frozenset({"glycan", "ligand", "constraints"}),
        constraint_resource="UBQG_glycan.csv",
    ),

    # -- Tier-2 two-sided parity smokes -----------------------------------
    # The three targets below (PTM, multi-ligand, large ligand) each
    # exercise an entity-dispatch code path that we want available in the
    # standard MLX↔CUDA comparison slate.
    #
    # PTM target ("PTM1"): 40-aa synthetic serum-albumin prefix
    # with an inline [SEP] phosphoserine at position 15. The
    # MLX-side featurizer requires ``CHAI_MLX_ALLOW_MODIFIED_RESIDUES=1``
    # to bypass the fail-fast validator; the CUDA-side featurizer
    # (chai-lab upstream) accepts the bracketed token directly.
    "PTM1": _target(
        "PTM1",
        (
            (
                "protein",
                "PTM1",
                "MKWVTFISLLLLFS[SEP]AYSRGVFRRDAHKSEVAHRFKDLGEE",
            ),
        ),
        description="40-aa protein with inline [SEP] phosphoserine (T2-3 PTM smoke)",
        kinds=frozenset({"monomer", "ptm"}),
    ),

    # Multi-ligand ("LYSM"): hen egg-white lysozyme (129 aa) with
    # two distinct ligands (NAG + FRC). Exercises the multi-ligand
    # entity-dispatch path.
    "LYSM": _target(
        "LYSM",
        (
            (
                "protein",
                "LYSM",
                "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSR"
                "WWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQA"
                "WIRGCRL",
            ),
            (
                "ligand",
                "NAG",
                "OC[C@H]1O[C@H](O)[C@H](NC(C)=O)[C@@H](O)[C@@H]1O",
            ),
            (
                "ligand",
                "FRC",
                "OC[C@@H]1OC(=O)[C@H](O)[C@@H](O)[C@H]1O",
            ),
        ),
        description="Lysozyme + NAG + fructose (T2-4 multi-ligand smoke)",
        kinds=frozenset({"ligand", "multiligand"}),
    ),

    # Large ligand ("BIGL"): 18-aa protein + 162-heavy-atom
    # bis-aromatic PEG chain (simulates a long-linker PROTAC). The
    # ligand heavy-atom count is ~3× the largest in the core slate
    # (FK506 = 51 atoms), so this exercises the atom-attention
    # encoder's 128-key / 32-query block structure near its budget.
    "BIGL": _target(
        "BIGL",
        (
            ("protein", "BIGL", "MKWVTFISLLLLFSSAYS"),
            # Two aromatic rings bridging 50 ethylene-glycol units;
            # 162 heavy atoms total (12 from the two benzenes + 150
            # from 50 × OCC). The original T2-5 smoke used the same
            # SMILES. Built programmatically so the
            # repeat count is explicit and audit-friendly.
            (
                "ligand",
                "L1",
                "c1ccccc1" + "OCC" * 50 + "c1ccccc1",
            ),
        ),
        description=(
            "18-aa protein + 162-heavy-atom bis-aromatic PEG ligand "
            "(T2-5 large-ligand smoke)"
        ),
        kinds=frozenset({"ligand", "largeligand"}),
    ),
}


def fasta_for(target: Target) -> str:
    """Build a FASTA string from a :class:`Target` (multi-entity aware)."""
    return target.to_fasta()


def filter_targets(target_kinds: str | None = None) -> dict[str, Target]:
    """Filter :data:`DEFAULT_TARGETS` by a comma-separated list of kind tags.

    Passing ``None`` or an empty string returns :data:`DEFAULT_TARGETS`
    unchanged.  Unknown kinds raise ``KeyError`` to catch typos in CLI
    arguments.
    """
    if not target_kinds:
        return dict(DEFAULT_TARGETS)
    wanted = {k.strip() for k in target_kinds.split(",") if k.strip()}
    known = {k for target in DEFAULT_TARGETS.values() for k in target.kinds}
    unknown = wanted - known
    if unknown:
        raise KeyError(
            f"Unknown target kinds {sorted(unknown)}; known kinds: {sorted(known)}"
        )
    return {
        name: target
        for name, target in DEFAULT_TARGETS.items()
        if wanted & target.kinds
    }
