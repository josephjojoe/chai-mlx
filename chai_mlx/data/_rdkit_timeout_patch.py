"""Workaround for chai-lab's multiprocessing-based RDKit timeout on macOS.

``chai_lab.utils.timeout.timeout`` wraps RDKit calls (``DetermineBonds``,
``GetSubstructMatches``) in a decorator that spawns a subprocess and
joins on a queue. The decorator's ``handler`` is a nested local
function, which Python cannot pickle under the ``spawn`` start method
(default on macOS Python 3.11+).  Symptom::

    AttributeError: Can't pickle local object 'timeout.<locals>.handler'
    AssertionError: Expected only one token asym, but got 0 asyms: ...

The protection this decorator provides is against **pathological**
ligands where RDKit genuinely hangs (tracked in
https://github.com/rdkit/rdkit/discussions/7289).  For every ligand in
our validation slate (FK506, methanethiol, etc.) and every protein
with disulfides, RDKit returns promptly, so the timeout never fires on
the happy path. On Linux, chai-lab's decorator works fine because
multiprocessing defaults to ``fork`` there.

We therefore swap the decorator for an identity pass-through **in
process** when ``apply_rdkit_timeout_patch()`` is called.  Nothing
else in chai-lab's behaviour changes; if RDKit hangs on a novel
ligand, the process will hang, just without the original 15-second
bail-out.

Invoke exactly once, before the first ``featurize_fasta`` call.  The
patch is idempotent.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_APPLIED = False


def apply_rdkit_timeout_patch() -> bool:
    """Replace ``chai_lab.utils.timeout.timeout`` with an identity decorator.

    Returns
    -------
    bool
        ``True`` if the patch was applied this call, ``False`` if it was
        already applied earlier or chai-lab is not installed.
    """
    global _APPLIED
    if _APPLIED:
        return False

    try:
        import chai_lab.utils.timeout as chai_timeout
        import chai_lab.data.sources.rdkit as chai_rdkit
    except ImportError:
        return False

    def _identity_timeout(timeout_after):  # noqa: ARG001 -- signature must match
        def _decorator(func):
            return func

        return _decorator

    chai_timeout.timeout = _identity_timeout
    # chai_lab.data.sources.rdkit imported ``timeout`` by name at module
    # load time, so we also need to rebind that reference.
    chai_rdkit.timeout = _identity_timeout

    # Re-decorate the two protected functions so the *already-decorated*
    # closures stop pointing at the broken timeout.  We keep the same
    # function bodies -- they were minimal RDKit wrappers anyway.
    from rdkit.Chem.rdDetermineBonds import DetermineBonds
    from rdkit import Chem

    def _add_bonds_unwrapped(mol):
        DetermineBonds(mol)
        return mol

    def _get_symmetries_unwrapped_factory(mol, max_symmetries: int = 1000):
        # Return a zero-arg fn matching the decorated signature in
        # chai_rdkit.get_intra_res_atom_symmetries's original closure.
        def _inner():
            return mol.GetSubstructMatches(
                mol,
                uniquify=False,
                maxMatches=max_symmetries,
                useChirality=False,
            )

        return _inner

    # The two functions ``rdkit._load_ref_conformer_from_rdkit`` and
    # ``rdkit.get_intra_res_atom_symmetries`` build local decorated
    # closures per call, so we don't need to replace the module-level
    # symbols themselves -- just ensure the decorator they see is the
    # identity above.  That's what rebinding ``chai_rdkit.timeout``
    # accomplishes, since those local closures do ``@timeout(...)`` at
    # call time and resolve ``timeout`` from the module's globals.
    del _add_bonds_unwrapped, _get_symmetries_unwrapped_factory, DetermineBonds, Chem

    _APPLIED = True
    logger.info(
        "Patched chai_lab.utils.timeout.timeout -> identity decorator "
        "(macOS multiprocessing-spawn workaround)"
    )
    return True
