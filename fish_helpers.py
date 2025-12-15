"""
Helper functions for working with MD filenames of the form:

    ChainLength_Polarity_Fluorination_#Replicates_sys#_(final|traj)(_solv).(pdb|dcd)

Examples:
    2C_NP_FH_20_sys1_final.pdb
    2C_NP_FH_20_sys1_traj.dcd
    2C_NP_FH_20_sys1_traj1.dcd
    2C_NP_FH_20_sys1_final_solv.pdb
"""

import os
from typing import Dict, Optional, Literal, Any, List

FluorPlacement = Literal["E", "M", "EM", "A", "N"]


class FilenameParseError(ValueError):
    """Raised when a filename does not match the expected convention."""


def _parse_pdb_filename(pdb_path: str) -> Dict[str, Any]:
    """
    Parse a topology .pdb filename.

    Returns a dict with:
        chain_len_token: str
        chain_length: int
        polarity: str
        fluor_pattern: str
        replicates: int
        system_index: int
        kind: str
        basename_no_ext: str
        directory: str
    """
    directory, filename = os.path.split(pdb_path)
    root, ext = os.path.splitext(filename)

    if ext.lower() != ".pdb":
        raise FilenameParseError(f"Expected a .pdb file, got {filename!r}")

    parts = root.split("_")
    if len(parts) < 6:
        raise FilenameParseError(
            f"Filename {filename!r} does not have at least 6 underscore-separated parts."
        )

    chain_len_token = parts[0]
    polarity = parts[1]
    fluor_pattern = parts[2]
    replicates_str = parts[3]
    sys_token = parts[4]
    kind = parts[5]

    if not chain_len_token.endswith("C"):
        raise FilenameParseError(
            f"Chain length token {chain_len_token!r} does not end with 'C'."
        )
    try:
        chain_length = int(chain_len_token[:-1])
    except ValueError:
        raise FilenameParseError(
            f"Cannot parse chain length from token {chain_len_token!r}."
        )

    if polarity not in {"P", "NP"}:
        raise FilenameParseError(
            f"Unexpected polarity {polarity!r}; expected 'P' or 'NP'."
        )

    try:
        replicates = int(replicates_str)
    except ValueError:
        raise FilenameParseError(
            f"Cannot parse replicate count from {replicates_str!r}."
        )

    if not sys_token.startswith("sys"):
        raise FilenameParseError(
            f"System token {sys_token!r} does not start with 'sys'."
        )
    try:
        system_index = int(sys_token[3:])
    except ValueError:
        raise FilenameParseError(
            f"Cannot parse system index from token {sys_token!r}."
        )

    return {
        "chain_len_token": chain_len_token,
        "chain_length": chain_length,
        "polarity": polarity,
        "fluor_pattern": fluor_pattern,
        "replicates": replicates,
        "system_index": system_index,
        "kind": kind,
        "basename_no_ext": root,
        "directory": directory or ".",
    }


def _get_traj_root_from_parsed(parsed: Dict[str, Any]) -> str:
    """
    Given parsed PDB info, return the *basename* (no directory, no extension)
    common root used for trajectory files, e.g.:

        2C_NP_FH_20_sys1_final(.pdb)      -> 2C_NP_FH_20_sys1_traj
        2C_NP_FH_20_sys1_final_solv.pdb   -> 2C_NP_FH_20_sys1_traj
    """
    base = parsed["basename_no_ext"]

    # If we're starting from a solvated final PDB, strip the "_solv" suffix first
    if base.endswith("_solv"):
        base = base[:-5]

    parts = base.split("_")
    # Usually last token is "final"; we replace it with "traj"
    # If it isn't, we still replace the last token with "traj" to stay consistent
    parts[-1] = "traj"
    traj_root = "_".join(parts)
    return traj_root


def find_trajectory_files(pdb_path: str, must_exist: bool = False) -> List[str]:
    """
    Given a topology .pdb file, return **all** trajectory .dcd paths that are
    continuations of each other.

    Convention:
        ..._final.pdb       ->  ..._traj.dcd, ..._traj1.dcd, ..._traj2.dcd, ...
        ..._final_solv.pdb  ->  same as above (ignores the '_solv' tag)

    Returns a list of existing file paths, ordered as:
        traj.dcd, traj1.dcd, traj2.dcd, ...

    If must_exist is True and no trajectory files are found, returns an empty list.
    """
    parsed = _parse_pdb_filename(pdb_path)
    dirpath = parsed["directory"] or "."
    traj_root = _get_traj_root_from_parsed(parsed)  # basename without extension
    rootname = os.path.basename(traj_root)

    if not os.path.isdir(dirpath):
        return []

    candidates: List[str] = []
    for fname in os.listdir(dirpath):
        if not fname.endswith(".dcd"):
            continue
        if not fname.startswith(rootname):
            continue

        # suffix between rootname and ".dcd"
        suffix = fname[len(rootname):-4]
        # Accept:
        #   rootname.dcd       (suffix == "")
        #   rootname1.dcd      (suffix is digits)
        if suffix == "" or suffix.isdigit():
            candidates.append(os.path.join(dirpath, fname))

    # Sort by the numeric suffix: "" -> 0, "1" -> 1, "2" -> 2, ...
    def _sort_key(path: str) -> int:
        fname = os.path.basename(path)
        suffix = fname[len(rootname):-4]
        return 0 if suffix == "" else int(suffix)

    candidates.sort(key=_sort_key)

    if must_exist and not candidates:
        return []

    return candidates


def find_trajectory_file(pdb_path: str, must_exist: bool = False) -> Optional[str]:
    """
    Given a topology .pdb file, return the expected *primary* trajectory .dcd path.

    Convention:
        ..._final.pdb  →  ..._traj.dcd

    NOTE: this is kept for backward compatibility. If you want *all* trajectory
    segments (including continuations ..._traj1.dcd, ..._traj2.dcd, ...),
    use `find_trajectory_files`.
    """
    parsed = _parse_pdb_filename(pdb_path)
    dirpath = parsed["directory"]
    base = parsed["basename_no_ext"]

    parts = base.split("_")
    if parts[-1] != "final":
        # If it's something else (e.g., final_solv), we don't enforce here;
        # we just replace the last token with "traj" to maintain old behavior.
        pass

    parts[-1] = "traj"
    traj_root = "_".join(parts)
    traj_filename = traj_root + ".dcd"
    traj_path = os.path.join(dirpath, traj_filename)

    if must_exist and not os.path.exists(traj_path):
        return None

    return traj_path


def find_solvated_pdb_file(pdb_path: str, must_exist: bool = False) -> Optional[str]:
    """
    Given a final .pdb file, return the expected solvated final .pdb path.

    Convention:
        ..._final.pdb      →  ..._final_solv.pdb
        ..._final_solv.pdb →  ..._final_solv.pdb
    """
    parsed = _parse_pdb_filename(pdb_path)
    dirpath = parsed["directory"]
    base = parsed["basename_no_ext"]

    if base.endswith("_solv"):
        solv_path = pdb_path
    else:
        solv_root = base + "_solv"
        solv_path = os.path.join(dirpath, solv_root + ".pdb")

    if must_exist and not os.path.exists(solv_path):
        return None

    return solv_path


def get_chain_length(pdb_path: str) -> int:
    """Return the integer chain length."""
    parsed = _parse_pdb_filename(pdb_path)
    return parsed["chain_length"]


def get_polarity(pdb_path: str) -> str:
    """Return the polarity string, 'P' or 'NP'."""
    parsed = _parse_pdb_filename(pdb_path)
    return parsed["polarity"]


def get_fluorination_percentage(pdb_path: str) -> float:
    """Return the percentage of residues that are fluorinated (0–100)."""
    parsed = _parse_pdb_filename(pdb_path)
    fluor_pattern = parsed["fluor_pattern"]

    if not fluor_pattern:
        return 0.0

    total = len(fluor_pattern)
    fluorinated = fluor_pattern.count("F")

    return (fluorinated / total) * 100.0


def get_fluorine_placement(pdb_path: str) -> FluorPlacement:
    """
    Return fluorine placement category:

        - 'A'  : all residues are fluorinated
        - 'E'  : only edge residues
        - 'M'  : only middle residues
        - 'EM' : both edge and middle residues
        - 'N'  : no residues are fluorinated
    """
    parsed = _parse_pdb_filename(pdb_path)
    pattern = parsed["fluor_pattern"]

    if not pattern:
        return "N"

    n = len(pattern)
    if all(c == "F" for c in pattern):
        return "A"

    if n == 1:
        return "N" if pattern[0] != "F" else "A"

    has_edge_F = (pattern[0] == "F") or (pattern[-1] == "F")
    has_middle_F = any(pattern[i] == "F" for i in range(1, n - 1))

    if not has_edge_F and not has_middle_F:
        return "N"
    if has_edge_F and has_middle_F:
        return "EM"
    if has_edge_F:
        return "E"
    if has_middle_F:
        return "M"

    return "N"


def get_replicate_count(pdb_path: str) -> int:
    """Return the number of chain replicates in the system."""
    parsed = _parse_pdb_filename(pdb_path)
    return parsed["replicates"]


def describe_system(pdb_path: str, must_exist_traj: bool = False) -> Dict[str, Any]:
    """
    Return a dictionary with parsed information:

        {
            "pdb_path": ...,
            "trajectory_path": ... or None,            # primary traj
            "trajectory_paths": [...],                 # all continuations
            "solvated_pdb_path": ...,
            "chain_length": ...,
            "polarity": ...,
            "fluor_pattern": ...,
            "fluorination_percent": ...,
            "fluorine_placement": ...,
            "replicates": ...,
            "system_index": ...,
        }
    """
    parsed = _parse_pdb_filename(pdb_path)

    traj = find_trajectory_file(pdb_path, must_exist=must_exist_traj)
    traj_list = find_trajectory_files(pdb_path, must_exist=must_exist_traj)
    solv = find_solvated_pdb_file(pdb_path, must_exist=False)
    fluor_pct = get_fluorination_percentage(pdb_path)
    placement = get_fluorine_placement(pdb_path)

    return {
        "pdb_path": pdb_path,
        "trajectory_path": traj,
        "trajectory_paths": traj_list,
        "solvated_pdb_path": solv,
        "chain_length": parsed["chain_length"],
        "polarity": parsed["polarity"],
        "fluor_pattern": parsed["fluor_pattern"],
        "fluorination_percent": fluor_pct,
        "fluorine_placement": placement,
        "replicates": parsed["replicates"],
        "system_index": parsed["system_index"],
    }


__all__ = [
    "FilenameParseError",
    "find_trajectory_file",
    "find_trajectory_files",
    "find_solvated_pdb_file",
    "get_chain_length",
    "get_polarity",
    "get_fluorination_percentage",
    "get_fluorine_placement",
    "get_replicate_count",
    "describe_system",
]
