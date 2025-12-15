import argparse
from pathlib import Path
import multiprocessing as mp
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
import seaborn as sns

from fish_helpers import (describe_system, FilenameParseError,)
import sys

PLACEMENT_ORDER = {"E": 0, "M": 1, "EM": 2, "A": 3, "N": 4, "None": 4, None: 4}
FLUORINATED_AMINO_ACIDS = ["V6G", "V3R", "V3S", "I3G", "E3G"]


def combo_sort_key(combo):
    if len(combo) == 3:
        fluor_pct, placement, chain_length = combo
        p_rank = PLACEMENT_ORDER.get(placement, 4)
        return (fluor_pct, p_rank, chain_length)
    else:
        fluor_pct, chain_length = combo
        return (fluor_pct, 0, chain_length)


def prepare_traj_with_pbc(topology, trajectory):
    try:
        traj = md.load(trajectory, top=topology)
    except Exception as e:
        print(
            "Failed to load trajectory with given topology:",
            file=sys.stderr,
        )
        print(f"  topology:   {topology}", file=sys.stderr)
        print(f"  trajectory: {trajectory}", file=sys.stderr)
        raise

    if traj.unitcell_lengths is not None:
        top = traj.topology
        anchor_molecules = top.find_molecules()

        try:
            traj = traj.image_molecules(anchor_molecules=anchor_molecules)
        except TypeError:
            traj = traj.image_molecules()

        try:
            traj = traj.make_molecules_whole(anchor_molecules=anchor_molecules)
        except TypeError:
            traj = traj.make_molecules_whole()

    return traj


def _get_residue_representatives(traj, selection="not resname HOH",
                                 chain_length=None, chunked_chains=False):
    """
    Treat each *residue* (within the selection) as one RDF particle.

    Returns
    -------
    rep_indices : np.ndarray (n_residues_sel,)
        Representative atom index for each selected residue (traj atom indices).
    fluorinated_mask : np.ndarray (n_residues_sel,) of bool
        True for residues whose name is in FLUORINATED_AMINO_ACIDS.
    chain_ids : np.ndarray (n_residues_sel,) of int
        Chain index for each residue. When chunked_chains is True and valid,
        chains are contiguous blocks of chain_length residues. Otherwise,
        chains are topological molecules (top.find_molecules()).
    """
    top = traj.topology
    sel_idx = top.select(selection)
    if sel_idx.size == 0:
        raise ValueError(f"Selection '{selection}' returned no atoms.")

    sel_set = set(sel_idx.tolist())

    selected_residues = []
    rep_indices = []
    fluorinated_mask = []

    for res in top.residues:
        atoms_in_sel = [a for a in res.atoms if a.index in sel_set]
        if not atoms_in_sel:
            continue

        selected_residues.append(res)
        rep_indices.append(atoms_in_sel[0].index)

        name = res.name.strip().upper()
        fluorinated_mask.append(name in FLUORINATED_AMINO_ACIDS)

    n_sel_res = len(selected_residues)
    rep_indices = np.array(rep_indices, dtype=int)
    fluorinated_mask = np.array(fluorinated_mask, dtype=bool)

    if n_sel_res == 0:
        return rep_indices, fluorinated_mask, np.zeros(0, dtype=int)

    chain_ids = np.zeros(n_sel_res, dtype=int)

    use_chunking = (
        chunked_chains
        and chain_length is not None
        and chain_length > 0
        and n_sel_res % int(chain_length) == 0
    )

    if chunked_chains and not use_chunking:
        print(
            f"Warning: --chunked-chains requested but chain_length={chain_length} "
            f"does not divide n_selected_residues={n_sel_res}; "
            f"falling back to topological molecules.",
            file=sys.stderr,
        )

    if use_chunking:
        chain_length = int(chain_length)
        for i in range(n_sel_res):
            chain_ids[i] = i // chain_length
    else:
        mols = top.find_molecules()
        atom_to_mol = np.empty(top.n_atoms, dtype=int)
        for im, mol in enumerate(mols):
            for atom in mol:
                atom_to_mol[atom.index] = im
        for i, rep_idx in enumerate(rep_indices):
            chain_ids[i] = atom_to_mol[rep_idx]

    return rep_indices, fluorinated_mask, chain_ids


def _make_time_slices(n_frames, n_slices):
    """Return a list of arrays of frame indices (np.ndarray) for each slice."""
    if n_frames == 0 or n_slices <= 0:
        return []
    all_indices = np.arange(n_frames, dtype=int)
    slices = np.array_split(all_indices, n_slices)
    return [sl for sl in slices if sl.size > 0]


def _compute_rdf_slices(traj, selection, r_max, n_bins, n_slices,
                        chain_length=None, chunked_chains=False):
    """
    Compute RDF g(r) between *residues* (represented by a single atom per residue)
    over multiple time slices, EXCLUDING pairs where the two residues are in
    the same chain (as defined by chain_ids from chunked-chains logic).

    Returns
    -------
    bin_centers : (n_bins,)
    slice_g : (n_slices_eff, n_bins)
        RDF curves for each time slice actually used.
    n_frames : int
    n_groups : int
        Number of residue "particles".
    fluorinated_mask : (n_groups,) bool
    """
    rep_indices, fluorinated_mask, chain_ids = _get_residue_representatives(
        traj,
        selection=selection,
        chain_length=chain_length,
        chunked_chains=chunked_chains,
    )

    n_frames = traj.n_frames
    n_groups = rep_indices.size

    bin_edges = np.linspace(0.0, r_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if n_groups < 2 or n_frames == 0:
        slice_g = np.zeros((0, n_bins), dtype=float)
        return bin_centers, slice_g, n_frames, n_groups, fluorinated_mask

    pairs = []
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            if chain_ids[i] == chain_ids[j]:
                continue
            pairs.append((rep_indices[i], rep_indices[j]))
    if not pairs:
        slice_g = np.zeros((0, n_bins), dtype=float)
        return bin_centers, slice_g, n_frames, n_groups, fluorinated_mask

    pairs = np.array(pairs, dtype=int)

    dr = np.diff(bin_edges)

    if traj.unitcell_lengths is not None:
        lengths = traj.unitcell_lengths
        volumes = lengths[:, 0] * lengths[:, 1] * lengths[:, 2]
    else:
        volumes = np.ones(n_frames, dtype=float)

    frame_slices = _make_time_slices(n_frames, n_slices)
    slice_g_list = []

    for frames in frame_slices:
        subtraj = traj[frames]
        dists = md.compute_distances(subtraj, pairs)
        dists_flat = dists.ravel()

        counts, _ = np.histogram(dists_flat, bins=bin_edges)

        n_frames_slice = frames.size
        if n_frames_slice == 0:
            slice_g_list.append(np.zeros_like(bin_centers))
            continue

        vol_mean = float(volumes[frames].mean())

        N = float(n_groups)
        rho = N / vol_mean
        shell_volumes = 4.0 * np.pi * (bin_centers ** 2) * dr
        ideal_counts = (n_frames_slice * N * rho * shell_volumes) / 2.0
        ideal_counts[ideal_counts == 0.0] = np.nan
        g_r = counts / ideal_counts

        slice_g_list.append(g_r)

    if slice_g_list:
        slice_g = np.vstack(slice_g_list)
    else:
        slice_g = np.zeros((0, n_bins), dtype=float)

    return bin_centers, slice_g, n_frames, n_groups, fluorinated_mask


def _process_system_worker(args):
    (
        pdb_path,
        selection,
        r_max,
        n_bins,
        n_slices,
        placement_combos,
        sys_type,
        chunked_chains,
        traj_cap,
    ) = args

    pdb_path = str(pdb_path)

    try:
        info = describe_system(pdb_path, must_exist_traj=True)
    except FilenameParseError as e:
        raise RuntimeError(f"Cannot parse filename {pdb_path!r}: {e}") from e

    traj_paths = info.get("trajectory_paths")
    traj_path_single = info.get("trajectory_path")

    if traj_paths and len(traj_paths) > 0:
        traj_inputs = [str(p) for p in traj_paths]
    elif traj_path_single is not None:
        traj_inputs = [str(traj_path_single)]
    else:
        raise FileNotFoundError(
            f"Could not find trajectory file(s) corresponding to {pdb_path!r}"
        )

    if traj_cap is not None and traj_cap > 0:
        traj_inputs = traj_inputs[:traj_cap]

    filtered_traj_inputs = []
    for t in traj_inputs:
        name = Path(t).name
        if name.endswith(".dcd") and "_water_lastframe" in name:
            print(
                "Skipping _water_lastframe DCD in trajectory list:",
                file=sys.stderr,
            )
            print(f"  topology:   {pdb_path}", file=sys.stderr)
            print(f"  trajectory: {t}", file=sys.stderr)
            continue
        filtered_traj_inputs.append(t)

    if not filtered_traj_inputs:
        raise FileNotFoundError(
            f"No usable trajectory files found for {pdb_path!r} "
            "(only _water_lastframe DCDs or missing trajectories)."
        )

    chain_length = info["chain_length"]
    fluor_pct = info["fluorination_percent"]
    placement = info.get("fluorine_placement")

    traj = prepare_traj_with_pbc(pdb_path, filtered_traj_inputs)

    bin_centers, slice_g, n_frames, n_groups, fluorinated_mask = _compute_rdf_slices(
        traj,
        selection=selection,
        r_max=r_max,
        n_bins=n_bins,
        n_slices=n_slices,
        chain_length=chain_length,
        chunked_chains=chunked_chains,
    )

    if placement_combos:
        combo = (fluor_pct, placement, chain_length)
    else:
        combo = (fluor_pct, chain_length)

    return {
        "pdb_path": pdb_path,
        "combo": combo,
        "bin_centers": bin_centers,
        "slice_g": slice_g,
        "n_frames": n_frames,
        "n_groups": n_groups,
        "fluorinated_mask": fluorinated_mask,
    }


def aggregate_rdf_by_combo(system_results, n_slices):
    """
    Aggregate RDF results across systems for each combo.

    Returns dict:
        combo -> {
            "bin_centers": (n_bins,),
            "slice_g_mean": (n_slices_eff, n_bins),
            "slice_g_std":  (n_slices_eff, n_bins),
            "n_slices_eff": int
        }

    Note: if some systems have fewer effective slices (e.g. fewer frames),
    we align by the minimum number of slices actually present up to n_slices.
    """
    by_combo = {}
    for res in system_results:
        combo = res["combo"]
        by_combo.setdefault(combo, []).append(res)

    agg = {}
    for combo, reps in by_combo.items():
        bin_centers = reps[0]["bin_centers"]
        n_bins = bin_centers.size

        slice_counts = [r["slice_g"].shape[0] for r in reps]
        if not slice_counts:
            continue
        max_slices_present = min(max(slice_counts), n_slices)

        if max_slices_present == 0:
            continue

        slice_g_mean = np.zeros((max_slices_present, n_bins), dtype=float)
        slice_g_std = np.zeros_like(slice_g_mean)

        for k in range(max_slices_present):
            curves = []
            for r in reps:
                if r["slice_g"].shape[0] > k:
                    curves.append(r["slice_g"][k])
            if not curves:
                continue
            curves = np.stack(curves, axis=0)
            slice_g_mean[k] = curves.mean(axis=0)
            slice_g_std[k] = curves.std(axis=0)

        agg[combo] = {
            "bin_centers": bin_centers,
            "slice_g_mean": slice_g_mean,
            "slice_g_std": slice_g_std,
            "n_slices_eff": max_slices_present,
        }

    return agg


def _collect_pdbs_from_input(input_path: str):
    p = Path(input_path)
    if p.is_dir():
        pdbs = sorted(
            pp for pp in p.glob("*.pdb")
            if "_solv" not in pp.name
        )
        return [str(pp) for pp in pdbs]
    else:
        if "_solv" in p.name:
            return []
        return [str(p)]


def _slice_labels(n_slices):
    """
    Simple labels like "0–20%", "20–40%", ... based on fraction of trajectory.
    """
    bounds = np.linspace(0, 100, n_slices + 1)
    labels = []
    for i in range(n_slices):
        lo = bounds[i]
        hi = bounds[i + 1]
        labels.append(f"{lo:.0f}–{hi:.0f}%")
    return labels


def plot_rdf_by_combo(rdf_results, output, n_slices, max_cols=3):
    """
    Make a subplot for each (fluor_pct, placement?, chain_length) combo.
    Within each subplot, plot RDF curves for each time slice.
    If a combo has no RDF data (e.g. only 1 residue or no cross-chain pairs),
    we still make a subplot but show a 'no RDF' message.
    """
    sns.set_theme(style="darkgrid")
    plt.style.use("dark_background")

    combos_sorted = sorted(rdf_results.keys(), key=combo_sort_key)
    n_combos = len(combos_sorted)
    if n_combos == 0:
        print("No RDF data to plot.")
        return

    n_cols = min(max_cols, n_combos)
    n_rows = int(np.ceil(n_combos / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False
    )

    slice_labels = _slice_labels(n_slices)
    palette = sns.color_palette("winter", n_colors=n_slices)

    for idx, combo in enumerate(combos_sorted):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        data = rdf_results[combo]
        r = data["bin_centers"]
        g_mean = data["slice_g_mean"]
        n_slices_eff = data.get("n_slices_eff", 0)
        has_data = data.get("has_data", True)

        if len(combo) == 3:
            fluor_pct, placement, chain_length = combo
            title = f"{fluor_pct:.1f}% F, {placement}, {chain_length}C"
        else:
            fluor_pct, chain_length = combo
            title = f"{fluor_pct:.1f}% F, {chain_length}C"

        if has_data and n_slices_eff > 0:
            for s in range(n_slices_eff):
                label = slice_labels[s] if s < len(slice_labels) else f"Slice {s+1}"
                ax.plot(r, g_mean[s], linewidth=2.0, color=palette[s], label=label)
        else:
            ax.text(
                0.5,
                0.5,
                "No RDF\n(only 1 residue,\nno cross-chain pairs,\n"
                "or no frames)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )

        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r"r (nm)", fontsize=11)
        ax.set_ylabel("g(r)", fontsize=11)
        ax.grid(alpha=0.3)

    for j in range(n_combos, n_rows * n_cols):
        row = j // n_cols
        col = j % n_cols
        fig.delaxes(axes[row][col])

    handles, labels = [], []
    for combo in combos_sorted:
        d = rdf_results[combo]
        if d.get("has_data", True) and d.get("n_slices_eff", 0) > 0:
            for s in range(min(d["n_slices_eff"], len(slice_labels))):
                label = slice_labels[s]
                handles.append(plt.Line2D([], [], color=palette[s], label=label))
                labels.append(label)
            break

    if handles:
        seen = set()
        uniq_handles = []
        uniq_labels = []
        for h, lab in zip(handles, labels):
            if lab not in seen:
                seen.add(lab)
                uniq_handles.append(h)
                uniq_labels.append(lab)

        fig.legend(
            uniq_handles,
            uniq_labels,
            loc="center right",
            bbox_to_anchor=(1.02, 0.5),
            framealpha=0.3,
        )

    fig.tight_layout(rect=[0, 0, 0.88, 1])
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute radial distribution functions (RDF) for *individual solute residues* "
            "using the same system/combination logic as the SASA script. For each "
            "(fluorination %, chain length and optional placement) combination, produce a "
            "subplot with RDF curves for different time slices along the trajectory. "
            "Residue–residue pairs within the same chain are excluded using --chunked-chains logic."
        )
    )
    parser.add_argument(
        "input",
        help=(
            "Directory containing .pdb files (following the naming convention)"
            " and their corresponding trajectories, or a single .pdb file. "
            "Files with '_solv' in the name are ignored."
        ),
    )
    parser.add_argument(
        "-s",
        "--selection",
        default="not resname HOH",
        help="MDTraj selection for solute atoms (default: 'not resname HOH').",
    )
    parser.add_argument(
        "--rdf-output",
        default="rdf_by_combo.png",
        help="Output filename for RDF plot (default: rdf_by_combo.png).",
    )
    parser.add_argument(
        "--r-max",
        type=float,
        default=3.0,
        help="Maximum distance (nm) for RDF (default: 3.0 nm).",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=75,
        help="Number of radial bins for RDF (default: 75).",
    )
    parser.add_argument(
        "--n-slices",
        type=int,
        default=5,
        help="Number of time slices along the trajectory (default: 5).",
    )
    parser.add_argument(
        "-j",
        "--n-workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: min(#systems, #CPUs)).",
    )
    parser.add_argument(
        "--chunked-chains",
        action="store_true",
        help=(
            "If set, define chains as contiguous blocks of chain_length residues "
            "within the selected solute residues (using chain_length from fish_helpers). "
            "Otherwise chains are defined by topological molecules via MDTraj."
        ),
    )
    parser.add_argument(
        "--placement-combos",
        action="store_true",
        help="If set, separate combinations by fluorine placement as well as % and chain length.",
    )
    parser.add_argument(
        "--sys-type",
        choices=["solo", "combo"],
        default="solo",
        help="System type: 'solo' (default) or 'combo' (fluorinated + non-fluorinated sets).",
    )
    parser.add_argument(
        "--traj-cap",
        type=int,
        default=None,
        help=(
            "Maximum number of trajectory segments to use per system. "
            "1 uses only *_traj.dcd, 2 uses *_traj.dcd and *_traj1.dcd, etc. "
            "Default: use all available segments."
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        if input_path.name not in ("solo_sys", "combo_sys"):
            subdir_name = "solo_sys" if args.sys_type == "solo" else "combo_sys"
            input_path = input_path / subdir_name
        input_str = str(input_path)
    else:
        input_str = args.input

    pdb_paths = _collect_pdbs_from_input(input_str)
    if not pdb_paths:
        raise RuntimeError(
            f"No usable .pdb files found in {input_str!r} "
            "(files with '_solv' are ignored as they only contain final frame information)."
        )

    n_systems = len(pdb_paths)

    if args.n_workers is None:
        n_workers = min(n_systems, mp.cpu_count())
    else:
        n_workers = max(1, args.n_workers)

    worker_args = []
    for pdb_path in pdb_paths:
        worker_args.append(
            (
                pdb_path,
                args.selection,
                args.r_max,
                args.n_bins,
                args.n_slices,
                args.placement_combos,
                args.sys_type,
                args.chunked_chains,
                args.traj_cap,
            )
        )

    if n_workers == 1 or n_systems == 1:
        system_results = [_process_system_worker(w) for w in worker_args]
    else:
        with mp.Pool(processes=n_workers) as pool:
            system_results = pool.map(_process_system_worker, worker_args)

    rdf_results = aggregate_rdf_by_combo(system_results, n_slices=args.n_slices)

    combo_set = {res["combo"] for res in system_results}

    if rdf_results:
        example_bins = next(iter(rdf_results.values()))["bin_centers"]
    else:
        bin_edges = np.linspace(0.0, args.r_max, args.n_bins + 1)
        example_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for combo in combo_set:
        if combo not in rdf_results:
            n_bins = example_bins.size
            rdf_results[combo] = {
                "bin_centers": example_bins,
                "slice_g_mean": np.zeros((0, n_bins), dtype=float),
                "slice_g_std":  np.zeros((0, n_bins), dtype=float),
                "n_slices_eff": 0,
                "has_data": False,
            }
        else:
            rdf_results[combo]["has_data"] = True

    plot_rdf_by_combo(rdf_results, args.rdf_output, n_slices=args.n_slices)


if __name__ == "__main__":
    main()
