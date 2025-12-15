import argparse
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import multiprocessing as mp
from pathlib import Path
import pandas as pd
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


def compute_sasa_per_molecule(
    traj,
    selection="not resname HOH",
    probe_radius=0.14,
    n_sphere_points=1000,
    chain_length=None,
    chunked_chains=False,
):
    top = traj.topology
    solute_idx = top.select(selection)
    if solute_idx.size == 0:
        raise ValueError(f"Selection '{selection}' returned no atoms.")

    traj_solute = traj.atom_slice(solute_idx)

    sasa_atom = md.shrake_rupley(
        traj_solute,
        probe_radius=probe_radius,
        n_sphere_points=n_sphere_points,
        mode="atom",
    )

    top_solute = traj_solute.topology
    residues = list(top_solute.residues)
    n_residues = len(residues)
    n_frames, n_atoms_solute = sasa_atom.shape

    use_chunking = (
        chunked_chains
        and chain_length is not None
        and chain_length > 0
        and n_residues % chain_length == 0
    )

    if chunked_chains and not use_chunking:
        print(
            f"Warning: --chunked-chains requested but chain_length={chain_length} "
            f"does not divide n_residues={n_residues}; falling back to find_molecules().",
            file=sys.stderr,
        )

    if use_chunking:
        n_groups = n_residues // chain_length
        atom_to_group = np.empty(n_atoms_solute, dtype=int)
        for atom in top_solute.atoms:
            resid_local = atom.residue.index
            chain_idx = resid_local // chain_length
            atom_to_group[atom.index] = chain_idx
    else:
        mols = top_solute.find_molecules()
        n_groups = len(mols)
        atom_to_group = np.empty(n_atoms_solute, dtype=int)
        for ig, atom_set in enumerate(mols):
            for atom in atom_set:
                atom_to_group[atom.index] = ig

    sasa_per_group = np.zeros((n_frames, n_groups), dtype=float)
    for ig in range(n_groups):
        in_group = (atom_to_group == ig)
        sasa_per_group[:, ig] = sasa_atom[:, in_group].sum(axis=1)

    fluorinated_mask = np.zeros(n_groups, dtype=bool)
    if use_chunking:
        for chain_idx in range(n_groups):
            start = chain_idx * chain_length
            end = start + chain_length
            names = [res.name.strip().upper() for res in residues[start:end]]
            if names:
                has_f = any(n in FLUORINATED_AMINO_ACIDS for n in names)
                fluorinated_mask[chain_idx] = has_f
    else:
        mols = top_solute.find_molecules()
        for ig, atom_set in enumerate(mols):
            names = [atom.residue.name.strip().upper() for atom in atom_set]
            if names:
                has_f = any(n in FLUORINATED_AMINO_ACIDS for n in names)
                fluorinated_mask[ig] = has_f

    times = traj.time
    return times, sasa_per_group, fluorinated_mask


def compute_cv_curve_across_systems(total_sasa_matrix, n_folds=5, seed=481516):
    n_systems, n_frames = total_sasa_matrix.shape

    indices = np.arange(n_systems)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    folds = np.array_split(indices, n_folds)

    fold_curves = []
    for fold in folds:
        if fold.size == 0:
            continue
        fold_data = total_sasa_matrix[fold, :]
        fold_mean = fold_data.mean(axis=0)
        fold_curves.append(fold_mean)

    if len(fold_curves) == 0:
        overall_mean = total_sasa_matrix.mean(axis=0)
        return overall_mean, np.zeros_like(overall_mean)

    fold_curves = np.stack(fold_curves, axis=0)
    mean_curve = fold_curves.mean(axis=0)
    std_curve = fold_curves.std(axis=0)

    return mean_curve, std_curve


def _process_system_worker(args):
    (
        pdb_path,
        selection,
        probe_radius,
        n_sphere_points,
        standardize_by_chain_length,
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

    _, sasa_per_mol, fluorinated_mask = compute_sasa_per_molecule(
        traj,
        selection=selection,
        probe_radius=probe_radius,
        n_sphere_points=n_sphere_points,
        chain_length=chain_length,
        chunked_chains=chunked_chains,
    )

    total_sasa_all = sasa_per_mol.sum(axis=1)

    n_chains = sasa_per_mol.shape[1]
    n_f_chains = int(fluorinated_mask.sum())
    n_nf_chains = int((~fluorinated_mask).sum())

    if sys_type == "combo":
        if fluorinated_mask.any():
            sasa_f = sasa_per_mol[:, fluorinated_mask].sum(axis=1)
        else:
            sasa_f = np.zeros_like(total_sasa_all)
        if (~fluorinated_mask).any():
            sasa_nf = sasa_per_mol[:, ~fluorinated_mask].sum(axis=1)
        else:
            sasa_nf = np.zeros_like(total_sasa_all)
    else:
        sasa_f = None
        sasa_nf = None

    if standardize_by_chain_length and chain_length > 0:
        total_sasa_all = total_sasa_all / chain_length
        if sasa_f is not None:
            sasa_f = sasa_f / chain_length
        if sasa_nf is not None:
            sasa_nf = sasa_nf / chain_length

    n_frames = total_sasa_all.shape[0]
    frame_indices = np.arange(n_frames, dtype=float)

    if placement_combos:
        combo = (fluor_pct, placement, chain_length)
    else:
        combo = (fluor_pct, chain_length)

    return {
        "pdb_path": pdb_path,
        "combo": combo,
        "times": frame_indices,
        "total_sasa": total_sasa_all,
        "sasa_f": sasa_f,
        "sasa_nf": sasa_nf,
        "n_chains": n_chains,
        "n_f_chains": n_f_chains,
        "n_nf_chains": n_nf_chains,
    }


def aggregate_by_combo(system_results, n_folds=5, seed=481516):
    by_combo = {}
    for res in system_results:
        key = res["combo"]
        by_combo.setdefault(key, []).append(res)

    has_split = any(("sasa_f" in r and r["sasa_f"] is not None) for r in system_results)

    agg = {}
    for combo_idx, (combo, reps) in enumerate(
        sorted(by_combo.items(), key=lambda item: combo_sort_key(item[0]))
    ):
        times0 = reps[0]["times"]

        curves = np.stack([r["total_sasa"] for r in reps], axis=0)
        mean_curve, std_curve = compute_cv_curve_across_systems(
            curves,
            n_folds=n_folds,
            seed=seed + combo_idx,
        )

        entry = {
            "times": times0,
            "mean": mean_curve,
            "std": std_curve,
        }

        if has_split:
            curves_f = np.stack([r["sasa_f"] for r in reps], axis=0)
            curves_nf = np.stack([r["sasa_nf"] for r in reps], axis=0)
            mean_f, std_f = compute_cv_curve_across_systems(
                curves_f,
                n_folds=n_folds,
                seed=seed + combo_idx + 1000,
            )
            mean_nf, std_nf = compute_cv_curve_across_systems(
                curves_nf,
                n_folds=n_folds,
                seed=seed + combo_idx + 2000,
            )
            entry["mean_f"] = mean_f
            entry["std_f"] = std_f
            entry["mean_nf"] = mean_nf
            entry["std_nf"] = std_nf

        agg[combo] = entry

    return agg


def _outline_violin_inners(ax,
                           inner_color="black",
                           outline_color="white",
                           outline_extra=1.0):
    """
    Give inner box/median lines in a violinplot a thin outline while
    preserving their original linewidths (so the middle section can stay bulky).
    """
    for line in ax.lines:
        orig_lw = line.get_linewidth()
        line.set_color(inner_color)
        line.set_path_effects([
            pe.Stroke(linewidth=orig_lw + outline_extra, foreground=outline_color),
            pe.Normal(),
        ])


def plot_sasa_combos_cv(cv_results, output, combo_to_color, frame_time_ps, ax=None):
    sns.set_theme(style="darkgrid")
    plt.style.use("dark_background")

    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        own_fig = True
    else:
        fig = ax.figure

    max_time_ns = 0.0

    for combo, data in sorted(cv_results.items(), key=lambda item: combo_sort_key(item[0])):
        if len(combo) == 3:
            fluor_pct, placement, chain_length = combo
            label = f"{fluor_pct:.1f}% F, {placement}, {chain_length}C"
        else:
            fluor_pct, chain_length = combo
            label = f"{fluor_pct:.1f}% F, {chain_length}C"

        frame_indices = data["times"]
        times_ns = (frame_indices * frame_time_ps) / 1000.0
        mean_curve = data["mean"]
        std_curve = data["std"]

        if times_ns.size > 0 and times_ns[-1] > max_time_ns:
            max_time_ns = times_ns[-1]

        color = combo_to_color.get(combo, None)

        ax.fill_between(
            times_ns,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
            color=color,
            zorder=1,
        )
        ax.plot(times_ns, mean_curve, linewidth=2.0, label=label, color=color, zorder=2)

    if max_time_ns > 0:
        ax.set_xlim(left=0.0, right=max_time_ns)
        n_ticks = 5
        xticks = np.linspace(0.0, max_time_ns, n_ticks)
        ax.set_xticks(xticks)
    else:
        ax.set_xlim(left=0.0)
    ax.margins(x=0)

    ax.set_xlabel("Time (ns)", fontsize=12)
    ax.set_ylabel(r"SASA (nm$^2$)", fontsize=12)
    ax.set_title(
        "SASA over Time (5-fold CV)",
        fontsize=14,
    )

    ax.grid(alpha=0.3)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_alpha(0.5)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        framealpha=0.3,
        borderaxespad=0.0,
    )

    if own_fig:
        fig.tight_layout()
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_sasa_combos_cv_split(cv_results, output, combo_to_color, frame_time_ps, ax=None):
    sns.set_theme(style="darkgrid")
    plt.style.use("dark_background")

    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        own_fig = True
    else:
        fig = ax.figure

    max_time_ns = 0.0

    for combo, data in sorted(cv_results.items(), key=lambda item: combo_sort_key(item[0])):
        if "mean_f" not in data or "mean_nf" not in data:
            continue

        if len(combo) == 3:
            fluor_pct, placement, chain_length = combo
            base_label = f"{fluor_pct:.1f}% F, {placement}, {chain_length}C"
        else:
            fluor_pct, chain_length = combo
            base_label = f"{fluor_pct:.1f}% F, {chain_length}C"

        frame_indices = data["times"]
        times_ns = (frame_indices * frame_time_ps) / 1000.0
        mean_f = data["mean_f"]
        std_f = data["std_f"]
        mean_nf = data["mean_nf"]
        std_nf = data["std_nf"]

        if times_ns.size > 0 and times_ns[-1] > max_time_ns:
            max_time_ns = times_ns[-1]

        color = combo_to_color.get(combo, None)

        ax.fill_between(
            times_ns,
            mean_f - std_f,
            mean_f + std_f,
            alpha=0.2,
            color=color,
            zorder=1,
        )
        ax.plot(
            times_ns,
            mean_f,
            linewidth=2.0,
            label=base_label + " (F)",
            color=color,
            zorder=2,
        )

        ax.fill_between(
            times_ns,
            mean_nf - std_nf,
            mean_nf + std_nf,
            alpha=0.1,
            color=color,
            zorder=1,
        )
        ax.plot(
            times_ns,
            mean_nf,
            linewidth=2.0,
            linestyle="--",
            label=base_label + " (non-F)",
            color=color,
            zorder=2,
        )

    if max_time_ns > 0:
        ax.set_xlim(left=0.0, right=max_time_ns)
        n_ticks = 6
        xticks = np.linspace(0.0, max_time_ns, n_ticks)
        ax.set_xticks(xticks)
    else:
        ax.set_xlim(left=0.0)
    ax.margins(x=0)

    ax.set_xlabel("Time (ns)", fontsize=12)
    ax.set_ylabel(r"SASA (nm$^2$)", fontsize=12)
    ax.set_title("SASA over Time by Fluorination (5-fold CV)", fontsize=14,)

    ax.grid(alpha=0.3)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_alpha(0.5)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        framealpha=0.3,
        borderaxespad=0.0,
    )

    if own_fig:
        fig.tight_layout()
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_delta_sasa_violin(system_results, output, total_time_ps, combo_to_color, ax=None):
    sns.set_theme(style="darkgrid")
    plt.style.use("dark_background")

    by_combo = {}
    for res in system_results:
        combo = res["combo"]
        total_sasa = res["total_sasa"]
        if total_sasa.size < 2:
            continue
        delta = total_sasa[-1] - total_sasa[0]
        by_combo.setdefault(combo, []).append(delta)

    combos_sorted = sorted(by_combo.keys(), key=combo_sort_key)

    categories = []
    values = []
    labels = []
    for combo in combos_sorted:
        if len(combo) == 3:
            fluor_pct, placement, chain_length = combo
            label = f"{fluor_pct:.1f}% F, {placement}, {chain_length}C"
        else:
            fluor_pct, chain_length = combo
            label = f"{fluor_pct:.1f}% F, {chain_length}C"
        labels.append(label)
        deltas = by_combo[combo]
        for d in deltas:
            categories.append(label)
            values.append(d)

    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        own_fig = True
    else:
        fig = ax.figure

    if categories:
        palette = {}
        for combo, label in zip(combos_sorted, labels):
            palette[label] = combo_to_color.get(combo, None)
        sns.violinplot(
            x=categories,
            y=values,
            ax=ax,
            cut=3,
            palette=palette,
            order=labels,
        )

        for coll in ax.collections:
            coll.set_alpha(0.85)
            coll.set_edgecolor("white")
            coll.set_linewidth(1.0)

        _outline_violin_inners(ax)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_xlabel("System (Fluorination %, Chain Length)", fontsize=12)
    ax.set_ylabel(r"Δ Total SASA (nm$^2$)", fontsize=12)
    ax.set_title("Δ SASA", fontsize=14)

    ax.grid(alpha=0.3)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_alpha(0.5)

    if own_fig:
        total_time_ns = total_time_ps / 1000.0
        fig.text(
            0.5,
            0.01,
            f"Total time: {total_time_ns:.2f} ns",
            ha="center",
            va="bottom",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_delta_sasa_violin_split(system_results, output, combo_to_color, ax=None):
    sns.set_theme(style="darkgrid")
    plt.style.use("dark_background")

    by_combo = {}
    for res in system_results:
        combo = res["combo"]
        sasa_f = res.get("sasa_f", None)
        sasa_nf = res.get("sasa_nf", None)
        if sasa_f is None or sasa_nf is None:
            continue
        if sasa_f.size < 2 or sasa_nf.size < 2:
            continue
        delta_f = sasa_f[-1] - sasa_f[0]
        delta_nf = sasa_nf[-1] - sasa_nf[0]
        by_combo.setdefault(combo, {"F": [], "NF": []})
        by_combo[combo]["F"].append(delta_f)
        by_combo[combo]["NF"].append(delta_nf)

    combos_sorted = sorted(by_combo.keys(), key=combo_sort_key)

    records = []
    labels_order = []
    for combo in combos_sorted:
        if len(combo) == 3:
            fluor_pct, placement, chain_length = combo
            label = f"{fluor_pct:.1f}% F, {placement}, {chain_length}C"
        else:
            fluor_pct, chain_length = combo
            label = f"{fluor_pct:.1f}% F, {chain_length}C"
        labels_order.append(label)
        for d in by_combo[combo]["F"]:
            records.append({"category": label, "delta": d, "status": "Fluorinated"})
        for d in by_combo[combo]["NF"]:
            records.append({"category": label, "delta": d, "status": "Non-fluorinated"})

    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        own_fig = True
    else:
        fig = ax.figure

    if records:
        df = pd.DataFrame(records)

        sns.violinplot(
            data=df,
            x="category",
            y="delta",
            hue="status",
            split=True,
            cut=3,
            ax=ax,
            order=labels_order,
            hue_order=["Fluorinated", "Non-fluorinated"],
        )

        if ax.legend_ is not None:
            ax.legend_.remove()

        label_to_combo = {label: combo for combo, label in zip(combos_sorted, labels_order)}

        collections = ax.collections
        for i, label in enumerate(labels_order):
            combo = label_to_combo.get(label)
            base_color = combo_to_color.get(combo, None)
            if base_color is None:
                continue

            idx_f = 2 * i
            idx_nf = 2 * i + 1
            if idx_nf >= len(collections):
                continue

            coll_f = collections[idx_f]
            coll_nf = collections[idx_nf]

            coll_f.set_facecolor(base_color)
            coll_nf.set_facecolor(base_color)

            for coll in (coll_f, coll_nf):
                coll.set_alpha(0.85)
                coll.set_edgecolor("white")
                coll.set_linewidth(1.0)

            coll_nf.set_hatch("//")

        _outline_violin_inners(ax)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_xlabel("System (Fluorination %, Chain Length)", fontsize=12)
    ax.set_ylabel(r"Δ SASA (nm$^2$)", fontsize=12)
    ax.set_title("Δ SASA by Fluorination", fontsize=14)

    ax.grid(alpha=0.3)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_alpha(0.5)

    if own_fig:
        fig.tight_layout()
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)


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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute total SASA over time for each system, then 5-fold CV across systems "
            "for each (fluorination %, chain length) combo (using fish_helpers), "
            "and plot each combo as a separate line on a dark-mode plot. "
            "Ignores any .pdb with '_solv' in the filename."
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
        "--time-output",
        default="sasa_combos_cv.png",
        help="Output plot filename for time series (default: sasa_combos_cv.png).",
    )
    parser.add_argument(
        "--delta-output",
        default="delta_sasa_violin.png",
        help="Output filename for ΔSASA violin plot (default: delta_sasa_violin.png).",
    )
    parser.add_argument(
        "--probe-radius",
        type=float,
        default=0.14,
        help="Probe radius in nm (default: 0.14 nm ~ 1.4 Å).",
    )
    parser.add_argument(
        "--n-sphere-points",
        type=int,
        default=1000,
        help="Number of sphere points for Shrake–Rupley (default: 1000).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds across systems (default: 5).",
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
            "If set, group solute into chains by fixed chain_length residue chunks. "
            "By default, chains are defined by MDTraj's find_molecules()."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=481516,
        help="Base random seed for CV splitting (default: 123).",
    )
    parser.add_argument(
        "--standardize-by-chain-length",
        action="store_true",
        help="If set, divide total SASA by chain length for each system.",
    )
    parser.add_argument(
        "--frame-time-ps",
        type=float,
        default=50.0,
        help="Time in ps represented by each frame (default: 250 ps).",
    )
    parser.add_argument(
        "--traj-cap",
        type=int,
        default=None,
        help=("Maximum number of trajectory segments to use per system." "Default: use all available segments."),)
    parser.add_argument(
        "--combine-plots",
        action="store_true",
        help="If set, combine time series and ΔSASA violin into one figure with subplots.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, print number of samples and chain counts used for each combination.",
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
                args.probe_radius,
                args.n_sphere_points,
                args.standardize_by_chain_length,
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

    cv_results = aggregate_by_combo(system_results, n_folds=args.n_folds, seed=args.seed)

    if system_results:
        n_frames = system_results[0]["total_sasa"].shape[0]
        if n_frames > 0:
            total_time_ps = float(n_frames * args.frame_time_ps)
        else:
            total_time_ps = 0.0
    else:
        total_time_ps = 0.0

    combo_set = set(cv_results.keys())
    for res in system_results:
        combo_set.add(res["combo"])
    combos_sorted = sorted(combo_set, key=combo_sort_key)
    palette_colors = sns.color_palette("hls", n_colors=len(combos_sorted)) if combos_sorted else []
    combo_to_color = {combo: col for combo, col in zip(combos_sorted, palette_colors)}

    if args.verbose:
        print("=== Per-system details ===")
        for idx, res in enumerate(system_results):
            combo = res["combo"]
            n_f = res.get("n_f_chains", 0)
            n_nf = res.get("n_nf_chains", 0)
            n_total = res.get("n_chains", n_f + n_nf)

            if len(combo) == 3:
                fluor_pct, placement, chain_length = combo
                label = f"{fluor_pct:.1f}% F, {placement}, {chain_length}C"
            else:
                fluor_pct, chain_length = combo
                label = f"{fluor_pct:.1f}% F, {chain_length}C"

            pdb_name = Path(res.get("pdb_path", f"system_{idx+1}")).name

            print(
                f"System {idx+1}: {pdb_name} | {label} | "
                f"{n_total} chains ({n_f} fluorinated, {n_nf} non-fluorinated)"
            )

        counts = {}
        f_chain_counts = {}
        nf_chain_counts = {}
        for res in system_results:
            combo = res["combo"]
            counts[combo] = counts.get(combo, 0) + 1
            n_f = res.get("n_f_chains", 0)
            n_nf = res.get("n_nf_chains", 0)
            f_chain_counts[combo] = f_chain_counts.get(combo, 0) + n_f
            nf_chain_counts[combo] = nf_chain_counts.get(combo, 0) + n_nf

        print("\n=== Per-combo summary ===")
        for combo in combos_sorted:
            n_samples = counts.get(combo, 0)
            n_f_tot = f_chain_counts.get(combo, 0)
            n_nf_tot = nf_chain_counts.get(combo, 0)
            n_chains_tot = n_f_tot + n_nf_tot

            if len(combo) == 3:
                fluor_pct, placement, chain_length = combo
                label = f"{fluor_pct:.1f}% F, {placement}, {chain_length}C"
            else:
                fluor_pct, chain_length = combo
                label = f"{fluor_pct:.1f}% F, {chain_length}C"

            print(
                f"Combo {label}: {n_samples} samples, "
                f"{n_chains_tot} total chains "
                f"({n_f_tot} fluorinated, {n_nf_tot} non-fluorinated)"
            )

    if args.combine_plots and system_results:
        sns.set_theme(style="darkgrid")
        plt.style.use("dark_background")
        if args.sys_type == "combo":
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            ax_delta_total = axs[0, 0]
            ax_time_total = axs[0, 1]
            ax_delta_split = axs[1, 0]
            ax_time_split = axs[1, 1]

            plot_delta_sasa_violin(system_results, None, total_time_ps, combo_to_color, ax=ax_delta_total)
            plot_sasa_combos_cv(cv_results, None, combo_to_color, args.frame_time_ps, ax=ax_time_total)
            plot_delta_sasa_violin_split(system_results, None, combo_to_color, ax=ax_delta_split)
            plot_sasa_combos_cv_split(cv_results, None, combo_to_color, args.frame_time_ps, ax=ax_time_split)
        else:
            fig, (ax_delta, ax_time) = plt.subplots(1, 2, figsize=(12, 5))
            plot_delta_sasa_violin(system_results, None, total_time_ps, combo_to_color, ax=ax_delta)
            plot_sasa_combos_cv(cv_results, None, combo_to_color, args.frame_time_ps, ax=ax_time)

        fig.tight_layout()
        fig.savefig("SASA-combined-plot.png", dpi=400, bbox_inches="tight")
        plt.close(fig)
    else:
        plot_sasa_combos_cv(cv_results, args.time_output, combo_to_color, args.frame_time_ps)
        if system_results:
            plot_delta_sasa_violin(system_results, args.delta_output, total_time_ps, combo_to_color)


if __name__ == "__main__":
    main()
