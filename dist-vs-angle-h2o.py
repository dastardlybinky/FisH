import argparse
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
from pathlib import Path
import pandas as pd
from fish_helpers import describe_system, FilenameParseError
import sys
import math
import os
from matplotlib.colors import LinearSegmentedColormap

PLACEMENT_ORDER = {"E": 0, "M": 1, "EM": 2, "A": 3, "N": 4, "None": 4, None: 4}
FLUORINATED_AMINO_ACIDS = ["V6G", "V3R", "V3S", "I3G", "E3G"]


def combo_sort_key(combo):
    if len(combo) == 3:
        fluor_pct, placement, chain_length = combo
        p_rank = PLACEMENT_ORDER.get(placement, 4)
        return (fluor_pct, chain_length, p_rank)
    else:
        fluor_pct, chain_length = combo
        return (fluor_pct, chain_length)


def load_pdb_with_pbc(pdb_path):
    try:
        traj = md.load(pdb_path)
    except Exception as e:
        print("Failed to load PDB:", file=sys.stderr)
        print(f"  pdb_path: {pdb_path}", file=sys.stderr)
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


def identify_chain_atom_groups(traj, selection="not resname HOH", chain_length=None, chunked_chains=False):
    top = traj.topology
    solute_idx = top.select(selection)
    if solute_idx.size == 0:
        return [], np.array([], dtype=bool)

    traj_solute = traj.atom_slice(solute_idx)
    top_solute = traj_solute.topology
    residues = list(top_solute.residues)
    n_residues = len(residues)

    use_chunking = (
        chunked_chains
        and chain_length is not None
        and chain_length > 0
        and n_residues % chain_length == 0
    )

    groups = []

    if use_chunking:
        n_groups = n_residues // chain_length
        for chain_idx in range(n_groups):
            start = chain_idx * chain_length
            end = start + chain_length
            atoms_in_chain = [
                int(solute_idx[atom.index])
                for atom in top_solute.atoms
                if start <= atom.residue.index < end
            ]
            if atoms_in_chain:
                groups.append(np.array(atoms_in_chain, dtype=int))
    else:
        mols = top_solute.find_molecules()
        for atom_set in mols:
            atoms_in_chain = [int(solute_idx[atom.index]) for atom in atom_set]
            if atoms_in_chain:
                groups.append(np.array(atoms_in_chain, dtype=int))

    top_full = traj.topology
    residues_full = list(top_full.residues)
    atom_to_res = np.array([atom.residue.index for atom in top_full.atoms], dtype=int)

    fluorinated_mask = []
    for grp in groups:
        res_indices = {int(atom_to_res[idx]) for idx in grp}
        names = [residues_full[ri].name.strip().upper() for ri in res_indices]
        has_f = any(name in FLUORINATED_AMINO_ACIDS for name in names)
        fluorinated_mask.append(has_f)

    return groups, np.array(fluorinated_mask, dtype=bool)


def identify_waters(top):
    water_O_indices = []
    angle_triplets = []

    for res in top.residues:
        if not (res.is_water or res.name.strip().upper() in ("HOH", "WAT", "TIP3", "TIP3P")):
            continue

        O_idx = None
        H_indices = []
        for atom in res.atoms:
            try:
                sym = atom.element.symbol
            except Exception:
                sym = atom.name[0].upper()

            if sym == "O":
                O_idx = atom.index
            elif sym == "H":
                H_indices.append(atom.index)

        if O_idx is not None and len(H_indices) >= 2:
            water_O_indices.append(O_idx)
            angle_triplets.append((H_indices[0], O_idx, H_indices[1]))

    return water_O_indices, angle_triplets


def _process_system_worker(args):
    (
        pdb_path,
        selection,
        placement_combos,
        chunked_chains,
        sys_type,
        standardize_by_chain_length,
    ) = args

    pdb_path = str(pdb_path)

    try:
        info = describe_system(pdb_path, must_exist_traj=False)
    except FilenameParseError as e:
        raise RuntimeError(f"Cannot parse filename {pdb_path!r}: {e}") from e

    solv_pdb_path = info.get("solvated_pdb_path", None)
    if solv_pdb_path is None or not os.path.exists(solv_pdb_path):
        raise FileNotFoundError(
            f"Could not find solvated PDB corresponding to {pdb_path!r} "
            f"(expected at {solv_pdb_path!r})."
        )

    chain_length = info["chain_length"]
    fluor_pct = info["fluorination_percent"]
    placement = info.get("fluorine_placement")

    traj = load_pdb_with_pbc(solv_pdb_path)
    top = traj.topology

    chain_groups, fluorinated_mask = identify_chain_atom_groups(
        traj,
        selection=selection,
        chain_length=chain_length,
        chunked_chains=chunked_chains,
    )

    water_O_indices, angle_triplets = identify_waters(top)

    distances_pm_flat = np.array([], dtype=float)
    angles_deg_flat = np.array([], dtype=float)
    distances_pm_f_flat = np.array([], dtype=float)
    angles_deg_f_flat = np.array([], dtype=float)
    distances_pm_nf_flat = np.array([], dtype=float)
    angles_deg_nf_flat = np.array([], dtype=float)

    weights_all = np.array([], dtype=float)
    weights_f = np.array([], dtype=float)
    weights_nf = np.array([], dtype=float)

    n_chains = len(chain_groups)
    n_f_chains = int(fluorinated_mask.sum()) if fluorinated_mask.size else 0
    n_nf_chains = n_chains - n_f_chains

    if chain_groups and water_O_indices and angle_triplets:
        xyz = traj.xyz
        n_frames = xyz.shape[0]

        chain_centroids = np.empty((n_frames, n_chains, 3), dtype=float)
        for ic, atom_idx_array in enumerate(chain_groups):
            coords = xyz[:, atom_idx_array, :]
            chain_centroids[:, ic, :] = coords.mean(axis=1)

        water_pos = xyz[:, water_O_indices, :]

        diff = chain_centroids[:, :, None, :] - water_pos[:, None, :, :]
        distances_nm = np.linalg.norm(diff, axis=3)
        distances_pm = distances_nm * 1000.0

        angles_rad = md.compute_angles(traj, angle_triplets)
        angles_deg = np.degrees(angles_rad)
        angles_deg_expanded = np.repeat(angles_deg[:, None, :], n_chains, axis=1)

        distances_pm_flat = distances_pm.ravel()
        angles_deg_flat = angles_deg_expanded.ravel()

        if chain_length > 0 and standardize_by_chain_length:
            weight_scalar = 1.0 / float(chain_length)
        else:
            weight_scalar = 1.0

        def make_weights(arr):
            if arr.size == 0:
                return np.array([], dtype=float)
            return np.full(arr.shape, weight_scalar, dtype=float)

        weights_all = make_weights(distances_pm_flat)

        if fluorinated_mask.size == n_chains:
            mask_f = fluorinated_mask
            mask_nf = ~fluorinated_mask

            if mask_f.any():
                distances_pm_f_flat = distances_pm[:, mask_f, :].ravel()
                angles_deg_f_flat = angles_deg_expanded[:, mask_f, :].ravel()
                weights_f = make_weights(distances_pm_f_flat)
            if mask_nf.any():
                distances_pm_nf_flat = distances_pm[:, mask_nf, :].ravel()
                angles_deg_nf_flat = angles_deg_expanded[:, mask_nf, :].ravel()
                weights_nf = make_weights(distances_pm_nf_flat)

    if placement_combos:
        combo = (fluor_pct, placement, chain_length)
    else:
        combo = (fluor_pct, chain_length)

    return {
        "pdb_path": pdb_path,
        "combo": combo,
        "distances_pm": distances_pm_flat,
        "angles_deg": angles_deg_flat,
        "distances_pm_f": distances_pm_f_flat,
        "angles_deg_f": angles_deg_f_flat,
        "distances_pm_nf": distances_pm_nf_flat,
        "angles_deg_nf": angles_deg_nf_flat,
        "weights_all": weights_all,
        "weights_f": weights_f,
        "weights_nf": weights_nf,
        "n_chains": n_chains,
        "n_f_chains": n_f_chains,
        "n_nf_chains": n_nf_chains,
    }


def aggregate_by_combo_water(system_results):
    by_combo = {}
    all_combos = set()
    for res in system_results:
        combo = res["combo"]
        all_combos.add(combo)

        d_all = res["distances_pm"]
        a_all = res["angles_deg"]
        d_f = res["distances_pm_f"]
        a_f = res["angles_deg_f"]
        d_nf = res["distances_pm_nf"]
        a_nf = res["angles_deg_nf"]

        w_all = res.get("weights_all", np.array([], dtype=float))
        w_f = res.get("weights_f", np.array([], dtype=float))
        w_nf = res.get("weights_nf", np.array([], dtype=float))

        entry = by_combo.setdefault(
            combo,
            {
                "dist": [],
                "ang": [],
                "dist_f": [],
                "ang_f": [],
                "dist_nf": [],
                "ang_nf": [],
                "w_all": [],
                "w_f": [],
                "w_nf": [],
            },
        )

        if d_all.size and a_all.size:
            entry["dist"].append(d_all)
            entry["ang"].append(a_all)
            if w_all.size:
                entry["w_all"].append(w_all)

        if d_f.size and a_f.size:
            entry["dist_f"].append(d_f)
            entry["ang_f"].append(a_f)
            if w_f.size:
                entry["w_f"].append(w_f)

        if d_nf.size and a_nf.size:
            entry["dist_nf"].append(d_nf)
            entry["ang_nf"].append(a_nf)
            if w_nf.size:
                entry["w_nf"].append(w_nf)

    agg = {}
    for combo in all_combos:
        parts = by_combo.get(combo, None)
        if parts is None:
            agg[combo] = {
                "distances_pm": np.array([], dtype=float),
                "angles_deg": np.array([], dtype=float),
                "distances_pm_f": np.array([], dtype=float),
                "angles_deg_f": np.array([], dtype=float),
                "distances_pm_nf": np.array([], dtype=float),
                "angles_deg_nf": np.array([], dtype=float),
                "weights_all": np.array([], dtype=float),
                "weights_f": np.array([], dtype=float),
                "weights_nf": np.array([], dtype=float),
            }
        else:
            def concat_or_empty(lst):
                if lst:
                    return np.concatenate(lst)
                return np.array([], dtype=float)

            agg[combo] = {
                "distances_pm": concat_or_empty(parts["dist"]),
                "angles_deg": concat_or_empty(parts["ang"]),
                "distances_pm_f": concat_or_empty(parts["dist_f"]),
                "angles_deg_f": concat_or_empty(parts["ang_f"]),
                "distances_pm_nf": concat_or_empty(parts["dist_nf"]),
                "angles_deg_nf": concat_or_empty(parts["ang_nf"]),
                "weights_all": concat_or_empty(parts["w_all"]),
                "weights_f": concat_or_empty(parts["w_f"]),
                "weights_nf": concat_or_empty(parts["w_nf"]),
            }

    return agg


def make_colormaps():
    cmap_all = LinearSegmentedColormap.from_list(
        "all_combo_cmap",
        ["#000000", "#00fffb", "#ffff00", "#ffffff"],
    )
    cmap_nf = LinearSegmentedColormap.from_list(
        "nf_cmap",
        ["#000000", "#00ff00", "#ccff00", "#ffffff"],
    )
    cmap_f = LinearSegmentedColormap.from_list(
        "f_cmap",
        ["#000000", "#ff00ff", "#ff9900", "#ffff00"],
    )
    return cmap_all, cmap_nf, cmap_f


def plot_occurrence_maps_per_combo(
    combo_results,
    output="water_distance_angle_maps.png",
    distance_range=(200.0, 1000.0),
    angle_range=(90.0, 120.0),
    bins=(200, 200),
    sys_type="solo",
):
    sns.set_theme(style="darkgrid")
    plt.style.use("dark_background")

    cmap_all, cmap_nf, cmap_f = make_colormaps()

    combos = sorted(combo_results.keys(), key=combo_sort_key)
    if not combos:
        print("No data to plot.")
        return

    split_status = (sys_type == "combo")

    if not split_status:
        n = len(combos)
        ncols = min(3, n)
        nrows = math.ceil(n / ncols)
        n_panels = n
    else:
        base_n = len(combos)
        ncols_base = min(3, base_n)
        nrows = math.ceil(base_n / ncols_base)
        ncols = ncols_base * 2
        n_panels = base_n * 2

    hist_by_combo = {}
    global_max = 0.0

    for combo in combos:
        data = combo_results[combo]
        d_all = data["distances_pm"]
        a_all = data["angles_deg"]
        d_f = data["distances_pm_f"]
        a_f = data["angles_deg_f"]
        d_nf = data["distances_pm_nf"]
        a_nf = data["angles_deg_nf"]

        w_all = data["weights_all"]
        w_f = data["weights_f"]
        w_nf = data["weights_nf"]

        H_all = H_f = H_nf = None
        xedges = yedges = None

        if d_all.size and a_all.size:
            H_all, xedges, yedges = np.histogram2d(
                d_all,
                a_all,
                bins=bins,
                range=[distance_range, angle_range],
                weights=w_all if w_all.size else None,
            )
            hmax = H_all.max() if H_all.size else 0.0
            if hmax > global_max:
                global_max = float(hmax)
        else:
            H_all, xedges, yedges = np.histogram2d(
                np.array([], dtype=float),
                np.array([], dtype=float),
                bins=bins,
                range=[distance_range, angle_range],
            )

        if d_f.size and a_f.size:
            H_f, _, _ = np.histogram2d(
                d_f,
                a_f,
                bins=[xedges, yedges],
                weights=w_f if w_f.size else None,
            )
        else:
            H_f = np.zeros_like(H_all)

        if d_nf.size and a_nf.size:
            H_nf, _, _ = np.histogram2d(
                d_nf,
                a_nf,
                bins=[xedges, yedges],
                weights=w_nf if w_nf.size else None,
            )
        else:
            H_nf = np.zeros_like(H_all)

        hist_by_combo[combo] = {
            "H_all": H_all,
            "H_f": H_f,
            "H_nf": H_nf,
            "xedges": xedges,
            "yedges": yedges,
        }

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols + 2, 4 * nrows + 1),
        sharex=True,
        sharey=True,
    )

    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    elif nrows == 1 or ncols == 1:
        axs = np.array(axs).reshape(nrows, ncols)

    fig.patch.set_facecolor("black")
    for ax_row in axs:
        for ax in np.atleast_1d(ax_row):
            ax.set_facecolor("#111111")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("white")

    im = None
    im_nf = None
    im_f = None

    if not split_status:
        for idx, combo in enumerate(combos):
            row = idx // ncols
            col = idx % ncols
            ax = axs[row, col]

            H_all = hist_by_combo[combo]["H_all"]
            xedges = hist_by_combo[combo]["xedges"]
            yedges = hist_by_combo[combo]["yedges"]

            if H_all.size == 0 or np.all(H_all == 0):
                ax.set_visible(False)
                continue

            im = ax.imshow(
                H_all.T,
                origin="lower",
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                aspect="auto",
                cmap=cmap_all,
                vmin=0.0,
                vmax=global_max if global_max > 0 else None,
            )

            if row == nrows - 1:
                ax.set_xlabel("chain centroid – O(water) distance (pm)", fontsize=10, color="white")
            if col == 0:
                ax.set_ylabel("H–O–H angle (deg)", fontsize=10, color="white")

            if len(combo) == 3:
                fluor_pct, placement, chain_length = combo
                title = f"{fluor_pct:.1f}% F, {placement}, {chain_length}C"
            else:
                fluor_pct, chain_length = combo
                title = f"{fluor_pct:.1f}% F, {chain_length}C"
            ax.set_title(title, fontsize=11, color="white")

            ax.set_xticks(np.linspace(distance_range[0], distance_range[1], 9))
            ax.set_yticks(np.linspace(angle_range[0], angle_range[1], 7))
            ax.grid(color="white", linewidth=0.6, alpha=0.6)

        for idx in range(len(combos), n_panels):
            row = idx // ncols
            col = idx % ncols
            axs[row, col].set_visible(False)

        fig.tight_layout(rect=[0.0, 0.0, 0.86, 1.0])

        if im is not None:
            cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label("Occurrence (a.u.)", fontsize=11, color="white")
            cbar.ax.yaxis.set_tick_params(color="white")
            for tick in cbar.ax.get_yticklabels():
                tick.set_color("white")

    else:
        base_n = len(combos)
        ncols_base = ncols // 2
        panel_idx = 0
        for base_idx, combo in enumerate(combos):
            row = base_idx // ncols_base
            base_col = base_idx % ncols_base

            ax_nf = axs[row, 2 * base_col]
            ax_f = axs[row, 2 * base_col + 1]

            H_f = hist_by_combo[combo]["H_f"]
            H_nf = hist_by_combo[combo]["H_nf"]
            xedges = hist_by_combo[combo]["xedges"]
            yedges = hist_by_combo[combo]["yedges"]

            has_nf = H_nf.size and np.any(H_nf > 0)
            has_f = H_f.size and np.any(H_f > 0)

            if has_nf:
                im_nf = ax_nf.imshow(
                    H_nf.T,
                    origin="lower",
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    aspect="auto",
                    cmap=cmap_nf,
                    vmin=0.0,
                    vmax=global_max if global_max > 0 else None,
                )
            else:
                ax_nf.set_visible(False)

            if has_f:
                im_f = ax_f.imshow(
                    H_f.T,
                    origin="lower",
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    aspect="auto",
                    cmap=cmap_f,
                    vmin=0.0,
                    vmax=global_max if global_max > 0 else None,
                )
            else:
                ax_f.set_visible(False)

            if row == nrows - 1:
                if has_nf:
                    ax_nf.set_xlabel("distance (pm)", fontsize=10, color="white")
                if has_f:
                    ax_f.set_xlabel("distance (pm)", fontsize=10, color="white")
            if base_col == 0:
                if has_nf:
                    ax_nf.set_ylabel("H–O–H angle (deg)", fontsize=10, color="white")

            if len(combo) == 3:
                fluor_pct, placement, chain_length = combo
                base_label = f"{fluor_pct:.1f}% F, {placement}, {chain_length}C"
            else:
                fluor_pct, chain_length = combo
                base_label = f"{fluor_pct:.1f}% F, {chain_length}C"

            if has_nf:
                ax_nf.set_title(base_label + "\nNon-fluorinated chains", fontsize=10, color="white")
            if has_f:
                ax_f.set_title(base_label + "\nFluorinated chains", fontsize=10, color="white")

            for ax in (ax_nf, ax_f):
                if not ax.get_visible():
                    continue
                ax.set_xticks(np.linspace(distance_range[0], distance_range[1], 9))
                ax.set_yticks(np.linspace(angle_range[0], angle_range[1], 7))
                ax.grid(color="white", linewidth=0.6, alpha=0.6)

            panel_idx += 2

        for idx in range(panel_idx, n_panels):
            row = idx // ncols
            col = idx % ncols
            axs[row, col].set_visible(False)

        fig.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])

        if im_nf is not None:
            cbar_nf_ax = fig.add_axes([0.82, 0.55, 0.02, 0.35])
            cbar_nf = fig.colorbar(im_nf, cax=cbar_nf_ax)
            cbar_nf.set_label("Non-fluorinated occurrence (a.u.)", fontsize=11, color="white")
            cbar_nf.ax.yaxis.set_tick_params(color="white")
            for tick in cbar_nf.ax.get_yticklabels():
                tick.set_color("white")

        if im_f is not None:
            cbar_f_ax = fig.add_axes([0.82, 0.10, 0.02, 0.35])
            cbar_f = fig.colorbar(im_f, cax=cbar_f_ax)
            cbar_f.set_label("Fluorinated occurrence (a.u.)", fontsize=11, color="white")
            cbar_f.ax.yaxis.set_tick_params(color="white")
            for tick in cbar_f.ax.get_yticklabels():
                tick.set_color("white")

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
            "For each system, compute distance between each residue chain centroid "
            "and each water oxygen (using solvated topology), paired with each water's "
            "H–O–H angle, and plot 2D occurrence maps per "
            "(fluorination %, [placement], chain length) combo."
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
        help="MDTraj selection for solute atoms / chains (default: 'not resname HOH').",
    )
    parser.add_argument(
        "--output",
        default="water_distance_angle_maps.png",
        help="Output filename for the grid of occurrence maps.",
    )
    parser.add_argument(
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
            "Otherwise, chains are defined by MDTraj's find_molecules()."
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
        "--dist-min",
        type=float,
        default=200.0,
        help="Minimum distance (pm) for histogram x-range (default: 200).",
    )
    parser.add_argument(
        "--dist-max",
        type=float,
        default=1000.0,
        help="Maximum distance (pm) for histogram x-range (default: 1000).",
    )
    parser.add_argument(
        "--angle-min",
        type=float,
        default=90.0,
        help="Minimum H–O–H angle (deg) for histogram y-range (default: 90).",
    )
    parser.add_argument(
        "--angle-max",
        type=float,
        default=120.0,
        help="Maximum H–O–H angle (deg) for histogram y-range (default: 120).",
    )
    parser.add_argument(
        "--bins-x",
        type=int,
        default=200,
        help="Number of bins in distance (x) dimension (default: 200).",
    )
    parser.add_argument(
        "--bins-y",
        type=int,
        default=200,
        help="Number of bins in angle (y) dimension (default: 200).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, print per-system and per-combo summaries.",
    )
    parser.add_argument(
        "--standardize-by-chain-length",
        action="store_true",
        help="If set, weight contributions by 1 / chain_length for each system.",
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
                args.placement_combos,
                args.chunked_chains,
                args.sys_type,
                args.standardize_by_chain_length,
            )
        )

    if n_workers == 1 or n_systems == 1:
        system_results = [_process_system_worker(w) for w in worker_args]
    else:
        with mp.Pool(processes=n_workers) as pool:
            system_results = pool.map(_process_system_worker, worker_args)

    combo_results = aggregate_by_combo_water(system_results)

    if args.verbose:
        combo_set = set(r["combo"] for r in system_results)
        combos_sorted = sorted(combo_set, key=combo_sort_key)

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

    plot_occurrence_maps_per_combo(
        combo_results,
        output=args.output,
        distance_range=(args.dist_min, args.dist_max),
        angle_range=(args.angle_min, args.angle_max),
        bins=(args.bins_x, args.bins_y),
        sys_type=args.sys_type,
    )


if __name__ == "__main__":
    main()
