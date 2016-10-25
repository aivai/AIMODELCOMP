#!/usr/bin/env python

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns
import os.path
from multiprocessing import Pool
import lims_utils
from allensdk.core.nwb_data_set import NwbDataSet
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor


NEURONAL_MODEL_TEMPLATES = {
    "glif_1": 395310469,
    "glif_2": 395310479,
    "glif_3": 395310475,
    "glif_4": 471355161,
    "glif_5": 395310498,
    "bp": 329230710,
    "baa": 491455321,
}

NWB_DOWNLOAD_TYPE_ID = 481007198

BASE_ORDER = [
    "baa",
    "bp",
    "glif_1",
    "glif_2",
    "glif_3",
    "glif_4",
    "glif_5",
]

LABELS = [
    "Biophys all active",
    "Biophys perisomatic",
    "GLIF level 1",
    "GLIF level 2",
    "GLIF level 3",
    "GLIF level 4",
    "GLIF level 5",
]


def collect_exp_var(ids):
    sql = """
        select sp.id, nmr.explained_variance_ratio from specimens sp
        join neuronal_models nm on nm.specimen_id = sp.id
        join neuronal_model_runs nmr on nmr.neuronal_model_id = nm.id
        where sp.id = any(%s)
        and nm.neuronal_model_template_id = %s
    """

    exp_var_data = {}
    for model_type, nmt_id in NEURONAL_MODEL_TEMPLATES.iteritems():
        results = lims_utils.query(sql, (list(ids), nmt_id))
        exp_var_data[model_type] = dict(results)

    return exp_var_data


def compare_fi_curves(ve_paths):
    ve_df = DataFrame(ve_paths)

    input_list = ve_df.reset_index().to_dict("records")
    pool = Pool()
    data_list = pool.map(compare_fi_curve, input_list)

    df = DataFrame(data_list)
    for model_type in NEURONAL_MODEL_TEMPLATES:
        df[model_type + "_slope_pct_diff"] = (df[model_type + "_slope"] - df["expt_slope"]) / df["expt_slope"]
        df[model_type + "_rheo_pct_diff"] = (df[model_type + "_rheo"] - df["expt_rheo"]) / df["expt_rheo"]

    slope_cols = [m + "_slope_pct_diff" for m in BASE_ORDER]
    rheo_cols = [m + "_rheo_pct_diff" for m in BASE_ORDER]

    plot_comparison(df.ix[:, slope_cols] * 100., "fI slope (% diff)", order=slope_cols, zeroline=True, filename="fi_slope.png")
    plot_comparison(df.ix[:, slope_cols] * 100., "fI slope (% diff)", order=slope_cols, xlim=(-100, 100), zeroline=True, filename="fi_slope_zoomed.png")
    plot_comparison(df.ix[:, rheo_cols] * 100., "rheobase (% diff)", order=rheo_cols, zeroline=True, filename="fi_rheo.png")
    plot_comparison(df.ix[:, rheo_cols] * 100., "rheobase (% diff)", order=rheo_cols, xlim=(-100, 100), zeroline=True, filename="fi_rheo_zoomed.png")


def compare_fi_curve(input_dict):
    specimen_key = "index"
    specimen_id = input_dict[specimen_key]
    expt_rheo, expt_slope = expt_fi_curve(specimen_id)
    info = {
        "expt_rheo": expt_rheo,
        "expt_slope": expt_slope,
    }

    for model_type in input_dict:
        if model_type == specimen_key:
            continue
        if type(input_dict[model_type]) != str:
            continue
        if not os.path.exists(input_dict[model_type]):
            continue
        ve_path = input_dict[model_type]
        rheo, slope = ve_fi_curve(specimen_id, ve_path)
        info[model_type + "_rheo"] = rheo
        info[model_type + "_slope"] = slope

    return info


def compare_ramp_latencies(ve_paths):
    ve_df = DataFrame(ve_paths)

    input_list = ve_df.reset_index().to_dict("records")
    pool = Pool()
    data_list = pool.map(compare_ramp_latency, input_list)
    df = DataFrame(data_list)
    for model_type in NEURONAL_MODEL_TEMPLATES:
        df[model_type + "_latency_pct_diff"] = (df[model_type + "_ramp_latency"] - df["expt_ramp_latency"]) / df["expt_ramp_latency"]
    cols = [m + "_latency_pct_diff" for m in BASE_ORDER]
    plot_comparison(df.ix[:, cols] * 100., "ramp latency (% diff)", order=cols, zeroline=True, filename="ramp_latency.png")
    plot_comparison(df.ix[:, cols] * 100., "ramp latency (% diff)", order=cols, zeroline=True, xlim=(-100, 100), filename="ramp_latency_zoomed.png")


def compare_ramp_latency(input_dict):
    specimen_key = "index"
    specimen_id = input_dict[specimen_key]
    expt_latency = expt_ramp_latency(specimen_id)

    info = {
        "expt_ramp_latency": expt_latency
    }
    for model_type in input_dict:
        if model_type == specimen_key:
            continue
        if type(input_dict[model_type]) != str:
            continue
        if not os.path.exists(input_dict[model_type]):
            continue
        ve_path = input_dict[model_type]
        info[model_type + "_ramp_latency"] = ve_ramp_latency(specimen_id, ve_path)
    return info


def compare_ap_dims(ve_paths, biophys_ids):
    ve_df = DataFrame(ve_paths)
    ve_df = ve_df.ix[biophys_ids, :]
    input_list = ve_df.reset_index().to_dict("records")
    pool = Pool()
    data_list = pool.map(compare_ap_dim, input_list)
    df = DataFrame(data_list)
    for model_type in ["bp", "baa"]:
        df[model_type + "_width_pct_diff"] = (df[model_type + "_width"] - df["expt_width"]) / df["expt_width"]
        df[model_type + "_height_pct_diff"] = (df[model_type + "_height"] - df["expt_height"]) / df["expt_height"]
    width_cols = [m + "_width_pct_diff" for m in BASE_ORDER]
    height_cols = [m + "_height_pct_diff" for m in BASE_ORDER]
    plot_comparison(df.ix[:, width_cols] * 100., "AP width (% diff)", order=width_cols, zeroline=True, filename="ap_width.png")
    plot_comparison(df.ix[:, height_cols] * 100., "AP height (% diff)", order=height_cols, zeroline=True, filename="ap_height.png")


def compare_ap_dim(input_dict):
    specimen_key = "index"
    specimen_id = input_dict[specimen_key]
    expt_width, expt_height = expt_ap_dim(specimen_id)

    info = {
        "expt_width": expt_width,
        "expt_height": expt_height,
    }
    for model_type in ["bp", "baa"]:
        if type(input_dict[model_type]) != str:
            continue
        if not os.path.exists(input_dict[model_type]):
            continue
        ve_path = input_dict[model_type]
        info[model_type + "_width"], info[model_type + "_height"] = ve_ap_dim(specimen_id, ve_path)
    return info


def expt_data_set(specimen_id):
    sql = """
        select wkf.storage_directory || wkf.filename from well_known_files wkf
        join specimens sp on sp.ephys_roi_result_id = wkf.attachable_id
        where sp.id = %s
        and wkf.well_known_file_type_id = %s
    """

    results = lims_utils.query(sql, (specimen_id, NWB_DOWNLOAD_TYPE_ID))
    nwb_path = results[0][0]
    return NwbDataSet(nwb_path)


def expt_fi_curve(specimen_id):
    data_set = expt_data_set(specimen_id)
    long_square_sweeps = lims_utils.get_sweeps_of_type("C1LSCOARSE", specimen_id, passed_only=True)
    fi_curve_data = dict([amp_and_spike_count(data_set, sweep) for sweep in long_square_sweeps])
    return fi_curve_stats(fi_curve_data)


def expt_ramp_latency(specimen_id):
    data_set = expt_data_set(specimen_id)
    ramp_sweeps = lims_utils.get_sweeps_of_type("C1RP25PR1S", specimen_id, passed_only=True)
    if len(ramp_sweeps) == 0:
        return np.nan
    return np.nanmean([data_set.get_spike_times(sweep)[0] if len(data_set.get_spike_times(sweep)) > 0 else np.nan
                      for sweep in ramp_sweeps])


def expt_ap_dim(specimen_id):
    data_set = expt_data_set(specimen_id)
    long_square_sweeps = lims_utils.get_sweeps_of_type("C1LSCOARSE", specimen_id, passed_only=True)
    fi_curve_data = dict([amp_and_spike_count(data_set, sweep) for sweep in long_square_sweeps])
    sweeps_by_amp = {amp_and_spike_count(data_set, sweep)[0]: sweep for sweep in long_square_sweeps}
    fi_arr = np.array([(amp, fi_curve_data[amp]) for amp in sorted(fi_curve_data.keys())])

    spiking_sweeps = np.flatnonzero(fi_arr[:, 1])
    if len(spiking_sweeps) == 0:
        return np.nan, np.nan
    rheo_sweep = sweeps_by_amp[fi_arr[spiking_sweeps[0], 0]]
#     print specimen_id, rheo_sweep
    v, i, t = lims_utils.get_sweep_v_i_t_from_set(data_set, rheo_sweep)
    swp_ext = EphysSweepFeatureExtractor(t, v, start=1.02, end=2.02)
    swp_ext.process_spikes()
    return (swp_ext.spike_feature("width")[0] * 1e3, swp_ext.spike_feature("peak_v")[0] - swp_ext.spike_feature("trough_v")[0])


def ve_ramp_latency(specimen_id, ve_path):
    data_set = NwbDataSet(ve_path)
    ramp_sweeps = lims_utils.get_sweeps_of_type("C1RP25PR1S", specimen_id, passed_only=True)
    if len(ramp_sweeps) == 0:
        return np.nan
    spike_times = data_set.get_spike_times(ramp_sweeps[0])
    if len(spike_times) > 0:
        return spike_times[0]
    else:
        return np.nan


def ve_fi_curve(specimen_id, ve_path):
    data_set = NwbDataSet(ve_path)
    expt_set = expt_data_set(specimen_id)
    long_square_sweeps = lims_utils.get_sweeps_of_type("C1LSCOARSE", specimen_id, passed_only=True)
    fi_curve_data = dict([amp_and_spike_count(data_set, sweep, expt_set) for sweep in long_square_sweeps])
    return fi_curve_stats(fi_curve_data)


def ve_ap_dim(specimen_id, ve_path):
    data_set = NwbDataSet(ve_path)
    expt_set = expt_data_set(specimen_id)
    long_square_sweeps = lims_utils.get_sweeps_of_type("C1LSCOARSE", specimen_id, passed_only=True)
    fi_curve_data = dict([amp_and_spike_count(data_set, sweep, expt_set) for sweep in long_square_sweeps])
    sweeps_by_amp = {amp_and_spike_count(data_set, sweep, expt_set)[0]: sweep for sweep in long_square_sweeps}
    fi_arr = np.array([(amp, fi_curve_data[amp]) for amp in sorted(fi_curve_data.keys())])

    spiking_sweeps = np.flatnonzero(fi_arr[:, 1])
    if len(spiking_sweeps) == 0:
        return np.nan, np.nan
    rheo_sweep = sweeps_by_amp[fi_arr[spiking_sweeps[0], 0]]
#     print specimen_id, rheo_sweep

    v, i, t = lims_utils.get_sweep_v_i_t_from_set(data_set, rheo_sweep)
    swp_ext = EphysSweepFeatureExtractor(t, v, start=1.02, end=2.02, filter=None)
    swp_ext.process_spikes()
    if len(swp_ext.spike_feature("width")) == 0:
        print "NO SPIKES FOR {:d} ON SWEEP {:d}".format(specimen_id, sweeps_by_amp[fi_arr[spiking_sweeps[0], 0]])
        print fi_arr
        print sweeps_by_amp
        return np.nan, np.nan
    return_vals = (swp_ext.spike_feature("width")[0] * 1e3, swp_ext.spike_feature("peak_v")[0] - swp_ext.spike_feature("trough_v")[0])
    return return_vals


def amp_and_spike_count(data_set, sweep, expt_set=None):
    spike_times = data_set.get_spike_times(sweep)
    start_t = 1.02
    end_t = 2.02

    if len(spike_times) == 0:
        n_spikes = 0
    else:
        n_spikes = len(spike_times[(spike_times >= start_t) & (spike_times <= end_t)])


    if expt_set is None:
        amp = data_set.get_sweep_metadata(sweep)["aibs_stimulus_amplitude_pa"]
    else:
        amp = expt_set.get_sweep_metadata(sweep)["aibs_stimulus_amplitude_pa"]

    return int(np.round(amp)), n_spikes


def fi_curve_stats(data):
    fi_arr = np.array([(amp, data[amp]) for amp in sorted(data.keys())])
    spiking_sweeps = np.flatnonzero(fi_arr[:, 1])
    if len(spiking_sweeps) == 0:
        return fi_arr[:, 0].max() + 20, 0

    rheo_idx = spiking_sweeps[0]
    rheobase = fi_arr[rheo_idx, 0]

    x = fi_arr[rheo_idx:, 0].astype(np.float64)
    y = fi_arr[rheo_idx:, 1]
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y)[0]
    return rheobase, m


def collect_virtual_experiments(ids):
    sql = """
        select sp.id, wkf.storage_directory || wkf.filename from specimens sp
        join neuronal_models nm on nm.specimen_id = sp.id
        join neuronal_model_runs nmr on nmr.neuronal_model_id = nm.id
        join well_known_files wkf on wkf.attachable_id = nmr.id
        where sp.id = any(%s)
        and nm.neuronal_model_template_id = %s
    """

    ve_paths = {}
    for model_type, nmt_id in NEURONAL_MODEL_TEMPLATES.iteritems():
        results = lims_utils.query(sql, (list(ids), nmt_id))
        ve_paths[model_type] = dict(results)

    return ve_paths


def plot_comparison(df, xlabel="", order=None, xlim=None, zeroline=False, filename=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    sns.boxplot(data=df, color="dodgerblue", whis=np.inf, order=order, orient="h", ax=ax)
    sns.stripplot(data=df, jitter=False, alpha=0.5, color="0.3", orient="h", ax=ax, linewidth=0)
    for i, r in df.iterrows():
        if order:
            vals = r[order].values
        else:
            vals = r.values
        plt.plot(vals, range(len(vals)), c="lightgray", linewidth=0.5)
    ax.set(xlabel=xlabel, ylabel="model type")
    if xlim:
        ax.set_xlim(xlim)
    if zeroline:
        ax.plot([0, 0], ax.get_ylim(), ":", c="k", zorder=-1)
    ax.set_yticklabels(LABELS)
    sns.despine()
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight")
#     plt.show()
    plt.close()

def main():
    sns.set_style("white")

    glif_bp_df = pd.read_csv("glif_bp.csv")
    glif_baa_df = pd.read_csv("glif_baa.csv")
    bp_baa_df = pd.read_csv("bp_baa.csv")

    glif_bp_ids = glif_bp_df["specimen__id"].values
    glif_baa_ids = glif_baa_df["specimen__id"].values
    bp_baa_ids = bp_baa_df["specimen__id"].values

    all_ids = np.unique(np.concatenate((glif_bp_ids, glif_baa_ids, bp_baa_ids)))

    plot_comparison(DataFrame(collect_exp_var(all_ids)), order=BASE_ORDER,
                    xlabel="temporal explained variance ratio", xlim=(0, 1), filename="exp_var.png")
    ve_paths = collect_virtual_experiments(all_ids)
    compare_ap_dims(ve_paths, bp_baa_ids)
    compare_ramp_latencies(ve_paths)
    compare_fi_curves(ve_paths)

if __name__ == "__main__":
    main()
