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
from allensdk.ephys.feature_extractor import EphysFeatureExtractor

#ctc = CellTypesCache(manifest_file='cell_types/cell_types_manifest.json')
from allensdk.api.queries.cell_types_api import CellTypesApi

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
    "glif_1",
    "glif_2",
    "glif_3",
    "glif_4",
    "glif_5",    
    "bp",
    "baa",
]

LABELS = [
    "GLIF level 1",
    "GLIF level 2",
    "GLIF level 3",
    "GLIF level 4",
    "GLIF level 5",
    "Biophys perisomatic",
    "Biophys all active",
]

VE_STIMULI = {
    "noise_1": "C1NSSEED_1",
    "noise_2": "C1NSSEED_2",
}

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
    exp_var_data = DataFrame(exp_var_data)  
    exp_var_data = exp_var_data.ix[:, BASE_ORDER]

    return exp_var_data


def compare_fi_curves(ve_paths, file_prefix):
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

    plot_comparison(df.ix[:, slope_cols] * 100., "fI slope (% diff)", order=slope_cols, zeroline=True, filename="fi_slope.png", file_prefix=file_prefix)
    plot_comparison(df.ix[:, slope_cols] * 100., "fI slope (% diff)", order=slope_cols, xlim=(-100, 100), zeroline=True, filename="fi_slope_zoomed.png", file_prefix=file_prefix)
    plot_comparison(df.ix[:, rheo_cols] * 100., "rheobase (% diff)", order=rheo_cols, zeroline=True, filename="fi_rheo.png", file_prefix=file_prefix)
    plot_comparison(df.ix[:, rheo_cols] * 100., "rheobase (% diff)", order=rheo_cols, xlim=(-100, 100), zeroline=True, filename="fi_rheo_zoomed.png", file_prefix=file_prefix)


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


def compare_ramp_latencies(ve_paths, file_prefix):
    ve_df = DataFrame(ve_paths)

    input_list = ve_df.reset_index().to_dict("records")
    pool = Pool()
    data_list = pool.map(compare_ramp_latency, input_list)
    df = DataFrame(data_list)
    for model_type in NEURONAL_MODEL_TEMPLATES:
        df[model_type + "_latency_pct_diff"] = (df[model_type + "_ramp_latency"] - df["expt_ramp_latency"]) / df["expt_ramp_latency"]
    cols = [m + "_latency_pct_diff" for m in BASE_ORDER]
    plot_comparison(df.ix[:, cols] * 100., "ramp latency (% diff)", order=cols, zeroline=True, filename="ramp_latency.png", file_prefix=file_prefix)
    plot_comparison(df.ix[:, cols] * 100., "ramp latency (% diff)", order=cols, zeroline=True, xlim=(-100, 100), filename="ramp_latency_zoomed.png", file_prefix=file_prefix)


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


def compare_ap_dims(ve_paths, biophys_ids, file_prefix):
    ve_df = DataFrame(ve_paths)
    ve_df = ve_df.ix[biophys_ids, :]
    input_list = ve_df.reset_index().to_dict("records")
    pool = Pool()
    data_list = pool.map(compare_ap_dim, input_list)
    df = DataFrame(data_list)
    biophys_models = ["baa", "bp"]
    for model_type in biophys_models:
        df[model_type + "_width_pct_diff"] = (df[model_type + "_width"] - df["expt_width"]) / df["expt_width"]
        df[model_type + "_height_pct_diff"] = (df[model_type + "_height"] - df["expt_height"]) / df["expt_height"]
    width_cols = [m + "_width_pct_diff" for m in biophys_models]
    height_cols = [m + "_height_pct_diff" for m in biophys_models]
    plot_comparison(df.ix[:, width_cols] * 100., "AP width (% diff)", ylabels = ["Biophys perisomatic", "Biophys all active"], order=width_cols, zeroline=True, filename="ap_width.png", file_prefix=file_prefix)
    plot_comparison(df.ix[:, height_cols] * 100., "AP height (% diff)", ylabels = ["Biophys perisomatic", "Biophys all active"], order=height_cols, zeroline=True, filename="ap_height.png", file_prefix=file_prefix)


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




def expt_tau(specimen_id):
    #print(chr(27) + "[2J") # To clear terminal screen    
    #print "START EXPT_TAU" + str(specimen_id)
    expt_taus = []
    data_set = expt_data_set(specimen_id)
    long_square_sweeps = lims_utils.get_sweeps_of_type("C1LSCOARSE", specimen_id, passed_only=True)
    #print "expt specimen id= " + str(specimen_id)
    for sweep in long_square_sweeps:
        print str(specimen_id) + " expt_sweep_number: " + str(sweep)
        if (data_set.get_sweep_metadata(sweep)["aibs_stimulus_amplitude_pa"] < 0):
            v, i, t = lims_utils.get_sweep_v_i_t_from_set(data_set, sweep)
            sweep_feat = EphysSweepFeatureExtractor(t, v) # Get time and voltage of each hyperpolarizing sweep
            try: 
                sweep_feat.estimate_time_constant()  # Try statement included because estimate_time_constant errors on some sweeps ("could not find interval for time constant estimate")          
            except:
                continue
            else:
                expt_taus.append(sweep_feat.estimate_time_constant()) # Append time constant of each sweep to list
    mean_expt_tau = np.nanmean(expt_taus) # Mean time constant for this cell
    #print "mean_expt_tau= " + str(mean_expt_tau)
    return mean_expt_tau

def ve_tau(specimen_id, ve_path):
    #print(chr(27) + "[2J") # To clear terminal screen 
    print "START VE_TAU " + str(specimen_id) + " " + str(ve_path)
    expt_taus = []
    data_set = NwbDataSet(ve_path)
    long_square_sweeps = lims_utils.get_sweeps_of_type("C1LSCOARSE", specimen_id, passed_only=True)
    print "ve specimen id= " + str(specimen_id)
    for sweep in long_square_sweeps: 
        #print "ve_sweep_number: " + str(sweep)
        #print(data_set.get_sweep_metadata(sweep))
        try: 
            (data_set.get_sweep_metadata(sweep)["aibs_stimulus_amplitude_pa"])
        except:
            continue
        else:
            if (data_set.get_sweep_metadata(sweep)["aibs_stimulus_amplitude_pa"] < 0):
                v, i, t = lims_utils.get_sweep_v_i_t_from_set(data_set, sweep)
                sweep_feat = EphysSweepFeatureExtractor(t, v) # Get time and voltage of each hyperpolarizing sweep
                if np.isnan(sweep_feat):
                    continue
                else:
                    expt_taus.append(sweep_feat.estimate_time_constant()) # Append time constant of each sweep to list
    mean_expt_tau = np.nanmean(expt_taus) # Mean time constant for this cell
    print "mean_ve_tau= " + str(mean_expt_tau)
    return mean_expt_tau

def compare_taus(ve_paths, file_prefix):
    print "START COMPARE_TAUS"    
    ve_df = DataFrame(ve_paths)
    
    input_list = ve_df.reset_index().to_dict("records")
    pool = Pool()
    data_list = pool.map(compare_tau, input_list) ###### ERROR HERE. 
    df = DataFrame(data_list)

    for model_type in NEURONAL_MODEL_TEMPLATES:
        df[model_type + "_tau_diff"] = (df[model_type + "_tau"] - df["expt_tau"]) / df["expt_tau"]
    tau_cols = [m + "_tau_diff" for m in BASE_ORDER]
    plot_comparison(df.ix[:, tau_cols] * 100., "Time Constant (% diff)", order=tau_cols, zeroline=True, filename="tau.png", file_prefix=file_prefix)


def compare_tau(input_dict):
    print "START COMPARE_TAU " 
    specimen_key = "index"
    specimen_id = input_dict[specimen_key]
    mean_expt_tau = expt_tau(specimen_id)

    info = {
        "tau": mean_expt_tau
    }
    for model_type in input_dict:
        print str(specimen_id) + " model type =" + str(model_type)
        if model_type == specimen_key:
            continue
        if type(input_dict[model_type]) != str:
            continue
        if not os.path.exists(input_dict[model_type]):
            continue
        #if np.isnan(input_dict[model_type]) == True:
         #   continue
        ve_path = input_dict[model_type]
        info[model_type + "_tau"] = ve_tau(specimen_id, ve_path)
    return info




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


def plot_comparison(df, xlabel="", ylabels="", order=None, xlim=None, zeroline=False, filename=str(), file_prefix=str()):
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
    if ylabels:
        ax.set_yticklabels(ylabels)
    else:
        ax.set_yticklabels(LABELS)
    sns.despine()
    plt.tight_layout()
    if filename:
        plt.savefig(str(file_prefix + "_" + filename), bbox_inches="tight")
    #plt.show()
    plt.close()

# Find cell ids that have data for all model types
def find_cells_with_all_models(ids):
    cells_with_all_models = []    
    
    e = DataFrame(collect_virtual_experiments(ids))
    for i in ids: 
        if all(n == True for n in e.loc[i].notnull()): # Select only complete rows (have all models)
            cells_with_all_models.append(i) # Make list of cell ids that have all models, to be used later
    
    return cells_with_all_models

   

# Make example plot showing real noise stimulus response for each cell and its modeled responses
def noise1_response_comparison(ids, ve_paths, file_prefix):
    filename='Noise2'
    
    ids1 = find_cells_with_all_models(ids)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for specimen_id in ids1:
        data_set = expt_data_set(specimen_id)
        exp_sweeps = lims_utils.get_sweeps_of_type("C1NSSEED_2", specimen_id, passed_only=True)
        
        if len(exp_sweeps) == 0:
            continue
    
        fig, axes = plt.subplots(3, 1, sharex=True)
        plt.subplots_adjust(hspace=0.1, wspace=0.1)
        axes[2].set_xlim([0, 25])
        sweep_data = data_set.get_sweep(exp_sweeps[0])
        spike_times = data_set.get_spike_times(exp_sweeps[0])
        index_range = sweep_data["index_range"]
        i = sweep_data["stimulus"][0:index_range[1]+1] # in A
        v = sweep_data["response"][0:index_range[1]+1] # in V
        i *= 1e12 # to pA
        v *= 1e3 # to mV
        sampling_rate = sweep_data["sampling_rate"] # in Hz
        t = np.arange(0, len(v)) * (1.0 / sampling_rate)

        axes[1].plot(t, v, color='k', linewidth=0.3)
        axes[0].scatter(spike_times, [10]*len(spike_times), color='k', s=100, marker="|", linewidth=0.3)
        axes[2].plot(t, i, color='k', linewidth=0.3)
        axes[1].set_ylabel("mV")
        axes[2].set_ylabel("pA")
        axes[2].set_xlabel("seconds")   
    
        for model, label, color in zip(BASE_ORDER, LABELS, colors):
            ve_path = ve_paths[model][specimen_id]
            data_set = NwbDataSet(ve_path)
            exp_sweeps = lims_utils.get_sweeps_of_type("C1NSSEED_2", specimen_id, passed_only=True)
            sweep_data = data_set.get_sweep(exp_sweeps[0])
            spike_times = data_set.get_spike_times(exp_sweeps[0])
            index_range = sweep_data["index_range"]
            i = sweep_data["stimulus"][0:index_range[1]+1] # in A
            v = sweep_data["response"][0:index_range[1]+1] # in V
            i *= 1e12 # to pA
            v *= 1e3 # to mV
            sampling_rate = sweep_data["sampling_rate"] # in Hz
            t = np.arange(0, len(v)) * (1.0 / sampling_rate)
            axes[0].scatter(spike_times, [80-(10*BASE_ORDER.index(model))]*len(spike_times), color='k', s=100, marker="|", linewidth=0.3)
        axes[0].set_yticklabels(["",'Experimental Data', LABELS[6], LABELS[5], LABELS[4], LABELS[3], LABELS[2], LABELS[1], LABELS[0]])
        axes[0].set_ylabel("model type")
        axes[0].set_title("Spike Times")

        plt.subplots_adjust(hspace=0.1, wspace=0.1)
        plt.savefig(str(file_prefix + "_" + filename + "_" + str(specimen_id)), bbox_inches="tight")
        #plt.show()
        plt.close()


def main():
    sns.set_style("white")

    glif_bp_df = pd.read_csv("glif_bp.csv")
    glif_baa_df = pd.read_csv("glif_baa.csv")
    bp_baa_df = pd.read_csv("bp_baa.csv")
    
    #global dendrite_type    
    
    for dendrite_type in ["spiny", "aspiny"]:
        glif_bp_ids = (glif_bp_df[glif_bp_df["tag__dendrite_type"]==dendrite_type])["specimen__id"].values
        glif_baa_ids =  (glif_baa_df[glif_baa_df["tag__dendrite_type"]==dendrite_type])["specimen__id"].values
        bp_baa_ids =  (bp_baa_df[bp_baa_df["tag__dendrite_type"]==dendrite_type])["specimen__id"].values

        all_ids = np.unique(np.concatenate((glif_bp_ids, glif_baa_ids, bp_baa_ids)))

        #plot_comparison(collect_exp_var(all_ids), order=BASE_ORDER,
        #            xlabel="temporal explained variance ratio", xlim=(0, 1), filename="exp_var.png", file_prefix=dendrite_type)
        ve_paths = collect_virtual_experiments(all_ids)
        #noise1_response_comparison(all_ids, ve_paths, file_prefix=dendrite_type)
        #compare_ap_dims(ve_paths, bp_baa_ids, file_prefix=dendrite_type)  ## Still errors
        #compare_ramp_latencies(ve_paths, file_prefix=dendrite_type)
        #compare_fi_curves(ve_paths, file_prefix=dendrite_type)
        compare_taus(ve_paths, file_prefix=dendrite_type)
if __name__ == "__main__":
    main()
