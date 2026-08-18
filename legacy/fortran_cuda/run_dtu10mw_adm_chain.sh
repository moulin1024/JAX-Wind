#!/usr/bin/env bash
set -euo pipefail

export WIRELES_TURBINE_MODEL=admnr
export WIRELES_USE_NVTX=OFF
export OMPI_COMM_WORLD_LOCAL_RANK=0

run_stage() {
    local case_name="$1"
    local case_dir="job/${case_name}"

    printf '%s\n' "$(date --iso-8601=seconds) preparing ${case_name}" | tee -a job/dtu10mw_adm_chain.log
    python3 prc/wireles.py pre "${case_name}" > "${case_dir}/pre.log" 2>&1
    python3 prc/wireles.py build "${case_name}" > "${case_dir}/build.log" 2>&1
    printf '%s\n' "$(date --iso-8601=seconds) running ${case_name}" | tee -a job/dtu10mw_adm_chain.log
    python3 prc/wireles.py run "${case_name}" > "${case_dir}/run.log" 2>&1
    printf '%s\n' "$(date --iso-8601=seconds) completed ${case_name}" | tee -a job/dtu10mw_adm_chain.log
}

run_stage dtu10mw_128_warmup_10h
run_stage dtu10mw_128_precursor_1h
run_stage dtu10mw_128_main_adm_1h
