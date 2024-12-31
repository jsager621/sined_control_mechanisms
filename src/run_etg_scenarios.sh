#!/bin/bash
this_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for ctrl_type in none tariff limits peak_price conditional_power
do
    date +"%F %T"
    echo starting $ctrl_type

    python $this_dir/run_etg_simulation.py $ctrl_type 1 1
    python $this_dir/run_etg_simulation.py $ctrl_type 1 0.66
    python $this_dir/run_etg_simulation.py $ctrl_type 1 0.33

    python $this_dir/run_etg_simulation.py $ctrl_type 0.66 1
    python $this_dir/run_etg_simulation.py $ctrl_type 0.66 0.66
    python $this_dir/run_etg_simulation.py $ctrl_type 0.66 0.33

    python $this_dir/run_etg_simulation.py $ctrl_type 0.33 1
    python $this_dir/run_etg_simulation.py $ctrl_type 0.33 0.66
    python $this_dir/run_etg_simulation.py $ctrl_type 0.33 0.33

    date +"%F %T"
    echo finished $ctrl_type
done
