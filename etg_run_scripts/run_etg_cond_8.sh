#!/bin/bash
this_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
this_dir=$this_dir/../src
cond=8
port_offset=3

# other mechanism runs
for ctrl_type in conditional_power
do
    date +"%F %T"
    echo starting $ctrl_type

    python $this_dir/run_etg_simulation.py $ctrl_type 1 1 $cond $port_offset
    python $this_dir/run_etg_simulation.py $ctrl_type 1 0.66 $cond $port_offset
    python $this_dir/run_etg_simulation.py $ctrl_type 1 0.33 $cond $port_offset

    python $this_dir/run_etg_simulation.py $ctrl_type 0.66 1 $cond $port_offset
    python $this_dir/run_etg_simulation.py $ctrl_type 0.66 0.66 $cond $port_offset
    python $this_dir/run_etg_simulation.py $ctrl_type 0.66 0.33 $cond $port_offset

    python $this_dir/run_etg_simulation.py $ctrl_type 0.33 1 $cond $port_offset
    python $this_dir/run_etg_simulation.py $ctrl_type 0.33 0.66 $cond $port_offset
    python $this_dir/run_etg_simulation.py $ctrl_type 0.33 0.33 $cond $port_offset
 
    date +"%F %T"
    echo finished $ctrl_type
done