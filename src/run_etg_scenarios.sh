# python call: 
# python run_etg_simulation.py <ctrl_type> <r_pv> <r_bss>
this_dir=dirname "$(realpath $0)"

for ctrl_type in none tariff limits peak_price conditional_power
do
    python $this_dir/run_etg_simulation.py $ctrl_type 1 1
    python $this_dir/run_etg_simulation.py $ctrl_type 1 0.66
    python $this_dir/run_etg_simulation.py $ctrl_type 1 0.33

    python $this_dir/run_etg_simulation.py $ctrl_type 0.66 1
    python $this_dir/run_etg_simulation.py $ctrl_type 0.66 0.66
    python $this_dir/run_etg_simulation.py $ctrl_type 0.66 0.33

    python $this_dir/run_etg_simulation.py $ctrl_type 0.33 1
    python $this_dir/run_etg_simulation.py $ctrl_type 0.33 0.66
    python $this_dir/run_etg_simulation.py $ctrl_type 0.33 0.33
done
