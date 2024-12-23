# sined_control_mechanisms

### To start a simulation:

```Powershell
$DIR_TO_PROJECT$> python src\\run_simulation.py
```

If socket address create problems, change the port number in 'run_simulations.py'.

### To print results:

```Powershell
$DIR_TO_PROJECT$> python src\\plot_results.py "outputs\\[sim_dir]" [day_to_plot] [day_to_end_plot]
```

### To compare multiple results:

```Powershell
$DIR_TO_PROJECT$> python src\\plot_results_compare.py "[sim_1]" "[sim_2]" ...
python src\\plot_results_compare.py "01_none_2m" "11_tariff_1loo" "12_tariff_5loo" "13_tariff_10loo" "14_tariff_20loo" "21_limits_1loo" "22_limits_5loo" "23_limits_10loo" "31_peak_1loo" "41_cond_1loo"
```