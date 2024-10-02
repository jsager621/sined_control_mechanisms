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