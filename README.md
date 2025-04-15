# Simulative setup for the forecast-based assessment of incentivation and control mechanisms for prosumer households

This repository offers a setup to investigate various incentivation and control mechanisms for prosumer households in the low-voltage grid.
This can be done with regard to the mitigation of grid congestions.
Some of the implemented mechanisms rely on an iterative procedure for scheduling days, like the adaption of variable tariffs based on congestions in specific timesteps of the upcoming day.

For an enhanced overview of how this repository is set up and can be used, we refer to the publication that is listed under [Cite](#Cite).

## Usage

First, you may need to install Python and the Code (preferably using a Git program).
Then, you have to set up the code, which can be done by cloning the repository to your local workspace.
Afterwards, set up your IDE (e.g. Visual Studio Code) by (creating a virtual environment ```python -m venv .venv``` and) installing the requirements ```pip install -r requirements.txt```.

Finally, run the setup by using the following instructions and possibly adapting the configurations (```/config/```) of the simulation.
For that, see [the config readme](https://github.com/jsager621/sined_control_mechanisms/tree/main/config).

### To start a simulation:

```Powershell
$DIR_TO_PROJECT$> python src\\run_simulation.py
```

If socket address create problems, change the port number in 'run_simulations.py'.

### To print results for a specific simulation run:

```Powershell
$DIR_TO_PROJECT$> python src\\plot_results.py "outputs\\[sim_dir]" [day_to_plot] [day_to_end_plot]
```

### To compare multiple results:

```Powershell
$DIR_TO_PROJECT$> python src\\plot_results_compare.py "[sim_1]" "[sim_2]" ...
python src\\plot_results_compare.py "01_none_2m" "11_tariff_1loo" "12_tariff_5loo" "13_tariff_10loo" "14_tariff_20loo" "21_limits_1loo" "22_limits_5loo" "23_limits_10loo" "31_peak_1loo" "41_cond_1loo"
```

## Authors

Jens Sager, Carl von Ossietzky Universität, Oldenburg
Carsten Wegkamp, elenia Institute for High Voltage Technology and Power Systems, Technical University Braunschweig

## Cite

If you want to cite this repository, you can use:

(Manuscript accepted for publication) C. Wegkamp, J. Sager, B. Engel, A. Niesse: Simulative Investigation of Control Mechanisms for Grid-Serving Utilization of Prosumer Household Flexibility. Proceedings der IEWT, Vienna, Austria, 2025

## Acknowledgements

The research project ’SiNED – Ancillary Services for Reliable Power Grids in Times of Progressive German Energiewende and Digital Transformation’ acknowledges the support of the Lower Saxony Ministry of Science and Culture through the ’Niedersächsisches Vorab’ grant programme (grant ZN3563) and of the Energy Research Centre of Lower Saxony (EFZN).
The responsibility for the content of this repository lies with the authors and does not necessarily reflect the opinion of the project consortium 'SiNED'.
