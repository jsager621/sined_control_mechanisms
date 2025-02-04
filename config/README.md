## Hints for config parameterization

### simulation

- start_time : simulation start time in format "YYYY-MM-DD HH:MM:SS"
- end_time : simulation end time (non-inclusive) in format "YYYY-MM-DD HH:MM:SS"
- seed : random seed for reproducibility of simulations (maybe keep value 10)

### prosumer

- electricity_price : 36.1 ct/kWh (06/24), per strom-report.com
- feedin_tariff : 8.11 ct/kWh for <10 kWp, 7.03 ct/kWh for 10+ kWp (since 2023), per Verbraucherzentrale
- baseload_kWh_year : 2105 (1 person), 3470 (2), 5411 kWh (3+), per destatis
- hp :

    - typically 30 to 40 kWh (el) per m2, for 150 m2 - 4500 to 6000 kWh (el), per thermondo.de
    - higher values range up to 96 kWh (el) per m2 - 14400 kWh (el) overall, per solarwatt.de

- pv : should be below 30; average before 2020 was 8.52 kWp per system; probably mostly 10-15 kWp now
- bss : average capacity 8.90 kWh, power 5.55 kW, per battery-charts.de
- ev : new cars range around 53 kWh in 2021, per Speichermonitoring resp. mobility-charts.de
- cs : typical charging powers include 3.7, 11, 22, or >50 kW - for future and current households a wallbox with 11 kW charging power seems reasonable

### grid

- control_type

    - "none"
        
        - MAX_NUM_LOOPS = 0
    
    - "tariff"
        
        - MAX_NUM_LOOPS >1, should be around 5, as congested time steps are continuously added to list of adjusted tariff steps
        - TARIFF_ADJ >0, small value of a couple cents sufficient (typically approx. 0.08 eur/kWh from grid charges, should be below)
        - TARIFF_ADJ_FEEDIN true/false, whether to apply adjustment for tariff values at schedule timesteps also to feedin tariff or just to the electricity consumption price (false for fixed feedin tariff)
    
    - "limits"
        
        - MAX_NUM_LOOPS >1, should be around 10, as congested time steps are continuously added to list of steps with limits and limits of already existent steps are continuosly decreased
        - P_MAX_INIT_kW >0, maximum power demand to start with
        - P_MAX_STEP_kW <0, value to continuously decrease maximum power demand if congestions persist for time steps
        - P_MIN_INIT_kW <0, minimum power (generation) to start with
        - P_MIN_STEP_kW >0, value to continuously increase minimum power (gen) if congestions persist for time steps
    
    - "peak_price"
        
        - MAX_NUM_LOOPS = 1, one iteration maximum, as values only applied
        - PEAK_PRICE_DEM >0, daily price of peak demand, small value of a couple cents sufficient
        - PEAK_PRICE_GEN >0, daily price of peak generation, small value of a couple cents sufficient
        - PEAK_PRICE_SEND_ALWAYS true/false, whether a signal should be sent no matter if there is a congestion or not
    
    - "conditional_power"
        
        - MAX_NUM_LOOPS = 1, one iteration maximum, as values only applied
        - COND_POWER_THRESHOLD_kW >0, value above which extra costs apply
        - COND_POWER_ADD_COSTS >0, extra costs for power above threshold
        - COND_POWER_SEND_ALWAYS true/false, whether a signal should be sent no matter if there is a congestion or not

- GRID = "kerber_dorfnetz", other [pandapower grids](https://pandapower.readthedocs.io/en/latest/networks.html) possible
- NUM_HOUSEHOLDS = 57 for "kerber_dorfnetz"
- NUM_PARTICIPANTS <= NUM_HOUSEHOLDS as only a ratio of households can participate
- R_PV <= 1: Ratio of households to have PV system
- R_BSS <= 1: Ratio of households with PV to have BSS
- R_EV <= 1: Ratio of households with an EV (and CS)
- R_HP <= 1: Ratio of households with a HP
