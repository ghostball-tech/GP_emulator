# GP_emulator
Gaussian process emulation of agent based models.

Data files to be emulated are .csv tables, created from python pandas data frames
The data files follow the naming convection
<model>_<data source>_example<n>.csv
  
Index:

MODEL: SPOTWELD
canonical example from R SAVE package (Bayarri et al)
Model output : spotweld_model_example1.csv
Field data : spotweld_field_example1.csv
  response variable : "diameter"
  controll variables : "current" "load" "thickness"
  model parameter : "tuning"
 
MODEL: COUZIN
influential model of collective animal movement
Model output : couzin_ABM_example<n>.csv
Simulated field data : couzin_simfield_example<n>.csv
  response variables : "group_dir" "group_rho"
  control variables : "n" "max_steps" "field_width" "field_height" "field_depth"
  model parameters : "r_o" "r_a"

Notes for simulated field data:
  Example 1: control variables held constant for 10 iterations. n = 20, max_steps = 4000, field = (24,12,16)
  /swarms/couzin_wrapper_sim_field_data.py
  Example 2: n in {10,12,20} max_steps random, two different tank sizes
  /swarms/couzin_wrapper_sim_field_data2.py

Notes for ABM output
  Example 1: /swarms/couzin_wrapper_sim.py
  Example 2: /swarms/couzin_wrapper_sim2.py
  Example 3: /swarms/couzin_wrapper_sim3.py
