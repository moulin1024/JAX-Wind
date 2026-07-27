# This is a example script to run the simulation of a single NREL 5MW turbine in larminar inflow
# Because no precursor simulation is available, the wake of the turbine will reente the domain.
# The aeroelasticity is enabled by default.

python prc/wl.py create Single-NREL5MW
cp -r example/Single-NREL5MW/input job/Single-NREL5MW
./local_run.sh Single-NREL5MW
python prc/wl.py anime Single-NREL5MW
python animation.py Single-NREL5MW 300