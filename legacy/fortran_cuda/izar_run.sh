casename=$1
module load nvhpc
python prc/wl.py clean $casename
python prc/wl.py pre $casename
python prc/wl.py solve $casename

