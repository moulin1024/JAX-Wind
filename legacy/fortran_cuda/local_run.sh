
casename=$1
python prc/wl.py clean $casename
python prc/wl.py pre $casename
python prc/wl.py build $casename
python prc/wl.py run $casename

