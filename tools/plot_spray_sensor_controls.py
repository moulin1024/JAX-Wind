"""Plot all nine paired sensor predictions from assess_inertial_centre JSON."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('comparison',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    runs=json.loads(args.comparison.read_text())
    expected=np.array(runs[0]['experimental'])
    for run in runs:
        assert run['case']==runs[0]['case']
        np.testing.assert_allclose(run['experimental'],expected)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    sensors=np.arange(1,10)
    for q,(ax,label) in enumerate(zip(axes,['DBT (°C)','WBT (°C)','Enthalpy (kJ/kg dry air)'])):
        ax.plot(sensors,expected[q],'ko',label='Experiment')
        for run in runs:
            model=run['carrier_model']
            ax.plot(sensors,np.array(run['predicted'])[q],'o-',label=f"{model}, dt={run['dt']:g} s")
        ax.axvspan(4.75,5.25,color='gray',alpha=.12)
        ax.set(xlabel='Sensor (5 = centre)',ylabel=label,xticks=sensors)
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"Case {runs[0]['case']}: all nine sensors, {runs[0]['window_seconds'][0]:g}–{runs[0]['window_seconds'][1]:g} s means")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
