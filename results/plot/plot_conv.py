import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../csv_files/convergence/conv_lin_adv_ader3_balsara.csv")

N = df["mesh"]  
err = df["err"]
plt.loglog(N, err, 's-', label='L1 Error (ADER 2nd Order MP PC)')

scale = err.iloc[0] / (N.iloc[0].astype(float) ** -3)
 
N_ref = np.linspace(N.min(), N.max(), 100)
err_ref = scale * (N_ref ** -3)

plt.loglog(N_ref, err_ref, '--r', label='Reference slope ~ N^-3 (3rd order)')

scale = err.iloc[0] / (N.iloc[0].astype(float) ** -1)

N_ref = np.linspace(N.min(), N.max(), 100)
err_ref = scale * (N_ref ** -1)

plt.loglog(N_ref, err_ref, '--k', label='Reference slope ~ N^-1 (1rd order)')
plt.xlabel("Number of mesh points (N)")
plt.ylabel("Error")
plt.title("Convergence Plot (Error vs. N)")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.savefig("convergence_plot_lin_adv_ader3.pdf")
plt.close()
