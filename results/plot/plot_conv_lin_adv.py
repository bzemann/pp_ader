import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../csv_files/convergence/conv_lin_adv_fo_ssp.csv")

N = df["mesh"]  
err_fo = df["err_fo"]
err_tvd = df["err_tvd"]
err_weno = df["err_weno"]

#plot all 
plt.loglog(N, err_fo,  'o-', label='L1 Error (Euler)')
plt.loglog(N, err_tvd, 's-', label='L1 Error (TVD)')
plt.loglog(N, err_weno, '^-', label='L1 Error (WENO)')

plt.xlabel("Number of mesh points (N)")
plt.ylabel("Error")
plt.title("Convergence Plot (Error vs. N)")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.savefig("convergence_plot_lin_adv_all.pdf")
plt.close()

#euler
plt.loglog(N, err_fo,  'o-', label='L1 Error (Euler)')

scale = err_fo.iloc[0] / (N.iloc[0].astype(float) ** -1)

N_ref = np.linspace(N.min(), N.max(), 100)
err_ref = scale * (N_ref ** -1)

plt.loglog(N_ref, err_ref, '--k', label='Reference slope ~ N^-1 (1st order)')
plt.xlabel("Number of mesh points (N)")
plt.ylabel("Error")
plt.title("Convergence Plot (Error vs. N)")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.savefig("convergence_plot_lin_adv_first_order.pdf")
plt.close()

#tvd
plt.loglog(N, err_tvd, 's-', label='L1 Error (TVD)')

scale = err_tvd.iloc[0] / (N.iloc[0].astype(float) ** -2)
 
N_ref = np.linspace(N.min(), N.max(), 100)
err_ref = scale * (N_ref ** -2)

plt.loglog(N_ref, err_ref, '--k', label='Reference slope ~ N^-2 (2rd order)')
plt.xlabel("Number of mesh points (N)")
plt.ylabel("Error")
plt.title("Convergence Plot (Error vs. N)")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.savefig("convergence_plot_lin_adv_tvd.pdf")
plt.close()

#weno
plt.loglog(N, err_weno, '^-', label='L1 Error (WENO)')

scale = err_weno.iloc[0] / (N.iloc[0].astype(float) ** -3)
 
_ref = np.linspace(N.min(), N.max(), 100)
err_ref = scale * (N_ref ** -3)

plt.loglog(N_ref, err_ref, '--k', label='Reference slope ~ N^-3 (3rd order)')
plt.xlabel("Number of mesh points (N)")
plt.ylabel("Error")
plt.title("Convergence Plot (Error vs. N)")
plt.grid(True, which="both", linestyle="--")
plt.legend()

plt.savefig("convergence_plot_lin_adv_weno.pdf")

plt.close()
