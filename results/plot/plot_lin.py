import pandas as pd
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
    input_string = sys.argv[1]

    df_1 = pd.read_csv("../csv_files/plot_files/solution_lin_adv_ader_3.csv")
    # df_2 = pd.read_csv("../csv_files/plot_files/solution_tvd.csv")
    df_3 = pd.read_csv("../csv_files/plot_files/solution_lin_adv_ader_3.csv")

    if input_string == "gauss":
        plt.plot(df_1["x"], df_1["u_initial"], linestyle="--", label="Initial")
        plt.plot(df_1["x"], df_1["u_final"], label="Final const")
        plt.plot(df_2["x"], df_2["u_final"], label="Final tvd")
        plt.plot(df_3["x"], df_3["u_final"], label="Final weno")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title("Linear advection gauss")
        plt.legend()
        plt.savefig("solution_lin_adv_gauss.pdf")
    elif input_string == "sin":
        plt.plot(df_3["x"], df_3["u_initial"], linestyle="--", label="Initial")
        plt.plot(df_1["x"], df_1["u_analytical"], label="Analytical")
        # plt.plot(df_1["x"], df_1["u_final"], label="Final const")
        # plt.plot(df_2["x"], df_2["u_final"], label="Final tvd")
        plt.plot(df_3["x"], df_3["u_final"], label="Final ADER third order")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title("Linear advection sin")
        plt.legend()
        plt.savefig("solution_lin_adv_sin.pdf")
    else:
        plt.plot(df_1["x"], df_1["u_initial"], linestyle="--", label="Initial")
        plt.plot(df_1["x"], df_1["u_final"], label="Final const")
        plt.plot(df_2["x"], df_2["u_final"], label="Final tvd")
        plt.plot(df_3["x"], df_3["u_final"], label="Final weno")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title("Linear advection square")
        plt.legend()
        plt.savefig("solution_lin_adv_square.pdf")       
