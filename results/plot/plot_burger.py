import pandas as pd
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
    input_string = sys.argv[1]

    df_1 = pd.read_csv("../csv_files/plot_files/burger_solution_fo.csv")
    df_2 = pd.read_csv("../csv_files/plot_files/solution_burger_pc.csv")
    df_3 = pd.read_csv("../csv_files/plot_files/solution_burger_pc_MP.csv")

    if input_string == "gauss":
        plt.plot(df_1["x"], df_1["u_initial"], linestyle="--", label="Initial")
        plt.plot(df_1["x"], df_1["u_final"], label="Final const")
        plt.plot(df_2["x"], df_2["u_final"], label="Final tvd")
        plt.plot(df_3["x"], df_3["u_final"], label="Final weno")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title("Burger gauss")
        plt.legend()
        plt.savefig("solution_burger_gauss.pdf")
    elif input_string == "sin":
        plt.plot(df_3["x"], df_3["u_initial"], linestyle="--", label="Initial")
        # plt.plot(df_1["x"], df_1["u_final"], label="Final const")
        # plt.plot(df_2["x"], df_2["u_final"], label="Final tvd")
        plt.plot(df_3["x"], df_3["u_final"], marker='.', linestyle='None', label="Final weno")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title("Burger sin")
        plt.legend()
        plt.savefig("solution_burger_sin.pdf")
    else:
        plt.plot(df_2["x"], df_2["u_initial"], linestyle="--", label="Initial")
        # plt.plot(df_1["x"], df_1["u_final"], label="Final const")
        plt.plot(df_2["x"], df_2["u_final"], label="Final pc")
        # plt.plot(df_3["x"], df_3["u_final"], label="Final weno")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title("Burger square")
        plt.legend()
        plt.savefig("solution_burger_square.pdf")       
