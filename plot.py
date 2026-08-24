import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("runs/p0_bidir/eval.csv")
plt.plot(df.global_step, df.A2B, label="A→B")
plt.plot(df.global_step, df.B2A, label="B→A")
plt.plot(df.global_step, df["mean"], "k--", label="mean")
plt.xlabel("environment steps"); plt.ylabel("held-out success rate")
plt.legend(); plt.savefig("phase0_curve.png", dpi=150)