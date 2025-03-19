import os
import pandas as pd
import matplotlib.pyplot as plt

results_dir = "results"
models = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

plt.figure(figsize=(10, 6))

for model in models:
    loss_file = os.path.join(results_dir, model, "loss_history.csv")

    if os.path.exists(loss_file):
        df = pd.read_csv(loss_file)
        print(f"Columns in {model} loss file: {df.columns.tolist()}")  # Debugging line

        if "Loss" in df.columns:
            epochs = range(1, len(df) + 1)
            plt.plot(epochs, df["Loss"], label=model)
        else:
            print(f"Warning: 'loss' column not found in {loss_file}")

plt.xlabel("Epoch")
plt.ylabel("Overall Loss")
plt.title("Training Loss for Different Models")
plt.legend()
plt.grid()
plt.savefig(os.path.join(results_dir, "overall_loss_plot.png"))
plt.show()