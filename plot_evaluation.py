import matplotlib.pyplot as plt

# Replace these with your actual evaluation results
bleu_score = 31.03
chrf_score = 66.04

# Combine metrics
metrics = {
    "BLEU Score": bleu_score,
    "chrF Score": chrf_score
}

# Plot
plt.figure(figsize=(6, 4))
bars = plt.bar(metrics.keys(), metrics.values(),
               color=["skyblue", "lightgreen"], width=0.6)

# Add value labels above bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1,
             f"{height:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.title("Evaluation Metrics for mBART-50 Multilingual Translator", fontsize=12, fontweight="bold")
plt.ylabel("Score (%)")
plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Save and show the graph
plt.tight_layout()
plt.savefig("evaluation_metrics.png")
plt.show()
