from plotnine import (
    ggplot, aes, geom_point, facet_wrap, scale_x_log10, theme, element_text,
    geom_text, theme_set, theme_gray, element_blank
)
from pandas import DataFrame
from mizani.formatters import percent_format

theme_set(theme_gray(base_family="Inter"))

MODEL_ORDER = ["LLaMA-7B", "LLaMA-13B", "Llama-2 7B", "RoBERTa-base", "RoBERTa-large"]
TASK_ORDER = ["Commonsense", "Arithmetic", "Instruct-tuning", "GLUE"]

stats = {
    "Commonsense": {
        "LLaMA-7B": [
            {"name": "PrefT", "params": 0.110, "score": 64.6},
            {"name": "AdapterS", "params": 0.990, "score": 70.8},
            {"name": "AdapterP", "params": 3.540, "score": 72.3},
            {"name": "LoRA", "params": 0.830, "score": 74.7},
            {"name": "DoRA (half)", "params": 0.430, "score": 77.5},
            {"name": "DoRA", "params": 0.840, "score": 78.1},
            {"name": "LoReFT", "params": 0.031, "score": 80.2}
        ],
        "LLaMA-13B": [
            {"name": "PrefT", "params": 0.03, "score": 68.4},
            {"name": "AdapterS", "params": 0.800, "score": 79.5},
            {"name": "AdapterP", "params": 2.890, "score": 81.5},
            {"name": "LoRA", "params": 0.670, "score": 80.5},
            {"name": "DoRA (half)", "params": 0.350, "score": 80.8},
            {"name": "DoRA", "params": 0.680, "score": 81.5},
            {"name": "LoReFT", "params": 0.025, "score": 83.3}
        ]
    },
    "Arithmetic": {
        "LLaMA-7B": [
            {"name": "PrefT", "params": 0.110, "score": 35.0},
            {"name": "AdapterS", "params": 0.990, "score": 44.6},
            {"name": "AdapterP", "params": 3.540, "score": 46.4},
            {"name": "LoRA", "params": 0.830, "score": 46.9},
            {"name": "LoReFT", "params": 0.031, "score": 42.6},
        ],
        "LLaMA-13B": [
            {"name": "PrefT", "params": 0.300, "score": 38.8},
            {"name": "AdapterS", "params": 0.800, "score": 48.9},
            {"name": "AdapterP", "params": 2.890, "score": 50.2},
            {"name": "LoRA", "params": 0.670, "score": 51.1},
            {"name": "LoReFT", "params": 0.025, "score": 49.6},
        ]
    },
    "Instruct-tuning": {
        "Llama-2 7B": [
            {"name": "FT", "params": 100.000, "score": 80.93},
            {"name": "LoRA", "params": 0.1245, "score": 81.48},
            {"name": "RED", "params": 0.0039, "score": 81.69},
            {"name": "LoReFT", "params": 0.0039, "score": 85.60}
        ]
    },
    "GLUE": {
        "RoBERTa-base": [
            {"name": "FT", "params": 100.000, "score": 85.6},
            {"name": "Adapter", "params": 0.318, "score": 85.0},
            {"name": "LoRA", "params": 0.239, "score": 84.7},
            {"name": "AdapterFNN", "params": 0.239, "score": 84.7},
            {"name": "BitFit", "params": 0.080, "score": 82.3},
            {"name": "RED", "params": 0.016, "score": 84.3},
            {"name": "LoReFT", "params": 0.015, "score": 84.2}
        ],
        "RoBERTa-large": [
            {"name": "FT", "params": 100.000, "score": 88.6},
            {"name": "Adapter", "params": 0.254, "score": 88.0},
            {"name": "LoRA", "params": 0.225, "score": 88.1},
            {"name": "AdapterFNN", "params": 0.225, "score": 87.7},
            {"name": "RED", "params": 0.014, "score": 88.0},
            {"name": "LoReFT", "params": 0.014, "score": 88.2}
        ]
    }
}
stats_flat = []
for task in stats:
    for model in stats[task]:
        for method in stats[task][model]:
            stats_flat.append({**method, "task": task, "model": model})

df = DataFrame(stats_flat)
df["params"] *= 0.01
df["color"] = ~df["name"].isin(["LoReFT"])
df["model"] = df["model"].astype("category")
df["model"].cat.set_categories(MODEL_ORDER, inplace=True)
df["task"] = df["task"].astype("category")
df["task"].cat.set_categories(TASK_ORDER, inplace=True)

# rename score to Score and params to Parameters
df = df.rename(columns={"score": "Score", "params": "Parameters"})

plot = (
    ggplot(df, aes(x="Parameters", y="Score", color="color")) +
    geom_point() + facet_wrap("~task+model", scales="free", nrow=2) +
    scale_x_log10(labels=percent_format()) +
    geom_text(aes(label="name"), size=7, adjust_text={"avoid_self": True}) +
    theme(legend_position="none", axis_text_x=element_text(angle=45, hjust=0.75), panel_spacing_x=0.4, panel_spacing_y=0.4,
          panel_grid_minor_x=element_blank(), panel_grid_minor_y=element_blank(), axis_text=element_text(size=7),
          strip_text=element_text(weight="bold"))
)
plot.save("plot.svg", width=9, height=4, dpi=300)