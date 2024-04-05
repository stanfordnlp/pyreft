from plotnine import (
    ggplot, aes, geom_point, facet_wrap, scale_x_log10, theme, element_text,
    geom_text, theme_set, theme_gray, element_blank, facet_grid, stat_smooth
)
from pandas import DataFrame
import json

theme_set(theme_gray(base_family="Inter"))

# LAYER

with open("data/elapse_per_layer.json") as f:
    per_layer = json.load(f)

data = []
for layer, stats in per_layer.items():
    l = list(layer.split(';'))
    if len(l) == 1:
        l = l[0]
    else:
        l = f"{l[0]}-{l[-1]}"
    print(l)
    for stat in stats:
        if stat[3] == "loreft": stat[3] = "LoReFT"
        else: stat[3] = "No intervention"
        data.append({
            "layer": l, "time": stat[0], "prompt_toks": stat[1],
            "total_toks": stat[2], "type": stat[3],
            "generated_toks": stat[2] - stat[1]
        })

df = DataFrame(data)
df = df[df["generated_toks"] == 256].reset_index(drop=True)
print(df)

plot = (
    ggplot(df, aes(x="prompt_toks", y="time", color="type")) + facet_wrap("~layer", nrow=1) +
    geom_point(alpha=0.5, stroke=0) + stat_smooth(method="lm") + theme(axis_text_x=element_text(angle=45, hjust=0.75))
)
plot.save("plot_layer.svg", width=10, height=2)

# POS

with open("data/elapse_per_position.json") as f:
    per_layer = json.load(f)

data = []
for position, stats in per_layer.items():
    for stat in stats:
        if stat[3] == "loreft": stat[3] = "LoReFT"
        else: stat[3] = "No intervention"
        data.append({
            "position": int(position), "time": stat[0], "prompt_toks": stat[1],
            "total_toks": stat[2], "type": stat[3],
            "generated_toks": stat[2] - stat[1]
        })

df = DataFrame(data)
df = df[df["generated_toks"] == 256].reset_index(drop=True)
print(df)

plot = (
    ggplot(df, aes(x="prompt_toks", y="time", color="type")) + facet_wrap("~position", nrow=1) +
    geom_point(alpha=0.5, stroke=0) + stat_smooth(method="lm") + theme(axis_text_x=element_text(angle=45, hjust=0.75))
)
plot.save("plot_position.svg", width=10, height=2)

# RANK

with open("data/elapse_per_rank.json") as f:
    per_layer = json.load(f)

data = []
for rank, stats in per_layer.items():
    for stat in stats:
        if stat[3] == "loreft": stat[3] = "LoReFT"
        else: stat[3] = "No intervention"
        data.append({
            "rank": int(rank), "time": stat[0], "prompt_toks": stat[1],
            "total_toks": stat[2], "type": stat[3],
            "generated_toks": stat[2] - stat[1]
        })

df = DataFrame(data)
df = df[df["generated_toks"] == 256].reset_index(drop=True)
print(df)

plot = (
    ggplot(df, aes(x="prompt_toks", y="time", color="type")) + facet_wrap("~rank", nrow=1) +
    geom_point(alpha=0.5, stroke=0) + stat_smooth(method="lm") + theme(axis_text_x=element_text(angle=45, hjust=0.75))
)
plot.save("plot_rank.svg", width=10, height=2)