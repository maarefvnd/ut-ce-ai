import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.rcParams.update({"font.size": 16, "axes.labelweight": "bold"})


def plot_pokemon(x, y, y_hat=None, x_range=[10, 130], y_range=[10, 130], dx=20, dy=20):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers", marker=dict(size=10), name="data")
    )
    if y_hat is not None:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_hat,
                line_color="red",
                mode="lines",
                line=dict(width=3),
                name="Fitted line",
            )
        )
        width = 550
        title_x = 0.46
    else:
        width = 500
        title_x = 0.5
    fig.update_layout(
        width=width,
        height=500,
        title="Pokemon stats",
        title_x=title_x,
        title_y=0.93,
        xaxis_title="defense",
        yaxis_title="attack",
        margin=dict(t=60),
    )
    fig.update_xaxes(range=x_range, tick0=x_range[0], dtick=dx)
    fig.update_yaxes(range=y_range, tick0=y_range[0], dtick=dy)
    return fig


def plot_grid_search(
    x,
    y,
    slopes,
    loss_function,
    title="Mean Squared Error",
    y_range=[0, 2500],
    y_title="MSE",
):
    mse = []
    df = pd.DataFrame()
    for m in slopes:
        df[f"{m:.2f}"] = m * x  # store predictions for plotting later
        mse.append(loss_function(y, m * x))  # calc MSE
    mse = pd.DataFrame({"slope": slopes, "squared_error": mse})
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Data & Fitted Line", title))
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers", marker=dict(size=10), name="Data"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df.iloc[:, 0],
            line_color="red",
            mode="lines",
            line=dict(width=3),
            name="Fitted line",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=mse["slope"],
            y=mse["squared_error"],
            mode="markers",
            marker=dict(size=7),
            name="MSE",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=mse.iloc[[0]]["slope"],
            y=mse.iloc[[0]]["squared_error"],
            line_color="red",
            mode="markers",
            marker=dict(size=14, line=dict(width=1, color="DarkSlateGrey")),
            name="MSE for line",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(width=900, height=475)
    fig.update_xaxes(
        range=[10, 130],
        tick0=10,
        dtick=20,
        row=1,
        col=1,
        title="defense",
        title_standoff=0,
    )
    fig.update_xaxes(
        range=[0.3, 1.6],
        tick0=0.3,
        dtick=0.2,
        row=1,
        col=2,
        title="slope",
        title_standoff=0,
    )
    fig.update_yaxes(
        range=[10, 130],
        tick0=10,
        dtick=20,
        row=1,
        col=1,
        title="attack",
        title_standoff=0,
    )
    fig.update_yaxes(range=y_range, row=1, col=2, title=y_title, title_standoff=0)
    frames = [
        dict(
            name=f"{slope:.2f}",
            data=[
                go.Scatter(x=x, y=y),
                go.Scatter(x=x, y=df[f"{slope:.2f}"]),
                go.Scatter(x=mse["slope"], y=mse["squared_error"]),
                go.Scatter(x=mse.iloc[[n]]["slope"], y=mse.iloc[[n]]["squared_error"]),
            ],
            traces=[0, 1, 2, 3],
        )
        for n, slope in enumerate(slopes)
    ]

    sliders = [
        {
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "slope: ",
                "visible": True,
            },
            "pad": {"b": 10, "t": 30},
            "steps": [
                {
                    "args": [
                        [f"{slope:.2f}"],
                        {
                            "frame": {
                                "duration": 0,
                                "easing": "linear",
                                "redraw": False,
                            },
                            "transition": {"duration": 0, "easing": "linear"},
                        },
                    ],
                    "label": f"{slope:.2f}",
                    "method": "animate",
                }
                for slope in slopes
            ],
        }
    ]
    fig.update(frames=frames), fig.update_layout(sliders=sliders)
    return fig
