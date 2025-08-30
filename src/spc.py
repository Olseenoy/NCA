import plotly.graph_objects as go
import numpy as np

def generate_spc_chart(values, title="SPC Chart"):
    x = np.arange(len(values))
    mean = np.mean(values)
    sigma = np.std(values)
    ucl = mean + 3 * sigma
    lcl = mean - 3 * sigma

    # Highlight points out of control
    colors = ['red' if v > ucl or v < lcl else 'blue' for v in values]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=values, mode='markers+lines',
        name='Values', marker=dict(color=colors, size=10)
    ))

    # Control lines
    fig.add_hline(y=mean, line_dash="dash", line_color="green", annotation_text="Mean")
    fig.add_hline(y=ucl, line_dash="dot", line_color="red", annotation_text="UCL")
    fig.add_hline(y=lcl, line_dash="dot", line_color="red", annotation_text="LCL")

    fig.update_layout(
        title=title,
        xaxis_title="Index",
        yaxis_title="Value",
        template="plotly_white",
        showlegend=True
    )
    return fig
