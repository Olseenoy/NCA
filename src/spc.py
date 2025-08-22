import plotly.graph_objects as go
import numpy as np

def generate_spc_chart(values):
    x = np.arange(len(values))
    mean = np.mean(values)
    sigma = np.std(values)
    ucl = mean + 3 * sigma
    lcl = mean - 3 * sigma

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=values, mode='lines+markers', name='Values'))
    fig.add_hline(y=mean, line_dash="dash", line_color="blue", annotation_text="Mean")
    fig.add_hline(y=ucl, line_dash="dot", line_color="red", annotation_text="UCL")
    fig.add_hline(y=lcl, line_dash="dot", line_color="red", annotation_text="LCL")
    fig.update_layout(title="SPC Chart", xaxis_title="Index", yaxis_title="Value")
    return fig
