# src/fishbone_visualizer.py
import plotly.graph_objects as go

def visualize_fishbone(fishbone_data: dict, title: str = "Fishbone Diagram"):
    """
    fishbone_data: dict where keys are categories and values are lists of causes
    Returns a Plotly figure object.
    """
    fig = go.Figure()

    categories = list(fishbone_data.keys())
    # assign y positions
    y_positions = list(range(len(categories)))

    # Draw a central spine line
    fig.add_shape(type="line", x0=0.1, y0=len(categories)/2, x1=1.0, y1=len(categories)/2,
                  line=dict(color="black", width=2))

    # Draw category anchors and cause text
    for i, cat in enumerate(categories):
        y = len(categories) - i - 1  # invert so top category appears at top
        # category label on spine
        fig.add_annotation(x=0.05, y=y, text=cat, showarrow=False, xanchor="right", font=dict(size=12))
        causes = fishbone_data.get(cat) or []
        # place causes extending to the right (spread)
        for j, cause in enumerate(causes):
            x = 0.25 + (j * 0.18)
            fig.add_annotation(x=x, y=y + 0.12, text=str(cause), showarrow=True,
                               ax=0.1, ay=0, arrowhead=1, arrowsize=1)

    fig.update_xaxes(visible=False, range=[0,1.1])
    fig.update_yaxes(visible=False, range=[-1, len(categories)+1])
    fig.update_layout(title=title, height=420, margin=dict(l=20, r=20, t=40, b=10), showlegend=False)
    return fig
