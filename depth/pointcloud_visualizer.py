import numpy as np
import plotly.graph_objects as go


def pointcloud_to_figure(
    points,
    colors,
    max_points=12000
):

    points = np.asarray(
        points,
        dtype=np.float32
    )

    colors = np.asarray(
        colors,
        dtype=np.uint8
    )

    if len(points) > max_points:

        indices = np.linspace(
            0,
            len(points) - 1,
            max_points,
            dtype=np.int64
        )

        points = points[indices]
        colors = colors[indices]

    cores = [
        f"rgb({int(r)},{int(g)},{int(b)})"
        for r, g, b in colors
    ]

    figura = go.Figure(

        data=[

            go.Scatter3d(

                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],

                mode="markers",

                marker=dict(
                    size=2,
                    color=cores,
                    opacity=0.9
                ),

                hoverinfo="none"
            )
        ]
    )

    figura.update_layout(

        scene=dict(

            xaxis=dict(
                title="X"
            ),

            yaxis=dict(
                title="Y"
            ),

            zaxis=dict(
                title="Z"
            ),

            aspectmode="data"
        ),

        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0
        ),

        showlegend=False
    )

    return figura
