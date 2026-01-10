import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

def run_clustervis_dashboard(X_data, y_labels, user_colors):
    """
    Run an interactive dashboard for clustering visualization, along with a probability table.
    """
    # 1. Data preparation (PCA)
    pca = PCA(n_components=2)
    X_PCA = pca.fit_transform(X_data)

    # 2. Model training
    model = BaggingClassifier(
        estimator=KNeighborsClassifier(n_neighbors=5),
        n_estimators=10, max_samples=0.05, n_jobs=-1, random_state=42
    )
    model.fit(X_PCA, y_labels)

    # 3. Inner function to construct the initial figure
    def build_figure():
        res = 120
        y_pts, x_pts = X_PCA[:, 0], X_PCA[:, 1]
        x_min, x_max = x_pts.min() - 1, x_pts.max() + 1
        y_min, y_max = y_pts.min() - 1, y_pts.max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, res), np.linspace(y_min, y_max, res))

        Z_proba = model.predict_proba(np.c_[yy.ravel(), xx.ravel()])
        grid_rgb = (Z_proba @ user_colors[:Z_proba.shape[1]]).reshape((res, res, 3)).astype(np.uint8)

        fig = go.Figure()
        fig.add_trace(go.Image(z=grid_rgb, x0=x_min, y0=y_min, dx=(x_max - x_min) / res, dy=(y_max - y_min) / res,
                               hoverinfo='skip'))

        colors = [f'rgb({user_colors[l][0]},{user_colors[l][1]},{user_colors[l][2]})' for l in y_labels]
        fig.add_trace(go.Scattergl(
            x=x_pts, y=y_pts, mode='markers',
            marker=dict(color=colors, size=10, line=dict(width=1.2, color='black')),
            customdata=y_labels,
            hovertemplate="<b>Cluster: %{customdata}</b><extra></extra>"
        ))

        fig.update_layout(
            title=dict(text="<b>Clustervis</b>", x=0.5, y=0.95, font=dict(size=20)),
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            template="plotly_white"
        )
        return fig

    # --- DASH APP ---
    app = Dash(__name__)

    app.layout = html.Div(style={
        'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center',
        'height': '100vh', 'backgroundColor': '#f0f2f5', 'fontFamily': 'Segoe UI, sans-serif', 'overflow': 'hidden'
    }, children=[
        html.Div(style={'display': 'flex', 'height': '90vh', 'width': '95vw', 'maxWidth': '1200px'}, children=[

            # Graph (Left)
            html.Div([
                dcc.Graph(id='main-graph', figure=build_figure(), config={'displayModeBar': False},
                          style={'height': '100%', 'width': '100%'})
            ], style={'flex': '2', 'backgroundColor': 'white', 'borderRadius': '20px',
                      'boxShadow': '0px 10px 30px rgba(0,0,0,0.08)', 'marginRight': '20px', 'padding': '10px'}),

            # Control panel (Right)
            html.Div(style={'flex': '1', 'backgroundColor': 'white', 'borderRadius': '20px', 'padding': '25px',
                            'boxShadow': '0px 10px 30px rgba(0,0,0,0.08)', 'display': 'flex',
                            'flexDirection': 'column'}, children=[
                html.H2("Statistics", style={'fontSize': '22px', 'marginBottom': '20px'}),

                # Styled table
                html.Div(id='stats-table-container'),

                html.Hr(style={'border': '0', 'borderTop': '1px solid #eee', 'margin': '25px 0'}),

                html.H3("Probabilities", style={'fontSize': '16px', 'color': '#666'}),
                html.Div(id='hover-probability-display',
                         style={'backgroundColor': '#fafafa', 'padding': '15px', 'borderRadius': '12px',
                                'flexGrow': '1'})
            ])
        ])
    ])

    @app.callback(
        Output('hover-probability-display', 'children'),
        Input('main-graph', 'hoverData')
    )
    def update_probs(hoverData):
        if not hoverData: return html.P("Hover over a point...", style={'color': '#aaa', 'textAlign': 'center'})
        pt = hoverData['points'][0]
        probs = model.predict_proba([[pt['y'], pt['x']]])[0]
        return [html.Div([
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '5px'}, children=[
                html.Span(f"Cluster {i}", style={'fontWeight': '600',
                                                 'color': f'rgb({user_colors[i][0]},{user_colors[i][1]},{user_colors[i][2]})'}),
                html.Span(f"{p:.1%}", style={'fontWeight': 'bold'})
            ]),
            html.Div(style={'height': '6px', 'width': '100%', 'backgroundColor': '#eee', 'borderRadius': '3px',
                            'marginBottom': '10px'}, children=[
                html.Div(style={'height': '100%', 'width': f'{p * 100}%',
                                'backgroundColor': f'rgb({user_colors[i][0]},{user_colors[i][1]},{user_colors[i][2]})',
                                'borderRadius': '3px'})
            ])
        ]) for i, p in enumerate(probs)]

    @app.callback(
        Output('stats-table-container', 'children'),
        Input('main-graph', 'id')
    )
    def update_table(_):
        counts = pd.Series(y_labels).value_counts().sort_index()
        header = html.Div([
            html.Div("ID", style={'flex': '1', 'fontWeight': 'bold', 'color': '#95a5a6', 'fontSize': '12px'}),
            html.Div("COLOR", style={'flex': '1', 'fontWeight': 'bold', 'color': '#95a5a6', 'fontSize': '12px',
                                     'textAlign': 'center'}),
            html.Div("TOTAL", style={'flex': '1', 'fontWeight': 'bold', 'color': '#95a5a6', 'fontSize': '12px',
                                     'textAlign': 'right'}),
        ], style={'display': 'flex', 'padding': '0 10px 10px 10px', 'borderBottom': '1px solid #f1f1f1'})

        rows = []
        for cid, count in counts.items():
            rows.append(html.Div([
                html.Div(f"C{cid}", style={'flex': '1', 'fontWeight': 'bold', 'color': '#2c3e50', 'fontSize': '14px'}),
                html.Div(html.Div(style={'width': '12px', 'height': '12px', 'borderRadius': '50%',
                                         'backgroundColor': f'rgb({user_colors[cid][0]},{user_colors[cid][1]},{user_colors[cid][2]})',
                                         'margin': 'auto', 'boxShadow': '0px 2px 4px rgba(0,0,0,0.1)'}),
                         style={'flex': '1', 'display': 'flex'}),
                html.Div(f"{count} pts",
                         style={'flex': '1', 'textAlign': 'right', 'color': '#7f8c8d', 'fontSize': '13px'})
            ], style={'display': 'flex', 'padding': '12px 10px', 'borderBottom': '1px solid #f8f9fa',
                      'alignItems': 'center'}))
        return html.Div([header] + rows)

    app.run(debug=True)