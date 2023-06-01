import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

title_font_size = 16
font_size = 10
color_discrete_map = {
        0: '#67001f', 
        1: '#e58267', 
        2: '#053061',
        3: '#fcf9f0',
        4: '#6bacd1', 
    }

def create_bar(
        df, x_col, y_col, group_col, barmode, x_title, y_title,
        ):

    data = [
        go.Bar(
            x=dg[x_col].values, 
            y=dg[y_col].values,
            marker_color=color_discrete_map[i],
            orientation='h',
            name=x
            ) for i, (x, dg) in enumerate(df.groupby(group_col))
    ]

    layout = go.Layout(
        title='Model Performance',
        title_font_family="Arial Black",
        title_font_size=title_font_size,
        barmode=barmode,
        yaxis=dict(
            title=y_title,
            showline=False,
            showgrid=False),
        xaxis=dict(
            title=x_title,
            showgrid=True,
            gridcolor='black',
            zeroline=True,
            showline=True),
        margin=dict(
            l=20,
            b=20,
            r=20,
            t=50),
        legend=dict(
            title_text='Zone',
            orientation="h", 
            x=0, 
            y=-0.2),
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#f8f9fa',
        font=dict(size=font_size),
        showlegend=True,
        )

    return go.Figure(data=data, layout=layout)

def create_pie_rating(df, cmin=1, cmax=5):
    df.rename(columns={'overall_rating' : 'Overall Rating'}, inplace=True)
    fig = px.sunburst(
        df, 
        path=['job_status', 'years_at_company'], values='count',
        color='Overall Rating', 
        color_continuous_scale='RdBu',
        color_continuous_midpoint=np.average(df['Overall Rating'], weights=df['count']))

    fig.update_layout(
        title='Employee Rating by Group',
        title_font_family="Arial Black",
        title_font_size=title_font_size,
        coloraxis=dict(cmin=cmin, cmax=cmax),
        margin=dict(
            autoexpand=False,
            l=10,
            b=10,
            r=50,
            t=50),
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#f8f9fa',
        font=dict(size=font_size)
        )
    return fig

def create_pie(labels, values):

    layout = go.Layout(
        title='Sample Size',
        title_font_family="Arial Black",
        title_font_size=title_font_size,
        # coloraxis=dict(cmin=cmin, cmax=cmax),
        margin=dict(
            autoexpand=False,
            l=10,
            b=20,
            r=20,
            t=50),
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#f8f9fa',
        showlegend=True,
        font=dict(size=font_size)
        )
    fig = go.Pie(labels=labels, values=values, marker=dict(colors=[x for x in color_discrete_map.values()]))
    fig = go.Figure(data=[fig], layout=layout)
    return fig