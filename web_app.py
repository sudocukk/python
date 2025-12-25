"""
Web-Based Visualization Application (Plotly Dash)
Visualizes happiness data with a modern, interactive web interface.
"""

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import webbrowser

# Import data loading function
try:
    from data_scraper import scrape_wikipedia_happiness_data, clean_scraped_data
except:
    # Fallback: create sample data
    def clean_scraped_data(df):
        return df


def load_data():
    """Load data"""
    try:
        # Try scraper first
        from data_scraper import scrape_wikipedia_happiness_data, clean_scraped_data
        raw_data = scrape_wikipedia_happiness_data()
        df = clean_scraped_data(raw_data)
        # Standardize column names
        column_mapping = {
            'Country': 'Country name',
            'Country name': 'Country name'
        }
        for old, new in column_mapping.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})
        return df
    except Exception as e:
        print(f"Data loading error: {e}. Creating sample data...")
        # Create sample data
        np.random.seed(42)
        countries = ['Finland', 'Denmark', 'Switzerland', 'Iceland', 'Netherlands', 
                    'Norway', 'Sweden', 'Luxembourg', 'New Zealand', 'Austria',
                    'Australia', 'Israel', 'Germany', 'Canada', 'Ireland',
                    'Costa Rica', 'United Kingdom', 'United States', 'Belgium',
                    'France', 'Spain', 'Italy', 'Japan', 'South Korea']
        n = len(countries)
        df = pd.DataFrame({
            'Country name': countries,
            'Happiness Score': np.random.normal(6, 1, n).clip(3, 8),
            'GDP per capita': np.exp(np.random.normal(9, 1, n)).clip(10000, 80000),
            'Social support': np.random.beta(2, 1, n) * 2,
            'Healthy life expectancy': np.random.beta(2, 1, n) * 1.5,
            'Freedom to make life choices': np.random.beta(2, 1, n) * 0.8,
            'Generosity': np.random.beta(1, 2, n) * 0.5,
            'Perceptions of corruption': np.random.beta(2, 2, n) * 0.6
        })
        return df


# Load data
df = load_data()

# Continent assignment function
def assign_continent(country):
    europe = ['Finland', 'Denmark', 'Switzerland', 'Iceland', 'Netherlands', 
              'Norway', 'Sweden', 'Luxembourg', 'Austria', 'Germany', 
              'United Kingdom', 'Belgium', 'France', 'Spain', 'Italy']
    asia = ['Japan', 'South Korea', 'Israel']
    americas = ['United States', 'Canada', 'Costa Rica']
    oceania = ['New Zealand', 'Australia']
    
    if country in europe:
        return 'Europe'
    elif country in asia:
        return 'Asia'
    elif country in americas:
        return 'Americas'
    elif country in oceania:
        return 'Oceania'
    else:
        return 'Other'

if 'Country name' in df.columns:
    df['Continent'] = df['Country name'].apply(assign_continent)

# Initialize Dash application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🌍 Global Happiness and Economic Prosperity Analysis", 
                   className="text-center mb-4"),
            html.P("World Happiness Report - Interactive Data Visualization", 
                  className="text-center text-muted mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("📊 Data Summary", className="card-title"),
                    html.P(f"Total Countries: {len(df)}", className="mb-1"),
                    html.P(f"Number of Columns: {len(df.columns)}", className="mb-0")
                ])
            ], className="mb-3")
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("📈 Average Happiness", className="card-title"),
                    html.H3(f"{df['Happiness Score'].mean():.2f}", 
                           className="text-success mb-0") if 'Happiness Score' in df.columns else "N/A"
                ])
            ], className="mb-3")
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("💰 Average GDP", className="card-title"),
                    html.H3(f"${df['GDP per capita'].mean():,.0f}", 
                           className="text-info mb-0") if 'GDP per capita' in df.columns else "N/A"
                ])
            ], className="mb-3")
        ], width=4)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Filters", className="card-title"),
                    html.Label("Continent:"),
                    dcc.Dropdown(
                        id='continent-filter',
                        options=[{'label': 'All', 'value': 'All'}] + 
                                [{'label': c, 'value': c} for c in df['Continent'].unique()] 
                                if 'Continent' in df.columns else [],
                        value='All',
                        className="mb-3"
                    ),
                    html.Label("Variable Selection:"),
                    dcc.Dropdown(
                        id='variable-selector',
                        options=[
                            {'label': 'GDP per Capita', 'value': 'GDP per capita'},
                            {'label': 'Social Support', 'value': 'Social support'},
                            {'label': 'Healthy Life Expectancy', 'value': 'Healthy life expectancy'},
                            {'label': 'Freedom', 'value': 'Freedom to make life choices'},
                            {'label': 'Generosity', 'value': 'Generosity'},
                            {'label': 'Corruption Perception', 'value': 'Perceptions of corruption'}
                        ],
                        value='GDP per capita',
                        className="mb-3"
                    )
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='scatter-plot')
                ])
            ])
        ], width=9)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='top-countries-chart')
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='continent-comparison')
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='correlation-heatmap')
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Data Table", className="card-title"),
                    dash_table.DataTable(
                        id='data-table',
                        columns=[{"name": col, "id": col} for col in df.columns],
                        data=df.to_dict('records'),
                        page_size=10,
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '10px'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                    )
                ])
            ])
        ], width=12)
    ])
], fluid=True)


# Callbacks
@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('top-countries-chart', 'figure'),
     Output('continent-comparison', 'figure'),
     Output('correlation-heatmap', 'figure'),
     Output('data-table', 'data')],
    [Input('continent-filter', 'value'),
     Input('variable-selector', 'value')]
)
def update_graphs(continent, variable):
    # Filter data
    filtered_df = df.copy()
    if continent != 'All' and 'Continent' in df.columns:
        filtered_df = filtered_df[filtered_df['Continent'] == continent]
    
    # 1. Scatter Plot
    if 'Happiness Score' in filtered_df.columns and variable in filtered_df.columns:
        fig_scatter = px.scatter(
            filtered_df,
            x=variable,
            y='Happiness Score',
            color='Continent' if 'Continent' in filtered_df.columns else None,
            size='GDP per capita' if 'GDP per capita' in filtered_df.columns else None,
            hover_data=['Country name'],
            title=f'{variable} vs Happiness Score',
            labels={variable: variable, 'Happiness Score': 'Happiness Score'}
        )
        fig_scatter.update_traces(marker=dict(opacity=0.7))
        fig_scatter.update_layout(height=400)
    else:
        fig_scatter = go.Figure()
        fig_scatter.add_annotation(text="Data not found", showarrow=False)
    
    # 2. Top Countries Chart
    if 'Happiness Score' in filtered_df.columns and 'Country name' in filtered_df.columns:
        top_10 = filtered_df.nlargest(10, 'Happiness Score')
        fig_top = px.bar(
            top_10,
            x='Happiness Score',
            y='Country name',
            orientation='h',
            title='Top 10 Happiest Countries',
            color='Happiness Score',
            color_continuous_scale='Viridis'
        )
        fig_top.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    else:
        fig_top = go.Figure()
    
    # 3. Continent Comparison
    if 'Continent' in filtered_df.columns and 'Happiness Score' in filtered_df.columns:
        continent_means = filtered_df.groupby('Continent')['Happiness Score'].mean().reset_index()
        fig_continent = px.bar(
            continent_means,
            x='Continent',
            y='Happiness Score',
            title='Average Happiness by Continent',
            color='Happiness Score',
            color_continuous_scale='Blues'
        )
        fig_continent.update_layout(height=400)
    else:
        fig_continent = go.Figure()
    
    # 4. Correlation Heatmap
    numeric_cols = ['Happiness Score', 'GDP per capita', 'Social support', 
                   'Healthy life expectancy', 'Freedom to make life choices',
                   'Generosity', 'Perceptions of corruption']
    available_cols = [col for col in numeric_cols if col in filtered_df.columns]
    if len(available_cols) > 1:
        corr_matrix = filtered_df[available_cols].corr()
        fig_heatmap = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            title="Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect="auto"
        )
        fig_heatmap.update_layout(height=500)
    else:
        fig_heatmap = go.Figure()
    
    # 5. Data Table
    table_data = filtered_df.to_dict('records')
    
    return fig_scatter, fig_top, fig_continent, fig_heatmap, table_data



    

if __name__ == '__main__':
    print("Starting web application...")
    url = "http://127.0.0.1:8050"
    print(f"Open {url} in your browser")
    
    # Tarayıcıyı otomatik aç
    webbrowser.open(url)
    
    app.run(debug=True, host='127.0.0.1', port=8050)

