# from Central_Package.all_dc_package import *
import pandas as pd

import plotly.express as px  # (version 4.7.0)
import plotly as py
from plotly import tools, subplots
import plotly.graph_objects as go

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
# https://community.plotly.com/t/how-do-i-use-dash-to-add-local-css/4914/14


def convert_df_cols_to_float(df, cols_list):
    df[cols_list] = df[cols_list].applymap(lambda x: x.strip().replace(',','') if isinstance(x, str) else x)
    df[cols_list] = df[cols_list].applymap(lambda x: x.strip().replace('(', '-').replace(')', '') if isinstance(x, str) else x)
    df[cols_list] = df[cols_list].applymap(lambda x: x.strip().replace('-', '0') if isinstance(x, str) else x)
    df[cols_list] = df[cols_list].astype(float)
    return df


# src_dir = '/Users/derrick/Desktop/Personal Projects/SGX/999. DB/'
src_dir = 'data/sgx/'
sectorName = 'For_Github'
src_file = src_dir + sectorName + '.csv'
df_raw = pd.read_csv(src_file)
df_filter_1 = df_raw.dropna(how='any')
df_filter_1 = convert_df_cols_to_float(df_filter_1, ['2019', '2018', '2017', '2016'])
all_sectors_ref = ['Retailers', 'Insurance', 'Industrial Conglomerates', 'Telecommunications Services',
                   'Automobiles & Auto Parts', 'Food & Beverages', 'Utilities',
                   'Personal & Household Products & Services', 'Software & IT Services', 'Food & Drug Retailing',
                   'Cyclical Consumer Products', 'Healthcare Services', 'Real Estate', 'Applied Resources',
                   'Technology Equipment', 'Energy - Fossil Fuels', 'Mineral Resources',
                   'Transportation', 'Chemicals', 'Banking & Investment Services', 'Industrial Goods',
                   'Cyclical Consumer Services', 'Industrial & Commercial Services', 'Collective Investments']
all_sectors_ref = sorted(all_sectors_ref)
all_sectors = []
for a in all_sectors_ref:
    all_sectors.append({"label": a, "value": a})

all_options = {
    'Income_Statement': ['netIncome', 'grossProfit', 'totalRevenue'],
    'Balance_Sheet': [u'totalLiab', 'totalAssets', 'cash', 'totalStockholderEquity'],
    'Cash_Flow': [u'dividendsPaid', 'totalCashflowsFromInvestingActivities', 'totalCashFromFinancingActivities', 'totalCashFromOperatingActivities', 'repurchaseOfStock', 'issuanceOfStock']
}


# src_dir2 = '/Users/derrick/Desktop/Personal Projects/SGX/999. DB/'

# sectorName2 = 'For_Github_2'
# src_file2 = src_dir2 + sectorName2 + '.csv'
# df_raw_2 = pd.read_csv(src_file2)
# df_filter_1_2 = df_raw_2
src_dir2 = 'data/sgx/'
mthName1,mthName3,mthName6,mthName12 = 'For_Github_2_1mth_cut.csv','For_Github_2_3mth_cut.csv','For_Github_2_6mth_cut.csv','For_Github_2_12mth_cut.csv'
df_raw_2_1mth,df_raw_2_3mth,df_raw_2_6mth,df_raw_2_12mth = pd.read_csv(src_dir2+mthName1),pd.read_csv(src_dir2+mthName3),pd.read_csv(src_dir2+mthName6),pd.read_csv(src_dir2+mthName12)

return_dur_options = [{'label': '1mth', 'value': '1mth'},
                      {'label': '3mth', 'value': '3mth'},
                      {'label': '6mth', 'value': '6mth'},
                      {'label': '12mth', 'value': '12mth'},
                      ]


app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    html.Link(
        rel='stylesheet',
        href='assets/css/sgx_view.css'
    ),

    # html.Div([]),
    html.H1("SGX Overview", style={'text-align': 'centre', 'font-weight': 'bold'}, className='header-Banner'),

    html.Div([
        html.Div([
            html.Li(['Done by DCMK'], style={'list-style-type': 'None'}),
            html.Li(['Last Updated: 24 June 2020'], style={'list-style-type': 'None'}),
            html.Li(['Data Source: Yahoo Finance'], style={'list-style-type': 'None'}),
            html.Li(['This dashboard aims to visualize infomation on the various stocks listed on the Straits Times Index in Singapore'], style={'list-style-type': 'None'}),

        ], className='Header-brief'),

        html.A(['Notes'], style={'font-weight': 'bold'}),
        html.Div([
            html.Li(['You may select items in the Legend to temporarily remove data in that specific year']),
            html.Li(['Financial Statements Data is only limited to the Top-10 Companies for better visualization']),
            html.Li(['Rolling returns only limited to the range of -100% to 100% for better visualization']),
            html.Li(['Do let me know if there are any discrepancies in the data'])
        ], className='Header-Notes'),

    ], className='Headers'),

    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    dcc.Dropdown(id="slct_sector",
                                 options=all_sectors,
                                 value='Insurance',
                                 # style={'width': "40%"}
                                 ),
                ], className="mainChart1-dropdown-col"),
                html.Div([
                    dcc.Dropdown(id="slct_stmt",
                                 options=[{'label': k, 'value': k} for k in all_options.keys()],
                                 multi=False,
                                 value='Income_Statement',
                                 # style={'width': "40%"}
                                 ),
                ], className="mainChart1-dropdown-col"),
                html.Div([
                    dcc.Dropdown(id="slct_label",
                                 value='totalRevenue',
                                 # style={'width': "40%"},
                                 placeholder="Select a Item"
                                 ),
                ], className="mainChart1-dropdown-col"),

            ], className="mainChart1-dropdown-row"),

            # html.Div(id='output_container', children=[]),
            html.Br(),
            dcc.Graph(id='my_bee_map', figure={}),
            html.Br(),
        ], className="inside-mainChart-Box"),




        html.Div([
            html.Div([
                html.Div([
                    dcc.Dropdown(id="slct_sector2",
                                 options=all_sectors,
                                 value='Insurance',
                                 # style={'width': "40%"}
                                 ),
                ], className="mainChart1-dropdown-col"),
                html.Div([
                    dcc.Dropdown(id="slct_dur",
                                 options=return_dur_options,
                                 value='1mth',
                                 # style={'width': "40%"}
                                 ),
                ], className="mainChart1-dropdown-col"),
            ], className="mainChart1-dropdown-row"),



            html.Br(),
            dcc.Graph(id='my_box_map', figure={})
        ], className="inside-mainChart-Box"),



    ], className='mainChart-Box'),



], className='DC-Backdrop')

app.css.append_css({
    'external_url': "https://codepen.io/chriddyp/pen/bWLwgP.css"
})
# ------------------------------------------------------------------------------


@app.callback(
    dash.dependencies.Output('slct_label', 'options'),
    [dash.dependencies.Input('slct_stmt', 'value')])
def set_cities_options(selected_country):
    return [{'label': i, 'value': i} for i in all_options[selected_country]]


@app.callback(
    dash.dependencies.Output('slct_stmt', 'value'),
    [dash.dependencies.Input('slct_stmt', 'options')])
def set_cities_value(available_options):
    return available_options[0]['value']

from plotly.graph_objects import *
# Connect the Plotly graphs with Dash Component
@app.callback(
    # [Output(component_id='output_container', component_property='children'),
    #  Output(component_id='my_bee_map', component_property='figure')],
    Output(component_id='my_bee_map', component_property='figure'),
    [Input(component_id='slct_sector', component_property='value'),
     Input(component_id='slct_stmt', component_property='value'),
     Input(component_id='slct_label', component_property='value')
     ]
)
def update_graph(option_slctd1, option_slctd2, option_slctd3):
    container = "The year chosen by user was: {}".format(option_slctd2)

    dff = df_filter_1.copy()
    dff = dff[dff["Sector"] == option_slctd1]
    dff = dff[dff["Statement_Type"] == option_slctd2]  # dff = dff[dff["Statement_Type"] == "Income_Statement"]
    dff = dff[dff["Info_Label"] == option_slctd3]

    print(dff)
    print(len(dff))
    # Plotly Express
    dff = dff.sort_values(by='2019')
    dff = dff.iloc[:10, :]
    df_long = pd.melt(dff, id_vars=['SGX_CoyName'], value_vars=['2019', '2018', '2017', '2016'])

    # fig = px.line(data_frame=df_long, x='SGX_CoyName', y='value', template='plotly_dark', color='variable')
    fig = px.bar(data_frame=df_long, x='SGX_CoyName', y='value', color='variable',
                 barmode='group',
                 template='plotly_dark',
                 width=1100, height=400
                 )
    cht1_title = 'Financial Statements Overview for {} Sector'.format(option_slctd1)

    # layout = Layout(paper_bgcolor='rgb(0,0,0,0',plot_bgcolor='rgb(0,0,0,0')
    fig.update_layout(title=cht1_title,
                      xaxis_title="Company Names",
                      yaxis_title="Value"
                      )
    # return container, fig
    return fig


# 2nd Chart
@app.callback(
    Output(component_id='my_box_map', component_property='figure'),
    [Input(component_id='slct_sector2', component_property='value'),
     Input(component_id='slct_dur', component_property='value')]
)
def update_graph_2(option_slctd1, option_slctd2):
    print(option_slctd1)
    p_sec = 'Healthcare Services'
    p_duration = '1mth'
    dff = df_raw_2_1mth.copy()
    if option_slctd2 =='1mth':
        dff = df_raw_2_1mth.copy()
    if option_slctd2 =='3mth':
        dff = df_raw_2_3mth.copy()
    if option_slctd2 =='6mth':
        dff = df_raw_2_6mth.copy()
    if option_slctd2 =='12mth':
        dff = df_raw_2_12mth.copy()
    # dff = df_filter_1_2.copy()
    dff = dff[['Sector', 'Company_Name', option_slctd2]]
    dff = dff[dff["Sector"] == option_slctd1]
    dff = dff.dropna(subset=[option_slctd2])

    dff[option_slctd2] = dff[option_slctd2].apply(lambda x: x*100)

    dff = dff.iloc[:, :]
    # dff = dff.loc[dff[option_slctd2] <= 1.5]
    # dff = dff.loc[dff[option_slctd2] >= -1.5]

    print(dff)
    fig = px.box(dff, x="Company_Name", y=option_slctd2,
                 template='plotly_dark',
                 width=1100, height=600,

                 )
    cht2_title = '{} Rolling Returns for {} Sector'.format(option_slctd2, option_slctd1)
    fig.update_layout(title=cht2_title,
                      xaxis_title="Company Names",
                      yaxis_title="{} % Change".format(option_slctd2),
                      )
    fig.update_yaxes(range=[-100, 100])
    return fig



if __name__ == '__main__':
    app.run_server() # debug=True
    x=1