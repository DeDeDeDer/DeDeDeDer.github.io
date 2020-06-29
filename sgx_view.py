# from Central_Package.all_dc_package import *
import pandas as pd
import numpy as np
import plotly.express as px  # (version 4.7.0)
import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
# import dash_table
#
# from sklearn.linear_model import RidgeClassifier, LogisticRegression, Lasso, Ridge, BayesianRidge
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
#
# from sklearn.neural_network import MLPRegressor
#
# from sklearn.neighbors import KNeighborsRegressor
#
# from sklearn.svm import SVC, LinearSVC
#
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
#
# from sklearn.model_selection import train_test_split


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

def convert_df_cols_to_float(df, cols_list):
    df[cols_list] = df[cols_list].applymap(lambda x: x.strip().replace(',','') if isinstance(x, str) else x)
    df[cols_list] = df[cols_list].applymap(lambda x: x.strip().replace('(', '-').replace(')', '') if isinstance(x, str) else x)
    #f[cols_list] = df[cols_list].applymap(lambda x: x.strip().replace('-', '0') if isinstance(x, str) else x)
    df[cols_list] = df[cols_list].astype(float)
    return df



# src_dir0 = '/Users/derrick/Desktop/Personal Projects/SGX/999. DB/A_Stock_Screener/'
src_dir0 = 'data/sgx/'
fn = 'For_Github_3.csv'
df_stock_screen = pd.read_csv(src_dir0 + fn)
xyz_cols = list(df_stock_screen.columns)
xyz_cols = sorted(xyz_cols)
ls_dict_xyz_options = [{'label': k, 'value': k} for k in xyz_cols]


src_dir2 = 'data/sgx/'
mthName1,mthName3,mthName6,mthName12 = 'For_Github_2_1mth_cut.csv','For_Github_2_3mth_cut.csv','For_Github_2_6mth_cut.csv','For_Github_2_12mth_cut.csv'
df_raw_2_1mth,df_raw_2_3mth,df_raw_2_6mth,df_raw_2_12mth = pd.read_csv(src_dir2+mthName1),pd.read_csv(src_dir2+mthName3),pd.read_csv(src_dir2+mthName6),pd.read_csv(src_dir2+mthName12)

return_dur_options = [{'label': '1mth', 'value': '1mth'},
                      {'label': '3mth', 'value': '3mth'},
                      {'label': '6mth', 'value': '6mth'},
                      {'label': '12mth', 'value': '12mth'},
                      ]





# Import Data - Financial Statements
# src_dir = '/Users/derrick/Desktop/Personal Projects/SGX/999. DB/'
# src_dir = 'data/sgx/'
src_dir = 'data/sgx/'
sectorName = 'For_Github'
src_file = src_dir + sectorName + '.csv'
df_fin_raw = pd.read_csv(src_file).fillna(0)
# df_fin_filt_1 = df_fin_raw.dropna(how='any')
# Clean Numerical Columns
df_fin_filt_1 = convert_df_cols_to_float(df_fin_raw, ['2019', '2018', '2017', '2016'])
# Prepare Sector Options
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
# Prepare Financial_Statement Options
fin_stmts_options_dict = {
    'Income_Statement': ['netIncome', 'grossProfit', 'totalRevenue'],
    'Balance_Sheet': [u'totalLiab', 'totalAssets', 'cash', 'totalStockholderEquity'],
    'Cash_Flow': [u'dividendsPaid', 'totalCashflowsFromInvestingActivities', 'totalCashFromFinancingActivities', 'totalCashFromOperatingActivities', 'repurchaseOfStock', 'issuanceOfStock']
}
year_options = [{'label': '2019', 'value': '2019'},
                      {'label': '2018', 'value': '2018'},
                      {'label': '2017', 'value': '2017'},
                      {'label': '2016', 'value': '2016'},
                      ]
tempo_coy_options = [{'label': 'S68', 'value': 'S68'},
                      {'label': '5PO', 'value': '5PO'},
                      ]

all_options = {
    'Income_Statement': ['netIncome', 'grossProfit', 'totalRevenue'],
    'Balance_Sheet': [u'totalLiab', 'totalAssets', 'cash', 'totalStockholderEquity'],
    'Cash_Flow': [u'dividendsPaid', 'totalCashflowsFromInvestingActivities', 'totalCashFromFinancingActivities', 'totalCashFromOperatingActivities', 'repurchaseOfStock', 'issuanceOfStock']
}


app = dash.Dash(__name__)
server = app.server

# html.Div([])
app.layout = html.Div([
    # Link CSS
    html.Link(
        rel='stylesheet',
        href='assets/css/sgx_view_2.css'
    ),
    html.Div([
        # mainContainer
        html.Div([
            # Column 1
            html.Div([
                # Header
                html.Div([
                    html.H1("Individual Report"),
                    html.A("Visualizes financial information based on the selected company"),
                    html.Li(["Note: An empty chart indicates no data was provided"], style={'list-style-type': 'None'}),

                ], className='col1-Header'),
                # Options 1
                html.Div([
                    # Option Sector
                    html.Div([
                        html.A('Sector'),
                        dcc.Dropdown(id="slct_sector",
                                     options=all_sectors,
                                     value='Banking & Investment Services',
                                     # style={'width': "40%"}
                                     optionHeight=50
                                     ),
                    ], className='col1-row1-col1'),
                    # Option Year
                    html.Div([
                        html.A('Year'),
                        dcc.Dropdown(id="slct_year",
                                     options=year_options,
                                     value='2019',
                                     # style={'width': "40%"}
                                     ),
                    ], className='col1-row1-col2'),
                ], className='col1-row1'),
                # Options 2
                html.Div([
                    # Option Company
                    html.Div([
                        html.A('Company'),
                        dcc.Dropdown(id="slct_company",
                                     # options=tempo_coy_options,
                                     value='SGX',
                                     # style={'width': "200%"}
                                     ),
                    ], className='col1-row2-col1'),
                    # Option Returns period
                    html.Div([
                        html.A('Period'),
                        dcc.Dropdown(id="slct_dur",
                                     options=return_dur_options,
                                     value='3mth',
                                     # style={'width': "40%"}
                                     ),
                    ], className='col1-row2-col2'),
                ], className='col1-row2'),

                # First row of charts
                html.Div([
                    # Income Statement Chart
                    html.Div([
                        dcc.Graph(id='Income_Statement', figure={}),
                    ], className='col1-row3-col1'),
                    # Balance Sheet Chart
                    html.Div([
                        dcc.Graph(id='Balance_Sheet', figure={}),
                    ], className='col1-row3-col2'),
                ], className='col1-row3'),

                # Second row of charts
                html.Div([
                    # Cash flow Chart
                    html.Div([
                        dcc.Graph(id='Cash_flow', figure={}),
                    ], className='col1-row4-col1'),
                    # Trailing Returns Chart
                    html.Div([
                        dcc.Graph(id='Trailing_Returns_Distribution', figure={}),
                    ], className='col1-row4-col2')
                ], className='col1-row4'),

                # Tempo Empty
                html.Div([
                ], className='col1-row5'),

            ], className='col1'),

            # Column 2
            html.Div([
                # Header
                html.Div([
                    html.H1("Benchmark Report"),
                    html.A("Comparing financial information against the respective sector-level"),
                ], className='col2-Header'),
                # Options 1
                html.Div([
                    # html.Div([
                    #     html.A('Sector'),
                    #     dcc.Dropdown(id="slct_sector",
                    #                  options=all_sectors,
                    #                  value='Banking & Investment Services',
                    #                  # style={'width': "40%"}
                    #                  optionHeight=50
                    #                  ),
                    # ], className="col2-row1-col1"),
                    html.Div([
                        html.A('Statement Type'),
                        dcc.Dropdown(id="slct_stmt",
                                     options=[{'label': k, 'value': k} for k in all_options.keys()],
                                     multi=False,
                                     value='Income_Statement',
                                     # style={'width': "40%"}
                                     ),
                    ], className="col2-row1-col1"),
                    html.Div([
                        html.A('Item'),
                        dcc.Dropdown(id="slct_label",
                                     value='totalRevenue',
                                     # style={'width': "40%"},
                                     placeholder="Select a Item"
                                     ),
                    ], className="col2-row1-col2"),
                ], className='col2-row1'),
                # Statements & Market Cap
                html.Div([
                    # Financial Statements Benchmark Chart
                    html.Div([
                        dcc.Graph(id='fin_stms_benchmark', figure={}),
                    ], className='col2-row2-col1'),
                    # Market Cap Chart market_cap
                    html.Div([
                        dcc.Graph(id='market_cap', figure={}),
                    ], className='col2-row2-col2'),
                ], className='col2-row2'),
                # Options for 3D-Plot
                html.Div([
                    # X Column
                    html.Div([
                        html.A('X Column'),
                        dcc.Dropdown(id="slct_xyz_col_x",
                                     options=ls_dict_xyz_options,
                                     multi=False,
                                     value='Mkt Cap ($M)',
                                     # style={'width': "40%"}
                                     ),
                    ], className='col2-row3-col1'),
                    # Y Column
                    html.Div([
                        html.A('Y Column'),
                        dcc.Dropdown(id="slct_xyz_col_y",
                                     options=ls_dict_xyz_options,
                                     multi=False,
                                     value='Tot. Rev ($M)',
                                     # style={'width': "40%"}
                                     ),
                    ], className='col2-row3-col2'),
                    # Z Column
                    html.Div([
                        html.A('Y Column'),
                        dcc.Dropdown(id="slct_xyz_col_z",
                                     options=ls_dict_xyz_options,
                                     multi=False,
                                     value='GTI Score',
                                     # style={'width': "40%"}
                                     ),
                    ], className='col2-row3-col3'),

                ], className='col2-row3'),
                # Box- Plot & 3D-Plot
                html.Div([
                    # Box- Plot Chart
                    html.Div([
                        dcc.Graph(id='xyz_benchmark', figure={}),
                    ], className='col2-row4-col1'),
                    # 3D-Plot
                    html.Div([
                        dcc.Graph(id='box_plot_returns', figure={}),
                    ], className='col2-row4-col2'),
                ], className='col2-row4'),

                # WIP
                html.Div([], className='col2-row5'),
            ], className='col2'),

        ], className='mainContainer'),

    ], className='BodyBack')
])


@app.callback(
    Output(component_id='slct_company', component_property='options'),
    [Input(component_id='slct_sector', component_property='value')]
)
def update_sub_companies(option_slctd1):
    df_ = df_fin_filt_1.copy()
    print('hereh')
    print(option_slctd1)
    print(df_.head(5))
    df_ = df_[df_["Sector"] == option_slctd1]
    df_coy = df_["SGX_CoyName"].unique()
    print(df_coy)
    ls_coy = list(df_coy)
    print(ls_coy)
    ls_coy_sorted = sorted(ls_coy)
    #ls_dict_coy_sorted = []
    #for a in ls_coy_sorted:
    #    ls_dict_coy_sorted.append({"label": a, "value": a})
    ls_dict_coy_sorted=[{'label': k, 'value': k} for k in ls_coy_sorted]
    print('final - coys for dropdown')
    print(ls_dict_coy_sorted)
    return ls_dict_coy_sorted


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


@app.callback(
    Output(component_id='Income_Statement', component_property='figure'),
    [Input(component_id='slct_sector', component_property='value'),
     Input(component_id='slct_year', component_property='value'),
     Input(component_id='slct_company', component_property='value')
     ]
)
def update_graph(option_slctd1, option_slctd2, option_slctd3):
    # Initial Filter
    df_is = df_fin_filt_1.copy()
    df_is = df_is[df_is["Sector"] == option_slctd1]
    df_is = df_is[df_is["SGX_CoyName"] == option_slctd3]
    df_is = df_is[[option_slctd2, "Info_Label", "Statement_Type"]]
    df_is[option_slctd2] = df_is[option_slctd2].abs()
    df_is = df_is[df_is["Statement_Type"] == 'Income_Statement']  # dff = dff[dff["Statement_Type"] == "Income_Statement"]
    # print('Incme Statement')
    # print(df_is)
    # Measure dictionary
    # label_sort_ = [
    #     {'totalRevenue': 'absolute'}, {'costOfRevenue': 'relative'},
    #     {'grossProfit': 'total'}, {'totalOperatingExpenses': 'relative'}, {'totalOtherIncomeExpenseNet': 'relative'},
    #     {'operatingIncome': 'total'},
    #     {'interestExpense': 'relative'}, {'researchDevelopment': 'relative'},
    #     {'incomeBeforeTax': 'total'}, {'incomeTaxExpense': 'relative'},
    #     {'netIncome': 'absolute'},
    # ]
    # df_corr = pd.DataFrame({
    #     'totalRevenue': [1], 'costOfRevenue': [-1],
    #     'grossProfit': [1], 'totalOperatingExpenses': [-1],'totalOtherIncomeExpenseNet': [-1],
    #     'operatingIncome': [1],
    #     'interestExpense': [-1], 'researchDevelopment': [-1],
    #     'incomeBeforeTax': [1], 'incomeTaxExpense': [-1],
    #     'netIncome': [1]
    # })
    # label_sort_ = [
    #     {'totalRevenue': 'absolute'}, {'costOfRevenue': 'relative'},
    #     {'grossProfit': 'total'},
    #     {'operatingIncome': 'total'},
    #     {'interestExpense': 'relative'}, {'researchDevelopment': 'relative'},
    #     {'incomeBeforeTax': 'total'}, {'incomeTaxExpense': 'relative'},
    #     {'netIncome': 'absolute'},
    # ]
    # df_corr = pd.DataFrame({
    #     'totalRevenue': [1], 'costOfRevenue': [-1],
    #     'grossProfit': [1],
    #     'operatingIncome': [1],
    #     'interestExpense': [-1], 'researchDevelopment': [-1],
    #     'incomeBeforeTax': [1], 'incomeTaxExpense': [-1],
    #     'netIncome': [1]
    # })
    label_sort_ = [
        {'totalRevenue': 'absolute'},
        {'grossProfit': 'total'},
        {'operatingIncome': 'total'},
        {'incomeBeforeTax': 'total'},
        {'netIncome': 'total'}
    ]
    df_is_corr = pd.DataFrame({
        'totalRevenue': [1],
        'grossProfit': [1],
        'operatingIncome': [1],
        'incomeBeforeTax': [1],
        'netIncome': [1]
    })
    # ERROR req_labels = list(set().union(*(d.keys() for d in label_sort_)))
    req_labels = [i for s in [d.keys() for d in label_sort_] for i in s]
    req_measures = [i for s in [d.values() for d in label_sort_] for i in s]
    # req_correctors = [i for s in [d.values() for d in label_correction] for i in s]

    df_is_filt = df_is[df_is["Info_Label"].isin(req_labels)]

    df_is_corr = df_is_corr[list(df_is_filt["Info_Label"])].T.reset_index()
    df_is_corr.columns = ['Info_Label', 'Corrector']

    df_is_filt_2 = pd.merge(df_is_filt, df_is_corr, on=['Info_Label'])

    df_is_filt_2 = convert_df_cols_to_float(df_is_filt_2, [option_slctd2])

    sorterIndex = dict(zip(req_labels, range(len(req_labels))))
    df_is_filt_2['Order_id'] = df_is_filt_2['Info_Label'].map(sorterIndex)
    df_is_filt_2.sort_values(['Order_id'], inplace=True)

    req_vals_cor = [a*b for a,b in zip(list(df_is_filt_2[option_slctd2]), list(df_is_filt_2['Corrector']))]
    df_is_filt_2[option_slctd2]=req_vals_cor

    x_labels = list(df_is_filt_2['Info_Label']) # req_labels
    y_labels = list(df_is_filt_2[option_slctd2])

    # adjust due to yahoo error
    pos0 = 1
    adjust0 = y_labels[x_labels.index("grossProfit")]-y_labels[x_labels.index("totalRevenue")]

    y_labels[pos0:pos0] = [adjust0]

    x_labels[pos0:pos0] = ['costOfRevenue']
    req_measures[pos0:pos0] = ['relative']

    adjust1 = y_labels[x_labels.index("operatingIncome")]-y_labels[x_labels.index("grossProfit")]

    y_labels[3:3] = [adjust1]

    x_labels[3:3] = ['operatingExpense']
    req_measures[3:3] = ['relative']

    pos2 = 5
    adjust2 = y_labels[x_labels.index("incomeBeforeTax")]-y_labels[x_labels.index("operatingIncome")]
    y_labels[pos2:pos2] = [adjust2]
    x_labels[pos2:pos2] = ['interestExpense']
    req_measures[pos2:pos2] = ['relative']

    pos3 = 7
    adjust3 = y_labels[x_labels.index("netIncome")]-y_labels[x_labels.index("incomeBeforeTax")]
    y_labels[pos3:pos3] = [adjust3]
    x_labels[pos3:pos3] = ['taxExpense']
    req_measures[pos3:pos3] = ['relative']

    # Graph
    import plotly.graph_objects as go
    fig = go.Figure(go.Waterfall(
        name="20", orientation="v",
        measure=req_measures,
        x=x_labels,
        textposition="outside",
        # text=["+60", "+80", "", "-40", "-20", "Total"],
        y=y_labels,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    # title_ = "Balance Sheet {}".format(option_slctd2)
    # fig.update_layout(
    #     title=title_,
    #
    # )
    cht1_title = 'Income Statement for {}'.format(option_slctd3)
    # layout = Layout(paper_bgcolor='rgb(0,0,0,0',plot_bgcolor='rgb(0,0,0,0')
    fig.update_layout(title=cht1_title,
                      xaxis_title="Content",
                      yaxis_title="Value",
                      # showlegend=True,
                      width=700, height=500,template='plotly_dark'
                      )
    return fig


@app.callback(
    Output(component_id='Balance_Sheet', component_property='figure'),
    [Input(component_id='slct_sector', component_property='value'),
     Input(component_id='slct_year', component_property='value'),
     Input(component_id='slct_company', component_property='value')
     ]
)
def update_graph2(option_slctd1, option_slctd2, option_slctd3):
    print('*'*100)
    # Initial Filter
    df_bs = df_fin_filt_1.copy()
    df_bs = df_bs[df_bs["Sector"] == option_slctd1]
    df_bs = df_bs[df_bs["SGX_CoyName"] == option_slctd3]
    df_bs = df_bs[[option_slctd2, "Info_Label", "Statement_Type"]]
    df_bs[option_slctd2] = df_bs[option_slctd2].abs()
    df_bs = df_bs[df_bs["Statement_Type"] == 'Balance_Sheet']  # dff = dff[dff["Statement_Type"] == "Income_Statement"]]
    label_sort_ = [
        {'totalCurrentAssets': 'absolute'},
        {'totalAssets': 'total'},
        {'totalCurrentLiabilities': 'relative'},
        {'totalLiab': 'total'},
        {'commonStock': 'relative'},
        {'totalStockholderEquity': 'total'}
    ]
    df_corr = pd.DataFrame({
        'totalCurrentAssets': [1],
        'totalAssets': [1],
        'totalCurrentLiabilities': [1],
        'totalLiab': [1],
        'commonStock': [1],
        'totalStockholderEquity': [1]
    })
    # ERROR req_labels = list(set().union(*(d.keys() for d in label_sort_)))
    req_labels = [i for s in [d.keys() for d in label_sort_] for i in s]
    req_measures = [i for s in [d.values() for d in label_sort_] for i in s]
    # Filter only required
    df_bs_filt = df_bs[df_bs["Info_Label"].isin(req_labels)]
    df_corr = df_corr[list(df_bs_filt["Info_Label"])].T.reset_index()
    df_corr.columns = ['Info_Label', 'Corrector']
    df_bs_filt_2 = pd.merge(df_bs_filt, df_corr, on=['Info_Label'])
    df_bs_filt_2 = convert_df_cols_to_float(df_bs_filt_2, [option_slctd2])
    # Order
    sorterIndex = dict(zip(req_labels, range(len(req_labels))))
    df_bs_filt_2['Order_id'] = df_bs_filt_2['Info_Label'].map(sorterIndex)
    df_bs_filt_2.sort_values(['Order_id'], inplace=True)
    req_vals_cor = [a*b for a,b in zip(list(df_bs_filt_2[option_slctd2]), list(df_bs_filt_2['Corrector']))]
    df_bs_filt_2[option_slctd2]=req_vals_cor
    # Set Chart X-values / Y-values
    x_labels = list(df_bs_filt_2['Info_Label']) # req_labels
    y_labels = list(df_bs_filt_2[option_slctd2])
    # adjust due to yahoo error
    pos0 = 1
    adjust0 = y_labels[x_labels.index("totalAssets")]-y_labels[x_labels.index("totalCurrentAssets")]
    y_labels[pos0:pos0] = [adjust0]
    x_labels[pos0:pos0] = ['totalNonCurrentAssets']
    req_measures[pos0:pos0] = ['relative']
    # y_labels[pos0:pos0] = [y_labels[x_labels.index("totalCurrentAssets")]]
    pos1 = 3
    adjust1 = y_labels[x_labels.index("totalLiab")]-y_labels[x_labels.index("totalCurrentLiabilities")]
    y_labels[pos1:pos1] = [-adjust1]
    x_labels[pos1:pos1] = ['totalNonCurrentLiabilities']
    req_measures[pos1:pos1] = ['relative']
    y_labels[4] = -y_labels[x_labels.index("totalCurrentLiabilities")]
    #y_labels[pos1 + 2] = [-y_labels[x_labels.index("totalCurrentLiabilities")]]
    # adjust due to yahoo error
    pos2 = 6
    adjust2 = y_labels[x_labels.index("totalStockholderEquity")] - y_labels[x_labels.index("commonStock")]
    y_labels[pos2:pos2] = [adjust2]
    x_labels[pos2:pos2] = ['retainedEarnings']
    req_measures[pos2:pos2] = ['relative']
    # adjust due to yahoo error
    # pos3 = len(y_labels)
    # adjust3 = y_labels[x_labels.index("totalStockholderEquity")]+abs(y_labels[x_labels.index("totalLiab")])
    # y_labels[pos3:pos3] = [adjust3]
    # x_labels[pos3:pos3] = ['totalLiability&StockholdersEquity']
    # req_measures[pos3:pos3] = ['absolute']
    # x_labels[x_labels.index("totalLiab")] = 'netAssetLiability'

    # Graph
    import plotly.graph_objects as go
    fig = go.Figure(go.Waterfall(
        name="20", orientation="v",
        measure=req_measures,
        x=x_labels,
        textposition="outside",
        # text=["+60", "+80", "", "-40", "-20", "Total"],
        y=y_labels,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    # title_ = "Balance Sheet {}".format(option_slctd2)
    # fig.update_layout(
    #     title=title_,
    # )
    cht1_title = 'Balance Sheet for {}'.format(option_slctd3)
    fig.update_layout(title=cht1_title,
                      xaxis_title="Content",
                      yaxis_title="Value",
    # showlegend = True,
                 width = 700, height = 500,template='plotly_dark'
                      )
    return fig


@app.callback(
    Output(component_id='Cash_flow', component_property='figure'),
    [Input(component_id='slct_sector', component_property='value'),
     Input(component_id='slct_year', component_property='value'),
     Input(component_id='slct_company', component_property='value')
     ]
)
def update_graph3(option_slctd1, option_slctd2, option_slctd3):
    print('*'*100)
    # Initial Filter
    df_bs = df_fin_filt_1.copy()
    df_bs = df_bs[df_bs["Sector"] == option_slctd1]
    df_bs = df_bs[df_bs["SGX_CoyName"] == option_slctd3]
    df_bs = df_bs[[option_slctd2, "Info_Label", "Statement_Type"]]
    df_bs[option_slctd2] = df_bs[option_slctd2]#.abs()
    df_bs = df_bs[df_bs["Statement_Type"] == 'Cash_Flow']  # dff = dff[dff["Statement_Type"] == "Income_Statement"]]
    label_sort_ = [
        {'netIncome': 'absolute'},
        {'totalCashFromOperatingActivities': 'relative'},
        {'totalCashflowsFromInvestingActivities': 'relative'},
        {'totalCashFromFinancingActivities': 'relative'},

    ]
    df_corr = pd.DataFrame({
        'netIncome': [1],
        'totalCashFromOperatingActivities': [1],
        'totalCashflowsFromInvestingActivities': [1],
        'totalCashFromFinancingActivities': [1],
    })
    # ERROR req_labels = list(set().union(*(d.keys() for d in label_sort_)))
    req_labels = [i for s in [d.keys() for d in label_sort_] for i in s]
    req_measures = [i for s in [d.values() for d in label_sort_] for i in s]
    # Filter only required
    df_bs_filt = df_bs[df_bs["Info_Label"].isin(req_labels)]
    df_corr = df_corr[list(df_bs_filt["Info_Label"])].T.reset_index()
    df_corr.columns = ['Info_Label', 'Corrector']
    df_bs_filt_2 = pd.merge(df_bs_filt, df_corr, on=['Info_Label'])
    df_bs_filt_2 = convert_df_cols_to_float(df_bs_filt_2, [option_slctd2])
    # Order
    sorterIndex = dict(zip(req_labels, range(len(req_labels))))
    df_bs_filt_2['Order_id'] = df_bs_filt_2['Info_Label'].map(sorterIndex)
    df_bs_filt_2.sort_values(['Order_id'], inplace=True)
    req_vals_cor = [a*b for a,b in zip(list(df_bs_filt_2[option_slctd2]), list(df_bs_filt_2['Corrector']))]
    df_bs_filt_2[option_slctd2]=req_vals_cor
    # Set Chart X-values / Y-values
    x_labels = list(df_bs_filt_2['Info_Label']) # req_labels
    y_labels = list(df_bs_filt_2[option_slctd2])
    # # adjust due to yahoo error
    # pos0 = 1
    # adjust0 = y_labels[x_labels.index("totalAssets")]-y_labels[x_labels.index("totalCurrentAssets")]
    # y_labels[pos0:pos0] = [adjust0]
    # x_labels[pos0:pos0] = ['totalNonCurrentAssets']
    # req_measures[pos0:pos0] = ['relative']
    # # y_labels[pos0:pos0] = [y_labels[x_labels.index("totalCurrentAssets")]]
    # pos1 = 3
    # adjust1 = y_labels[x_labels.index("totalLiab")]-y_labels[x_labels.index("totalCurrentLiabilities")]
    # y_labels[pos1:pos1] = [-adjust1]
    # x_labels[pos1:pos1] = ['totalNonCurrentLiabilities']
    # req_measures[pos1:pos1] = ['relative']
    # y_labels[4] = -y_labels[x_labels.index("totalCurrentLiabilities")]
    # #y_labels[pos1 + 2] = [-y_labels[x_labels.index("totalCurrentLiabilities")]]
    # # adjust due to yahoo error
    # pos2 = 6
    # adjust2 = y_labels[x_labels.index("totalStockholderEquity")] - y_labels[x_labels.index("commonStock")]
    # y_labels[pos2:pos2] = [adjust2]
    # x_labels[pos2:pos2] = ['retainedEarnings']
    # req_measures[pos2:pos2] = ['relative']
    # adjust due to yahoo error
    pos3 = len(y_labels)
    # adjust3 = y_labels[x_labels.index("netIncome")]+y_labels[x_labels.index("totalCashFromOperatingActivities")]
    adjust3 = sum(map(float,y_labels))
    y_labels[pos3:pos3] = [adjust3]
    x_labels[pos3:pos3] = ['Cash']
    req_measures[pos3:pos3] = ['total']

    # Graph
    import plotly.graph_objects as go
    fig = go.Figure(go.Waterfall(
        name="20", orientation="v",
        measure=req_measures,
        x=x_labels,
        textposition="outside",
        # text=["+60", "+80", "", "-40", "-20", "Total"],
        y=y_labels,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    title_ = 'Cash flow for {}'.format(option_slctd3)
    fig.update_layout(
        title=title_,
        # showlegend=True,
        width=700, height=500,
        xaxis_title = "Content",
        yaxis_title = "Value",template='plotly_dark'
    )

    return fig



# Histogram Returns
@app.callback(
    Output(component_id='Trailing_Returns_Distribution', component_property='figure'),
    [Input(component_id='slct_dur', component_property='value'),
        Input(component_id='slct_sector', component_property='value'),
    Input(component_id='slct_company', component_property='value')
]
)
def update_graph_4(option_slctd1, option_slctd2, option_slctd3):
    print(option_slctd1)
    p_sec = 'Healthcare Services'
    p_duration = '1mth'
    dff = df_raw_2_1mth.copy()
    if option_slctd1 =='1mth':
        dff = df_raw_2_1mth.copy()
    if option_slctd1 =='3mth':
        dff = df_raw_2_3mth.copy()
    if option_slctd1 =='6mth':
        dff = df_raw_2_6mth.copy()
    if option_slctd1 =='12mth':
        dff = df_raw_2_12mth.copy()
    # dff = df_filter_1_2.copy()
    dff = dff[['Sector', 'Company_Name', option_slctd1]]
    dff = dff[dff["Sector"] == option_slctd2]
    dff = dff[dff["Company_Name"] == option_slctd3]
    dff = dff.dropna(subset=[option_slctd1])
    print(dff)
    dff[option_slctd2] = dff[option_slctd1].apply(lambda x: x*100)

    fig = px.histogram(dff, option_slctd2, nbins=50) #, histnorm='probability'
    title_ = 'Distribution of Returns for {}'.format(option_slctd3)
    fig.update_layout(
        title=title_,
        # showlegend=True,
        width=700, height=500,
        xaxis_title = "% Return",
        yaxis_title = "Value",template='plotly_dark'
    )
    return fig



# Finanicla Statemtns Benchmark
@app.callback(
    Output(component_id='fin_stms_benchmark', component_property='figure'),
    [Input(component_id='slct_sector', component_property='value'),
     Input(component_id='slct_stmt', component_property='value'),
     Input(component_id='slct_label', component_property='value')
     ]
)
def update_graph5(option_slctd1, option_slctd2, option_slctd3):
    container = "The year chosen by user was: {}".format(option_slctd2)

    dff = df_fin_filt_1.copy()
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
                 width=700, height=500
                 )
    cht1_title = 'Financial Statements Benchmark for {} Sector'.format(option_slctd1)

    # layout = Layout(paper_bgcolor='rgb(0,0,0,0',plot_bgcolor='rgb(0,0,0,0')
    fig.update_layout(title=cht1_title,
                      xaxis_title="Company Names",
                      yaxis_title="Value",legend_orientation='h'
                      )
    # return container, fig
    return fig



# XYZ GRapg
@app.callback(
    Output(component_id='xyz_benchmark', component_property='figure'),
    [Input(component_id='slct_xyz_col_x', component_property='value'),
        Input(component_id='slct_xyz_col_y', component_property='value'),
    Input(component_id='slct_xyz_col_z', component_property='value'),
Input(component_id='slct_sector', component_property='value'),
]
)
def update_graph_6(option_slctd1, option_slctd2, option_slctd3, option_slctd4):

    dff = df_stock_screen.copy()
    dff = dff[dff["Sector"] == option_slctd4]
    dff = dff[[option_slctd1, option_slctd2, option_slctd3, 'Trading Name']]
    dff = dff.dropna(how='any')
    print('3D PLOT'*50)
    print(dff)
    fig = px.scatter_3d(dff, x=option_slctd1,y=option_slctd2,z=option_slctd3, color='Trading Name')

    title_ = 'Dimensions for {} Sector'.format(option_slctd4)
    fig.update_layout(
        title=title_,
        # showlegend=True,
        width=700, height=500,
        xaxis_title = "% Return",
        yaxis_title = "Value",template='plotly_dark'
    )
    return fig


# Market Cap Pie Chart
@app.callback(
    Output(component_id='market_cap', component_property='figure'),
    [
Input(component_id='slct_sector', component_property='value'),
]
)
def update_graph_7(option_slctd1):

    dff = df_stock_screen.copy()
    dff = dff[dff["Sector"] == option_slctd1]
    dff = dff[['Mkt Cap ($M)', 'Trading Name']]
    dff = dff.dropna(how='any')
    print(dff)
    fig = px.pie(dff, values='Mkt Cap ($M)',names='Trading Name',)

    title_ = 'Market Cap for {} Sector'.format(option_slctd1)
    fig.update_layout(
        title=title_,
        # showlegend=True,
        width=700, height=500,
        template='plotly_dark'

    )
    fig.update_traces(textinfo='none', marker=dict(line=dict(color="#000000", width=2))) # eeeeee
    return fig


# 2nd Chart
@app.callback(
    Output(component_id='box_plot_returns', component_property='figure'),
    [Input(component_id='slct_sector', component_property='value'),
     Input(component_id='slct_dur', component_property='value')]
)
def update_graph_8(option_slctd1, option_slctd2):
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
                 width=700, height=500,
                 )

    # fig.add_trace(px.bar())

    cht2_title = '{} Rolling Returns for {} Sector'.format(option_slctd2, option_slctd1)
    fig.update_layout(title=cht2_title,
                      xaxis_title="Company Names",
                      yaxis_title="{} % Change".format(option_slctd2),
                      )
    fig.update_yaxes(range=[-100, 100])





    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
    x=1