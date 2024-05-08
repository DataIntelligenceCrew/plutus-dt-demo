from dash import Dash, html, dcc, Input, Output, callback, State, dash_table
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dt
import numpy as np

app = Dash(__name__)

# Define global variables for bookkeeping

active_task = None
# Data used to remember historical performance of model over iters
# and to return 
global_data = {
  "flights-classify": {
    "iter": 0,
    "stage": "dt",
    "train_agg_losses": [],
    "test_agg_losses": [],
    "train_losses": None,
    "test_losses": None,
    "slice_train_losses": [],
    "slice_test_losses": [],
    "slices": None,
    "sliceline_stats": None,
    "counts": None,
    "sources_stats": None,
    "dt_result": None,
    "train": None,
    "test": None,
  },
  "flights-regress": {
    "iter": 0,
    "stage": "dt",
    "train_agg_losses": [],
    "test_agg_losses": [],
    "train_losses": None,
    "test_losses": None,
    "slice_train_losses": [],
    "slice_test_losses": [],
    "slices": None,
    "sliceline_stats": None,
    "counts": None,
    "sources_stats": None,
    "dt_result": None,
    "train": None,
    "test": None,
  }
}

# Main app layout

app.layout = html.Div(
  style = {
    'font-family': 'Arial',
  },
  children = [
  html.Link(
      rel='stylesheet',
      href='styles.css'  # Specify the path to your .css file
  ),
  html.Div(
    id = 'title-bar',
    children = [
      html.H1(
        children ='PLUTUS: Understanding Distribution Tailoring for Machine Learning',
      )
    ],
  ),
  html.Div(
    id = 'vis-container',
    children = [
      html.Div(
        id = 'vis-model',
        style = {
          "display": "flex",
          "flexDirection": "column"
        },
        children = [
          html.H2('Model Training'),
          html.Div(
            id = 'vis-model-datachoice',
            children = [
              html.H3('① Choose task & train model'),
              dcc.RadioItems(
                id = 'vis-model-datachoice-radio',
                options = [
                  {
                    "label": html.Span(
                        children = [
                          html.Span("Flights arrival delay regression"),
                          html.Span(
                            className="tooltip",children=['(?)',html.Span(className="tooltiptext",
                                children="US Department of Transport airline on-time performance delay measured by the nearest minute.")]
                          )
                        ]
                      ),
                    "value": "flights-regress"
                  },
                  {
                    "label": html.Span(
                        children = [
                          html.Span("Flights arrival status classification"),
                          html.Span(
                            className="tooltip",children=['(?)',html.Span(className="tooltiptext",
                                children="US Department of Transport airline arrival code classification: on-time, delayed, or cancelled.")]
                          )
                        ]
                      ),
                    "value": "flights-classify"
                  }
                ]
              ),
              html.Div(
                className='button-wrapper',
                children=[
                  html.Div(
                    className='button-inner',
                    children = [
                      html.Button(
                        id = 'vis-model-run-button',
                        children ='Run Model'
                      ),
                      html.Span(
                        className="tooltip",children=['(?)',html.Span(className="tooltiptext",
                          children="XGBoost with default parameters.")]
                      ) 
                    ]
                  )
                ]
              )
            ]
          ),
          html.Div(
            id = 'vis-model-performance',
            children = [
              html.H3('Model performance'),
              html.H4(
                children = [
                  'Average performance by iteration',
                  html.Span(
                    className="tooltip",children=['(?)',html.Span(className="tooltiptext",
                      children="Performance metric is classification accuracy or square regression loss.")]
                  ),
                ]
              ),
              html.Div(
                children = dcc.Graph(id = 'vis-model-acc-graph')
              ),
              html.H4(
                children = [
                  'Performance per top-level slice',
                  html.Span(
                    className="tooltip",children=['(?)',html.Span(className="tooltiptext",
                      children="A top-level slice is a subset of data with a feature-value combination. Numeric types are binned.")]
                  )
                ]
              ),
              html.Div(
                children = dcc.Graph(id = 'vis-model-slice-plot')
              ),
              html.H4(
                children = [
                  'Validation performance of top-level slices by iteration',
                  html.Span(
                    className = "tooltip", children = ['(?)', html.Span(
                      className = 'tooltiptext',
                      children = 'A top-level slice is a subset of data with a feature-value combination. Numeric types are binned.'
                    )]
                  )
                ]
              ),
              html.Div(
                children = dcc.Graph(id = 'vis-model-slice-trend-graph')
              )
            ]
          )
        ]
      ),
      html.Div(
        id = 'vis-sliceline',
        children = [
          html.H2(children = [
            'Sliceline',
            html.Span(
              className="tooltip",children=['(?)',html.Span(className="tooltiptext",
                children="Sliceline efficiently identifies underperforming subsets of data using sparse linear algebra.")]
            )
          ]),
          html.Div(
            id = 'vis-sliceline-info',
            children = [
              dcc.Markdown("$$\\text{score}(S) = \\underbrace{\\alpha \left(\\frac{\\text{avg. slice error}}{\\text{avg. error}} - 1\\right)}_{\\text{poorly performing slices}} - \\underbrace{(1 - \\alpha) \left(\\frac{n}{|S|} - 1\\right)}_{\\text{large slices}}$$", mathjax=True)
            ]
          ),
          html.Div(
            id = 'vis-sliceline-params',
            children = [
              html.H3('② Run Sliceline'),
              html.Div(
                id = 'vis-sliceline-params-alpha',
                children = [
                  html.Label(
                    children = [
                      html.Span('α: Scoring parameter')
                    ]
                  ),
                  dcc.Slider(
                    id = 'vis-sliceline-params-alpha-slider',
                    min = 0.01,
                    max = 1.0,
                    step = 0.01,
                    value = 0.98,
                    marks = {
                      0: '0',
                      0.1: '0.1',
                      0.2: '0.2',
                      0.3: '0.3',
                      0.4: '0.4',
                      0.5: '0.5',
                      0.6: '0.6',
                      0.7: '0.7',
                      0.8: '0.8',
                      0.9: '0.9',
                      1: '1'
                    }
                  ),
                  html.Div(
                    style = {
                      'display': 'flex',
                      'justify-content': 'space-between'
                    },
                    children = [
                      html.Small('Prioritize large slices'),
                      html.Small('Prioritize poorly performing slices')
                    ]
                  )
                ]
              ),
              html.Div(
                id = 'vis-sliceline-params-k',
                children = [
                  html.P(
                    children = [
                      html.Span('k: Total number of slices'),
                      html.Span(
                        className="tooltip",children=['(?)',html.Span(className="tooltiptext",
                          children=html.Span([
                            "Sliceline could return more than k slices if scores are tied."
                          ]))]
                      )
                    ]
                  ),
                  dcc.Slider(
                    id = 'vis-sliceline-params-k-slider',
                    min = 1,
                    max = 10,
                    step = 1,
                    value = 5
                  )
                ]
              ),
              html.Div(
                id = 'vis-sliceline-params-l',
                children = [
                  html.P(
                    children = [
                      html.Span('l: Maximum lattice level'),
                      html.Span(
                        className="tooltip",children=['(?)',html.Span(className="tooltiptext",
                          children=html.Span([
                            "Up to l feature-value combinations, in conjunction, will define a slice. Warning: exponential time and memory cost."
                          ]))]
                      )
                    ]
                  ),
                  dcc.Slider(
                    id = 'vis-sliceline-params-l-slider',
                    min = 1,
                    max = 5,
                    step = 1,
                    value = 1
                  )
                ]
              ),
              html.Div(
                id = 'vis-sliceline-params-min-sup',
                children = [
                  html.P(
                    children = [
                      html.Span('min_sup: Minimum count requirement'),
                      html.Span(
                        className="tooltip",children=['(?)',html.Span(
                            className="tooltiptext",
                            children=html.Span([
                              "Slices smaller than min_sup are pruned to ensure statistical significance."
                            ])
                          )
                        ]
                      )
                    ]
                  ),
                  dcc.Slider(
                    id = 'vis-sliceline-params-min-sup-input',
                    min = 0,
                    max = 100,
                    step = 1,
                    value = 20,
                    marks = {
                      0: '0',
                      10: '10',
                      20: '20',
                      30: '30',
                      40: '40',
                      50: '50',
                      60: '60',
                      70: '70',
                      80: '80',
                      90: '90',
                      100: '100',
                    }
                  )
                ]
              ),
              html.Div(
                className = 'button-wrapper',
                children = html.Div(
                  className = 'button-inner',
                  children = html.Button(
                    id = 'vis-sliceline-run-button',
                    children = 'Run Sliceline'
                  )
                )
              )
            ]
          ),
          #html.Div(
          #  id = 'vis-sliceline-run',
          #  children = [
          #    html.H3('③ Run Sliceline'),
          #    html.Div(
          #      className = 'button-wrapper',
          #      children = html.Button(
          #        id = 'vis-sliceline-run-button',
          #        children = 'Run Sliceline'
          #      )
          #    )
          #  ]
          #),
          html.Div(
            id = 'vis-sliceline-results',
            children = [
              html.H3('③ Set slices query count'),
              dash_table.DataTable(
                id = 'vis-sliceline-results-table'
              )
            ]
          )
        ]
      ),
      html.Div(
        id = 'vis-dt',
        children = [
          html.H2(children = [
            'Data Distribution Tailoring',
            html.Span(
              className = 'tooltip',
              children = [
                '(?)',
                html.Span(
                  className = 'tooltiptext',
                  children = html.Span(
                    'DT algorithms efficiently satisfy quota sampling queries from heterogeneous sources through adaptive sampling.'
                  )
                )
              ]
            )
          ]),
          html.Div(
            id = 'vis-dt-sources',
            children = [
              html.H3('④ Retrieve data sources statistics'),
              html.Div(
                className = 'button-wrapper',
                children = html.Div(
                  className = 'button-inner',
                  children = [
                    html.Button(
                      id = 'vis-dt-sources-button',
                      children = 'Retrieve sources statistics'
                    ),
                    html.Span(
                      className = "tooltip",
                      children = [
                        "(?)",
                        html.Span(
                          className = "tooltiptext",
                          children=html.Span([
                            "For each source, retrieve the probability of randomly sampling the slices identified."
                          ])
                        )
                      ]
                    )                      
                  ]
                )
              ),
              html.Div(
                id = 'vis-dt-sources-graph-container',
                children = [
                  dcc.Graph(
                    id = 'vis-dt-sources-graph'
                  )
                ]
              )
            ]
          ),
          html.Div(
            id = 'vis-dt-algos',
            children = [
              html.H3('⑤ Choose & run DT algorithms'),
              dcc.Checklist(
                id = "vis-dt-algos-radio",
                options = [
                  {
                    "label": html.Span(
                      children = [
                        'Random',
                        html.Span(
                        className="tooltip",children=['(?)',html.Span(className="tooltiptext",
                          children=html.Span([
                            "Baseline that randomly queries a source in each iteration."
                          ]))]
                      )
                      ]
                    ),
                    "value": 'random'
                  },
                  {
                    "label": html.Span(
                      children = [
                        'RatioColl\n',
                        html.Span(
                        className="tooltip",children=['(?)',html.Span(className="tooltiptext",
                          children=html.Span([
                            "Heuristic algorithm that requires precomputed data sources statistics. It prioritizes groups with high query count and groups that are rare to minimize query cost."
                          ]))]
                      )
                      ]
                    ),
                    "value": 'ratiocoll'
                  },
                  {
                    "label": html.Span(
                      children = [
                        'ExploreExploit\n',
                        html.Span(
                        className="tooltip",children=['(?)',html.Span(className="tooltiptext",
                          children=html.Span([
                            "Algorithm that first samples each source to estimate statistics, then calls RatioColl."
                          ]))]
                      )
                      ]
                    ),
                    "value": 'exploreexploit'
                  }
                ],
                value = ['random', 'ratiocoll', 'exploreexploit']
              )
            ]
          ),
          html.Div(
            id = 'vis-dt-run',
            children = [
              html.Div(
                className = 'button-wrapper',
                children = html.Div(
                  className = 'button-inner',
                  children = html.Button(
                  id = 'vis-dt-run-button',
                  children = 'Run DT'
                  ),
                )
              )
            ]
          ),
          html.Div(
            id = 'vis-dt-results',
            children = [
              html.H3(children = [
                'DT result statistics',
                  html.Span(
                  className = 'tooltip',
                  children = [
                    '?',
                    html.Span(
                      className = 'tooltiptext',
                      children = 'Proportion of each source sampled by each of the chosen algorithms. Size of pie chart is proportional to the total query cost issued, denoted in paranthesis.'
                    )
                  ]
                )
              ]),
              html.Div(
                id = 'vis-dt-stats',
                children = [
                  html.H4('Total cost issued by each algorithm and proportion of sources queried'),
                  dcc.Graph(
                    id='vis-dt-stats-sources-chart'
                  )
                ]
              )
            ]
          ),
          html.Div(
            id = 'vis-dt-combine',
            children = [
              html.H3('⑥ Enrich Training Data'),
              html.P("Choose which algorithm's output to use"),
              dcc.Dropdown(
                options = [
                  {'label': 'Random', 'value': 'random'},
                  {'label': 'RatioColl', 'value': 'ratiocoll'},
                  {'label': 'ExploreExploit', 'value': 'exploreexploit'}
                ],
                id = 'vis-dt-combine-dropdown',
                value = 'random'
              ),
              html.Div(
                className = 'button-wrapper',
                children = html.Button(
                  id = "vis-dt-combine-button",
                  children = 'Enrich Data'
                ),
              ),
              html.P(
                id = 'vis-dt-combine-placeholder',
                children = ''
              )
            ]
          )
        ]
      )
    ]
  )
])

# Callbacks

@callback(
  Output('vis-dt-combine-placeholder', 'children'),
  Input('vis-dt-combine-button', 'n_clicks'),
  State('vis-dt-combine-dropdown', 'value'),
  State('vis-model-datachoice-radio', 'value'),
  prevent_initial_call = True
)
def dt_combine(n_clicks, algo, task):
  current_stage = global_data[task]['stage']
  if task is None or current_stage == 'train' or current_stage == 'sliceline':
    raise PreventUpdate
  # Retrieve train set, aug set
  train = global_data[task]['train']
  aug = global_data[task]['dt_result'][algo][0]
  # Combine train, aug
  aug_train = pd.concat([train, aug], ignore_index=True).sample(frac=1)
  # Update train
  global_data[task]['train'] = aug_train
  return ''

# Compute sources statistics
@callback(
  Output('vis-dt-sources-graph', 'figure'),
  Input('vis-dt-sources-button', 'n_clicks'),
  State('vis-model-datachoice-radio', 'value'),
  prevent_initial_call = True
)
def visualize_dt_sources(n_clicks, task):
  # Obtain slices statistics table
  slices = global_data[task]['slices']
  gt_stats = dt.dbsource.construct_stats_table(slices, task)
  # Update gt stats
  global_data[task]['sources_stats'] = gt_stats
  print("gt stats:", gt_stats)
  # Create the graph
  n = len(gt_stats)
  m = len(gt_stats[0])
  fig = go.Figure()
  for j in range(m):
    fig.add_trace(go.Bar(
      x = [f'Source {i}' for i in range(n)],
      y = gt_stats[:,j] * 100,
      name = f'Slice {j}'
    ))
  fig.update_layout(
    barmode='group',
    xaxis_title='Source',
    yaxis_title='Chance of sampling slice',
    legend_title='Slices',
    coloraxis=dict(colorscale='Viridis')
  )
  return fig

algo_to_name = {
  'random': 'Random',
  'ratiocoll': "RatioColl",
  'exploreexploit': "ExploreExploit"
}

@callback(
  Output('vis-dt-stats-sources-chart', 'figure'),
  Input('vis-dt-run-button', 'n_clicks'),
  State('vis-model-datachoice-radio', 'value'),
  State('vis-sliceline-results-table', 'data'),
  State('vis-dt-algos-radio', 'value'),
  prevent_initial_call = True
)
def run_dt(n_clicks, task, data, algos):
  current_stage = global_data[task]['stage']
  if task is None or current_stage == 'train' or current_stage == 'sliceline':
    raise PreventUpdate
  # Run DT using the gt stats
  slices = global_data[task]['slices']
  train = global_data[task]['train']
  query_counts = [row['counts'] for row in data]
  sources_stats = global_data[task]['sources_stats']
  slices_stats = global_data[task]['sliceline_stats']
  explore_scale = len(train) / sum(slices_stats['sizes'])
  costs = dt.const.SOURCES[task]['costs']
  # Run DT & update
  dt_result = dt.pipeline.pipeline_dt_py(slices, costs, query_counts, "random", train, algos, explore_scale, sources_stats, task)
  global_data[task]['dt_result'] = dt_result
  # Aggregate cost visualization
  #stats_vis_children = []
  #cost_data = pd.DataFrame({"Algorithms": algos, "Costs": [dt_result[algo][1]['cost'] for algo in algos]})
  #cost_fig = px.bar(cost_data, x="Algorithms", y="Costs")
  #cost_fig.update_layout(
  #  xaxis_title="Total Cost",
  #  yaxis_title="Algorithms"
  #)
  # Sources subfig
  a = len(algos)
  sources_fig = make_subplots(rows=1, cols=a, 
    specs=[[{"type": "pie"} for _ in range(a)]],
    subplot_titles = [algo_to_name[algo] + '<br>(' + str(int(dt_result[algo][1]['cost'])) + ')' for algo in algos]
  )
  n = dt.const.SOURCES[task]['n']
  for idx, algo in enumerate(algos):
    pie_chart = go.Pie(
      labels = ["Source " + str(i) for i in range(n)], 
      values = dt_result[algo][1]['sources'], 
      domain = {'x': [idx * 1.0 / a, (idx + 1) * 1.0 / a]},
      scalegroup = 'one',
      textinfo = 'none'
    )
    sources_fig.add_trace(pie_chart, row=1, col=idx+1)
  return sources_fig

# Update summary/description of dataset based on choice menu
#@callback(
#  Output('vis-model-dataset-description', 'children'),
#  Input('vis-model-datachoice-radio', 'value')
#)
#def update_dataset_description(task):
#  if task is None:
#    return "Choose a dataset to begin."
#  else:
#    active_task = task
#    return dt.const.DATASET_DESCRIPTIONS[task]

@callback(
  Output('vis-sliceline-results', 'children'),
  Input('vis-sliceline-run-button', 'n_clicks'),
  State('vis-sliceline-params-alpha-slider', 'value'), # alpha
  State('vis-sliceline-params-k-slider', 'value'), # k
  State('vis-sliceline-params-l-slider', 'value'), # max_l
  State('vis-sliceline-params-min-sup-input', 'value'), # min_sup
  State('vis-model-datachoice-radio', 'value'), # task
  prevent_initial_call = True
)
def run_sliceline(n_clicks, alpha, k, max_l, min_sup, task):
  # If current stage is sliceline, reload the most recent slice result
  train_sliceline = global_data[task]['test']
  train_losses = global_data[task]['test_losses']
  slices, slices_stats = dt.pipeline.pipeline_sliceline_dml(train_sliceline, train_losses, alpha, max_l, min_sup, k, task)
  # Update slices
  global_data[task]['slices'] = slices
  global_data[task]['sliceline_stats'] = slices_stats
  # Convert slices to DF
  print("DEBUG LINE")
  print(slices)
  print(task)
  slices_df = dt.utils.recode_slice_to_df(slices, task)
  # Add slice-related info to slices
  slices_df.insert(0, "sizes", slices_stats["sizes"])
  slices_df.insert(0, "errors", slices_stats["errors"])
  slices_df.insert(0, "scores", slices_stats["scores"])
  slices_df.insert(0, "slice_id", list(range(len(slices_df))))
  # Construct slices table
  slices_df.dropna(axis=1, how='all', inplace=True)
  table = dash_table.DataTable(
    id = 'vis-sliceline-results-table',
    data = slices_df.to_dict('records'), 
    columns = [
      {"name": ["DT", "Count"], "id": "counts", "type": "numeric", "editable":True},
      {"name": ["Sliceline", "Id"], "id": "slice_id", "type": "numeric", "editable":False},
      {"name": ["Sliceline", "Score"], "id": "scores", "type": "numeric", 
        "format": dash_table.Format.Format(precision=2, 
          scheme=dash_table.Format.Scheme.fixed), "editable":False},
      {"name": ["Sliceline", "Errors"], "id": "errors", "type": "numeric", 
        "format": dash_table.Format.Format(precision=2, 
          scheme=dash_table.Format.Scheme.decimal_or_exponent), "editable":False},
      {"name": ["Sliceline", "Size"], "id": "sizes", "editable":False}
    ] + [{ "name": ["Features", feature], "id": feature , "editable":False} 
      for feature in slices_df.columns 
      if feature not in ["scores", "errors", "sizes", "slice_id"]],
    merge_duplicate_headers=True,
    style_cell={'textAlign': 'left', 'paddingLeft': '5px'},
    style_table={'overflowX': 'auto'},
    fixed_columns={'headers': False, 'data': 0},
    editable=True,
    style_data_conditional=[
      {
        "if": {'column_editable': True},
        'backgroundColor': 'lightyellow',
        'color': 'black'
      }
    ]
  )
  print(table)
  return [html.H3('③ Set Slices query count'), table]

# Run model button updates model performance and model running report
@callback(
  Output('vis-model-acc-graph', 'figure'),
  Output('vis-model-slice-plot', 'figure'),
  Output('vis-model-slice-trend-graph', 'figure'),
  Input('vis-model-run-button', 'n_clicks'),
  State('vis-model-datachoice-radio', 'value'),
  prevent_initial_call = True
)
def run_model(n_clicks, task):
  global_data[task]['train_agg_losses'].append(0)
  global_data[task]['test_agg_losses'].append(0)
  global_data[task]['slice_test_losses'].append(0)
  global_data[task]['slice_train_losses'].append(0)
  global_data
  # If current stage is train, then reload the most recent running result
  # Load datasets
  train = global_data[task]['train']
  test = global_data[task]['test']
  # Run pipeline function
  train_losses, test_losses, train_stats = dt.pipeline.pipeline_train_py(train, test, 1, task)
  global_data[task]['train_losses'] = train_losses
  global_data[task]['test_losses'] = test_losses
  # Append new aggregate accuracies/losses
  itr = global_data[task]['iter']
  global_data[task]['train_agg_losses'][itr] =  train_stats['agg_train_loss']
  global_data[task]['test_agg_losses'][itr] = train_stats['agg_test_loss']
  # Update aggregate accuracy graph
  agg_graph = get_agg_accuracy_graph(task)
  # Update slice losses
  global_data[task]['slice_train_losses'][itr] = train_stats['slice_train_losses']
  global_data[task]['slice_test_losses'][itr] = train_stats['slice_test_losses']
  # Update slice losses graph
  slice_graph = get_slice_accuracy_graph(task)
  global_data[task]['iter'] += 1
  # Update slice losses trend graph
  traces = []
  for i, train_losses in enumerate(global_data[task]['slice_test_losses']):
    print("SLICES DEBUG")
    print(global_data[task]['slice_train_losses'])
    trace = go.Box(y = train_losses, name='Iter ' + str(i))
    traces.append(trace)
  slice_trend_fig = go.Figure(data=traces)
  return agg_graph, slice_graph, slice_trend_fig

def get_slice_accuracy_graph(task: str):
  # Grab data
  itr = global_data[task]['iter']
  train_ys = global_data[task]['slice_train_losses'][itr]
  test_ys = global_data[task]['slice_test_losses'][itr]
  # Construct figure
  fig = go.Figure()
  # Add train set losses
  fig.add_trace(go.Bar(
    x = list(range(len(train_ys))),
    y = train_ys,
    name = "Train"
  ))
  # Add test set losses
  fig.add_trace(go.Bar(
    x = list(range(len(test_ys))),
    y = test_ys,
    name = "Val"
  ))
  loss_or_acc = 'Accuracy' if dt.const.Y_IS_CATEGORICAL[task] else 'Loss'
  fig.update_layout(
    xaxis_title='Slice Accuracy Rank',
    yaxis_title=loss_or_acc,
    margin = {'l': 0, 'r': 0, "t": 0, 'b': 0}
  )
  return fig


def get_agg_accuracy_graph(task: str):
  # Get x axis
  itr = global_data[task]['iter']
  acc_x = list(range(itr + 1))
  # Get y axes
  acc_train = global_data[task]['train_agg_losses']
  acc_test = global_data[task]['test_agg_losses']
  print("GRAPH DEBUG")
  print(acc_train, acc_test)
  # Get y axes range
  if dt.const.Y_IS_CATEGORICAL[task]:
    range_y = [0,1]
  else:
    range_y = [0, max(np.max(acc_test), np.max(acc_train)) * 1.1]
  # Label of each line
  loss_or_acc = 'Accuracy' if dt.const.Y_IS_CATEGORICAL[task] else 'Loss'
  # Marker scale
  marker_size = 10
  # Construct figure
  fig = go.Figure()
  # Add train line
  fig.add_trace(
      go.Scatter(
        x=acc_x, 
        y=acc_train, 
        mode='lines+markers', 
        marker=dict(size=marker_size), 
        name='Train ' + loss_or_acc
      )
    )
  # Add test line
  fig.add_trace(
    go.Scatter(
      x=acc_x, 
      y=acc_test, 
      mode='lines+markers', 
      marker=dict(size=marker_size), 
      name='Val ' + loss_or_acc
    )
  )
  # Set axis titles
  fig.update_layout(
    xaxis_title='Iteration',
    yaxis_title=loss_or_acc,
    margin = {'l': 0, 'r': 0, "t": 0, 'b': 0}
  )
  # Set x, y axes ranges
  fig.update_yaxes(range=range_y)
  fig.update_xaxes(range=[-0.5, itr + 0.5])
  return fig

def progress_stage(task):
  current = global_data[task]['stage']
  if current == 'dt':
    global_data[task]['stage'] = 'train'
  elif current == 'train':
    global_data[task]['stage'] = 'sliceline'
  elif current == 'sliceline':
    global_data[task]['stage'] = 'dt'

if __name__ == '__main__':
  # Preloading datasets
  tasks = ['flights-regress']
  for task in tasks:
    train = dt.dbsource.get_train(task)
    test = dt.dbsource.get_test(task)
    global_data[task]['train'] = train
    global_data[task]['test'] = test
  
  # Start Dash app
  app.run(debug=True)