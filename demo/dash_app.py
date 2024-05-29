import json
import os

from dash import Dash, html, Input, Output, callback, State, dash_table
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dt import *
from dash_data import *
from layout import *

app = Dash(__name__)


# Method for initializing tasks from a config file in directory
def get_dashdata_from_configs(path):
    result = dict()
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".json"):
                file_name = os.path.splitext(file)[0]
                with open(os.path.join(root, file), 'r') as json_file:
                    config = json.load(json_file)
                    simple_task = SimpleTask.new_from_config(config)
                    dash_data = DashData(simple_task)
                    result[file_name] = dash_data
    return result


ALGO_TO_NAME = {
    'random': 'Random',
    'ratiocoll': "RatioColl",
    'exploreexploit': "ExploreExploit"
}
global_data = None


# Callbacks

@callback(
    Output('vis-dt-combine-placeholder', 'children'),
    Input('vis-dt-combine-button', 'n_clicks'),
    State('vis-dt-combine-dropdown', 'value'),
    State('vis-model-datachoice-radio', 'value'),
    prevent_initial_call=True
)
def dt_combine(_, algo, task_key):
    if task_key is None:  # No task is currently active
        raise PreventUpdate
    else:
        # Retrieve train set, aug set
        prev_train = global_data[task_key].train
        aug = global_data[task_key].dt_result[algo]['data']
        # Combine train, aug
        aug_train = pd.concat([prev_train, aug], ignore_index=True).sample(frac=1)
        # Update train
        global_data[task_key].train = aug_train
        return ''


# Compute sources statistics
@callback(
    Output('vis-dt-sources-graph', 'figure'),
    Input('vis-dt-sources-button', 'n_clicks'),
    State('vis-model-datachoice-radio', 'value'),
    prevent_initial_call=True
)
def visualize_dt_sources(_, task_key):
    # Obtain slices statistics table
    slices = global_data[task_key].slices
    sql_readable_slices = global_data[task_key].task.recode_slice_to_sql_readable(slices)
    gt_stats = [source_.slices_count(sql_readable_slices) for source_ in global_data[task_key].task.additional_sources]
    source_sizes = [source_.total_size() for source_ in global_data[task_key].task.additional_sources]
    gt_stats = np.array(gt_stats).astype(np.float64)
    source_sizes = np.array(source_sizes).astype(np.float64)
    print("gt_stats:", gt_stats)
    print("source sizes:", source_sizes)
    gt_stats /= source_sizes[:, np.newaxis]
    print("normalized gt stats:", gt_stats)
    # Update gt stats
    global_data[task_key].sources_stats = gt_stats
    # Create the graph
    n = len(gt_stats)
    m = len(gt_stats[0])
    fig = go.Figure()
    for j in range(m):
        fig.add_trace(go.Bar(
            x=[f'Source {i}' for i in range(n)],
            y=gt_stats[:, j] * 100,
            name=f'Slice {j}'
        ))
    fig.update_layout(
        barmode='group',
        xaxis_title='Source',
        yaxis_title='Percentage chance of sampling slice',
        legend_title='Slices',
        coloraxis=dict(colorscale='Viridis')
    )
    return fig


@callback(
    Output('vis-dt-stats-sources-chart', 'figure'),
    Input('vis-dt-run-button', 'n_clicks'),
    State('vis-model-datachoice-radio', 'value'),
    State('vis-sliceline-results-table', 'data'),
    State('vis-dt-algos-radio', 'value'),
    prevent_initial_call=True
)
def run_dt(_, task_key, sliceline_data, algos):
    task_data = global_data[task_key]
    if task_key is None:
        raise PreventUpdate
    # Run DT using the gt stats
    query_counts = np.array([row['counts'] for row in sliceline_data])
    print("query counts:", query_counts)
    explore_scale = len(task_data.train) / sum(task_data.sliceline_stats['sizes'])
    # Run DT & update
    dt_result = pipeline_dt_py(task_data.task, task_data.slices, query_counts, algos, explore_scale)
    print(dt_result)
    task_data.dt_result = dt_result
    # Sources subfig
    a = len(algos)
    sources_fig = make_subplots(rows=1, cols=a,
                                specs=[[{"type": "pie"} for _ in range(a)]],
                                subplot_titles=[
                                    ALGO_TO_NAME[algo] + '<br>(' + str(int(dt_result[algo]['cost'])) + ')' for algo
                                    in algos]
                                )
    n = len(task_data.task.additional_sources)
    for idx, algo in enumerate(algos):
        pie_chart = go.Pie(
            labels=["Source " + str(i) for i in range(n)],
            values=dt_result[algo]['sources'],
            domain={'x': [idx * 1.0 / a, (idx + 1) * 1.0 / a]},
            scalegroup='one',
            textinfo='none'
        )
        sources_fig.add_trace(pie_chart, row=1, col=idx + 1)
    return sources_fig


@callback(
    Output('vis-sliceline-results', 'children'),
    Input('vis-sliceline-run-button', 'n_clicks'),
    State('vis-sliceline-params-alpha-slider', 'value'),  # alpha
    State('vis-sliceline-params-k-slider', 'value'),  # k
    State('vis-sliceline-params-l-slider', 'value'),  # max_l
    State('vis-sliceline-params-min-sup-input', 'value'),  # min_sup
    State('vis-model-datachoice-radio', 'value'),  # task
    prevent_initial_call=True
)
def run_sliceline(n_clicks, alpha, k, max_l, min_sup, task_key):
    # If current stage is sliceline, reload the most recent slice result
    task_data = global_data[task_key]
    train_sliceline = task_data.test
    train_losses = task_data.test_losses
    sliceline_result = pipeline_sliceline_dml(task_data.task, train_sliceline, train_losses, alpha, max_l, min_sup, k)
    # Update slices
    task_data.slices = sliceline_result['slices']
    print("Slices:")
    print(task_data.slices)
    task_data.sliceline_stats = sliceline_result
    print("Sliceline result:")
    print(sliceline_result)
    # Convert slices to DF
    slices_human_readable = task_data.task.recode_slice_to_human_readable(task_data.slices)
    print("Human readable slices:")
    print(slices_human_readable)
    print("Columns:")
    print(task_data.task.all_column_names())
    slices_df = pd.DataFrame(slices_human_readable, columns=task_data.task.x_column_names())
    # Add slice-related info to slices
    slices_df.insert(0, "sizes", task_data.sliceline_stats["sizes"])
    slices_df.insert(0, "errors", task_data.sliceline_stats["errors"])
    slices_df.insert(0, "scores", task_data.sliceline_stats["scores"])
    slices_df.insert(0, "slice_id", np.arange(slices_df.shape[0]))
    # Construct slices table
    slices_df.dropna(axis=1, how='all', inplace=True)
    table = dash_table.DataTable(
        id='vis-sliceline-results-table',
        data=slices_df.to_dict('records'),
        columns=[
                    {"name": ["DT", "Count"], "id": "counts", "type": "numeric", "editable": True},
                    {"name": ["Sliceline", "Id"], "id": "slice_id", "type": "numeric", "editable": False},
                    {"name": ["Sliceline", "Score"], "id": "scores", "type": "numeric",
                     "format": dash_table.Format.Format(precision=2,
                                                        scheme=dash_table.Format.Scheme.fixed), "editable": False},
                    {"name": ["Sliceline", "Errors"], "id": "errors", "type": "numeric",
                     "format": dash_table.Format.Format(precision=2,
                                                        scheme=dash_table.Format.Scheme.decimal_or_exponent),
                     "editable": False},
                    {"name": ["Sliceline", "Size"], "id": "sizes", "editable": False}
                ] + [{"name": ["Features", feature], "id": feature, "editable": False}
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
    prevent_initial_call=True
)
def run_model(n_clicks, task_key):
    task_data = global_data[task_key]
    task_data.train_agg_losses.append(0)
    task_data.test_agg_losses.append(0)
    task_data.slice_test_losses.append(0)
    task_data.slice_train_losses.append(0)
    # If current stage is train, then reload the most recent running result
    # Load datasets
    # Run pipeline function
    train_result = pipeline_train_py(task_data.task, task_data.train, task_data.test)
    task_data.train_losses = train_result['train_losses']
    task_data.test_losses = train_result['test_losses']
    # Append new aggregate accuracies/losses
    itr = task_data.iter
    task_data.train_agg_losses[itr] = train_result['agg_train_loss']
    task_data.test_agg_losses[itr] = train_result['agg_test_loss']
    # Update aggregate accuracy graph
    agg_graph = get_agg_accuracy_graph(task_key)
    # Update slice losses
    task_data.slice_train_losses[itr] = train_result['slice_train_losses']
    task_data.slice_test_losses[itr] = train_result['slice_test_losses']
    # Update slice losses graph
    slice_graph = get_slice_accuracy_graph(task_key)
    task_data.iter += 1
    # Update slice losses trend graph
    traces = []
    for i, train_losses in enumerate(task_data.slice_test_losses):
        trace = go.Box(y=train_losses, name='Iter ' + str(i))
        traces.append(trace)
    slice_trend_fig = go.Figure(data=traces)
    return agg_graph, slice_graph, slice_trend_fig


def get_slice_accuracy_graph(task_key: str):
    # Grab data
    task_data = global_data[task_key]
    itr = task_data.iter
    train_ys = task_data.slice_train_losses[itr]
    test_ys = task_data.slice_test_losses[itr]
    # Construct figure
    fig = go.Figure()
    # Add train set losses
    fig.add_trace(go.Bar(
        x=list(range(len(train_ys))),
        y=train_ys,
        name="Train"
    ))
    # Add test set losses
    fig.add_trace(go.Bar(
        x=list(range(len(test_ys))),
        y=test_ys,
        name="Val"
    ))
    loss_or_acc = 'Loss'
    fig.update_layout(
        xaxis_title='Slice Accuracy Rank',
        yaxis_title=loss_or_acc,
        margin={'l': 0, 'r': 0, "t": 0, 'b': 0}
    )
    return fig


def get_agg_accuracy_graph(task_key: str):
    # Get x axis
    task_data = global_data[task_key]
    itr = task_data.iter
    acc_x = list(range(itr + 1))
    # Get y axes
    acc_train = task_data.train_agg_losses
    acc_test = task_data.test_agg_losses
    # Get y axes range
    if task_data.task.y_is_categorical:
        range_y = [0, 1]
    else:
        range_y = [0, max(np.max(acc_test), np.max(acc_train)) * 1.1]
    # Label of each line
    loss_or_acc = 'Loss'
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
        margin={'l': 0, 'r': 0, "t": 0, 'b': 0}
    )
    # Set x, y axes ranges
    fig.update_yaxes(range=range_y)
    fig.update_xaxes(range=[-0.5, itr + 0.5])
    return fig


if __name__ == '__main__':
    # Preloading datasets
    global_data = get_dashdata_from_configs('configs-active')

    # Main app layout
    app.layout = get_layout(global_data)

    # Start Dash app
    app.run_server(host='0.0.0.0', port=8050, debug=True)
