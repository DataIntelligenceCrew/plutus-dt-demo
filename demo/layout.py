from dash import *


def get_layout() -> html.Div:
    layout = html.Div(
        children=[
            html.Link(rel='stylesheet', herf='styles.css'),
            get_title_bar(),
            get_container()
        ]
    )
    return layout


def get_title_bar() -> html.Div:
    title_bar = html.Div(
        id='title-bar',
        children=[html.H1(children='PLUTUS: Understanding Distribution Tailoring for Machine Learning', )],
    )
    return title_bar


def get_container() -> html.Div:
    container = html.Div(
        id='vis-container',
        children=[
            get_vis_model(),
            get_vis_sliceline(),
            get_vis_dt()
        ]
    )
    return container


def get_vis_model() -> html.Div:
    model = html.Div(
        id='vis-model',
        children=[
            html.H2('Model Training'),
            get_vis_model_data_choice(),
            get_vis_model_performance(),
        ]
    )
    return model


def get_vis_sliceline() -> html.Div:
    sliceline = html.Div(
        id='vis-sliceline',
        children=[
            html.H2(children=['Sliceline',
                              html.Span(
                                  className="tooltip",
                                  children=['(?)',
                                            html.Span(className="tooltiptext",
                                                      children=[
                                                          "Sliceline efficiently identifies underperforming subsets "
                                                          "of data using sparse linear algebra."]
                                                      )
                                            ]
                              )
                              ]),
            get_vis_sliceline_info(),
            get_vis_sliceline_params(),
            get_vis_sliceline_results(),
        ]
    )
    return sliceline


def get_vis_dt() -> html.Div:
    dt = html.Div(
        id='vis-dt',
        children=[
            html.H2(children=[
                'Data Distribution Tailoring',
                html.Span(
                    className='tooltip',
                    children=[
                        '(?)',
                        html.Span(
                            className='tooltiptext',
                            children=html.Span(
                                'DT algorithms efficiently satisfy quota sampling queries from heterogeneous sources through adaptive sampling.'
                            )
                        )
                    ]
                )
            ]),
            get_vis_dt_sources(),
            get_vis_dt_algos(),
            get_vis_dt_run(),
            get_vis_dt_results(),
            get_vis_dt_combine_button()
        ]
    )
    return dt


def get_vis_model_data_choice() -> html.Div:
    data_choice = html.Div(
        html.H3('① Choose task & train model'),
        get_vis_model_data_choice_radio(),
        get_vis_model_data_choice_run_button()
    )
    return data_choice


def get_vis_model_data_choice_radio() -> dcc.RadioItems:
    radio = dcc.RadioItems(
        id='vis-model-datachoice-radio',
        options=[
            {
                "label": html.Span(
                    children=[
                        html.Span("Flights arrival delay regression"),
                        html.Span(
                            className="tooltip",
                            children=['(?)',
                                      html.Span(className="tooltiptext",
                                                children="US Department of Transport airline on-time performance "
                                                         "delay measured by the nearest minute.")]
                        )
                    ]
                ),
                "value": "flights-regress"
            },
            {
                "label": html.Span(
                    children=[
                        html.Span("Flights arrival status classification"),
                        html.Span(
                            className="tooltip",
                            children=['(?)',
                                      html.Span(className="tooltiptext",
                                                children="US Department of Transport airline arrival code "
                                                         "classification: on-time, delayed, or cancelled.")]
                        )
                    ]
                ),
                "value": "flights-classify"
            }
        ]
    )
    return radio


def get_vis_model_data_choice_run_button() -> html.Div:
    button = html.Div(
        className='button-wrapper',
        children=[
            html.Div(
                className='button-inner',
                children=[
                    html.Button(id='vis-model-run-button', children='Run Model'),
                    html.Span(
                        className="tooltip",
                        children=['(?)',
                                  html.Span(className="tooltiptext",
                                            children="XGBoost with default parameters.")]
                    )
                ]
            )
        ]
    )
    return button


def get_vis_model_performance() -> html.Div:
    model_performance = html.Div(
        id='vis-model-performance',
        childre=[
            html.H3('Model performance'),
            get_vis_model_performance_per_iteration(),
            get_vis_model_performance_per_slice(),
            get_vis_model_performance_per_slice_per_iteration(),
        ]
    )
    return model_performance


def get_vis_model_performance_per_iteration() -> html.Div:
    per_iter = html.Div(
        children=[
            html.H4(
                children=[
                    'Average performance by iteration',
                    html.Span(
                        className="tooltip",
                        children=['(?)',
                                  html.Span(className="tooltiptext",
                                            children="Performance metric is classification accuracy or square "
                                                     "regression loss.")]
                    ),
                ]
            ),
            html.Div(
                children=dcc.Graph(id='vis-model-acc-graph')
            )
        ]
    )
    return per_iter


def get_vis_model_performance_per_slice() -> html.Div:
    per_slice = html.Div(
        children=[
            html.H4(
                children=[
                    'Performance per top-level slice',
                    html.Span(
                        className="tooltip", children=['(?)', html.Span(className="tooltiptext",
                                                                        children="A top-level slice is a subset of data with a feature-value combination. Numeric types are binned.")]
                    )
                ]
            ),
            html.Div(
                children=dcc.Graph(id='vis-model-slice-plot')
            ),
        ]
    )
    return per_slice


def get_vis_model_performance_per_slice_per_iteration() -> html.Div:
    slice_per_iter = html.Div(
        children=[
            html.H4(
                children=[
                    'Validation performance of top-level slices by iteration',
                    html.Span(
                        className="tooltip", children=['(?)', html.Span(
                            className='tooltiptext',
                            children='A top-level slice is a subset of data with a feature-value combination. Numeric '
                                     'types are binned.'
                        )]
                    )
                ]
            ),
            html.Div(
                children=dcc.Graph(id='vis-model-slice-trend-graph')
            )
        ]
    )
    return slice_per_iter


def get_vis_sliceline_info() -> html.Div:
    info = html.Div(
        id='vis-sliceline-info',
        children=[
            dcc.Markdown("$$\\text{score}(S) = \\underbrace{\\alpha \left(\\frac{\\text{avg. slice error}}{\\text{"
                         "avg. error}} - 1\\right)}_{\\text{poorly performing slices}} - \\underbrace{(1 - \\alpha) "
                         "\left(\\frac{n}{|S|} - 1\\right)}_{\\text{large slices}}$$", mathjax=True)
        ]
    )
    return info


def get_vis_sliceline_params() -> html.Div:
    params = html.Div(
        id='vis-sliceline-params',
        children=[
            html.H3('② Run Sliceline'),
            html.Div(
                id='vis-sliceline-params-alpha',
                children=[
                    html.Label(
                        children=[
                            html.Span('α: Scoring parameter')
                        ]
                    ),
                    dcc.Slider(
                        id='vis-sliceline-params-alpha-slider',
                        min=0.01,
                        max=1.0,
                        step=0.01,
                        value=0.98,
                        marks={
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
                        style={
                            'display': 'flex',
                            'justify-content': 'space-between'
                        },
                        children=[
                            html.Small('Prioritize large slices'),
                            html.Small('Prioritize poorly performing slices')
                        ]
                    )
                ]
            ),
            html.Div(
                id='vis-sliceline-params-k',
                children=[
                    html.P(
                        children=[
                            html.Span('k: Total number of slices'),
                            html.Span(
                                className="tooltip", children=['(?)', html.Span(className="tooltiptext",
                                                                                children=html.Span([
                                                                                    "Sliceline could return more than k slices if scores are tied."
                                                                                ]))]
                            )
                        ]
                    ),
                    dcc.Slider(
                        id='vis-sliceline-params-k-slider',
                        min=1,
                        max=10,
                        step=1,
                        value=5
                    )
                ]
            ),
            html.Div(
                id='vis-sliceline-params-l',
                children=[
                    html.P(
                        children=[
                            html.Span('l: Maximum lattice level'),
                            html.Span(
                                className="tooltip", children=['(?)', html.Span(className="tooltiptext",
                                                                                children=html.Span([
                                                                                    "Up to l feature-value combinations, in conjunction, will define a slice. Warning: exponential time and memory cost."
                                                                                ]))]
                            )
                        ]
                    ),
                    dcc.Slider(
                        id='vis-sliceline-params-l-slider',
                        min=1,
                        max=5,
                        step=1,
                        value=1
                    )
                ]
            ),
            html.Div(
                id='vis-sliceline-params-min-sup',
                children=[
                    html.P(
                        children=[
                            html.Span('min_sup: Minimum count requirement'),
                            html.Span(
                                className="tooltip", children=['(?)', html.Span(
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
                        id='vis-sliceline-params-min-sup-input',
                        min=0,
                        max=100,
                        step=1,
                        value=20,
                        marks={
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
                className='button-wrapper',
                children=html.Div(
                    className='button-inner',
                    children=html.Button(
                        id='vis-sliceline-run-button',
                        children='Run Sliceline'
                    )
                )
            )
        ]
    )
    return params


def get_vis_sliceline_results() -> html.Div:
    results = html.Div(
        id='vis-sliceline-results',
        children=[
            html.H3('③ Set slices query count'),
            dash_table.DataTable(
                id='vis-sliceline-results-table'
            )
        ]
    )
    return results


def get_vis_dt_sources() -> html.Div:
    sources = html.Div(
        id='vis-dt-sources',
        children=[
            html.H3('④ Retrieve data sources statistics'),
            html.Div(
                className='button-wrapper',
                children=html.Div(
                    className='button-inner',
                    children=[
                        html.Button(
                            id='vis-dt-sources-button',
                            children='Retrieve sources statistics'
                        ),
                        html.Span(
                            className="tooltip",
                            children=[
                                "(?)",
                                html.Span(
                                    className="tooltiptext",
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
                id='vis-dt-sources-graph-container',
                children=[
                    dcc.Graph(
                        id='vis-dt-sources-graph'
                    )
                ]
            )
        ]
    )
    return sources


def get_vis_dt_algos() -> html.Div:
    algos = html.Div(
        id='vis-dt-algos',
        children=[
            html.H3('⑤ Choose & run DT algorithms'),
            dcc.Checklist(
                id="vis-dt-algos-radio",
                options=[
                    {
                        "label": html.Span(
                            children=[
                                'Random',
                                html.Span(
                                    className="tooltip",
                                    children=['(?)',
                                              html.Span(className="tooltiptext",
                                                        children=html.Span([
                                                            "Baseline that randomly queries a source in each iteration."]))]
                                )
                            ]
                        ),
                        "value": 'random'
                    },
                    {
                        "label": html.Span(
                            children=[
                                'RatioColl\n',
                                html.Span(
                                    className="tooltip",
                                    children=['(?)',
                                              html.Span(className="tooltiptext",
                                                        children=html.Span([
                                                            "Heuristic algorithm that requires precomputed data"
                                                            "sources statistics. It prioritizes groups with high "
                                                            "query count and groups that are rare to minimize query "
                                                            "cost."
                                                        ]))]
                                )
                            ]
                        ),
                        "value": 'ratiocoll'
                    },
                    {
                        "label": html.Span(
                            children=[
                                'ExploreExploit\n',
                                html.Span(
                                    className="tooltip",
                                    children=['(?)',
                                              html.Span(className="tooltiptext",
                                                       children=html.Span([
                                                           "Algorithm that first samples each source to estimate "
                                                           "statistics, then calls RatioColl."
                                                       ]))]
                                )
                            ]
                        ),
                        "value": 'exploreexploit'
                    }
                ],
                value=['random', 'ratiocoll', 'exploreexploit']
            )
        ]
    )
    return algos


def get_vis_dt_run() -> html.Div:
    run_button = html.Div(
        id='vis-dt-run',
        children=[
            html.Div(
                className='button-wrapper',
                children=html.Div(
                    className='button-inner',
                    children=html.Button(
                        id='vis-dt-run-button',
                        children='Run DT'
                    ),
                )
            )
        ]
    )
    return run_button


def get_vis_dt_results() -> html.Div:
    results = html.Div(
        id='vis-dt-results',
        children=[
            html.H3(children=[
                'DT result statistics',
                html.Span(
                    className='tooltip',
                    children=[
                        '?',
                        html.Span(
                            className='tooltiptext',
                            children='Proportion of each source sampled by each of the chosen algorithms. Size of pie chart is proportional to the total query cost issued, denoted in paranthesis.'
                        )
                    ]
                )
            ]),
            html.Div(
                id='vis-dt-stats',
                children=[
                    html.H4('Total cost issued by each algorithm and proportion of sources queried'),
                    dcc.Graph(
                        id='vis-dt-stats-sources-chart'
                    )
                ]
            )
        ]
    )
    return results


def get_vis_dt_combine_button() -> html.Div:
    button = html.Div(
        id='vis-dt-combine',
        children=[
            html.H3('⑥ Enrich Training Data'),
            html.P("Choose which algorithm's output to use"),
            dcc.Dropdown(
                options=[
                    {'label': 'Random', 'value': 'random'},
                    {'label': 'RatioColl', 'value': 'ratiocoll'},
                    {'label': 'ExploreExploit', 'value': 'exploreexploit'}
                ],
                id='vis-dt-combine-dropdown',
                value='random'
            ),
            html.Div(
                className='button-wrapper',
                children=html.Button(
                    id="vis-dt-combine-button",
                    children='Enrich Data'
                ),
            ),
            html.P(
                id='vis-dt-combine-placeholder',
                children=''
            )
        ]
    )
    return button
