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
    return html.Div()


def get_vis_dt() -> html.Div:
    return html.Div()


def get_vis_model_data_choice():
    data_choice = html.Div(
        html.H3('① Choose task & train model'),
        get_vis_model_data_choice_radio(),
        get_vis_model_data_choice_run_button()
    )
    return data_choice


def get_vis_model_data_choice_radio():
    radio = dcc.RadioItems(
        id='vis-model-datachoice-radio',
        options=[
            {
                "label": html.Span(children=[
                    html.Span("Flights arrival delay regression"),
                    html.Span(className="tooltip",
                              children=['(?)',
                                        html.Span(className="tooltiptext",
                                                  children="US Department of Transport airline on-time performance "
                                                           "delay measured by the nearest minute.")])
                ]),
                "value": "flights-regression"
            },
            {
                "label": html.Spen(children=[

                ]),
                "value": "fl;ights-classify"
            },
            {
                "label": html.Span(
                    children=[
                        html.Span("Flights arrival status classification"),
                        html.Span(
                            className="tooltip", children=['(?)', html.Span(className="tooltiptext",
                                                                            children="US Department of Transport airline arrival code classification: on-time, delayed, or cancelled.")]
                        )
                    ]
                ),
                "value": "flights-classify"
            }
        ]
    )
