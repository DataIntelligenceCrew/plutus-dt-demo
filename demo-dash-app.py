from dash import Dash, html, dcc, Input, Output, callback
import plotly.express as px
import pandas as pd

app = Dash(__name__)

# Global constants

dataset_descriptions = {
  "flights": """
* Dataset: US Bureau of Transportation Statistics [On-Time Performance](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGK)\n
* n = 40mil\n
* Data sources: Split by marketing airline

**X variables**

| **Variable** | **Type** | **Description** |
|---|---|---|
| year | int | 2018 - 2023 |
| month | int | Month |
| day | int | Day of month |
| weekday | int | Day of week |
| scheduled departure time | int | 00:00 to 23:59 |
| marketing carrier | categorical | Airline that sold the tickets (9 total) |
| operating carrier | categorical | Airline that operated the airpline (21 total) |
| origin & destination location | float | longitude & latitude of origin & destination |
| origin & destination state | categorical | State of origin & destination (51 total) |
| distance | int | Travel distance (in miles) |

**y variable**

| **Variable**        | **Type**    | **Values**                  |
|---------------------|-------------|-----------------------------|
| arrival performance | categorical | on-time, delayed, or cancelled |
""",
  "census": """
TODO: write a description for the census data
"""
}

# Main app layout

app.layout = html.Div([
  html.Div(
    id = 'title-bar',
    children = [
      html.H1('End-to-end Model Improvement Demo')
    ]
  ),
  html.Div(
    id = 'vis-container',
    style = {
      'display': 'flex',
      'justify-content': 'space-evenly',
      'align-items': 'stretch'
    },
    children = [
      html.Div(
        id = 'vis-model',
        style = {
          'width': '33%',
        },
        children = [
          html.H2('Model Training'),
          html.Div(
            id = 'vis-model-datachoice',
            children = [
              html.H3('Choose task'),
              dcc.RadioItems(
                id = 'vis-model-datachoice-radio',
                options = [
                  {
                    "label": "Airline on-time performance classification",
                    "value": "flights"
                  },
                  {
                    "label": "Census income regression",
                    "value": "census"
                  }
                ]
              )
            ]
          ),
          html.Div(
            id = 'vis-model-dataset',
            children = [
              html.H3('Dataset summary'),
              dcc.Markdown(
                id = 'vis-model-dataset-description',
                children = "Choose a dataset to begin."
              )
            ]
          ),
          html.Div(
            id = 'vis-model-performance',
            children = [
              html.H3('Model performance over iterations')
            ]
          )
        ]
      ),
      html.Div(
        id = 'vis-sliceline',
        style = {
          'width': '33%',
        },
        children = [
          html.H2('Sliceline: Fast Liear-Algebra Based Slice Finding'),
          html.Div(
            id = 'vis-sliceline-params',
            children = [
              html.H3('Set Sliceline parameters'),
              html.Div(
                id = 'vis-sliceline-params-alpha',
                children = [
                  html.Label(
                    children = [
                      html.B('α'),
                      html.Span(': Adjust size of chosen slices')
                    ]
                  ),
                  dcc.Slider(
                    min = 0.0,
                    max = 1.0,
                    step = 0.01,
                    value = 0.5,
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
                  html.Label(
                    children = [
                      html.B('k'),
                      html.Span(': Total number of slices')
                    ]
                  ),
                  dcc.Slider(
                    min = 1,
                    max = 10,
                    step = 1,
                    value = 1
                  )
                ]
              ),
              html.Div(
                id = 'vis-sliceline-params-l',
                children = [
                  html.Label(
                    children = [
                      html.B('l'),
                      html.Span(': Maximum level (# of features in a slice)')
                    ]
                  ),
                  dcc.Slider(
                    min = 1,
                    max = 5,
                    step = 1,
                    value = 3
                  )
                ]
              )
            ]
          ),
          html.Div(
            id = 'vis-sliceline-run',
            children = [
              html.H3('Run Sliceline'),
              html.Button(children = 'Run Sliceline'),
              html.P('TODO: display result of running sliceline'),
            ]
          )
        ]
      ),
      html.Div(
        id = 'vis-dt',
        style = {
          'width': '33%',
        },
        children = [
          html.H2('Data Distribution Tailoring: Cost-Efficient Data Acquisition'),
          html.H3('Run DT'),
          html.H3('DT Results')
        ]
      )
    ]
  )
])

# Callbacks

@callback(
  Output('vis-model-dataset-description', 'children'),
  Input('vis-model-datachoice-radio', 'value'))
def update_dataset_description(dataset_value):
  if dataset_value is None:
    return "Choose a dataset to begin."
  else:
    return dataset_descriptions[dataset_value]

# Define global variables for bookkeeping

active_task = None
# Data used to remember historical performance of model over iters
# and to return 
global_data = {
  "flights": {
    "iter": 0,
    "model_history": [],
    "slices": None,
    "counts": None,
    "sources_stats": None,
    "dt_stats": None,
  },
  "census": {
    "iter": 0,
    "model_history": [],
    "slices": None,
    "counts": None,
    "sources_stats": None,
    "dt_stats": None,
  },
}

if __name__ == '__main__':
  # Start Dash app
  app.run(debug=True)
  
