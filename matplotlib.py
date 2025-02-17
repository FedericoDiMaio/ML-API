import numpy as np
from sklearn.model_selection import learning_curve
import plotly.graph_objects as go


def plot_learning_curves(model, X_train, y_train):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=2,
        scoring='accuracy'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='blue'),
        error_y=dict(type='data', array=train_std, visible=True)
    ))

    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_mean,
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='orange'),
        error_y=dict(type='data', array=val_std, visible=True)
    ))

    fig.update_layout(
        title='Learning Curves',
        xaxis_title='Training Examples',
        yaxis_title='Score',
        legend=dict(x=0.75, y=0.15),
        template='plotly_white',
        width=800,
        height=600
    )

    fig.show()
