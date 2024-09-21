import json
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from model import train_model, load_data

def generate_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Setosa', 'Versicolor', 'Virginica']
    
    fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=labels,
                    y=labels,
                    hoverongaps = False,
                    texttemplate="%{z}"))
    fig.update_layout(title='Confusion Matrix',
                      xaxis_title='Predicted',
                      yaxis_title='Actual')
    return fig.to_html(full_html=False)

def generate_feature_importance(model, feature_names):
    importances = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
    importances = importances.sort_values('importance', ascending=False)
    
    fig = go.Figure(go.Bar(x=importances['feature'], y=importances['importance']))
    return fig.to_html(full_html=False)

def generate_report():
    model, X_test, y_test = train_model()
    y_pred = model.predict(X_test)
    
    confusion_matrix_chart = generate_confusion_matrix(y_test, y_pred)
    feature_importance_chart = generate_feature_importance(model, X_test.columns)
    
    classification_rep = classification_report(y_test, y_pred, target_names=['Setosa', 'Versicolor', 'Virginica'])
    
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('report_template.html')
    
    html_out = template.render(
        confusion_matrix_chart=confusion_matrix_chart,
        feature_importance_chart=feature_importance_chart,
        classification_report=classification_rep
    )
    
    with open('ml_report.html', 'w') as f:
        f.write(html_out)

if __name__ == "__main__":
    generate_report()
