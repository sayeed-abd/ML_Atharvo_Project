from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_csv():
    file = request.files['file']
    df = pd.read_csv(file)

    # Split data into features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression model
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    logistic_regression_results = logistic_regression.predict(X_test)

    # Train Random Forest model
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)
    random_forest_results = random_forest.predict(X_test)

    return jsonify({
        'logistic_regression': logistic_regression_results.tolist(),
        'random_forest': random_forest_results.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)