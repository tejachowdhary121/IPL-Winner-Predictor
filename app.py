from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('winner_prediction_catboost_classifier.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input values from the form
        batting_team = request.form.get('batting_team')
        bowling_team = request.form.get('bowling_team')
        city = request.form.get('city')

        # Use default values if fields are empty
        def get_float_value(field):
            try:
                return float(request.form.get(field, '0'))
            except ValueError:
                return 0

        runs_left = get_float_value('runs_left')
        balls_left = get_float_value('balls_left')
        wickets_left = get_float_value('wickets_left')
        current_run_rate = get_float_value('current_run_rate')
        required_run_rate = get_float_value('required_run_rate')
        target = get_float_value('target')

        # Prepare DataFrame for prediction
        data = [[batting_team, bowling_team, city, runs_left, balls_left, wickets_left,
                 current_run_rate, required_run_rate, target]]
        columns = ['BattingTeam', 'BowlingTeam', 'City', 'runs_left', 'balls_left',
                   'wickets_left', 'current_run_rate', 'required_run_rate', 'target']
        input_df = pd.DataFrame(data, columns=columns)

        # Make the prediction
        prediction = model.predict_proba(input_df)
        probability1 = int(prediction[0, 0] * 100)
        probability2 = int(prediction[0, 1] * 100)

        return render_template('result.html',
                               team1=batting_team,
                               team2=bowling_team,
                               probability1=probability1,
                               probability2=probability2)
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred. Please check the server logs."


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
