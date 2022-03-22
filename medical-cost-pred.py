"""
Author: Minhyuk Choi
Email: minhyukc@usc.edu
Description: This code provides a web app that make users to input their personal data in order to estimate the medical cost.
             It will also display the graph that shows relationship between age and charges in next 5 years.
"""

from flask import Flask, url_for, render_template, redirect, request, session, send_file
import os
import sqlite3 as sl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
from io import BytesIO
from matplotlib.figure import Figure

matplotlib.use("Agg")

app = Flask(__name__)
db = "insurance.db"


# Renders the home.html
@app.route("/")
def home():
    return render_template("home.html", message="Please provide your information.")


# Get the user information.
@app.route("/action/getinfo", methods=["POST", "GET"])
def set_info():
    if request.method == "POST":
        # If the user is male, set the value equal to 1
        if request.form["sex_request"] == "Male":
            session["sex"] = 1
        else:
            session["sex"] = 0
        # If the user is a smoker, set the value equal to 1
        if request.form["smoker_request"] == "Smoker":
            session["smoker"] = 1
        else:
            session["smoker"] = 0
        # Set session["region"] variable accordingly
        if request.form["region"] == "Southwest":
            session["southwest"] = 1
            session["southeast"] = 0
            session["northwest"] = 0
        elif request.form["region"] == "Southeast":
            session["southwest"] = 0
            session["southeast"] = 1
            session["northwest"] = 0
        elif request.form["region"] == "Northwest":
            session["southwest"] = 0
            session["southeast"] = 0
            session["northwest"] = 1
        else:
            session["southwest"] = 0
            session["southeast"] = 0
            session["northwest"] = 0

        # Calculate BMI
        session["height"] = float(request.form["height"])
        session["weight"] = float(request.form["weight"])
        session["bmi"] = float((session["weight"] / session["height"] / session["height"]) * 703)

        session["username"] = request.form["username"]
        session["age"] = int(request.form["age"])
        session["children"] = int(request.form["children"])

    return render_template("user.html", username=session["username"], cost=linear_regression(session["age"]))


# If the user hits the button, redirect it to data_viz endpoint.
@app.route("/future", methods=["POST", "GET"])
def future_prediction():
    if request.form["submit"]:
        return redirect(url_for("data_viz"))


# Create ndarrays for future 5 years for age and costs.
def future_pred():
    # Create an array for ages
    np_age = [session["age"]]
    for i in range(1, 6, 1):
        j = session["age"] + i
        np_age = np.append(np_age, j)

    # Create an array for medical costs
    np_cost = []
    for i in np_age:
        predicted = linear_regression(i)
        np_cost = np.append(np_cost, predicted)
    np_cost = np.array(np_cost)

    return np_age, np_cost


# Create a plot and save the image.
@app.route("/fig/plot")
def figures():
    age, cost = future_pred()
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(title="Medical cost vs age")
    plt.xlabel("Ages in year")
    plt.ylabel("Estimated medical costs")
    ax.plot(age, cost, marker='o')
    img_bytes = BytesIO()
    fig.savefig(img_bytes)
    img_bytes.seek(0)
    return send_file(img_bytes, mimetype='image/png')


# Template the dataviz.html with a plot image
@app.route("/images/plot")
def data_viz():
    np_age, np_cost = future_pred()
    return render_template("dataviz.html", username=session["username"], age=np_age[5], cost=np_cost[5])


def linear_regression(age):
    conn = sl.connect(db)
    df = pd.read_sql("SELECT * FROM insurance", conn)
    X = df.drop("charges", axis=1)  # Every column but charges
    y = df["charges"]  # Only charges

    # Do the regression
    reg = LinearRegression()
    reg.fit(X.values, y)
    arr = np.array([age, session["sex"], session["bmi"], session["children"], session["smoker"],
                    session["southwest"], session["southeast"], session["northwest"]], ndmin=2)
    prediction = reg.predict(arr)

    conn.close()
    return round(float(prediction), 2)


# Create a database called insurance.db, and add values from csv.
def db_create_database():
    conn = sl.connect(db)
    curs = conn.cursor()
    # Create a new table for the new db
    curs.execute("""
         CREATE TABLE IF NOT EXISTS insurance
              ([age] INT, [sex] INT, [bmi] FLOAT, [children] INT, [smoker] INT, [southwest] INT, [southeast] INT,
              [northwest] INT, [charges] FLOAT)""")
    # CSV to DB
    csv_file = "insurance.csv"
    df = pd.read_csv(csv_file)
    df.to_sql("insurance", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True)
