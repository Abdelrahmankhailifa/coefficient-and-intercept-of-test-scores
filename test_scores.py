import pandas as pd
import numpy as np
from sklearn import linear_model 
import math


def linear_regression():
    df = pd.read_csv("E:\\Downloads\\ML_excel\\test_scores.csv")
    reg = linear_model.LinearRegression()
    reg.fit(df[['math']],df.cs)
    print(reg.coef_, reg.intercept_)

    return reg.coef_, reg.intercept_




def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 1000000
    n = len(x)
    learning_rate = 0.0002
    cost_prev = 0
    for i in range(iterations):
        y_pred = m_curr * x + b_curr
        cost =(1/n)* sum([val**2 for val in (y - y_pred)])
        md = -(2/n)*sum(x*(y-y_pred))
        bd = -(2/n)*sum(y-y_pred)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if (math.isclose(cost,cost_prev, rel_tol=1e-20)):
            break
        cost_prev = cost
    return m_curr , b_curr



if __name__ == "__main__":
    df = pd.read_csv("E:\\Downloads\\ML_excel\\test_scores.csv")
    x = np.array(df["math"])
    y = np.array(df["cs"])

    m, b = gradient_descent(x,y)
    print("Gradient descent function: Coef {} Intercept {}".format(m, b))

    m_reg, b_reg = linear_regression()
    print("Linear regression: Coef {} Intercept {}".format(m_reg, b_reg))