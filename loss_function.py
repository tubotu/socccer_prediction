"""
It is a library used in "main_cross_validation.py" and "main_predict_test.py".
It calculates RPS.
"""

def RPS(y_true, y_pred) -> float:
    """
    Calcurate loss by RPS.
    
    @param y_true: the answer list of target variable. ex.)[(1,0,0),(0,1,0)]
    @param y_pred: the predict list of target variable. ex.)[(0.4,0.4,0.2),(0.2,0.5,0.3)]
    @return: the value of loss
    """
    output = 0.
    data_num = len(y_true)
    for i in range(data_num):
        times = len(y_true[i]) - 1 
        cumulative_sum = 0.
        score = 0.
        for time in range(times):
            cumulative_sum += y_true[i,time] - y_pred[i,time]
            score += cumulative_sum ** 2
        score /= times
        output += score
    
    output /= data_num
    return output
