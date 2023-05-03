import numpy as np
from numpy.linalg import inv

# Reads in data from files and returns them 
# as numpy arrays
def read_in_data(filename):
    with open(filename) as fl:
        year = list()
        value = list()
        for line in fl:
            data = line.strip().split(" ")
            year.append((int)(data[0]))
            value.append((float)(data[1]))
        return np.float64(year), np.float64(value)

# Splits the data into a matrix with k rows.
# Allows for easy use in k-folds CV
def create_k_sets(data, k):
    i = 0
    
    sets = list()
    ret_sets = list()

    for element in data: # loops thorugh rows
        if i > len(data)/k-1: # determines how many rows are in final set matrix
            i = 0
            ret_sets.append(np.array(sets))
            sets = list()
        sets.append(element) # adds entire row to sets list
        i = i + 1
    ret_sets.append(np.array(sets))
    return np.float64(ret_sets)

################################################
#                 Normalization                #
################################################

# Creats a new set without the hold out set
# loops through the rows of the input data and skips of the test set
def cv_train_sets(data, test_set_index):
    scaled_list = list()
    for i in range(0, len(data)):
        if i != test_set_index:
            for element in data[i]:
                scaled_list.append(element)
    return np.array(scaled_list)

# Normalize the data
# Lops through data and scales it using standard scaling
def normalize(data, mean, std):
    arr = data.copy()
    for i in range(len(data)):
        arr[i] = (data[i]-mean)/std
    return arr

################################################
#                   K-Folds                    #
################################################
def k_folds(years, percents, k, d, lambda_value):
    errors = dict()

    for j in range(0, k): # Loops through k for the k-folds CV
        test_set_index = (k-1)-j # index of the hold out set

        # create set for hold out set
        years_test_set = years[test_set_index] 
        percents_test_set = percents[test_set_index]

        # Create new sets without the hold out set
        cv_train_year= cv_train_sets(years, test_set_index)
        cv_train_percent= cv_train_sets(percents, test_set_index)
       
        # Calculate the mean and standard deviation for the inputs and outputs
        year_mean = np.mean(cv_train_year)
        year_std = np.std(cv_train_year)

        percent_mean = np.mean(cv_train_percent)
        percent_std = np.std(cv_train_percent)

        # Normalize sets using the means and standard deviations above
        cv_train_year = normalize(cv_train_year, year_mean, year_std)
        cv_train_percent = normalize(cv_train_percent, percent_mean, percent_std)

        # Normalize the test set using the mean and standard deviation for the train inputs
        cv_test_year = normalize(years_test_set, year_mean, year_std)

        # Loop through each dimension and find the sum of RMSEs for each dimension
        
        if type(lambda_value) != np.float64 and type(lambda_value) != int:
            for i in range(len(lambda_value)):
                w = calculate_w(create_matrix(cv_train_year, 13),cv_train_percent,lambda_value[i])
                error = rmse(len(cv_test_year),12,w,cv_test_year,percents_test_set,percent_mean,percent_std)
                if j==0: # On first fold create an entry in a dictonary
                    errors[lambda_value[i]] = error

                else: # on each fold after the first, update the entry
                    errors.update({lambda_value[i]:errors[lambda_value[i]]+error})
        else:
            for i in range(d+1):
                w = calculate_w(create_matrix(cv_train_year, i+1),cv_train_percent,lambda_value)
                error = rmse(len(cv_test_year),i,w,cv_test_year,percents_test_set,percent_mean,percent_std)

                if j==0: # On first fold create an entry in a dictonary
                    errors[i] = error

                else: # on each fold after the first, update the entry
                    errors.update({i:errors[i]+error})
       
    return errors

#rows correspond to data value 
# columns correspond to the jth-dimesion
def create_matrix(data, d):
    matrix = list()
    # loops through the input data and creates a 35x13 matrix
    for i in range(len(data)):
        row = list()
        # loops through the d-dimensions and adds data[i]^j to the row list
        for j in range(0, d):
            row.append(np.power(data[i], j)) 
        matrix.append(np.array(row)) # turns the row list into an array and ads to the matrix as a row
    return np.array(matrix)

################################################
#                calculate w                   #
################################################
def calculate_w(matrix, y, lambda_value):
    # inverse((Φ.transpose) * Φ + λ*I) * Φ.transpose()) * y-values
    transpose = matrix.transpose()
    return np.matmul(np.matmul(inv(np.matmul(transpose, matrix) + lambda_value*np.identity(len(matrix[0]))), transpose), y)

def calculate_w_star(x, y, d, lambda_value):
    year_mean = np.mean(x)
    year_std = np.std(x)

    percent_mean = np.mean(y)
    percent_std = np.std(y)

    x = normalize(x, year_mean, year_std)
    y = normalize(y, percent_mean, percent_std)
    return calculate_w(create_matrix(x, d+1),y,lambda_value)

################################################
#                Loss Function                 #
################################################
def rmse(m,d,w,x,y,mean,std):
    sum = 0
    for l in range(0,m):
        # Find the Means Square Error
        sum = sum + np.square(y[l] - denormalize(predicted_value(d,w,x[l]),mean,std)) 
    return np.sqrt(sum/m)

def loss(m,d,w,x,y,mean,std,lambda_value):
    sum = 0
    for l in range(0,m):
        # Find the Means Square Error
        sum = sum + np.square(y[l] - denormalize(predicted_value(d,w,x[l]),mean,std)) # predicted value is the 
                                                                                    # output estimated by the algorith in the d-dimension
                                                                                    # and must be descaled for RMSE
    return 0.5 * sum + 0.5 * lambda_value * np.sum(w) # If λ!=0 then the penalization will take effect

# loop through the a range up to the desired dimension
# multiply w of each dimension by the x raised to that same dimension
def predicted_value(d,w,x):
    sum = 0
    for i in range(d+1): 
        sum = sum + (w[i]*np.power(x, i))
    return sum

# Descale the output byt multiplying it by the standard deviation
# and adding the mean
def denormalize(y,mean,std):
    return y*std+mean

################################################
#                   Predict                    #
################################################  

def predict(final, x, y, d, w):
    x_mean = np.mean(x) # calculate mean of inputs
    x_std = np.std(x) # calculate standard deviation of inputs
    x = normalize(final, x_mean, x_std) # Normalize the data using standard scaling

    y_mean = np.mean(y) # Calculate mean of outputs
    y_std = np.std(y) # Calculate standard deviation of outputs

    # loop though the inputs and
    predicted = list()
    for i in range(0, len(x)):
        # Calculate the output value for the onput x[i] in the d dimension
        # Descale that value back into the original scale and append it to a list
        # of predicted values
        predicted.append(denormalize(predicted_value(d,w,x[i]),y_mean,y_std))
    
    return np.array(predicted)

def determine_loss(x, y, d, w_star, lambda_star):
    x_mean = np.mean(x)
    x_std = np.std(x)

    y_mean = np.mean(y)
    y_std = np.std(y)

    x = normalize(x, x_mean, x_std)
    
    return loss(len(y), d, w_star, x, y,y_mean, y_std, lambda_star)

################################################
#              Helper Functions                #
################################################

# Loop through dict and divide errors by the # of k-folds
# Return an array of avg RMSE
def turn_into_array_d_star(errors, k):
    list1 = list()
    for element in errors:
        list1.append(errors.get(element)/k)
    return np.array(list1, dtype='float64')