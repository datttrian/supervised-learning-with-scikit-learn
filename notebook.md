# Supervised Learning with scikit-learn

## Classification

### k-Nearest Neighbors: Fit

In this exercise, you will build your first classification model using
the `churn_df` dataset, which has been preloaded for the remainder of
the chapter.

The target, `"churn"`, needs to be a single column with the same number
of observations as the feature data. The feature data has already been
converted into `numpy` arrays.

**Instructions**

- Import `KNeighborsClassifier` from `sklearn.neighbors`.
- Instantiate a `KNeighborsClassifier` called `knn` with `6` neighbors.
- Fit the classifier to the data using the `.fit()` method.

**Answer**

```{python}

```

### k-Nearest Neighbors: Predict

Now you have fit a KNN classifier, you can use it to predict the label
of new data points. All available data was used for training, however,
fortunately, there are new observations available. These have been
preloaded for you as `X_new`.

The model `knn`, which you created and fit the data in the last
exercise, has been preloaded for you. You will use your classifier to
predict the labels of a set of new data points:

    X_new = np.array([[30.0, 17.5],
                      [107.0, 24.1],
                      [213.0, 10.9]])

**Instructions**

- Create `y_pred` by predicting the target values of the unseen features
  `X_new`.
- Print the predicted labels for the set of predictions.

**Answer**

```{python}

```

### Train/test split + computing accuracy

It's time to practice splitting your data into training and test sets
with the `churn_df` dataset!

NumPy arrays have been created for you containing the features as `X`
and the target variable as `y`.

**Instructions**

- Import `train_test_split` from `sklearn.model_selection`.
- Split `X` and `y` into training and test sets, setting `test_size`
  equal to 20%, `random_state` to `42`, and ensuring the target label
  proportions reflect that of the original dataset.
- Fit the `knn` model to the training data.
- Compute and print the model's accuracy for the test data.

**Answer**

```{python}

```

### Overfitting and underfitting

Interpreting model complexity is a great way to evaluate supervised
learning performance. Your aim is to produce a model that can interpret
the relationship between features and the target variable, as well as
generalize well when exposed to new observations.

The training and test sets have been created from the `churn_df` dataset
and preloaded as `X_train`, `X_test`, `y_train`, and `y_test`.

In addition, `KNeighborsClassifier` has been imported for you along with
`numpy` as `np`.

**Instructions**

- Create `neighbors` as a `numpy` array of values from `1` up to and
  including `12`.
- Instantiate a KNN classifier, with the number of neighbors equal to
  the `neighbor` iterator.
- Fit the model to the training data.
- Calculate accuracy scores for the training set and test set separately
  using the `.score()` method, and assign the results to the index of
  the `train_accuracies` and `test_accuracies` dictionaries,
  respectively.

**Answer**

```{python}

```

### Visualizing model complexity

Now you have calculated the accuracy of the KNN model on the training
and test sets using various values of `n_neighbors`, you can create a
model complexity curve to visualize how performance changes as the model
becomes less complex!

The variables `neighbors`, `train_accuracies`, and `test_accuracies`,
which you generated in the previous exercise, have all been preloaded
for you. You will plot the results to aid in finding the optimal number
of neighbors for your model.

**Instructions**

- Add a title `"KNN: Varying Number of Neighbors"`.
- Plot the `.values()` method of `train_accuracies` on the y-axis
  against `neighbors` on the x-axis, with a label of
  `"Training Accuracy"`.
- Plot the `.values()` method of `test_accuracies` on the y-axis against
  `neighbors` on the x-axis, with a label of `"Testing Accuracy"`.
- Display the plot.

**Answer**

```{python}

```

## Regression

### Creating features

In this chapter, you will work with a dataset called `sales_df`, which
contains information on advertising campaign expenditure across
different media types, and the number of dollars generated in sales for
the respective campaign. The dataset has been preloaded for you. Here
are the first two rows:

         tv        radio      social_media    sales
    1    13000.0   9237.76    2409.57         46677.90
    2    41000.0   15886.45   2913.41         150177.83

You will use the advertising expenditure as features to predict sales
values, initially working with the `"radio"` column. However, before you
make any predictions you will need to create the feature and target
arrays, reshaping them to the correct format for scikit-learn.

**Instructions**

- Create `X`, an array of the values from the `sales_df` DataFrame's
  `"radio"` column.
- Create `y`, an array of the values from the `sales_df` DataFrame's
  `"sales"` column.
- Reshape `X` into a two-dimensional NumPy array.
- Print the shape of `X` and `y`.

**Answer**

```{python}

```

### Building a linear regression model

Now you have created your feature and target arrays, you will train a
linear regression model on all feature and target values.

As the goal is to assess the relationship between the feature and target
values there is no need to split the data into training and test sets.

`X` and `y` have been preloaded for you as follows:

    y = sales_df["sales"].values
    X = sales_df["radio"].values.reshape(-1, 1)

**Instructions**

- Import `LinearRegression`.
- Instantiate a linear regression model.
- Predict sales values using `X`, storing as `predictions`.

**Answer**

```{python}

```

### Visualizing a linear regression model

Now you have built your linear regression model and trained it using all
available observations, you can visualize how well the model fits the
data. This allows you to interpret the relationship between `radio`
advertising expenditure and `sales` values.

The variables `X`, an array of `radio` values, `y`, an array of `sales`
values, and `predictions`, an array of the model's predicted values for
`y` given `X`, have all been preloaded for you from the previous
exercise.

**Instructions**

- Import `matplotlib.pyplot` as `plt`.
- Create a scatter plot visualizing `y` against `X`, with observations
  in blue.
- Draw a red line plot displaying the predictions against `X`.
- Display the plot.

**Answer**

```{python}

```

### Fit and predict for regression

Now you have seen how linear regression works, your task is to create a
multiple linear regression model using all of the features in the
`sales_df` dataset, which has been preloaded for you. As a reminder,
here are the first two rows:

         tv        radio      social_media    sales
    1    13000.0   9237.76    2409.57         46677.90
    2    41000.0   15886.45   2913.41         150177.83

You will then use this model to predict sales based on the values of the
test features.

`LinearRegression` and `train_test_split` have been preloaded for you
from their respective modules.

**Instructions**

- Create `X`, an array containing values of all features in `sales_df`,
  and `y`, containing all values from the `"sales"` column.
- Instantiate a linear regression model.
- Fit the model to the training data.
- Create `y_pred`, making predictions for `sales` using the test
  features.

**Answer**

```{python}

```

### Regression performance

Now you have fit a model, `reg`, using all features from `sales_df`, and
made predictions of sales values, you can evaluate performance using
some common regression metrics.

The variables `X_train`, `X_test`, `y_train`, `y_test`, and `y_pred`,
along with the fitted model, `reg`, all from the last exercise, have
been preloaded for you.

Your task is to find out how well the features can explain the variance
in the target values, along with assessing the model's ability to make
predictions on unseen data.

**Instructions**

- Import `mean_squared_error`.
- Calculate the model's R-squared score by passing the test feature
  values and the test target values to an appropriate method.
- Calculate the model's root mean squared error using `y_test` and
  `y_pred`.
- Print `r_squared` and `rmse`.

**Answer**

```{python}

```

### Cross-validation for R-squared

Cross-validation is a vital approach to evaluating a model. It maximizes
the amount of data that is available to the model, as the model is not
only trained but also tested on all of the available data.

In this exercise, you will build a linear regression model, then use
6-fold cross-validation to assess its accuracy for predicting sales
using social media advertising expenditure. You will display the
individual score for each of the six-folds.

The `sales_df` dataset has been split into `y` for the target variable,
and `X` for the features, and preloaded for you. `LinearRegression` has
been imported from `sklearn.linear_model`.

**Instructions**

- Import `KFold` and `cross_val_score`.
- Create `kf` by calling `KFold()`, setting the number of splits to six,
  `shuffle` to `True`, and setting a seed of `5`.
- Perform cross-validation using `reg` on `X` and `y`, passing `kf` to
  `cv`.
- Print the `cv_scores`.

**Answer**

```{python}

```

### Analyzing cross-validation metrics

Now you have performed cross-validation, it's time to analyze the
results.

You will display the mean, standard deviation, and 95% confidence
interval for `cv_results`, which has been preloaded for you from the
previous exercise.

`numpy` has been imported for you as `np`.

**Instructions**

- Calculate and print the mean of the results.
- Calculate and print the standard deviation of `cv_results`.
- Display the 95% confidence interval for your results using
  `np.quantile()`.

**Answer**

```{python}

```

### Regularized regression: Ridge

Ridge regression performs regularization by computing the *squared*
values of the model parameters multiplied by alpha and adding them to
the loss function.

In this exercise, you will fit ridge regression models over a range of
different alpha values, and print their \\R^2\\ scores. You will use all
of the features in the `sales_df` dataset to predict `"sales"`. The data
has been split into `X_train`, `X_test`, `y_train`, `y_test` for you.

A variable called `alphas` has been provided as a list containing
different alpha values, which you will loop through to generate scores.

**Instructions**

- Import `Ridge`.
- Instantiate `Ridge`, setting alpha equal to `alpha`.
- Fit the model to the training data.
- Calculate the \\R^2\\ score for each iteration of `ridge`.

**Answer**

```{python}

```

### Lasso regression for feature importance

In the video, you saw how lasso regression can be used to identify
important features in a dataset.

In this exercise, you will fit a lasso regression model to the
`sales_df` data and plot the model's coefficients.

The feature and target variable arrays have been pre-loaded as `X` and
`y`, along with `sales_columns`, which contains the dataset's feature
names.

**Instructions**

- Import `Lasso` from `sklearn.linear_model`.
- Instantiate a Lasso regressor with an alpha of `0.3`.
- Fit the model to the data.
- Compute the model's coefficients, storing as `lasso_coef`.

**Answer**

```{python}

```

## Fine-Tuning Your Model

### Assessing a diabetes prediction classifier

In this chapter you'll work with the `diabetes_df` dataset introduced
previously.

The goal is to predict whether or not each individual is likely to have
diabetes based on the features body mass index (BMI) and age (in years).
Therefore, it is a binary classification problem. A target value of `0`
indicates that the individual does *not* have diabetes, while a value of
`1` indicates that the individual *does* have diabetes.

`diabetes_df` has been preloaded for you as a pandas DataFrame and split
into `X_train`, `X_test`, `y_train`, and `y_test`. In addition, a
`KNeighborsClassifier()` has been instantiated and assigned to `knn`.

You will fit the model, make predictions on the test set, then produce a
confusion matrix and classification report.

**Instructions**

- Import `confusion_matrix` and `classification_report`.
- Fit the model to the training data.
- Predict the labels of the test set, storing the results as `y_pred`.
- Compute and print the confusion matrix and classification report for
  the test labels versus the predicted labels.

**Answer**

```{python}

```

### Building a logistic regression model

In this exercise, you will build a logistic regression model using all
features in the `diabetes_df` dataset. The model will be used to predict
the probability of individuals in the test set having a diabetes
diagnosis.

The `diabetes_df` dataset has been split into `X_train`, `X_test`,
`y_train`, and `y_test`, and preloaded for you.

**Instructions**

- Import `LogisticRegression`.
- Instantiate a logistic regression model, `logreg`.
- Fit the model to the training data.
- Predict the probabilities of each individual in the test set having a
  diabetes diagnosis, storing the array of positive probabilities as
  `y_pred_probs`.

**Answer**

```{python}

```

### The ROC curve

Now you have built a logistic regression model for predicting diabetes
status, you can plot the ROC curve to visualize how the true positive
rate and false positive rate vary as the decision threshold changes.

The test labels, `y_test`, and the predicted probabilities of the test
features belonging to the positive class, `y_pred_probs`, have been
preloaded for you, along with `matplotlib.pyplot` as `plt`.

You will create a ROC curve and then interpret the results.

**Instructions**

- Import `roc_curve`.
- Calculate the ROC curve values, using `y_test` and `y_pred_probs`, and
  unpacking the results into `fpr`, `tpr`, and `thresholds`.
- Plot true positive rate against false positive rate.

**Answer**

```{python}

```

### ROC AUC

The ROC curve you plotted in the last exercise looked promising.

Now you will compute the area under the ROC curve, along with the other
classification metrics you have used previously.

The `confusion_matrix` and `classification_report` functions have been
preloaded for you, along with the `logreg` model you previously built,
plus `X_train`, `X_test`, `y_train`, `y_test`. Also, the model's
predicted test set labels are stored as `y_pred`, and probabilities of
test set observations belonging to the positive class stored as
`y_pred_probs`.

A `knn` model has also been created and the performance metrics printed
in the console, so you can compare the `roc_auc_score`,
`confusion_matrix`, and `classification_report` between the two models.

**Instructions**

- Import `roc_auc_score`.
- Calculate and print the ROC AUC score, passing the test labels and the
  predicted positive class probabilities.
- Calculate and print the confusion matrix.
- Call `classification_report()`.

**Answer**

```{python}

```

### Hyperparameter tuning with GridSearchCV

Now you have seen how to perform grid search hyperparameter tuning, you
are going to build a lasso regression model with optimal hyperparameters
to predict blood glucose levels using the features in the `diabetes_df`
dataset.

`X_train`, `X_test`, `y_train`, and `y_test` have been preloaded for
you. A `KFold()` object has been created and stored for you as `kf`,
along with a lasso regression model as `lasso`.

**Instructions**

- Import `GridSearchCV`.
- Set up a parameter grid for `"alpha"`, using `np.linspace()` to create
  20 evenly spaced values ranging from `0.00001` to `1`.
- Call `GridSearchCV()`, passing `lasso`, the parameter grid, and
  setting `cv` equal to `kf`.
- Fit the grid search object to the training data to perform a
  cross-validated grid search.

**Answer**

```{python}

```

### Hyperparameter tuning with RandomizedSearchCV

As you saw, `GridSearchCV` can be computationally expensive, especially
if you are searching over a large hyperparameter space. In this case,
you can use `RandomizedSearchCV`, which tests a fixed number of
hyperparameter settings from specified probability distributions.

Training and test sets from `diabetes_df` have been pre-loaded for you
as `X_train`. `X_test`, `y_train`, and `y_test`, where the target is
`"diabetes"`. A logistic regression model has been created and stored as
`logreg`, as well as a `KFold` variable stored as `kf`.

You will define a range of hyperparameters and use `RandomizedSearchCV`,
which has been imported from `sklearn.model_selection`, to look for
optimal hyperparameters from these options.

**Instructions**

- Create `params`, adding `"l1"` and `"l2"` as `penalty` values, setting
  `C` to a range of `50` float values between `0.1` and `1.0`, and
  `class_weight` to either `"balanced"` or a dictionary containing
  `0:0.8, 1:0.2`.
- Create the Randomized Search CV object, passing the model and the
  parameters, and setting `cv` equal to `kf`.
- Fit `logreg_cv` to the training data.
- Print the model's best parameters and accuracy score.

**Answer**

```{python}

```

## Preprocessing and Pipelines

### Creating dummy variables

Being able to include categorical features in the model building process
can enhance performance as they may add information that contributes to
prediction accuracy.

The `music_df` dataset has been preloaded for you, and its shape is
printed. Also, `pandas` has been imported as `pd`.

Now you will create a new DataFrame containing the original columns of
`music_df` plus dummy variables from the `"genre"` column.

**Instructions**

- Use a relevant function, passing the entire `music_df` DataFrame, to
  create `music_dummies`, dropping the first binary column.
- Print the shape of `music_dummies`.

**Answer**

```{python}

```

### Regression with categorical features

Now you have created `music_dummies`, containing binary features for
each song's genre, it's time to build a ridge regression model to
predict song popularity.

`music_dummies` has been preloaded for you, along with `Ridge`,
`cross_val_score`, `numpy` as `np`, and a `KFold` object stored as `kf`.

The model will be evaluated by calculating the average RMSE, but first,
you will need to convert the scores for each fold to positive values and
take their square root. This metric shows the average error of our
model's predictions, so it can be compared against the standard
deviation of the target value—`"popularity"`.

**Instructions**

- Create `X`, containing all features in `music_dummies`, and `y`,
  consisting of the `"popularity"` column, respectively.
- Instantiate a ridge regression model, setting `alpha` equal to 0.2.
- Perform cross-validation on `X` and `y` using the ridge model, setting
  `cv` equal to `kf`, and using negative mean squared error as the
  scoring metric.
- Print the RMSE values by converting negative `scores` to positive and
  taking the square root.

**Answer**

```{python}

```

### Dropping missing data

Over the next three exercises, you are going to tidy the `music_df`
dataset. You will create a pipeline to impute missing values and build a
KNN classifier model, then use it to predict whether a song is of the
`"Rock"` genre.

In this exercise specifically, you will drop missing values accounting
for less than 5% of the dataset, and convert the `"genre"` column into a
binary feature.

**Instructions**

- Print the number of missing values for each column in the `music_df`
  dataset, sorted in ascending order.

**Answer**

```{python}

```

### Pipeline for song genre prediction: I

Now it's time to build a pipeline. It will contain steps to impute
missing values using the mean for each feature and build a KNN model for
the classification of song genre.

The modified `music_df` dataset that you created in the previous
exercise has been preloaded for you, along with `KNeighborsClassifier`
and `train_test_split`.

**Instructions**

- Import `SimpleImputer` and `Pipeline`.
- Instantiate an imputer.
- Instantiate a KNN classifier with three neighbors.
- Create `steps`, a list of tuples containing the imputer variable you
  created, called `"imputer"`, followed by the `knn` model you created,
  called `"knn"`.

**Answer**

```{python}

```

### Pipeline for song genre prediction: II

Having set up the steps of the pipeline in the previous exercise, you
will now use it on the `music_df` dataset to classify the genre of
songs. What makes pipelines so incredibly useful is the simple interface
that they provide.

`X_train`, `X_test`, `y_train`, and `y_test` have been preloaded for
you, and `confusion_matrix` has been imported from `sklearn.metrics`.

**Instructions**

- Create a pipeline using the steps you previously defined.
- Fit the pipeline to the training data.
- Make predictions on the test set.
- Calculate and print the confusion matrix.

**Answer**

```{python}

```

### Centering and scaling for regression

Now you have seen the benefits of scaling your data, you will use a
pipeline to preprocess the `music_df` features and build a lasso
regression model to predict a song's loudness.

`X_train`, `X_test`, `y_train`, and `y_test` have been created from the
`music_df` dataset, where the target is `"loudness"` and the features
are all other columns in the dataset. `Lasso` and `Pipeline` have also
been imported for you.

Note that `"genre"` has been converted to a binary feature where `1`
indicates a rock song, and `0` represents other genres.

**Instructions**

- Import `StandardScaler`.
- Create the steps for the pipeline object, a `StandardScaler` object
  called `"scaler"`, and a lasso model called `"lasso"` with `alpha` set
  to `0.5`.
- Instantiate a pipeline with steps to scale and build a lasso
  regression model.
- Calculate the R-squared value on the test data.

**Answer**

```{python}

```

### Centering and scaling for classification

Now you will bring together scaling and model building into a pipeline
for cross-validation.

Your task is to build a pipeline to scale features in the `music_df`
dataset and perform grid search cross-validation using a logistic
regression model with different values for the hyperparameter `C`. The
target variable here is `"genre"`, which contains binary values for rock
as `1` and any other genre as `0`.

`StandardScaler`, `LogisticRegression`, and `GridSearchCV` have all been
imported for you.

**Instructions**

- Build the steps for the pipeline: a `StandardScaler()` object named
  `"scaler"`, and a logistic regression model named `"logreg"`.
- Create the `parameters`, searching 20 equally spaced float values
  ranging from `0.001` to `1.0` for the logistic regression model's `C`
  hyperparameter within the pipeline.
- Instantiate the grid search object.
- Fit the grid search object to the training data.

**Answer**

```{python}

```

### Visualizing regression model performance

Now you have seen how to evaluate multiple models out of the box, you
will build three regression models to predict a song's `"energy"`
levels.

The `music_df` dataset has had dummy variables for `"genre"` added.
Also, feature and target arrays have been created, and these have been
split into `X_train`, `X_test`, `y_train`, and `y_test`.

The following have been imported for you: `LinearRegression`, `Ridge`,
`Lasso`, `cross_val_score`, and `KFold`.

**Instructions**

- Write a for loop using `model` as the iterator, and `model.values()`
  as the iterable.
- Perform cross-validation on the training features and the training
  target array using the model, setting `cv` equal to the `KFold`
  object.
- Append the model's cross-validation scores to the results list.
- Create a box plot displaying the results, with the x-axis labels as
  the names of the models.

**Answer**

```{python}

```

### Predicting on the test set

In the last exercise, linear regression and ridge appeared to produce
similar results. It would be appropriate to select either of those
models; however, you can check predictive performance on the test set to
see if either one can outperform the other.

You will use root mean squared error (RMSE) as the metric. The
dictionary `models`, containing the names and instances of the two
models, has been preloaded for you along with the training and target
arrays `X_train_scaled`, `X_test_scaled`, `y_train`, and `y_test`.

**Instructions**

- Import `mean_squared_error`.
- Fit the model to the scaled training features and the training labels.
- Make predictions using the scaled test features.
- Calculate RMSE by passing the test set labels and the predicted
  labels.

**Answer**

```{python}

```

### Visualizing classification model performance

In this exercise, you will be solving a classification problem where the
`"popularity"` column in the `music_df` dataset has been converted to
binary values, with `1` representing popularity more than or equal to
the median for the `"popularity"` column, and `0` indicating popularity
below the median.

Your task is to build and visualize the results of three different
models to classify whether a song is popular or not.

The data has been split, scaled, and preloaded for you as
`X_train_scaled`, `X_test_scaled`, `y_train`, and `y_test`.
Additionally, `KNeighborsClassifier`, `DecisionTreeClassifier`, and
`LogisticRegression` have been imported.

**Instructions**

- Create a dictionary of `"Logistic Regression"`, `"KNN"`, and
  `"Decision Tree Classifier"`, setting the dictionary's values to a
  call of each model.
- Loop through the values in `models`.
- Instantiate a `KFold` object to perform 6 splits, setting `shuffle` to
  `True` and `random_state` to `12`.
- Perform cross-validation using the model, the scaled training
  features, the target training set, and setting `cv` equal to `kf`.

**Answer**

```{python}

```

### Pipeline for predicting song popularity

For the final exercise, you will build a pipeline to impute missing
values, scale features, and perform hyperparameter tuning of a logistic
regression model. The aim is to find the best parameters and accuracy
when predicting song genre!

All the models and objects required to build the pipeline have been
preloaded for you.

**Instructions**

- Create the steps for the pipeline by calling a simple imputer, a
  standard scaler, and a logistic regression model.
- Create a pipeline object, and pass the `steps` variable.
- Instantiate a grid search object to perform cross-validation using the
  pipeline and the parameters.
- Print the best parameters and compute and print the test set accuracy
  score for the grid search object.

**Answer**

```{python}

```
