# What I learned? - Abanole Classification


**- Importing `.data` is no different than importing `.csv`**
Data file extension was `.data`, I first thought I would need to modify the file, but after checking the contents I saw it was basically a comma separated file.

**- Add Header Row to dataframe**
I realized that the data file had no header row, I could simply add it on top of the file manually, but I found a `pandas` solution to it.

```python
data_file = pd.read_csv("data/abalone.data",
                        names=["sex",
                               "length",
                               "diameter",
                               "height",
                               "whole_weight",
                               "shcuked_weight",
                               "viscera_weight",
                               "shell_weight",
                               "rings"
                              ])
```

If you do not specify `names=` for the file you are reading `pandas` assumes that the first line is the header. And you will get n-1 records, since first is used as header. To prevent this either set `names` or set `header=None`.

**- Add Subplot: Plot multiple plots in single figure**

Plotting different plots in the same figure is a good way to visualize the data distributions better. We use `seaborn`'s `.add_plot()` method for that. The first parameter it take is a `3-digit number` such as `RCP`. This RCP creates a Grid of `R rows`, `C columns` and `P` is the position of that subplot.

**Example usage:**

```python
fig = plt.figure(figsize=(16,5))
ax1 = fig.add_subplot(121) # R=1, C=2, P=1, 1st position in the grid of 1x2
ax2 = fig.add_subplot(122) # R=1, C=2, P=2, 2nd position in the grid of 1x2

# Plot weight information
sns.distplot(X_train_explore['whole_weight'],   label='Whole',   ax=ax1)
sns.distplot(X_train_explore['shucked_weight'], label='Shucked', ax=ax1)
sns.distplot(X_train_explore['viscera_weight'], label='Viscera', ax=ax1)
sns.distplot(X_train_explore['shell_weight'],   label='Shell',   ax=ax1)
ax1.legend()

# Plot growth information
sns.distplot(X_train_explore['length'],   label='Length', ax=ax2)
sns.distplot(X_train_explore['diameter'], label='Diameter', ax=ax2)
sns.distplot(X_train_explore['height'],   label='Height', ax=ax2)
ax2.legend()


ax1.set_xlabel('')
ax2.set_xlabel('')
ax1.set_title('Weights Distributions')
ax2.set_title('Size Distributions')
```


**- Map Categorical/Text values with Corresponding Values**

First we create the dictionary of the replacement.
We then assign a new column to the DataFrame.

```python
adulthood_mapper = {"M":0,"F":0,"I":1}
df_explore = df_explore.assign(infant=df_explore.sex.replace(adulthood_mapper))
```


**- Leaving out some part of the data to have a better model**

This is an advice I got from a analysis that is already completed on the same topic. After identifying distributions for Infant and Adult abalones, we seen that infants have measurements with smaller means and higher variance than adults. So to build a better model we may leave out those, infants, and just train our model on `adults`.

**- Having Large Variance**

Large variance values indicate that the parameter which that variance belongs to is less correlated to the target compared to other parameters with smaller variance values.

**- Plot Lower Triangle Heatmap**

In seaborn we can use **Heatmap** to plot the correlation of each parameter with each other but to get a better view we better plot the half of it since the result is already symmetrical.

1. We first create an array of zeroes, with the same shape/size of correlation map using `np.zeroes_like(x)`.
2. We take all upper triangle indices by calling `np.triu_indices_from(x)` and set all to `True`.
3. We set the background of the heatmap to `white`.
4. We set the `mask` parameter. (Also calling `square=True` makes our plot to show as a square)

```python
plt.figure(figsize=(8,8))
corr_map = df_explore.corr()
mask = np.zeros_like(corr_map)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    sns.heatmap(data=corr_map, mask=mask, annot=True, fmt=".2f",square=True)
```


**- Choosing a Scaler**

When applying Feature Scaling there are four main options that comes with `sklearn.preprocessing`:

- StandartScaler: 
      + The StandardScaler assumes your data is normally distributed within each feature and will scale them such that the distribution is now **centred around 0, with a standard deviation of 1.**
      + If your data is normally distributed this scaler is not the best option for you.
- MinMaxScaler
      + It essentially shrinks the range such that the range is now **between 0 and 1** (or -1 to 1 if there are negative values).
      + This scaler works better for cases in which the standard scaler might not work so well. If the distribution is not Gaussian or the standard deviation is very small, the min-max scaler works better.
      + However, it is sensitive to outliers, so if there are outliers in the data, you might want to consider the Robust Scaler below.
- RobustScaler
      + Uses a similar method to the Min-Max scaler but it instead uses the **interquartile range**, rather than the min-max, so that it is robust to outliers. 
      + It is using the less of the data for scaling so it’s more suitable for when there are outliers in the data.
- Normalizer
      + The normalizer scales each value by dividing each value by its magnitude in n-dimensional space for n number of features.
      + Note that the points are all brought within a sphere that is at most 1 away from the origin at any point. Also, the axes that were previously different scales are now all one scale.

**- Model Selection**

We use validation strategies for 3 broad objectives:

- Algorithm selection: Selecting the class of models which is best-suited for the data at hand (tree-based models vs neural networks vs linear models)

- Hyperparameter tuning: Tuning the hyperparameters of a model to increase this predictive power

- Measure of generalizability: Computing an unbiased estimate of the predictive power of the model

**- Selecting target variable**

After I first tried to classify instances by their `sex`  values, the accuracy scores I got was around 0.54 and it was not improving no matter what I do... So, with the help of a friend, I realized that I got into a big assumption, and my assumption was wrong. When I was making that prediction I was saying secretly to myself that "Male, Female and Infant alabones can be distinguished by looking at these variables.". And TA DA! No. Female and Male abalones almost had the same distribution for other parameters and only Infant class is different that Male and Female.

So I wanted to change my perspective, and do a Adult/Infant Classification for the sake of learning.

- Do you analysis well next time.

- Lesson Learned 1: Never stop looking for the reason of your low performing models.
- Lesson Learned 2: Never make assumptions secretly, either prove them or don't trust them.

**- Setting k for KNN**

K is the number of neighbors to consider when setting a class for the current instance. There is no rule of thumb for this value, it is best tried out with different values and using cross-validation. There's only 2 accepted suggestions I have found:
- It should not be power of the number of possible classses. (2 in our case, 1.Adult, 2.Infant)
- It must be less than n, where n is number of instances. And square root of n can be used as a valid value.

So what I did is:
```python
size_of_a_fold = np.sqrt(len(X_train)*0.9)
neighbors_list = list(range(1,size_of_a_fold))
neighbors_list = filter(lambda x: x % 2 != 0, neighbors_list) # take out the ones that are powers of number of classes
```