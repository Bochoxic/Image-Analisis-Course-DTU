import numpy as np 
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt 
from sklearn import decomposition


#Exercise 1
in_dir = r'/data//'
txt_name = 'irisdata.txt'
import numpy as np
iris_data = np.loadtxt(in_dir + txt_name, comments="%")
# x is a matrix with 50 rows and 4 columns
x = iris_data[0:50, 0:4]

n_feat = x.shape[1]
n_obs = x.shape[0]
print(f"Number of features: {n_feat} and number of observations: {n_obs}")


#Exercise 2
sep_l = x[:, 0]
sep_w = x[:, 1]
pet_l = x[:, 2]
pet_w = x[:, 3]


# Use ddof = 1 to make an unbiased estimate
var_sep_l = sep_l.var(ddof=1)


#Exercise 3
def covar(A,B):
    cv=(1/(len(A)-1))*sum(A*B)
    return cv

cv_length_lenght = covar(sep_l,pet_l)
cv_length_width = covar(sep_l,sep_w)
print(f"Covariance  sepal length and the petal length: {cv_length_lenght}")
print(f"Covariance  sepal length and the sepal width: {cv_length_width}")

# Exericse 4

plt.figure() # Added this to make sure that the figure appear
# Transform the data into a Pandas dataframe
d = pd.DataFrame(x, columns=['Sepal length', 'Sepal width',
'Petal length', 'Petal width'])
sns.pairplot(d)
plt.show()

# Exercise 5
mn = np.mean(x, axis=0)
data = x - mn
c_x = 1 / (50 - 1) * data.T @ data 
print(c_x)
print(np.cov(data.T))


# Exercise 6
values, vectors = np.linalg.eig(c_x)

# Exercise 7 
v_norm = values / values.sum() * 100
plt.plot(v_norm)
plt.xlabel('Principal component')
plt.ylabel('Percent explained variance')
plt.ylim([0, 100])
plt.show()

# Exercise 8 
pc_proj = vectors.T.dot(data.T)
plt.figure() 
d = pd.DataFrame(pc_proj.T, columns=['PC1', 'PC2','PC4', 'PC3'])
sns.pairplot(d)
plt.show()

# Exercise 9 
pca = decomposition.PCA()
pca.fit(x)
values_pca = pca.explained_variance_
exp_var_ratio = pca.explained_variance_ratio_
vectors_pca = pca.components_
data_transform = pca.transform(data)

plt.figure() 
d = pd.DataFrame(data_transform, columns=['PC1', 'PC2',
'PC4', 'PC3'])
sns.pairplot(d)
plt.show()

print(vectors)
print(vectors_pca)