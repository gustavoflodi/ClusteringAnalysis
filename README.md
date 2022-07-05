# ClusteringAnalysis
Learning concepts and algorithms to solve Clustering problems.

## What is Clustering?

Clustering is an unsupervised approach of **grouping instances or object representations into clusters or classes**, in which the instances are similar.

> Cluster is a subset of objects which are similar to one another in some features and, at the same time, dissimilar to the objects of other clusters.

- It's important to note that the attribution of similarity and the criteria chosen depends highly on the domain of expertise and ideally a professional of the area will be responsable to select the traits to differentiate the objects.
- To cluster objects or images is a very special ability of us humans. It's natural to us to group objects based on **Proximity, Similarity, Common fate or orientation, Common region, Parts of a bigger image, ...**

## What do we want to achieve with Clustering?

- Divide objects first to then identify qualities. 
  - Ex.: What are the common features of the best selling products?
- Obtain insights as to how the data is distributed.
  - Ex.: What is my e-commerce audience like? Their age, their hobbies, their behaviour in the online domain, ...
- Simply to detect different groups of apparent similar objects.
  - Ex.: A biologist wanting to classify different groups of sharks.
- Prior step to further analysis.
  - Ex.: A doctor wanting to explore the caractheristics the patients immune to a certain virus.
- Prior step to other Data Mining algorithms.
  - Ex.: labelling the data with classes to use supervised learning approaches.
- Outlier detection.

## Clustering in a nutshell

1. *Attribute selection.*
  
  What attribute(s) will be used for the similarity measure.

2. *Proximity calculation.*
  
  What is the distance calculation used?

3. *Clustering algorithm.*
  
  Different approaches:
  
  - Partitioning methods.
  - Hierarchical methods.
  - Model-based methods.
  - Density-based methods.
  - Mixes.
4. *Interpretation.*
  
  Evaluation of the partitioning of the subsets. Does it make sense?

## How do we measure similarity?

In a cartesian space with each dimension being one attribute or feature of the dataset, how do we measure similarity and, consequently, dissimilarity between instances. An easy answer is to compute the distance.

**The smaller the distance between the instances' attributes, the bigger the similarity and smaller the dissimilarity.**

### Different distance metrics

1. Euclidian Distance (L2 Norm).
2. Manhattan Distance (L1 Norm).
3. Minkowski Distance.
4. Pearson Correlation.
5. Canberra.
6. Mahalanobis.

### Distance Matrix

Visual representation of distances between instances in a specific distance metric with main diagonal as a vector of zeros.

L1     | p1  | p2  | p3  | ... | pn  |
---    |---  |---  |---  |---  |---  |
**p1** | 0   | d1  | d2  |     | dn1 |
**p2** | d1  | 0   | d3  |     | dn2 |
**p3** | d2  | d3  | 0   |     | dn3 |
...    |     |     |     | 0   |     |
**pn** | dn1 | dn2 | dn3 |     | 0   |

- Calculation should not depend on the choice of measurement units. Thus, if necessary, some normalization may come into hand not to distort cluster allocation with the outweighting of attributes. Such as:
  - Decimal scaling normalization.
  - Min-max normalization.
  - Z-score normalization.

## Set of partitioning based algorithms

The ideia in this specific set is to allocate the instances into a pre-defined number of sets. It's done iteratively by calculating the distances to specific points of the attributes' dimensions space until some criteria is fulfilled.

**Example of algorithm**: K-means and K-medoids (a.k.a PAM - Partition around Medoids)

### K-Means Algorithm

1. Instances are randomly selected as clusters centers.
2. The distances of each instance is then calculated to the centers and they are assigned a cluster.
3. New iteration: calculate new cluster centers, this time points, not instances.
4. Repeat from 2.
5. If the cluster centers don't change, the algorithm has come to an end.

#### Considerations

- Greedy algorithm: does not seek for optimal solution, just for the best at some point in time (local optimal).
- Depending on the dimensionality of the instances, time may be a bottleneck.
- The choice of initial points play an important role to the final clusters and to the performance.
- The choice of number of clusters (k) play an important role on the results as well.
- Converges easily and fast depending on initial choices.
- Outliers influence cluster allocation due to distance distortion.
- Best for compact, distinct and, sometimes, sphere-shaped clusters.

### K-Medoids Algorithm

Alternative to solve K-Means' problems.

- Usage of cluster medoid instead of cluster average center point: solve for outlier influence.
- Cluster centers always represented by instances, not points.
- Replace cluster center instance with another with error is reduced.
- Error calculated with cost function E as the sum of distances of each instance to its cluster center instance across all clusters.

$$ E = ∑  k Clusters  ∑  Inside Cluster J  |p - cj| $$

1. Choose k.
2. Choose cluster centers instances randomly.
3. For each instance, calculate distance to every cluster center and allocate it to the nearest.
4. New iteration, selecting the criteria as Eafter - Ebefore to each possibility to a non-center instance to be the new center.
5. If Eafter - Ebefore < 0, then swap centers.
6. Repeat from 3.
7. Terminates if no changes.

**Differences from K-Means**

- Cluster medoid instead of cluster mean.
- Manhattan distance for distances between instances and clusters centers.
- Cluster centers as instances always.
- Swap cluster centers based on cost function.

## Non-numerical attributes

*How do you measure the distance between the fact that a banana is yellow and an apple is red?* They are nominal attributes and can't be ordered.

Well you use non-numerical distance metrics:

#### Hamming distance for **Nominal Data**:
> Counts the number os attributes different between two instances and add 1 for each one dissimilar. The number is then divided by the number of attributes and falls into the range [0 1] for **Dissimilarity Measure**.

#### Distance of binary data:
> Convert the values to 1 and 0 and put the values in a table.

![image](https://user-images.githubusercontent.com/86890905/177174328-db2eb269-4a4d-414a-8620-7e448f5fe0d4.png)

Xi and Xj different instances with their attributes added to 1 or 0 depending if True or False. Distance between instances as **dissimilarity measure**:

$$ d(Xi, Xj) = (b + c) / (a + b + c + d) $$ or 
$$ d(Xi, Xj) = (b + c) / (a + b + c) $$ if data distribution is not symmetric (skewed). 

#### Distance of ordinal data:
> Convert the ordinal value into a rank assigning to it a number from the range [1 R], with R being the final rank of the data.

- Normalize the data into values of the range [0 1] by doing:

$$ rn = (r - 1) / (R - 1) $$

## Different approaches to solve Clustering Problems

Following the given list before of different approaches, our second one will be **Hierarchical Clustering**. The procedure follows:

1. Compute proximities between instances with the distance metric most appropriate.
2. Starts with each instance as being one cluster.
3. Identify the two closest clusters.
4. Merge them.
5. Repeat 3 and 4 until there's only one cluster remaining.

- One could go **Bottom-up**, merging the clusters as described or **Top-down** (rare), splitting the clusters from one big cluster.

The difference of performance comes from the computation of the proximity of two clusters.

### How to calculate proximity of two clusters

#### Single Linkage

Pick the closest two instances of the two clusters and calculate the distance.

- Tends to chaining.
- If there's noise, it could be non effective.

#### Complete Linkage

Pick the furthest two instances of the two clusters and calculate the distance.

- Tends to break large clusters.
- If there's noise, it could still be effective.

#### Average Linkage

Take the average of computing the distance between all instances of the two clusters.

- Biased to globular clusters.
- If there's noise, it could still be effective.

#### Centroid Linkage

Take the distance between centroids of the two clusters.

- Less popular technique.

#### Ward's Method

Take the sum of the square of the distances between each instance from the two clusters.

- Biased to globular clusters.
- If there's noise, it could still be effective.

## Yet another approach for clustering

Following the set of approaches for clustering algorithms, now we have our third type: **Model-based clustering methods (Gaussian Mixed Model)**.

### Cluster membership

- For k-means and k-medoids, if the instance was within a certain distance from a cluster, the probability of it to belong to that cluster is 1 and to others is 0.
  - It either is or is not. Binary.
  - Each instance belongs to exactly one cluster (partition).
- For the **Gaussian Mixed Models**, an instance is vulnerable to normal distributions in the space, with each normal/bell/gaussian distribution being a cluster. Therefore, it has different values of probability to belong to each cluster and it's not 0 nor 1 anywhere, but continuous in the space.
  - Each instance belongs to every cluster.
  - No strict boundaries between clusters since they are now distributions of probabilities.
  - There is no actual assignment to a single cluster for an instance.
  
### Expectation Maximization Algorithm (EM)

1. Randomly pick k distribution curves by choosing the μ (mean) σ (standard deviation), randomly as well, for each.
2. E-Step: calculate expected cluster memberships.
3. M-Step: re-estimate model parameters (refine memberships).
4. Use Maximum Likelihood Function to describe the probability of the instances to belong to the clusters.
5. Repeat 2, 3 and 4 until function cannot be increased.

In simpler words, we want to **maximize the probabilities** of the instances to belong to the clusters. And this can be done with Bayes' Theorem:
- Probability for instance x~i~ to be allocated to a cluster A:

$$ p(A|x~i~) = (p(x~i~|A) * p(A)) / p(x~i~) $$ 

And p(x~i~) can be considered ~ 1.

- Probability for instance x~i~ to be in a cluster A:

$$ p(x~i~|A) $$

- Probability for any instance to be in a cluster A:

$$ p(A) $$

> Then the ML Function becomes a product across all instances of the probability of it to belong to every cluster.

$$ ML Function: Π~i~ (p(A) * p(x~i~|A) + p(B) * p(x~i~|B) + ...)

#### Considerations of the Algorithm

- If a cluster is too **narrow**: widening it, could increase the contributions from nearby instances.
- If a cluster is too **wide**: separating it in more clusters, could also increase contribution.

#### Comparison of K-means and EM

| k-means                                           | EM                         |
| ---                                               | ---                        |
| Mean                                              | Center μ of Gaussian Curve |
| Cluster membership                                | Standard Deviation σ       |
| Non-Change in Cluster membership as stop criteria | Non-Change of ML Function  |

- Both estimated randomly the mean, the center and the standard deviation.
  - These parameters are later improved through iteration.
- Advantages of EM:
  - It evaluates the k parameters: if it returns a high probability, then it's a good choice. One could also run k-Means before to estimate the center for EM.
  - Robust against outliers: the center is not influenced by the distance of outliers since their probabilities are small.
 
## Fourth set of methods

Now for the final approach, we have **Density based clustering methods**. Its principles follow:

- Clusters are built where instances are dense, i.e., close to each other. Therefore, a pre-defined distance has to be chosen to evidence closeness.
- If an instance has few neighbors, it's an outlier without a cluster.

**Procedure**:

1. Location of instances in coordinate system.
2. Define density measure. And calculate how close the instances are from their neighbors.
3. Determine cluster boundaries. Define a threshold to be the criteria of cluster membership.

### DENCLUE Algorithm

Specific algorithm from the density based methods.

- Each instance posses an influence function.
- The more instances close to that instance the higher the aggregated influence.
- Influence in a point is calculated by summing all influence functions there.
- Two parameters to control:
  - σ Influence, Spread: the higher the sigma, the more you will have overlaps from the influence functions.
  - ξ Threshold for Influence: "water level" that denote sufficient influence. The higher the xii, the smaller the cluster areas.

#### Common influence functions

- Gaussian normal distribution.

$$ f~Gauss~ (x, y) = e^-(d(x, y)^2) / 2σ^2 $$

- Square-function. f~s~ = 0 if d(x, y) > σ otherwise 1.

#### Consideration of DENCLUE

- Domain expert evaluates the clusters' quality.
- Cluster is a continuous not necessarily sphere-shaped area where influence is what surpasses the threshold.
- Clusters could have holes and be complex.
- Outliers' influences remain below threshold.
- Sigma and xii need to be set. And then iterated for optimal solution.

## Validating clusters

After the model is built, we have to validate the final results. In order to do that, we could have:

- Internal parameters: often based in **compactness, connectedness and separation**. Properties of data points to specific parameters.
- External parameters: comparing the results of the clustering to external known results. Expert's analysis.
- Relative criteria: other clustering analyses with different parameters or algorithms resulting in similar patterns.

### How to measure the internal parameters of your clusters

#### Compactness

> Measuring distance inside cluster. If high, that means that intraclass similarity is high.

**1. Cluster density:** 

$$ (1/n) * Σ(xi -c) $$ 

c being the cluster center and xi being an instance.

**2. Variance:** 

$$ Σ(xi -c)^2 $$ 

c being the cluster center and xi being an instance.

**3. Average Variance:** 

$$ (1/n) * Σ(xi -c)^2 $$ 

c being the cluster center and xi being an instance.

- In a chart of **Average Cluster Density X k**, it's important to locate the **elbow** for an optimal number of k.
- To apply the formular above to the whole model, you could take the average in regards to the clusters.

#### Separation

> Measuring distance between clusters. If high, that means that interclass dissimilarity is high.

**1. Single Linkage**: distance between the two closest instances from two different clusters.

**2. Complete Linkage**: distance between two most distant instances.

**3. Average Linkage**: average distance between each instance of one cluster to every instance from other one.

**4. Distance of cluster centers**: distance between the centers of the clusters.

#### Other distance measures

They try to combine compactness and separation into a metric.

- Average cluster density.
- BIC.
- Davies-Bouldin.
- Silhouette.
- Dunn.
- Niva.
- ...




























