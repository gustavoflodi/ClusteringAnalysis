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














































