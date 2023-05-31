# ML_Graph_Coloring
machine learning treatment of the graph coloring problem

The process consisted of the following steps:
1. Prepare the data: Represent the graph as a data structure, such as an adjacency
matrix, adjacency list, or feature matrix. Encode the nodes and edges, and create
the adjacency list
2. Split the data: Split the data into training, validation, and test sets.
3. Define the GraphSAGE model: Define the GraphSAGE model architecture, which
involves defining the number of layers, the number of hidden units per layer, the
activation function, and the dropout rate.
4. Train the GraphSAGE model: Train the GraphSAGE model on the training data
using stochastic gradient descent (SGD) or a similar optimization algorithm.
5. Evaluate the GraphSAGE model: Evaluate the performance of the GraphSAGE
model on the validation data, and tune the hyperparameters as necessary.
6. Define the Graph Convolutional Network (GCN) model: Define the GCN model
architecture, which involves defining the number of layers, the number of hidden
units per layer, the activation function, and the dropout rate.
7. Train the GCN model: Train the GCN model on the training data using stochastic
gradient descent (SGD) or a similar optimization algorithm.
8. Evaluate the GCN model: Evaluate the performance of the GCN model on the test
data, and tune the hyperparameters as necessary.
9. Compare the both coloring: Compare the coloring generated by the GraphSAGE
model and the coloring generated by the GCN model and comparing them with
the coloring generated by the traditional approach (Welsh Powell algorithm),and
analyze the performance of the overall solution.

--Testing

The Testing process consisted of the following steps:
1-Initialization: An input graph was generated with its corresponding edge connections.
2-Traditional Approach: The input graph was solved using the Welsh Powell algorithm, a traditional graph coloring method. The resulting coloring was saved in the 
test labels, which were later used for visualization.
3-Graph Conversion: The input graph was passed through a conversion function to represent it as an adjacency matrix. Additionally, the degrees of the nodes were 
calculated. Both the adjacency matrix and node degrees were returned as output.
4-Model Initialization: An instance of the trained model was created. The saved parameters from the previously trained model were loaded into the instance.
5-Predictions: The trained model was used to make predictions on the input graph. This process involved feeding the adjacency matrix and node degrees into the 
model, which then generated predicted colors for the graph nodes.
6-Visualization: The input graph, the graph colored using the Welsh Powell algorithm, and the graph colored using the predictions from the model were plotted. 
These visualizations provided a comparison between the traditional approach and the model's predictions.

By following this experimental procedure, the input graph was transformed, analyzed using both traditional and machine learning approaches, and visualized to assess the performance and effectiveness of the trained model in graph coloring.

--illustration

1(Firstly, a set of random graphs was generated, each consisting of 15 nodes. These graphs were represented as adjacency matrices, which captured the connections between nodes, and feature matrices, which contained the degree of each node as its feature.
Next, the generated graphs were solved using the Welsh Powell coloring algorithm. This algorithm assigns colors to the nodes of a graph, aiming to minimize the number of adjacent nodes with the same color. The resulting colors of the nodes were recorded and saved as a labels list.
Finally, to ensure uniformity in the data representation, the adjacency and feature matrices were padded to the same size. Padding is a process of adding dummy values or placeholders to adjust the dimensions of matrices or tensors to a consistent shape.
By following these steps, a dataset was prepared for further analysis and modeling, with consistent adjacency and feature matrices representing the graphs, and a labels list containing the colors assigned to the nodes by the Welsh Powell algorithm.)

2(The dataset was divided into two separate sets: the training data and the evaluation data. The division was performed using a ratio of 80:20, where 80\% of the data was allocated for training, and the remaining 20\% was set aside for evaluation purposes.)

3(The GraphSAGE model was defined with three layers, each consisting of 32, 64, and 128 units respectively. Additionally, L2 regularization technique was applied to the model, which helps prevent overfitting by adding a penalty term to the loss function to discourage large weights.
The Adam optimizer was chosen for training the model, with a learning rate of 0.001. Adam is a popular optimization algorithm that adapts the learning rate based on the gradients of the model parameters, enabling efficient convergence during training.
The model was trained for a total of 200 epochs, iterating over the training data to update the model's parameters and minimize the loss. By training the model for multiple epochs, it had the opportunity to learn and adjust its weights based on the given data.
After the training process, the model was evaluated using the evaluation data to assess its performance. This evaluation step helps in understanding the model's accuracy.
Upon completion of the training and evaluation, the trained model was saved for future testing and inference purposes. Saving the model ensures that the learned parameters and architecture are preserved, allowing for its utilization in subsequent tasks without the need for retraining.
By following these steps, the GraphSAGE model was trained, evaluated, and saved, providing a foundation for further analysis, fine-tuning of hyperparameters if necessary, and subsequent testing on unseen data.)

4(The GCN (Graph Convolutional Network) model was defined with three layers, each consisting of 16, 32, and 64 units respectively. Additionally, L2 regularization technique was applied to the model, which helps prevent overfitting by adding a penalty term to the loss function to discourage large weights.
For optimization, the Adam optimizer was selected with a learning rate of 0.01. Adam is an adaptive optimization algorithm that adjusts the learning rate based on the gradients of the model parameters, enabling efficient convergence during training.
The GCN model was trained for a total of 200 epochs, iterating over the training data to update the model's parameters and minimize the loss. Training the model for multiple epochs allows it to learn and adjust its weights to capture the underlying patterns in the data.
Following the training process, the model was evaluated using the evaluation data to assess its performance. This evaluation step helps in analyzing the model's accuracy, precision, recall, or other relevant metrics, enabling insights into its predictive capabilities and identification of potential areas for hyperparameter tuning.
After completing the training and evaluation, the trained GCN model was saved for future testing and inference. Saving the model preserves its learned parameters and architecture, allowing for utilization in subsequent tasks without the need for retraining.
By following these steps, the GCN model was defined, trained, evaluated, and saved, providing a foundation for further analysis, potential hyperparameter tuning, and subsequent testing on unseen data.)

