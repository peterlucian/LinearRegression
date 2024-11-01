#include "Eigen/Dense"
#include <iostream>


int main(int argc, char* argv[]){


    // Step 1: Define the data
    Eigen::VectorXd y(5);  // Target vector
    y << 2, 2.8, 3.6, 4.5, 5.1;

    Eigen::MatrixXd X(5, 2);  // Feature matrix with a column for the intercept (ones)
    X << 1, 1,
         1, 2,
         1, 3,
         1, 4,
         1, 5;

    // Initialize parameters (intercept and slope)
    Eigen::VectorXd beta(2);
    beta << 0, 0;

    // Hyperparameters
    double learning_rate = 0.01;
    int num_iterations = 100;

    // Gradient descent loop
    for (int iter = 0; iter < num_iterations; ++iter) {
        // Step 2: Compute predictions
        Eigen::VectorXd y_pred = X * beta;

        // Step 3: Calculate the loss (Mean Squared Error)
        Eigen::VectorXd error = y_pred - y;
        double loss = (error.array().square().sum()) / y.size();

        // Output the loss every 100 iterations
        if (iter % 10 == 0) {
            std::cout << "Iteration " << iter << ", Loss: " << loss << std::endl;
        }

        // Step 4: Calculate gradients
        Eigen::VectorXd gradients = (2.0 / y.size()) * X.transpose() * error;

        // Step 5: Update parameters
        beta -= learning_rate * gradients;
    }

    // Output the final parameters
    std::cout << "Final Intercept (beta_0): " << beta(0) << std::endl;
    std::cout << "Final Slope (beta_1): " << beta(1) << std::endl;

    return 0;
}