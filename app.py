
import sys
sys.path.insert(0, "src")  # Add src directory to Python path for module import

# Import required libraries
import streamlit as st  # Web interface framework
import numpy as np      # Numerical computations
import plotly.graph_objects as go  # Interactive plotting
import mlcpp  # Our C++ implementation binding

def matrix_to_numpy(matrix):
    """Convert mlcpp Matrix to numpy array for visualization
    Args:
        matrix: mlcpp.Matrix object
    Returns:
        numpy.ndarray: Converted matrix
    """
    rows, cols = matrix.rows(), matrix.cols()
    result = np.zeros((rows, cols))
    # Copy each element individually due to C++/Python binding
    for i in range(rows):
        for j in range(cols):
            result[i, j] = matrix.at(i, j)
    return result

def numpy_to_matrix(array):
    """Convert numpy array to mlcpp Matrix for computation
    Args:
        array: numpy.ndarray input
    Returns:
        mlcpp.Matrix: Converted matrix
    """
    rows, cols = array.shape
    matrix = mlcpp.Matrix(rows, cols)
    # Set each element individually due to C++/Python binding
    for i in range(rows):
        for j in range(cols):
            val = array[i, j]
            # Two-step assignment required by C++ binding
            ref = matrix.at(i, j)
            ref = val
    return matrix

def main():
    """Main application entry point"""
    # Set up web interface
    st.title("Neural Network Matrix Operations Demo")

    # Matrix Operations Section
    st.header("Matrix Operations")

    # Create two-column layout
    col1, col2 = st.columns(2)

    with col1:
        # Input controls for matrix dimensions
        rows = st.number_input("Rows", min_value=1, value=3)
        cols = st.number_input("Columns", min_value=1, value=3)

    try:
        # Generate random test matrices
        matrix_a_np = np.random.rand(rows, cols)  # NumPy matrix A
        matrix_b_np = np.random.rand(rows, cols)  # NumPy matrix B

        # Convert to C++ matrices for computation
        matrix_a = numpy_to_matrix(matrix_a_np)
        matrix_b = numpy_to_matrix(matrix_b_np)

        # Operation selection dropdown
        operation = st.selectbox("Select Operation", ["Add", "Subtract", "Multiply"])

        if st.button("Compute"):
            try:
                # Perform selected operation using C++ implementation
                if operation == "Add":
                    result = matrix_a + matrix_b
                elif operation == "Subtract":
                    result = matrix_a - matrix_b
                else:  # Multiply
                    # Check dimensions for multiplication
                    if matrix_a.cols() != matrix_b.rows():
                        st.error("Matrix dimensions must match for multiplication")
                        return
                    result = matrix_a * matrix_b

                # Convert result back to NumPy for visualization
                result_matrix = matrix_to_numpy(result)

                # Create interactive heatmap
                fig = go.Figure(data=[go.Heatmap(z=result_matrix)])
                fig.update_layout(title="Result Matrix Heatmap")
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error during computation: {str(e)}")

    except Exception as e:
        st.error(f"Error initializing matrices: {str(e)}")

    # Neural Network Demo Section
    st.header("Neural Network Demo")

    # Create simple feedforward neural network
    nn = mlcpp.NeuralNetwork()
    nn.add_layer(2, 3, mlcpp.ActivationType.ReLU)    # Hidden layer
    nn.add_layer(3, 1, mlcpp.ActivationType.Sigmoid) # Output layer

    # Generate input space for visualization
    x = np.linspace(-5, 5, 20)  # X-axis points
    y = np.linspace(-5, 5, 20)  # Y-axis points
    X, Y = np.meshgrid(x, y)    # Create 2D grid
    Z = np.zeros_like(X)        # Initialize output surface

    # Process each point through neural network
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Create input matrix for current point
            input_matrix = mlcpp.Matrix(1, 2)
            # Get references for modification
            x_val = input_matrix.at(0, 0)
            y_val = input_matrix.at(0, 1)
            # Set input values
            x_val = X[i, j]
            y_val = Y[i, j]
            # Forward pass through network
            output = nn.forward(input_matrix)
            Z[i, j] = output.at(0, 0)

    # Create 3D surface plot of network output
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
    fig.update_layout(
        title="Neural Network Output Surface",
        scene=dict(
            xaxis_title="Input X",
            yaxis_title="Input Y",
            zaxis_title="Output"
        )
    )
    st.plotly_chart(fig)

    # Display project documentation
    st.header("Project Details")
    st.markdown("""
    This demo showcases:
    1. **C++ Matrix Operations**: Efficient implementation with SIMD optimizations
    2. **Neural Network**: Custom implementation with forward propagation
    3. **Python Integration**: Seamless C++/Python binding using pybind11
    4. **Interactive Visualization**: Real-time matrix operations and NN output

    Key technical features:
    - AVX-optimized matrix operations
    - Cache-friendly block matrix multiplication
    - Custom activation functions (ReLU, Sigmoid, Tanh)
    - Memory-aligned data structures
    """)

if __name__ == "__main__":
    main()
