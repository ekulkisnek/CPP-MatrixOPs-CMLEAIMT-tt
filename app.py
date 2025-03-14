import streamlit as st
import numpy as np
import plotly.graph_objects as go
import mlcpp

def matrix_to_numpy(matrix):
    """Convert mlcpp Matrix to numpy array"""
    rows, cols = matrix.rows(), matrix.cols()
    result = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            result[i, j] = matrix.at(i, j)
    return result

def numpy_to_matrix(array):
    """Convert numpy array to mlcpp Matrix"""
    rows, cols = array.shape
    matrix = mlcpp.Matrix(rows, cols)
    # Fill matrix values using a temporary reference
    for i in range(rows):
        for j in range(cols):
            val = array[i, j]
            # Get reference and set in two steps
            ref = matrix.at(i, j)
            ref = val
    return matrix

def main():
    st.title("Neural Network Matrix Operations Demo")

    st.header("Matrix Operations")

    col1, col2 = st.columns(2)

    with col1:
        rows = st.number_input("Rows", min_value=1, value=3)
        cols = st.number_input("Columns", min_value=1, value=3)

    # Create matrices with random values
    matrix_a_np = np.random.rand(rows, cols)
    matrix_b_np = np.random.rand(rows, cols)

    # Convert to our C++ matrices
    matrix_a = numpy_to_matrix(matrix_a_np)
    matrix_b = numpy_to_matrix(matrix_b_np)

    operation = st.selectbox("Select Operation", ["Add", "Subtract", "Multiply"])

    if st.button("Compute"):
        if operation == "Add":
            result = matrix_a + matrix_b
        elif operation == "Subtract":
            result = matrix_a - matrix_b
        else:
            result = matrix_a * matrix_b

        # Visualize results
        result_matrix = matrix_to_numpy(result)

        fig = go.Figure(data=[go.Heatmap(z=result_matrix)])
        fig.update_layout(title="Result Matrix Heatmap")
        st.plotly_chart(fig)

    st.header("Neural Network Demo")

    # Create a simple neural network
    nn = mlcpp.NeuralNetwork()
    nn.add_layer(2, 3, mlcpp.ActivationType.ReLU)
    nn.add_layer(3, 1, mlcpp.ActivationType.Sigmoid)

    # Generate sample data for visualization
    x = np.linspace(-5, 5, 20)
    y = np.linspace(-5, 5, 20)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Process each point through the neural network
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            input_matrix = mlcpp.Matrix(1, 2)
            # Set values using reference variables
            x_val = input_matrix.at(0, 0)
            y_val = input_matrix.at(0, 1)
            x_val = X[i, j]
            y_val = Y[i, j]
            output = nn.forward(input_matrix)
            Z[i, j] = output.at(0, 0)

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

    # Display project details
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