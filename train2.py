from sklearn.kernel_ridge import KernelRidge
import misc # Our custom functions

def main():
    # Load and preprocess data using functions from misc.py
    df = misc.load_data()
    X_train, X_test, y_train, y_test = misc.preprocess_data(df)

    # Initialize the Kernel Ridge model
    kr_model = KernelRidge(alpha=1.0, kernel='rbf')

    # Train the model
    trained_model = misc.train_model(kr_model, X_train, y_train)

    # Evaluate the model
    mse = misc.evaluate_model(trained_model, X_test, y_test)

    print(f"Kernel Ridge Regressor - Average MSE on Test Set: {mse:.4f}")

if __name__ == '__main__':
    main()
