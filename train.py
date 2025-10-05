from sklearn.tree import DecisionTreeRegressor
import misc # Our custom functions

def main():
    # Load and preprocess data using functions from misc.py
    df = misc.load_data()
    X_train, X_test, y_train, y_test = misc.preprocess_data(df)

    # Initialize the Decision Tree Regressor model
    dt_model = DecisionTreeRegressor(random_state=42)

    # Train the model
    trained_model = misc.train_model(dt_model, X_train, y_train)

    # Evaluate the model
    mse = misc.evaluate_model(trained_model, X_test, y_test)

    print(f"Decision Tree Regressor - Average MSE on Test Set: {mse:.4f}")

if __name__ == '__main__':
    main()