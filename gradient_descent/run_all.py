"""Run Linear, Logistic Regression and MultiLayer Perceptron demos."""

from gradient_descent.linear_regression import main as linear_main
from gradient_descent.logistic_regression import main as logistic_main
from gradient_descent.mlp_backpropagation import main as mlp_main


def main() -> None:
    print("=" * 60)
    print("  GRADIENT DESCENT FROM SCRATCH")
    print("=" * 60)

    print("\n--- 1. Linear Regression (MSE Loss) ---\n")
    linear_main()

    print("\n--- 2. Logistic Regression (Cross-Entropy Loss) ---\n")
    logistic_main()

    print("\n--- 3. MultiLayer Perceptron (Backpropagation) ---\n")
    mlp_main()

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
