################################################################################
# Backward NNexplainer (Regression Only)
# Interpretation of predictions using backward contribution calculation
# All layers use ReLU activation, NO BIAS
################################################################################
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Input
from keras import Model
from keras import utils
from keras import initializers

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

## BEGIN CLASS ####################################################################
class BackwardNNexplainer():
    def __init__(self, random_state=1, scaler=None):
        self._random_state = random_state
        self.scaler = scaler
        self._model = None
        self._feature_names = None
        self._fmin = []              # min value of each feature
        self._fmax = []              # max value of each feature

    # build a neural network model
    def build_NN(self, X_train, y_train, params):

        # min max of each feature
        for i in range(len(X_train.columns)):
            self._fmin.append(X_train.iloc[:,i].min())
            self._fmax.append(X_train.iloc[:,i].max())

        epochs = params['epochs']
        batch_size = params['batch_size']
        lr = params['lr']
        self._feature_names = params['feature_names']
        X_train_scaled = self.scaler.transform(X_train)

        # Set random seed
        utils.set_random_seed(self._random_state)
        initializer = initializers.he_normal()
        last_layer = len(params['layers'])-1
        activator = 'relu'

        # All layers use ReLU activation, NO BIAS
        self._model = Sequential()
        self._model.add(Input(shape=(X_train_scaled.shape[1],)))
        for i in range(0, len(params['layers'])-1):
            self._model.add(Dense(params['layers'][i],
                                  activation=activator,
                                  use_bias=False,  # NO BIAS
                                  kernel_initializer=initializer))

        # Output layer (linear activation, NO BIAS)
        self._model.add(Dense(params['layers'][last_layer],
                              activation='linear',
                              use_bias=False,  # NO BIAS
                              kernel_initializer=initializer))

        # Compile model
        adam = optimizers.Adam(learning_rate=lr)
        self._model.compile(loss='mse', optimizer=adam, metrics=['mae'])

        # model fitting (learning)
        disp = self._model.fit(X_train_scaled, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.2)

        print("\n" + "="*70)
        print("Model Architecture (NO BIAS):")
        print("="*70)
        self._model.summary()

        print("\nActivation function for each layer:")
        for i, layer in enumerate(self._model.layers):
            config = layer.get_config()
            print(f"Layer {i}: {config.get('activation', 'None')}, use_bias: {config.get('use_bias', True)}")

        return disp

    # Backward explainer for regression ##################################
    def explainer_backward(self, input_data, verbose=True):
        """
        Calculate the contribution of each input feature using the backward chain rule
        Goal: Σ C_xi = y_pred (NO BIAS)
        All layers use ReLU activation
        """
        WEIGHTS = []

        layers = len(self._model.layers)

        # Loading model weights (NO BIAS)
        for layer in self._model.layers:
            weights = layer.get_weights()
            WEIGHTS.append(weights[0])

        if verbose:
            print(f"\n=== Debug ===")

        # Forward propagation (NO BIAS)
        current_input = input_data[np.newaxis]
        activations = [input_data]
        pre_activations = []

        if verbose:
            print(f"\ninput (scaled): {input_data}")

        for ln in range(layers):
            # Pre-activation: z = W^T × h (NO BIAS)
            if ln == 0:
                z = np.dot(WEIGHTS[ln].T, input_data)
            else:
                z = np.dot(WEIGHTS[ln].T, activations[-1])
            pre_activations.append(z)
            if verbose:
                print(f"z{ln} (W^T×h): {z}")

            # Activation
            extractor = Model(inputs=self._model.inputs,
                            outputs=self._model.layers[ln].output)
            h = extractor(current_input)[0].numpy()
            activations.append(h)
            if verbose:
                print(f"h{ln} (after activation): {h}")

        y_pred = activations[-1]
        if isinstance(y_pred, np.ndarray) and len(y_pred) == 1:
            y_pred = y_pred[0]

        if verbose:
            print(f"\nFinal output: {y_pred:.6f}")

        # Backpropagation
        if verbose:
            print(f"\n=== Backpropagation gradient calculation ===")

        # Output layer: linear activation
        grad_z_out = np.ones_like(pre_activations[-1])
        if verbose:
            print(f"output layer z (= y): {pre_activations[-1]}")
            print(f"output layer linear grad: 1.0")
            print(f"∂y/∂z_out: {grad_z_out}")

        # Storing gradients of all layers
        all_grad_z = [None] * layers
        all_grad_z[-1] = grad_z_out

        # Propagate to the last hidden layer
        grad_h_last = np.dot(WEIGHTS[-1], grad_z_out)
        if verbose:
            print(f"∂y/∂h_last = W_out^T × ∂y/∂z_out: {grad_h_last}")

        current_grad_h = grad_h_last

        # Process in each hidden layer
        for ln in range(layers - 2, -1, -1):
            # ReLU gradient: 1 if z > 0, 0 otherwise
            relu_grad = np.where(pre_activations[ln] > 0, 1.0, 0.0)
            if verbose:
                print(f"\nLayer {ln}:")
                print(f"  ReLU grad: {relu_grad}")

            # ∂y/∂z = ∂y/∂h × relu_grad
            grad_z = current_grad_h * relu_grad
            if verbose:
                print(f"  ∂y/∂z{ln}: {grad_z}")
            all_grad_z[ln] = grad_z

            if ln > 0:
                current_grad_h = np.dot(WEIGHTS[ln], grad_z)
                if verbose:
                    print(f"  ∂y/∂h{ln-1}: {current_grad_h}")

        # Contribution calculation (NO BIAS)
        if verbose:
            print(f"\n=== Contribution calculation ===")
        contributions = np.zeros(len(input_data))

        grad_z_first = all_grad_z[0]
        if verbose:
            print(f"First layer gradient: {grad_z_first}")

        # Contribution of the input
        for i in range(len(input_data)):
            contrib = 0.0
            for j in range(len(grad_z_first)):
                # C_xi += ∂y/∂z₀[j] × W₀[i,j] × xi
                term = grad_z_first[j] * WEIGHTS[0][i, j] * input_data[i]
                contrib += term
                if verbose:
                    print(f"  C_x{i} += ∂y/∂z0[{j}]({grad_z_first[j]:.4f}) × W0[{i},{j}]({WEIGHTS[0][i,j]:.4f}) × x{i}({input_data[i]:.4f}) = {term:.6f}")
            contributions[i] = contrib
            if verbose:
                print(f"  Total C_x{i} = {contrib:.6f}")

        if verbose:
            print(f"\nContribution of the input: {contributions}")
            print(f"Sum of contributions: {contributions.sum():.6f}")
            print(f"Predicted output: {y_pred:.6f}")
            print(f"Difference: {abs(contributions.sum() - y_pred):.10f}")

        return contributions, y_pred

    # plot the contribution of each feature (Regression) ####################################
    def plot_contribution_R(self, input, show_table=True):
        no_feature = len(self._feature_names)

        input_scaled = self.scaler.transform(input.to_frame().T)
        prediction = self._model.predict(input_scaled, verbose=0)[0][0]
        cont_list, _ = self.explainer_backward(input_scaled[0], verbose=True)
        cont_list = np.array(cont_list)

        # Contribution Summary Table
        if show_table:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axis('tight')
            ax.axis('off')

            table_data = []
            for i in range(no_feature):
                table_data.append([
                    self._feature_names[i],
                    f"{input.iloc[i]:.4f}",
                    f"{input_scaled[0][i]:.4f}",
                    f"{cont_list[i]:.4f}"
                ])

            table_data.append(['', '', '', ''])
            table_data.append(['Sum of Contributions', '', '', f"{cont_list.sum():.4f}"])
            table_data.append(['Predicted Output', '', '', f"{prediction:.4f}"])
            table_data.append(['Difference', '', '', f"{abs(cont_list.sum() - prediction):.4f}"])

            col_labels = ['Feature', 'Original Value', 'Scaled Value', 'Contribution']

            table = ax.table(cellText=table_data,
                           colLabels=col_labels,
                           cellLoc='right',
                           loc='center',
                           colWidths=[0.25, 0.25, 0.25, 0.25])

            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

            for i in range(len(col_labels)):
                cell = table[(0, i)]
                cell.set_facecolor('#ADD8E6')
                cell.set_text_props(weight='bold', color='white')

            for i in range(1, no_feature + 1):
                for j in range(len(col_labels)):
                    cell = table[(i, j)]
                    cell.set_facecolor('white')

            plt.title('Contribution Summary Table', fontsize=14, weight='bold', pad=20)
            plt.tight_layout()
            plt.show()

        # Backward Contribution Plot
        max_lim = max(cont_list.max(), abs(cont_list.min()))
        min_lim = -max_lim

        f_name = []
        for j in range(no_feature):
            f_name.append(self._feature_names[j] + ' = ' + f"{input.iloc[j]:.2f}")

        bar_colors = ['salmon' if x < 0 else 'steelblue' for x in cont_list]

        title_text = f'Backward Contribution Plot\n(Prediction: {prediction:.4f})'
        plt.figure(figsize=(10, 6))
        plt.barh(f_name, cont_list, color=bar_colors)

        plt.xlim([min_lim, max_lim])
        plt.xlabel('Contribution', fontsize=12)
        plt.title(title_text, fontsize=14, weight='bold')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.tight_layout()
        plt.show()

        return pd.Series(cont_list, index=self._feature_names)

    # Feature importance based on permutation test
    def plot_feature_importance(self, X_data, y_data):
        # Baseline mae
        X_data_scaled = self.scaler.transform(X_data)
        base_score = self._model.evaluate(X_data_scaled, y_data, verbose=0)[1]

        # Permutation test
        print('Permutation test.....')
        perm_scores = []
        for i in range(10):
            this_score = []
            for f in range(len(self._feature_names)):
                X_data_perm = X_data_scaled.copy()
                X_data_perm[:,f] = np.random.RandomState(seed=i).permutation(
                    X_data_scaled[:,f])
                this_score.append(
                    self._model.evaluate(X_data_perm, y_data, verbose=0)[1])

            perm_scores.append(this_score)

        perm_scores = pd.DataFrame(perm_scores, columns=self._feature_names)
        diff = perm_scores.mean(axis=0) - base_score

        df_sorted = diff.sort_values()

        plt.figure(figsize=(10, 6))
        plt.barh(df_sorted.index, df_sorted)
        plt.xlabel('Importance', fontsize=12)
        plt.title('Feature Importance (Permutation Test)', fontsize=14, weight='bold')
        plt.tight_layout()
        plt.show()

## END CLASS #######################################################################


################################################################################
# Example: California Housing Dataset
################################################################################
if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print("=" * 70)
    print("Example of backward contribution calculation using the California Housing Dataset")
    print("=" * 70)

    # Load data
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name='Price')

    # Select a subset of features, excluding Latitude and Longitude
    selected_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                         'Population', 'AveOccup']
    X = X[selected_features]

    print(f"\nDataset size: {X.shape}")
    print(f"Features: {selected_features}")

    # Train Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Build the model
    explainer = BackwardNNexplainer(random_state=42, scaler=scaler)

    params = {
        'epochs': 50,
        'batch_size': 32,
        'lr': 0.001,
        'layers': [8, 4, 1],  # hidden: 8, 4 / output: 1
        'feature_names': selected_features
    }

    print("\n" + "=" * 70)
    print("Start model training")
    print("=" * 70)

    history = explainer.build_NN(X_train, y_train, params)

    # Contribution analysis
    print("\n" + "=" * 70)
    print("Test Sample Analysis")
    print("=" * 70)

    test_sample = X_test.iloc[0]
    print(f"\nTest sample:\n{test_sample}")
    print(f"\nActual value: {y_test.iloc[0]:.4f}")

    # Contribution plot with table
    contributions = explainer.plot_contribution_R(test_sample, show_table=True)

    # Feature importance
    print("\n" + "=" * 70)
    print("Feature Importance Analysis")
    print("=" * 70)
    explainer.plot_feature_importance(X_test, y_test)