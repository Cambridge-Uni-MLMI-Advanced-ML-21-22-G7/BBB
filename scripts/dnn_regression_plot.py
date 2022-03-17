import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

from bbb.utils.pytorch_setup import DEVICE
from bbb.utils.plotting import plot_bbb_regression_predictions
from bbb.config.parameters import Parameters, PriorParameters
from bbb.models.dnn import RegressionDNN
from bbb.data import generate_regression_data



MODEL_PATHS = glob('saved_models/DNN_regression/baseline/*/model.pt')
SAVE_DIR = os.path.join('plots', 'dnn_regression', 'baseline')


def main():
    DNN_REGRESSION_PARAMETERS = Parameters(
        name = "DNN_regression",
        input_dim = 1,
        output_dim = 1,
        hidden_layers = 3,
        hidden_units = 400,
        batch_size = 128,
        lr = 1e-3,
        epochs = 1000,
        step_size = 250,
        early_stopping = False,
        early_stopping_thresh = 1e-4,
        dropout = False,
        dropout_p = 0.5
    )

    X_train = generate_regression_data(train=True, size=8*DNN_REGRESSION_PARAMETERS.batch_size, batch_size=DNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)
    X_val = generate_regression_data(train=False, size=DNN_REGRESSION_PARAMETERS.batch_size, batch_size=DNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)

    X_train_arr = np.array(X_train.dataset, dtype=float)
    X_val_arr = np.array(X_val.dataset, dtype=float)

    reg_results = np.zeros((len(MODEL_PATHS), X_val_arr.shape[0]))

    for i, model_path in enumerate(MODEL_PATHS):
        net = RegressionDNN(params=DNN_REGRESSION_PARAMETERS, eval_mode=True).to(DEVICE)
        net.load_saved(model_path=model_path)

        Y_val_pred, _, _ = net.predict(X_val.dataset[:][0])
        Y_val_pred = Y_val_pred.detach().cpu().numpy().flatten()
        
        reg_results[i, :] = Y_val_pred

    Y_val_pred_mean = reg_results.mean(axis=0)
    Y_val_pred_var = reg_results.var(axis=0)
    Y_val_pred_quartiles = np.quantile(reg_results, np.array((0.05,0.25,0.75,0.95)), axis=0)

    plot_bbb_regression_predictions(
        X_train_arr=X_train_arr,
        X_val_arr=X_val_arr,
        Y_val_pred_mean=Y_val_pred_mean,
        Y_val_pred_var=Y_val_pred_var,
        Y_val_pred_quartiles=Y_val_pred_quartiles,
        save_dir=SAVE_DIR
    )


if __name__ == "__main__":
    main()
