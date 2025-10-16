import os
import json
import joblib
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def process(input_path: str, scalers: dict, batch_size: int = 64):
    """
    Preprocess input CSV and return a DataLoader for prediction.
    - input_path: path to CSV file
    - scalers: dict with keys 'eq','dnn','struct' containing fitted scalers
    Returns: DataLoader
    """
    df = pd.read_csv(input_path, delimiter=',', encoding='utf-8')
    df = df.dropna()
    # avoid zero when taking log
    df.loc[df['Bracketed Duration'] == 0, 'Bracketed Duration'] = 1e-4

    # ensure expected categorical columns and one-hot encoding consistent with training
    categories1 = ['C1', 'C2', 'C3', 'RM1', 'RM2', 'S1', 'S2', 'S3', 'S4', 'S5', 'URM', 'W1', 'W2']
    df['strutype'] = pd.Categorical(df['strutype'], categories=categories1, ordered=False)
    df = pd.get_dummies(df, columns=['strutype'], drop_first=False)

    # coarse year grouping consistent with training
    df.loc[df['year'] > 1989, 'yeartype'] = 1
    df.loc[(df['year'] <= 1989) & (df['year'] > 1978), 'yeartype'] = 2
    df.loc[df['year'] <= 1978, 'yeartype'] = 3
    df['yeartype'] = pd.Categorical(df['yeartype'], categories=[1,2,3], ordered=False)
    df = pd.get_dummies(df, columns=['yeartype'], drop_first=False)

    # drop possible non-feature columns (keep consistent with original processing)
    df = df.iloc[:, 2:]
    num_cols = df.columns[:26]
    df = df[(df.iloc[:, :26] >= 0).all(axis=1)]
    df[num_cols] = df[num_cols].astype('float32')
    # take logarithm for numeric features as in training
    df[num_cols] = np.log(df[num_cols])

    # indices follow original script: adjust if your column order differs
    X_eq = df.iloc[:, 6:25].values      # earthquake features for attention (19 dim)
    X_dnn = df.iloc[:, 0:3].values      # small DNN inputs (3 dim)
    X_struct = df.iloc[:, 26:].values   # structural categorical features
    y = df.iloc[:, 25].values

    # apply pre-fitted scalers
    scaler_eq = scalers["eq"]
    scaler_dnn = scalers["dnn"]
    scaler_struct = scalers["struct"]
    X_eq = scaler_eq.transform(X_eq)
    X_dnn = scaler_dnn.transform(X_dnn)
    X_struct = scaler_struct.transform(X_struct)

    # convert to torch tensors
    X_eq = torch.FloatTensor(X_eq)
    X_struct = torch.FloatTensor(X_struct)
    X_dnn = torch.FloatTensor(X_dnn)
    y = torch.FloatTensor(y).view(-1, 1)

    dataset = TensorDataset(X_struct, X_eq, X_dnn, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

def predict(best_pth: str, input_path: str, output_path: str, scalers: dict, device: str = None):
    """
    Load model, run inference on input data, save plots and metrics.
    - best_pth: path to .pth model file
    - input_path: CSV input
    - output_path: folder to save results
    - scalers: dict of fitted scalers
    - device: 'cpu' or 'cuda' (auto-detected if None)
    """
    # choose device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from model import CombinedModel

    os.makedirs(output_path, exist_ok=True)

    # prepare model and data
    model = CombinedModel(eq_input_dim=19, dnn_input_dim=3, struct_input_dim=16)  # clearer param name
    try:
        # load weights onto chosen device
        state = torch.load(best_pth, map_location=device)
        model.load_state_dict(state, strict=True)
    except Exception as e:
        # try loading with 'weights_only' compatibility if saved differently
        try:
            model.load_state_dict(torch.load(best_pth, map_location=device), strict=False)
        except Exception as ee:
            raise RuntimeError(f"Failed to load model weights: {e} / {ee}")

    model.to(device)
    model.eval()

    predict_loader = process(input_path, scalers)

    predictions = []
    actuals = []
    with torch.no_grad():
        for SX_batch, EX_batch, DX_batch, y_batch in predict_loader:
            SX_batch = SX_batch.to(device)
            EX_batch = EX_batch.to(device)
            DX_batch = DX_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(SX_batch, EX_batch, DX_batch)
            # move to cpu numpy for metric calculation/saving
            preds = output.view(-1).cpu().numpy()
            trues = y_batch.view(-1).cpu().numpy()

            predictions.extend(preds.tolist())
            actuals.extend(trues.tolist())

    # compute metrics
    r2 = float(r2_score(actuals, predictions))
    mae = float(mean_absolute_error(actuals, predictions))
    mse = float(mean_squared_error(actuals, predictions))
    rmse = float(np.sqrt(mse))
    r = float(np.corrcoef(actuals, predictions)[0, 1])

    # print brief summary
    print(f'R^2: {r2:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, R: {r:.6f}')

    # scatter plot (saved to file)
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r')  # 45-degree line
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'R^2: {r2:.6f}')
    plt.text(0.01, 0.99, f'MAE: {mae:.6f}', fontsize=10, verticalalignment='top', transform=plt.gca().transAxes)
    plt.text(0.01, 0.95, f'MSE: {mse:.6f}', fontsize=10, verticalalignment='top', transform=plt.gca().transAxes)
    plt.text(0.01, 0.91, f'RMSE: {rmse:.6f}', fontsize=10, verticalalignment='top', transform=plt.gca().transAxes)
    plt.savefig(os.path.join(output_path, "predicted_vs_actual.png"))
    plt.close()

    # save predictions and metrics
    results_df = pd.DataFrame({"true": actuals, "predicted": predictions})
    results_df.to_csv(os.path.join(output_path, "predictions.csv"), index=False)

    metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "r": r}
    with open(os.path.join(output_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    scalers_path = os.path.join(script_dir, "model", "scalers.pkl")
    pth_file = os.path.join(script_dir, "model", "TPFA.pth")
    output_dir = os.path.join(script_dir, "pre_output")
    input_file = os.path.join(script_dir, "example_input_data.csv")

    scalers = joblib.load(scalers_path)
    os.makedirs(output_dir, exist_ok=True)

    predict(pth_file, input_file, output_dir, scalers)