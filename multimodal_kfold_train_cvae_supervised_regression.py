#!/home/songlinzhao/anaconda3/envs/normodiff/bin/python3
"""Script to train the deterministic supervised adversarial autoencoder."""
from pathlib import Path
import random as rn
import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import torch
from cVAE import cVAE_multimodal_regression
from utils import get_column_name, get_datasets_name
from utils_vae import MyDataset
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class MyDataset_labels_with_fi(MyDataset):
    def __init__(self, data, covariates, fi):
        self.data = data
        self.covariates = covariates
        self.fi = fi

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.covariates[idx], self.fi[idx]

def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}

def train_and_test(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    PROJECT_ROOT = Path.cwd()
    output_dir = PROJECT_ROOT / 'regression_outputs'
    output_dir.mkdir(exist_ok=True)

    dataset_names = get_datasets_name(args.dataset_resourse, args.procedure)
    participants_path = PROJECT_ROOT / 'data' / args.dataset_resourse / 'y.csv'
    ids_df = pd.read_csv(participants_path)

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(ids_df)):
        print(f"=== Fold {fold} ===")

        train_df = ids_df.iloc[train_idx]
        test_df = ids_df.iloc[test_idx]

        generator_train_list = []
        generator_test_list = []
        input_dim_list = []

        for dataset_name in dataset_names:
            columns_name = get_column_name(args.dataset_resourse, dataset_name)
            freesurfer_path = PROJECT_ROOT / 'data' / args.dataset_resourse / f'{dataset_name}.csv'

            train_ids = train_df['IID'].tolist()
            test_ids = test_df['IID'].tolist()

            modality_df = pd.read_csv(freesurfer_path)
            demo_df = pd.read_csv(participants_path)

            train_dataset_df = pd.merge(modality_df[modality_df['IID'].isin(train_ids)], demo_df, on='IID')
            test_dataset_df = pd.merge(modality_df[modality_df['IID'].isin(test_ids)], demo_df, on='IID')

            train_data = train_dataset_df[columns_name].values
            test_data = test_dataset_df[columns_name].values

            scaler = RobustScaler()
            train_data = scaler.fit_transform(train_data)
            test_data = scaler.transform(test_data)

            train_cov = train_dataset_df[['AGE', 'PTGENDER']].values
            test_cov = test_dataset_df[['AGE', 'PTGENDER']].values

            train_fi = train_dataset_df['FI'].values.astype(np.float32).reshape(-1, 1)
            test_fi = test_dataset_df['FI'].values.astype(np.float32).reshape(-1, 1)

            input_dim_list.append(train_data.shape[1])

            train_dataset = MyDataset_labels_with_fi(train_data.astype(np.float32), train_cov.astype(np.float32), train_fi)
            test_dataset = MyDataset_labels_with_fi(test_data.astype(np.float32), test_cov.astype(np.float32), test_fi)

            generator_train = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            generator_test = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            generator_train_list.append(generator_train)
            generator_test_list.append(generator_test)

        h_dim = args.hz_para_list[:-1]
        z_dim = args.hz_para_list[-1]

        model = cVAE_multimodal_regression(
            input_dim_list=input_dim_list,
            hidden_dim=h_dim,
            latent_dim=z_dim,
            c_dim=2,
            learning_rate=args.base_learning_rate,
            modalities=len(dataset_names),
            non_linear=True
        ).to(device)

        model.train()
        for epoch in range(args.epochs):
            for batch_list in zip(*generator_train_list):
                xes = [batch[0].to(device) for batch in batch_list]
                covs = [batch[1].to(device) for batch in batch_list]
                fi_target = batch_list[0][2].to(device)

                out = model.forward_multimodal(xes, covs, args.combine)
                losses = model.loss_function_multimodal(xes, out, fi_target, lambda_reg=1.0)

                model.optimizer1.zero_grad()
                losses['total'].backward()
                model.optimizer1.step()

            if epoch % 10 == 0 or epoch == args.epochs - 1:
                print(f"[Fold {fold}][Epoch {epoch}] Loss: {losses['total'].item():.4f}, FI MSE: {losses['regression'].item():.4f}")

        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for batch_list in zip(*generator_test_list):
                xes = [batch[0].to(device) for batch in batch_list]
                covs = [batch[1].to(device) for batch in batch_list]
                fi_true = batch_list[0][2].to(device)

                out = model.forward_multimodal(xes, covs, args.combine)
                fi_pred = out['fi_pred']
                preds.append(fi_pred.cpu().numpy())
                trues.append(fi_true.cpu().numpy())

        preds = np.vstack(preds)
        trues = np.vstack(trues)

        np.save(output_dir / f'fold_{fold}_pred.npy', preds)
        np.save(output_dir / f'fold_{fold}_true.npy', trues)

        scores = evaluate_regression(trues, preds)
        print(f"[Fold {fold}] RMSE: {scores['RMSE']:.4f}, MAE: {scores['MAE']:.4f}, R²: {scores['R2']:.4f}, MAPE: {scores['MAPE']:.2f}%")

        plt.figure(figsize=(6, 6))
        plt.scatter(trues, preds, alpha=0.5)
        plt.plot([trues.min(), trues.max()], [trues.min(), trues.max()], 'r--')
        plt.xlabel("True FI")
        plt.ylabel("Predicted FI")
        plt.title(f"Fold {fold} - FI Prediction")
        plt.grid(True)
        plt.savefig(output_dir / f'fold_{fold}_scatter.png')
        plt.close()

        all_ids = ids_df['IID'].tolist()
        demo_df = pd.read_csv(participants_path)

        for modal_idx, dataset_name in enumerate(dataset_names):
            print(f"[Fold {fold}] Extracting ROI-wise deviation for {dataset_name}...")
            freesurfer_path = PROJECT_ROOT / 'data' / args.dataset_resourse / f'{dataset_name}.csv'
            modality_df = pd.read_csv(freesurfer_path)
            full_df = pd.merge(modality_df[modality_df['IID'].isin(all_ids)], demo_df, on='IID')

            columns_name = get_column_name(args.dataset_resourse, dataset_name)
            X = full_df[columns_name].values.astype(np.float32)
            C = full_df[['AGE', 'PTGENDER']].values.astype(np.float32)
            IIDs = full_df['IID'].tolist()

            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)

            x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            c_tensor = torch.tensor(C, dtype=torch.float32).to(device)

            with torch.no_grad():
                mu, logvar = model.encode(x_tensor, c_tensor, modal_idx)
                z = model.reparameterise(mu, logvar)
                x_recon_dist = model.decode(z, c_tensor, modal_idx)
                x_recon = x_recon_dist.loc
                deviation_roi = ((x_tensor - x_recon) ** 2).cpu().numpy()

            df_out = pd.DataFrame(deviation_roi, columns=[f'ROI_{i}' for i in range(deviation_roi.shape[1])])
            df_out.insert(0, 'IID', IIDs)
            df_out.to_csv(output_dir / f'deviation_fold_{fold}_{dataset_name}_roiwise.csv', index=False)

    print("Training & evaluation complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-R', '--dataset_resourse', type=str, default='ADNI')
    parser.add_argument('-H', '--hz_para_list', nargs='+', type=int, default=[110, 110, 10])
    parser.add_argument('-C', '--combine', type=str, default='gpoe')
    parser.add_argument('-P', '--procedure', type=str, default='UCA-gPoE')
    parser.add_argument('-E', '--epochs', type=int, default=500)
    parser.add_argument('-K', '--n_splits', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('-BaseLR', '--base_learning_rate', type=float, default=0.0001)
    args = parser.parse_args()

    train_and_test(args)
