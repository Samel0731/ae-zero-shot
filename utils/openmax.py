"""
Code adapted from: https://github.com/ma-xu/Open-Set-Recognition/blob/master/OSR/OpenMax/openmax.py

Copy-paste from Open-Set-Recognition with modifications:
    * 
"""
import numpy as np
import scipy.spatial.distance as spd
import torch


def compute_channel_distances(mavs, features, eu_weight=0.5):
    """
    Input:
        mavs (channel, C)
        features: (N, channel, C)
    Output:
        channel_distances: dict of distance distribution from MAV for each channel.
    """
    eucos_dists, eu_dists, cos_dists = [], [], []
    for channel, mcv in enumerate(mavs):  # Compute channel specific distances
        eu_dists.append([spd.euclidean(mcv, feat[channel]) for feat in features])
        cos_dists.append([spd.cosine(mcv, feat[channel]) for feat in features])
        eucos_dists.append([spd.euclidean(mcv, feat[channel]) * eu_weight +
                            spd.cosine(mcv, feat[channel]) for feat in features])

    return {'eucos': np.array(eucos_dists), 'cosine': np.array(cos_dists), 'euclidean': np.array(eu_dists)}


def compute_train_score_and_mavs_and_dists(train_class_num, trainloader, net, device=None, threshold=0.0002, batch_size=32, local_rank=0):
    if device==None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.eval()

    scores = [[] for _ in range(train_class_num)]
    with torch.no_grad():
        for batch_idx, (inputs, _, targets) in enumerate(trainloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            train_loader_len = int(trainloader._size / batch_size)

            if local_rank == 0 and batch_idx % 200 == 0:
                print(f"computing train score {batch_idx}\t/{train_loader_len}...")

            latent, outputs = net(inputs)
            mse_metric = MeanSquaredError()
            mse_metric.update(outputs, inputs)
            mse = mse_metric.compute()

            for i, t in enumerate(targets):
                if mse[i] < threshold:
                    scores[t].append(latent[i].unsqueeze(dim=0).unsqueeze(dim=0))

            for score, t in zip(outputs, targets):
                # print(f"torch.argmax(score) is {torch.argmax(score)}, t is {t}")
                if torch.argmax(score) == t:
                    scores[t].append(score.unsqueeze(dim=0).unsqueeze(dim=0))
    scores = [torch.cat(x).cpu().numpy() for x in scores]  # (N_c, 1, C) * C
    mavs = np.array([np.mean(x, axis=0) for x in scores])  # (C, 1, C)
    dists = [compute_channel_distances(mcv, score) for mcv, score in zip(mavs, scores)]
    return scores, mavs, dists