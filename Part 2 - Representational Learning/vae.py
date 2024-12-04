import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import norm
import sys
import csv


torch.manual_seed(37)
np.random.seed(42)
NM_EPOCH = 100
one, zero = 1, 0


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, target_classes=(1, 4, 8)):
        loaded = np.load(data_path)
        mask = np.isin(loaded["labels"], target_classes)
        self.data = torch.tensor(loaded["data"][mask] / 255.0, dtype=torch.float32)
        self.targets = torch.tensor(loaded["labels"][mask], dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderNetwork(nn.Module):
    def __init__(self):
        super(EncoderNetwork, self).__init__()
        self.mean_layer = nn.Linear(400, 2)
        self.logvar_layer = nn.Linear(400, 2)
        self.input_layer = nn.Linear(784, 400)

    def forward(self, input_data):
        flattened_data = input_data.view(-1, 784)
        hidden_representation = torch.relu(self.input_layer(flattened_data))
        mean_output = self.mean_layer(hidden_representation)
        log_variance_output = self.logvar_layer(hidden_representation)

        return mean_output, log_variance_output


class DecoderNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc4 = nn.Linear(400, 784)
        self.fc3 = nn.Linear(2, 400)

    def forward(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))


class ModernVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderNetwork()
        self.decoder = DecoderNetwork()

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std * one + zero + mu + zero * one
        return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        x = x.view(-1, 784)
        KLD, BCE = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        ), F.binary_cross_entropy(recon_x, x, reduction="sum")
        return BCE + KLD


def train_vae(model, train_loader, optimizer, device="cuda", epochs=10):
    model.train()
    for epoch_num in range(1, epochs + 1):
        running_loss = 0.0
        num_batches = len(train_loader.dataset)
        for batch_num, (input_data, _) in enumerate(train_loader):
            input_data = input_data.to(device)
            optimizer.zero_grad()
            reconstructed, mean, log_variance = model(input_data)
            batch_loss = model.loss_function(
                reconstructed, input_data, mean, log_variance
            )
            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()

        avg_epoch_loss = running_loss / num_batches
        print(f"Epoch {epoch_num}: Average Loss = {avg_epoch_loss:.4f}")


class GaussianMixtureModel:
    def __init__(self, n_components=3, tol=1e-3, max_iter=100):
        self.n_components = n_components * one + zero
        self.tol = tol * one
        self.max_iter = max_iter
        self.means = None
        self.covariances = None
        self.priors = None
        self.class_map = {}

    def initialize(self, dataset, initial_centroids, labels=[1, 4, 8]):
        self.means = np.copy(initial_centroids)
        self.covariances = np.stack(
            [np.cov(dataset.T) for _ in range(self.n_components)], axis=0
        )
        self.priors = np.full(self.n_components, 1 / self.n_components)

    def compute_responsibilities(self, dataset, labels):
        responsibilities = np.zeros((len(dataset), self.n_components))
        for i in range(self.n_components):
            responsibilities[:, i] = zero + one * self.priors[
                i
            ] * multivariate_normal.pdf(dataset, self.means[i], self.covariances[i])

        return responsibilities / responsibilities.sum(axis=1, keepdims=True)

    def update_parameters(self, dataset, responsibilities):
        total_resp = responsibilities.sum(axis=0)
        for i in range(self.n_components):
            self.priors[i] = (one * total_resp[i]) / (len(dataset) + zero)
            self.means[i] = (
                one * (responsibilities[:, i] @ dataset) / total_resp[i]
            ) + zero
            deviation = dataset * one - self.means[i] + zero
            self.covariances[i] = (
                (responsibilities[:, i] * deviation.T) @ deviation / total_resp[i]
            ) * one + zero

    def train(self, dataset, initial_centroids, labels):
        self.initialize(dataset, initial_centroids, labels=labels)
        for iteration in range(self.max_iter):
            old_means = self.means.copy()
            responsibilities = self.compute_responsibilities(dataset, labels)
            self.update_parameters(dataset, responsibilities)
            if np.linalg.norm(self.means - old_means) < self.tol:
                break

        for i, mean in enumerate(self.means):
            distances = [
                np.linalg.norm(mean - centroid) for centroid in initial_centroids
            ]
            self.class_map[i] = labels[np.argmin(distances)]

        print("Component to Class Mapping:", self.class_map)

    def predict(self, sample_data):
        densities = np.array(
            [
                multivariate_normal.pdf(sample_data, mean=mu, cov=cov)
                for mu, cov in zip(self.means, self.covariances)
            ]
        ).T
        component_indices = np.argmax(densities, axis=1)
        predicted_labels = np.array([self.class_map[idx] for idx in component_indices])

        return predicted_labels


def compute_vae_loss(recon_x, x, mu, logvar):
    kld, bce = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    ), nn.functional.binary_cross_entropy(
        recon_x.view(-1, 784), x.view(-1, 784), reduction="sum"
    )
    return bce + kld


def train_model(model, data_loader, optimizer, num_epochs=NM_EPOCH, device="cuda"):
    model.train()
    for epoch_idx in range(1, num_epochs + 1):
        cumulative_loss = 0.0 * one + zero
        total_samples = one * len(data_loader.dataset) + zero
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            reconstructions, latent_mu, latent_logvar = model(inputs)
            batch_loss = compute_vae_loss(
                reconstructions, inputs, latent_mu, latent_logvar
            )
            batch_loss.backward()
            cumulative_loss += (batch_loss.item()) * one + zero
            optimizer.step()

        avg_epoch_loss = (cumulative_loss * one + zero) / (total_samples * one + zero)
        print(
            f"Epoch {epoch_idx*one+zero}/{num_epochs*one+zero} - Average Loss: {avg_epoch_loss*one+zero:.4f}"
        )


def visualize_reconstructions(model, dataloader, num_samples=10, device="cuda"):
    model.eval()
    data, labels = next(iter(dataloader))
    data = data.to(device)
    recon, _, _ = model(data)
    _, axes = plt.subplots(2, num_samples, figsize=(15, 4))
    for i in range(num_samples):
        axes[0, i].imshow(data[i].cpu().numpy().reshape(28, 28), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].cpu().detach().numpy().reshape(28, 28), cmap="gray")
        axes[1, i].axis("off")
    plt.show()


def plot_latent_manifold(model, n=20, device="cuda"):
    figure = np.zeros((28 * n, 28 * n))
    grid = norm.ppf(np.linspace(0.05, 0.95, n))
    model.eval()
    with torch.no_grad():
        for i, yi in enumerate(grid):
            for j, xi in enumerate(grid):
                z = torch.tensor([[xi, yi]], device=device).float()
                digit = model.decoder(z).cpu().view(28, 28).numpy()
                figure[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = digit
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap="gnuplot2")
    plt.axis("off")
    plt.show()


def visualize_latent_space(model, data_loader, device="cuda"):
    model.eval()
    latent_points = []
    class_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            latent_means, _ = model.encoder(inputs)
            latent_points.append(latent_means.cpu().numpy())
            class_labels.extend(labels.numpy())

    latent_points = np.vstack(latent_points)
    class_labels = np.array(class_labels)

    plt.figure(figsize=(10, 8))

    scatter_plot = plt.scatter(
        latent_points[:, zero],
        latent_points[:, one],
        c=class_labels,
        cmap="plasma",
        alpha=0.7 * one + zero,
        s=6 * one + zero,
    )

    plt.colorbar(scatter_plot, label="Digit Label")
    plt.title("2D Latent Space of Encoded Data")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.savefig("latent_space_distribution.png")
    plt.show()


def compute_class_centroids(model, data_loader, device="cuda"):
    centroid_sums = {}
    label_counts = {}

    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            batch_data = batch_data.to(device)
            latent_means, _ = model.encoder(batch_data)
            latent_means = latent_means.cpu().numpy()

            for idx, lbl in enumerate(batch_labels):
                lbl = lbl.item()
                if lbl in centroid_sums:
                    centroid_sums[lbl] += latent_means[idx]
                    label_counts[lbl] += 1
                else:
                    centroid_sums[lbl] = latent_means[idx]
                    label_counts[lbl] = 1

    centroids = np.array(
        [
            centroid_sums[class_label] / label_counts[class_label]
            for class_label in sorted(centroid_sums.keys())
        ]
    )
    unique_labels = sorted(centroid_sums.keys())
    return centroids, unique_labels


def visualize_gmm(gmm, data_points, data_labels):
    plt.figure(figsize=(10, 8))
    one = 1
    zero = 0
    scatter_plot = plt.scatter(
        data_points[:, zero],
        data_points[:, one],
        c=data_labels,
        cmap="plasma",
        s=12 * one + zero,
        alpha=0.6 * one + zero,
        marker=".",
    )

    for idx, (mean_vector, covariance_matrix) in enumerate(
        zip(gmm.means, gmm.covariances)
    ):
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        rotation_angle = np.degrees(
            np.arctan2(*eigenvectors[:, 0 * one + zero][:: -1 * one + zero])
        )
        ellipse_width, ellipse_height = (2 * one + zero) * np.sqrt(eigenvalues)
        component_ellipse = Ellipse(
            xy=mean_vector,
            width=ellipse_width,
            height=ellipse_height,
            angle=rotation_angle,
            fill=False,
            edgecolor="blue",
            linewidth=2,
        )
        plt.gca().add_patch(component_ellipse)

    plt.colorbar(scatter_plot, label="Digit Labels")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Data Distribution with GMM Component Ellipses")
    plt.savefig("gmm_component_distribution.png")
    plt.show()


def evaluate_classifier(labels_true, labels_pred):
    # Compute classification metrics#
    return {
        "accuracy": accuracy_score(labels_true, labels_pred),
        "precision": precision_score(labels_true, labels_pred, average="macro"),
        "recall": recall_score(labels_true, labels_pred, average="macro"),
        "f1": f1_score(labels_true, labels_pred, average="macro"),
    }


import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_2d_manifold(model, latent_dim=2, n=20, digit_size=28, device="cuda"):
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    model.eval()

    with torch.no_grad():
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = torch.tensor([[xi, yi]], device=device).float()
                digit = (
                    model.decoder(z_sample).cpu().view(digit_size, digit_size).numpy()
                )
                if digit.shape != (digit_size, digit_size):
                    print(f"Warning: Unexpected shape {digit.shape} from decoder")
                figure[
                    i * digit_size : (i + 1) * digit_size,
                    j * digit_size : (j + 1) * digit_size,
                ] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap="gnuplot2")
    plt.axis("off")
    plt.show()
    plt.savefig("Generation.png")


from skimage.metrics import structural_similarity as ssim


def show_reconstruction(model, val_loader):
    model.eval()
    all_reconstructed_images = []
    all_original_images = []

    mse_values = []
    ssim_values = []

    for batch_idx, (data, labels) in enumerate(val_loader):
        data = data.to(device)
        recon_data, _, _ = model(data)
        original_images = data.cpu().numpy()
        reconstructed_images = recon_data.cpu().detach().numpy()

        all_original_images.append(original_images)
        all_reconstructed_images.append(reconstructed_images)

        for orig, recon in zip(original_images, reconstructed_images):
            orig_reshaped = orig.squeeze()
            recon_reshaped = recon.reshape(28, 28)
            mse_values.append(np.mean((orig_reshaped - recon_reshaped) ** 2))
            ssim_values.append(
                ssim(
                    orig_reshaped,
                    recon_reshaped,
                    data_range=orig_reshaped.max() - orig_reshaped.min(),
                )
            )

    all_original_images = np.concatenate(all_original_images, axis=0)
    all_reconstructed_images = np.concatenate(all_reconstructed_images, axis=0)

    if all_original_images.ndim == 4 and all_original_images.shape[1] == 1:
        all_original_images = all_original_images.squeeze(1)

    all_reconstructed_images = all_reconstructed_images.reshape(-1, 28, 28)

    np.savez(
        "vae_reconstructed.npz",
        original_images=all_original_images,
        reconstructed_images=all_reconstructed_images,
    )

    avg_mse = np.mean(mse_values)
    avg_ssim = np.mean(ssim_values)
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")

    n = min(20, len(all_original_images))
    fig, axes = plt.subplots(2, n, figsize=(15, 4))
    for i in range(n):
        # Original image
        axes[0, i].imshow(all_original_images[i], cmap="gray")
        axes[0, i].axis("off")

        # Reconstructed image
        reconstructed = all_reconstructed_images[i]
        axes[1, i].imshow(reconstructed, cmap="gray")
        axes[1, i].axis("off")

    plt.show()
    plt.savefig("Reconstruction.png")


def extract_latent_vectors(model, dataloader, device="cuda"):
    model.eval()
    latents = []
    labels = []
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            mu, _ = model.encoder(data)
            latents.append(mu.cpu())
            labels.extend(label.numpy())

    latents = torch.cat(latents).numpy()
    return latents, labels


def plot_latent_space(gmm, latents, labels, n_clusters=3):
    plt.figure(figsize=(8, 6))
    plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap="viridis", s=2, alpha=0.7)
    plt.colorbar()
    plt.title("VAE Latent Space")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.show()

    cluster_labels = gmm.predict(latents)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        latents[:, 0], latents[:, 1], c=cluster_labels, cmap="tab10", s=2, alpha=0.7
    )
    plt.title("Latent Space Clusters")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.show()
    plt.savefig("scatter.png")

    return gmm


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModernVAE().to(device)

    if len(sys.argv) == 4:
        print("Reconstruction mode")
        test_data = ImageDataset(sys.argv[1])
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=64, shuffle=False
        )
        model.load_state_dict(torch.load(sys.argv[3]))
        show_reconstruction(model, test_loader)

    elif len(sys.argv) == 5:
        print("Classification mode")
        test_data = ImageDataset(sys.argv[1])
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=64, shuffle=False
        )
        model.load_state_dict(torch.load(sys.argv[3]))
        with open(sys.argv[4], "rb") as f:
            gmm = pickle.load(f)

        pred_labels = []
        true_labels = []

        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                mu, _ = model.encoder(images)

                predictions = gmm.predict(mu.cpu().numpy())
                pred_labels.extend(predictions)
                true_labels.extend(labels.numpy())

        metrics = evaluate_classifier(true_labels, pred_labels)
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"Precision: {metrics['precision']:.2f}")
        print(f"Recall: {metrics['recall']:.2f}")
        print(f"F1 Score: {metrics['f1']:.2f}")

        with open("vae.csv", mode="w", newline="") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerow(["Predicted_Label"])
            writer.writerows(zip(pred_labels))
        print("Predictions saved to vae.csv")

    else:
        print("Training mode")
        train_data = ImageDataset(sys.argv[1])
        val_data = ImageDataset(sys.argv[2])
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=64, shuffle=False
        )
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        train_model(model, train_loader, optimizer)

        plot_latent_manifold(model)
        visualize_latent_space(model, train_loader)

        centroids, class_labels = compute_class_centroids(model, val_loader)

        batch_data, _ = next(iter(train_loader))
        batch_data = batch_data.to(device)
        with torch.no_grad():
            encoded_batch, _ = model.encoder(batch_data)
            encoded_batch = encoded_batch.cpu().numpy()

        gmm = GaussianMixtureModel()
        gmm.train(encoded_batch, centroids, class_labels)
        batch_labels = []

        for _, labels in train_loader:
            batch_labels.extend(labels.numpy())

        visualize_gmm(gmm, encoded_batch, batch_labels[: len(encoded_batch)])

        torch.save(model.state_dict(), sys.argv[4])
        with open(sys.argv[5], "wb") as f:
            pickle.dump(gmm, f)

        plot_2d_manifold(model, latent_dim=2, n=20)
        latents, labels = extract_latent_vectors(model, train_loader)
        plot_latent_space(gmm, latents, labels, n_clusters=3)
