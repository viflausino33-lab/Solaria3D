import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# ============================================================
# CAMINHO DO PROJETO
# ============================================================

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from training.dataset import SolariaDepthDataset
from depth.model import SolariaDepth


# ============================================================
# CONFIGURAÇÃO
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 2

EPOCHS = 1

LEARNING_RATE = 1e-4

MAX_TRAIN_SAMPLES = 20

MAX_VAL_SAMPLES = 5

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "solaria_depth_v1.pth"
)


# ============================================================
# INFORMAÇÕES
# ============================================================

print()
print("========================================")
print("       SOLARIA DEPTH TRAINING")
print("========================================")

print()
print("Dispositivo:", DEVICE)


if DEVICE == "cpu":

    print()
    print("AVISO: treinamento será executado na CPU.")


# ============================================================
# DATASET
# ============================================================

print()
print("Carregando dataset de treinamento...")

train_dataset = SolariaDepthDataset(
    split="train"
)

val_dataset = SolariaDepthDataset(
    split="val"
)


# Limita a quantidade somente para o primeiro teste
if MAX_TRAIN_SAMPLES is not None:

    train_dataset.dataset = (
        train_dataset.dataset.select(
            range(
                min(
                    MAX_TRAIN_SAMPLES,
                    len(train_dataset)
                )
            )
        )
    )


if MAX_VAL_SAMPLES is not None:

    val_dataset.dataset = (
        val_dataset.dataset.select(
            range(
                min(
                    MAX_VAL_SAMPLES,
                    len(val_dataset)
                )
            )
        )
    )


print(
    "Treino:",
    len(train_dataset)
)

print(
    "Validação:",
    len(val_dataset)
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODELO
# ============================================================

print()
print("Criando SolariaDepth...")

model = SolariaDepth()

model = model.to(
    DEVICE
)


parametros = sum(
    p.numel()
    for p in model.parameters()
)


print(
    "Parâmetros:",
    f"{parametros:,}"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.L1Loss()


# ============================================================
# OTIMIZADOR
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# DIRETÓRIO
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# TREINAMENTO
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    barra = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for imagens, depths in barra:

        imagens = imagens.to(
            DEVICE
        )

        depths = depths.to(
            DEVICE
        )

        optimizer.zero_grad()

        previsao = model(
            imagens
        )

        loss = criterion(
            previsao,
            depths
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        barra.set_postfix(
            loss=f"{loss.item():.6f}"
        )


    train_loss = (
        total_loss /
        len(train_loader)
    )


    # ========================================================
    # VALIDAÇÃO
    # ========================================================

    model.eval()

    total_val_loss = 0.0

    with torch.no_grad():

        for imagens, depths in val_loader:

            imagens = imagens.to(
                DEVICE
            )

            depths = depths.to(
                DEVICE
            )

            previsao = model(
                imagens
            )

            loss = criterion(
                previsao,
                depths
            )

            total_val_loss += loss.item()


    val_loss = (
        total_val_loss /
        len(val_loader)
    )


    print()
    print(
        f"Epoch {epoch + 1}"
    )

    print(
        f"Train Loss: {train_loss:.6f}"
    )

    print(
        f"Val Loss: {val_loss:.6f}"
    )


# ============================================================
# SALVAR
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "epoch": EPOCHS,
        "train_loss": train_loss,
        "val_loss": val_loss,
    },
    MODEL_PATH
)


print()
print("========================================")
print("Treinamento concluído.")
print("Modelo salvo em:")
print(MODEL_PATH)
print("========================================")
