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

EPOCHS = 10

LEARNING_RATE = 1e-4

MODEL_DIR = os.path.join(
    ROOT,
    "models"
)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "solaria_depth_best.pth"
)

LAST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "solaria_depth_last.pth"
)


# ============================================================
# LOSS
# ============================================================

def gradient_loss(predicao, alvo):
    """
    Compara os gradientes horizontais e verticais
    da profundidade prevista e da profundidade real.

    Isso ajuda a rede a aprender estruturas espaciais
    e preservar transições de profundidade.
    """

    pred_x = predicao[:, :, :, 1:] - predicao[:, :, :, :-1]
    alvo_x = alvo[:, :, :, 1:] - alvo[:, :, :, :-1]

    pred_y = predicao[:, :, 1:, :] - predicao[:, :, :-1, :]
    alvo_y = alvo[:, :, 1:, :] - alvo[:, :, :-1, :]

    loss_x = torch.mean(
        torch.abs(pred_x - alvo_x)
    )

    loss_y = torch.mean(
        torch.abs(pred_y - alvo_y)
    )

    return loss_x + loss_y


def calcular_loss(predicao, alvo):
    """
    Loss principal da SolariaDepth.

    L1:
        aproxima os valores de profundidade.

    Gradient:
        ajuda a preservar a estrutura espacial.
    """

    l1 = torch.mean(
        torch.abs(predicao - alvo)
    )

    grad = gradient_loss(
        predicao,
        alvo
    )

    loss = (
        l1 +
        0.5 * grad
    )

    return loss, l1, grad


# ============================================================
# INFORMAÇÕES
# ============================================================

print()
print("========================================")
print("       SOLARIA DEPTH TRAINING")
print("========================================")

print()
print("Dispositivo:", DEVICE)

print("Batch size:", BATCH_SIZE)
print("Épocas:", EPOCHS)
print("Learning rate:", LEARNING_RATE)

if DEVICE == "cpu":

    print()
    print(
        "AVISO: treinamento será executado na CPU."
    )


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


print()
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
# HISTÓRICO
# ============================================================

historico = []

melhor_val_loss = float(
    "inf"
)


# ============================================================
# TREINAMENTO
# ============================================================

for epoch in range(EPOCHS):

    print()
    print(
        "========================================"
    )

    print(
        f"ÉPOCA {epoch + 1}/{EPOCHS}"
    )

    print(
        "========================================"
    )


    # ========================================================
    # TREINO
    # ========================================================

    model.train()

    total_loss = 0.0
    total_l1 = 0.0
    total_grad = 0.0

    barra = tqdm(
        train_loader,
        desc=f"Treino {epoch + 1}/{EPOCHS}"
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


        loss, l1, grad = calcular_loss(
            previsao,
            depths
        )


        loss.backward()


        optimizer.step()


        total_loss += loss.item()
        total_l1 += l1.item()
        total_grad += grad.item()


        barra.set_postfix(
            loss=f"{loss.item():.5f}",
            l1=f"{l1.item():.5f}",
            grad=f"{grad.item():.5f}"
        )


    train_loss = (
        total_loss /
        len(train_loader)
    )

    train_l1 = (
        total_l1 /
        len(train_loader)
    )

    train_grad = (
        total_grad /
        len(train_loader)
    )


    # ========================================================
    # VALIDAÇÃO
    # ========================================================

    model.eval()

    total_val_loss = 0.0
    total_val_l1 = 0.0
    total_val_grad = 0.0


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


            loss, l1, grad = calcular_loss(
                previsao,
                depths
            )


            total_val_loss += loss.item()
            total_val_l1 += l1.item()
            total_val_grad += grad.item()


    val_loss = (
        total_val_loss /
        len(val_loader)
    )

    val_l1 = (
        total_val_l1 /
        len(val_loader)
    )

    val_grad = (
        total_val_grad /
        len(val_loader)
    )


    # ========================================================
    # HISTÓRICO
    # ========================================================

    historico.append(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_l1": train_l1,
            "train_grad": train_grad,
            "val_loss": val_loss,
            "val_l1": val_l1,
            "val_grad": val_grad,
        }
    )


    # ========================================================
    # RESULTADOS
    # ========================================================

    print()

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.6f}"
    )

    print(
        f"Train L1:   {train_l1:.6f}"
    )

    print(
        f"Train Grad: {train_grad:.6f}"
    )

    print()

    print(
        f"Val Loss:   {val_loss:.6f}"
    )

    print(
        f"Val L1:     {val_l1:.6f}"
    )

    print(
        f"Val Grad:   {val_grad:.6f}"
    )


    # ========================================================
    # SALVAR ÚLTIMO MODELO
    # ========================================================

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "epoch":
            epoch + 1,

        "train_loss":
            train_loss,

        "val_loss":
            val_loss,

        "train_l1":
            train_l1,

        "train_grad":
            train_grad,

        "val_l1":
            val_l1,

        "val_grad":
            val_grad,

        "historico":
            historico,
    }


    torch.save(
        checkpoint,
        LAST_MODEL_PATH
    )


    # ========================================================
    # SALVAR MELHOR MODELO
    # ========================================================

    if val_loss < melhor_val_loss:

        melhor_val_loss = val_loss

        torch.save(
            checkpoint,
            BEST_MODEL_PATH
        )

        print()
        print(
            "NOVO MELHOR MODELO!"
        )

        print(
            "Val Loss:",
            f"{val_loss:.6f}"
        )

        print(
            "Salvo em:",
            BEST_MODEL_PATH
        )


# ============================================================
# FINAL
# ============================================================

print()
print("========================================")
print("       TREINAMENTO CONCLUÍDO")
print("========================================")

print()

print(
    "Melhor Val Loss:",
    f"{melhor_val_loss:.6f}"
)

print()

print(
    "Melhor modelo:"
)

print(
    BEST_MODEL_PATH
)

print()

print(
    "Último modelo:"
)

print(
    LAST_MODEL_PATH
)

print()
print("========================================")
