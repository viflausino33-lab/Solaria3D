import os
import torch

from depth.model import SolariaDepth


MODEL_PATH = "models/solaria_depth_v1.pth"


print("========================================")
print("     TESTE DE CARREGAMENTO DO MODELO")
print("========================================")


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Modelo não encontrado: {MODEL_PATH}"
    )


print()
print("Modelo encontrado:")
print(MODEL_PATH)


print()
print("Criando SolariaDepth...")

model = SolariaDepth()


print()
print("Carregando pesos...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


print()
print("Pesos carregados com sucesso.")


print()
print("Epoch:", checkpoint["epoch"])
print("Train Loss:", checkpoint["train_loss"])
print("Val Loss:", checkpoint["val_loss"])


parametros = sum(
    p.numel()
    for p in model.parameters()
)


print()
print("Parâmetros:", f"{parametros:,}")


print()
print("========================================")
print("TESTE CONCLUÍDO COM SUCESSO")
print("========================================")
