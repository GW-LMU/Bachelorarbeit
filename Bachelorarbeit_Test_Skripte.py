# pip install botorch gpytorch torch

import torch
import sys
print(sys.executable)
import plotly.graph_objects as go

from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

#Importiernen der Simulierden Funktionen 
from test_functions import f1_sphere

torch.manual_seed(42)
dtype = torch.double
device = torch.device("cpu")


# 2) Initiale Daten
train_X = (torch.rand(5000, 2, dtype=dtype, device=device) * 20000.0) - 10000.0
print("x-Traingsdaten:", train_X)
train_Y = f1_sphere.sphere_function(train_X)

    

# 3) Grenzen des Suchraums
bounds = torch.tensor(
    [[-10000.0, -10000.0],
     [10000.0,  10000.0]],
    dtype=dtype,
    device=device
)

print("y-Traingsdaten",train_Y)

#--------------------------------------------------------------------------------------------#
                                #Traings des Suggrate-Models
#--------------------------------------------------------------------------------------------#

model = SingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    input_transform=Normalize(d=2, bounds=bounds),
    outcome_transform=Standardize(m=1),
)

# Marginal Log Likelihood
mll = ExactMarginalLogLikelihood(model.likelihood, model)

# Modell fitten
fit_gpytorch_mll(mll)

print("GP erfolgreich gefittet.")

# ----------------------------------
# 3) Bestes bisheriges y bestimmen
# ----------------------------------
# Sphere minimieren wir.
# LogExpectedImprovement ist standardmäßig für maximize=True gedacht.
# Deshalb drehen wir train_Y einfach um.
best_f = (-train_Y).max().item()

# ----------------------------------
# 4) Acquisition Function definieren
# ----------------------------------
acqf = LogExpectedImprovement(
    model=model,
    best_f=best_f,
    maximize=True,
)

# ----------------------------------
# 5) Nächsten Kandidaten optimieren
# ----------------------------------
candidate, acq_value = optimize_acqf(
    acq_function=acqf,
    bounds=bounds,
    q=1,
    num_restarts=10,
    raw_samples=100,
)

print("Nächster Kandidat:", candidate)
print("Acquisition-Wert:", acq_value)

# ----------------------------------
# 6) Funktion am neuen Punkt auswerten
# ----------------------------------
new_Y = f1_sphere.sphere_function(candidate)

print("Neuer Funktionswert:", new_Y)

# ----------------------------------
# 7) Datensatz erweitern
# ----------------------------------
train_X = torch.cat([train_X, candidate], dim=0)
train_Y = torch.cat([train_Y, new_Y], dim=0)

print("Neue train_X shape:", train_X.shape)
print("Neue train_Y shape:", train_Y.shape)


#--------------------------------------------------------------------------------------------#
                                #Plot des Simulierten Daten 
#--------------------------------------------------------------------------------------------#
x1 = train_X[:, 0].cpu().numpy()
x2 = train_X[:, 1].cpu().numpy()
y = train_Y[:, 0].cpu().numpy()

# 3D Scatter Plot
fig = go.Figure(data=[go.Scatter3d(
    x=x1,
    y=x2,
    z=y,
    mode='markers',
    marker=dict(size=5)
)])

fig.update_layout(
    title="Trainingspunkte (3D)",
    scene=dict(
        xaxis_title="x1",
        yaxis_title="x2",
        zaxis_title="f(x)"
    )
)

fig.show()



