import streamlit as st
import cocoex
import torch
import pandas as pd
import numpy as np

st.set_page_config(page_title="COCO Testfunktionen", layout="wide")

st.title("COCO / BBOB Testfunktionen mit Streamlit")
st.write(
    "Wähle eine COCO-Testfunktion aus, sample x-Werte und lasse sie zu y-Werten evaluieren."
)

# Mapping: Anzeigename -> COCO function index
FUNCTION_MAP = {
    "f1_sphere": 1,
    "f2_ellipsoid": 2,
    "f3_rastrigin": 15,   # nur als Beispielname, COCO-intern ist das f015
    "f4_rosenbrock": 8,   # nur als Beispielname, COCO-intern ist das f008
}

# Session State
if "test_function" not in st.session_state:
    st.session_state.test_function = "f1_sphere"

if "train_X" not in st.session_state:
    st.session_state.train_X = None

if "train_Y" not in st.session_state:
    st.session_state.train_Y = None

# Sidebar
st.sidebar.header("Einstellungen")

selected_function_name = st.sidebar.selectbox(
    "Test-Funktion auswählen",
    options=list(FUNCTION_MAP.keys()),
    index=list(FUNCTION_MAP.keys()).index(st.session_state.test_function),
)

st.session_state.test_function = selected_function_name

dimension = st.sidebar.number_input("Dimension", min_value=2, max_value=20, value=2, step=1)
n_samples = st.sidebar.number_input("Anzahl Samples", min_value=1, max_value=10000, value=20, step=1)
lower_bound = st.sidebar.number_input("Untergrenze", value=-5.0, step=0.5)
upper_bound = st.sidebar.number_input("Obergrenze", value=5.0, step=0.5)

dtype_name = st.sidebar.selectbox("Torch Datentyp", ["float32", "float64"], index=0)
dtype = torch.float32 if dtype_name == "float32" else torch.float64

device_name = st.sidebar.selectbox("Device", ["cpu"], index=0)
device = torch.device(device_name)

seed = st.sidebar.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)

generate_button = st.sidebar.button("Samples erzeugen und evaluieren")


@st.cache_resource
def get_problem(function_index: int, dimension: int):
    """
    Holt genau ein COCO-Problem für gegebene Funktion und Dimension.
    Instanz wird hier auf 1 gesetzt.
    """
    suite = cocoex.Suite(
        "bbob",
        "",
        f"function_indices:{function_index} dimensions:{dimension} instance_indices:1"
    )
    problem = next(iter(suite))
    return problem


def sample_x(n: int, dim: int, low: float, high: float, dtype, device, seed_value: int):
    torch.manual_seed(seed_value)
    return (torch.rand(n, dim, dtype=dtype, device=device) * (high - low)) + low


def evaluate_problem(problem, x_tensor: torch.Tensor):
    """
    COCO ist nicht batchfähig, deshalb zeilenweise evaluieren.
    """
    x_np = x_tensor.detach().cpu().numpy()
    y_values = np.array([problem(x) for x in x_np], dtype=np.float64)
    return torch.tensor(y_values, dtype=x_tensor.dtype, device=x_tensor.device).unsqueeze(1)


if generate_button:
    if lower_bound >= upper_bound:
        st.error("Die Untergrenze muss kleiner als die Obergrenze sein.")
    else:
        function_index = FUNCTION_MAP[selected_function_name]
        problem = get_problem(function_index=function_index, dimension=dimension)

        train_X = sample_x(
            n=n_samples,
            dim=dimension,
            low=lower_bound,
            high=upper_bound,
            dtype=dtype,
            device=device,
            seed_value=seed,
        )

        train_Y = evaluate_problem(problem, train_X)

        st.session_state.train_X = train_X
        st.session_state.train_Y = train_Y

        st.success("Samples wurden erzeugt und evaluiert.")

# Anzeige
st.subheader("Ausgewählte Funktion")
st.write(f"**Name:** {st.session_state.test_function}")
st.write(f"**COCO Function Index:** {FUNCTION_MAP[st.session_state.test_function]}")

if st.session_state.train_X is not None and st.session_state.train_Y is not None:
    train_X_cpu = st.session_state.train_X.detach().cpu().numpy()
    train_Y_cpu = st.session_state.train_Y.detach().cpu().numpy()

    st.subheader("x-Samples")
    df_x = pd.DataFrame(
        train_X_cpu,
        columns=[f"x{i+1}" for i in range(train_X_cpu.shape[1])]
    )
    st.dataframe(df_x, use_container_width=True)

    st.subheader("Evaluierte y-Werte")
    df_y = pd.DataFrame(train_Y_cpu, columns=["y"])
    st.dataframe(df_y, use_container_width=True)

    st.subheader("Kombinierte Tabelle")
    df_all = pd.concat([df_x, df_y], axis=1)
    st.dataframe(df_all, use_container_width=True)

    csv = df_all.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="CSV herunterladen",
        data=csv,
        file_name="coco_samples.csv",
        mime="text/csv",
    )
else:
    st.info("Klicke links auf 'Samples erzeugen und evaluieren'.")