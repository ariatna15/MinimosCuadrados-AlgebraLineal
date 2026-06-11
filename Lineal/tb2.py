import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="Mínimos Cuadrados",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .stApp { background: #0a0a0f; }
    .main .block-container { background: #0a0a0f; padding-top: 1.5rem; max-width: 1300px; }
    section[data-testid="stSidebar"] { background: #0f0f18 !important; border-right: 1px solid #1e1e2e; }
    section[data-testid="stSidebar"] * { color: #e2e2f0 !important; }
    section[data-testid="stSidebar"] input { background: #1a1a28 !important; border: 1px solid #2e2e48 !important; border-radius: 8px !important; color: #e2e2f0 !important; }
    section[data-testid="stSidebar"] select { background: #1a1a28 !important; border: 1px solid #2e2e48 !important; }
    .stButton > button { background: linear-gradient(135deg, #6c63ff, #3ecfcf) !important; color: #ffffff !important; border: none !important; border-radius: 10px !important; padding: 0.6rem 2rem !important; font-weight: 700; font-family: 'Syne', sans-serif !important; letter-spacing: 0.03em; }
    .stButton > button:hover { opacity: 0.85; transform: translateY(-1px); }
    h1,h2,h3,h4,h5,h6,label,p,span { color: #e2e2f0 !important; }
    .stDataFrame thead th { background: #1a1a2e !important; color: #9090c0 !important; font-size: 0.75rem !important; }
    .stDataFrame tbody td { background: #11111f !important; color: #e2e2f0 !important; font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='padding:0.5rem 0 1.2rem;'>
    <h1 style='font-size:2rem; font-weight:800; color:#ffffff; letter-spacing:-0.02em; margin:0;'>
        Método de Mínimos Cuadrados
    </h1>
    <p style='font-size:0.9rem; color:#7070a0; margin-top:6px; font-family: JetBrains Mono, monospace;'>
        x = (AᵀA)⁻¹ Aᵀ B
    </p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Datos de entrada")
    x_input = st.text_input("Valores de X", value="1, 2, 3, 4, 5")
    y_input = st.text_input("Valores de Y", value="9, 7.5, 4.2, 3, 2.1")
    st.markdown("---")
    st.markdown("### Modelos")
    modelo1 = st.selectbox("Modelo 1", ["Lineal", "Cuadrático", "Exponencial"], index=0)
    modelo2 = st.selectbox("Modelo 2", ["Ninguno", "Lineal", "Cuadrático", "Exponencial"], index=2)
    modelo3 = st.selectbox("Modelo 3", ["Ninguno", "Lineal", "Cuadrático", "Exponencial"], index=0)
    st.markdown("---")
    calcular = st.button("▶ Calcular Modelos", use_container_width=True)

# ── PARSING ────────────────────────────────────────────────────────────────
try:
    X = np.array([float(v.strip()) for v in x_input.split(",")])
    Y = np.array([float(v.strip()) for v in y_input.split(",")])
    if len(X) != len(Y): st.error("X e Y deben tener la misma cantidad de elementos."); st.stop()
    if len(X) < 3: st.error("Se necesitan al menos 3 puntos."); st.stop()
except ValueError:
    st.error("Formato de datos inválido."); st.stop()

COLORES = {"Lineal": "#6c63ff", "Cuadrático": "#ff9f40", "Exponencial": "#3ecfcf"}
DASH    = {"Lineal": "solid",   "Cuadrático": "dash",    "Exponencial": "dot"}

# ── LÓGICA MATRICIAL ───────────────────────────────────────────────────────
def resolver(X, Y, tipo):
    if tipo == "Exponencial" and np.any(Y <= 0):
        return {"error": True}

    if tipo == "Lineal":
        A = np.vstack([X, np.ones(len(X))]).T
        B = Y.reshape(-1, 1)
    elif tipo == "Cuadrático":
        A = np.vstack([X**2, X, np.ones(len(X))]).T
        B = Y.reshape(-1, 1)
    else:
        A = np.vstack([X, np.ones(len(X))]).T
        B = np.log(Y).reshape(-1, 1)

    AtA = A.T @ A
    AtB = A.T @ B
    sol = np.linalg.inv(AtA) @ AtB

    if tipo == "Lineal":
        m, b = sol[0,0], sol[1,0]
        Y_pred = m*X + b
        sb = "+" if b >= 0 else "-"
        eq = f"y = {m:.2f}x {sb} {abs(b):.2f}"
        params = {"m": m, "b": b}
    elif tipo == "Cuadrático":
        a, b, c = sol[0,0], sol[1,0], sol[2,0]
        Y_pred = a*X**2 + b*X + c
        sb = "+" if b >= 0 else "-"
        sc = "+" if c >= 0 else "-"
        eq = f"y = {a:.2f}x² {sb} {abs(b):.2f}x {sc} {abs(c):.2f}"
        params = {"a": a, "b": b, "c": c}
    else:
        b_exp, ln_a = sol[0,0], sol[1,0]
        a_real = np.exp(ln_a)
        Y_pred = a_real * np.exp(b_exp * X)
        eq = f"y = {a_real:.4f} · e^({b_exp:.4f}x)"
        params = {"a": a_real, "b": b_exp}

    # ── ANÁLISIS DE ERROR ───────────────────────────────────
    n = len(X)
    residuos = Y - Y_pred
    sse = np.sum(residuos ** 2)                         
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r2  = 1 - sse / ss_tot if ss_tot > 0 else 0       

    return {
        "Y_pred": Y_pred, "eq": eq, "params": params,
        "AtA": AtA, "AtB": AtB, "sol": sol,
        "residuos": residuos, "sse": sse, "r2": r2
    }

def curva_suave(X, res, tipo):
    xs = np.linspace(X.min() - 0.3, X.max() + 0.3, 300)
    p = res["params"]
    if tipo == "Lineal":
        return xs, p["m"]*xs + p["b"]
    elif tipo == "Cuadrático":
        return xs, p["a"]*xs**2 + p["b"]*xs + p["c"]
    else:
        return xs, p["a"]*np.exp(p["b"]*xs)

# ── SEL POR PUNTO ──────────────────────────────────────────────────────────
def generar_sel(X, Y, tipo):
    filas = []
    for i in range(len(X)):
        xi, yi, idx = X[i], Y[i], i + 1
        xi_fmt, yi_fmt = f"{xi:.4g}", f"{yi:.4g}"
        if tipo == "Lineal":
            fila = f"y{idx} = m·x{idx} + b   →   {yi_fmt} = m·({xi_fmt}) + b"
        elif tipo == "Cuadrático":
            xi2_fmt = f"{xi**2:.4g}"
            fila = f"y{idx} = a·x{idx}² + b·x{idx} + c   →   {yi_fmt} = a·({xi2_fmt}) + b·({xi_fmt}) + c"
        else:
            lnyi_fmt = f"{np.log(yi):.4f}"
            fila = f"ln(y{idx}) = b·x{idx} + ln(a)   →   {lnyi_fmt} = b·({xi_fmt}) + ln(a)"
        filas.append(fila)
    return filas

# ── TARJETA SUPERIOR ──────────────────────
def mostrar_tarjeta(res, tipo, color, es_mejor=False):
    if res.get("error"):
        st.markdown(f"""
        <div style="background:#11111f; border-radius:14px; border:1px dashed #ff6b6b44;
                    padding:36px 18px; text-align:center;">
            <div style="display:flex; justify-content:center; align-items:center;
                        gap:8px; margin-bottom:14px;">
                <div style="width:10px;height:10px;border-radius:50%;background:#ff6b6b"></div>
                <span style="font-size:0.72rem; font-weight:700; color:#ff6b6b;
                             letter-spacing:0.08em; text-transform:uppercase">{tipo}</span>
            </div>
            <div style="font-size:0.88rem; color:#7070a0; line-height:1.6;">
                Modelo no disponible para estos datos.<br>
                El ajuste exponencial requiere <span style="color:#e2e2f0; font-weight:600">Y &gt; 0</span>
                en todos los puntos.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    badge = "  Mejor ajuste" if es_mejor else ""
    r2_color = "#3ecfcf" if res["r2"] >= 0.95 else ("#ff9f40" if res["r2"] >= 0.80 else "#ff6b6b")

    st.markdown(f"""
    <div style="background:#11111f; border-radius:14px; border:1px solid {color}55;
                padding:16px 18px; box-sizing:border-box;">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px">
            <div style="width:10px;height:10px;border-radius:50%;background:{color}"></div>
            <span style="font-size:0.72rem; font-weight:700; color:#9090c0;
                         letter-spacing:0.08em; text-transform:uppercase">{tipo}{badge}</span>
        </div>
        <code style="display:block; background:#1a1a2e; border:1px solid {color}66;
                     border-radius:8px; padding:9px 13px; font-size:0.88rem;
                     color:{color}; font-family:'JetBrains Mono',monospace; margin-bottom:10px;">{res['eq']}</code>
        <div style="display:flex; gap:8px; margin-bottom:10px;">
            <div style="background:#1a1a2e; border-radius:8px; padding:7px 10px; flex:1; text-align:center;">
                <div style="font-size:0.6rem; color:#7070a0; text-transform:uppercase; letter-spacing:0.05em;">SSE</div>
                <div style="font-size:1rem; font-weight:700; color:#ffffff; font-family:'JetBrains Mono',monospace;">{res['sse']:.4f}</div>
            </div>
            <div style="background:#1a1a2e; border-radius:8px; padding:7px 10px; flex:1; text-align:center;">
                <div style="font-size:0.6rem; color:#7070a0; text-transform:uppercase; letter-spacing:0.05em;">R²</div>
                <div style="font-size:1rem; font-weight:700; color:{r2_color}; font-family:'JetBrains Mono',monospace;">{res['r2']:.4f}</div>
            </div>
        </div>
        <div style="font-size:0.65rem; color:#7070a0; text-transform:uppercase;
                    letter-spacing:0.06em; margin-bottom:6px;">Sistema (fila por punto)</div>
    </div>
    """, unsafe_allow_html=True)

    filas = generar_sel(X, Y, tipo)
    st.code("\n".join(filas), language=None)

# ── TABLA DE RESIDUOS ──────────────────────────────────────────────────────
def mostrar_tabla_residuos(res, tipo, color):
    """Tabla al estilo del libro: x | y_real | y_estimado | e = y_real - y_est | e²"""
    if res.get("error") or np.isnan(res.get("sse", np.nan)):
        st.markdown(f"""
        <div style="background:#11111f; border-radius:12px; border:1px dashed #ff6b6b44;
                    padding:18px 16px; margin-top:4px; text-align:center;">
            <div style="font-size:0.65rem; color:#ff6b6b; text-transform:uppercase;
                        letter-spacing:0.07em; margin-bottom:6px;">{tipo}</div>
            <div style="font-size:0.82rem; color:#7070a0;">
                Tabla no disponible — Y contiene valores ≤ 0
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    Y_pred = res["Y_pred"]
    residuos = res["residuos"]
    e2 = residuos ** 2

    rows = []
    for i in range(len(X)):
        rows.append({
            "x":            round(float(X[i]), 4),
            "y real":       round(float(Y[i]), 4),
            "ŷ estimado":   round(float(Y_pred[i]), 4),
            "e = y − ŷ":   round(float(residuos[i]), 4),
            "e²":           round(float(e2[i]), 4),
        })

    df = pd.DataFrame(rows)

    # Fila de totales
    totales = pd.DataFrame([{
        "x": "—",
        "y real": "—",
        "ŷ estimado": "—",
        "e = y − ŷ": round(float(np.sum(residuos)), 4),
        "e²": round(float(res["sse"]), 4),
    }])
    df_full = pd.concat([df.astype(str), totales.astype(str)], ignore_index=True)

    st.markdown(f"""
    <div style="background:#11111f; border-radius:12px; border:1px solid {color}33;
                padding:12px 16px; margin-top:4px;">
        <div style="font-size:0.65rem; color:#7070a0; text-transform:uppercase;
                    letter-spacing:0.07em; margin-bottom:8px;">
            Tabla de residuos  ·  SSE = Σe² = {res['sse']:.4f}  ·  R² = {res['r2']:.4f}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df_full, use_container_width=True, hide_index=True)

# ── RENDERIZADO PRINCIPAL ──────────────────────────────────────────────────
if calcular or st.session_state.get('calc', False):
    st.session_state['calc'] = True

    modelos_activos = [(modelo1, COLORES[modelo1])]
    if modelo2 != "Ninguno":
        modelos_activos.append((modelo2, COLORES[modelo2]))
    if modelo3 != "Ninguno" and modelo3 != modelo2:
        modelos_activos.append((modelo3, COLORES[modelo3]))

    resultados = {m: resolver(X, Y, m) for m, _ in modelos_activos}
    validos_para_mejor = [m for m, _ in modelos_activos if not resultados[m].get("error") and not np.isnan(resultados[m].get("sse", np.nan))]
    mejor_modelo = min(validos_para_mejor, key=lambda m: resultados[m]["sse"]) if validos_para_mejor else None

    # ── FILA: tarjetas + tabla inicial ────────────────────────────────────
    modelos_mostrar = [(m, c) for m, c in modelos_activos
                       if not resultados[m].get("error") and not np.isnan(resultados[m].get("sse", np.nan))]
    n_modelos = len(modelos_mostrar)
    widths = [2] * n_modelos + [1.2]
    cols = st.columns(widths)

    for i, (tipo, color) in enumerate(modelos_mostrar):
        with cols[i]:
            mostrar_tarjeta(resultados[tipo], tipo, color, es_mejor=(mejor_modelo is not None and tipo == mejor_modelo))

    with cols[-1]:
        st.markdown("""
        <div style="background:#11111f; border-radius:14px; border:1px solid #2e2e4e;
                    padding:14px 16px;">
            <div style="font-size:0.65rem; color:#7070a0; text-transform:uppercase;
                        letter-spacing:0.08em; font-weight:700; margin-bottom:8px">
                Datos iniciales
            </div>
        </div>
        """, unsafe_allow_html=True)
        df_ini = pd.DataFrame({"X": X, "Y": Y})
        st.dataframe(
            df_ini.style.format({"X": "{:.4g}", "Y": "{:.4g}"}),
            use_container_width=True, hide_index=True,
            height=min(36 * len(X) + 38, 420)
        )

    st.markdown("<div style='margin-top:1.4rem'></div>", unsafe_allow_html=True)

    # ── TABLAS DE RESIDUOS ────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:0.68rem; color:#7070a0; text-transform:uppercase;
                letter-spacing:0.08em; font-weight:700; margin-bottom:8px">
        📋 Análisis de error por modelo  (e = y − ŷ,  SSE = Σe²)
    </div>""", unsafe_allow_html=True)

    err_cols = st.columns(max(n_modelos, 1))
    for i, (tipo, color) in enumerate(modelos_mostrar):
        with err_cols[i]:
            mostrar_tabla_residuos(resultados[tipo], tipo, color)

    st.markdown("<div style='margin-top:1.4rem'></div>", unsafe_allow_html=True)

    # ── GRÁFICA ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:0.68rem; color:#7070a0; text-transform:uppercase;
                letter-spacing:0.08em; font-weight:700; margin-bottom:6px">
        📊 Gráfica de ajuste
    </div>""", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X, y=Y, mode='markers', name='Datos reales',
        marker=dict(color='#ffffff', size=11, symbol='circle',
                    line=dict(width=2, color='#6c63ff'))
    ))
    for tipo, color in modelos_activos:
        if resultados[tipo].get("error") or np.isnan(resultados[tipo].get("sse", np.nan)):
            continue
        xs, ys = curva_suave(X, resultados[tipo], tipo)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines',
            name=f'{tipo}: {resultados[tipo]["eq"]}',
            line=dict(color=color, width=3, dash=DASH[tipo])
        ))

    fig.update_layout(
        plot_bgcolor='#11111f', paper_bgcolor='#0a0a0f',
        height=390, margin=dict(l=50, r=30, t=20, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color="#c0c0e0", size=11, family="JetBrains Mono"),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=True, gridcolor="#1e1e2e", linecolor="#2e2e4e",
                   tickfont=dict(color="#7070a0", family="JetBrains Mono"),
                   title=dict(text="X", font=dict(color="#7070a0"))),
        yaxis=dict(showgrid=True, gridcolor="#1e1e2e", linecolor="#2e2e4e",
                   tickfont=dict(color="#7070a0", family="JetBrains Mono"),
                   title=dict(text="Y", font=dict(color="#7070a0"))),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── VEREDICTO FINAL ───────────────────────────────────────────────────
    modelos_validos = [
        m for m, _ in modelos_activos
        if not resultados[m].get("error") and not np.isnan(resultados[m].get("sse", np.nan))
    ]

    if len(modelos_validos) > 1:
        sses_v = {m: resultados[m]["sse"] for m in modelos_validos}
        sorted_v = sorted(sses_v, key=sses_v.get)

        ranking_partes = [f"{m}  (SSE={sses_v[m]:.4f})" for m in sorted_v]
        ranking_txt = "  <  ".join(ranking_partes)

        sse_min = sses_v[sorted_v[0]]
        empatados = [m for m in modelos_validos if abs(sses_v[m] - sse_min) < 1e-10]

        if len(empatados) > 1:
            nombres = " y ".join(f'<span style="color:{COLORES[m]};font-weight:700">{m}</span>' for m in empatados)
            conclusion = f"Los modelos {nombres} tienen el <strong style='color:#ffffff'>mismo SSE = {sse_min:.4f}</strong>, por lo que cualquiera es igualmente óptimo para estos datos."
            border_color = "#7070a0"
        else:
            mejor = sorted_v[0]
            mejor_color = COLORES[mejor]
            border_color = mejor_color
            conclusion = f'El modelo <span style="color:{mejor_color};font-weight:700">{mejor}</span> presenta el <strong style="color:#ffffff">menor SSE = {sse_min:.4f}</strong>, por lo que es el <span style="color:{mejor_color};font-weight:700">mejor ajuste</span> para estos datos.'

        st.markdown(f"""
        <div style="background:#11111f; border-radius:12px; border:1px solid {border_color}44;
                    padding:14px 20px; margin-top:0.4rem; display:flex; align-items:flex-start; gap:14px">
            <span style="font-size:1.4rem; margin-top:2px">🏆</span>
            <div>
                <div style="font-size:0.68rem; color:#7070a0; text-transform:uppercase;
                            letter-spacing:0.06em; margin-bottom:4px">
                    Comparación de modelos · menor SSE = mejor ajuste
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.82rem; color:#c0c0e0; margin-bottom:6px;">
                    {ranking_txt}
                </div>
                <div style="font-size:0.88rem; color:#e2e2f0;">{conclusion}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="background:#11111f; border-radius:14px; border:1px dashed #2e2e4e;
                padding:40px; text-align:center; margin-top:2rem">
        <div style="font-size:2rem; margin-bottom:12px">📐</div>
        <p style="color:#7070a0; margin:0">Ingresa los datos en el panel lateral y presiona
        <strong style="color:#6c63ff">Calcular Modelos</strong></p>
    </div>
    """, unsafe_allow_html=True)