import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit 
from sklearn.metrics import r2_score

def func_exp(x, a, b, c):
    return a * np.exp(b * (x - 1990)) + c

def func_lin(x, a, b):
    return a * (x - 1990) + b

def Brazil (df, titulo):
    cols = st.columns(2)
    df = df.loc[df['Local'] == 'Brasil']

    # Criando gráfico base
    fig = go.Figure()

    # Adicionando pontos de dados reais com destaque e nome
    fig.add_trace(go.Scatter(
        x=df["Ano"], 
        y=df["Morte5"], 
        mode="markers",
        marker=dict(size=10, color="black", line=dict(width=2, color="white")), 
        name="Dados Reais"
    ))
    
    # Ajustando uma curva linear
    params, _ = curve_fit(func_lin, df['Ano'], df['Morte5'], p0=(0, 0))
    x_pred = np.linspace(1985, 2030, (2030-1985)*10)
    y_pred = func_lin(x_pred, *params)
    fig.add_trace(go.Scatter(x=x_pred, y=y_pred, mode='lines', name="Ajuste Linear", line=dict(dash='dash')))
    with cols[0]:
        st.write("### Ajuste linear")
        st.markdown(f'$y = {params[0]:.2f}(x - 1990) + {params[1]:.2f}$')
        st.markdown(f'$R^2 = {r2_score(df["Morte5"], func_lin(df["Ano"], *params)):.4f}$')

    # Adicionando uma curva exponencial
    params, _ = curve_fit(func_exp, df['Ano'], df['Morte5'], p0=(50, -0.1, -1))
    y_pred = func_exp(x_pred, *params)
    fig.add_trace(go.Scatter(x=x_pred, y=y_pred, mode='lines', name="Ajuste Exponencial", line=dict(color='green', dash='dash')))
    with cols[1]:
        st.write("### Ajuste exponencial")
        st.markdown(f'$y = {params[0]:.2f}e^{{{params[1]:.2f}(x-1990)}} {"+" if params[2] >= 0 else ""} {params[2]:.2f}$')
        st.markdown(f'$R^2 = {r2_score(df["Morte5"], func_exp(df["Ano"], *params)):.4f}$')

    # Adicionando a meta (25 mortes por 1000 nascidos vivos até 2030)
    fig.add_hline(y=25, line_dash="dot", line_color="red", annotation_text="Meta 2030", annotation_position="bottom right")
    st.plotly_chart(fig, use_container_width=True)

st.write("# Valores no Brasil")
with st.expander("Brasil (1991, 2000 e 2010) - Atlas Brasil"):
    Brazil(pd.read_csv('Dados Tratados\Atlas.csv'), 'Mortalidade Infantil no Brasil (1991-2010)')
with st.expander("Brasil (1991, 2000 e 2010 + 2020) - Atlas + IBGE (Tabela 6695)"):
    Brazil(pd.read_csv('Dados Tratados\Atlas_ODS.csv'), 'Mortalidade Infantil no Brasil (1991-2010 + 2020)')
with st.expander("Brasil (1990 a 2022) - IBGE (Tabela 6695)"):
    Brazil(pd.read_csv('Dados Tratados\ODS.csv'), 'Mortalidade Infantil no Brasil (1990-2022)')







def Regioes(df, titulo):
    regioes = st.multiselect("Selecione as regiões", ['Brasil', "Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"], default=['Brasil', "Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"])
    cores = px.colors.qualitative.Set1
    fig = go.Figure()

    for i, regiao in enumerate(regioes):
        df_regiao = df[df["Local"] == regiao]
        if df_regiao.empty:
            continue  # Pula caso não tenha dados da região

        # Ajuste exponencial
        params, _ = curve_fit(func_exp, df_regiao['Ano'], df_regiao['Morte5'], p0=(50, -0.1, -1))
        x_pred = np.linspace(1985, 2030, 100)  # Intervalo suave
        y_pred = func_exp(x_pred, *params)  # Calcula valores preditos

        # Adiciona os pontos da região
        fig.add_trace(go.Scatter(
            x=df_regiao["Ano"], 
            y=df_regiao["Morte5"], 
            mode="markers",
            marker=dict(size=8, color=cores[i], line=dict(width=1, color="black")),
            name=f"{regiao}"
        ))

        # Adiciona a curva exponencial ajustada
        fig.add_trace(go.Scatter(
            x=x_pred, 
            y=y_pred, 
            mode='lines', 
            name=f"{regiao}",
            line=dict(color=cores[i], dash='dash'),
            # Oculta a legenda para não poluir
            showlegend=False
        ))


    fig.add_hline(y=25, line_dash="dot", line_color="red", annotation_text="Meta 2030", annotation_position="bottom right")
    st.plotly_chart(fig, use_container_width=True)

def proporcao_regiao(df, regioes):
    vals = []
    for regiao in regioes:
        df_regiao = df[df["Local"].isin(regioes[regiao])]
        for ano in range(df_regiao["Ano"].min(), df_regiao["Ano"].max() + 1):
            df_ano = df_regiao[df_regiao["Ano"] == ano]
            p = len(df_ano[df_ano["Morte5"] <= 25]) / len(df_ano)
            vals.append({"Ano": ano, "Região": regiao, "Proporção": p})
    
    df_prop = pd.DataFrame(vals)
    
    fig = px.bar(df_prop, x="Região", y="Proporção",
                 animation_frame="Ano", color="Região",
                 range_y=[0, 1], title="Proporção de Estados que atingiram a meta por Região")
    st.plotly_chart(fig, use_container_width=True)

def ranking_regiao(df, regioes):
    df = df[df["Local"].isin(regioes.keys())]
    fig = px.bar(df, 
                 x="Morte5", 
                 y="Local", 
                 color="Morte5",
                 animation_frame="Ano",
                 orientation="h",
                 title="Ranking de Mortalidade Infantil por Estado (1990-2022)",
                    range_x=[0, df["Morte5"].max() + 5],
                    range_color=[0, df["Morte5"].max() + 5])
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)





st.write("# Valores por Região")
with st.expander("Evolução por Região"):
    Regioes(pd.read_csv('Dados Tratados\ODS.csv'), 'Mortalidade Infantil por Região (1990-2022)')
with st.expander("Mapa da Evolução por Região"):
    st.write("Em construção... (O mapa das regiões vem aqui)")
with st.expander("Proporção de Estados que atingiram a meta por Região"):
    regioes = {
        'Norte': ['Rondônia', 'Acre', 'Amazonas', 'Roraima', 'Pará', 'Amapá', 'Tocantins'],
        'Nordeste': ['Maranhão', 'Piauí', 'Ceará', 'Rio Grande do Norte', 'Paraíba', 'Pernambuco', 'Alagoas', 'Sergipe', 'Bahia'],
        'Sudeste': ['Minas Gerais', 'Espírito Santo', 'Rio de Janeiro', 'São Paulo'],
        'Sul': ['Paraná', 'Santa Catarina', 'Rio Grande do Sul'],
        'Centro-Oeste': ['Mato Grosso do Sul', 'Mato Grosso', 'Goiás', 'Distrito Federal']
    }
    regioes['Brasil'] = regioes['Norte'] + regioes['Nordeste'] + regioes['Sudeste'] + regioes['Sul'] + regioes['Centro-Oeste']

    proporcao_regiao(pd.read_csv('Dados Tratados\ODS.csv'), regioes)
with st.expander("Ranking de Mortalidade Infantil por Estado"):
    ranking_regiao(pd.read_csv('Dados Tratados\ODS.csv'), regioes)








st.write("# Valores por Estado")
with st.expander("Evolução por Estado"):
    st.write("Em construção... (O gráfico de evolução por estado vem aqui)")