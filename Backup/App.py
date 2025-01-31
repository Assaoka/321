import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import Analises

Analises.analise()

Nivel = ['Brasil', 'Região', 'Estado']
def nivel(lvl):
    if lvl == 'Brasil': return 'BR'
    elif lvl == 'Região': return 'GR'
    elif lvl == 'Estado': return 'UF'
lvl = st.selectbox('Nível', Nivel)        

#ano = st.slider('Ano', 1990, 2022, 2022)
 
ods = pd.read_csv('Dados Tratados/ODS.csv')
df = ods.loc[(ods['Nível'] == nivel(lvl))]# & (ods['Ano'] == ano)]
#st.dataframe(df, use_container_width=True)  

regioes = {
    'Norte': ['Rondônia', 'Acre', 'Amazonas', 'Roraima', 'Pará', 'Amapá', 'Tocantins'],
    'Nordeste': ['Maranhão', 'Piauí', 'Ceará', 'Rio Grande do Norte', 'Paraíba', 'Pernambuco', 'Alagoas', 'Sergipe', 'Bahia'],
    'Sudeste': ['Minas Gerais', 'Espírito Santo', 'Rio de Janeiro', 'São Paulo'],
    'Sul': ['Paraná', 'Santa Catarina', 'Rio Grande do Sul'],
    'Centro-Oeste': ['Mato Grosso do Sul', 'Mato Grosso', 'Goiás', 'Distrito Federal']
}
regioes['Brasil'] = regioes['Norte'] + regioes['Nordeste'] + regioes['Sudeste'] + regioes['Sul'] + regioes['Centro-Oeste']

# Quero pegar quantos estados cumpriram a meta por região por ano
vals = []
for região in ods.loc[ods['Nível'] == 'GR'].iterrows():
    df_reg = df.loc[(df['Local'].isin(regioes[região[1]['Local']]))
                    & (df['Ano'] == região[1]['Ano'])]
    total = len(df_reg)
    cumpriram = len(df_reg.loc[df_reg['Morte5'] <= 25])
    vals.append([região[1]['Ano'], região[1]['Local'], cumpriram, total])
vals = pd.DataFrame(vals, columns=['Ano', 'Local', 'Cumpriram', 'Total'])
vals['%'] = vals['Cumpriram'] / vals['Total'] * 100
#vals

# Gráfico de barras 100% empilhadas com a proporção de estados que cumpriram a meta por região (animado por ano)
# Gostaria que tivesse uma barra mais clara empilhada em cima da barra escura, com o que não cumpriu a meta

fig = px.bar(vals,
                x='Local',
                y='%',
                color='Local',
                animation_frame='Ano',
                title='Proporção de estados que cumpriram a meta de mortalidade infantil por região (1990-2022)',
                labels={'Cumpriram': 'Estados que cumpriram a meta', 'Local': 'Região'},
                range_y=[0, 100],
                range_color=[0, 100])
st.write(fig)



# Criar gráfico animado
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
st.write(fig)


# Gráfico dos dados por região + regressões (Linear, Exponencial). X = Ano, Y = Morte5, Curvas = Local (Norte, Nordeste, Sudeste, Sul, Centro-Oeste)
fig = px.scatter(df,
                 x="Ano", 
                 y="Morte5", 
                 color="Local",  
                 title="Mortalidade Infantil por Região (1990-2022)",
                 labels={'Morte5': 'Mortalidade Infantil', 'Local': 'Região'})

from scipy.optimize import curve_fit
def func_exp(x, a, b, c):
    return a * np.exp(b * (x - 1990)) + c

def ajustar_curva(x, y):
    params, _ = curve_fit(func_exp, x, y, p0=(50, -0.1, -1))
    return params

ano_meta = 2030
meta = 25
fig.add_shape(type="line", x0=1985, x1=ano_meta, y0=meta, y1=meta, line=dict(color="red", width=3))

x_pred = np.linspace(1985, 2035, 100)
for regiao in regioes:
    if regiao == 'Brasil': continue
    a = df.loc[df['Local'] == regiao]
    x = a['Ano']
    y = a['Morte5']
    params = ajustar_curva(x, y)
    y_pred = func_exp(x_pred, *params)
    fig.add_trace(px.line(x=x_pred, y=y_pred).data[0])

def func_lin(x, a, b):
    return a * x + b

def ajustar_curva_lin(x, y):
    params, _ = curve_fit(func_lin, x, y, p0=(0, 0))
    return params

for regiao in regioes:
    if regiao == 'Brasil': continue
    a = df.loc[df['Local'] == regiao]
    x = a['Ano']
    y = a['Morte5']
    params = ajustar_curva_lin(x, y)
    y_pred = func_lin(x_pred, *params)
    fig.add_trace(px.line(x=x_pred, y=y_pred).data[0])

st.write(fig)


# Histograma da mortalidade infantil por estado
mun = pd.read_csv('Dados Tratados\Atlas_Municipios_SP.csv')
mun['Morte5'] = mun['Morte5'].apply(lambda x: (x.replace(',', '.')))

fig = px.histogram(mun, 
                   x="Morte5",
                   animation_frame="Ano",
                   range_x=[0, 50],
                   # Queria que os bins fossem sempre do mesmo tamanho, mas eles mudam com o ano
                   title="Histograma da Mortalidade Infantil por Estado (1990-2022)")
st.write(fig)


# Bloxplot da mortalidade infantil por região
df = ods.loc[ods['Nível'] == 'UF']
df['Região'] = df['Local'].apply(lambda x: [regiao for regiao in regioes if x in regioes[regiao]][0])

fig = px.box(df, 
             x="Região", 
             y="Morte5",
             range_y=[0, 100],
             animation_frame="Ano", 
             title="Mortalidade Infantil por Região (1990-2022)")
st.write(fig)
