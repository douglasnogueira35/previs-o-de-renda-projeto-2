import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------
# Configuração da página
# ---------------------------
sns.set(context='talk', style='ticks')
st.set_page_config(
     page_title="Análise de Previsão de Renda",
     page_icon="💰",
     layout="wide",
)

st.write('# Análise exploratória da previsão de renda')

# ---------------------------
# Entendimento do negócio
# ---------------------------
st.markdown("## Entendimento do Negócio")
st.markdown("""
Este projeto busca analisar variáveis demográficas e financeiras dos clientes para identificar padrões relacionados à renda e risco de inadimplência.
""")

# ---------------------------
# Dicionário de Dados
# ---------------------------
st.markdown("## Dicionário de Dados")
st.markdown("""
- **data_ref**: data de referência da coleta  
- **renda**: renda mensal declarada  
- **posse_de_imovel**: indicador de posse de imóvel  
- **posse_de_veiculo**: indicador de posse de veículo  
- **qtd_filhos**: número de filhos  
- **tipo_renda**: categoria da fonte de renda  
- **educacao**: nível de escolaridade  
- **estado_civil**: estado civil  
- **tipo_residencia**: tipo de residência  
- **valor_credito**: valor de crédito disponível  
""")

# ---------------------------
# Upload do CSV
# ---------------------------
arquivo = st.file_uploader("Envie o arquivo CSV de renda", type="csv")

if arquivo is not None:
    renda = pd.read_csv(arquivo)

    # ---------------------------
    # Limpeza de Dados
    # ---------------------------
    st.write("## Limpeza de Dados")
    st.write("Valores nulos por coluna:")
    st.write(renda.isnull().sum())

    # Exemplo de tratamento simples
    renda = renda.dropna()

    # ---------------------------
    # Feature Engineering
    # ---------------------------
    if "idade" in renda.columns:
        renda["faixa_idade"] = pd.cut(renda["idade"], bins=[18,30,45,60,80],
                                      labels=["18-30","31-45","46-60","61-80"])
    if "valor_credito" in renda.columns and "renda" in renda.columns:
        renda["ratio_credito_renda"] = renda["valor_credito"] / renda["renda"]

    # ---------------------------
    # Gráficos Interativos
    # ---------------------------
    st.write("## Gráficos Interativos")
    opcoes = ["posse_de_imovel","posse_de_veiculo","qtd_filhos","tipo_renda","educacao","estado_civil","tipo_residencia"]
    var = st.selectbox("Escolha a variável para análise:", opcoes)

    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x=var, y="renda", data=renda, ax=ax)
    st.pyplot(fig)

    # ---------------------------
    # Modelagem e Métricas
    # ---------------------------
    st.write("## Modelagem e Avaliação")

    # Exemplo simples: regressão linear para prever renda
    if "idade" in renda.columns and "qtd_filhos" in renda.columns:
        X = renda[["idade","qtd_filhos"]]
        y = renda["renda"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        modelo = LinearRegression()
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)

        st.write("R²:", r2_score(y_test, y_pred))
        st.write("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

else:
    st.warning("Por favor, envie o arquivo CSV para visualizar os gráficos e análises.")