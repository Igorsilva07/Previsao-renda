# app.py (ou Projeto_Igor.py)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from lightgbm import LGBMRegressor # Necessário para carregar o modelo LightGBM

# --- Configurações Iniciais ---
# Configurações de layout (opcional, mas pode ajudar na aparência)
st.set_page_config(layout="centered", initial_sidebar_state="expanded") # ou "wide" para mais espaço


# --- 1. Carregar o Modelo e Parâmetros de Pré-processamento ---
# AQUI USAMOS CAMINHOS RELATIVOS, QUE SÃO A MELHOR PRÁTICA.
# ISSO SIGNIFICA QUE O APP.PY E A PASTA 'MODELS' DEVEM ESTAR NO MESMO DIRETÓRIO.
try:
    # Caminhos relativos para os arquivos salvos dentro da pasta 'models'
    model_path = os.path.join('.', 'models', 'lgbm_regressor_renda.joblib')
    median_tempo_emprego_path = os.path.join('.', 'models', 'mediana_tempo_emprego.joblib')
    training_columns_path = os.path.join('.', 'models', 'training_columns.joblib')

    # Carregar o modelo
    model = joblib.load(model_path)
    
    # Carregar a mediana do tempo_emprego (para imputação de nulos)
    mediana_tempo_emprego_treino = joblib.load(median_tempo_emprego_path)
    
    # Carregar a lista de colunas usadas no treinamento (para One-Hot Encoding consistente)
    expected_model_columns = joblib.load(training_columns_path)
    
    st.sidebar.success("Modelo e parâmetros de pré-processamento carregados com sucesso!")

except FileNotFoundError as e:
    st.sidebar.error(f"Erro ao carregar arquivos necessários: {e}")
    st.sidebar.warning("Por favor, certifique-se de que o modelo e os arquivos de parâmetros (.joblib) foram salvos corretamente na pasta 'models' executando o notebook completo. A pasta 'models' deve estar no mesmo diretório que este script.")
    st.stop() # Interrompe a execução do app se os arquivos não forem encontrados
except Exception as e:
    # Captura qualquer outro erro inesperado
    st.sidebar.error(f"Ocorreu um erro inesperado ao carregar os arquivos: {e}")
    st.sidebar.warning("Isso pode ser devido a uma incompatibilidade de versão de bibliotecas (LightGBM, Scikit-learn) ou ao modelo ter sido salvo de forma inesperada. Verifique as versões no ambiente do notebook e do Streamlit.")
    st.stop()

# --- Funções de Pré-processamento (MIMICAM A ETAPA 3 DO NOTEBOOK) ---
# Esta função é CRÍTICA. Ela precisa replicar EXATAMENTE o pré-processamento do seu notebook.
def preprocess_input(input_df, mediana_tempo_emprego, expected_cols):
    
    # Garante que a entrada é um DataFrame, mesmo que seja uma única linha
    if not isinstance(input_df, pd.DataFrame):
        input_df = pd.DataFrame([input_df])

    # 1. Tratar NaN (tempo_emprego)
    input_df['tempo_emprego'] = input_df['tempo_emprego'].fillna(mediana_tempo_emprego)

    # 2. Criar mes_ref, ano_ref
    # Para uma previsão pontual, usamos a data atual (ou uma data de referência fixa se preferir)
    hoje = datetime.now() # Ou datetime(2023, 1, 1) se quiser uma data fixa para consistência
    input_df['mes_ref'] = hoje.month
    input_df['ano_ref'] = hoje.year

    # 3. Criar faixa_idade
    bins_idade = [0, 18, 30, 45, 60, 100]
    labels_idade = ['0-18', '19-30', '31-45', '46-60', '61+']
    # 'include_lowest=True' e 'right=False' para replicar pd.cut padrão se usado no notebook
    input_df['faixa_idade'] = pd.cut(input_df['idade'], bins=bins_idade, labels=labels_idade, right=False, include_lowest=True)

    # 4. Converter booleanos para int
    input_df['posse_de_veiculo'] = input_df['posse_de_veiculo'].astype(int)
    input_df['posse_de_imovel'] = input_df['posse_de_imovel'].astype(int)

    # 5. One-Hot Encoding
    categorical_cols_for_ohe = [
        'sexo', 'tipo_renda', 'educacao', 'estado_civil',
        'tipo_residencia', 'mes_ref', 'ano_ref', 'faixa_idade'
    ]
    
    # Aplicar One-Hot Encoding
    processed_df = pd.get_dummies(input_df, columns=categorical_cols_for_ohe, drop_first=True)
    
    # Garantir que todas as colunas esperadas pelo modelo (do treino) estão presentes
    # Preencher com 0 se a coluna dummy não existir para a entrada atual
    for col in expected_cols:
        if col not in processed_df.columns:
            processed_df[col] = 0 
    
    # Remover colunas extras que não estavam no treino (se o Streamlit gerou algo inesperado)
    extra_cols = set(processed_df.columns) - set(expected_cols)
    if extra_cols:
        processed_df = processed_df.drop(columns=list(extra_cols))

    # Reordenar as colunas para que correspondam EXATAMENTE à ordem das colunas de treinamento
    # Isso é fundamental para a previsão correta.
    processed_df = processed_df[expected_cols]

    return processed_df

# --- 2. Interface do Streamlit ---
st.title('💰 Previsão de Renda do Cliente')
st.markdown("Utilize este aplicativo para prever a renda mensal de um cliente com base em suas características.")

# Barra lateral para entrada de dados
st.sidebar.header('Dados do Cliente')

# Entradas do usuário - Os nomes das chaves (e.g., 'sexo', 'idade') devem corresponder aos nomes das colunas originais
# no seu DataFrame ANTES do pré-processamento.
sexo_input = st.sidebar.selectbox('Sexo', ['F', 'M'])
posse_veiculo_input = st.sidebar.checkbox('Possui Veículo?', value=False) # Adicione um valor padrão
posse_imovel_input = st.sidebar.checkbox('Possui Imóvel?', value=True) # Adicione um valor padrão
qtd_filhos_input = st.sidebar.slider('Quantidade de Filhos', 0, 10, 0)
tipo_renda_input = st.sidebar.selectbox('Tipo de Renda', ['Assalariado', 'Empresário', 'Servidor público', 'Pensionista', 'Bolsista', 'Autônomo'])
educacao_input = st.sidebar.selectbox('Educação', ['Primário', 'Secundário', 'Superior incompleto', 'Superior completo', 'Pós graduação'])
estado_civil_input = st.sidebar.selectbox('Estado Civil', ['Solteiro', 'Casado', 'Viúvo', 'Separado', 'União'])
tipo_residencia_input = st.sidebar.selectbox('Tipo de Residência', ['Casa', 'Governamental', 'Com os pais', 'Aluguel', 'Estúdio', 'Comunitário'])
idade_input = st.sidebar.slider('Idade', 18, 80, 30)
tempo_emprego_input = st.sidebar.slider('Tempo de Emprego (anos)', 0.0, 35.0, 5.0, step=0.1) # Adicionado step para floats
qt_pessoas_residencia_input = st.sidebar.slider('Pessoas na Residência', 1.0, 10.0, 1.0, step=1.0) # Adicionado step


# Botão para Prever
if st.sidebar.button('Prever Renda'):
    # Coletar os dados de entrada em um dicionário
    input_data_dict = {
        'sexo': sexo_input,
        'posse_de_veiculo': posse_veiculo_input,
        'posse_de_imovel': posse_imovel_input,
        'qtd_filhos': qtd_filhos_input,
        'tipo_renda': tipo_renda_input,
        'educacao': educacao_input,
        'estado_civil': estado_civil_input,
        'tipo_residencia': tipo_residencia_input,
        'idade': idade_input,
        'tempo_emprego': tempo_emprego_input,
        'qt_pessoas_residencia': qt_pessoas_residencia_input
        # data_ref não é passada diretamente pois mes_ref e ano_ref são gerados no pré-processamento
    }

    try:
        # Pré-processar os dados de entrada
        processed_input_df = preprocess_input(input_data_dict, mediana_tempo_emprego_treino, expected_model_columns)
        
        # Realizar a previsão
        prediction = model.predict(processed_input_df)[0]
        
        st.subheader('Previsão de Renda:')
        st.success(f'R$ {prediction:,.2f}') # Formato monetário com 2 casas decimais
        st.balloons() # Um efeito visual legal!

    except Exception as e:
        st.error(f"Ocorreu um erro ao fazer a previsão: {e}")
        st.warning("Por favor, verifique se todos os dados de entrada estão corretos e se o modelo foi carregado sem problemas.")

st.markdown("---")
st.markdown("Para mais detalhes sobre o modelo e o projeto, consulte o notebook CRISP-DM.")