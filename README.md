# Análise Bibliométrica e de Conteúdo

Este projeto realiza uma análise bibliométrica e de conteúdo de artigos científicos, com o objetivo de identificar padrões, tópicos e métricas relevantes sobre um determinado tema. A análise é dividida em duas partes principais: análise bibliométrica e análise de conteúdo, e é apresentada em um dashboard interativo utilizando o Streamlit.

## Requisitos

- **Python** 3.x
- **Anaconda** ou **Miniconda** para gerenciamento do ambiente
- As seguintes bibliotecas Python (que serão instaladas automaticamente usando o arquivo `streamlit_env.yml`):
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `sklearn`
  - `streamlit`
  - `nltk`
  - `wordcloud`
  - ... (adicione outras bibliotecas conforme necessário)

## Como configurar o ambiente

1. **Instale o Anaconda** (se ainda não tiver instalado): [Download Anaconda](https://www.anaconda.com/products/distribution)
2. **Crie um ambiente Conda utilizando o arquivo `streamlit_env.yml`**:
    - Abra o **Anaconda Prompt** ou o terminal.
    - Navegue até a pasta do projeto onde o arquivo `streamlit_env.yml` está localizado.
    - Execute o comando:
      ```bash
      conda env create -f streamlit_env.yml
      ```
    - Isso criará um ambiente Conda com todas as bibliotecas necessárias.

3. **Ative o ambiente**:
    - Para ativar o ambiente criado, execute:
      ```bash
      conda activate streamlit_env
      ```

## Como rodar o código

1. **Execute a Análise Bibliométrica**:
    - Abra o **Jupyter Notebook** (se estiver usando o Jupyter):
      ```bash
      jupyter notebook
      ```
    - Execute o código de análise bibliométrica no notebook.

2. **Execute a Análise de Conteúdo**:
    - Realize as etapas de pré-processamento e análise de conteúdo conforme indicado no notebook.

3. **Execute o Dashboard**:
    - No terminal, na pasta do projeto, execute o Streamlit com o comando:
      ```bash
      streamlit run dashboard.py
      ```
    - O dashboard será aberto automaticamente no seu navegador. Você pode visualizar os gráficos interativos e a análise.

## Arquivos de Entrada

- O arquivo de entrada usado para análise bibliométrica é o arquivo `.csv` da Scopus ou outro arquivo de dados relevante. Ele deve estar localizado na pasta do projeto.



