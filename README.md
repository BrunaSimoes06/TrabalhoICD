# O Papel da Ciência de Dados nos Avanços dos Cuidados de Saúde

Este projeto realiza uma análise bibliométrica e de conteúdo de artigos científicos, com o objetivo de identificar padrões, tópicos e métricas relevantes sobre o papel da Ciência de Dados nos Avanços dos Cuidados de Saúde. A análise é dividida em duas partes principais: análise bibliométrica e análise de conteúdo, e é apresentada num dashboard interativo utilizando o Streamlit.

## Requisitos

Para instalar este projeto e as dependências, deve garantir os seguintes requisitos:

- **Python** 3.x
- **Anaconda** ou **Miniconda** para gerir o ambiente
- As seguintes bibliotecas Python (que serão instaladas automaticamente usando o arquivo `streamlit_env.yml`):
  - `streamlit`
  - `pandas`
  - `numpy`
  - `geopy`
  - `plotly`
  - `time`
  - `matplotlib.pyplot`
  - `collections`
  - `wordcloud`
  - `sklearn.feature_extraction.text`
  - `sklearn.cluster`
  - `seaborn`
  - `geopandas`
  - `folium`
  - `sklearn.decomposition`

## Para reprodução da análise 

**Pela grande dimensão dos dados extraídos da Scopus (scopus (6).csv) e após a limpeza e processamento (df_limpo.csv), foi necessário enviá-los via Google Drive.** 
Poderão ser encontrados aqui: https://drive.google.com/drive/folders/1x1bU2DDUJmvV0JjL-Q-VphkCj5reXHil?usp=sharing 

#### Instruções para uso dos dados: 
  - **Descarregue os arquidos da Drive**
  - **Coloque os arquivos dentro da pasta "notebook", que deverá ser o diretório. Pois o código foi projetado para carregar os dados diretamente do diretório** 


## Configuração do ambiente e do diretório

1. **Instalar o Anaconda, caso não esteja** 
2. **Criar o ambiente Conda utilizando o arquivo `streamlit_env.yml`**:
    Este parte deve ser apenas executada uma vez, para outras ocasiões que replicar o projeto, não será necessário.
    - Abra o **Anaconda Prompt** ou o terminal.
    - Navegue até à pasta do projeto onde está o arquivo `streamlit_env.yml`.
    - Execute o comando:
      ```bash
      conda env create --file streamlit_env.yml
      ```
    - Isso criará um ambiente Conda com todas as bibliotecas necessárias.

3. **Ative o ambiente**:
    - Para ativar o ambiente criado, execute:
      ```bash
      conda activate streamlit_env
      ```
    Deverá ficar com algo: (streamlit_env) C:\Users\Utilizador

4. **Verificar a instalação:**
    - Execute:
    ```bash
      conda list
      ```
    - Deve garantir as bibliotecas acima mencionadas, caso haja algum problema via conda, pode instalar o pacote pelo pip:
    ```bash
      pip install nome_do_pacote
      ```
5. **Configurar o diretório:**
    - Após ativar o ambiente, deverá configurar o diretório, execute:
    ```bash
      cd C:\Users\Utilizador\caminho\pasta\zip\notebook
      ```
    
## Organização do Conteúdo

### Análise Bibliométrica
 A análise bibliométrica foi realizada para examinar as tendências e o impacto de publicações científicas em um determinado conjunto de dados extraídos da Scopus. 

1. **Execute a Análise Bibliométrica através do Jupyter Notebook**:
Nota: Na pasta zip tem o ficheiro em formato .ipynb e .html

    - Primeiro, garanta que o ambiente está devidamente configurado como explicado na secção anterior e no diretório correto: 
    No Anaconda Prompt deverá ser: 
**(streamlit_env) C: caminho\diretório**

    - Abra o **Jupyter Notebook**:
      ```bash
      (streamlit_env) cd caminho\diretório > jupyter notebook
      ```
    - Execute o código de análise bibliométrica no arquivo analise_bibliometrica.ipynb.

2. **Execute a Análise de Conteúdo**:
    - Se ainda tiver no Jupyter Notebook poderá também encontrar o ficheiro .ipynb da Análise de Conteúdo, realize as etapas de pré-processamento e análise de conteúdo conforme indicado no notebook.
    - Caso não tenha aberto, reabra através:
     ```bash
      (streamlit_env) cd caminho\diretório > jupyter notebook
      ``` 
      E execute o código de análise de conteúdo no arquivo analise_conteudo.ipynb.
    
    - Após execução do código, será apresentado no diretório em formato .png, o resultado do cleaning e do lda, que foram assim configurados para utilização no relatório. 

    - Neste código é também descarregado os ficheiros em formato .csv dos dados limpos, o top 10 de termos do TF-IDF e os topicos da LDA para facilitar na execução da dashboard.


3. **Execução da Dashboard**
A dashboard foi desenvolvida utilizando o Streamlit para apresentar visualmente os resultados das análises bibliométrica e de conteúdo.

    - Garanta que o ambiente está ativo e o diretório correto.

    - No terminal, execute o Streamlit com o comando:
      ```bash
      streamlit run dashboard.py
      ```
    - O dashboard será aberto automaticamente no seu navegador e poderá visualizar os gráficos interativos e a análise.

    Nota: Caso este último passo não ocorra, copie e cole no navegador o link fornecido no terminal (geralmente algo como http://localhost:8501).




