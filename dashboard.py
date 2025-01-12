import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import geopandas as gpd
import folium
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer 

# Título
st.title('O papel da Ciência de Dados nos Avanços dos Cuidados de Saúde')

# Introdução
st.write("""
         Nesta dashboard, exploramos como a ciência de dados tem impulsionado os avanços nos cuidados de saúde,
         especialmente em áreas como a tomada de decisão clínica, medicina personalizada e telemedicina.
         Através de análises bibliométricas e de conteúdo, vamos explorar insights e visualizações!
         """)


df_scopus = pd.read_csv('C:\\Users\\Utilizador\\Desktop\\Universidade\\1º semestre 24.25\\ICD\\notebook\\scopus (6).csv')
df_limpo = pd.read_csv('C:\\Users\\Utilizador\\Desktop\\Universidade\\1º semestre 24.25\\ICD\\notebook\\df_limpo.csv')
df_tfidf = pd.read_csv('C:\\Users\\Utilizador\\Desktop\\Universidade\\1º semestre 24.25\\ICD\\notebook\\top_10_avg_tfidf_terms.csv')
df_lda_topics = pd.read_csv('C:\\Users\\Utilizador\\Desktop\\Universidade\\1º semestre 24.25\\ICD\\notebook\\lda_topics.csv')

# Exibir as primeiras linhas de cada DataFrame para confirmação
#print("df_scopus:")
#print(df_scopus.head())  # Exibe as primeiras 5 linhas do DataFrame df_scopus

#print("\ndf_limpo:")
#print(df_limpo.head())  # Exibe as primeiras 5 linhas do DataFrame df_limpo

#print("\ndf_tfidf:")
#print(df_tfidf.head())  # Exibe as primeiras 5 linhas do DataFrame df_tfidf

#print("\ndf_lda_topics:")
#print(df_lda_topics.head())  # Exibe as primeiras 5 linhas do DataFrame df_lda_topics


# Configuração da sidebar com diferentes opções
st.sidebar.title("Escolha a Análise")
menu = st.sidebar.radio(
    "Selecione a Análise",
    ["Análise Bibliométrica", "Análise de Conteúdo"]
)

if menu == "Análise Bibliométrica":
    

    st.sidebar.title("Escolha a Fase de Análise")
    secoes = ["Temporal", "Domínios", "Keywords", "Citações"]
    sub_menu = st.sidebar.radio("", secoes)

    if sub_menu == "Temporal":
        st.write("""
            A análise temporal mostra o crescimento das publicações sobre Ciência de Dados na área da Saúde ao longo dos anos.
            O gráfico abaixo ilustra o número de publicações por ano e os anos de pico de publicações.
        """)
        st.subheader('Análise Temporal')

        # Informações sobre o número de artigos e variáveis após a limpeza
        total_artigos_inicio = df_scopus.shape[0]  # Número total de artigos (linhas)
        total_variaveis_inicio = df_scopus.shape[1]  # Número total de variáveis (colunas)

        # Exibindo tabela com informações de Data Cleaning
        data_cleaning_info_inicio = {
            'Total de Artigos': [total_artigos_inicio],
            'Total de Variáveis': [total_variaveis_inicio]
        }
        df_cleaning_info_inicio = pd.DataFrame(data_cleaning_info_inicio)
        st.write("Após extração da Scopus, os resultados iniciais foram:")
        st.dataframe(df_cleaning_info_inicio)

        # Análise Temporal - Contagem de Publicações por Ano
        publications_per_year = df_scopus['Year'].value_counts().sort_index()

        # Gráfico de barras com o número de publicações por ano
        fig_bar = px.bar(publications_per_year, x=publications_per_year.index, y=publications_per_year.values, 
                         labels={'x': 'Ano', 'y': 'Número de Publicações'},
                         title="Publicações por Ano")
        fig_bar.update_traces(marker_color='#00CED1')  # Cor para as barras
        st.plotly_chart(fig_bar)

        # Gráfico de Linha - Número de Publicações ao Longo do Tempo
        df_soma_por_ano = df_scopus.groupby('Year').size().reset_index(name='Número de Publicações')
        ano_pico = df_soma_por_ano['Year'][df_soma_por_ano['Número de Publicações'].idxmax()]
        publicacoes_pico = df_soma_por_ano['Número de Publicações'].max()

        fig = px.line(df_soma_por_ano, x='Year', y='Número de Publicações', title='Número de Publicações ao Longo do Tempo')
        fig.update_traces(line=dict(color='#1f77b4'))  # Azul escuro
        fig.add_annotation(
            x=ano_pico,
            y=publicacoes_pico,
            text=f"{publicacoes_pico}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            font=dict(size=12, color="black")
        )
        st.plotly_chart(fig)

        # Gráfico de Linha - Citações por Ano
        citations_per_year = df_scopus.groupby('Year')['Cited by'].sum().reset_index(name='Total de Citações')

        # Identificar o ano com o pico de citações
        ano_pico_cit = citations_per_year['Year'][citations_per_year['Total de Citações'].idxmax()]
        citacoes_pico = citations_per_year['Total de Citações'].max()

        # Gráfico com Plotly
        fig_cit = px.line(citations_per_year, x='Year', y='Total de Citações', title='Número Total de Citações ao Longo do Tempo')

        # Personalizar a linha e adicionar anotação
        fig_cit.update_traces(line=dict(color='#FF4500', width=2))  # Cor laranja
        fig_cit.add_annotation(
            x=ano_pico_cit,
            y=citacoes_pico,
            text=f"{citacoes_pico}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            font=dict(size=12, color="black")
        )

        # Exibir o gráfico com Streamlit (caso esteja usando Streamlit para o dashboard)
        st.plotly_chart(fig_cit)

    elif sub_menu == "Domínios":
        st.write("""
            A análise de domínios ajuda a entender quais são as áreas científicas mais relevantes no contexto das publicações sobre Ciência de Dados na Saúde.
        """)
        st.subheader('Análise de Domínios')

        # Contagem de Tipos de Documentos
        document_types = df_scopus['Document Type'].value_counts()

        # Agrupar os dados pelo nome da revista (Source title) e somar as citações
        journal_citations = df_scopus.groupby('Source title')['Cited by'].sum().sort_values(ascending=False).head(10)

        # Criar o gráfico de barras para Top 10 Journals Mais Citados
        plt.figure(figsize=(14, 10))
        ax = journal_citations.plot(kind='bar', color='skyblue', title='Top 10 Journals Mais Citados')

        # Adicionar rótulos de barra
        ax.bar_label(ax.containers[0])

        # Adicionar títulos e rótulos
        plt.xlabel("")
        plt.ylabel("Número de Citações")
        plt.xticks(rotation=45, ha='right')  # Rotaciona os rótulos para uma leitura melhor
        plt.tight_layout()  # Ajusta o layout para evitar sobreposição de texto

        # Exibir o gráfico de barras com Streamlit
        st.pyplot(plt)  # Exibe o gráfico Matplotlib

        st.write("""
            Pela análise dos Journals mais citados, podemos tirar conclusões sobre os domínios de pesquisa predominantes e as áreas de maior impacto no campo da ciência de dados aplicada aos cuidados de saúde. Destacam-se áreas como **Ciências da Computação**, **Inteligência Artificial**,**Informática Biomédica**,**Saúde Digital** e **Oncologia**.
                 """)
        st.write(""" As áreas de Ciências da Computação e Inteligência Artificial, representadas por revistas como Artificial Intelligence in Medicine e Nature Machine Intelligence, desempenham um papel crucial no desenvolvimento de novas tecnologias para o setor de saúde. Além disso, outras áreas, como Oncologia Digital e Informática Biomédica, com journals como Frontiers in Oncology e Journal of Biomedical Informatics, evidenciam a crescente aplicação de técnicas computacionais na medicina, particularmente no diagnóstico e tratamento de doenças complexas. Outro ponto importante a ressaltar é a presença de áreas emergentes como Saúde Digital e **Segurança Cibernética**, destacadas por revistas como npj Digital Medicine e Proceedings of the ACM Conference on Computer and Communications Security. Essas áreas indicam uma tendência crescente de incorporar soluções digitais não apenas para melhorar os cuidados aos pacientes, mas também para proteger e gerenciar de forma segura os dados sensíveis de saúde.
                """)

        # Layout com 2 colunas para os gráficos interativos
        col1, col2 = st.columns(2)

        with col1:
            # Gráfico de Tipos de Documentos - Gráfico de Barras Interativo com Plotly
            fig_doc_types = px.bar(document_types, 
                                x=document_types.index, 
                                y=document_types.values,
                                labels={'x': 'Tipo de Documento', 'y': 'Número de Publicações'},
                                title="Distribuição dos Tipos de Documentos")
            fig_doc_types.update_traces(marker_color='#00CED1')  # Cor para as barras
            st.plotly_chart(fig_doc_types)

        with col2:
            # Gráfico de Setores (Pizza) - Proporção dos Tipos de Documentos
            fig_pie = px.pie(document_types, 
                            names=document_types.index, 
                            values=document_types.values,
                            title="Proporção dos Tipos de Documentos")
            st.plotly_chart(fig_pie)
        
    elif sub_menu == "Keywords":
        st.write("""
            A análise de keywords ajuda a identificar as palavras-chave mais frequentes nas publicações e os principais tópicos pesquisados.
        """)

        from collections import Counter

        # Limpeza das keywords
        keywords = df_scopus['Author Keywords'].dropna().str.lower().str.strip().str.split(';').sum()  # Converte para minúsculas e remove espaços
        keywords = [kw.strip() for kw in keywords if kw.strip().isalpha() or ' ' in kw.strip()]  # Remove itens que não são alfanuméricos

        # Unificação de termos
        keywords = ['machine learning' if kw in ['machine learning', 'Machine Learning', 'Machine learning', 'deep learning', 'Deep Learning'] else kw for kw in keywords]
        keywords = ['artificial intelligence' if kw in ['artificial intelligence', 'Artificial Intelligence'] else kw for kw in keywords]
        keywords = ['precision medicine' if kw in ['precision medicine', 'Precision Medicine'] else kw for kw in keywords]

        # Contagem das palavras-chave
        keywords_count = Counter(keywords)
        top_keywords = pd.DataFrame(keywords_count.most_common(10), columns=['Keyword', 'Count'])

        # Streamlit interface
        st.subheader("Análise das Keywords")
        st.write("Aqui estão as palavras-chave mais frequentes nas publicações analisadas.")

        # Gráfico interativo das top 10 keywords
        fig, ax = plt.subplots(figsize=(12, 7))
        top_keywords.set_index('Keyword').plot(kind='bar', color='#4CAF50', edgecolor='black', legend=False, ax=ax)
        ax.set_title("Top 10 Keywords", fontsize=16, fontweight='bold')
        ax.set_xlabel("Keyword", fontsize=12)
        ax.set_ylabel("Frequência", fontsize=12)
        ax.set_xticklabels(top_keywords['Keyword'], rotation=45, ha='right', fontsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Adicionar rótulos de frequência no topo de cada barra
        for index, value in enumerate(top_keywords['Count']):
            ax.text(index, value + 5, str(value), ha='center', va='bottom', fontsize=10)

        # Exibir gráfico no Streamlit
        st.pyplot(fig)

        # Criação da WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(keywords))

        # Plotando a WordCloud
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')  # Desliga os eixos
        st.pyplot(plt)  # Exibe diretamente no Streamlit

    elif sub_menu == "Citações":
        st.write("""
            As citações não apenas refletem a popularidade de um artigo, mas também o seu impacto intelectual e a contribuição significativa ao avanço do conhecimento, neste caso, na área da saúde. """)
        st.subheader('Análise de Citações')

        # Selecionando e ordenando os Top 10 Artigos Mais Citados
        top_cited_articles = df_scopus[['Title', 'Cited by', 'Authors', 'Year']].sort_values(by='Cited by', ascending=False).head(10)

        # Exibir no Streamlit a tabela
        st.write("""
        Aqui estão os 10 artigos mais citados, com as respectivas informações de citações, título, autores e ano de publicação.
        """)

        # Exibir a tabela
        st.dataframe(top_cited_articles)

elif menu == "Análise de Conteúdo":

    # Sub-menu de Análise de Conteúdo
    st.sidebar.title("Escolha a Fase de Análise")
    secoes = ["Data Cleaning e Text Processing", "TF-IDF", "LDA"]
    sub_menu = st.sidebar.radio("", secoes)

    if sub_menu == "Data Cleaning e Text Processing":
        st.subheader('Processo de Data Cleaning')
        st.write("""
            Nesta seção, apresentamos primeiramente o processo de limpeza dos dados textuais.
        """)

        # Adicionar os passos de limpeza 
        st.write("""
            As etapas de limpeza incluíram:
            - Verificação de duplicados;
            - Verificação de valores ausentes por coluna;
            - Remoção de Abstracts duplicados, mantendo apenas a primeira ocorrência.
        """)

        # Informações sobre o número de artigos e variáveis após a limpeza
        total_artigos = df_limpo.shape[0]  # Número total de artigos (linhas)
        total_variaveis = df_limpo.shape[1]  # Número total de variáveis (colunas)

        # Exibindo tabela com informações de Data Cleaning
        data_cleaning_info = {
            'Total de Artigos': [total_artigos],
            'Total de Variáveis': [total_variaveis]
        }
        df_cleaning_info = pd.DataFrame(data_cleaning_info)

        st.write("Após a limpeza, temos os seguintes resultados:")
        st.dataframe(df_cleaning_info)

        # Criar uma lista com os pontos do processo de Data Cleaning
        data_cleaning_steps = [
    {"Passo": "Remoção de URLs e Tags HTML", "Descrição": "Limpeza de links e elementos HTML."},
    {"Passo": "Remoção de Pontuação", "Descrição": "Retira os sinais de pontuação."},
    {"Passo": "Transformação para Minúsculas", "Descrição": "Converte todas as palavras para minúsculas."},
    {"Passo": "Remoção de Stop Words e Lista Customizada", "Descrição": "Elimina palavras comuns que não contribuem significativamente para a análise, incluindo uma lista personalizada de stop words."},
    {"Passo": "Tokenização", "Descrição": "Divide o texto em palavras individuais."},
    {"Passo": "Remoção de Números", "Descrição": "Retira tokens que contenham números."},
    {"Passo": "Lematização", "Descrição": "Reduz palavras a suas formas base."},
    {"Passo": "Correção de Contrações", "Descrição": "Expande contrações para sua forma completa."}
]
        # Criar o DataFrame
        df_steps = pd.DataFrame(data_cleaning_steps)

        # Exibir a tabela na dashboard
        st.subheader("Processo de Text Processing")
        st.table(df_steps)

    # Gráfico de barras para os 10 termos mais frequentes
        st.subheader("Top 10 Termos mais Frequentes")
        
        # Criar a contagem de palavras (baseado no df_limpo)
        from sklearn.feature_extraction.text import CountVectorizer
        
        # Supondo que os textos limpos estão na coluna 'text' do df_limpo
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(df_limpo['clean_Abstract'])  # Ajuste a coluna conforme necessário
        
        # Obter as palavras mais frequentes
        word_freq = X.sum(axis=0).A1
        words = vectorizer.get_feature_names_out()
        
        # Criar um DataFrame com os termos e suas frequências
        word_freq_df = pd.DataFrame(list(zip(words, word_freq)), columns=["Termo", "Frequência"])
        word_freq_df = word_freq_df.sort_values(by="Frequência", ascending=False).head(10)
        
        # Gráfico de barras
        import plotly.express as px
        fig_bar = px.bar(word_freq_df, x='Termo', y='Frequência', 
                         title="Top 10 Termos mais Frequentes")
        st.plotly_chart(fig_bar)

        # Gerar e exibir a Word Cloud
        st.subheader("Word Cloud")
        
        # Gerar a WordCloud
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        
        # Concatenar todos os textos
        text = " ".join(df_limpo['clean_Abstract'])
        
        # Gerar a WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        
        # Exibir a WordCloud
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

    elif sub_menu == "TF-IDF":
        st.subheader('Term Frequency - Inverse Document Frequency')
        st.write("""
            O TF-IDF é uma técnica que avalia a importância de palavras em um conjunto de documentos.
            Abaixo, você pode ver os termos com maior valor de TF-IDF.
        """)

        # Exibindo os 10 primeiros termos TF-IDF
        st.dataframe(df_tfidf.head(10))

        # Gráfico de barras dos 10 principais termos TF-IDF
        df_tfidf.rename(columns={'Unnamed: 0': 'Terms'}, inplace=True)

        # Gráfico de barras
        fig_tfidf = px.bar(
        df_tfidf.head(10),
        x='Terms',  # Nova coluna renomeada
        y='Average TF-IDF',
        title="Top 10 Termos mais Relevantes (TF-IDF)",
        labels={'Terms': 'Terms', 'Average TF-IDF': 'Average TF-IDF'}  # Ajustar rótulos
        )

        st.plotly_chart(fig_tfidf)

    elif sub_menu == "LDA":
        st.write("""
            O modelo LDA (Latent Dirichlet Allocation) é utilizado para descobrir os tópicos latentes num conjunto de documentos.
            Abaixo estão os tópicos identificados e os seus termos mais relevantes.
        """)

        # Criar o modelo de contagem de palavras
        count_vectorizer = CountVectorizer(stop_words='english')
        X_count = count_vectorizer.fit_transform(df_limpo['clean_Abstract'])

        # Definir o número de tópicos
        n_topics = 10  # Número de tópicos que você quer
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)

        # Treinar o modelo LDA
        lda.fit(X_count)

        # Obter os termos (palavras) mais relevantes para cada tópico
        feature_names = count_vectorizer.get_feature_names_out()
        n_words = 10  # Número de palavras a exibir por tópico

        # Criar lista de tópicos com palavras mais relevantes
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            topic_words = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
            topics.append(topic_words)

            # Criar um DataFrame com os tópicos
            df_topics = pd.DataFrame(topics, columns=[f"Word {i+1}" for i in range(n_words)])

            # Mostrar os tópicos como uma tabela
            st.dataframe(df_topics)

        # Gerar os gráficos de barras para cada tópico usando Plotly
        for topic_idx, topic in enumerate(lda.components_):
            # Obter as palavras mais relevantes para o tópico
            top_words_idx = topic.argsort()[:-n_words - 1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_words_scores = [topic[i] for i in top_words_idx]

            # Criar o gráfico de barras usando Plotly
            fig_lda = px.bar(x=top_words, y=top_words_scores,
                            labels={'x': 'Palavras', 'y': 'Relevância'},
                            title=f'Tópico {topic_idx + 1} - Termos Relevantes')

            # Exibir o gráfico na interface
            st.plotly_chart(fig_lda)


        