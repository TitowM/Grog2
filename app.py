import streamlit as st
import pandas as pd
import plotly.express as px
from pypdf import PdfReader
from openai import AzureOpenAI

# Configuração da Página
st.set_page_config(
    page_title="Suporte de TI - Chatbot RAG & Analytics",
    page_icon="🛠️",
    layout="wide"
)

# Inicialização do estado de sessão (State)
if "chat_iniciado" not in st.session_state:
    st.session_state.chat_iniciado = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conteudo_base" not in st.session_state:
    st.session_state.conteudo_base = ""

if "df_dados" not in st.session_state:
    st.session_state.df_dados = None


def extrair_texto_e_dados(uploaded_file):
    """Extrai texto e tenta carregar DataFrame se for CSV ou texto estruturado."""
    nome = uploaded_file.name.lower()
    df_result = None
    texto_result = ""
    try:
        if nome.endswith(".txt"):
            texto_result = uploaded_file.read().decode("utf-8", errors="ignore")
            uploaded_file.seek(0)
            try:
                df_result = pd.read_csv(uploaded_file, sep=None, engine="python")
            except Exception:
                pass
        elif nome.endswith(".csv"):
            df_result = pd.read_csv(uploaded_file, sep=None, engine="python")
            texto_result = df_result.to_string(index=False)
        elif nome.endswith(".pdf"):
            leitor = PdfReader(uploaded_file)
            for pagina in leitor.pages:
                t_pag = pagina.extract_text()
                if t_pag:
                    texto_result += t_pag + "\n"
    except Exception as e:
        st.sidebar.error(f"Erro ao processar o arquivo: {e}")
    
    return texto_result, df_result


# ==========================================
# BARRA LATERAL: Configurações e Conexão
# ==========================================
with st.sidebar:
    st.title("⚙️ Configurações da IA")
    st.caption("Microsoft Foundry / Azure OpenAI Service")

    endpoint = st.text_input(
        "Endpoint Azure:",
        value=st.session_state.get("saved_endpoint", ""),
        placeholder="https://seu-recurso.openai.azure.com/",
    )
    api_key = st.text_input(
        "API Key:",
        value=st.session_state.get("saved_key", ""),
        placeholder="Cole sua API Key",
        type="password"
    )
    modelo = st.text_input(
        "Nome do Deployment / Modelo:",
        value=st.session_state.get("saved_model", "gpt-4o-mini"),
        placeholder="ex: gpt-4o-mini"
    )

    st.divider()

    persona_padrao = (
        "Você é um Especialista de Suporte Técnico de TI nível 2/3. "
        "Seu tom é profissional, analítico, objetivo e didático. "
        "Utilize a base de conhecimento anexada para resolver os chamados e incidentes "
        "com passos estruturados de troubleshooting."
    )
    persona = st.text_area(
        "Definição da Persona (Prompt do Sistema):",
        value=persona_padrao,
        height=120
    )

    st.divider()
    arquivo_rag = st.file_uploader(
        "Base de Conhecimento (RAG):",
        type=["pdf", "txt", "csv"],
        help="Envie documentações, CSVs de problemas/valores, etc."
    )

    if arquivo_rag is not None:
        texto_extraido, df_extraido = extrair_texto_e_dados(arquivo_rag)
        st.session_state.conteudo_base = texto_extraido
        st.session_state.df_dados = df_extraido
        st.success(f"Base carregada: {arquivo_rag.name}")
    else:
        st.session_state.conteudo_base = ""
        st.session_state.df_dados = None

    if st.button("🚀 Iniciar / Atualizar Chat", use_container_width=True, type="primary"):
        if not endpoint or not api_key or not modelo:
            st.error("Preencha Endpoint, API Key e Nome do Modelo.")
        else:
            st.session_state.saved_endpoint = endpoint
            st.session_state.saved_key = api_key
            st.session_state.saved_model = modelo
            st.session_state.persona = persona
            st.session_state.chat_iniciado = True
            st.rerun()

    if st.session_state.chat_iniciado:
        if st.button("🔄 Novo Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ==========================================
# PAINEL PRINCIPAL: Dashboard e WhatsApp Chat
# ==========================================
st.subheader("🛠️ Central de Atendimento & Analytics de TI")

col_dashboard, col_chat = st.columns([1.2, 0.8], gap="medium")

with col_dashboard:
    st.markdown("### 📊 Visão Geral dos Dados")
    if st.session_state.df_dados is not None and not st.session_state.df_dados.empty:
        df = st.session_state.df_dados
        
        total_linhas = len(df)
        col_kpi1, col_kpi2 = st.columns(2)
        col_kpi1.metric("Total de Registros / Serviços", total_linhas)
        
        coluna_valor = None
        for col in df.columns:
            if any(term in col.lower() for term in ["valor", "preço", "custo", "r$"]):
                coluna_valor = col
                break
        
        if coluna_valor:
            try:
                valores_numericos = df[coluna_valor].astype(str).str.extract(r'(\d+[\d\.]*)')[0].astype(float)
                media_val = valores_numericos.mean()
                col_kpi2.metric("Média Estimada (Ref)", f"R$ {media_val:.2f}")
            except Exception:
                col_kpi2.metric("Col. Valores Detectada", coluna_valor)
        else:
            col_kpi2.metric("Colunas Identificadas", len(df.columns))

        st.markdown("---")
        st.markdown("#### Gráficos Dinâmicos")
        
        if len(df.columns) >= 2:
            try:
                fig = px.bar(
                    df.head(10), 
                    x=df.columns[0], 
                    y=coluna_valor if coluna_valor else df.columns[1],
                    title="Top Ocorrências / Valores por Serviço",
                    labels={df.columns[0]: "Item", coluna_valor or df.columns[1]: "Métrica"}
                )
                fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=320)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.bar_chart(df.select_dtypes(include="number").head(10))
        
        with st.expander("🔍 Visualizar Tabela Completa de Dados"):
            st.dataframe(df, use_container_width=True)
    else:
        st.info("💡 Envie um arquivo estruturado (CSV) na barra lateral para gerar os cartões de resumo e gráficos automáticos.")

with col_chat:
    st.markdown("### 💬 Chat de Suporte")
    
    # CSS refinado imitando perfeitamente o visual e paleta de cores do WhatsApp Dark Mode
    st.markdown("""
    <style>
    .wa-chat-window {
        background-color: #0b141a;
        border-radius: 12px;
        padding: 16px;
        height: 480px;
        overflow-y: auto;
        border: 1px solid #222d34;
        display: flex;
        flex-direction: column;
        gap: 8px;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.4);
    }
    .wa-msg-bubble-user {
        background-color: #005c4b;
        color: #e9edef;
        padding: 8px 12px;
        border-radius: 8px 0px 8px 8px;
        max-width: 80%;
        align-self: flex-end;
        margin-left: auto;
        font-size: 13.5px;
        line-height: 1.4;
        word-break: break-word;
        box-shadow: 0 1px 0.5px rgba(0,0,0,0.2);
    }
    .wa-msg-bubble-bot {
        background-color: #202c33;
        color: #e9edef;
        padding: 8px 12px;
        border-radius: 0px 8px 8px 8px;
        max-width: 80%;
        align-self: flex-start;
        margin-right: auto;
        font-size: 13.5px;
        line-height: 1.4;
        word-break: break-word;
        box-shadow: 0 1px 0.5px rgba(0,0,0,0.2);
    }
    .wa-author-user {
        font-size: 10px;
        color: #53bdeb;
        font-weight: 600;
        text-align: right;
        margin-bottom: 2px;
        display: block;
    }
    .wa-author-bot {
        font-size: 10px;
        color: #00a884;
        font-weight: 600;
        text-align: left;
        margin-bottom: 2px;
        display: block;
    }
    </style>
    """, unsafe_allow_html=True)

    if not st.session_state.chat_iniciado:
        st.warning("Inicie a sessão na barra lateral para conversar com o assistente.")
    else:
        # Janela container de mensagens com rolagem limpa
        chat_container = st.container(height=480)
        
        with chat_container:
            st.markdown('<div class="wa-chat-window">', unsafe_allow_html=True)
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="wa-msg-bubble-user">'
                        f'<span class="wa-author-user">Você</span>'
                        f'{msg["content"]}'
                        f'</div>', 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="wa-msg-bubble-bot">'
                        f'<span class="wa-author-bot">🛠️ Suporte TI</span>'
                        f'{msg["content"]}'
                        f'</div>', 
                        unsafe_allow_html=True
                    )
            st.markdown('</div>', unsafe_allow_html=True)

        # Entrada de prompt na parte inferior idêntica ao campo de digitação
        if prompt_usuario := st.chat_input("Digite sua dúvida de TI..."):
            st.session_state.messages.append({"role": "user", "content": prompt_usuario})
            
            system_instruction = st.session_state.persona
            if st.session_state.conteudo_base:
                system_instruction += (
                    f"\n\n--- DOCUMENTAÇÃO DE SUPORTE (RAG) ---\n"
                    f"{st.session_state.conteudo_base}\n"
                    f"--- FIM DA DOCUMENTAÇÃO ---"
                )

            mensagens_api = [{"role": "system", "content": system_instruction}]
            for m in st.session_state.messages:
                mensagens_api.append({"role": m["role"], "content": m["content"]})

            try:
                client = AzureOpenAI(
                    azure_endpoint=st.session_state.saved_endpoint.strip(),
                    api_key=st.session_state.saved_key.strip(),
                    api_version="2024-06-01"
                )

                response = client.chat.completions.create(
                    model=st.session_state.saved_model.strip(),
                    messages=mensagens_api,
                    temperature=0.2
                )
                resposta_ia = response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": resposta_ia})
                st.rerun()

            except Exception as erro:
                st.error(f"Falha na comunicação com a Azure: {erro}")
