import streamlit as st
import openai
import pandas as pd
import io

# Layout wide para permitir controle fino das colunas
st.set_page_config(page_title="Chatbot Didático: Groq + RAG + Gráficos", page_icon="🤖", layout="wide")

# CSS para alinhar o st.chat_input e estilizar o chat com visual estilo WhatsApp
st.markdown("""
<style>
    /* Trava o chat_input na largura da coluna do meio e centraliza */
    [data-testid="stChatInput"] {
        max-width: 60% !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    
    /* Estilo WhatsApp para as caixas de chat */
    [data-testid="stChatMessage"] {
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Laboratório de IA: Chatbot, RAG, Gráficos e Personas")

# Inicialização do histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================================================================
# PROPORÇÃO DAS COLUNAS: LATERAIS (0.8) E CHAT EXPANDIDO (2.4)
# ==============================================================================
col_controles, col_chat, col_preview = st.columns([0.8, 2.4, 0.8], gap="medium")

# ==============================================================================
# 1. COLUNA DA ESQUERDA: API, PERSONA, UPLOAD E CONTROLES
# ==============================================================================
with col_controles:
    st.subheader("⚙️ Controles")

    # Bloco API
    st.markdown("##### 🔑 API & Modelo")
    user_api_key = st.text_input(
        "API Key (Groq):", 
        type="password", 
        placeholder="gsk_...",
        help="Insira a chave obtida em console.groq.com"
    )

    BASE_URL = "https://api.groq.com/openai/v1"
    modelo_selecionado = None
    client = None

    if user_api_key:
        try:
            client = OpenAI(base_url=BASE_URL, api_key=user_api_key.strip())
            
            modelos_disponiveis = [
                m.id for m in client.models.list().data 
                if not any(termo in m.id.lower() for termo in ["whisper", "guard", "audio"])
            ]
            
            if modelos_disponiveis:
                padrao_idx = 0
                for idx, nome in enumerate(sorted(modelos_disponiveis)):
                    if "gpt-oss-20b" in nome:
                        padrao_idx = idx
                        break
                        
                modelo_selecionado = st.selectbox(
                    "Modelo Ativo:", 
                    sorted(modelos_disponiveis),
                    index=padrao_idx
                )
                st.success("Conectado!", icon="✅")
            else:
                st.error("Nenhum modelo de chat liberado.")
        except Exception as e:
            st.error(f"Erro: {e}")
            client = None
    else:
        st.info("Cole sua chave acima.")

    st.markdown("---")

    # Bloco Persona
    st.markdown("##### 🎭 Persona da IA")
    persona_padrao = "Você é um analista de dados especialista em interpretar gráficos, tabelas CSV e responder dúvidas com precisão."
    persona_customizada = st.text_area(
        "Instrução da IA:",
        value=persona_padrao,
        height=100,
        help="Orienta o comportamento do modelo durante as respostas."
    )

    st.markdown("---")

    # Bloco Upload
    st.markdown("##### 📁 Documento / Dataset (RAG)")
    arquivo_enviado = st.file_uploader(
        "Upload (.txt, .csv, .md):",
        type=["txt", "csv", "md"]
    )
    
    conteudo_documento = ""
    df_csv = None
    
    if arquivo_enviado:
        try:
            nome_arquivo = arquivo_enviado.name.lower()
            if nome_arquivo.endswith(".csv"):
                # Lê o CSV usando pandas
                string_io = io.StringIO(arquivo_enviado.getvalue().decode("utf-8", errors="replace"))
                df_csv = pd.read_csv(string_io)
                conteudo_documento = df_csv.to_string(index=False)
            else:
                conteudo_documento = arquivo_enviado.read().decode("utf-8", errors="replace")
            
            st.caption(f"Status: OK ({len(conteudo_documento)} chars)")
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {e}")

    st.markdown("---")

    # Botões de Controle
    if st.button("🧹 Limpar Histórico", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==============================================================================
# 2. COLUNA DO CENTRO: CHAT (ESTILO MENSAGEIRO)
# ==============================================================================
with col_chat:
    st.subheader("💬 Diálogo com o Assistente")
    
    chat_container = st.container(height=580)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

# ==============================================================================
# 3. COLUNA DA DIREITA: PRÉ-VISUALIZAÇÃO E GRÁFICOS DO CSV
# ==============================================================================
with col_preview:
    st.subheader("📄 Visualização & Gráficos")
    if arquivo_enviado:
        st.caption(f"Arquivo: **{arquivo_enviado.name}**")
        
        if df_csv is not None:
            # Mostra tabela resumida e gráfico automático
            with st.expander("Ver Tabela Completa", expanded=False):
                st.dataframe(df_csv, use_container_width=True, height=200)
            
            st.markdown("##### 📈 Gráfico Gerado")
            colunas_numericas = df_csv.select_dtypes(include=["number"]).columns.tolist()
            colunas_texto = df_csv.select_dtypes(include=["object", "category"]).columns.tolist()
            
            if colunas_numericas:
                try:
                    if colunas_texto:
                        # Define a primeira coluna de texto como índice para o eixo X do gráfico
                        chart_data = df_csv.set_index(colunas_texto[0])[colunas_numericas]
                        st.line_chart(chart_data)
                    else:
                        st.line_chart(df_csv[colunas_numericas])
                except Exception as e:
                    st.warning(f"Não foi possível plotar o gráfico automaticamente: {e}")
            else:
                st.info("O CSV não possui colunas numéricas para gerar gráficos.")
        else:
            st.text_area(
                label="Conteúdo em memória:",
                value=conteudo_documento,
                height=580,
                disabled=True
            )
    else:
        st.info("Nenhum arquivo carregado. Anexe um arquivo .csv ou .txt no painel esquerdo.")

# ==============================================================================
# 4. CAMPO DE ENTRADA DO CHAT
# ==============================================================================
if user_input := st.chat_input("Perquise sobre os dados do gráfico ou converse com o bot..."):
    if not client or not modelo_selecionado:
        st.warning("⚠️ Insira uma chave de API válida no painel de controle antes de enviar perguntas.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with chat_container:
        with st.chat_message("user"):
            st.write(user_input)

    # Constrói o System Prompt (Persona + Contexto RAG do CSV/Documento)
    mensagens_api = []
    prompt_sistema = f"### IDENTIDADE E REGRAS DA PERSONA:\n{persona_customizada.strip()}\n\n"

    if conteudo_documento:
        prompt_sistema += (
            "### REGRAS ESPECÍFICAS DE CONSULTA (RAG/DADOS):\n"
            "1. Responda à dúvida do usuário usando EXCLUSIVAMENTE as informações contidas nos dados/documento abaixo.\n"
            "2. Seja direto, claro e ajude a extrair insights dos dados numéricos ou textuais apresentados.\n"
            "3. Se a informação solicitada não constar nos dados, responda estritamente: "
            "'A informação solicitada não consta no documento/dataset fornecido.'\n\n"
            f"--- INÍCIO DOS DADOS ---\n{conteudo_documento}\n--- FIM DOS DADOS ---"
        )
    else:
        prompt_sistema += "Responda de forma clara, objetiva e coerente com sua persona."

    mensagens_api.append({"role": "system", "content": prompt_sistema})

    for m in st.session_state.messages:
        mensagens_api.append({"role": m["role"], "content": m["content"]})

    # Chamada à API
    try:
        with chat_container:
            with st.spinner("Analisando dados..."):
                response = client.chat.completions.create(
                    model=modelo_selecionado,
                    messages=mensagens_api,
                    temperature=0.2 if conteudo_documento else 0.7,
                    max_tokens=400
                )
                resposta = response.choices[0].message.content

            st.session_state.messages.append({"role": "assistant", "content": resposta})
            with st.chat_message("assistant"):
                st.write(resposta)

    except Exception as e:
        st.error(f"Erro na geração da resposta: {e}")
