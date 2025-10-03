import os
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from waitress import serve

from testeDaIa import perguntar_ollama, buscar_na_web, get_persona_texto
from banco.banco import (
    criar_banco, criarUsuario, procurarUsuarioPorEmail,
    pegarHistorico, salvarMensagem, carregar_conversas, carregar_memorias,
    pegarPersonaEscolhida, escolherApersona
)
from classificadorDaWeb.classificador_busca_web import deve_buscar_na_web

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'chave_default_secreta_mude_em_producao')

IS_PRODUCTION = os.environ.get('RENDER', False)

# Sessão
app.config.update(
    SESSION_COOKIE_SAMESITE="None",  # necessário para cross-domain
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=IS_PRODUCTION,  # True no Render (HTTPS), False local
    PERMANENT_SESSION_LIFETIME=604800  # 7 dias
)

# CORS
allowed_origins = [
    "http://localhost:5173",
    "http://localhost:3000"
]

if IS_PRODUCTION:
    allowed_origins.append("https://lyriafront.onrender.com")  # produção

CORS(app, 
    resources={r"/Lyria/*": {
        "origins": allowed_origins,
        "allow_headers": ["Content-Type", "Authorization"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "supports_credentials": True
    }}
)

# Inicializa banco
try:
    criar_banco()
    print("✅ Tabelas criadas/verificadas com sucesso!")
except Exception as e:
    print(f"❌ Erro ao criar tabelas: {e}")

# ---------------- FUNÇÕES AUXILIARES ----------------
def verificar_login():
    """Retorna o email do usuário logado ou None."""
    email = session.get('usuario_email')
    if email:
        print(f"✅ Usuário autenticado: {email}")
    else:
        print("❌ Nenhum usuário autenticado na sessão")
    return email

def validar_persona(persona):
    return persona in ['professor', 'empresarial', 'social']

# ---------------- ROTAS ----------------

# --- Autenticação ---
@app.route('/Lyria/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    email = data.get('email')
    senha_hash = data.get('senha_hash')

    if not email:
        return jsonify({"erro": "Campo 'email' é obrigatório"}), 400

    try:
        usuario = procurarUsuarioPorEmail(email)
        if not usuario:
            return jsonify({"erro": "Usuário não encontrado"}), 404

        if usuario.get('senha_hash') and senha_hash != usuario['senha_hash']:
            return jsonify({"erro": "Senha incorreta"}), 401

        # Configura a sessão
        session.clear()  # Limpa qualquer sessão anterior
        session.permanent = True  # Mantém a sessão ativa
        session.update({
            'usuario_email': usuario['email'],
            'usuario_nome': usuario['nome'],
            'usuario_id': usuario['id']
        })
        
        print(f"✅ Login bem-sucedido: {email}")
        print(f"📦 Sessão criada: {dict(session)}")

        return jsonify({
            "status": "ok",
            "mensagem": "Login realizado com sucesso",
            "usuario": usuario['nome'],
            "persona": usuario.get('persona_escolhida')
        })

    except Exception as e:
        print(f"❌ Erro no login: {e}")
        return jsonify({"status": "erro", "mensagem": str(e)}), 500


@app.route('/Lyria/logout', methods=['POST'])
def logout():
    email = session.get('usuario_email')
    session.clear()
    print(f"✅ Logout realizado: {email}")
    return jsonify({"status": "ok", "mensagem": "Logout realizado com sucesso"}), 200

# --- Conversa ---
@app.route('/Lyria/conversar', methods=['POST'])
def conversar_sem_conta():
    data = request.get_json() or {}
    pergunta = data.get('pergunta')
    persona = data.get('persona')

    if not pergunta or not persona:
        return jsonify({"erro": "Campos 'pergunta' e 'persona' são obrigatórios"}), 400

    try:
        contexto_web = buscar_na_web(pergunta) if deve_buscar_na_web(pergunta) else None
        resposta = perguntar_ollama(pergunta, None, None, persona, contexto_web)
        return jsonify({"resposta": resposta})
    except Exception as e:
        print(f"❌ Erro em conversar_sem_conta: {e}")
        return jsonify({"erro": str(e)}), 500


@app.route('/Lyria/conversar-logado', methods=['POST'])
def conversar_logado():
    usuario = verificar_login()
    if not usuario:
        print("❌ Tentativa de acesso não autorizado em /conversar-logado")
        return jsonify({"erro": "Usuário não está logado"}), 401

    data = request.get_json() or {}
    pergunta = data.get('pergunta')
    if not pergunta:
        return jsonify({"erro": "Campo 'pergunta' é obrigatório"}), 400

    try:
        persona_tipo = pegarPersonaEscolhida(usuario)
        if not persona_tipo:
            return jsonify({"erro": "Usuário não tem persona definida"}), 400

        conversas = carregar_conversas(usuario)
        memorias = carregar_memorias(usuario)
        contexto_web = buscar_na_web(pergunta) if deve_buscar_na_web(pergunta) else None
        persona_texto = get_persona_texto(persona_tipo)

        resposta = perguntar_ollama(pergunta, conversas, memorias, persona_texto, contexto_web)
        salvarMensagem(usuario, pergunta, resposta, modelo_usado="hf", tokens=None)

        return jsonify({"resposta": resposta})
    except Exception as e:
        print(f"❌ Erro em conversar_logado: {e}")
        return jsonify({"erro": str(e)}), 500

# --- Histórico e conversas ---
@app.route('/Lyria/conversas', methods=['GET'])
def get_conversas_logado():
    usuario = verificar_login()
    if not usuario:
        print("❌ Tentativa de acesso não autorizado em /conversas")
        return jsonify({"erro": "Usuário não está logado"}), 401

    try:
        conversas = carregar_conversas(usuario)
        return jsonify({"conversas": conversas or []})
    except Exception as e:
        print(f"❌ Erro em get_conversas_logado: {e}")
        return jsonify({"erro": str(e)}), 500


@app.route('/Lyria/historico', methods=['GET'])
def get_historico_logado():
    usuario = verificar_login()
    if not usuario:
        return jsonify({"erro": "Usuário não está logado"}), 401

    limite = min(request.args.get('limite', 10, type=int), 50)
    try:
        historico = pegarHistorico(usuario, limite)
        return jsonify({"historico": historico})
    except Exception as e:
        print(f"❌ Erro em get_historico_logado: {e}")
        return jsonify({"erro": str(e)}), 500

# --- Persona ---
@app.route('/Lyria/PersonaEscolhida', methods=['GET'])
def get_persona_logado():
    usuario = verificar_login()
    if not usuario:
        print("❌ Tentativa de acesso não autorizado em /PersonaEscolhida GET")
        return jsonify({"erro": "Usuário não está logado"}), 401

    try:
        persona = pegarPersonaEscolhida(usuario)
        if persona:
            return jsonify({"persona": persona})
        return jsonify({"erro": "Usuário não encontrado"}), 404
    except Exception as e:
        print(f"❌ Erro em get_persona_logado: {e}")
        return jsonify({"erro": str(e)}), 500


@app.route('/Lyria/PersonaEscolhida', methods=['PUT'])
def atualizar_persona_logado():
    usuario = verificar_login()
    if not usuario:
        print("❌ Tentativa de acesso não autorizado em /PersonaEscolhida PUT")
        return jsonify({"erro": "Usuário não está logado"}), 401

    data = request.get_json() or {}
    persona = data.get('persona')
    if not validar_persona(persona):
        return jsonify({"erro": "Persona inválida. Use 'professor', 'empresarial' ou 'social'"}), 400

    try:
        escolherApersona(persona, usuario)
        print(f"✅ Persona atualizada para {persona} - usuário: {usuario}")
        return jsonify({"sucesso": "Persona atualizada com sucesso"})
    except Exception as e:
        print(f"❌ Erro em atualizar_persona_logado: {e}")
        return jsonify({"erro": str(e)}), 500

# --- Usuários ---
@app.route('/Lyria/usuarios', methods=['POST'])
def criar_usuario_route():
    data = request.get_json() or {}
    nome = data.get('nome')
    email = data.get('email')
    persona = data.get('persona')
    senha_hash = data.get('senha_hash')

    if not nome or not email:
        return jsonify({"erro": "Campos 'nome' e 'email' são obrigatórios"}), 400
    if persona and not validar_persona(persona):
        return jsonify({"erro": "Persona inválida. Use 'professor', 'empresarial' ou 'social'"}), 400

    try:
        usuario_id = criarUsuario(nome, email, persona, senha_hash)
        print(f"✅ Usuário criado: {email} com persona {persona}")
        return jsonify({"sucesso": "Usuário criado com sucesso", "id": usuario_id, "persona": persona}), 201
    except Exception as e:
        if "UNIQUE constraint" in str(e):
            return jsonify({"erro": "Usuário já existe"}), 409
        print(f"❌ Erro em criar_usuario_route: {e}")
        return jsonify({"erro": str(e)}), 500


@app.route('/Lyria/usuarios/<usuarioEmail>', methods=['GET'])
def get_usuario(usuarioEmail):
    try:
        usuario = procurarUsuarioPorEmail(usuarioEmail)
        if usuario:
            return jsonify({"usuario": usuario})
        return jsonify({"erro": "Usuário não encontrado"}), 404
    except Exception as e:
        print(f"❌ Erro em get_usuario: {e}")
        return jsonify({"erro": str(e)}), 500

# --- Personas disponíveis ---
@app.route('/Lyria/personas', methods=['GET'])
def listar_personas():
    try:
        personas = {
            "professor": "Persona professor",
            "empresarial": "Persona empresarial",
            "social": "Persona social"
        }
        return jsonify({"personas": personas}), 200
    except Exception as e:
        print(f"❌ Erro em /Lyria/personas: {e}")
        return jsonify({"erro": str(e)}), 500

# --- Rota de verificação de sessão (útil para debugging) ---
@app.route('/Lyria/check-session', methods=['GET'])
def check_session():
    print(f"📦 Headers recebidos: {dict(request.headers)}")
    print(f"🍪 Cookies recebidos: {request.cookies}")
    print(f"📋 Sessão atual: {dict(session)}")
    
    usuario = verificar_login()
    if usuario:
        return jsonify({
            "autenticado": True,
            "usuario": session.get('usuario_nome'),
            "email": usuario,
            "session_id": request.cookies.get('lyria_session', 'Não encontrado')
        })
    return jsonify({
        "autenticado": False,
        "cookies_recebidos": list(request.cookies.keys()),
        "mensagem": "Nenhuma sessão ativa"
    }), 401


# ---------------- INÍCIO DO SERVIDOR ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Servidor iniciando na porta {port}")
    serve(app, host="0.0.0.0", port=port)