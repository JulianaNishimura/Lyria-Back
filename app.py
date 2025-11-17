import os
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from waitress import serve
from flask_session import Session
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from testeDaIa import perguntar_ollama, buscar_na_web, get_persona_texto
import secrets
from datetime import datetime, timedelta
from banco.banco import (
    criar_banco, criarUsuario, procurarUsuarioPorEmail,
    pegarHistorico, salvarMensagem, carregar_conversas, carregar_memorias,
    pegarPersonaEscolhida, escolherApersona, deleta_conversa, criar_nova_conversa,
    salvar_token_redefinicao, procurarUsuarioPorToken, atualizar_senha,
    carregar_mensagens_por_conversa_id
)
from classificadorDaWeb.classificador_busca_web import deve_buscar_na_web

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY')

IS_PRODUCTION = os.environ.get('RENDER', False)

app.config.update(
    SESSION_TYPE='filesystem',  
    SESSION_COOKIE_NAME='lyria_session',
    SESSION_COOKIE_SAMESITE='None' if IS_PRODUCTION else 'Lax',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=IS_PRODUCTION,
    SESSION_COOKIE_PATH='/',
    SESSION_COOKIE_DOMAIN=None,  
    PERMANENT_SESSION_LIFETIME=604800
)

Session(app)

allowed_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://10.110.12.20:5173"
]

if IS_PRODUCTION:
    allowed_origins.append("https://lyriafront.onrender.com")

CORS(app, 
    resources={r"/*": {
        "origins": allowed_origins,
        "allow_headers": ["Content-Type", "Authorization"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "supports_credentials": True,
        "expose_headers": ["Set-Cookie"]
    }}
)

try:
    criar_banco()
    print("✅ Tabelas criadas/verificadas com sucesso!")
except Exception as e:
    print(f"❌ Erro ao criar tabelas: {e}")

def verificar_login():
    email = session.get('usuario_email')
    if email:
        print(f"✅ Usuário autenticado: {email}")
    else:
        print("❌ Nenhum usuário autenticado na sessão")
    return email

def validar_persona(persona):
    return persona in ['professor', 'empresarial', 'social']

@app.route('/Lyria/login', methods=['POST'])
def login():
    print(f"   Origin: {request.headers.get('Origin')}")
    print(f"   Cookies recebidos: {dict(request.cookies)}")
    
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

        session.clear()  
        session.permanent = True 
        session['usuario_email'] = usuario['email']
        session['usuario_nome'] = usuario['nome']
        session['usuario_id'] = usuario['id']
                
        session.modified = True
        
        print(f"✅ Sessão criada:")
        print(f"   Email: {session.get('usuario_email')}")
        print(f"   Nome: {session.get('usuario_nome')}")
        print(f"   ID: {session.get('usuario_id')}")
        print(f"{'='*60}\n")

        response = jsonify({
            "status": "ok",
            "mensagem": "Login realizado com sucesso",
            "usuario": usuario['nome'],
            "persona": usuario.get('persona_escolhida'),
            "conversa_id": None  # Sem conversa inicial
        })
        
        return response

    except Exception as e:
        print(f"❌ Erro no login: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({"status": "erro", "mensagem": str(e)}), 500

@app.route('/Lyria/conversas', methods=['POST'])
def criar_nova_conversa_route():
    usuario = verificar_login()
    if not usuario:
        return jsonify({"erro": "Usuário não está logado"}), 401
    
    try:
        conversa_id = criar_nova_conversa(usuario)
        session['conversa_id'] = conversa_id
        session.modified = True
        
        print(f"✅ Nova conversa criada: {conversa_id}")
        return jsonify({"conversa_id": conversa_id, "sucesso": "Nova conversa criada"}), 201
    except Exception as e:
        print(f"❌ Erro ao criar nova conversa: {e}")
        return jsonify({"erro": str(e)}), 500

@app.route('/Lyria/logout', methods=['POST'])
def logout():
    email = session.get('usuario_email')
    session.clear()
    print(f"✅ Logout realizado: {email}")
    return jsonify({"status": "ok", "mensagem": "Logout realizado com sucesso"}), 200

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
        return jsonify({"erro": "Usuário não está logado"}), 401

    data = request.get_json() or {}
    pergunta = data.get('pergunta')
    conversa_id = data.get('conversa_id')

    if not pergunta or not conversa_id:
        return jsonify({"erro": "Campos 'pergunta' e 'conversa_id' são obrigatórios"}), 400

    try:
        print(f"📌 Usando conversa_id recebido do frontend: {conversa_id}")

        persona_tipo = pegarPersonaEscolhida(usuario)
        if not persona_tipo:
            return jsonify({"erro": "Usuário não tem persona definida"}), 400
        
        historico_conversa = carregar_mensagens_por_conversa_id(conversa_id)
        print(f"🧠 Histórico da conversa ({conversa_id}) carregado para a IA: {historico_conversa}")
        memorias = carregar_memorias(usuario)
        contexto_web = buscar_na_web(pergunta) if deve_buscar_na_web(pergunta) else None
        persona_texto = get_persona_texto(persona_tipo)

        resposta = perguntar_ollama(pergunta, historico_conversa, memorias, persona_texto, contexto_web)
        
        conversa_id_retornado = salvarMensagem(usuario, pergunta, resposta, modelo_usado="hf", tokens=None, conversa_id=conversa_id)

        return jsonify({"resposta": resposta, "conversa_id": conversa_id_retornado})
    except Exception as e:
        print(f"❌ Erro detalhado em conversar_logado: {str(e)}")
        import traceback
        print(f"❌ Traceback completo:\n{traceback.format_exc()}")
        return jsonify({"erro": str(e)}), 500
    
@app.route('/Lyria/conversas', methods=['GET'])
def get_conversas_logado():
    usuario = verificar_login()
    if not usuario:
        print("❌ Tentativa de acesso não autorizado em /conversas")
        return jsonify({"erro": "Usuário não está logado"}), 401

    try:
        conversas = carregar_conversas(usuario)
        conversa_ativa = session.get('conversa_id')
        return jsonify({
            "conversas": conversas or [],
            "conversa_ativa": conversa_ativa
        })
    except Exception as e:
        print(f"❌ Erro em get_conversas_logado: {e}")
        return jsonify({"erro": str(e)}), 500
    
@app.route('/Lyria/conversas/<id>', methods=['DELETE'])
def remove_conversa_id(id):
    usuario = verificar_login()
    if not usuario:
        return jsonify({"erro": "Usuário não está logado"}), 401
    
    try:
        deletou = deleta_conversa(id)
        
        if str(session.get('conversa_id')) == str(id):
            session.pop('conversa_id', None)
            session.modified = True
            print(f"🗑️ Conversa ativa {id} removida da sessão")
        
        return jsonify({"sucesso": "Deletado com sucesso!"})
    except Exception as e:
        print(f"Erro em deletar a conversa")
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

@app.route('/Lyria/PersonaEscolhida', methods=['GET'])
def get_persona_logado():
    usuario = verificar_login()
    if not usuario:
        print("❌ Tentativa de acesso não autorizado em /PersonaEscolhida GET")
        return jsonify({"erro": "Usuário não está logado"}), 401

    try:
        persona = pegarPersonaEscolhida(usuario)
        if persona:
            return jsonify({"persona_escolhida": persona})
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

def send_password_reset_email(user_email, token):
    """Envia um e-mail de redefinição de senha usando SendGrid."""
    frontend_url = os.environ.get('FRONTEND_URL', 'http://localhost:5173')
    sender_email = os.environ.get('SENDER_EMAIL')
    sendgrid_api_key = os.environ.get('SENDGRID_API_KEY')

    if not sender_email or not sendgrid_api_key:
        print("❌ Variáveis de ambiente SENDER_EMAIL ou SENDGRID_API_KEY não configuradas.")
        # Retornar um erro genérico para não expor detalhes de implementação
        raise Exception("O serviço de e-mail não está configurado.")

    reset_link = f"{frontend_url}/reset-password?token={token}"

    message = Mail(
        from_email=sender_email,
        to_emails=user_email,
        subject='[LyrIA] Redefinição de Senha',
        html_content=f"""
            <p>Olá,</p>
            <p>Você solicitou a redefinição da sua senha. Clique no link abaixo para criar uma nova senha:</p>
            <p><a href="{reset_link}">Redefinir Senha</a></p>
            <p>Se você não solicitou isso, por favor, ignore este e-mail.</p>
            <p>O link expirará em 1 hora.</p>
            <p>Atenciosamente,<br>Equipe LyrIA</p>
        """
    )
    try:
        sg = SendGridAPIClient(sendgrid_api_key)
        response = sg.send(message)
        print(f"✅ E-mail de redefinição enviado para {user_email}. Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Erro ao enviar e-mail via SendGrid: {e}")
        raise e

@app.route('/Lyria/esqueci-minha-senha', methods=['POST'])
def esqueci_minha_senha():
    data = request.get_json() or {}
    email = data.get('email')

    if not email:
        return jsonify({"erro": "Campo 'email' é obrigatório"}), 400

    try:
        usuario = procurarUsuarioPorEmail(email)
        if not usuario:
            # Still return a success message to avoid user enumeration
            return jsonify({"status": "ok", "mensagem": "Se um usuário com este e-mail existir, um link de redefinição de senha será enviado."}), 200

        token = secrets.token_urlsafe(32)
        expiracao = datetime.utcnow() + timedelta(hours=1)
        salvar_token_redefinicao(email, token, expiracao)

        # Envia o e-mail de redefinição de senha
        send_password_reset_email(email, token)

        return jsonify({"status": "ok", "mensagem": "Se um usuário com este e-mail existir, um link de redefinição de senha será enviado."}), 200

    except Exception as e:
        print(f"❌ Erro em esqueci_minha_senha: {e}")
        return jsonify({"status": "erro", "mensagem": str(e)}), 500

@app.route('/Lyria/redefinir-senha', methods=['POST'])
def redefinir_senha():
    data = request.get_json() or {}
    token = data.get('token')
    nova_senha = data.get('nova_senha')

    if not token or not nova_senha:
        return jsonify({"erro": "Campos 'token' e 'nova_senha' são obrigatórios"}), 400

    try:
        usuario = procurarUsuarioPorToken(token)
        if not usuario:
            return jsonify({"erro": "Token inválido ou expirado"}), 400

        expiracao = usuario.get('token_redefinicao_expiracao')
        if not expiracao or datetime.utcnow() > expiracao:
            return jsonify({"erro": "Token inválido ou expirado"}), 400

        atualizar_senha(token, nova_senha)

        return jsonify({"status": "ok", "mensagem": "Senha redefinida com sucesso"}), 200

    except Exception as e:
        print(f"❌ Erro em redefinir_senha: {e}")
        return jsonify({"status": "erro", "mensagem": str(e)}), 500

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
            "conversa_id": session.get('conversa_id'),
            "session_id": request.cookies.get('lyria_session', 'Não encontrado')
        })
    return jsonify({
        "autenticado": False,
        "cookies_recebidos": list(request.cookies.keys()),
        "mensagem": "Nenhuma sessão ativa"
    }), 401


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Servidor iniciando na porta {port}")
    serve(app, host="0.0.0.0", port=port)