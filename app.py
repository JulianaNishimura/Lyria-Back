from flask import Flask, request, jsonify, session
from flask_cors import CORS
from testeDaIa import perguntar_ollama, buscar_na_web, get_persona_texto
from banco.banco import (
    pegarPersonaEscolhida, 
    escolherApersona, 
    criarUsuario, 
    procurarUsuarioPorEmail, 
    pegarHistorico,
    salvarMensagem,
    criar_banco,
    carregar_conversas,
    carregar_memorias
)
from classificadorDaWeb.classificador_busca_web import deve_buscar_na_web
from waitress import serve
import os

app = Flask(__name__)

# 🔑 Configurações de sessão e CORS
app.secret_key = os.environ.get('SECRET_KEY', 'sua_chave_secreta_aqui')
app.config['SESSION_COOKIE_SAMESITE'] = "None"   
app.config['SESSION_COOKIE_SECURE'] = True       

CORS(
    app, 
    resources={r"/Lyria/*": {"origins": [
        "http://10.110.12.27:5173",   
        "https://seufront.onrender.com"  
    ]}},
    supports_credentials=True
)

try:
    criar_banco()
    print("✅ Tabelas criadas/verificadas com sucesso!")
except Exception as e:
    print(f"❌ Erro ao criar tabelas: {e}")

# ------------------ ROTAS ------------------

# Login
@app.route('/Lyria/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or 'email' not in data:
        return jsonify({"erro": "Campo 'email' é obrigatório"}), 400
    
    email = data['email']
    senha_hash = data.get('senha_hash')
    
    try:
        usuario = procurarUsuarioPorEmail(email)
        if not usuario:
            return jsonify({"erro": "Usuário não encontrado"}), 404
        
        if usuario.get('senha_hash') and senha_hash != usuario['senha_hash']:
            return jsonify({"erro": "Senha incorreta"}), 401
        
        session['usuario_email'] = usuario['email']
        session['usuario_nome'] = usuario['nome'] 
        session['usuario_id'] = usuario['id']
        
        return jsonify({
            "status": "ok",
            "mensagem": "Login realizado com sucesso",
            "usuario": usuario['nome'],
            "persona": usuario.get('persona_escolhida')
        })
        
    except Exception as e:
        return jsonify({"status": "erro", "mensagem": f"Erro interno: {str(e)}"}), 500

# Logout
@app.route('/Lyria/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"status": "ok", "mensagem": "Logout realizado com sucesso"}), 200

# Função auxiliar de login
def verificar_login():
    return session.get('usuario_email')

# Conversar sem login
@app.route('/Lyria/conversar', methods=['POST'])
def conversar_sem_conta():
    data = request.get_json()
    if not data or 'pergunta' not in data or 'persona' not in data:
        return jsonify({"erro": "Campos 'pergunta' e 'persona' são obrigatórios"}), 400
    
    pergunta = data['pergunta']
    persona = data['persona']
    
    try:
        contexto_web = buscar_na_web(pergunta) if deve_buscar_na_web(pergunta) else None
        resposta = perguntar_ollama(pergunta, None, None, persona, contexto_web)
        return jsonify({"resposta": resposta})
        
    except Exception as e:
        return jsonify({"erro": f"Erro interno: {str(e)}"}), 500

# Conversar logado
@app.route('/Lyria/conversar-logado', methods=['POST'])
def conversar_logado():
    usuario = verificar_login()
    if not usuario:
        return jsonify({"erro": "Usuário não está logado"}), 401
    
    data = request.get_json()
    if not data or 'pergunta' not in data:
        return jsonify({"erro": "Campo 'pergunta' é obrigatório"}), 400
    
    pergunta = data['pergunta']
    persona_tipo = pegarPersonaEscolhida(usuario)
    if not persona_tipo:
        return jsonify({"erro": "Usuário não tem persona definida"}), 400
    
    try:
        conversas = carregar_conversas(usuario)
        memorias = carregar_memorias(usuario)
        contexto_web = buscar_na_web(pergunta) if deve_buscar_na_web(pergunta) else None
        persona = get_persona_texto(persona_tipo)
        resposta = perguntar_ollama(pergunta, conversas, memorias, persona, contexto_web)
        salvarMensagem(usuario, pergunta, resposta, modelo_usado="hf", tokens=None)
        return jsonify({"resposta": resposta})
        
    except Exception as e:
        return jsonify({"erro": f"Erro interno: {str(e)}"}), 500

# Conversas logado
@app.route('/Lyria/conversas', methods=['GET'])
def get_conversas_logado():
    usuario = verificar_login()
    if not usuario:
        return jsonify({"erro": "Usuário não está logado"}), 401
    
    try:
        conversas = carregar_conversas(usuario)
        return jsonify({"conversas": conversas or []})
    except Exception as e:
        return jsonify({"erro": f"Erro ao buscar conversas: {str(e)}"}), 500

# Persona do usuário
@app.route('/Lyria/PersonaEscolhida', methods=['GET'])
def get_persona_escolhida_logado():
    usuario = verificar_login()
    if not usuario:
        return jsonify({"erro": "Usuário não está logado"}), 401
    
    try:
        persona = pegarPersonaEscolhida(usuario)
        if persona:
            return jsonify({"persona": persona})
        return jsonify({"erro": "Usuário não encontrado"}), 404
    except Exception as e:
        return jsonify({"erro": f"Erro ao buscar persona: {str(e)}"}), 500

@app.route('/Lyria/PersonaEscolhida', methods=['PUT'])
def atualizar_persona_escolhida_logado():
    usuario = verificar_login()
    if not usuario:
        return jsonify({"erro": "Usuário não está logado"}), 401
    
    data = request.get_json()
    if not data or 'persona' not in data:
        return jsonify({"erro": "Campo 'persona' é obrigatório"}), 400

    persona = data['persona']
    if persona not in ['professor', 'empresarial', 'social']:
        return jsonify({"erro": "Persona inválida. Use 'professor', 'empresarial' ou 'social'"}), 400

    try:
        escolherApersona(persona, usuario) 
        return jsonify({"sucesso": "Persona atualizada com sucesso"}), 200
    except Exception as e:
        return jsonify({"erro": f"Erro ao atualizar persona: {str(e)}"}), 500

# Criar usuário
@app.route('/Lyria/usuarios', methods=['POST'])
def criar_usuario_route():
    data = request.get_json()
    if not data or 'nome' not in data or 'email' not in data:
        return jsonify({"erro": "Campos 'nome' e 'email' são obrigatórios"}), 400

    nome = data['nome']
    email = data['email']
    persona = data.get('persona')
    senha_hash = data.get('senha_hash')
    
    if persona not in ['professor', 'empresarial', 'social']:
        return jsonify({"erro": "Persona inválida. Use 'professor', 'empresarial' ou 'social'"}), 400

    try:
        usuario_id = criarUsuario(nome, email, persona, senha_hash)
        return jsonify({
            "sucesso": "Usuário criado com sucesso", 
            "id": usuario_id,
            "persona": persona
        }), 201
    except Exception as e:
        if "UNIQUE constraint" in str(e):
            return jsonify({"erro": "Usuário já existe"}), 409
        return jsonify({"erro": f"Erro ao criar usuário: {str(e)}"}), 500

# Buscar usuário por email
@app.route('/Lyria/usuarios/<usuarioEmail>', methods=['GET'])
def get_usuario(usuarioEmail):
    try:
        result = procurarUsuarioPorEmail(usuarioEmail)
        if result:
            return jsonify({"usuario": result})
        return jsonify({"erro": "Usuário não encontrado"}), 404
    except Exception as e:
        return jsonify({"erro": f"Erro ao buscar usuário: {str(e)}"}), 500

# Histórico logado
@app.route('/Lyria/historico', methods=['GET'])
def get_historico_recente_logado():
    usuario = verificar_login()
    if not usuario:
        return jsonify({"erro": "Usuário não está logado"}), 401
    
    try:
        limite = request.args.get('limite', 10, type=int)
        if limite > 50: 
            limite = 50
            
        historico = pegarHistorico(usuario, limite)
        return jsonify({"historico": historico})
    except Exception as e:
        return jsonify({"erro": f"Erro ao buscar histórico: {str(e)}"}), 500

# Listar personas
@app.route('/Lyria/personas', methods=['GET'])
def listar_personas():
    personas = { ... } 
    return jsonify({"personas": personas})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  
    serve(app, host="0.0.0.0", port=port)
