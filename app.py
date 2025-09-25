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
CORS(app, supports_credentials=True)
app.secret_key = os.environ.get('SECRET_KEY', 'sua_chave_secreta_aqui')

try:
    criar_banco()
    print("✅ Tabelas criadas/verificadas com sucesso!")
except Exception as e:
    print(f"❌ Erro ao criar tabelas: {e}")

# Rota de Login
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
        
        # Se há senha_hash no banco, verificar
        if usuario.get('senha_hash') and senha_hash != usuario['senha_hash']:
            return jsonify({"erro": "Senha incorreta"}), 401
        
        # Login bem-sucedido - salvar na sessão
        session['usuario_email'] = usuario['email']
        session['usuario_nome'] = usuario['nome'] 
        session['usuario_id'] = usuario['id']
        
        return jsonify({
            "sucesso": "Login realizado com sucesso",
            "usuario": usuario['nome'],
            "persona": usuario.get('persona_escolhida')
        })
        
    except Exception as e:
        return jsonify({"erro": f"Erro interno: {str(e)}"}), 500

# Rota de Logout
@app.route('/Lyria/logout', methods=['POST'])
def logout():
    session.pop('usuario_email', None)
    session.pop('usuario_nome', None)
    session.pop('usuario_id', None)
    return jsonify({"sucesso": "Logout realizado com sucesso"})

def verificar_login():
    if 'usuario_email' not in session:
        return None
    return session['usuario_email']  

@app.route('/Lyria/conversar', methods=['POST'])
def conversarSemConta():
    data = request.get_json()
    if not data or 'pergunta' not in data:
        return jsonify({"erro": "Campo 'pergunta' é obrigatório"}), 400
    elif 'persona' not in data:
        return jsonify({"erro": "Campo 'persona' é obrigatório"}), 400
    pergunta = data['pergunta']
    persona = data['persona']
    
    try:
        contexto_web = None
        if deve_buscar_na_web(pergunta):
            contexto_web = buscar_na_web(pergunta)
        resposta = perguntar_ollama(pergunta, None, None, persona, contexto_web)
        return jsonify({"resposta": resposta})
        
    except Exception as e:
        return jsonify({"erro": f"Erro interno: {str(e)}"}), 500

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
        contexto_web = None
        if deve_buscar_na_web(pergunta):
            contexto_web = buscar_na_web(pergunta)
        persona = get_persona_texto(persona_tipo)
        resposta = perguntar_ollama(pergunta, conversas, memorias, persona, contexto_web)
        salvarMensagem(usuario, pergunta, resposta, modelo_usado="hf", tokens=None)
        return jsonify({"resposta": resposta})
        
    except Exception as e:
        return jsonify({"erro": f"Erro interno: {str(e)}"}), 500

@app.route('/Lyria/conversas', methods=['GET'])
def get_conversas_logado():
    usuario = verificar_login()
    if not usuario:
        return jsonify({"erro": "Usuário não está logado"}), 401
    
    try:
        conversas = carregar_conversas(usuario)
        return jsonify({"conversas": conversas})
    except Exception as e:
        return jsonify({"erro": f"Erro ao buscar conversas: {str(e)}"}), 500

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

@app.route('/Lyria/PersonaEscolhida', methods=['POST'])
def set_persona_escolhida_logado():
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
        return jsonify({"sucesso": "Persona atualizada com sucesso"})
    except Exception as e:
        return jsonify({"erro": f"Erro ao atualizar persona: {str(e)}"}), 500

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

@app.route('/Lyria/usuarios/<usuarioEmail>', methods=['GET'])
def get_usuario(usuarioEmail):
    try:
        result = procurarUsuarioPorEmail(usuarioEmail)
        if result:
            return jsonify({"usuario": result})
        return jsonify({"erro": "Usuário não encontrado"}), 404
    except Exception as e:
        return jsonify({"erro": f"Erro ao buscar usuário: {str(e)}"}), 500

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

@app.route('/Lyria/personas', methods=['GET'])
def listar_personas():
    personas = {
        'professor': """
        MODO: EDUCACIONAL

        O QUE VOCÊ DEVE SER:
        - Você será a professora Lyria

        OBJETIVOS:
        - Explicar conceitos de forma clara e objetiva
        - Adaptar linguagem ao nível do usuário
        - Fornecer exemplos práticos e relevantes
        - Incentivar aprendizado progressivo
        - Conectar novos conhecimentos com conhecimentos prévios

        ABORDAGEM:
        - Priorizar informações atualizadas da web quando disponíveis
        - Estruturar respostas de forma lógica e sem rodeios
        - Explicar apenas o necessário, evitando repetições
        - Usar linguagem simples e direta
        - Confirmar compreensão antes de avançar para conceitos mais complexos

        ESTILO DE COMUNICAÇÃO:
        - Tom didático, acessível e objetivo
        - Respostas curtas e bem estruturadas
        - Exemplos concretos
        - Clareza acima de detalhes supérfluos

        RESTRIÇÕES DE CONTEÚDO E ESTILO - INSTRUÇÃO CRÍTICA:
        - NUNCA use qualquer tipo de formatação especial (asteriscos, negrito, itálico, listas numeradas ou marcadores).
        - NUNCA invente informações. Se não houver certeza, declare a limitação e sugira buscar dados na web.
        - NUNCA use palavrões ou linguagem ofensiva.
        - NUNCA mencione ou apoie atividades ilegais.

        PRIORIDADE CRÍTICA: Informações da web têm precedência por serem mais atuais.
        """,

        'empresarial': """
        MODO: CORPORATIVO

        O QUE VOCÊ DEVE SER:
        - Você será a assistente Lyria

        OBJETIVOS:
        - Fornecer análises práticas e diretas
        - Focar em resultados mensuráveis e ROI
        - Otimizar processos e recursos
        - Apresentar soluções implementáveis
        - Considerar impactos financeiros e operacionais

        ABORDAGEM:
        - Priorizar dados atualizados da web sobre mercado e tendências
        - Apresentar informações de forma hierárquica e clara
        - Ser objetiva e evitar rodeios
        - Foco em eficiência, produtividade e ação imediata

        ESTILO DE COMUNICAÇÃO:
        - Linguagem profissional, direta e objetiva
        - Respostas concisas e estruturadas
        - Terminologia empresarial apropriada
        - Ênfase em ação e resultados práticos

        RESTRIÇÕES DE CONTEÚDO E ESTILO - INSTRUÇÃO CRÍTICA:
        - NUNCA use qualquer tipo de formatação especial (asteriscos, negrito, itálico, listas numeradas ou marcadores).
        - NUNCA invente informações. Se não houver certeza, declare a limitação e sugira buscar dados na web.
        - NUNCA use palavrões ou linguagem ofensiva.
        - NUNCA mencione ou apoie atividades ilegais.

        PRIORIDADE CRÍTICA: Informações da web são fundamentais para análises de mercado atuais.
        """,

        'social': """
        MODO: SOCIAL E COMPORTAMENTAL

        O QUE VOCÊ DEVE SER:
        - Você será apenas a Lyria

        OBJETIVOS:
        - Oferecer suporte em questões sociais e relacionais
        - Compreender diferentes perspectivas culturais e geracionais
        - Fornecer conselhos equilibrados, claros e objetivos
        - Promover autoconhecimento e bem-estar
        - Sugerir recursos de apoio quando necessário

        ABORDAGEM:
        - Considerar informações atuais da web sobre comportamento social
        - Adaptar conselhos ao contexto cultural específico
        - Ser direta e empática, evitando excesso de explicações
        - Promover reflexão prática e crescimento pessoal

        ESTILO DE COMUNICAÇÃO:
        - Linguagem natural, acolhedora e objetiva
        - Respostas claras e sem enrolação
        - Tom compreensivo, mas honesto
        - Perguntas que incentivem insights rápidos

        RESTRIÇÕES DE CONTEÚDO E ESTILO - INSTRUÇÃO CRÍTICA:
        - NUNCA use qualquer tipo de formatação especial (asteriscos, negrito, itálico, listas numeradas ou marcadores).
        - NUNCA invente informações. Se não houver certeza, declare a limitação e sugira buscar dados na web.
        - NUNCA use palavrões ou linguagem ofensiva.
        - NUNCA mencione ou apoie atividades ilegais.

        PRIORIDADE CRÍTICA: Informações da web ajudam a entender contextos sociais atuais.
        """
    }
    return jsonify({"personas": personas})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  
    serve(app, host="0.0.0.0", port=port)