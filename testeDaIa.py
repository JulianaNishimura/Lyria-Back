import requests
import sqlite3
import os
from classificadorDaWeb.classificador_busca_web import deve_buscar_na_web
from banco.banco import (
    carregar_conversas, 
    salvarMensagem, 
    pegarPersonaEscolhida,
    escolherApersona,
    criarUsuario,
    criar_banco
)

LIMITE_HISTORICO = 12
SERPAPI_KEY = os.getenv("KEY_SERP_API")
OLLAMA_HOST = 'https://testeollama.onrender.com'
OLLAMA_MODEL = "gemma3n:latest"

def carregar_memorias(usuario):
    from banco.banco import carregar_memorias as carregar_memorias_db
    return carregar_memorias_db(usuario)

def perguntar_ollama(pergunta, conversas, memorias, persona, contexto_web=None): 
    LIMITE_HISTORICO_REDUZIDO = 6
    
    prompt_parts = [persona]
    
    if conversas:
        prompt_parts.append("Histórico recente:")
        for msg in conversas[-LIMITE_HISTORICO_REDUZIDO:]:
            prompt_parts.append(f"Usuário: {msg['pergunta']}")
            prompt_parts.append(f"Lyria: {msg['resposta']}")
    
    if contexto_web:
        contexto_limitado = contexto_web[:500] 
        prompt_parts.append(f"Info web atual: {contexto_limitado}")
    
    prompt_parts.append(f"Usuário: {pergunta}")
    prompt_parts.append("Lyria:")
    
    prompt = "\n".join(prompt_parts)
    
    payload = {
        'model': OLLAMA_MODEL,
        'prompt': prompt,
        'stream': False,
        'options': {
            'temperature': 0.7, 
            'top_p': 0.9, 
            'num_predict': 200, 
            'stop': ['\n\nUsuário:', 'Usuário:'] 
        }
    }
    
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=60, 
            headers={'Content-Type': 'application/json'}
        )
        
        response.raise_for_status()
        data = response.json()
        
        resposta = data.get('response', '').strip()
        
        if not resposta:
            return "Desculpe, não consegui gerar uma resposta adequada."
        
        if resposta.startswith('Lyria:'):
            resposta = resposta[6:].strip()
            
        return resposta
        
    except requests.exceptions.Timeout:
        print("Timeout - Ollama demorou mais que 45s")
        return "Desculpe, estou processando mais devagar hoje. Pode repetir a pergunta?"
        
    except requests.exceptions.ConnectionError:
        print("Erro de conexão - Ollama pode estar offline")
        return "Não consegui me conectar ao sistema. Verifique se o Ollama está rodando."
        
    except requests.exceptions.HTTPError as e:
        print(f"Erro HTTP {e.response.status_code}: {e.response.text}")
        return "Houve um problema no processamento. Tente novamente."
        
    except Exception as e:
        print(f"Erro inesperado: {e}")
        return "Erro interno. Tente novamente em alguns instantes."

def verificar_ollama_status():
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            modelos = response.json().get('models', [])
            modelo_existe = any(m.get('name', '').startswith(OLLAMA_MODEL.split(':')[0]) for m in modelos)
            return {
                'status': 'online',
                'modelo_existe': modelo_existe,
                'modelos_disponiveis': [m.get('name') for m in modelos]
            }
        else:
            return {'status': 'erro', 'detalhes': f'HTTP {response.status_code}'}
            
    except requests.exceptions.ConnectionError:
        return {'status': 'offline', 'detalhes': 'Não foi possível conectar'}
    except Exception as e:
        return {'status': 'erro', 'detalhes': str(e)}

def buscar_na_web(pergunta):
    try:
        params = {"q": pergunta, "hl": "pt-br", "gl": "br", "api_key": SERPAPI_KEY}
        res = requests.get("https://serpapi.com/search", params=params, timeout=10)
        res.raise_for_status()
        
        resultados = res.json().get("organic_results", [])
        trechos = [r.get("snippet", "") for r in resultados[:2] if r.get("snippet")]
        return " ".join(trechos) if trechos else None
        
    except Exception as e:
        print(f"Erro na busca web: {e}")
        return None

def get_persona_texto(persona_tipo):
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
    
    return personas.get(persona_tipo, personas['professor'])

if __name__ == "__main__":
    criar_banco()
    
    print("Do que você precisa?")
    print("1. Professor")
    print("2. Empresarial")
    escolha = input("Escolha: ").strip()

    if escolha == '1':
        persona_tipo = 'professor'
    elif escolha == '2':
        persona_tipo = 'empresarial'
    else:
        print("Opção inválida")
        exit()

    usuario = input("Informe seu nome: ").strip().lower()
    
    try:
        criarUsuario(usuario, f"{usuario}@local.com", persona_tipo)
    except:
        escolherApersona(persona_tipo, usuario)
    
    persona = get_persona_texto(persona_tipo)
    
    print("\nModo texto ativo (digite 'sair' para encerrar)")
    while True:
        entrada = input("Você: ").strip()
        if entrada.lower() == 'sair':
            break
            
        contexto_web = None
        if deve_buscar_na_web(entrada):
            contexto_web = buscar_na_web(entrada)
            
        resposta = perguntar_ollama(
            entrada, 
            carregar_conversas(usuario), 
            carregar_memorias(usuario), 
            persona,
            contexto_web
        )
        
        print(f"Lyria: {resposta}")
        salvarMensagem(usuario, entrada, resposta, modelo_usado="ollama", tokens=None)