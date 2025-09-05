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

import json
import time

def chamar_groq_api(prompt, max_tokens=300):
    """Fallback usando Groq API (gratuita e rápida)"""
    if not GROQ_API_KEY:
        return None
        
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "model": "llama3-8b-8192",  # Modelo gratuito e rápido
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    try:
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data['choices'][0]['message']['content']
    except Exception as e:
        print(f"Erro Groq API: {e}")
        return None

def chamar_hf_inference(prompt, max_new_tokens=200, temperature=0.7, max_tentativas=2):
    if HUGGING_FACE_API_KEY is None or HUGGING_FACE_API_KEY.strip() == "":
        print("Chave HF não encontrada, tentando Groq...")
        return chamar_groq_api(prompt) or "Erro: nenhuma API disponível"
    
    headers = {
        "Authorization": f"Bearer {HUGGING_FACE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Prompt ainda mais limitado
    prompt_limitado = prompt[:600] if len(prompt) > 600 else prompt
    
    payload = {
        "inputs": prompt_limitado,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "return_full_text": False
        },
        "options": {
            "wait_for_model": True,
            "use_cache": False
        }
    }
    
    for tentativa in range(max_tentativas):
        try:
            timeout = 8 + (tentativa * 5)  # 8s, 13s
            print(f"HF API tentativa {tentativa + 1}/{max_tentativas} (timeout: {timeout}s)")
            
            resp = requests.post(HF_MODEL_ENDPOINT, headers=headers, json=payload, timeout=timeout)
            
            if resp.status_code == 503:
                print("Modelo HF carregando, tentando Groq...")
                fallback = chamar_groq_api(prompt)
                if fallback:
                    return fallback
                time.sleep(5)
                continue
            
            if resp.status_code == 429:
                print("Rate limit HF, tentando Groq...")
                fallback = chamar_groq_api(prompt)
                if fallback:
                    return fallback
                time.sleep(10)
                continue
            
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, list) and len(data) > 0 and 'generated_text' in data[0]:
                return data[0]['generated_text'].strip()
            
            if isinstance(data, dict) and 'generated_text' in data:
                return data['generated_text'].strip()
            
            return "Resposta inesperada da HF API"
            
        except requests.exceptions.Timeout:
            print(f"Timeout HF API (tentativa {tentativa + 1})")
            if tentativa == 0:  # Na primeira tentativa de timeout, tenta Groq
                print("Tentando Groq como fallback...")
                fallback = chamar_groq_api(prompt)
                if fallback:
                    return fallback
            continue
        except requests.exceptions.RequestException as e:
            print(f"Erro HF API: {e}")
            # Tenta Groq em caso de erro
            fallback = chamar_groq_api(prompt)
            if fallback:
                return fallback
            continue
    
    # Se todas as tentativas falharam, tenta Groq uma última vez
    print("Todas tentativas HF falharam, tentando Groq...")
    fallback = chamar_groq_api(prompt)
    if fallback:
        return fallback
    
    # Se tudo falhou, resposta de emergência
    return gerar_resposta_offline(prompt)

def gerar_resposta_offline(prompt):
    """Resposta de emergência quando todas as APIs falham"""
    pergunta = prompt.split("Usuário:")[-1].split("Lyria:")[0].strip() if "Usuário:" in prompt else prompt
    
    respostas_genericas = {
        "como": "Para fazer isso, você pode seguir alguns passos básicos. Preciso de mais detalhes para te ajudar melhor.",
        "o que": "Essa é uma pergunta interessante. Posso explicar de forma simples se você me der mais contexto.",
        "por que": "Existem algumas razões principais para isso. Você gostaria que eu explique alguma específica?",
        "onde": "A localização ou lugar específico depende do contexto. Pode me dar mais informações?",
        "quando": "O timing varia dependendo da situação. Precisa de informações sobre um período específico?",
        "quem": "Isso envolve pessoas ou organizações específicas. Quer saber sobre alguém em particular?",
    }
    
    pergunta_lower = pergunta.lower()
    for palavra, resposta in respostas_genericas.items():
        if palavra in pergunta_lower:
            return resposta
    
    return "Desculpe, estou com dificuldades técnicas no momento. Pode reformular sua pergunta ou tentar novamente em alguns minutos?"

LIMITE_HISTORICO = 12
SERPAPI_KEY = os.getenv("KEY_SERP_API")
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Nova variável para Groq

# Modelo mais leve e confiável
HF_MODEL_ENDPOINT = "https://api-inference.huggingface.co/models/distilgpt2"

def carregar_memorias(usuario):
    from banco.banco import carregar_memorias as carregar_memorias_db
    return carregar_memorias_db(usuario)

def perguntar_ollama(pergunta, conversas, memorias, persona, contexto_web=None):
    # Prompt super otimizado
    prompt_parts = []
    
    # Persona ultra-condensada
    if 'professor' in persona.lower():
        prompt_parts.append("Você é Lyria, professora. Responda de forma didática e clara.")
    elif 'empresarial' in persona.lower():
        prompt_parts.append("Você é Lyria, assistente corporativa. Responda de forma profissional e objetiva.")
    elif 'social' in persona.lower():
        prompt_parts.append("Você é Lyria. Responda de forma empática e compreensiva.")
    else:
        prompt_parts.append("Você é Lyria, assistente inteligente. Responda de forma útil.")
    
    # Apenas 1 conversa anterior se existir
    if conversas and len(conversas) > 0:
        ultima = conversas[-1]
        prompt_parts.append(f"\nAnterior - U: {str(ultima.get('pergunta', ''))[:50]}")
        prompt_parts.append(f" L: {str(ultima.get('resposta', ''))[:50]}")
    
    # Contexto web muito limitado
    if contexto_web:
        prompt_parts.append(f"\nInfo: {str(contexto_web)[:100]}")
    
    # Pergunta atual
    prompt_parts.append(f"\nUsuário: {str(pergunta)}")
    prompt_parts.append("\nLyria:")
    
    prompt_final = "".join(prompt_parts)
    print(f"Prompt: {len(prompt_final)} chars")
    
    resposta = chamar_hf_inference(prompt_final)
    return resposta

def verificar_ollama_status():
    status = "Usando HF Inference API"
    if GROQ_API_KEY:
        status += " com fallback Groq"
    if not HUGGING_FACE_API_KEY and not GROQ_API_KEY:
        status = "Nenhuma API configurada - modo offline"
    return {'status': 'info', 'detalhes': status}

def buscar_na_web(pergunta):
    try:
        params = {"q": pergunta, "hl": "pt-br", "gl": "br", "api_key": SERPAPI_KEY}
        res = requests.get("https://serpapi.com/search", params=params, timeout=10)
        res.raise_for_status()
        
        resultados = res.json().get("organic_results", [])
        trechos = [r.get("snippet", "") for r in resultados[:2] if r.get("snippet")]
        return " ".join(trechos) if trechos else None
        
    except Exception as e:
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

    print(f"\n{verificar_ollama_status()['detalhes']}")
    print("Modo texto ativo (digite 'sair' para encerrar)")
    
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
        salvarMensagem(usuario, entrada, resposta, modelo_usado="hf", tokens=None)