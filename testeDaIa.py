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

def chamar_hf_inference(prompt, max_new_tokens=512, temperature=0.3, top_p=0.95):
    if HUGGING_FACE_API_KEY is None or HUGGING_FACE_API_KEY.strip() == "":
        raise RuntimeError("Variavel de ambiente HUGGING_FACE_API_KEY nao encontrada.")
    headers = {
        "Authorization": "Bearer " + HUGGING_FACE_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_full_text": False
        },
        "options": {
            "wait_for_model": True
        }
    }
    try:
        resp = requests.post(HF_MODEL_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=60)
        if resp.status_code == 503:
            # Modelo aquecendo; tentar novamente rapidamente
            resp = requests.post(HF_MODEL_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # Formatos possiveis: lista com dicts contendo 'generated_text' ou dict com 'generated_text'
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and 'generated_text' in data[0]:
            return data[0]['generated_text']
        if isinstance(data, dict) and 'generated_text' in data:
            return data['generated_text']
        # Alguns endpoints retornam {'choices':[{'text': ...}]}
        if isinstance(data, dict) and 'choices' in data and isinstance(data['choices'], list) and len(data['choices']) > 0:
            choice = data['choices'][0]
            if isinstance(choice, dict) and 'text' in choice:
                return choice['text']
        return str(data)
    except requests.RequestException as e:
        return "Ocorreu um erro ao chamar a Inference API: " + str(e)

LIMITE_HISTORICO = 12
SERPAPI_KEY = os.getenv("KEY_SERP_API")
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
HF_MODEL_ENDPOINT = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

def carregar_memorias(usuario):
    from banco.banco import carregar_memorias as carregar_memorias_db
    return carregar_memorias_db(usuario)



def perguntar_ollama(pergunta, conversas, memorias, persona, contexto_web=None):
    LIMITE_HISTORICO_REDUZIDO = 6
    prompt_parts = []
    prompt_parts.append(persona)
    if conversas:
        prompt_parts.append("Historico recente:")
        for msg in conversas[-LIMITE_HISTORICO_REDUZIDO:]:
            prompt_parts.append("Usuario: " + str(msg.get('pergunta', '')))
            prompt_parts.append("Lyria: " + str(msg.get('resposta', '')))
    if contexto_web:
        contexto_limitado = str(contexto_web)[:500]
        prompt_parts.append("Info web atual: " + contexto_limitado)
    prompt_parts.append("Pergunta atual: " + str(pergunta))
    prompt = "".join(prompt_parts)
    resposta = chamar_hf_inference(prompt)
    return resposta

def verificar_ollama_status():
    return {'status': 'info', 'detalhes': 'Usando a API de Inferência do Hugging Face.'}

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
        salvarMensagem(usuario, entrada, resposta, modelo_usado="hf", tokens=None)