import sqlite3
from datetime import datetime
import os

DB_NOME = os.path.join(os.path.dirname(__file__), "lyria.db")

def criar_banco():
    conn = sqlite3.connect(DB_NOME, timeout=10, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS usuarios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nome TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        senha_hash TEXT,
        persona_escolhida TEXT,
        criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        ultimo_acesso TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        usuario_id INTEGER NOT NULL,
        mensagens TEXT,
        iniciado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        atualizado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT,
        FOREIGN KEY(usuario_id) REFERENCES usuarios(id)
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        usuario_id INTEGER NOT NULL,
        conversa_id INTEGER NOT NULL,
        conteudo TEXT NOT NULL,
        criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(usuario_id) REFERENCES usuarios(id),
        FOREIGN KEY(conversa_id) REFERENCES conversas(id)
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ai_responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id INTEGER NOT NULL,
        conteudo TEXT NOT NULL,
        criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        modelo_usado TEXT,
        tokens INTEGER,
        FOREIGN KEY(request_id) REFERENCES user_requests(id)
    );    
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS mensagens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversa_id INTEGER NOT NULL,
        request_id INTEGER NOT NULL,
        response_id INTEGER NOT NULL,
        criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(conversa_id) REFERENCES conversas(id),
        FOREIGN KEY(request_id) REFERENCES user_requests(id),
        FOREIGN KEY(response_id) REFERENCES ai_responses(id)
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memorias (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        usuario_id INTEGER NOT NULL,
        chave TEXT NOT NULL,
        valor TEXT NOT NULL,
        tipo TEXT,
        relevancia INTEGER DEFAULT 0,
        conversa_origem INTEGER,
        criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        atualizado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expira_em TIMESTAMP,
        FOREIGN KEY(usuario_id) REFERENCES usuarios(id)
    );
    """)
    conn.commit()
    conn.close()

def carregar_memorias(usuario, limite=20):
    conn = sqlite3.connect(DB_NOME, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ur.conteudo AS usuario_disse,
               ar.conteudo AS ia_respondeu,
               m.criado_em AS quando
        FROM mensagens m
        JOIN user_requests ur ON m.request_id = ur.id
        JOIN ai_responses ar ON m.response_id = ar.id
        JOIN conversas c ON m.conversa_id = c.id
        JOIN usuarios u ON c.usuario_id = u.id
        WHERE u.nome = ?
        ORDER BY m.criado_em DESC
        LIMIT ?
    """, (usuario, limite))
    results = cursor.fetchall()
    conn.close()
    memorias = []
    for row in results:
        memorias.append(f"Usuário: {row['usuario_disse']}")
        memorias.append(f"IA: {row['ia_respondeu']}")
    return list(reversed(memorias))

def pegarPersonaEscolhida(usuario):
    conn = sqlite3.connect(DB_NOME, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT persona_escolhida FROM usuarios WHERE nome = ?", (usuario,))
    result = cursor.fetchone()
    conn.close()
    return result["persona_escolhida"] if result else None

def escolherApersona(persona, usuario):
    conn = sqlite3.connect(DB_NOME, timeout=10, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("UPDATE usuarios SET persona_escolhida = ? WHERE nome = ?", (persona, usuario))
    conn.commit()
    conn.close()

def criarUsuario(nome, email, persona, senha_hash=None):
    conn = sqlite3.connect(DB_NOME, timeout=10, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO usuarios (nome, email, persona_escolhida, senha_hash, criado_em, ultimo_acesso)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (nome, email, persona, senha_hash, datetime.now(), datetime.now()))
    conn.commit()
    usuario_id = cursor.lastrowid
    conn.close()
    return usuario_id

def procurarUsuarioPorEmail(usuarioEmail):
    conn = sqlite3.connect(DB_NOME, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM usuarios WHERE email = ?", (usuarioEmail,))
    result = cursor.fetchone()
    conn.close()
    return dict(result) if result else None

def pegarHistorico(usuario, limite=3):
    conn = sqlite3.connect(DB_NOME, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ur.conteudo AS pergunta,
               ar.conteudo AS resposta,
               m.criado_em AS timestamp
        FROM mensagens m
        JOIN user_requests ur ON m.request_id = ur.id
        JOIN ai_responses ar ON m.response_id = ar.id
        JOIN conversas c ON m.conversa_id = c.id
        JOIN usuarios u ON c.usuario_id = u.id
        WHERE u.nome = ?
        ORDER BY m.criado_em DESC
        LIMIT ?
    """, (usuario, limite))
    results = cursor.fetchall()
    conn.close()
    return [dict(row) for row in results]

def carregar_conversas(usuario, limite=12):
    conn = sqlite3.connect(DB_NOME, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ur.conteudo AS pergunta, ar.conteudo AS resposta
        FROM mensagens m
        JOIN user_requests ur ON m.request_id = ur.id
        JOIN ai_responses ar ON m.response_id = ar.id
        JOIN conversas c ON m.conversa_id = c.id
        JOIN usuarios u ON c.usuario_id = u.id
        WHERE u.nome = ?
        ORDER BY m.criado_em ASC
        LIMIT ?
    """, (usuario, limite))
    results = cursor.fetchall()
    conn.close()
    return [{"pergunta": row["pergunta"], "resposta": row["resposta"]} for row in results]

def salvarMensagem(usuario, pergunta, resposta, modelo_usado=None, tokens=None):
    conn = sqlite3.connect(DB_NOME, timeout=10, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id FROM conversas 
        WHERE usuario_id = (SELECT id FROM usuarios WHERE nome=?)
        ORDER BY iniciado_em DESC LIMIT 1
    """, (usuario,))
    conversa = cursor.fetchone()
    if conversa:
        conversa_id = conversa[0]
    else:
        cursor.execute("""
            INSERT INTO conversas (usuario_id) VALUES ((SELECT id FROM usuarios WHERE nome=?))
        """, (usuario,))
        conversa_id = cursor.lastrowid
    cursor.execute("""
        INSERT INTO user_requests (usuario_id, conversa_id, conteudo)
        VALUES ((SELECT id FROM usuarios WHERE nome=?), ?, ?)
    """, (usuario, conversa_id, pergunta))
    request_id = cursor.lastrowid
    cursor.execute("""
        INSERT INTO ai_responses (request_id, conteudo, modelo_usado, tokens)
        VALUES (?, ?, ?, ?)
    """, (request_id, resposta, modelo_usado, tokens))
    response_id = cursor.lastrowid
    cursor.execute("""
        INSERT INTO mensagens (conversa_id, request_id, response_id)
        VALUES (?, ?, ?)
    """, (conversa_id, request_id, response_id))
    cursor.execute("""
        INSERT INTO memorias (usuario_id, chave, valor, tipo, conversa_origem)
        VALUES ((SELECT id FROM usuarios WHERE nome=?), ?, ?, 'conversa', ?)
    """, (usuario, f"pergunta_{request_id}", pergunta, conversa_id))
    cursor.execute("""
        INSERT INTO memorias (usuario_id, chave, valor, tipo, conversa_origem)
        VALUES ((SELECT id FROM usuarios WHERE nome=?), ?, ?, 'conversa', ?)
    """, (usuario, f"resposta_{response_id}", resposta, conversa_id))
    conn.commit()
    conn.close()
