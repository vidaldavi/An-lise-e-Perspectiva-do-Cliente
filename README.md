# Google Maps Review Summarizer

Coleta avaliações públicas de estabelecimentos no Google Maps com Playwright e gera análise/resumo usando GroqCloud.

## Instalação

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
playwright install chromium
```

## Configuração

Copie o exemplo de ambiente:

```bash
copy .env.example .env
```

No Linux/macOS:

```bash
cp .env.example .env
```

Abra o `.env` e coloque sua chave real:

```env
GROQ_API_KEY=gsk_sua_chave_aqui
GROQ_MODEL=openai/gpt-oss-20b
```

> Se alguma chave foi colada em logs, prints ou conversa, revogue essa chave no painel do GroqCloud e gere outra.

## Executar

```bash
python app.py
```

Abra:

```text
http://localhost:5000
```

## URLs aceitas

- URL direta do estabelecimento: `https://www.google.com/maps/place/...`
- Link curto: `https://maps.app.goo.gl/...`
- Link curto novo: `https://share.google/...`
- Busca do Maps: `https://www.google.com/maps/search/...`

A URL mais confiável continua sendo a do estabelecimento direto. Se uma unidade abre sem avaliações, abra a aba **Avaliações** no Google Maps, copie o link da barra de endereços e cole no app.

## Problemas comuns

| Problema | Solução |
|---|---|
| `No module named flask` | Ative o venv e execute `pip install -r requirements.txt`. |
| `GROQ_API_KEY não definida` | Crie o `.env` com sua chave e reinicie o servidor. |
| `GROQ_API_KEY inválida` | Gere uma chave nova no GroqCloud e substitua no `.env`. |
| `Nenhuma avaliação foi coletada` | Use URL direta do estabelecimento ou copie a URL com a aba Avaliações aberta. |
| Poucas avaliações coletadas | O Google Maps pode limitar/lazy-load; tente novamente ou reduza para 20/30. |
| Erro do Playwright | Rode `playwright install chromium`. No Linux, se necessário, rode também `playwright install-deps`. |

---

## Seleção de modelos Groq e verificação da chave

Esta versão carrega o `.env` com `override=True`, então o valor do arquivo `.env` passa a ter prioridade sobre variáveis antigas abertas no PowerShell. Isso evita o problema de o servidor usar uma `GROQ_API_KEY` antiga mesmo depois de você editar o `.env`.

Na tela inicial há agora:

- um seletor de modelo Groq;
- um botão **Verificar Groq**, que faz uma chamada real mínima à API para confirmar se a chave funciona;
- exibição mascarada da chave carregada, por exemplo `gsk_abc...WXYZ`, para você confirmar se o servidor está usando a chave certa sem expor o segredo inteiro.

O botão de verificação mostra headers de limite quando a Groq os retorna, como requisições restantes, tokens restantes e `retry-after`. O histórico diário/mensal completo continua sendo consultado no console/logs da Groq.
