web/
├── app/                      # Rotas e páginas estruturadas via App Router
│   ├── agenda/               # Gerenciamento diário de horários
│   ├── clientes/             # Visualização e gestão de clientes
│   ├── equipe/               # Gestão da equipe de profissionais
│   ├── novo-agendamento/     # Formulário interativo com busca assíncrona
│   ├── visagismo/            # Upload de fotos e histórico de análise facial
│   └── page.tsx              # Dashboard Geral & Analytics
├── components/               # Componentes compartilhados e atômicos
│   ├── Header.tsx            # Barra superior de navegação contextual
│   ├── Sidebar.tsx           # Menu de navegação lateral fixa/responsiva
│   ├── ImageUploadModule.tsx # Componente de Drag & Drop para análise visagista
│   └── OccupancyChart.tsx    # Gráficos analíticos de horários de maior fluxo
└── services/
    └── api.ts                # Configuração unificada do Axios e Interceptors
    
```bash
# Instalar dependências do ecossistema front
npm install

# Configuração de Variáveis de Ambiente
# Crie um arquivo chamado .env.local na raiz da pasta 'web':
NEXT_PUBLIC_API_URL=http://localhost:3333
```

```bash
npm run dev
```
