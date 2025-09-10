# 🚢 Deteção de Navios - Classificador Binário

Um classificador binário determinístico, pronto para produção, para detetar navios em imagens de satélite utilizando o conjunto de dados Airbus Ship Detection Challenge. Construído com PyTorch e otimizado para treino apenas em CPU em contentores Docker.

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Funcionalidades](#funcionalidades)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Início Rápido](#início-rápido)
- [Utilização](#utilização)
- [Configuração](#configuração)
- [Detalhes do Treino](#detalhes-do-treino)
- [Processamento de Dados](#processamento-de-dados)
- [Arquitetura do Modelo](#arquitetura-do-modelo)
- [Reprodutibilidade](#reprodutibilidade)
- [Otimização de Desempenho](#otimização-de-desempenho)
- [Resolução de Problemas](#resolução-de-problemas)
- [Contribuir](#contribuir)
- [Licença](#licença)

## 🎯 Visão Geral

Este projeto transforma o complexo desafio de segmentação do Airbus Ship Detection num problema mais simples de classificação binária: **navio vs. sem navio**. É concebido para ambientes de produção onde não existem recursos de GPU, utilizando técnicas avançadas de otimização para CPU e treino determinístico para total reprodutibilidade.

**Principais casos de utilização:**

- Vigilância e monitorização marítima
- Análise de imagens de satélite
- Deteção de navios em dados de deteção remota
- Fins educativos/de investigação em visão por computador

## ✨ Funcionalidades

### 🚀 **Capacidades Principais**

- **Classificação Binária**: Deteção de navios (1) vs. sem navio (0)
- **Treino Determinístico**: Reprodutibilidade total entre execuções
- **Paragem Antecipada (Early Stopping)**: Com base em PR-AUC com paciência configurável
- **Retoma por Checkpoint**: Retomar o treino a partir de qualquer checkpoint
- **Otimização para CPU**: Otimizado para ARM64 com melhorias de threading

### 🔧 **Funcionalidades Técnicas**

- **Contentorização com Docker**: Pronto para implantação em produção
- **Aumento de Dados (Data Augmentation)**: Transformações no treino para robustez
- **Gestão de Desbalanceamento de Classes**: Cálculo automático de `pos_weight`
- **Amostragem Estratificada**: Mantém o equilíbrio de classes em subconjuntos
- **Registo Abrangente**: Progresso e métricas em tempo real
- **Gestão de Recursos**: Limites configuráveis de memória e CPU

### 📊 **Métricas e Monitorização**

- **Métricas de Treino**: Loss, taxa de aprendizagem, tempo por batch
- **Métricas de Validação**: Acurácia, Precisão, Revocação, F1, ROC-AUC, PR-AUC
- **Acompanhamento de Progresso**: Estimativas de ETA e barras de progresso em tempo real
- **Gestão de Checkpoints**: Guarda o melhor e o último modelo

## 📁 Estrutura do Projeto

```
final-project/
├── airbus-ship-detection/          # Diretório do dataset (é necessário fazer o download no Kaggle)
│   ├── train_v2/                   # Imagens de treino
├── labels/                         # Ficheiros de rótulos e divisões
│   ├── binary_labels.csv           # Rótulos binários navio/sem navio
│   ├── segmentations_labels.csv    # Dados originais de segmentação (vem com o dataset do Kaggle)
│   └── splits/                     # Divisões treino/validação
│       ├── train.csv               # Conjunto de treino
│       └── val.csv                 # Conjunto de validação
├── utils/                          # Utilitários de pré-processamento de dados
│   ├── make_binary_labels.py       # Converter segmentação para binário
│   └── make_train_val_split.py     # Criar divisões treino/validação
├── outputs/                        # Saídas do modelo e checkpoints
│   └── models/                     # Checkpoints guardados
├── docker-compose.yml              # Orquestração de contentores
├── Dockerfile                      # Definição do contentor
├── requirements.txt                # Dependências Python
├── train_binary_classifier.py      # Script principal de treino
└── README.md                       # Este ficheiro
```

## 📋 Pré-requisitos

### **Requisitos do Sistema**

- **SO**: Linux, macOS ou Windows com Docker
- **RAM**: Mínimo 16 GB, recomendado 64 GB+
- **CPU**: Processador multi‑core (ARM64 ou x86_64)
- **Armazenamento**: 10 GB+ de espaço livre para dataset e modelos

### **Requisitos de Software**

- **Docker**: Versão 20.10+
- **Docker Compose**: Versão 2.0+
- **Git**: Para clonar o repositório

### **Conjunto de Dados**

- **Airbus Ship Detection Challenge**: Transferir de [Kaggle](https://www.kaggle.com/c/airbus-ship-detection)

## 🚀 Início Rápido

### **1. Clonar e Configurar**

```bash
git clone https://github.com/gabriel-n-carvalho/ship-detection-binary-classifier
cd ship-detection-binary-classifier
```

### **2. Transferir o dataset**

- Transfira o dataset do Kaggle
- Copie a pasta `train_v2/` do dataset `airbus-ship-detection` para o diretório raiz do projeto, para que a sua estrutura de pastas corresponda ao exemplo acima.
- Coloque o ficheiro de rótulos de segmentação `train_ship_segmentations_v2.csv` em `labels/train_ship_segmentations_v2.csv`.
- Execute os seguintes comandos para converter os rótulos de segmentação em rótulos binários e criar uma divisão estratificada dos dados em conjuntos de treino e validação.

### **3. Preparar Dados**

```bash
# Converter rótulos de segmentação para binário (se necessário). Este script converte os rótulos originais em rótulos binários.
python utils/make_binary_labels.py

# Criar divisões de rótulos (se ainda não existirem). Este script cria uma divisão estratificada em treino e validação.
python utils/make_train_val_split.py
```

### **4. Iniciar o Treino com Docker**

```bash
# Este comando inicia o processo de treino e verá as barras de progresso no primeiro plano.
docker-compose run --rm efficientnet-training

```

**Nota:** Usar `docker-compose up` pode fazer com que as barras de progresso do tqdm sejam colocadas em buffer e apenas apareçam após o fim do treino, devido à forma como o output é tratado ([ver issue #771 do tqdm](https://github.com/tqdm/tqdm/issues/771)). Para atualizações em tempo real, recomenda-se usar `docker-compose run`.

### **5. Monitorizar o Treino**

```bash
# Com `docker-compose run`, verá o progresso do tqdm em tempo real no primeiro plano.

# Se optar por usar `up`, pode seguir os logs, mas o tqdm pode não atualizar em tempo real:
docker-compose logs -f

# Verificar o estado dos contentores
docker-compose ps
```

## 📖 Utilização

### **Interface de Linha de Comandos**

O script de treino suporta uma configuração extensa por linha de comandos:

```bash
python train_binary_classifier.py \
    --seed 42 \
    --fold 0 \
    --batch-size 32 \
    --epochs 100 \
    --lr 3e-4 \
    --img-size 256 \
    --grad-accum-steps 1 \
    --early-stop-patience 20 \
    --model-name tf_efficientnetv2_b0.in1k \
    --subset-size 1000
```

### **Variáveis de Ambiente do Docker**

Configure os parâmetros de treino via variáveis de ambiente:

```bash
# Substituir valores por defeito
export SEED=123
export BATCH_SIZE=64
export EPOCHS=200
export LR=1e-4

# Iniciar treino
docker-compose up
```

### **Subconjuntos de Dados para Desenvolvimento**

Use subconjuntos de dados para iteração mais rápida durante o desenvolvimento:

```bash
# Usar 10% dos dados
python train_binary_classifier.py --subset-fraction 0.1

# Usar um número específico de amostras
python train_binary_classifier.py --subset-size 1000
```

## ⚙️ Configuração

### **Hiperparâmetros de Treino**

| Parâmetro             | Predefinição | Descrição                                   |
| --------------------- | ------------ | ------------------------------------------- |
| `SEED`                | 42           | Semente aleatória para reprodutibilidade    |
| `BATCH_SIZE`          | 32           | Tamanho do batch de treino                  |
| `EPOCHS`              | 100          | Número máximo de épocas                     |
| `LR`                  | 3e-4         | Taxa de aprendizagem                        |
| `IMG_SIZE`            | 256          | Tamanho da imagem de entrada                |
| `GRAD_ACCUM_STEPS`    | 1            | Passos de acumulação de gradientes          |
| `EARLY_STOP_PATIENCE` | 20           | Paciência para paragem antecipada           |

### **Arquitetura do Modelo**

- **Modelo Base**: EfficientNet-V2-B0 (pré‑treinado no ImageNet)
- **Tamanho de Entrada**: Imagens RGB 256x256
- **Saída**: Único sigmoid (probabilidade 0-1)
- **Função de Perda**: BCEWithLogitsLoss com balanceamento de classes

### **Aumento de Dados**

**Transformações de Treino:**

- Redimensionar para 256x256
- Flip horizontal aleatório (50%)
- Flip vertical aleatório (20%)
- Rotação aleatória (±5°)
- Variação de cor (color jitter)
- Normalização do ImageNet

**Transformações de Validação:**

- Redimensionar para 256x256
- Normalização do ImageNet

## 🎓 Detalhes do Treino

### **Ciclo de Treino**

1. **Inicialização da Época**: Definir sementes aleatórias, criar barras de progresso
2. **Fase de Treino**: Forward, cálculo de loss, backpropagation
3. **Fase de Validação**: Avaliação do modelo, cálculo de métricas
4. **Checkpoints**: Guardar o melhor modelo (PR-AUC) e o último checkpoint
5. **Paragem Antecipada**: Monitorizar a melhoria do PR-AUC

### **Função de Perda**

```python
# Balanceamento automático de classes
pos_weight = max(1.0, negative_samples / positive_samples)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### **Otimização**

- **Otimizador**: AdamW com weight decay
- **Agendador**: Cosine annealing da taxa de aprendizagem
- **Clipping de Gradiente**: Clipping da norma em 1.0
- **Acumulação de Gradientes**: Configurável para batches efetivos maiores

### **Métricas**

- **Acurácia**: Acurácia global de classificação
- **Precisão**: Precisão na deteção de navios
- **Revocação**: Revocação na deteção de navios
- **F1-Score**: Média harmónica de precisão e revocação
- **ROC-AUC**: Área sob a curva ROC
- **PR-AUC**: Área sob a curva precisão‑revocação (métrica principal)

## 🔄 Processamento de Dados

### **Pipeline de Dados**

1. **Carregamento de Imagem**: Conversão para RGB baseada em PIL com tratamento de erros
2. **Transformação**: Transforms do PyTorch com aumento de dados
3. **Batching**: DataLoader determinístico com seed nos workers
4. **Transferência para Dispositivo**: Movimentação de dados otimizada para CPU

### **Processamento de Rótulos**

- **Conversão Binária**: Máscaras de segmentação → rótulos binários
- **Divisão Estratificada**: Mantém o equilíbrio de classes em treino/validação
- **Estabilidade de Tipo**: `float32` consistente para rótulos

### **Validação de Dados**

- **Verificação de Balanceamento**: Garante amostras positivas e negativas
- **Carregamento de Imagem**: Tratamento gracioso de imagens corrompidas
- **Consistência de Rótulos**: Validação do formato e valores dos rótulos

## 🏗️ Arquitetura do Modelo

### **EfficientNet-V2-B0**

- **Arquitetura**: Escalonamento composto com blocos residuais invertidos
- **Parâmetros**: ~7,1M parâmetros treináveis
- **Entrada**: 256x256x3 imagens RGB
- **Saída**: Único logit para classificação binária
- **Pré‑treino**: Pesos do ImageNet‑1K

### **Modificações no Modelo**

```python
model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=1  # Classificação binária
)
```

## 🔒 Reprodutibilidade

### **Treino Determinístico**

- **Controlo de Sementes**: Todos os geradores aleatórios com a mesma seed
- **Estado do RNG**: Guarda/restaura o estado completo para retomar
- **Seed dos Workers**: Inicialização determinística dos workers do DataLoader
- **Seleção de Algoritmos**: Algoritmos determinísticos do PyTorch forçados

### **Sistema de Checkpoints**

- **Melhor Modelo**: Guardado com base na melhoria do PR-AUC
- **Último Checkpoint**: Sempre guardado para retomar
- **Estado do RNG**: Preservação completa do estado aleatório
- **Metadados**: Configuração do modelo e histórico de treino

## ⚡ Otimização de Desempenho

### **Otimização para CPU**

- **Threading**: Configuração ótima de threads OpenMP/MKL/BLAS
- **Gestão de Memória**: Definições conservadoras de memória partilhada
- **Carregamento de Dados**: Um único worker no Docker para evitar conflitos
- **Processamento em Batch**: Operações de tensores eficientes

### **Otimizações no Docker**

- **Limites de Recursos**: Alocação configurável de memória e CPU
- **Memória Partilhada**: `shm_size` de 16 GB para estabilidade do DataLoader
- **Montagens de Volumes**: Acesso eficiente a dados e persistência
- **Variáveis de Ambiente**: Definições do PyTorch otimizadas

### **Gestão de Memória**

- **Acumulação de Gradientes**: Batches efetivos maiores
- **Limpeza de Checkpoints**: Garbage collection automática
- **Otimização de Tensores**: Operações específicas para CPU
- **Memória Partilhada**: Otimização de memória no contentor Docker

## 🐛 Resolução de Problemas

### **Problemas Comuns**

#### **Falta de Memória (OOM)**

```bash
# Reduzir o tamanho do batch
export BATCH_SIZE=16

# Reduzir o tamanho da imagem
export IMG_SIZE=128

# Usar subconjunto de dados
--subset-size 500
```

#### **Treino Lento**

```bash
# Aumentar o batch (se a memória permitir)
export BATCH_SIZE=64

# Reduzir o tamanho da imagem
export IMG_SIZE=128

# Usar acumulação de gradientes
export GRAD_ACCUM_STEPS=2
```

#### **Problemas com Docker**

```bash
# Ver logs do contentor
docker-compose logs

# Reiniciar o contentor
docker-compose restart

# Ver consumo de recursos
docker stats
```

### **Modo de Depuração**

Ativar logs verbosos para depuração:

```bash
# Definir variável de ambiente
export PYTHONUNBUFFERED=1

# Executar com output de debug
python -u train_binary_classifier.py --subset-size 100
```

