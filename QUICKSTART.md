# GrowingSparseSNN - Guia Rápido de Início 🚀

## ✅ O que foi criado

### 1. **Repositório GitHub**
- URL: https://github.com/nfriacowboy/growingSparseSNN
- Repositório público com licença MIT
- Commit inicial feito com toda a estrutura

### 2. **Arquitetura GrowingSparseSNN** (`src/models/growing_snn.py`)
- SNN dinâmica com neurônios LIF (Leaky Integrate-and-Fire)
- **Neurogenesis**: Adiciona neurônios quando firing rate < 0.05
- **Pruning**: Remove neurônios com firing rate < 0.005
- Começa com 64 neurônios, cresce até 512-1024
- Otimizado para GPU AMD com ROCm

### 3. **Ambiente de Teste** (`src/environments/grid_world.py`)
- Grid 15×15 com agente e comida
- Agente aprende a forragear (coletar comida)
- Observação: 2 canais (posição do agente, posições de comida)
- 4 ações: cima, baixo, esquerda, direita

### 4. **Sistema de Treinamento** (`src/training/trainer.py`)
- Algoritmo REINFORCE com baseline
- Growth automático a cada 100 episódios
- Pruning automático a cada 50 episódios
- Métricas exportadas para Prometheus

### 5. **Monitoramento** (`src/monitoring/metrics.py`)
- Prometheus/OpenMetrics integrado
- Métricas: neuron count, firing rates, sparsity, rewards, energy
- Porta: 8000
- Grafana dashboards via docker-compose

### 6. **Docker + ROCm** (`docker/`)
- Dockerfile baseado em `rocm/pytorch:rocm6.0`
- docker-compose.yml com SNN + Prometheus + Grafana
- Suporte completo para GPU AMD

### 7. **Testes Completos** (`tests/`)
- `test_growth.py`: Testa neurogenesis
- `test_pruning.py`: Testa poda de neurônios
- `test_learning.py`: Testa treinamento e REINFORCE
- `test_environment.py`: Testa ambiente de simulação
- Executar com: `pytest tests/ -v --cov=src`

### 8. **Scripts Úteis**
- `setup.sh`: Setup inicial do ambiente
- `run_tests.sh`: Executa testes com coverage
- `build_docker.sh`: Build da imagem Docker
- `demo.py`: Demo com visualizações

### 9. **Documentação**
- `README.md`: Documentação principal
- `docs/architecture.md`: Arquitetura detalhada
- `configs/training_config.yaml`: Configuração de hiperparâmetros

## 🎯 Próximos Passos

### 1. Setup Local (sem Docker)
```bash
# Instalar dependências
pip install -r requirements.txt

# Rodar testes
./run_tests.sh

# Demo rápido (treina 500 episódios)
python demo.py
```

### 2. Setup com Docker + ROCm (Recomendado para GPU AMD)
```bash
# Build imagem
./build_docker.sh

# Ou usar docker-compose
cd docker
docker-compose up -d

# Ver logs
docker-compose logs -f snn-training

# Acessar container
docker exec -it growing-snn-train bash
```

### 3. Treinar Modelo Completo
```bash
# Com configuração padrão
python src/training/train.py

# Com configuração customizada
python src/training/train.py --config configs/training_config.yaml

# Com mais episódios
python src/training/train.py --episodes 5000 --lr 0.001
```

### 4. Monitorar Treinamento
```bash
# Iniciar serviços de monitoramento
cd docker
docker-compose up prometheus grafana

# Acessar:
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
# - Métricas raw: http://localhost:8000/metrics
```

### 5. Experimentos Sugeridos

#### Baseline: Rede Fixa
```python
# Modificar training_config.yaml:
training:
  grow_interval: 999999  # Nunca cresce
  prune_interval: 999999  # Nunca poda
```

#### Growth Agressivo
```python
training:
  grow_interval: 25       # Cresce mais frequentemente
  grow_threshold: 0.1     # Threshold mais alto
  neurons_per_grow: 64    # Adiciona mais neurônios
```

#### Pruning Agressivo
```python
training:
  prune_interval: 20
  prune_threshold: 0.01   # Remove neurônios menos ativos
```

## 📊 Métricas Implementadas

| Métrica | Descrição |
|---------|-----------|
| `snn_neuron_count` | Número atual de neurônios |
| `snn_avg_firing_rate` | Taxa média de disparo |
| `snn_sparsity` | Proporção de neurônios inativos |
| `snn_episode_reward` | Reward do episódio atual |
| `snn_growth_events_total` | Total de eventos de crescimento |
| `snn_pruning_events_total` | Total de eventos de poda |
| `snn_energy_estimate` | Estimativa de energia (spikes × neurons) |

## 🔬 Hipótese Experimental

**H0**: Uma SNN com crescimento dinâmico (64→512 neurons) + pruning aprende melhor que uma rede fixa de 512 neurons.

**Métricas para validar**:
1. **Sample efficiency**: Episódios até convergência
2. **Final performance**: Reward médio após convergência
3. **Energy efficiency**: Energia total consumida
4. **Adaptação**: Performance em novas tarefas

## 🛠 Troubleshooting

### Problema: Norse não encontrado
```bash
pip install norse
```

### Problema: ROCm não detectado
```bash
# Verificar instalação ROCm
rocm-smi

# Verificar PyTorch com ROCm
python -c "import torch; print(torch.__version__)"
# Deve mostrar algo como: 2.1.0+rocm5.7

# Se não, reinstalar PyTorch para ROCm:
pip install torch --index-url https://download.pytorch.org/whl/rocm5.7
```

### Problema: GPU não detectada
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Se False, verificar drivers e ROCm
```

### Problema: Porta 8000 em uso
```bash
# Modificar porta no código ou:
python src/training/train.py --metrics-port 8001
```

## 📚 Estrutura de Arquivos

```
growingSparseSNN/
├── src/
│   ├── models/
│   │   └── growing_snn.py          # ⭐ Modelo principal
│   ├── environments/
│   │   └── grid_world.py           # Ambiente de simulação
│   ├── training/
│   │   ├── trainer.py              # ⭐ Loop de treinamento
│   │   └── train.py                # Script principal
│   └── monitoring/
│       └── metrics.py              # Prometheus metrics
├── tests/                          # Testes unitários
├── docker/                         # Docker + ROCm
├── configs/                        # Configurações YAML
├── docs/                           # Documentação
├── demo.py                         # ⭐ Demo rápido
└── README.md                       # Documentação principal
```

## 🎓 Conceitos Chave

### Neurogenesis (Crescimento)
- Adiciona neurônios quando capacidade é insuficiente
- Trigger: avg_firing_rate < 0.05
- Preserva pesos existentes
- Inicializa novos com Kaiming + noise

### Pruning (Poda)
- Remove neurônios inativos
- Trigger: firing_rate < 0.005
- Mantém no mínimo 32 neurônios
- Reconstrói rede menor

### LIF Neurons
- Leaky Integrate-and-Fire
- τ_mem = 20ms, τ_syn = 10ms
- Threshold = 1.0
- Spikes binários (0 ou 1)

### REINFORCE Learning
- Policy gradient com baseline
- Discount γ = 0.99
- Adam optimizer
- Gradient clipping (max_norm=1.0)

## 🚀 Status do Projeto

✅ Repositório GitHub criado  
✅ Arquitetura implementada  
✅ Testes completos  
✅ Docker + ROCm configurado  
✅ Monitoramento Prometheus/Grafana  
✅ Demo funcional  
✅ Documentação completa  

🔄 **Próximo**: Treinar e validar a hipótese experimental!

## 📞 Recursos

- **Repositório**: https://github.com/nfriacowboy/growingSparseSNN
- **Norse Docs**: https://norse.github.io/norse/
- **ROCm Docs**: https://rocm.docs.amd.com/
- **PyTorch SNN Tutorial**: https://snntorch.readthedocs.io/

---

**Criado em**: 2026-02-06  
**Autor**: nfriacowboy  
**GPU Target**: AMD Radeon RX 6900 XT (ROCm 6.0)  
**Framework**: PyTorch + Norse + OpenMetrics
