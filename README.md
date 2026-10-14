# Modelagem da Taxa de Câmbio com o Processo de Ornstein-Uhlenbeck

Projeto da disciplina **Introdução à Modelagem Matemática**.

O programa modela e projeta a taxa de câmbio do real usando o **processo de Ornstein-Uhlenbeck (OU)**, um modelo que assume que o valor tende a reverter para uma média de longo prazo. Os dados são as cotações diárias do real desde **01/07/1994**.

## O modelo

O processo OU é descrito pela equação diferencial estocástica:

```
dS = θ(μ − S)dt + σ dW
```

| Parâmetro | Significado |
|-----------|-------------|
| θ | velocidade de reversão à média |
| μ | nível de equilíbrio de longo prazo |
| σ | volatilidade |
| W | movimento browniano |

A estimação é feita por **mínimos quadrados (OLS)** na forma discreta:

```
ΔS_t = α + β·S_{t-1} + ε_t
θ = −β        μ = α/θ        σ = desvio-padrão dos resíduos
meia-vida = ln(2)/θ
```

## Como o programa funciona

1. **Tratamento de dados e filtro de período**
   Lê o arquivo histórico (`cambio.csv`) e permite escolher o período analisado pela linha de comando: data de início, data de fim ou apenas os últimos N anos.

2. **Ajuste do modelo**
   Estima θ, μ e σ, além da meia-vida e do R² da regressão. A estabilidade da série é verificada com um teste de estacionaridade **ADF** otimizado (número de lags escolhido por AIC com *early-stopping*; valor crítico de 5% = −2,86).

3. **Simulação e projeção futura**
   Roda uma **simulação de Monte Carlo** (1000 trajetórias) para projetar o câmbio 2 anos à frente (diário e mensal), com bandas de incerteza de 50% e 90%. A projeção é comparada com o valor realizado, quando ele existe nos dados.

### Módulo de validação (Estudo BC nº 25/2018)

O código também faz um **backtest rolling** que compara o modelo OU com o **passeio aleatório** (benchmark), calculando:

- EQM relativo (OU/RW): valores menores que 1 indicam que o OU supera o passeio aleatório
- Teste de **Diebold-Mariano** (com correção de Harvey-Leybourne-Newbold)
- **Decomposição de Theil** do erro (viés, variância e covariância)
- RMSE e MAE por horizonte
- Tabela de EQM relativo por fator *k* (percentis de variação), comparada com os valores de referência do paper do BC

## Requisitos

- Python 3.8+
- `pandas`, `numpy`, `matplotlib`, `scipy`

```bash
pip install pandas numpy matplotlib scipy
```

## Formato dos dados

O CSV de entrada deve ter as colunas `data` e `valor`:

```csv
data,valor
1994-07-01,0.9
1994-07-04,0.89
...
```

O código aceita números com vírgula ou ponto decimal.

## Uso

```bash
# Todos os dados disponíveis
python cambio_ou_modelo.py

# A partir de uma data
python cambio_ou_modelo.py --inicio 2015-01-01

# Intervalo específico
python cambio_ou_modelo.py --inicio 2015-01-01 --fim 2023-12-31

# Últimos 5 anos (sobrepõe --inicio/--fim)
python cambio_ou_modelo.py --anos_recentes 5

# Ajustando o backtest do BC
python cambio_ou_modelo.py --bc_janela 3 --bc_horizonte 6
```

### Argumentos

| Argumento | Descrição | Padrão |
|-----------|-----------|--------|
| `--csv` | Arquivo CSV de entrada | `cambio.csv` |
| `--inicio` | Data inicial (AAAA-MM-DD) | — |
| `--fim` | Data final (AAAA-MM-DD) | — |
| `--anos_recentes` | Usa apenas os últimos N anos | — |
| `--bc_janela` | Janela (anos) do backtest BC | 5 |
| `--bc_horizonte` | Horizonte máximo (períodos) do backtest BC | 12 |

## Saídas

As pastas são nomeadas pelo período analisado e **nunca sobrescrevem** resultados anteriores (se já existir, é criado o sufixo `_v2`, `_v3`, ...).

```
resultados_ou_AAAAMMDD-AAAAMMDD/
├── diario/
│   ├── 01_serie_projecao_diario.png     # série + projeção Monte Carlo
│   ├── 02_residuos_diario.png
│   ├── 03_qqplot_diario.png
│   ├── 04_distribuicao_residuos_diario.png
│   ├── 05_acf_residuos_diario.png
│   ├── 06_forca_reversao_diario.png
│   ├── 07_dispersao_delta_diario.png
│   ├── OU_global.csv                    # parâmetros estimados
│   └── debug/
└── mensal/
    ├── 01_... a 07_... (idem)
    ├── 08_mensal_dia1_projecao.png
    ├── OU_global.csv
    └── debug/

resultados_bc_AAAAMMDD-AAAAMMDD/
├── diario/ e mensal/
│   ├── bc_A_eqm_relativo_*.png
│   ├── bc_B_diebold_mariano_*.png
│   ├── bc_C_decomposicao_theil_*.png
│   ├── bc_D_rmse_mae_*.png
│   ├── bc_E_tabela2_*.png
│   ├── tabela1_*.csv
│   └── tabela2_*.csv
```

## Observações

- O gráfico principal da série diária abre uma **janela interativa** (zoom, pan e salvar). Feche a janela para a execução continuar. O PNG também é salvo automaticamente. Em ambientes sem interface gráfica, essa etapa pode falhar.
- Se o ADF indicar série **não estacionária**, o programa avisa que o modelo OU é suspeito para aquele período.
- Se θ ≤ 0 (sem reversão à média), o modelo é considerado inválido e nenhum resultado é salvo para aquela frequência.
- O modelo não é uma recomendação financeira. É um exercício acadêmico de modelagem.

## Autor

**Matheus Henrique Gomes dos Santos**
Instituto de Matemática e Estatística (IME), Universidade Federal de Goiás (UFG)

