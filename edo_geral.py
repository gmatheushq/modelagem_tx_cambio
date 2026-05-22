

import pandas as pd
import numpy as np
import re
import os
import sys
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy import stats

# =============================================================================
# CONFIGURAÇÃO DE PERÍODO — via argparse (linha de comando)
# =============================================================================
# Uso:
#   python cambio_ou_modelo.py                            → usa todos os dados
#   python cambio_ou_modelo.py --inicio 2015-01-01        → a partir de 2015
#   python cambio_ou_modelo.py --inicio 2015-01-01 --fim 2023-12-31
#   python cambio_ou_modelo.py --anos_recentes 5          → últimos 5 anos
# =============================================================================

parser = argparse.ArgumentParser(
    description="Modelo OU para câmbio com validação BC",
    formatter_class=argparse.RawTextHelpFormatter,
    epilog="""
Exemplos de uso:
  python cambio_ou_modelo.py
      → roda com todos os dados disponíveis

  python cambio_ou_modelo.py --inicio 2015-01-01
      → usa dados a partir de 01/01/2015

  python cambio_ou_modelo.py --inicio 2015-01-01 --fim 2023-12-31
      → usa apenas o intervalo 2015–2023

  python cambio_ou_modelo.py --anos_recentes 5
      → usa apenas os últimos 5 anos (sobrepõe --inicio/--fim)

  python cambio_ou_modelo.py --bc_janela 3 --bc_horizonte 6
      → backtest BC com janela de 3 anos e horizonte de 6 períodos
"""
)
parser.add_argument("--inicio",         type=str, default=None,
                    help="Data inicial no formato AAAA-MM-DD")
parser.add_argument("--fim",            type=str, default=None,
                    help="Data final no formato AAAA-MM-DD")
parser.add_argument("--anos_recentes",  type=int, default=None,
                    help="Usar apenas os últimos N anos (sobrepõe --inicio/--fim)")
parser.add_argument("--bc_janela",      type=int, default=5,
                    help="Janela (anos) para backtest BC [padrão: 5]")
parser.add_argument("--bc_horizonte",   type=int, default=12,
                    help="Horizonte máximo (períodos) para backtest BC [padrão: 12]")
parser.add_argument("--csv",            type=str, default="cambio.csv",
                    help="Arquivo CSV de entrada [padrão: cambio.csv]")

# parse_known_args evita erros quando rodado com outros flags (ex: Jupyter)
args, _ = parser.parse_known_args()


# =============================================================================
# 0. Helpers de pasta — nomeação por período e anti-sobrescrita
# =============================================================================
def montar_nome_pasta_periodo(df_filtrado, prefixo="resultados"):
    """
    Gera um nome de pasta baseado no período efetivo dos dados filtrados.
    Formato: {prefixo}_AAAAMMDD-AAAAMMDD
    Se a pasta já existir, adiciona sufixo _v2, _v3, ... para não sobrescrever.
    """
    d_ini = df_filtrado["data"].min().strftime("%Y%m%d")
    d_fim = df_filtrado["data"].max().strftime("%Y%m%d")
    base  = f"{prefixo}_{d_ini}-{d_fim}"

    # Anti-sobrescrita: adiciona versão se já existir
    pasta = base
    v = 2
    while os.path.exists(pasta):
        pasta = f"{base}_v{v}"
        v += 1

    return pasta


# Raiz dos resultados — será definida após filtrar o df principal
# (declarada como variável global após o filtro)
PASTA_RAIZ_OU = None
PASTA_RAIZ_BC = None


# =============================================================================
# 1. Limpeza de números
# =============================================================================
def limpar_numero(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    x = re.sub(r"[^\d,.\-]", "", x)
    if "," in x and "." in x:
        if x.rfind(",") > x.rfind("."):
            x = x.replace(".", "").replace(",", ".")
        else:
            x = x.replace(",", "")
    elif "," in x:
        x = x.replace(",", ".")
    try:
        return float(x)
    except Exception:
        return np.nan


# =============================================================================
# 2. Ler CSV
# =============================================================================
df_full = pd.read_csv(args.csv, sep=",", dtype=str, engine="python")
df_full.columns = df_full.columns.str.strip()

if "data" not in df_full.columns or "valor" not in df_full.columns:
    raise ValueError("CSV deve ter colunas: data, valor")

df_full["valor"] = df_full["valor"].apply(limpar_numero)
df_full["data"]  = pd.to_datetime(df_full["data"], errors="coerce")
df_full = df_full.dropna().copy()
df_full = df_full.sort_values("data").reset_index(drop=True)


# =============================================================================
# 3. Filtro de período (linha de comando)
# =============================================================================
def aplicar_filtro_periodo(df_in, inicio=None, fim=None, anos_recentes=None):
    """
    Aplica filtro de período ao DataFrame.
    Prioridade: anos_recentes > inicio/fim.
    Imprime no console o período efetivo usado.
    """
    df_out = df_in.copy()

    if anos_recentes is not None:
        data_corte = df_out["data"].max() - pd.DateOffset(years=anos_recentes)
        df_out = df_out[df_out["data"] >= data_corte]
        print(f"\n  Filtro: últimos {anos_recentes} anos "
              f"(a partir de {data_corte.date()})")
    else:
        if inicio is not None:
            df_out = df_out[df_out["data"] >= pd.to_datetime(inicio)]
            print(f"\n  Filtro início: {inicio}")
        if fim is not None:
            df_out = df_out[df_out["data"] <= pd.to_datetime(fim)]
            print(f"  Filtro fim:    {fim}")

    if df_out.empty:
        raise ValueError("Nenhum dado após aplicar filtro de período.")

    print(f"  Período efetivo: {df_out['data'].min().date()} "
          f"→ {df_out['data'].max().date()} | {len(df_out)} obs\n")
    return df_out.reset_index(drop=True)


# Aplica filtro global ao df principal
df = aplicar_filtro_periodo(
    df_full,
    inicio=args.inicio,
    fim=args.fim,
    anos_recentes=args.anos_recentes
)

# Define raízes de resultados nomeadas pelo período — NUNCA sobrescreve pasta existente
PASTA_RAIZ_OU = montar_nome_pasta_periodo(df, prefixo="resultados_ou")
PASTA_RAIZ_BC = montar_nome_pasta_periodo(df, prefixo="resultados_bc")

print(f"  Pasta OU : {PASTA_RAIZ_OU}/")
print(f"  Pasta BC : {PASTA_RAIZ_BC}/")


# =============================================================================
# 3.1. Série real após o período de ajuste
# =============================================================================
def obter_realizado_futuro(df_full, data_ultima_amostra, data_final_proj, modo):
    """
    Retorna os valores reais disponíveis entre o fim da amostra ajustada e o fim da projeção.
    Inclui o último ponto da amostra para ligar visualmente a curva histórica com a pós-amostra.

    modo = "diario"  -> compara por datas diárias
    modo = "mensal"  -> compara apenas com datas no dia 1 do mês
    """
    df_real = df_full.sort_values("data").copy()

    if modo == "mensal":
        df_real = df_real[df_real["data"].dt.day == 1].copy()

    df_real = df_real[
        (df_real["data"] >= data_ultima_amostra) &
        (df_real["data"] <= data_final_proj)
    ].copy()

    return df_real.reset_index(drop=True)


# =============================================================================
# 4. ADF Otimizado com busca por bisseção (Golden Section Search)
# =============================================================================
def teste_adf_manual(y, lags=1):
    """
    ADF simplificado (OLS puro, sem statsmodels).
    Regride Δy_t em y_{t-1} e Δy_{t-1},...,Δy_{t-lags} + constante.
    Retorna estatística t do coeficiente de y_{t-1}.

    Valores críticos aproximados (MacKinnon):
      1%: -3.43   5%: -2.86   10%: -2.57
    """
    dy = np.diff(y)
    n  = len(dy)
    if n < lags + 3:
        return np.nan

    cols = [y[lags:-1]] if lags > 0 else [y[:-1]]
    for lag in range(1, lags + 1):
        cols.append(dy[lags - lag: n - lag])
    cols.append(np.ones(n - lags))

    X = np.column_stack(cols)
    Y = dy[lags:]

    try:
        coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        resid  = Y - X @ coef
        s2     = np.sum(resid**2) / max(len(Y) - X.shape[1], 1)
        XtX_inv = np.linalg.pinv(X.T @ X)
        se     = np.sqrt(np.diag(s2 * XtX_inv))
        t_stat = coef[0] / se[0] if se[0] > 1e-12 else np.nan
    except Exception:
        t_stat = np.nan

    return t_stat


def selecionar_lags_aic(y, max_lags=10):
    """
    Seleciona número ótimo de lags para o ADF via AIC.
    Mais eficiente que calcular todos os lags sequencialmente:
    usa early-stopping quando AIC começa a subir consistentemente.

    Retorna: (melhor_lag, aic_min)
    """
    dy = np.diff(y)
    n  = len(dy)
    max_lags = min(max_lags, n // 4)

    melhor_lag = 0
    melhor_aic = np.inf
    pioras_consecutivas = 0

    for lag in range(0, max_lags + 1):
        if n - lag < lag + 3:
            break

        if lag == 0:
            X_cols = [y[:-1], np.ones(n)]
        else:
            X_cols = [y[lag:-1]]
            for l in range(1, lag + 1):
                X_cols.append(dy[lag - l: n - l])
            X_cols.append(np.ones(n - lag))

        X = np.column_stack(X_cols)
        Y = dy[lag:] if lag > 0 else dy

        try:
            coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            resid = Y - X @ coef
            k     = X.shape[1]
            T     = len(Y)
            sigma2 = np.sum(resid**2) / T
            aic = np.log(sigma2) + 2 * k / T

            if aic < melhor_aic:
                melhor_aic = aic
                melhor_lag = lag
                pioras_consecutivas = 0
            else:
                pioras_consecutivas += 1
                # early-stopping: 3 pioras consecutivas → para busca
                if pioras_consecutivas >= 3:
                    break
        except Exception:
            continue

    return melhor_lag, melhor_aic


def teste_adf_otimizado(y, max_lags=10, verbose=False):
    """
    Versão otimizada do ADF:
    1. Seleciona lags via AIC com early-stopping (O(max_lags) no pior caso)
    2. Roda ADF com o lag ótimo
    3. Retorna dict com estatística, lag escolhido e resultado do AIC

    Antes: calculava todos lags comparando um a um → O(n * max_lags)
    Agora: para assim que encontra mínimo AIC → O(n * k*) onde k* << max_lags
    """
    melhor_lag, aic_min = selecionar_lags_aic(y, max_lags=max_lags)
    t_stat = teste_adf_manual(y, lags=melhor_lag)

    if verbose:
        print(f"    ADF: lag_ótimo={melhor_lag}, AIC={aic_min:.4f}, t={t_stat:.3f}")

    return {
        "t_stat":    t_stat,
        "lag_otimo": melhor_lag,
        "aic_min":   aic_min
    }


# =============================================================================
# 5. Ajuste OU com validações
# =============================================================================
def ajustar_ou(y, pasta, nome="", freq_anual=252):
    """
    Estima o processo OU por OLS na forma discreta:
      ΔS_t = α + β·S_{t-1} + ε_t

    Parâmetros estimados:
      θ = -β          (reversão à média)
      μ = α / θ       (equilíbrio de longo prazo)
      σ = std(ε)      (volatilidade residual)

    Equivalência com a EDP contínua:
      dS = θ(μ - S)dt + σ dW
    """
    if len(y) < 10:
        print(f"   [{nome}] Série muito curta ({len(y)} obs). Pulando.")
        return None

    # ADF otimizado (seleção de lags via AIC com early-stopping)
    adf_res       = teste_adf_otimizado(y, max_lags=min(10, len(y)//5))
    t_adf         = adf_res["t_stat"]
    critico_5pct  = -2.86
    estacionaria  = (not np.isnan(t_adf)) and (t_adf < critico_5pct)

    if not estacionaria:
        print(f"   [{nome}] ADF t={t_adf:.3f} > {critico_5pct} "
              f"→ série possivelmente não estacionária. Modelo OU suspeito.")

    dy    = np.diff(y)
    y_lag = y[:-1]
    n     = len(dy)

    A = np.vstack([np.ones(n), y_lag]).T
    coef, _, _, _ = np.linalg.lstsq(A, dy, rcond=None)
    alpha, beta   = coef

    theta = -beta
    mu    = alpha / theta if abs(theta) > 1e-10 else np.nan

    dy_hat    = alpha + beta * y_lag
    residuals = dy - dy_hat
    sigma     = np.std(residuals, ddof=2)

    theta_anual = theta * freq_anual
    sigma_anual = sigma * np.sqrt(freq_anual)

    if theta <= 0:
        print(f"  [{nome}] θ={theta:.4f} ≤ 0 → sem reversão à média. "
              f"Modelo OU inválido para esta série.")
        return None

    meia_vida = np.log(2) / theta

    # R² da regressão
    #ss_res = np.sum(residuals**2)
    #ss_tot = np.sum((dy - np.mean(dy))**2)
    #r2     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    c = np.mean(dy)

    term1 = np.sum((dy - c)**2)
    term2 = np.sum((dy_hat - c)**2)
    term3 = -2 * np.sum((dy - c) * (dy_hat - c))

    ss_res = term1 + term2 + term3
    ss_tot = term1

    r2 = 1 - ss_res / ss_tot

    if pasta:
        tabela = pd.DataFrame({
            "t":         np.arange(1, n + 1),
            "y_(t-1)":   y_lag,
            "y_t":       y[1:],
            "Δy_real":   dy,
            "Δy_modelo": dy_hat,
            "Erro":      residuals
        })
        os.makedirs(f"{pasta}/debug", exist_ok=True)
        tabela.to_csv(f"{pasta}/debug/debug_{nome}.csv", index=False)

    y_last        = y[-1]
    dy_esperado   = theta * (mu - y_last)
    y_proj_1passo = y_last + dy_esperado

    return {
        "theta":                   theta,
        "theta_anual":             theta_anual,
        "mu":                      mu,
        "sigma":                   sigma,
        "sigma_anual":             sigma_anual,
        "meia_vida_periodos":      meia_vida,
        "ADF_t":                   t_adf,
        "ADF_lag_otimo":           adf_res["lag_otimo"],
        "serie_estacionaria_5pct": estacionaria,
        "R2_regressao":            r2,
        "Último Valor":            y_last,
        "Δy_esperado_1passo":      dy_esperado,
        "Projeção_1passo":         y_proj_1passo,
        "n_obs":                   n
    }


# =============================================================================
# 6. Simulação Monte Carlo
# =============================================================================
def simular_ou_mc(res, passos, n_sim=1000):
    theta = res["theta"]
    mu    = res["mu"]
    sigma = res["sigma"]
    y0    = res["Último Valor"]

    trajetorias    = np.zeros((passos + 1, n_sim))
    trajetorias[0] = y0

    rng = np.random.default_rng(42)

    for t in range(1, passos + 1):
        eps = rng.standard_normal(n_sim)
        trajetorias[t] = (trajetorias[t - 1]
                          + theta * (mu - trajetorias[t - 1])
                          + sigma * eps)
    return trajetorias


# =============================================================================
# 7. Plots separados — cada gráfico em seu próprio arquivo
# =============================================================================

def _titulo_ou(res, modo):
    """Gera string de título padrão com parâmetros OU."""
    aviso = " NÃO ESTACIONÁRIA (ADF)" if not res["serie_estacionaria_5pct"] else ""
    return (
        f"OU — {modo.capitalize()}  |  "
        f"θ={res['theta']:.4f}  μ={res['mu']:.4f}  σ={res['sigma']:.4f}  "
        f"meia-vida={res['meia_vida_periodos']:.1f} per.  "
        f"R²={res['R2_regressao']:.3f}{aviso}"
    )


def plotar_serie_projecao(df_plot, df_full, res, pasta, modo, interativo=False):
    """
    Subplot 1 — Série histórica + MMs + projeção Monte Carlo.

    interativo=True  → abre janela matplotlib com toolbar (zoom/pan/salvar).
    interativo=False → salva PNG diretamente (modo batch).
    """
    df_plot = df_plot.sort_values("data").copy()
    y       = df_plot["valor"].values
    datas   = df_plot["data"]

    passos        = 2*365 if modo == "diario" else 2*12
    trajetorias   = simular_ou_mc(res, passos, n_sim=1000)
    media_proj    = trajetorias.mean(axis=1)
    p5            = np.percentile(trajetorias, 5,  axis=1)
    p25           = np.percentile(trajetorias, 25, axis=1)
    p75           = np.percentile(trajetorias, 75, axis=1)
    p95           = np.percentile(trajetorias, 95, axis=1)
    freq_fut      = "B" if modo == "diario" else "MS"
    datas_futuras = pd.date_range(start=datas.iloc[-1], periods=passos + 1,
                                  freq=freq_fut)

    data_ultima_amostra = datas.iloc[-1]
    data_fim_proj       = datas_futuras[-1]

    df_real_futuro = obter_realizado_futuro(
        df_full=df_full,
        data_ultima_amostra=data_ultima_amostra,
        data_final_proj=data_fim_proj,
        modo=modo
    )

    mm_w1 = 20 if modo == "diario" else 3
    mm_w2 = 60 if modo == "diario" else 6
    df_plot[f"MM{mm_w1}"] = df_plot["valor"].rolling(mm_w1).mean()
    df_plot[f"MM{mm_w2}"] = df_plot["valor"].rolling(mm_w2).mean()

    # ── Backend: interativo usa TkAgg/Qt5Agg; batch usa Agg ──────────────────
    if interativo:
        # Tenta backends gráficos na ordem de preferência
        for backend in ("TkAgg", "Qt5Agg", "Qt4Agg", "WXAgg", "MacOSX"):
            try:
                matplotlib.use(backend)
                break
            except Exception:
                continue

    fig, ax = plt.subplots(figsize=(22, 9), dpi=110)
    fig.suptitle(_titulo_ou(res, modo), fontsize=12, fontweight="bold")

    ax.plot(datas, y, color="#1a1a2e", linewidth=1.2, label="Série Real")
    ax.plot(datas, df_plot[f"MM{mm_w1}"], color="royalblue",
            linewidth=1, label=f"MM{mm_w1}")
    ax.plot(datas, df_plot[f"MM{mm_w2}"], color="darkorange",
            linewidth=1, label=f"MM{mm_w2}")

    if not df_real_futuro.empty:
        df_mm_futuro = df_full.sort_values("data").copy()
        if modo == "mensal":
            df_mm_futuro = df_mm_futuro[df_mm_futuro["data"].dt.day == 1].copy()
        df_mm_futuro = df_mm_futuro[df_mm_futuro["data"] <= data_fim_proj].copy()
        df_mm_futuro[f"MM{mm_w1}"] = df_mm_futuro["valor"].rolling(mm_w1).mean()
        df_mm_futuro[f"MM{mm_w2}"] = df_mm_futuro["valor"].rolling(mm_w2).mean()
        df_mm_futuro = df_mm_futuro[df_mm_futuro["data"] >= data_ultima_amostra].copy()

        ax.plot(df_real_futuro["data"], df_real_futuro["valor"],
                color="black", linestyle="--", linewidth=2,
                marker="o", markersize=3, label="Realizado pós-amostra")
        ax.plot(df_mm_futuro["data"], df_mm_futuro[f"MM{mm_w1}"],
                color="royalblue", linestyle="--", linewidth=1.5, alpha=0.9,
                label=f"MM{mm_w1} pós-amostra")
        ax.plot(df_mm_futuro["data"], df_mm_futuro[f"MM{mm_w2}"],
                color="darkorange", linestyle="--", linewidth=1.5, alpha=0.9,
                label=f"MM{mm_w2} pós-amostra")

    ax.plot(datas_futuras, media_proj, "--", color="crimson",
            linewidth=2, label="Projeção OU (média MC)")
    ax.fill_between(datas_futuras, p5, p95, alpha=0.15,
                    color="crimson", label="IC 90%")
    ax.fill_between(datas_futuras, p25, p75, alpha=0.25,
                    color="crimson", label="IC 50%")
    ax.axhline(res["mu"], linestyle=":", color="green",
               linewidth=1.5, label=f"μ={res['mu']:.4f}")
    ax.set_title("Série Histórica + Projeção Monte Carlo (Cadeia de Markov)", fontsize=11)
    ax.legend(fontsize=9, ncol=4)
    ax.grid(alpha=0.25)
    ax.tick_params(axis="x", rotation=30)
    ax.set_xlabel("Data")
    ax.set_ylabel("Câmbio")

    if interativo:
        # Exibe janela interativa com toolbar (zoom, pan, salvar em qualquer resolução)
        plt.tight_layout()
        print(f"\n  Janela interativa aberta — use a toolbar para zoom/pan/salvar.")
        print(f"      Feche a janela para continuar a execução.")
        plt.show()
        # Após fechar, salva também o PNG padrão
        fig2, ax2 = plt.subplots(figsize=(22, 9), dpi=150)
        # recria (fig já foi fechada pelo show()) — salva cópia estática
        matplotlib.use("Agg")
        fig2, ax2 = _recriar_serie_projecao(df_plot, df_full, res, modo,
                                             mm_w1, mm_w2, datas, y,
                                             datas_futuras, media_proj,
                                             p5, p25, p75, p95,
                                             df_real_futuro, data_ultima_amostra,
                                             data_fim_proj)
        caminho = f"{pasta}/01_serie_projecao_{modo}.png"
        fig2.savefig(caminho, bbox_inches="tight", dpi=150)
        plt.close(fig2)
        print(f"  Série+Projeção salva: {caminho}")
    else:
        caminho = f"{pasta}/01_serie_projecao_{modo}.png"
        fig.savefig(caminho, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Série+Projeção salva: {caminho}")


def _recriar_serie_projecao(df_plot, df_full, res, modo,
                             mm_w1, mm_w2, datas, y,
                             datas_futuras, media_proj,
                             p5, p25, p75, p95,
                             df_real_futuro, data_ultima_amostra,
                             data_fim_proj):
    """Helper: recria figura estática após janela interativa ser fechada."""
    fig, ax = plt.subplots(figsize=(22, 9), dpi=150)
    fig.suptitle(_titulo_ou(res, modo), fontsize=12, fontweight="bold")

    ax.plot(datas, y, color="#1a1a2e", linewidth=1.2, label="Série Real")
    ax.plot(datas, df_plot[f"MM{mm_w1}"], color="royalblue",
            linewidth=1, label=f"MM{mm_w1}")
    ax.plot(datas, df_plot[f"MM{mm_w2}"], color="darkorange",
            linewidth=1, label=f"MM{mm_w2}")

    if not df_real_futuro.empty:
        df_mm_futuro = df_full.sort_values("data").copy()
        if modo == "mensal":
            df_mm_futuro = df_mm_futuro[df_mm_futuro["data"].dt.day == 1].copy()
        df_mm_futuro = df_mm_futuro[df_mm_futuro["data"] <= data_fim_proj].copy()
        df_mm_futuro[f"MM{mm_w1}"] = df_mm_futuro["valor"].rolling(mm_w1).mean()
        df_mm_futuro[f"MM{mm_w2}"] = df_mm_futuro["valor"].rolling(mm_w2).mean()
        df_mm_futuro = df_mm_futuro[df_mm_futuro["data"] >= data_ultima_amostra].copy()

        ax.plot(df_real_futuro["data"], df_real_futuro["valor"],
                color="black", linestyle="--", linewidth=2,
                marker="o", markersize=3, label="Realizado pós-amostra")
        ax.plot(df_mm_futuro["data"], df_mm_futuro[f"MM{mm_w1}"],
                color="royalblue", linestyle="--", linewidth=1.5, alpha=0.9,
                label=f"MM{mm_w1} pós-amostra")
        ax.plot(df_mm_futuro["data"], df_mm_futuro[f"MM{mm_w2}"],
                color="darkorange", linestyle="--", linewidth=1.5, alpha=0.9,
                label=f"MM{mm_w2} pós-amostra")

    ax.plot(datas_futuras, media_proj, "--", color="crimson",
            linewidth=2, label="Projeção OU (média MC)")
    ax.fill_between(datas_futuras, p5, p95, alpha=0.15,
                    color="crimson", label="IC 90%")
    ax.fill_between(datas_futuras, p25, p75, alpha=0.25,
                    color="crimson", label="IC 50%")
    ax.axhline(res["mu"], linestyle=":", color="green",
               linewidth=1.5, label=f"μ={res['mu']:.4f}")
    ax.set_title("Série Histórica + Projeção Monte Carlo (Cadeia de Markov)", fontsize=11)
    ax.legend(fontsize=9, ncol=4)
    ax.grid(alpha=0.25)
    ax.tick_params(axis="x", rotation=30)
    ax.set_xlabel("Data")
    ax.set_ylabel("Câmbio")
    plt.tight_layout()
    return fig, ax


def plotar_residuos(df_plot, res, pasta, modo):
    """Subplot 2 — Resíduos da regressão OU."""
    df_plot = df_plot.sort_values("data").copy()
    y       = df_plot["valor"].values
    datas   = df_plot["data"]

    dy        = np.diff(y)
    y_lag     = y[:-1]
    alpha     = res["theta"] * res["mu"]
    beta      = -res["theta"]
    dy_hat    = alpha + beta * y_lag
    residuals = dy - dy_hat

    fig, ax = plt.subplots(figsize=(14, 5), dpi=110)
    fig.suptitle(_titulo_ou(res, modo), fontsize=10)

    ax.plot(datas[1:], residuals, color="#444", linewidth=0.7, alpha=0.8)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.fill_between(datas[1:], residuals, 0,
                    where=residuals > 0, color="steelblue", alpha=0.3)
    ax.fill_between(datas[1:], residuals, 0,
                    where=residuals < 0, color="salmon", alpha=0.3)
    ax.set_title("Resíduos da Regressão OU", fontsize=11)
    ax.set_ylabel("ε_t")
    ax.set_xlabel("Data")
    ax.grid(alpha=0.2)
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    caminho = f"{pasta}/02_residuos_{modo}.png"
    fig.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Resíduos salvo: {caminho}")


def plotar_qq(df_plot, res, pasta, modo):
    """Subplot 3 — QQ-plot dos resíduos."""
    y         = df_plot.sort_values("data")["valor"].values
    dy        = np.diff(y)
    y_lag     = y[:-1]
    alpha     = res["theta"] * res["mu"]
    beta      = -res["theta"]
    residuals = dy - (alpha + beta * y_lag)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
    fig.suptitle(_titulo_ou(res, modo), fontsize=9)

    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, s=10, color="steelblue", alpha=0.6)
    ax.plot(osm, slope * np.array(osm) + intercept, color="red",
            linewidth=1.5, label=f"r={r:.3f}")
    ax.set_title("QQ-Plot Resíduos (Normal)", fontsize=11)
    ax.set_xlabel("Quantis Teóricos")
    ax.set_ylabel("Quantis Observados")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    caminho = f"{pasta}/03_qqplot_{modo}.png"
    fig.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  QQ-Plot salvo: {caminho}")


def plotar_distribuicao_residuos(df_plot, res, pasta, modo):
    """Subplot 4 — Histograma + curva normal dos resíduos."""
    y         = df_plot.sort_values("data")["valor"].values
    dy        = np.diff(y)
    y_lag     = y[:-1]
    alpha     = res["theta"] * res["mu"]
    beta      = -res["theta"]
    residuals = dy - (alpha + beta * y_lag)

    skew_val = stats.skew(residuals)
    kurt_val = stats.kurtosis(residuals)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    fig.suptitle(_titulo_ou(res, modo), fontsize=9)

    ax.hist(residuals, bins=40, color="steelblue", alpha=0.7,
            density=True, edgecolor="white", linewidth=0.3)
    xr = np.linspace(residuals.min(), residuals.max(), 200)
    ax.plot(xr, stats.norm.pdf(xr, residuals.mean(), residuals.std()),
            color="red", linewidth=2, label="Normal teórica")
    ax.set_title(f"Distribuição Resíduos\n"
                 f"Assimetria={skew_val:.2f}  Curtose={kurt_val:.2f}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    caminho = f"{pasta}/04_distribuicao_residuos_{modo}.png"
    fig.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Distribuição resíduos salvo: {caminho}")


def plotar_acf(df_plot, res, pasta, modo):
    """Subplot 5 — ACF dos resíduos."""
    y         = df_plot.sort_values("data")["valor"].values
    dy        = np.diff(y)
    y_lag     = y[:-1]
    alpha     = res["theta"] * res["mu"]
    beta      = -res["theta"]
    residuals = dy - (alpha + beta * y_lag)

    max_lags_acf = min(30, len(residuals) // 4)
    acf_vals = []
    for lag in range(0, max_lags_acf + 1):
        if lag == 0:
            acf_vals.append(1.0)
        else:
            c = np.corrcoef(residuals[lag:], residuals[:-lag])[0, 1]
            acf_vals.append(c)
    lags_arr  = np.arange(len(acf_vals))
    conf_band = 1.96 / np.sqrt(len(residuals))

    fig, ax = plt.subplots(figsize=(10, 5), dpi=110)
    fig.suptitle(_titulo_ou(res, modo), fontsize=9)

    ax.bar(lags_arr, acf_vals, color="steelblue", alpha=0.7, width=0.8)
    ax.axhline(conf_band,  color="red", linestyle="--",
               linewidth=1, label=f"IC 95% (±{conf_band:.3f})")
    ax.axhline(-conf_band, color="red", linestyle="--", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("ACF dos Resíduos", fontsize=11)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelação")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    caminho = f"{pasta}/05_acf_residuos_{modo}.png"
    fig.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ACF salvo: {caminho}")


def plotar_forca_reversao(df_plot, res, pasta, modo):
    """Subplot 6 — Força de reversão θ(μ − S_t)."""
    df_plot = df_plot.sort_values("data").copy()
    y       = df_plot["valor"].values
    datas   = df_plot["data"]
    force   = res["theta"] * (res["mu"] - y)

    fig, ax = plt.subplots(figsize=(14, 5), dpi=110)
    fig.suptitle(_titulo_ou(res, modo), fontsize=9)

    ax.plot(datas, force, color="purple", linewidth=0.8, alpha=0.8)
    ax.fill_between(datas, force, 0,
                    where=force > 0, color="green",  alpha=0.25, label="Pressão alta")
    ax.fill_between(datas, force, 0,
                    where=force < 0, color="orange", alpha=0.25, label="Pressão baixa")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Força de Reversão: θ(μ − S_t)", fontsize=11)
    ax.set_ylabel("dS esperado")
    ax.set_xlabel("Data")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    caminho = f"{pasta}/06_forca_reversao_{modo}.png"
    fig.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Força de reversão salvo: {caminho}")


def plotar_dispersao_delta(df_plot, res, pasta, modo):
    """Subplot 7 — Dispersão ΔS real vs ΔS modelo."""
    y         = df_plot.sort_values("data")["valor"].values
    dy        = np.diff(y)
    y_lag     = y[:-1]
    alpha     = res["theta"] * res["mu"]
    beta      = -res["theta"]
    dy_hat    = alpha + beta * y_lag

    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
    fig.suptitle(_titulo_ou(res, modo), fontsize=9)

    ax.scatter(dy_hat, dy, s=5, alpha=0.3, color="steelblue")
    lim = max(abs(dy_hat).max(), abs(dy).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1.5, label="y=x (perfeito)")
    ax.set_title(f"ΔS Real vs ΔS Modelo\nR²={res['R2_regressao']:.3f}", fontsize=11)
    ax.set_xlabel("ΔS Modelo (fitted)")
    ax.set_ylabel("ΔS Real")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    plt.tight_layout()
    caminho = f"{pasta}/07_dispersao_delta_{modo}.png"
    fig.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Dispersão ΔS salvo: {caminho}")


# =============================================================================
# 7b. Dashboard legado (mantido para compatibilidade — chama os plots separados)
# =============================================================================
def plotar_dashboard_ou(df_plot, df_full, res, pasta, modo):
    """
    Mantido para compatibilidade.
    Agora delega para as funções individuais de plot — cada gráfico em arquivo separado.
    O gráfico principal diário (série + projeção) é aberto em janela interativa.
    """
    interativo_principal = (modo == "diario")

    plotar_serie_projecao(df_plot, df_full, res, pasta, modo,
                          interativo=interativo_principal)
    plotar_residuos(df_plot, res, pasta, modo)
    plotar_qq(df_plot, res, pasta, modo)
    plotar_distribuicao_residuos(df_plot, res, pasta, modo)
    plotar_acf(df_plot, res, pasta, modo)
    plotar_forca_reversao(df_plot, res, pasta, modo)
    plotar_dispersao_delta(df_plot, res, pasta, modo)


# =============================================================================
# 8. Plot mensal (dia 1) com Monte Carlo — arquivo separado
# =============================================================================
def plotar_mensal_primeiro_dia(df, df_full, res, pasta):
    df        = df.sort_values("data").copy()
    df        = df[df["data"].dt.day == 1].copy()
    df["MM3"] = df["valor"].rolling(3).mean()
    df["MM6"] = df["valor"].rolling(6).mean()

    y     = df["valor"].values
    datas = df["data"]
    mu    = res["mu"]

    passos        = 2*12
    trajetorias   = simular_ou_mc(res, passos, n_sim=1000)
    media_proj    = trajetorias.mean(axis=1)
    p5            = np.percentile(trajetorias, 5,  axis=1)
    p95           = np.percentile(trajetorias, 95, axis=1)
    datas_futuras = pd.date_range(start=datas.iloc[-1], periods=passos + 1, freq="MS")

    data_ultima_amostra = datas.iloc[-1]
    data_fim_proj       = datas_futuras[-1]

    df_real_futuro = obter_realizado_futuro(
        df_full=df_full,
        data_ultima_amostra=data_ultima_amostra,
        data_final_proj=data_fim_proj,
        modo="mensal"
    )

    fig, ax = plt.subplots(figsize=(18, 8), dpi=120)
    ax.plot(datas, y, marker="o", markersize=4,
            label="Dias 1 (Real)", color="black", linewidth=1.5)
    ax.plot(datas, df["MM3"], label="MM3", color="royalblue", linewidth=1)
    ax.plot(datas, df["MM6"], label="MM6", color="darkorange", linewidth=1)

    if not df_real_futuro.empty:
        df_mm_futuro = df_full.sort_values("data").copy()
        df_mm_futuro = df_mm_futuro[df_mm_futuro["data"].dt.day == 1].copy()
        df_mm_futuro = df_mm_futuro[df_mm_futuro["data"] <= data_fim_proj].copy()
        df_mm_futuro["MM3"] = df_mm_futuro["valor"].rolling(3).mean()
        df_mm_futuro["MM6"] = df_mm_futuro["valor"].rolling(6).mean()
        df_mm_futuro = df_mm_futuro[df_mm_futuro["data"] >= data_ultima_amostra].copy()

        ax.plot(df_real_futuro["data"], df_real_futuro["valor"],
                linestyle="-", marker="o", markersize=4,
                color="black", linewidth=1.5, alpha=0.8,
                label="Realizado pós-amostra")
        ax.plot(df_mm_futuro["data"], df_mm_futuro["MM3"],
                linestyle="--", color="royalblue", linewidth=1.5, alpha=0.9,
                label="MM3 pós-amostra")
        ax.plot(df_mm_futuro["data"], df_mm_futuro["MM6"],
                linestyle="--", color="darkorange", linewidth=1.5, alpha=0.9,
                label="MM6 pós-amostra")

    ax.plot(datas_futuras, media_proj, linestyle="--", marker="o", markersize=4,
            label="Projeção OU (média MC)", color="red", linewidth=1.5)
    ax.fill_between(datas_futuras, p5, p95, alpha=0.2, color="red", label="IC 90%")
    ax.axhline(mu, linestyle=":", label=f"μ={mu:.4f}", color="green", linewidth=1)
    ax.set_title(
        f"OU — Mensal (dia 1) | θ={res['theta']:.4f}, "
        f"μ={mu:.4f}, σ={res['sigma']:.4f}, "
        f"meia-vida={res['meia_vida_periodos']:.1f} meses"
        + (" Série não estacionária (ADF)" if not res["serie_estacionaria_5pct"] else ""),
        fontsize=11
    )
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Câmbio")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    caminho = f"{pasta}/08_mensal_dia1_projecao.png"
    fig.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Gráfico mensal (dia 1) salvo: {caminho}")


# =============================================================================
# 9. Função principal
# =============================================================================
def rodar_modelo(df_base, df_full, modo):
    print(f"\n{'='*60}")
    print(f" Rodando modelo OU — {modo.upper()}")
    print(f"  EDP: dS = θ(μ–S)dt + σdW  (processo Ornstein-Uhlenbeck)")
    print(f"{'='*60}")

    df         = df_base.copy()
    freq_anual = 252 if modo == "diario" else 2*12

    if modo == "mensal":
        df = df[df["data"].dt.day == 1].copy()

    df = df.sort_values("data")

    pasta = f"{PASTA_RAIZ_OU}/{modo}"
    os.makedirs(f"{pasta}/debug", exist_ok=True)

    y = df["valor"].values
    print(f"  Observações: {len(y)}")

    res_global = ajustar_ou(y, pasta, "GLOBAL", freq_anual=freq_anual)

    if res_global is None:
        print(f"  Modelo OU inválido para {modo}. Nenhum resultado salvo.")
        return

    print(f"\n  ── Parâmetros OU ──────────────────────────────────────")
    print(f"  θ (discr.)   = {res_global['theta']:.4f} "
          f"| θ (anual) = {res_global['theta_anual']:.4f}")
    print(f"  μ (equilíbrio) = {res_global['mu']:.4f}")
    print(f"  σ (discr.)   = {res_global['sigma']:.4f} "
          f"| σ (anual) = {res_global['sigma_anual']:.4f}")
    print(f"  Meia-vida    = {res_global['meia_vida_periodos']:.1f} períodos")
    print(f"  R²           = {res_global['R2_regressao']:.4f}")
    print(f"  ADF t-stat   = {res_global['ADF_t']:.3f} "
          f"(lag_ótimo={res_global['ADF_lag_otimo']}) "
          f"| Estacionária(5%): {res_global['serie_estacionaria_5pct']}")
    print(f"  Último valor = {res_global['Último Valor']:.4f}")
    print(f"  Proj. 1 passo= {res_global['Projeção_1passo']:.4f} "
          f"(Δ esperado: {res_global['Δy_esperado_1passo']:+.4f})")

    # ── Gráficos individuais (cada um em arquivo separado) ───────────────────
    # Diário: gráfico principal abre janela interativa com zoom/pan/salvar
    plotar_dashboard_ou(df, df_full, res_global, pasta, modo)

    if modo == "mensal":
        plotar_mensal_primeiro_dia(df, df_full, res_global, pasta)

    pd.DataFrame([res_global]).to_csv(f"{pasta}/OU_global.csv", index=False)
    print(f" Parâmetros salvos: {pasta}/OU_global.csv")


# ╔══════════════════════════════════════════════════════════════╗
# ║     MÓDULO DE VALIDAÇÃO BC Nº 25/2018                       ║
# ║     + Decomposição Viés/Variância                           ║
# ║     + Teste Diebold-Mariano                                 ║
# ║     + Tratamento de Datas de Referência                     ║
# ╚══════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────
# BC-1. Prepara série em LOG
# ─────────────────────────────────────────────────────────────────────────────
def preparar_log(df_base, modo, anos_recentes=10):
    """
    Retorna série com log_valor e datas de referência alinhadas.
    Datas de referência: para modo mensal, usa o último dia útil
    do mês anterior (padrão adotado pelo BCB para taxa de fechamento).
    """
    df = df_base.sort_values("data").copy()

    data_corte = df["data"].max() - pd.DateOffset(years=anos_recentes)
    df = df[df["data"] >= data_corte].copy()

    if modo == "mensal":
        # Agrupa por mês e pega o ÚLTIMO dia disponível (fechamento mensal)
        df["mes"] = df["data"].dt.to_period("M")
        df = df.groupby("mes").last().reset_index()
        df["data"] = df["mes"].dt.to_timestamp("M")
        df = df.drop(columns=["mes"])
        print(f"  [BC] Modo mensal: usando último dia disponível por mês "
              f"(alinhamento BCB)")

    df["log_valor"] = np.log(df["valor"])

    print(f"  [BC] Período: {df['data'].min().date()} → {df['data'].max().date()} "
          f"| {len(df)} obs ({modo})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# BC-2. Previsão OU h-passos à frente
# ─────────────────────────────────────────────────────────────────────────────
def previsao_ou_h(res, y0_log, h):
    """
    Projeção iterativa discreta do OU para h passos.
    Forma determinística (sem ruído): E[S_{t+h}].
    """
    theta = res["theta"]
    mu    = res["mu"]
    y = y0_log
    for _ in range(h):
        y = y + theta * (mu - y)
    return y


# ─────────────────────────────────────────────────────────────────────────────
# BC-3. Teste Diebold-Mariano
# ─────────────────────────────────────────────────────────────────────────────
def diebold_mariano(e1, e2, h=1):
    """
    Teste Diebold-Mariano (1995) para comparação de precisão preditiva.

    H0: E[d_t] = 0  (modelos igualmente precisos)
    H1: E[d_t] ≠ 0  (um modelo é mais preciso)

    onde d_t = e1_t² - e2_t²  (diferença de perdas quadráticas)

    Implementação com correção de Harvey, Leybourne e Newbold (1997)
    para amostras finitas, com estimação de variância HAC.

    Retorna:
        dm_stat    : estatística DM (t-like)
        p_value    : p-valor bilateral
        conclusao  : string interpretando o resultado
    """
    e1 = np.array(e1)
    e2 = np.array(e2)
    d  = e1**2 - e2**2
    T  = len(d)

    if T < 5:
        return np.nan, np.nan, "Amostra insuficiente"

    d_bar = np.mean(d)

    # Variância HAC (Newey-West com h-1 lags)
    gamma0 = np.mean((d - d_bar)**2)
    hac_var = gamma0
    for k in range(1, h):
        if k >= T:
            break
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        hac_var += 2 * (1 - k / h) * gamma_k
    hac_var = max(hac_var, 1e-12)

    # Correção HLN para amostras finitas
    correction = np.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)
    dm_stat    = (d_bar / np.sqrt(hac_var / T)) * correction

    # Distribuição t com T-1 graus de liberdade (HLN)
    p_value = 2 * stats.t.sf(abs(dm_stat), df=T - 1)

    if p_value < 0.01:
        if dm_stat < 0:
            conclusao = "OU significativamente melhor (1%)"
        else:
            conclusao = "RW significativamente melhor (1%)"
    elif p_value < 0.05:
        if dm_stat < 0:
            conclusao = "OU melhor (5%)"
        else:
            conclusao = "RW melhor (5%)"
    elif p_value < 0.10:
        if dm_stat < 0:
            conclusao = "OU melhor (10%)"
        else:
            conclusao = "RW melhor (10%)"
    else:
        conclusao = "Sem diferença significativa"

    return dm_stat, p_value, conclusao


# ─────────────────────────────────────────────────────────────────────────────
# BC-4. Decomposição Viés/Variância
# ─────────────────────────────────────────────────────────────────────────────
def decompor_eqm(erros_y_real, erros_y_pred):
    """
    Decomposição de Theil do EQM em três componentes:
      EQM = Viés² + Variância_relativa + Covariância_imperfeita

    Componentes normalizados (somam 1):
      UM = Viés / EQM               → erro sistemático
      US = Variância / EQM          → erro de escala de flutuação
      UC = Covariância / EQM        → erro de covariância

    Interpretação:
      UM alto → previsão tem viés sistemático (sempre alta/baixa demais)
      US alto → previsão não captura a magnitude da variação
      UC alto → erro aleatório (inevitável, é o "melhor caso")
    """
    y_real = np.array(erros_y_real)
    y_pred = np.array(erros_y_pred)

    mean_real = np.mean(y_real)
    mean_pred = np.mean(y_pred)
    std_real  = np.std(y_real)
    std_pred  = np.std(y_pred)

    vies      = (mean_pred - mean_real)**2
    variancia = (std_pred - std_real)**2
    eqm_total = np.mean((y_real - y_pred)**2)

    if eqm_total < 1e-15:
        return {"vies": 0, "variancia": 0, "covariancia": 0,
                "UM": 0, "US": 0, "UC": 1}

    um = vies / eqm_total
    us = variancia / eqm_total
    uc = max(0, 1 - um - us)

    return {
        "vies":         vies,
        "variancia":    variancia,
        "covariancia":  eqm_total * uc,
        "UM":           um,
        "US":           us,
        "UC":           uc,
        "EQM":          eqm_total,
        "RMSE":         np.sqrt(eqm_total),
        "MAE":          np.mean(np.abs(y_real - y_pred))
    }


# ─────────────────────────────────────────────────────────────────────────────
# BC-5. Backtest rolling — EQM relativo + DM + Viés/Variância
# ─────────────────────────────────────────────────────────────────────────────
def backtest_bc_rolling(df_log, modo="mensal", max_h=12, janela_anos=5):
    freq   = 12 if modo == "mensal" else 252
    janela = janela_anos * freq
    y_full = df_log["log_valor"].values

    if len(y_full) < janela + max_h + 5:
        print(f"  ⚠️  [BC] Dados insuficientes para backtest {modo}.")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print(f"  VALIDAÇÃO BC — {modo.upper()} | janela={janela_anos}a | h=1..{max_h}")
    print(f"  EQM_rel < 1 → OU supera Passeio Aleatório")
    print(f"{'='*60}")
    print(f"  {'h':>3} | {'EQM_rel':>8} | {'DM_stat':>8} | {'p-val':>7} | "
          f"{'UM%':>6} | {'US%':>6} | {'UC%':>6} | Interpretação")
    print(f"  {'-'*80}")

    resultados = []

    for h in range(1, max_h + 1):
        erros_ou, erros_rw   = [], []
        y_reais, y_preds_ou  = [], []

        for t in range(janela, len(y_full) - h):
            y_train = y_full[t - janela: t]
            res = ajustar_ou(y_train, pasta=None, nome="bt", freq_anual=freq)
            if res is None:
                continue

            y_t    = y_full[t]
            y_real = y_full[t + h]

            y_ou = previsao_ou_h(res, y_t, h)
            y_rw = y_t

            erros_ou.append(y_real - y_ou)
            erros_rw.append(y_real - y_rw)
            y_reais.append(y_real)
            y_preds_ou.append(y_ou)

        if not erros_ou:
            continue

        # EQM relativo
        eqm_ou  = np.mean(np.array(erros_ou)**2)
        eqm_rw  = np.mean(np.array(erros_rw)**2)
        eqm_rel = eqm_ou / eqm_rw

        # Diebold-Mariano
        dm_stat, p_val, dm_conc = diebold_mariano(erros_ou, erros_rw, h=h)

        # Decomposição Viés/Variância
        decomp = decompor_eqm(y_reais, y_preds_ou)

        sinal = "y" if eqm_rel < 1 else "n"
        print(f"  {h:>3} | {eqm_rel:>8.3f} | {dm_stat:>8.3f} | {p_val:>7.4f} | "
              f"{decomp['UM']*100:>5.1f}% | {decomp['US']*100:>5.1f}% | "
              f"{decomp['UC']*100:>5.1f}% | {sinal} {dm_conc}")

        resultados.append({
            "h":         h,
            "EQM_OU":    eqm_ou,
            "EQM_RW":    eqm_rw,
            "EQM_rel":   eqm_rel,
            "RMSE_OU":   decomp["RMSE"],
            "MAE_OU":    decomp["MAE"],
            "DM_stat":   dm_stat,
            "DM_pval":   p_val,
            "DM_conc":   dm_conc,
            "UM":        decomp["UM"],
            "US":        decomp["US"],
            "UC":        decomp["UC"],
            "n":         len(erros_ou)
        })

    return pd.DataFrame(resultados)


# ─────────────────────────────────────────────────────────────────────────────
# BC-6. Backtest com fator k
# ─────────────────────────────────────────────────────────────────────────────
def backtest_bc_fator_k(df_log, modo="mensal", max_h=12, janela_anos=5,
                        percentis_k=(0, 25, 50, 75, 85, 90)):
    freq   = 12 if modo == "mensal" else 252
    janela = janela_anos * freq
    y_full = df_log["log_valor"].values

    var4 = np.abs(np.diff(y_full, n=4))
    var4 = np.concatenate([[np.nan] * 4, var4])

    print(f"\n{'='*60}")
    print(f"  TABELA 2 BC — {modo.upper()} | fator k | h=1..{max_h}")
    print(f"{'='*60}")

    tabela  = {}
    h_index = [f"h={h}" for h in range(1, max_h + 1)]

    for k in percentis_k:
        limite_k = np.nanpercentile(var4, k)
        col      = []

        for h in range(1, max_h + 1):
            erros_ou, erros_rw = [], []

            for t in range(janela, len(y_full) - h):
                if np.isnan(var4[t]) or var4[t] < limite_k:
                    continue
                y_train = y_full[t - janela: t]
                res     = ajustar_ou(y_train, pasta=None, nome="k", freq_anual=freq)
                if res is None:
                    continue
                y_t    = y_full[t]
                y_real = y_full[t + h]
                erros_ou.append((y_real - previsao_ou_h(res, y_t, h))**2)
                erros_rw.append((y_real - y_t)**2)

            if erros_ou:
                col.append(round(np.mean(erros_ou) / np.mean(erros_rw), 3))
            else:
                col.append(np.nan)

        tabela[f"k={k}"] = col

    df_tab = pd.DataFrame(tabela, index=h_index)
    print("\n  EQM relativo OU/RW (< 1 = OU melhor):")
    print(df_tab.to_string())
    print("\n  Referência paper BC (Focus vs RW, amostra completa, k=0):")
    ref = [1.13, 1.10, 1.08, 1.07, 1.06, 1.04, 1.01, 0.99, 0.99, 0.99, 0.99, 1.00]
    print("  h= " + "  ".join([f"{h:2d}" for h in range(1, 13)]))
    print("  BC " + "  ".join([f"{v:.2f}" for v in ref]))

    return df_tab


# ─────────────────────────────────────────────────────────────────────────────
# BC-7. Gráficos BC — cada um em arquivo separado
# ─────────────────────────────────────────────────────────────────────────────
def plotar_eqm_relativo(df_res, modo, pasta):
    """BC Plot A — EQM relativo OU vs RW por horizonte."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=110)
    fig.suptitle(f"Validação BC nº 25/2018 — {modo.upper()} — EQM Relativo",
                 fontsize=12, fontweight="bold")

    ax.plot(df_res["h"], df_res["EQM_rel"], marker="o",
            color="steelblue", linewidth=2, label="EQM OU / EQM RW")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5,
               label="Benchmark RW (=1.0)")
    ax.fill_between(df_res["h"], df_res["EQM_rel"], 1.0,
                    where=df_res["EQM_rel"] < 1.0,
                    alpha=0.25, color="green", label="OU melhor")
    ax.fill_between(df_res["h"], df_res["EQM_rel"], 1.0,
                    where=df_res["EQM_rel"] >= 1.0,
                    alpha=0.15, color="red", label="RW melhor")
    ref   = [1.13, 1.10, 1.08, 1.07, 1.06, 1.04, 1.01, 0.99, 0.99, 0.99, 0.99, 1.00]
    h_ref = list(range(1, 13))
    if max(df_res["h"]) >= 12:
        ax.plot(h_ref, ref, marker="s", linestyle=":", color="gray",
                linewidth=1.2, alpha=0.7, label="Paper BC (referência)")
    ax.set_xlabel("Horizonte h")
    ax.set_ylabel("EQM relativo (OU/RW)")
    ax.set_title("EQM Relativo por Horizonte")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    caminho = f"{pasta}/bc_A_eqm_relativo_{modo}.png"
    fig.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  BC EQM relativo salvo: {caminho}")


def plotar_diebold_mariano(df_res, modo, pasta):
    """BC Plot B — Teste Diebold-Mariano."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=110)
    fig.suptitle(f"Validação BC nº 25/2018 — {modo.upper()} — Diebold-Mariano",
                 fontsize=12, fontweight="bold")

    colors_dm = ["green" if p < 0.05 else ("orange" if p < 0.10 else "lightgray")
                 for p in df_res["DM_pval"]]
    ax.bar(df_res["h"], df_res["DM_stat"], color=colors_dm,
           alpha=0.8, edgecolor="white")
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(-1.96, color="red", linestyle="--", linewidth=1,
               label="t-crítico 5% (±1.96)")
    ax.axhline(1.96,  color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Horizonte h")
    ax.set_ylabel("Estatística DM (HLN)")
    ax.set_title("Teste Diebold-Mariano\n(negativo = OU melhor)")
    patches = [
        mpatches.Patch(color="green",     label="sig. 5%"),
        mpatches.Patch(color="orange",    label="sig. 10%"),
        mpatches.Patch(color="lightgray", label="não sig.")
    ]
    ax.legend(handles=patches, fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    caminho = f"{pasta}/bc_B_diebold_mariano_{modo}.png"
    fig.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f" BC Diebold-Mariano salvo: {caminho}")


def plotar_decomposicao_vies(df_res, modo, pasta):
    """BC Plot C — Decomposição de Theil Viés/Variância/Covariância."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=110)
    fig.suptitle(f"Validação BC nº 25/2018 — {modo.upper()} — Decomposição Theil",
                 fontsize=12, fontweight="bold")

    h_vals = df_res["h"].values
    um = df_res["UM"].values * 100
    us = df_res["US"].values * 100
    uc = df_res["UC"].values * 100
    ax.bar(h_vals, uc, label="UC — Covariância (aleatório)", color="#2ecc71", alpha=0.8)
    ax.bar(h_vals, us, bottom=uc, label="US — Variância (escala)", color="#f39c12", alpha=0.8)
    ax.bar(h_vals, um, bottom=uc+us, label="UM — Viés (sistemático)", color="#e74c3c", alpha=0.8)
    ax.set_xlabel("Horizonte h")
    ax.set_ylabel("% do EQM")
    ax.set_title("Decomposição de Theil do EQM\nViés (UM) + Variância (US) + Covariância (UC)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    caminho = f"{pasta}/bc_C_decomposicao_theil_{modo}.png"
    fig.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  BC Decomposição Theil salvo: {caminho}")


def plotar_rmse_mae(df_res, modo, pasta):
    """BC Plot D — RMSE e MAE por horizonte."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=110)
    fig.suptitle(f"Validação BC nº 25/2018 — {modo.upper()} — RMSE e MAE",
                 fontsize=12, fontweight="bold")

    ax.plot(df_res["h"], df_res["RMSE_OU"], marker="o", color="steelblue",
            linewidth=2, label="RMSE OU")
    ax.plot(df_res["h"], df_res["MAE_OU"], marker="s", color="darkorange",
            linewidth=2, linestyle="--", label="MAE OU")
    ax.set_xlabel("Horizonte h")
    ax.set_ylabel("Erro (log-câmbio)")
    ax.set_title("RMSE e MAE por Horizonte")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    caminho = f"{pasta}/bc_D_rmse_mae_{modo}.png"
    fig.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f" BC RMSE/MAE salvo: {caminho}")


def plotar_eqm_bc(df_res, modo, pasta="resultados_bc"):
    """
    Mantido para compatibilidade.
    Agora delega para as 4 funções individuais de plot BC.
    """
    if df_res.empty:
        return
    os.makedirs(pasta, exist_ok=True)
    plotar_eqm_relativo(df_res, modo, pasta)
    plotar_diebold_mariano(df_res, modo, pasta)
    plotar_decomposicao_vies(df_res, modo, pasta)
    plotar_rmse_mae(df_res, modo, pasta)


# ─────────────────────────────────────────────────────────────────────────────
# BC-8. Heatmap Tabela 2 — arquivo separado
# ─────────────────────────────────────────────────────────────────────────────
def plotar_tabela2_bc(df_tab, modo, pasta="resultados_bc"):
    os.makedirs(pasta, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    im = ax.imshow(df_tab.values.astype(float), cmap="RdYlGn_r",
                   vmin=0.7, vmax=1.4, aspect="auto")
    plt.colorbar(im, ax=ax, label="EQM relativo (OU/RW)")

    ax.set_xticks(range(len(df_tab.columns)))
    ax.set_xticklabels(df_tab.columns)
    ax.set_yticks(range(len(df_tab.index)))
    ax.set_yticklabels(df_tab.index)

    for i in range(len(df_tab.index)):
        for j in range(len(df_tab.columns)):
            val = df_tab.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black")

    ax.set_title(f"Tabela 2 Replicada — {modo.upper()}\n"
                 f"EQM relativo OU/RW | Verde = OU melhor | Vermelho = RW melhor")
    plt.tight_layout()
    caminho = f"{pasta}/bc_E_tabela2_{modo}.png"
    fig.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  BC Heatmap Tabela 2 salvo: {caminho}")


# ─────────────────────────────────────────────────────────────────────────────
# BC-9. Executor do módulo BC completo
# ─────────────────────────────────────────────────────────────────────────────
def rodar_validacao_bc(df_base, modo="mensal", max_h=12,
                       janela_anos=5, anos_recentes=10):
    print(f"\n{'#'*60}")
    print(f"#  MÓDULO BC — VALIDAÇÃO ESTUDO Nº 25/2018 — {modo.upper()}")
    print(f"{'#'*60}")

    pasta_bc = f"{PASTA_RAIZ_BC}/{modo}"
    os.makedirs(pasta_bc, exist_ok=True)

    df_log = preparar_log(df_base, modo, anos_recentes=anos_recentes)

    df_res = backtest_bc_rolling(df_log, modo=modo, max_h=max_h,
                                 janela_anos=janela_anos)
    if not df_res.empty:
        df_res.to_csv(f"{pasta_bc}/tabela1_{modo}.csv", index=False)
        plotar_eqm_bc(df_res, modo, pasta=pasta_bc)

    df_tab2 = backtest_bc_fator_k(df_log, modo=modo, max_h=max_h,
                                  janela_anos=janela_anos,
                                  percentis_k=(0, 25, 50, 75, 85, 90))
    df_tab2.to_csv(f"{pasta_bc}/tabela2_{modo}.csv")
    plotar_tabela2_bc(df_tab2, modo, pasta=pasta_bc)

    print(f"\n Resultados BC salvos em: {pasta_bc}/")
    return df_res, df_tab2


# ╔══════════════════════════════════════════════════════════════╗
# ║                      EXECUÇÃO                               ║
# ╚══════════════════════════════════════════════════════════════╝

print("\n" + "═"*60)
print(" MODELO OU — CÂMBIO")
print(" EDP: dS_t = θ(μ - S_t)dt + σ dW_t")
print("═"*60)
print(f" Arquivo CSV  : {args.csv}")
print(f" Início filtro: {args.inicio or 'automático'}")
print(f" Fim filtro   : {args.fim or 'automático'}")
print(f" Anos recentes: {args.anos_recentes or 'todos'}")
print(f" BC janela    : {args.bc_janela} anos")
print(f" BC horizonte : {args.bc_horizonte} períodos")
print("═"*60)
print(f"\n  Resultados OU → {PASTA_RAIZ_OU}/")
print(f"  Resultados BC → {PASTA_RAIZ_BC}/")
print()

# Modelos OU
rodar_modelo(df, df_full, "diario")
rodar_modelo(df, df_full, "mensal")

# Validação BC
rodar_validacao_bc(df, modo="mensal",
                   max_h=args.bc_horizonte,
                   janela_anos=args.bc_janela,
                   anos_recentes=10)

rodar_validacao_bc(df, modo="diario",
                   max_h=min(args.bc_horizonte, 21),
                   janela_anos=max(args.bc_janela - 2, 2),
                   anos_recentes=10)

print("\n Concluído!")
print(f" Resultados OU  → {PASTA_RAIZ_OU}/")
print(f" Resultados BC  → {PASTA_RAIZ_BC}/")
print()
print("Estrutura de arquivos gerados:")
print(f"  {PASTA_RAIZ_OU}/diario/")
print("    01_serie_projecao_diario.png   ← Série + MC (tb abriu janela interativa)")
print("    02_residuos_diario.png")
print("    03_qqplot_diario.png")
print("    04_distribuicao_residuos_diario.png")
print("    05_acf_residuos_diario.png")
print("    06_forca_reversao_diario.png")
print("    07_dispersao_delta_diario.png")
print("    OU_global.csv")
print(f"  {PASTA_RAIZ_OU}/mensal/")
print("    01_serie_projecao_mensal.png")
print("    02..07 ... (idem)")
print("    08_mensal_dia1_projecao.png")
print(f"  {PASTA_RAIZ_BC}/mensal/")
print("    bc_A_eqm_relativo_mensal.png")
print("    bc_B_diebold_mariano_mensal.png")
print("    bc_C_decomposicao_theil_mensal.png")
print("    bc_D_rmse_mae_mensal.png")
print("    bc_E_tabela2_mensal.png")
print()
print("Para rodar com período específico:")
print("  python3 cambio_ou_modelo.py --inicio 2015-01-01 --fim 2023-12-31")
print("  python3 cambio_ou_modelo.py --anos_recentes 5")
print("  python3 cambio_ou_modelo.py --bc_janela 3 --bc_horizonte 6")
