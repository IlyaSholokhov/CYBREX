import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def sales_recommendations(sku_df):
    rec = {"recommendations": {
        "sales_trend":{"rec":None, "chart":None},
        "promo_effect":{"rec":None, "chart":None},
        "low_days":{"rec":None, "chart":None},
        "spikes_and_dips":{"rec":None, "chart":None},
        "past_period":{"rec":None, "chart":None},
        "all_rec":{"rec":None, "chart":None},
    }}

    # =================== Фичи ===================
    sku_df["lag_7"] = sku_df["sales"].shift(7)
    sku_df["lag_30"] = sku_df["sales"].shift(30)
    sku_df["rolling_14"] = sku_df["sales"].rolling(14).mean()
    sku_df["rolling_30"] = sku_df["sales"].rolling(30).mean()
    mean_sales = sku_df["sales"].mean()
    latest_sales = sku_df["sales"].iloc[-1]
    rolling_30_latest = sku_df["rolling_30"].iloc[-1]

    # =================== Правила ===================
    # ===============================================================
    # Тренд продаж
    # ===============================================================
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=sku_df["ds"], y=sku_df["rolling_30"], mode="lines", name="30-дневное среднее"))
    fig_trend.add_trace(go.Scatter(x=[sku_df["ds"].iloc[0], sku_df["ds"].iloc[-1]],
                                    y=[sku_df["rolling_30"].iloc[0], sku_df["rolling_30"].iloc[-1]],
                                    mode="lines", line=dict(dash="dash", color="orange"), name="Тренд"))
    fig_trend.update_layout(title="Тренд продаж", height=400)

    trend_slope = (sku_df["rolling_30"].iloc[-1] - sku_df["rolling_30"].iloc[0]) / len(sku_df)
    if trend_slope < 0:
        rec["recommendations"]["sales_trend"]["rec"] = "Тренд продаж падает — усили маркетинг, промо или рекламу."
    elif trend_slope > 0 and latest_sales < mean_sales:
        rec["recommendations"]["sales_trend"]["rec"] = "Продажи растут, но текущий уровень низкий — активизируйте маркетинг или видимость продукта."
    rec["recommendations"]["sales_trend"]["chart"] = fig_trend

    # ===============================================================
    # Промо-эффект
    # ===============================================================
    fig_promo = go.Figure()
    fig_promo.add_trace(go.Box(y=sku_df[sku_df["promo"]==0]["sales"], name="Без промо"))
    fig_promo.add_trace(go.Box(y=sku_df[sku_df["promo"]==1]["sales"], name="С промо"))
    fig_promo.update_layout(title="Эффект промо-акций", height=400)

    promo_effect = sku_df[sku_df["promo"]==1]["sales"].mean() - sku_df[sku_df["promo"]==0]["sales"].mean()
    if promo_effect > 0.1 * mean_sales:
        rec["recommendations"]["promo_effect"]["rec"] = "Промо-акции приносят рост продаж — увеличьте количество промо-дней."
    elif promo_effect < 0:
        rec["recommendations"]["promo_effect"]["rec"] = "Промо-акции неэффективны — пересмотрите формат/размер скидки."
    rec["recommendations"]["promo_effect"]["chart"] = fig_promo

    # ===============================================================
    # Сезонность по дням недели
    # ===============================================================
    dow_mean = sku_df.groupby(sku_df["ds"].dt.dayofweek)["sales"].mean()
    low_days = dow_mean[dow_mean < dow_mean.mean() * 0.9].index.tolist()

    fig_dow = go.Figure()
    fig_dow.add_trace(go.Bar(x=[["Пн","Вт","Ср","Чт","Пт","Сб","Вс"][i] for i in dow_mean.index],
                                y=dow_mean.values))
    fig_dow.update_layout(title="Продажи по дням недели", height=400)
    
    if low_days:
        rec["recommendations"]["low_days"]["rec"] = f"Низкие продажи в дни недели {low_days} — усилить рекламу и акции в эти дни."
    rec["recommendations"]["low_days"]["chart"] = fig_dow

    # ===============================================================
    # Всплески и провалы
    # ===============================================================
    fig_spike = go.Figure()
    fig_spike.add_trace(go.Scatter(x=sku_df["ds"], y=sku_df["sales"], mode="lines", name="Продажи"))
    fig_spike.add_hline(y=rolling_30_latest, line=dict(color="gray", dash="dash"))
    fig_spike.add_hrect(y0=rolling_30_latest*0.7, y1=rolling_30_latest*1.3,
                        fillcolor="lightgreen", opacity=0.2)
    fig_spike.update_layout(title="Всплески и провалы продаж", height=400)

    if latest_sales < rolling_30_latest * 0.7:
        rec["recommendations"]["spikes_and_dips"]["rec"] = "Внезапный провал продаж — проверить поставки, маркетинг, конкурентов."
    elif latest_sales > rolling_30_latest * 1.3:
        rec["recommendations"]["spikes_and_dips"]["rec"] = "Всплеск продаж — проанализируйте причины и повторите успешные действия."
    rec["recommendations"]["spikes_and_dips"]["chart"] = fig_spike

    # ===============================================================
    # Сравнение с прошлым периодом
    # ===============================================================
    fig_compare = make_subplots(rows=1, cols=1)
    fig_compare.add_trace(go.Scatter(x=sku_df["ds"], y=sku_df["sales"], mode="lines", name="Продажи"))
    fig_compare.add_trace(go.Scatter(x=sku_df["ds"], y=sku_df["lag_7"], mode="lines",
                                        name="Неделю назад", line=dict(dash="dot")))
    fig_compare.add_trace(go.Scatter(x=sku_df["ds"], y=sku_df["lag_30"], mode="lines",
                                        name="Месяц назад", line=dict(dash="dash")))
    fig_compare.update_layout(title="Сравнение с прошлыми периодами", height=400)

    if sku_df["lag_7"].iloc[-1] is not None and latest_sales < sku_df["lag_7"].iloc[-1] * 0.9:
        rec["recommendations"]["past_period"]["rec"] = "Продажи ниже прошлой недели — стимулируйте активность через промо или рекламу."
    if sku_df["lag_30"].iloc[-1] is not None and latest_sales < sku_df["lag_30"].iloc[-1] * 0.9:
        rec["recommendations"]["past_period"]["rec"] = "Продажи ниже прошлого месяца — пересмотрите стратегию для SKU."
    rec["recommendations"]["past_period"]["chart"] = fig_compare

    # ===============================================================
    # Общая рекомендация, если нет проблем
    # ===============================================================
    if not rec["recommendations"]:
        rec["recommendations"]["all_rec"]["rec"] = "Продажи стабильны — продолжайте текущую стратегию."

    return rec    


