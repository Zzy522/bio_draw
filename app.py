import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import io
import warnings

warnings.filterwarnings('ignore')

# --- 核心计算函数 ---
def four_pl_model(x, bottom, top, ec50, hill_slope):
    x = np.maximum(x, 1e-10) 
    return bottom + (top - bottom) / (1 + (ec50 / x)**hill_slope)

def process_and_plot(title, xlabel, ylabel, unit, n_reps, groups_data, dpi_val, show_params):
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=dpi_val)
    plt.subplots_adjust(right=0.65) 
    
    all_x_values = []
    report_texts = []
    
    for group in groups_data:
        if not group['enabled'] or not group['raw_text'].strip():
            continue
            
        try:
            df = pd.read_csv(io.StringIO(group['raw_text'].strip()), sep=r'\s+', header=None)
            x_raw = df.iloc[:, 0].astype(float).values
            y_replicates = df.iloc[:, 1:n_reps+1].astype(float).values
            
            if np.max(y_replicates) <= 2.0: y_replicates = y_replicates * 100
            if group['data_mode'] == '保留率 (Retention)': y_replicates = 100 - y_replicates

            non_zero_x = x_raw[x_raw > 0]
            if len(non_zero_x) == 0: continue
            
            min_non_zero = np.min(non_zero_x)
            x_mapped = x_raw.copy()
            x_mapped[x_raw == 0] = min_non_zero / 10 
            all_x_values.extend(x_mapped)
            
            max_conc = np.max(x_raw)
            dc50_list, dmax_list = [], []
            y_mean = np.mean(y_replicates, axis=1)
            y_sd = np.std(y_replicates, axis=1, ddof=1) if n_reps > 1 else np.zeros_like(y_mean)

            for i in range(n_reps):
                y_rep = y_replicates[:, i]
                idx_max_conc = np.argmax(x_mapped)
                dmax_list.append(y_rep[idx_max_conc]) 
                
                p0 = [np.min(y_rep), np.max(y_rep), np.median(x_mapped), 1.0]
                try:
                    popt, _ = curve_fit(four_pl_model, x_mapped, y_rep, p0=p0, maxfev=10000)
                    dc50_list.append(popt[2])
                except RuntimeError:
                    pass 
                    
            mean_dmax = np.mean(dmax_list)
            sd_dmax = np.std(dmax_list, ddof=1) if len(dmax_list) > 1 else 0
            
            dmax_str = f"{mean_dmax:.2f}±{sd_dmax:.2f}%" if len(dmax_list) > 1 else f"{mean_dmax:.2f}%"
            
            if mean_dmax < 50.0:
                dc50_str = f"> {max_conc} {unit}"
            else:
                if len(dc50_list) > 0:
                    mean_dc50 = np.mean(dc50_list)
                    sd_dc50 = np.std(dc50_list, ddof=1) if len(dc50_list) > 1 else 0
                    dc50_str = f"{mean_dc50:.2f}±{sd_dc50:.2f} {unit}" if len(dc50_list) > 1 else f"{mean_dc50:.2f} {unit}"
                else:
                    dc50_str = "拟合失败"
            
            report_texts.append(f"**{group['name']}**: DC50 = {dc50_str} ; Dmax = {dmax_str}")
            
            legend_label = f"{group['name']}\n$DC_{{50}}$={dc50_str} ; $D_{{max}}$={dmax_str}" if show_params else group['name']
            
            if mean_dmax >= 50.0 and n_reps > 1:
                ax.errorbar(x_mapped, y_mean, yerr=y_sd, fmt=group['marker'], color=group['color'], ecolor=group['color'], elinewidth=1.5, capsize=group['cap'], label=legend_label, zorder=5)
            else:
                ax.scatter(x_mapped, y_mean, color=group['color'], marker=group['marker'], s=60, label=legend_label, zorder=5)

            try:
                popt_mean, _ = curve_fit(four_pl_model, x_mapped, y_mean, p0=[np.min(y_mean), np.max(y_mean), np.median(x_mapped), 1.0], maxfev=10000)
                x_smooth = np.logspace(np.log10(np.min(x_mapped)), np.log10(np.max(x_mapped)), 200)
                y_smooth = four_pl_model(x_smooth, *popt_mean)
                ax.plot(x_smooth, y_smooth, color=group['color'], linewidth=2, zorder=4)
            except: pass
                
        except Exception as e:
            st.error(f"[{group['name']}] 处理报错: {str(e)}")

    if len(all_x_values) > 0:
        ax.set_xscale('log')
        unique_x_mapped = sorted(list(set(all_x_values)))
        tick_labels = [str(np.round(val, 6)).rstrip('0').rstrip('.') for val in unique_x_mapped]
        tick_labels[0] = '0' 
        ax.set_xticks(unique_x_mapped)
        ax.set_xticklabels(tick_labels)

    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel(f"{xlabel} ({unit})", fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_ylim(-5, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(direction='in', length=6, width=1.5)
    ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5), labelspacing=1.2)
    
    return fig, report_texts

# --- 网页 UI 构建 ---
st.set_page_config(page_title="Dose-Response Fitter", layout="wide")
st.title("🔬 药物剂量-效应曲线在线拟合工具")
st.markdown("支持降解剂 ($DC_{50}$) / 抑制剂 ($IC_{50}$) 的多参数 4PL 拟合，由 Streamlit 强力驱动。")

# 全局设置区域
st.sidebar.header("⚙️ 全局设置")
n_reps = st.sidebar.number_input("实验重复数 (n):", min_value=1, value=3)
num_groups = st.sidebar.number_input("数据组数:", min_value=1, max_value=10, value=3)
ui_unit = st.sidebar.selectbox("浓度单位:", ['nM', 'uM', 'pM', 'mM'])
ui_dpi = st.sidebar.selectbox("图表清晰度 (DPI):", [150, 300, 600])
ui_show_params = st.sidebar.checkbox("图例中显示参数", value=False)

st.sidebar.markdown("---")
ui_title = st.sidebar.text_input("图表标题:", "Degradation Curve")
ui_xlabel = st.sidebar.text_input("X轴名称:", "Concentration")
ui_ylabel = st.sidebar.text_input("Y轴名称:", "Degradation Ratio (%)")

# 颜色和形状映射表
COLORS = {'深蓝色': '#1f77b4', '深红色': '#d62728', '森林绿': '#2ca02c', '紫色': '#9467bd', '橙色': '#ff7f0e', '黑色': 'black'}
MARKERS = {'圆点 (●)': 'o', '方块 (■)': 's', '正三角 (▲)': '^', '倒三角 (▼)': 'v'}

# 数据输入区域
tabs = st.tabs([f"数据组 {i+1}" for i in range(num_groups)])
groups_data = []

default_data = "0.1\t0.095\t0.101\t0.334\n1\t0.116\t0.130\t0.396\n10\t0.140\t0.164\t0.478\n100\t0.183\t0.204\t0.643\n1000\t0.367\t0.376\t0.855\n10000\t0.698\t0.667\t0.922\n100000\t1.002\t0.998\t0.996"

for i in range(num_groups):
    with tabs[i]:
        col1, col2, col3, col4 = st.columns(4)
        g_enable = col1.checkbox("启用此组", value=True, key=f"en_{i}")
        g_name = col2.text_input("化合物名称:", f"Compound {i+1}", key=f"name_{i}")
        g_mode = col3.selectbox("数据类型:", ['降解率 (Degradation)', '保留率 (Retention)'], key=f"mode_{i}")
        g_color = col4.selectbox("散点颜色:", list(COLORS.keys()), index=i%len(COLORS), key=f"color_{i}")
        
        col1_2, col2_2, col3_2 = st.columns(3)
        g_marker = col1_2.selectbox("散点形状:", list(MARKERS.keys()), index=i%len(MARKERS), key=f"marker_{i}")
        g_cap = col2_2.number_input("误差棒短线长:", value=3.0, key=f"cap_{i}")
        
        g_raw = st.text_area("从 Excel 粘贴数据 (第1列浓度，后续为比例/百分比):", value=default_data if i==0 else "", height=150, key=f"raw_{i}")
        
        groups_data.append({
            'enabled': g_enable, 'name': g_name, 'data_mode': g_mode,
            'color': COLORS[g_color], 'marker': MARKERS[g_marker], 'cap': g_cap, 'raw_text': g_raw
        })

# 绘图生成
if st.button("🚀 生成图表与分析报告", use_container_width=True, type="primary"):
    with st.spinner("正在进行 4PL 曲线拟合..."):
        fig, reports = process_and_plot(ui_title, ui_xlabel, ui_ylabel, ui_unit, n_reps, groups_data, ui_dpi, ui_show_params)
        
        # 将图和报告分为左右两列显示
        col_chart, col_report = st.columns([2, 1])
        
        with col_chart:
            st.pyplot(fig)
            
            # 提供高清图片下载
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=ui_dpi, bbox_inches='tight')
            st.download_button(label="📥 下载高清 PNG 图片", data=buf.getvalue(), file_name=f"{ui_title}.png", mime="image/png")
            
        with col_report:
            st.success("分析完成！")
            st.markdown("### 📊 参数报告")
            for text in reports:
                st.markdown(text)