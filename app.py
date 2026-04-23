import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import io
import os
import datetime
import warnings

# 忽略曲线拟合时的数学警告
warnings.filterwarnings('ignore')

# ================= 1. 访问统计与反馈持久化逻辑 =================
def get_visitor_count():
    count_file = "visitor_count.txt"
    if not os.path.exists(count_file):
        with open(count_file, "w") as f: f.write("100")
    with open(count_file, "r") as f:
        try: count = int(f.read())
        except: count = 100
    return count

def update_visitor_count():
    count = get_visitor_count() + 1
    with open("visitor_count.txt", "w") as f: f.write(str(count))
    return count

def save_feedback(text):
    feedback_file = "feedback_log.csv"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(feedback_file):
        with open(feedback_file, "w", encoding="utf-8-sig") as f:
            f.write("时间,内容\n")
    with open(feedback_file, "a", encoding="utf-8-sig") as f:
        # 替换英文逗号防止破坏 CSV 格式
        safe_text = text.replace(',', '，').replace('\n', ' ')
        f.write(f"{timestamp},{safe_text}\n")

# 防止同一用户刷新页面导致重复计数
if 'visited' not in st.session_state:
    st.session_state.visited = True
    current_visits = update_visitor_count()
else:
    current_visits = get_visitor_count()

# ================= 2. 核心数学与绘图引擎 =================
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
            if df.shape[1] < n_reps + 1:
                st.warning(f"[{group['name']}] 数据列数不足，跳过绘制。")
                continue
                
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
                # 按照最高浓度点提取 Dmax
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
            
            # Dmax < 50% 判定逻辑
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
            
            # 画点/误差棒
            if mean_dmax >= 50.0 and n_reps > 1:
                ax.errorbar(x_mapped, y_mean, yerr=y_sd, fmt=group['marker'], color=group['color'], ecolor=group['color'], elinewidth=1.5, capsize=group['cap'], label=legend_label, zorder=5)
            else:
                ax.scatter(x_mapped, y_mean, color=group['color'], marker=group['marker'], s=60, label=legend_label, zorder=5)

            # 画平滑曲线
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
    
    # 固定纵坐标范围
    ax.set_ylim(-5, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    
    # 工业标准排版风格
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(direction='in', length=6, width=1.5)
    ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5), labelspacing=1.2)
    
    return fig, report_texts

# ================= 3. 网页 UI 构建 =================
st.set_page_config(page_title="Dose-Response Fitter", layout="wide")

# --- 侧边栏：宣传与全局设置 ---
st.sidebar.title("秦冲课题组 (Qin Lab)")
st.sidebar.markdown(f"📊 **累计使用人次**: `{current_visits}`")
st.sidebar.success("📢 **生信、AI交流**\n\n欢迎 +v: **Jingzbdcjl**")
st.sidebar.markdown("---")

st.sidebar.header("⚙️ 全局图表设置")
n_reps = st.sidebar.number_input("实验重复数 (n):", min_value=1, value=3)
num_groups = st.sidebar.number_input("需要几组数据?:", min_value=1, max_value=10, value=3)
ui_unit = st.sidebar.selectbox("浓度单位:", ['nM', 'uM', 'pM', 'mM'])
ui_dpi = st.sidebar.selectbox("图表清晰度 (DPI):", [150, 300, 600])
ui_show_params = st.sidebar.checkbox("在图例中显示参数", value=False)

st.sidebar.markdown("---")
ui_title = st.sidebar.text_input("图表标题:", "Degradation Curve")
ui_xlabel = st.sidebar.text_input("X轴名称:", "Concentration")
ui_ylabel = st.sidebar.text_input("Y轴名称:", "Degradation Ratio (%)")

# --- 主页面：标题与数据输入 ---
st.title("🔬 药物剂量-效应曲线在线拟合工具")
st.markdown("支持降解剂 ($DC_{50}$) / 抑制剂 ($IC_{50}$) 的多参数 4PL 拟合。")

COLORS = {'深蓝色': '#1f77b4', '深红色': '#d62728', '森林绿': '#2ca02c', '紫色': '#9467bd', '橙色': '#ff7f0e', '黑色': 'black'}
MARKERS = {'圆点 (●)': 'o', '方块 (■)': 's', '正三角 (▲)': '^', '倒三角 (▼)': 'v', '菱形 (◆)': 'D', '叉号 (×)': 'x'}

tabs = st.tabs([f"数据组 {i+1}" for i in range(num_groups)])
groups_data = []

default_data = "0.01\t0\t0\t0\n1\t20.01\t27.28\t16.27\n10\t31.67\t30.58\t27.72\n100\t61.83\t59.60\t41.46\n1000\t66.34\t62.44\t45.98\n10000\t63.18\t64.93\t52.60"

for i in range(num_groups):
    with tabs[i]:
        col1, col2, col3, col4 = st.columns(4)
        g_enable = col1.checkbox("启用此组", value=True, key=f"en_{i}")
        g_name = col2.text_input("化合物名称:", f"Compound {i+1}", key=f"name_{i}")
        g_mode = col3.selectbox("数据类型:", ['降解率 (Degradation)', '保留率 (Retention)'], key=f"mode_{i}")
        g_color = col4.selectbox("散点颜色:", list(COLORS.keys()), index=i%len(COLORS), key=f"color_{i}")
        
        col1_2, col2_2, col3_2 = st.columns([1, 1, 2])
        g_marker = col1_2.selectbox("散点形状:", list(MARKERS.keys()), index=i%len(MARKERS), key=f"marker_{i}")
        g_cap = col2_2.number_input("误差棒短线长:", value=3.0, key=f"cap_{i}")
        
        g_raw = st.text_area("从 Excel 粘贴数据 (制表符/空格分隔，第1列浓度，后续为比例/百分比):", 
                             value=default_data if i==0 else "", height=180, key=f"raw_{i}")
        
        groups_data.append({
            'enabled': g_enable, 'name': g_name, 'data_mode': g_mode,
            'color': COLORS[g_color], 'marker': MARKERS[g_marker], 'cap': g_cap, 'raw_text': g_raw
        })

# --- 绘图生成与报告展示 ---
if st.button("🚀 生成图表与分析报告", use_container_width=True, type="primary"):
    with st.spinner("正在执行 4PL 曲线拟合..."):
        fig, reports = process_and_plot(ui_title, ui_xlabel, ui_ylabel, ui_unit, n_reps, groups_data, ui_dpi, ui_show_params)
        
        col_chart, col_report = st.columns([2, 1])
        with col_chart:
            st.pyplot(fig)
            
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=ui_dpi, bbox_inches='tight')
            st.download_button(label="📥 下载高清 PNG 图片", data=buf.getvalue(), file_name=f"Dose_Response_{datetime.datetime.now().strftime('%H%M%S')}.png", mime="image/png")
            
        with col_report:
            st.success("分析完成！")
            st.markdown("### 📊 参数报告")
            for text in reports:
                st.markdown(text)

st.markdown("---")

# --- 底部：说明文档与反馈 ---
# ================= 4. 底部：说明文档与互动论坛 =================
st.markdown("---")

# 定义管理员密码 (请自行修改)
ADMIN_PASSWORD = "5225249"
FORUM_FILE = "forum_messages.csv"

# 初始化论坛文件
def init_forum():
    if not os.path.exists(FORUM_FILE):
        df = pd.DataFrame(columns=["时间", "昵称", "留言内容"])
        df.to_csv(FORUM_FILE, index=False, encoding="utf-8-sig")

init_forum()

col_info, col_forum = st.columns([1, 1])

with col_info:
    # st.markdown("### 📖 工具说明")
    st.markdown("""
    ### 🛠 功能说明
    1. **数据输入**：支持从 Excel 复制粘贴多列数据。第一列为浓度，后续列为对应的重复实验值。小数(如0.95)或百分比(95)系统会自动兼容。
    2. **DC50 计算逻辑**：
        - 程序会独立对各列重复实验进行非线性拟合，最终对结果取均值并计算标准差 (SD)。
        - 智能判定：若 $D_{max} < 50\%$，认定目标药物效能过低，直接输出 $DC_{50}$  `> 最大给药浓度`。
    3. **Dmax 计算逻辑**：$D_{max}$ 严格提取最高浓度点处的观测均值，如PROTAC的hook效应，请注意分析。
    4. **坐标系与 0 浓度映射**：对数 (Log) 坐标系中 0 是不存在的。本工具会自动寻找极小非零浓度，将溶媒对照组 (0 浓度) 映射至其左侧以便直观显示。
    5. **呈现效果**：可以多组数据做在同一张图上，例如多个化合物（调整所需数据组数量）在同一张图上体现。
    6. **未完待续**：不同效果可自行体验，如有建议，请留言。
    ### ⚖️ 免责声明
    本工具采用 Python 生态标准算法库构建，仅供科学研究与学术交流使用。重要结论请结合实验原始图谱人工复核。
        
    *系统架构: Python/Streamlit | 开发者: Zzy522* 
    """)
    
    # 将管理员专区放在左侧下方的一个折叠面板里，隐藏得更深一些
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🛡️ 管理员后台专区", expanded=False):
        pwd = st.text_input("请输入管理员密码解锁：", type="password")
        if pwd == ADMIN_PASSWORD:
            st.success("身份验证成功！您现在可以管理所有留言。")
            
            # 读取当前数据
            df_forum = pd.read_csv(FORUM_FILE)
            
            # 开启数据编辑器 (允许删除行)
            st.markdown("提示：选中表格左侧的复选框，按 `Delete` 键即可删除不良留言。")
            edited_df = st.data_editor(df_forum, num_rows="dynamic", use_container_width=True)
            
            col_save, col_export = st.columns(2)
            with col_save:
                if st.button("💾 保存修改 / 删除", type="primary"):
                    edited_df.to_csv(FORUM_FILE, index=False, encoding="utf-8-sig")
                    st.success("后台数据已同步更新！")
                    st.rerun() # 刷新页面展示最新状态
            with col_export:
                with open(FORUM_FILE, "rb") as file:
                    st.download_button("📥 导出全部留言 (CSV)", data=file, file_name="forum_backup.csv", mime="text/csv")
        elif pwd != "":
            st.error("密码错误！")

with col_forum:
    st.markdown("### 💬 提问交流与建议反馈")
    
    # --- 展示历史留言 ---
    df_msg = pd.read_csv(FORUM_FILE)
    
    # 设定一个固定高度的滚动容器展示留言，避免留言太多把网页撑得太长
    msg_container = st.container(height=300)
    with msg_container:
        if len(df_msg) == 0:
            st.info("暂无留言，快来抢沙发吧！")
        else:
            # 倒序遍历，让最新的留言显示在最上面
            for index, row in df_msg.iloc[::-1].iterrows():
                st.markdown(f"**👤 {row['昵称']}** `<span style='color:gray;font-size:0.8em;'>{row['时间']}</span>`", unsafe_allow_html=True)
                st.markdown(f"> {row['留言内容']}")
                st.markdown("---")
    
    # --- 发表新留言 ---
    with st.form("post_msg_form", clear_on_submit=True):
        col_name, _ = st.columns([1, 1])
        with col_name:
            user_name = st.text_input("您的昵称 (选填):", value="热心科研狗")
        
        user_msg = st.text_area("留言内容:", placeholder="提问、Bug反馈或新功能建议...")
        submitted = st.form_submit_button("发送留言", type="primary")
        
        if submitted:
            if user_msg.strip() == "":
                st.warning("留言内容不能为空哦！")
            else:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                # 追加到 DataFrame 并保存
                new_row = pd.DataFrame([{"时间": timestamp, "昵称": user_name, "留言内容": user_msg.replace('\n', ' ')}])
                df_updated = pd.concat([df_msg, new_row], ignore_index=True)
                df_updated.to_csv(FORUM_FILE, index=False, encoding="utf-8-sig")
                
                st.toast("留言发表成功！", icon="🎉")
                st.rerun() # 立即刷新页面，让用户看到自己的留言
