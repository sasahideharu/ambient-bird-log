import base64
import streamlit as st
import pandas as pd
import os
import io
import requests
import re
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from pydub import AudioSegment
import uuid
import math
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont
import numpy as np           # 🔥 これを追加
import matplotlib.pyplot as plt # 🔥 これを追加

# --- 1. 環境変数とSupabase接続 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, "../.env"))

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
BUCKET_NAME = "bird-wav"

if not url or not key:
    st.error("⚠️ .envファイルからSupabaseのAPIキーが見つからないぜ。")
    st.stop()

supabase: Client = create_client(url, key)

from matplotlib.figure import Figure # 🔥 これがスレッドセーフの鍵

# --- 2. リモート音声の取得と切り出し (完全スレッドセーフ＆バイナリ版) ---
@st.cache_data(show_spinner=False)
def get_sliced_remote_audio(file_url, original_start, original_end):
    response = requests.get(file_url)
    if response.status_code != 200:
        raise Exception("ファイルのダウンロードに失敗したぜ。")
        
    audio = AudioSegment.from_file(io.BytesIO(response.content))
    channels = audio.channels
    
    start_ms = max(0, int((original_start - 1.5) * 1000))
    end_ms = min(len(audio), int((original_end + 1.5) * 1000))
    
    sliced_audio = audio[start_ms:end_ms]
    
    out_io = io.BytesIO()
    sliced_audio.export(out_io, format="mp3", bitrate="192k")
    
    samples = np.array(sliced_audio.get_array_of_samples())
    if channels == 2:
        samples = samples.reshape((-1, 2))
        samples = samples[:, 0]
        
    # 🔥 GEʍlNEʍ Hack: pyplotを使わず、独立したFigureオブジェクトを生成（競合回避）
    fig = Figure(figsize=(5, 1.5))
    ax = fig.add_subplot(111)
    
    # 🔥 GEʍlNEʍ Hack: カラーマップを深淵な 'inferno' に変更し、オブジェクトを受け取る
    Pxx, freqs, bins, im = ax.specgram(samples, Fs=audio.frame_rate, cmap='inferno', NFFT=1024, noverlap=512)
    
    # 🔥 鳴き声の最大音量(ピーク)を基準に、そこから下50dBだけを描画（ノイズフロアを黒に沈めるダイナミクス処理）
    max_db = 10 * np.log10(Pxx.max())
    im.set_clim(vmin=max_db - 50, vmax=max_db + 5)
    
    ax.set_ylim(0, 12000)
    
    for freq in range(2000, 12000, 2000):
        ax.plot([0.98, 1.0], [freq, freq], color='white', alpha=0.6, linewidth=0.8, transform=ax.get_yaxis_transform())
        ax.text(0.97, freq, f'{freq//1000}kHz', color='white', alpha=0.7, fontsize=5, ha='right', va='center', transform=ax.get_yaxis_transform())
    
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    img_io = io.BytesIO()
    fig.savefig(img_io, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    
    # 🔥 GEʍlNEʍ Hack: BytesIOオブジェクトではなく、.getvalue()で純粋なバイナリ(bytes)を取り出して返す
    return out_io.getvalue(), start_ms / 1000.0, end_ms / 1000.0, channels, img_io.getvalue()
    
    
# --- 2.5. 日付情報の自動抽出と曜日の計算 ---
def extract_date(filename):
    match = re.match(r'^(\d{6})', filename)
    if match:
        try:
            dt = datetime.strptime(match.group(1), '%y%m%d')
            weekdays = ['月', '火', '水', '木', '金', '土', '日']
            weekday_str = weekdays[dt.weekday()]
            return f"{dt.strftime('%Y/%m/%d')} ({weekday_str})"
        except ValueError:
            return "不明な日付"
    return "不明な日付"

# 🔥 GEʍlNEʍ Hack: 連続する3秒の解析結果を1つの長いトラックに結合する
def merge_consecutive_detections(df, max_gap=3.0):
    # 必要なカラムでソート（ファイル > 鳥の種類 > 開始時間）
    df = df.sort_values(['wav_filename', 'common_name', 'start_sec'])
    
    # 直前のデータとの「終了時間」と「開始時間」の差（ギャップ）を計算
    df['prev_end'] = df.groupby(['wav_filename', 'common_name'])['end_sec'].shift(1)
    df['gap'] = df['start_sec'] - df['prev_end']
    
    # ギャップが指定秒数(3秒)より大きい、または最初のデータの場合は「新しい鳴き声グループ(True)」とする
    df['new_group'] = (df['gap'].isnull()) | (df['gap'] > max_gap)
    
    # 累積和を使って、連続している行に同じグループIDを割り振る
    df['group_id'] = df.groupby(['wav_filename', 'common_name'])['new_group'].cumsum()
    
    # グループごとにデータを集計（結合）
    merged_df = df.groupby(['wav_filename', 'common_name', 'group_id']).agg({
        'start_sec': 'min',         # 開始時間は一番早いものを採用
        'end_sec': 'max',           # 終了時間は一番遅いものを採用
        'confidence': 'max',        # 信頼度はその中で一番高かったものを採用
        'scientific_name': 'first', # 🔥 ここを追加！学名のロストを防ぐ
        'location_name': 'first',
        'latitude': 'first',
        'longitude': 'first',
        'record_date': 'first'
    }).reset_index()
    
    return merged_df
    
# 🔥 GEʍlNEʍ Hack: ファイル名から録音開始時間を抽出 (例: _1322.mp3 -> 13:22)
def extract_recording_time(filename):
    match = re.search(r'_(\d{2})(\d{2})\.mp3$', filename)
    if match:
        return f"{match.group(1)}:{match.group(2)}"
    return None

# 🔥 ファイル名の_hhmmから「収録時刻 hh時mm分」表示用テキストを作る
def format_recording_time_label(filename):
    rec_time = extract_recording_time(filename)
    if rec_time:
        hh, mm = rec_time.split(":")
        return f"🕒 収録時刻 {hh}時{mm}分"
    return ""

# --- 2.8. 画面遷移のコントロール（Session State & Query Params） ---
# アプリを開いた瞬間に、URLにパラメータがあれば読み込む
if 'page' not in st.session_state: 
    st.session_state.page = st.query_params.get('page', 'main')
if 'selected_date' not in st.session_state: 
    st.session_state.selected_date = st.query_params.get('date', None)
if 'selected_bird' not in st.session_state: 
    st.session_state.selected_bird = st.query_params.get('bird', None)
if 'selected_loc' not in st.session_state: 
    st.session_state.selected_loc = st.query_params.get('loc', None)
if 'loaded_audio' not in st.session_state: 
    st.session_state.loaded_audio = set()

def load_audio_clip(audio_key):
    st.session_state.loaded_audio.add(audio_key)

# ページ遷移時、セッションステートと一緒にブラウザのURLも書き換える
def go_to_date_detail(date_str):
    st.session_state.page = 'date_detail'
    st.session_state.selected_date = date_str
    st.query_params["page"] = "date_detail"
    st.query_params["date"] = date_str

def go_to_bird_detail(bird_name):
    st.session_state.page = 'bird_detail'
    st.session_state.selected_bird = bird_name
    st.query_params["page"] = "bird_detail"
    st.query_params["bird"] = bird_name

def go_to_loc_detail(loc_name):
    st.session_state.page = 'loc_detail'
    st.session_state.selected_loc = loc_name
    st.query_params["page"] = "loc_detail"
    st.query_params["loc"] = loc_name

def go_to_main_bird():
    st.session_state.page = 'main'
    st.session_state.active_main_tab = 0
    st.session_state.selected_date = None
    st.session_state.selected_bird = None
    st.session_state.selected_loc = None
    st.query_params.clear()

def go_to_main_loc():
    st.session_state.page = 'main'
    st.session_state.active_main_tab = 1
    st.session_state.selected_date = None
    st.session_state.selected_bird = None
    st.session_state.selected_loc = None
    st.query_params.clear()

# --- 3. メインUI ---
st.markdown("<h2 style='font-family: \"Yusei Magic\", sans-serif; font-size: 26px; font-weight: bold; padding-top: 10px; text-align: center; color: #3A3A3A;'>🎧 Ambient Bird Log 🐦</h2>", unsafe_allow_html=True)

# 🔥 GEʍlNEʍ's CSS Hack (トップ余白の最適化版 + フォントの折り返し禁止)
st.markdown("""
    <style>
    /* 🔥 新デザイン: Googleフォントの読み込み */
    @import url('https://fonts.googleapis.com/css2?family=Yusei+Magic&family=Zen+Maru+Gothic:wght@500;700;900&display=swap');

    /* 🔥 新デザイン: アプリ全体のフォントと配色を淡いウグイス色系に統一 */
    .stApp {
        background-color: #F4F2EC;
        font-family: 'Zen Maru Gothic', sans-serif;
    }
    .stApp, .stApp p, .stApp span, .stApp label {
        color: #3A3A3A;
    }

    /* --- 🔥 NEW: ネイティブヘッダーを避けつつ余白を削る黄金比 --- */
    .block-container {
        padding-top: 3.5rem !important; 
        padding-bottom: 1rem !important;
    }
    
    /* 既存のレイアウト調整（レスポンシブ対応のみ） */
    div[data-testid="stVerticalBlock"] { gap: 0rem; }
    img { 
        border-radius: 6px; 
        object-fit: cover !important; 
        aspect-ratio: 1 / 1 !important; 
        width: 100% !important; 
        margin-bottom: 4px !important;
    }
    
    /* 🔥 GEʍlNEʍ Hack: ボタンの文字を強制的に1行にする */
    button p { 
        font-size: 12px !important; 
        white-space: nowrap !important;      /* 折り返し禁止 */
        overflow: hidden !important;         /* はみ出しを隠す */
        text-overflow: ellipsis !important;  /* 限界を超えたら...にする */
    }

    /* 🔥 新デザイン: カード・ボタンの角丸とアクセントカラー */
    .stButton button, .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        border-radius: 12px !important;
        border-color: #E5E0D2 !important;
    }
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #B4CF9E !important;
        border-color: #B4CF9E !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #6F8F5E !important;
        border-bottom-color: #6F8F5E !important;
    }
    
    @media (max-width: 640px) {
        div[data-testid="stHorizontalBlock"] { flex-direction: row !important; flex-wrap: nowrap !important; }
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] { min-width: 0 !important; padding: 0 3px !important; }
        button p { font-size: 10px !important; } /* スマホは10pxで固定。十分に読めるぜ */
    }
    </style>
""", unsafe_allow_html=True)

try:
    response_all = supabase.table("detections").select("*").limit(10000).execute()
    response_master = supabase.table("bird_master").select("*").execute()
    bird_images = {row['common_name']: row['image_url'] for row in response_master.data} if response_master.data else {}
    
    if response_all.data:
        raw_df = pd.DataFrame(response_all.data)
        raw_df['record_date'] = raw_df['wav_filename'].apply(extract_date)
        
        # 🔥 GEʍlNEʍ Hack: ここで細切れの3秒データを結合！
        df_all = merge_consecutive_detections(raw_df, max_gap=3.0)
        
        if st.session_state.page == 'main':
            bird_names = sorted(df_all['common_name'].dropna().unique().tolist())
            is_admin = st.query_params.get("admin") == "true"
            
            if is_admin:
                tab_bird, tab_location, tab_admin, tab_data = st.tabs(["🐦 鳥から探す", "📍 場所から探す", "⚙️ 画像管理", "📁 データ登録"])
            else:
                tab_bird, tab_location = st.tabs(["🐦 鳥から探す", "📍 場所から探す"])

            # --- 🔥 GEʍlNEʍ Hack: JSで裏側からタブを強制クリック ---
            if 'active_main_tab' in st.session_state:
                if st.session_state.active_main_tab == 1:
                    components.html("""
                        <script>
                        // iframeの中から親要素(StreamlitのDOM)にアクセスし、2番目のタブをクリック
                        const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
                        if (tabs.length > 1) {
                            tabs[1].click();
                        }
                        </script>
                    """, height=0)
                # ループ実行を防ぐためにステートから削除
                st.session_state.pop('active_main_tab')

            with tab_bird:
                min_confidence = st.slider("信頼度", min_value=0, max_value=100, value=60, format="%d%%")
                st.markdown("<div style='min-height: 40px;'></div>", unsafe_allow_html=True)
                search_query = st.text_input("検索窓", label_visibility="collapsed", placeholder="和名で検索")
                st.markdown("<div style='min-height: 20px;'></div>", unsafe_allow_html=True)
                
                df_filtered = df_all[df_all['confidence'] >= (min_confidence / 100.0)]
                bird_counts = df_filtered['common_name'].value_counts()
                filtered_birds = {name: count for name, count in bird_counts.items() if not search_query or search_query in name}
                
                # 🔥 GEʍlNEʍ Hack: 鳥一覧のページネーション（表示件数制限でUI描画を爆速化）
                bird_count_key = "display_count_main_bird"
                if bird_count_key not in st.session_state:
                    st.session_state[bird_count_key] = 21  # 初期は21件(3列x7行)を上限とする
                
                # 辞書をリスト化して、上限数だけスライス
                filtered_birds_list = list(filtered_birds.items())
                birds_to_show = filtered_birds_list[:st.session_state[bird_count_key]]
                
                cols = st.columns(3)
                for i, (bird_name, count) in enumerate(birds_to_show):
                    col_idx = i % 3
                    with cols[col_idx]:
                        with st.container():
                            img_url = bird_images.get(bird_name)
                            
                            # 🔥 GEʍlNEʍ Hack: st.imageの代わりに <a> タグを使って画像自体をリンク化する
                            if img_url:
                                st.markdown(f"""
                                    <a href="?page=bird_detail&bird={bird_name}" target="_self" style="display: block; margin-bottom: 4px;">
                                        <img src="{img_url}" style="width: 100%; border-radius: 6px; aspect-ratio: 1/1; object-fit: cover;">
                                    </a>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                    <a href="?page=bird_detail&bird={bird_name}" target="_self" style="display: block; margin-bottom: 4px; text-decoration: none;">
                                        <div style='background-color:#E5E0D2; border-radius:6px; width: 100%; aspect-ratio: 1/1; display:flex; align-items:center; justify-content:center;'>
                                            <span style='color:#9A9A8A; font-size:10px;'>No Img</span>
                                        </div>
                                    </a>
                                """, unsafe_allow_html=True)
                                
                            # 既存のボタン（文字をタップした時の導線もキープ）
                            if st.button(f"{bird_name}", key=f"btn_bird_{bird_name}", use_container_width=True):
                                go_to_bird_detail(bird_name)
                                st.rerun()
                        st.markdown("<div style='min-height: 20px;'></div>", unsafe_allow_html=True)
                
                # 🔥 さらに読み込むボタン
                if st.session_state[bird_count_key] < len(filtered_birds_list):
                    if st.button("🔽 さらに21件読み込む", use_container_width=True, key="btn_load_more_main_bird"):
                        st.session_state[bird_count_key] += 21
                        st.rerun()

            with tab_location:
                st.markdown("### 🗺️ 場所から探す")
                df_loc = df_all.dropna(subset=['latitude', 'longitude'])
                if not df_loc.empty:
                    
                    # 1. 選択肢の先頭に「すべての場所」をデフォルトとして追加
                    unique_locations = ["すべての場所"] + sorted(df_loc['location_name'].unique().tolist())

                    # 2. ステート管理（選択中の場所と、表示件数）
                    loc_count_key = "display_count_loc_tab"
                    loc_state_key = "selected_loc_tab"
                    
                    if loc_count_key not in st.session_state:
                        st.session_state[loc_count_key] = 10
                    if loc_state_key not in st.session_state:
                        st.session_state[loc_state_key] = "すべての場所"

                    # ドロップダウンリストを先に描画
                    selected_loc = st.selectbox("場所を選択してくれ:", unique_locations, index=unique_locations.index(st.session_state[loc_state_key]), label_visibility="collapsed")
                    
                    # 🔥 GEʍlNEʍ Hack: 中身(&nbsp;)を入れて空Divの消滅を防ぎ、確実に30pxの隙間を確保する完全版
                    st.markdown("<div style='font-size: 0px; padding-top: 30px;'>&nbsp;</div>", unsafe_allow_html=True)
                    
                    # 場所が切り替わったら表示件数を10件にリセットして再描画
                    if selected_loc != st.session_state[loc_state_key]:
                        st.session_state[loc_state_key] = selected_loc
                        st.session_state[loc_count_key] = 10
                        st.rerun()

                    # 3. データのフィルタリング
                    if selected_loc == "すべての場所":
                        loc_filtered = df_loc
                    else:
                        loc_filtered = df_loc[df_loc['location_name'] == selected_loc]

                    # 🔥 GEʍlNEʍ Hack: フィルタリングされたデータ(loc_filtered)だけを使ってマップを描画
                    if not loc_filtered.empty:
                        df_map = loc_filtered[['latitude', 'longitude', 'location_name']].drop_duplicates(subset=['latitude', 'longitude']).copy()
                        
                        lat_min, lat_max = df_map['latitude'].min(), df_map['latitude'].max()
                        lon_min, lon_max = df_map['longitude'].min(), df_map['longitude'].max()
                        
                        # 余白を計算 (1ピンだけでもダミーポイントのおかげで5kmスケールが保たれる)
                        lat_pad = max((lat_max - lat_min) * 0.8, 0.05)
                        lon_pad = max((lon_max - lon_min) * 0.8, 0.05)
                        
                        df_map['dot_color'] = '#6F8F5E'
                        df_map['dot_size'] = 150
                        
                        dummy_data = pd.DataFrame({
                            'latitude': [lat_min - lat_pad, lat_max + lat_pad],
                            'longitude': [lon_min - lon_pad, lon_max + lon_pad],
                            'location_name': ['dummy', 'dummy'],
                            'dot_color': ['#6F8F5E05', '#6F8F5E05'],
                            'dot_size': [1, 1]
                        })
                        
                        df_map_padded = pd.concat([df_map, dummy_data], ignore_index=True)
                        st.map(df_map_padded, latitude='latitude', longitude='longitude', color='dot_color', size='dot_size', height=250)
                    
                    st.divider()
                    st.markdown("**📍 過去の記録（場所別）**")
                        
                    if not loc_filtered.empty:
                        # 4. 日付と場所でグループ化し、件数を集計して降順にソート
                        summary_df = loc_filtered.groupby(['record_date', 'location_name']).size().reset_index(name='count')
                        summary_df = summary_df.sort_values(by=['record_date', 'location_name'], ascending=[False, True])
                        
                        # 10件ずつスライスして表示
                        current_limit = st.session_state[loc_count_key]
                        summary_df_to_show = summary_df.head(current_limit)
                        
                        for index, row in summary_df_to_show.iterrows():
                            date_str = row['record_date']
                            loc_name = row['location_name']
                            count = row['count']
                            
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"### 📅 {date_str}")
                                    st.caption(f"📍 **{loc_name}** ｜ 🎧 検出: {count} 件")
                                with col2:
                                    st.button("詳細", on_click=go_to_date_detail, args=(date_str,), key=f"btn_loc_tab_{loc_name}_{date_str}_{index}")
                            st.divider()
                            
                        # 5. 10件追加ロードボタン
                        if current_limit < len(summary_df):
                            if st.button("🔽 さらに10件読み込む", use_container_width=True, key="btn_load_more_loc_tab"):
                                st.session_state[loc_count_key] += 10
                                st.rerun()
                    else:
                        st.info("データが見つからないぜ。")

            if is_admin:
                with tab_admin:
                    st.markdown("### 📷 野鳥画像のアップロード")
                    st.info("自らのレンズで捉えた最高のShotを、図鑑に登録するぜ。")
                    upload_bird = st.selectbox("画像を登録する鳥を選んでくれ:", bird_names, key="upload_select")
                    uploaded_file = st.file_uploader("画像をドロップするか選択してくれ (JPG/PNG)", type=["jpg", "jpeg", "png"])
                    if uploaded_file is not None and upload_bird:
                        if st.button("🚀 Cloudへアップロード", use_container_width=True):
                            with st.spinner("Uploading to Supabase..."):
                                try:
                                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                                    file_ext = uploaded_file.name.split('.')[-1]
                                    safe_file_name = f"img_{timestamp}_{uuid.uuid4().hex[:8]}.{file_ext}"
                                    supabase.storage.from_("bird-images").upload(safe_file_name, uploaded_file.getvalue())
                                    public_image_url = supabase.storage.from_("bird-images").get_public_url(safe_file_name)
                                    supabase.table("bird_master").upsert({"common_name": upload_bird, "image_url": public_image_url}).execute()
                                    st.success(f"🔥 {upload_bird} の画像をCloudに刻み込んだぜ！")
                                except Exception as e:
                                    st.error(f"アップロードに失敗したぜ: {e}")
                
                with tab_data:
                    st.markdown("### 📁 解析データ & 音声のアップロード")
                    
                    # 既存データの取得
                    loc_df = df_all.dropna(subset=['latitude', 'longitude'])
                    loc_master = loc_df[['location_name', 'latitude', 'longitude']].drop_duplicates(subset=['location_name'])
                    
                    # セレクトボックスで既存場所を選択
                    loc_options = ["(新規追加)"] + loc_master['location_name'].tolist()
                    selected_loc = st.selectbox("📍 場所を選択 (新規なら左記を選択)", loc_options)
                    
                    # オートフィル用の初期値
                    initial_lat, initial_lon = 35.319200, 139.546700
                    initial_name = ""
                    
                    if selected_loc != "(新規追加)":
                        match = loc_master[loc_master['location_name'] == selected_loc].iloc[0]
                        initial_name = match['location_name']
                        initial_lat = float(match['latitude'])
                        initial_lon = float(match['longitude'])
                    
                    loc_name_input = st.text_input("場所の名前", value=initial_name)
                    
                    col_lat, col_lon = st.columns(2)
                    with col_lat:
                        lat_input = st.number_input("🌐 緯度", format="%.6f", value=initial_lat)
                    with col_lon:
                        lon_input = st.number_input("🌐 経度", format="%.6f", value=initial_lon)
                        
                    uploaded_csvs = st.file_uploader("📄 BirdNETのCSVを選択 (複数OK)", type=["csv"], accept_multiple_files=True)
                    # 🔥 ここをMP3対応に変更！
                    uploaded_mp3s = st.file_uploader("🎵 録音データ(MP3)を選択 (複数OK)", type=["mp3"], accept_multiple_files=True)
                    
                    if st.button("🚀 DB & Storageへ一括登録", use_container_width=True):
                        if not loc_name_input:
                            st.warning("⚠️ 場所の名前を入力してくれ！")
                        elif not uploaded_csvs and not uploaded_mp3s:
                            st.warning("⚠️ CSVかMP3ファイルを選んでくれ！")
                        else:
                            with st.spinner("Supabaseへ同期中..."):
                                try:
                                    # 1. ローカルで変換済みのMP3をそのままStorageへアップロード
                                    if uploaded_mp3s:
                                        for mp3_file in uploaded_mp3s:
                                            # 🔥 圧縮処理を削ぎ落とし、ファイルを直接クラウドへスルーパス
                                            supabase.storage.from_(BUCKET_NAME).upload(
                                                mp3_file.name, 
                                                mp3_file.getvalue(), 
                                                file_options={"content-type": "audio/mpeg", "upsert": "true"}
                                            )
                                        st.success(f"🎵 {len(uploaded_mp3s)} 個のMP3ファイルをStorageにアップロードしたぜ！")

                                    # 2. CSVデータをパースしてDBへ登録
                                    if uploaded_csvs:
                                        # 🔥 GEʍlNEʍ Hack: StorageのAPI仕様に合わせ、確実にファイルリストを取得
                                        try:
                                            # list() に空文字 "" (ルートパス) を渡すのが supabase-py の正しい仕様
                                            # 🔥 デフォルトの取得件数上限(100件)だと、ファイルが増えたときに
                                            #    古いmp3がリストから漏れて紐付けに失敗するため、上限を引き上げる
                                            storage_objects = supabase.storage.from_(BUCKET_NAME).list(
                                                "", {"limit": 5000, "sortBy": {"column": "name", "order": "asc"}}
                                            )
                                            storage_files = [f['name'] for f in storage_objects] if storage_objects else []
                                        except Exception as e:
                                            st.error(f"Storageのリスト取得エラー: {e}") # 万が一失敗したら画面に表示させる
                                            storage_files = []
                                            
                                        # 同時にアップロードされたMP3と、Storage内の既存ファイルを合体させた「検索リスト」
                                        available_mp3s = [f.name for f in (uploaded_mp3s or [])] + storage_files
                                        
                                        all_data = []
                                        for uploaded_csv in uploaded_csvs:
                                            df_csv = pd.read_csv(uploaded_csv)
                                            df_csv = df_csv.rename(columns={'Start (s)': 'start_sec', 'End (s)': 'end_sec', 'Scientific name': 'scientific_name', 'Common name': 'common_name', 'Confidence': 'confidence'})
                                            
                                            if 'File' in df_csv.columns:
                                                unmatched_names = []  # 🔥 マッチ失敗した元ファイル名を集めて後で警告表示する

                                                def match_mp3_name(wav_filename):
                                                    base_name = os.path.basename(str(wav_filename)).rsplit('.', 1)[0]
                                                    # 🔥 前方10文字 (YYMMDD_nnn 形式) の一致で紐付ける。
                                                    #    CSV側にはhhmmが付かず、mp3側にだけ_hhmmが付くため、
                                                    #    完全一致・アンダースコア区切りに頼らずプレフィックスで判定する。
                                                    key = base_name[:10]
                                                    for mp3_name in available_mp3s:
                                                        if mp3_name[:10] == key and mp3_name.endswith(".mp3"):
                                                            return mp3_name
                                                    unmatched_names.append(base_name)
                                                    return base_name + ".mp3"
                                                    
                                                df_csv['wav_filename'] = df_csv['File'].apply(match_mp3_name)
                                                df_csv = df_csv.drop(columns=['File'])

                                                # 🔥 紐付けに失敗した行があれば、登録前にここで警告する
                                                #    (放置すると閲覧時にファイル未発見エラーになるため)
                                                if unmatched_names:
                                                    unique_unmatched = sorted(set(unmatched_names))
                                                    st.warning(
                                                        "⚠️ 以下のファイルはStorage内で対応するmp3が見つからず、"
                                                        "_hhmmなしのファイル名で登録されます（閲覧時にエラーになる可能性あり）: "
                                                        + ", ".join(unique_unmatched)
                                                    )
                                                
                                            df_csv['location_name'] = loc_name_input
                                            df_csv['latitude'] = lat_input
                                            df_csv['longitude'] = lon_input
                                            all_data.append(df_csv)
                                            
                                        if all_data:
                                            final_df = pd.concat(all_data, ignore_index=True).replace({float('nan'): None})
                                            # 🔥 GEʍlNEʍ Hack: 重複エラー(23505)を回避し、衝突時は上書きする最強のUpsert
                                            supabase.table("detections").upsert(
                                                final_df.to_dict(orient='records'),
                                                on_conflict='wav_filename,start_sec,end_sec,scientific_name'
                                            ).execute()
                                            st.success(f"🔥 {len(final_df)} 件の解析データをDBに刻み込んだぜ！")
                                            
                                    st.rerun()  # 画面をリフレッシュ
                                except Exception as e:
                                    st.error(f"システムエラー: {e}")

        # --- 🐦 鳥の詳細画面 ---
        elif st.session_state.page == 'bird_detail':
            # 🔥 疑似タブナビゲーション（各画面共通）
            col_nav1, col_nav2 = st.columns(2)
            with col_nav1:
                st.button("🐦 鳥から探す", on_click=go_to_main_bird, type="tertiary", use_container_width=True, key=f"nav_{st.session_state.page}_1")
            with col_nav2:
                st.button("📍 場所から探す", on_click=go_to_main_loc, type="tertiary", use_container_width=True, key=f"nav_{st.session_state.page}_2")
            st.markdown("<hr style='margin-top: -10px; margin-bottom: 16px; border-color: #E5E0D2;'>", unsafe_allow_html=True)
            
            target_bird = st.session_state.selected_bird
            
            # --- 🔥 表示件数のステート管理 (初期値10件) ---
            count_key = f"display_count_bird_{target_bird}"
            if count_key not in st.session_state:
                st.session_state[count_key] = 10
                
            bird_data = df_all[df_all['common_name'] == target_bird].sort_values(by='confidence', ascending=False)
            
            if not bird_data.empty:
                scientific_name = bird_data.iloc[0]['scientific_name']
                st.markdown(f"## 🐦 {target_bird}")
                st.caption(f"学名: *{scientific_name}*")
                
                img_response = supabase.table("bird_master").select("image_url").eq("common_name", target_bird).execute()
                if img_response.data and img_response.data[0]['image_url']:
                    st.image(img_response.data[0]['image_url'], use_container_width=True)
                else:
                    st.info("🖼️ まだ画像が登録されていないぜ。「⚙️ 画像管理」タブからUploadしてくれ。")
                
                st.markdown("<div style='min-height: 10px;'></div>", unsafe_allow_html=True)
                
                # --- 🔥 NEW: 信頼性ゲージ ---
                min_confidence = st.slider(
                    "信頼度で絞り込む", 
                    min_value=0, 
                    max_value=100, 
                    value=60, 
                    format="%d%%", 
                    key=f"slider_conf_{target_bird}"
                )
                
                # ゲージの値でデータを事前フィルタリング
                bird_data = bird_data[bird_data['confidence'] >= (min_confidence / 100.0)]
                
                st.markdown("<div style='min-height: 10px;'></div>", unsafe_allow_html=True)

                # --- 🔥 連動型フィルター（カスケード）のロジック ---
                date_key = f"filter_date_{target_bird}"
                loc_key = f"filter_loc_{target_bird}"
                
                # 現在の選択状態を取得 (初期値は未選択)
                current_date = st.session_state.get(date_key, "日にちを選択")
                current_loc = st.session_state.get(loc_key, "場所を選択")
                
                # 📅 場所の選択状態に基づいて、選択可能な「日にち」を絞り込む
                if current_loc != "場所を選択":
                    valid_dates_df = bird_data[bird_data['location_name'] == current_loc]
                else:
                    valid_dates_df = bird_data
                available_dates = ["日にちを選択"] + sorted(valid_dates_df['record_date'].dropna().unique().tolist(), reverse=True)
                
                # 📍 日にちの選択状態に基づいて、選択可能な「場所」を絞り込む
                if current_date != "日にちを選択":
                    valid_locs_df = bird_data[bird_data['record_date'] == current_date]
                else:
                    valid_locs_df = bird_data
                available_locs = ["場所を選択"] + sorted(valid_locs_df['location_name'].dropna().unique().tolist())
                
                # 万が一、前回の選択肢が新しいリストに存在しない場合は未選択状態にリセット
                if current_date not in available_dates:
                    current_date = "日にちを選択"
                if current_loc not in available_locs:
                    current_loc = "場所を選択"
                
                # UIの描画
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    selected_date = st.selectbox("日にち", available_dates, index=available_dates.index(current_date), key=date_key, label_visibility="collapsed")
                with col_f2:
                    selected_loc = st.selectbox("場所", available_locs, index=available_locs.index(current_loc), key=loc_key, label_visibility="collapsed")
                
                # 実データへのフィルター適用 (AND条件)
                if selected_date != "日にちを選択":
                    bird_data = bird_data[bird_data['record_date'] == selected_date]
                if selected_loc != "場所を選択":
                    bird_data = bird_data[bird_data['location_name'] == selected_loc]
                
                st.divider()
                
                # --- 🔥 リスト描画 (10件表示・ロードの処理) ---
                if bird_data.empty:
                    st.warning("指定した条件に一致する録音データは見つからなかったぜ。")
                else:
                    # 上位10件をスライスして表示
                    current_limit = st.session_state[count_key]
                    bird_data_to_show = bird_data.head(current_limit)
                    
                    for index, row in bird_data_to_show.iterrows():
                        with st.container():
                            wav_filename = row['wav_filename']
                            public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(wav_filename)
                            duration = round(float(row['end_sec']) - float(row['start_sec']), 1)
                            confidence_pct = int(row['confidence'] * 100)
                            # 🔥 ファイル名から収録時刻をパースして表示用に成形
                            time_str = format_recording_time_label(wav_filename)

                            
                            # 🔥 GEʍlNEʍ Hack: プレイヤーを描画する前に音声をロードして判定
                            try:
                                with st.spinner("Loading Audio..."):
                                    # 🔥 spec_img も一緒に受け取る
                                    sliced_audio, actual_start, actual_end, channels, spec_img = get_sliced_remote_audio(public_url, float(row['start_sec']), float(row['end_sec']))
                                # バッジのCSSから余計なマージンを削除
                                badge = "<span style='color:#6F8F5E; border:1px solid #6F8F5E; padding:2px 6px; border-radius:4px; font-size:10px; vertical-align:middle;'>🎧 Stereo</span>" if channels >= 2 else "<span style='color:#9A9A8A; border:1px solid #9A9A8A; padding:2px 6px; border-radius:4px; font-size:10px; vertical-align:middle;'>🔈 Mono</span>"
                                audio_loaded = True
                            except Exception as e:
                                badge = ""
                                audio_loaded = False
                                error_msg = e
                            
                            col_meta1, col_meta2 = st.columns([1, 1])
                            with col_meta1:
                                # 🔥 信頼度の横にバッジを自然に配置
                                st.markdown(f"**信頼度:** `{confidence_pct}%` &nbsp; {badge}", unsafe_allow_html=True)
                                st.markdown(f"**再生時間:** `{duration}秒`")
                                if time_str:
                                    st.caption(time_str)
                            with col_meta2:
                                st.button(f"📅 {row['record_date']}", on_click=go_to_date_detail, args=(row['record_date'],), key=f"link_date_{index}")
                                loc_name = row['location_name'] if pd.notna(row['location_name']) else "場所不明"
                                st.button(f"📍 {loc_name}", on_click=go_to_loc_detail, args=(loc_name,), key=f"link_loc_{index}")
                            
                            # 🔥 GEʍlNEʍ Hack: 音と波形が完全同期するカスタムHTMLプレイヤー
                            if audio_loaded:
                                # 🔥 すでにバイナリデータなので、そのままエンコードする
                                audio_b64 = base64.b64encode(sliced_audio).decode()
                                spec_b64 = base64.b64encode(spec_img).decode()
                                
                                custom_player_html = f"""
                                <!DOCTYPE html>
                                <html>
                                <head>
                                <style>
                                    /* ダークモード対応と余白の完全排除 */
                                    body {{ margin: 0; padding: 0; background-color: transparent; color-scheme: dark; overflow: hidden; }}
                                </style>
                                </head>
                                <body>
                                    <div style="display: flex; flex-direction: column; width: 100%;">
                                        <!-- 波形画像と動くライン -->
                                        <div style="position: relative; width: 100%; height: 120px; margin-bottom: 8px; border-radius: 6px; overflow: hidden;">
                                            <img src="data:image/png;base64,{spec_b64}" style="width: 100%; height: 100%; object-fit: fill; display: block;" />
                                            <div id="playhead" style="position: absolute; top: 0; left: 0%; width: 2px; height: 100%; background-color: #6F8F5E; box-shadow: 0 0 8px #6F8F5E; pointer-events: none;"></div>
                                        </div>
                                        <!-- オーディオプレイヤー -->
                                        <audio id="player" controls src="data:audio/mpeg;base64,{audio_b64}" style="width: 100%; height: 40px; outline: none;"></audio>
                                    </div>
                                    <script>
                                        const audio = document.getElementById('player');
                                        const playhead = document.getElementById('playhead');
                                        
                                        // 再生位置に合わせてラインのCSS(left)をパーセンテージで動かす
                                        const updatePlayhead = () => {{
                                            if (audio.duration) {{
                                                const percent = (audio.currentTime / audio.duration) * 100;
                                                playhead.style.left = percent + '%';
                                            }}
                                        }};
                                        
                                        audio.addEventListener('timeupdate', updatePlayhead);
                                        audio.addEventListener('seeked', updatePlayhead);
                                    </script>
                                </body>
                                </html>
                                """
                                # カスタムUIをiframeとして画面にマウント (スクロールバーが出ない絶妙な高さを指定)
                                components.html(custom_player_html, height=175)
                            else:
                                st.error(f"ロードエラー: {error_msg}")
                        st.divider()
                                            
                    # 10件追加ロードボタン
                    if current_limit < len(bird_data):
                        if st.button("🔽 さらに10件読み込む", use_container_width=True, key=f"btn_load_more_bird_{target_bird}"):
                            st.session_state[count_key] += 10
                            st.rerun()

        # --- 📅 日付の詳細画面 ---
        elif st.session_state.page == 'date_detail':
            # 🔥 疑似タブナビゲーション（各画面共通）
            col_nav1, col_nav2 = st.columns(2)
            with col_nav1:
                st.button("🐦 鳥から探す", on_click=go_to_main_bird, type="tertiary", use_container_width=True, key=f"nav_{st.session_state.page}_1")
            with col_nav2:
                st.button("📍 場所から探す", on_click=go_to_main_loc, type="tertiary", use_container_width=True, key=f"nav_{st.session_state.page}_2")
            st.markdown("<hr style='margin-top: -10px; margin-bottom: 16px; border-color: #E5E0D2;'>", unsafe_allow_html=True)
            
            target_date = st.session_state.selected_date
            
            # --- 🔥 表示件数のステート管理 (初期値10件) ---
            count_key = f"display_count_date_{target_date}"
            if count_key not in st.session_state:
                st.session_state[count_key] = 10
                
            st.markdown(f"## 📅 {target_date} の記録")
            day_data = df_all[df_all['record_date'] == target_date]
            
            if not day_data.empty:
                visited_locations = day_data['location_name'].dropna().unique()
                loc_text = "、".join(visited_locations) if len(visited_locations) > 0 else "場所不明"
                st.info(f"📍 **その日に行った場所:** {loc_text}")
                
                # --- 🔥 GEʍlNEʍ Hack: その日の足跡をマップに可視化 ---
                df_map = day_data[['latitude', 'longitude', 'location_name']].dropna().drop_duplicates(subset=['latitude', 'longitude']).copy()
                if not df_map.empty:
                    lat_min, lat_max = df_map['latitude'].min(), df_map['latitude'].max()
                    lon_min, lon_max = df_map['longitude'].min(), df_map['longitude'].max()
                    
                    # 1箇所だけでも美しくズームアウトさせるダミーポイントの計算
                    lat_pad = max((lat_max - lat_min) * 0.8, 0.05)
                    lon_pad = max((lon_max - lon_min) * 0.8, 0.05)
                    
                    df_map['dot_color'] = '#6F8F5E'
                    df_map['dot_size'] = 150
                    
                    dummy_data = pd.DataFrame({
                        'latitude': [lat_min - lat_pad, lat_max + lat_pad],
                        'longitude': [lon_min - lon_pad, lon_max + lon_pad],
                        'location_name': ['dummy', 'dummy'],
                        'dot_color': ['#6F8F5E05', '#6F8F5E05'],
                        'dot_size': [1, 1]
                    })
                    
                    df_map_padded = pd.concat([df_map, dummy_data], ignore_index=True)
                    st.map(df_map_padded, latitude='latitude', longitude='longitude', color='dot_color', size='dot_size', height=200)
                
                st.markdown("<div style='min-height: 10px;'></div>", unsafe_allow_html=True)
                
                # --- 🔥 NEW: 信頼性ゲージ ---
                min_confidence = st.slider(
                    "信頼度で絞り込む", 
                    min_value=0, 
                    max_value=100, 
                    value=60, 
                    format="%d%%", 
                    key=f"slider_conf_date_{target_date}"
                )
                
                # 1. まず信頼度ゲージの値でデータを事前フィルタリング
                day_data = day_data[day_data['confidence'] >= (min_confidence / 100.0)]
                
                st.markdown("<div style='min-height: 10px;'></div>", unsafe_allow_html=True)
                
                # --- 🔥 NEW: 鳥で絞り込むフィルター (完全画像化による高さ統一) ---
                if not day_data.empty:
                    available_birds = sorted(day_data['common_name'].dropna().unique().tolist())
                    
                    filter_state_key = f"filter_bird_date_{target_date}"
                    if filter_state_key not in st.session_state:
                        st.session_state[filter_state_key] = "すべて"
                        
                    st.markdown("<br>**🐦 鳥で絞り込む**", unsafe_allow_html=True)

                    # 🔥 GEʍlNEʍ Hack: 動的にダミー画像を生成する関数 (フォント巨大化＆完全中央揃え版)
                    def create_dummy_image(text, bg_color="#E5E0D2", text_color="#9A9A8A"):
                        # 300x300の正方形画像を作成
                        img = Image.new('RGB', (300, 300), color=bg_color)
                        d = ImageDraw.Draw(img)
                        
                        # お前の環境(Pillow 12.3.0)のパワーを使ってデフォルトフォントを巨大化
                        try:
                            font = ImageFont.load_default(size=80)
                        except:
                            font = ImageFont.load_default()
                            
                        # anchor="mm" (Middle-Middle) を使って、(150, 150)のど真ん中に完璧にセンタリング
                        d.text((150, 150), text, fill=text_color, font=font, anchor="mm")
                        return img

                    MAX_COLS = 5
                    filter_items = ["ALL_RESET"] + available_birds
                    
                    for i in range(0, len(filter_items), MAX_COLS):
                        cols = st.columns(MAX_COLS)
                        chunk = filter_items[i:i + MAX_COLS]
                        
                        for j, item in enumerate(chunk):
                            with cols[j]:
                                if item == "ALL_RESET":
                                    # Allボタン用の画像を動的生成して表示
                                    all_img = create_dummy_image("ALL", bg_color="#262730", text_color="#FFFFFF")
                                    st.image(all_img, use_container_width=True)
                                    
                                    if st.button("解除", key=f"btn_filter_all_{target_date}", use_container_width=True):
                                        st.session_state[filter_state_key] = "すべて"
                                        st.rerun()
                                else:
                                    # 鳥のサムネイル画像
                                    img_url = bird_images.get(item)
                                    if img_url:
                                        st.image(img_url, use_container_width=True)
                                    else:
                                        # No Img用の画像を動的生成して表示
                                        no_img = create_dummy_image("No Img")
                                        st.image(no_img, use_container_width=True)
                                    
                                    # 選択状態のボタン
                                    is_selected = (st.session_state[filter_state_key] == item)
                                    btn_label = "✅" if is_selected else item[:3]
                                    
                                    if st.button(btn_label, key=f"btn_filter_{item}_{target_date}", use_container_width=True, help=item):
                                        if is_selected:
                                            st.session_state[filter_state_key] = "すべて"
                                        else:
                                            st.session_state[filter_state_key] = item
                                        st.rerun()

                    if st.session_state[filter_state_key] != "すべて":
                        day_data = day_data[day_data['common_name'] == st.session_state[filter_state_key]]
                        st.info(f"🔍 **{st.session_state[filter_state_key]}** に絞り込み中")
                        
                        
                
                if day_data.empty:
                    st.warning("指定した条件に一致する録音データは見つからなかったぜ。")
                else:
                    # フィルタリング後の件数を表示
                    st.markdown(f"### 🐦 その日の鳥 (計 {len(day_data)} 件)")
                    
                    day_data_sorted = day_data.sort_values(by='confidence', ascending=False)
                    
                    # --- 🔥 上位10件をスライスして表示 ---
                    current_limit = st.session_state[count_key]
                    day_data_to_show = day_data_sorted.head(current_limit)
                    
                    for index, row in day_data_to_show.iterrows():
                        with st.container():
                            wav_filename = row['wav_filename']
                            public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(wav_filename)
                            # 🔥 ファイル名から収録時刻をパースして表示用に成形
                            time_str = format_recording_time_label(wav_filename)
                            
                            # 🔥 プレイヤーを描画する前に音声をロード
                            try:
                                with st.spinner("Loading Audio..."):
                                    # 🔥 spec_img も一緒に受け取る
                                    sliced_audio, actual_start, actual_end, channels, spec_img = get_sliced_remote_audio(public_url, float(row['start_sec']), float(row['end_sec']))
                                # 余計なマイナスマージンを完全削除
                                badge = "<span style='color:#6F8F5E; border:1px solid #6F8F5E; padding:2px 6px; border-radius:4px; font-size:10px; vertical-align:middle;'>🎧 Stereo</span>" if channels >= 2 else "<span style='color:#9A9A8A; border:1px solid #9A9A8A; padding:2px 6px; border-radius:4px; font-size:10px; vertical-align:middle;'>🔈 Mono</span>"
                                audio_loaded = True
                            except Exception as e:
                                badge = ""
                                audio_loaded = False
                                error_msg = e

                            confidence_pct = int(row['confidence'] * 100)
                            
                            # 🔥 カラム分けの比率を変え、ボタンとバッジだけを横に並べる
                            col_head1, col_head2 = st.columns([2, 3])
                            with col_head1:
                                st.button(f"**{row['common_name']}**", on_click=go_to_bird_detail, args=(row['common_name'],), key=f"link_bird_date_{index}", use_container_width=True)
                            with col_head2:
                                # スマホの幅でも被らないように適度な上余白を設定
                                st.markdown(f"<div style='padding-top: 8px;'>{badge}</div>", unsafe_allow_html=True)
                            
                            # 🔥 ゲージとプレイヤーはカラムの「外」に出し、フル幅で贅沢に使う
                            if time_str:
                                st.caption(time_str)
                            st.progress(row['confidence'], text=f"信頼度: {confidence_pct}%")
                            
                            # 🔥 GEʍlNEʍ Hack: 音と波形が完全同期するカスタムHTMLプレイヤー
                            if audio_loaded:
                                # 🔥 すでにバイナリデータなので、そのままエンコードする
                                audio_b64 = base64.b64encode(sliced_audio).decode()
                                spec_b64 = base64.b64encode(spec_img).decode()
                                
                                custom_player_html = f"""
                                <!DOCTYPE html>
                                <html>
                                <head>
                                <style>
                                    /* ダークモード対応と余白の完全排除 */
                                    body {{ margin: 0; padding: 0; background-color: transparent; color-scheme: dark; overflow: hidden; }}
                                </style>
                                </head>
                                <body>
                                    <div style="display: flex; flex-direction: column; width: 100%;">
                                        <!-- 波形画像と動くライン -->
                                        <div style="position: relative; width: 100%; height: 120px; margin-bottom: 8px; border-radius: 6px; overflow: hidden;">
                                            <img src="data:image/png;base64,{spec_b64}" style="width: 100%; height: 100%; object-fit: fill; display: block;" />
                                            <div id="playhead" style="position: absolute; top: 0; left: 0%; width: 2px; height: 100%; background-color: #6F8F5E; box-shadow: 0 0 8px #6F8F5E; pointer-events: none;"></div>
                                        </div>
                                        <!-- オーディオプレイヤー -->
                                        <audio id="player" controls src="data:audio/mpeg;base64,{audio_b64}" style="width: 100%; height: 40px; outline: none;"></audio>
                                    </div>
                                    <script>
                                        const audio = document.getElementById('player');
                                        const playhead = document.getElementById('playhead');
                                        
                                        // 再生位置に合わせてラインのCSS(left)をパーセンテージで動かす
                                        const updatePlayhead = () => {{
                                            if (audio.duration) {{
                                                const percent = (audio.currentTime / audio.duration) * 100;
                                                playhead.style.left = percent + '%';
                                            }}
                                        }};
                                        
                                        audio.addEventListener('timeupdate', updatePlayhead);
                                        audio.addEventListener('seeked', updatePlayhead);
                                    </script>
                                </body>
                                </html>
                                """
                                # カスタムUIをiframeとして画面にマウント (スクロールバーが出ない絶妙な高さを指定)
                                components.html(custom_player_html, height=175)
                            else:
                                st.error(f"ロードエラー: {error_msg}")
                                
                        st.divider()
                    
                    # --- 🔥 10件追加ロードボタン ---
                    if current_limit < len(day_data):
                        if st.button("🔽 さらに10件読み込む", use_container_width=True, key=f"btn_load_more_date_{target_date}"):
                            st.session_state[count_key] += 10
                            st.rerun()

        # --- 📍 場所の詳細画面 ---
        elif st.session_state.page == 'loc_detail':
            # 🔥 疑似タブナビゲーション（各画面共通）
            col_nav1, col_nav2 = st.columns(2)
            with col_nav1:
                st.button("🐦 鳥から探す", on_click=go_to_main_bird, type="tertiary", use_container_width=True, key=f"nav_{st.session_state.page}_1")
            with col_nav2:
                st.button("📍 場所から探す", on_click=go_to_main_loc, type="tertiary", use_container_width=True, key=f"nav_{st.session_state.page}_2")
            st.markdown("<hr style='margin-top: -10px; margin-bottom: 16px; border-color: #E5E0D2;'>", unsafe_allow_html=True)
            
            target_loc = st.session_state.selected_loc
            
            st.markdown(f"## 📍 {target_loc} の記録")
            loc_data = df_all[df_all['location_name'] == target_loc].sort_values(by='record_date', ascending=False)
            
            if not loc_data.empty:
                # 🔥 GEʍlNEʍ Hack: 場所の詳細画面にもピン付きのマップをドロップ
                df_map = loc_data[['latitude', 'longitude', 'location_name']].drop_duplicates(subset=['latitude', 'longitude']).copy()
                if not df_map.empty:
                    lat_min, lat_max = df_map['latitude'].min(), df_map['latitude'].max()
                    lon_min, lon_max = df_map['longitude'].min(), df_map['longitude'].max()
                    
                    lat_pad = max((lat_max - lat_min) * 0.8, 0.05)
                    lon_pad = max((lon_max - lon_min) * 0.8, 0.05)
                    
                    df_map['dot_color'] = '#6F8F5E'
                    df_map['dot_size'] = 150
                    
                    dummy_data = pd.DataFrame({
                        'latitude': [lat_min - lat_pad, lat_max + lat_pad],
                        'longitude': [lon_min - lon_pad, lon_max + lon_pad],
                        'location_name': ['dummy', 'dummy'],
                        'dot_color': ['#6F8F5E05', '#6F8F5E05'],
                        'dot_size': [1, 1]
                    })
                    
                    df_map_padded = pd.concat([df_map, dummy_data], ignore_index=True)
                    st.map(df_map_padded, latitude='latitude', longitude='longitude', color='dot_color', size='dot_size', height=200)
                    
                    st.markdown("<div style='min-height: 10px;'></div>", unsafe_allow_html=True)

                dates_at_loc = loc_data['record_date'].unique()
                for date_str in dates_at_loc:
                    count = len(loc_data[loc_data['record_date'] == date_str])
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"### 📅 {date_str}")
                            st.caption(f"🎧 検出: {count} 件")
                        with col2:
                            st.button("詳細", on_click=go_to_date_detail, args=(date_str,), key=f"btn_loc_detail_{target_loc}_{date_str}")
                    st.divider()
            else:
                st.warning("この場所のデータは見つからなかったぜ。")
    else:
        st.warning("クラウドのデータベースにデータがないぜ。")
except Exception as e:
    st.error(f"システムエラー: {e}")