import streamlit as st
import pandas as pd
import os
import io
import requests
import re
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from scipy.io import wavfile
import uuid
import math

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

# --- 2. リモート音声の取得と切り出し ---
@st.cache_data(show_spinner=False)
def get_sliced_remote_wav(file_url, original_start, original_end):
    response = requests.get(file_url)
    if response.status_code != 200:
        raise Exception("ファイルのダウンロードに失敗したぜ。")
        
    wav_io = io.BytesIO(response.content)
    sample_rate, data = wavfile.read(wav_io)
    total_duration = len(data) / sample_rate
    
    start_sec = max(0.0, original_start - 1.5)
    end_sec = min(total_duration, original_end + 1.5)
    
    start_frame = int(start_sec * sample_rate)
    end_frame = int(end_sec * sample_rate)
    
    sliced_data = data[start_frame:end_frame]
    
    out_wav = io.BytesIO()
    wavfile.write(out_wav, sample_rate, sliced_data)
    out_wav.seek(0)
    
    return out_wav, start_sec, end_sec

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

# --- 2.8. 画面遷移のコントロール（Session State） ---
if 'page' not in st.session_state: st.session_state.page = 'main'
if 'selected_date' not in st.session_state: st.session_state.selected_date = None
if 'selected_bird' not in st.session_state: st.session_state.selected_bird = None
if 'selected_loc' not in st.session_state: st.session_state.selected_loc = None
if 'loaded_audio' not in st.session_state: st.session_state.loaded_audio = set()

def load_audio_clip(audio_key):
    st.session_state.loaded_audio.add(audio_key)

def go_to_date_detail(date_str):
    st.session_state.page = 'date_detail'
    st.session_state.selected_date = date_str

def go_to_bird_detail(bird_name):
    st.session_state.page = 'bird_detail'
    st.session_state.selected_bird = bird_name

def go_to_loc_detail(loc_name):
    st.session_state.page = 'loc_detail'
    st.session_state.selected_loc = loc_name

def go_to_main():
    st.session_state.page = 'main'
    st.session_state.selected_date = None
    st.session_state.selected_bird = None
    st.session_state.selected_loc = None

# --- 3. メインUI ---
st.markdown("<h2 style='font-size: 26px; font-weight: bold; padding-top: 10px; text-align: center;'>🎧 Ambient Bird Log 🐦</h2>", unsafe_allow_html=True)

# 🔥 GEʍlNEʍ's CSS Hack (トップ余白削減・安定版)
st.markdown("""
    <style>
    /* --- 🔥 NEW: Streamlitデフォルトの巨大な上部余白を削ぎ落とす --- */
    .block-container {
        padding-top: 1.5rem !important; /* デフォルトの約6remから大幅に削減 */
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
    @media (max-width: 640px) {
        div[data-testid="stHorizontalBlock"] { flex-direction: row !important; flex-wrap: nowrap !important; }
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] { min-width: 0 !important; padding: 0 3px !important; }
        button p { font-size: 10px !important; }
    }
    button p { font-size: 12px !important; }
    </style>
""", unsafe_allow_html=True)

try:
    response_all = supabase.table("detections").select("*").limit(10000).execute()
    response_master = supabase.table("bird_master").select("*").execute()
    bird_images = {row['common_name']: row['image_url'] for row in response_master.data} if response_master.data else {}
    
    if response_all.data:
        df_all = pd.DataFrame(response_all.data)
        df_all['record_date'] = df_all['wav_filename'].apply(extract_date)
        
        if st.session_state.page == 'main':
            bird_names = sorted(df_all['common_name'].dropna().unique().tolist())
            is_admin = st.query_params.get("admin") == "true"
            
            if is_admin:
                tab_bird, tab_location, tab_admin, tab_data = st.tabs(["🐦 鳥から探す", "📍 場所から探す", "⚙️ 画像管理", "📁 データ登録"])
            else:
                tab_bird, tab_location = st.tabs(["🐦 鳥から探す", "📍 場所から探す"])

            with tab_bird:
                min_confidence = st.slider("信頼度", min_value=0, max_value=100, value=60, format="%d%%")
                st.markdown("<div style='min-height: 40px;'></div>", unsafe_allow_html=True)
                search_query = st.text_input("検索窓", label_visibility="collapsed", placeholder="和名で検索")
                st.markdown("<div style='min-height: 20px;'></div>", unsafe_allow_html=True)
                
                df_filtered = df_all[df_all['confidence'] >= (min_confidence / 100.0)]
                bird_counts = df_filtered['common_name'].value_counts()
                filtered_birds = {name: count for name, count in bird_counts.items() if not search_query or search_query in name}
                
                cols = st.columns(3)
                for i, (bird_name, count) in enumerate(filtered_birds.items()):
                    col_idx = i % 3
                    with cols[col_idx]:
                        with st.container():
                            img_url = bird_images.get(bird_name)
                            if img_url:
                                st.image(img_url, use_container_width=True)
                            else:
                                st.markdown("<div style='background-color:#1E1E1E; border-radius:6px; width: 100%; aspect-ratio: 1/1; display:flex; align-items:center; justify-content:center; margin-bottom: 4px;'><span style='color:#8E8E93; font-size:10px;'>No Img</span></div>", unsafe_allow_html=True)
                            if st.button(f"{bird_name}", key=f"btn_bird_{bird_name}", use_container_width=True):
                                go_to_bird_detail(bird_name)
                                st.rerun()
                        st.markdown("<div style='min-height: 20px;'></div>", unsafe_allow_html=True)

            with tab_location:
                st.markdown("### 🗺️ 場所から探す")
                df_loc = df_all.dropna(subset=['latitude', 'longitude'])
                if not df_loc.empty:
                    df_map = df_loc[['latitude', 'longitude', 'location_name']].drop_duplicates(subset=['latitude', 'longitude'])
                    st.map(df_map, latitude='latitude', longitude='longitude', color='#39FF14', size=150)
                    st.divider()
                    st.markdown("**📍 過去の記録（場所別）**")
                    unique_locations = sorted(df_loc['location_name'].unique().tolist())
                    selected_loc = st.selectbox("場所を選択してくれ:", unique_locations, label_visibility="collapsed")
                    if selected_loc:
                        loc_filtered = df_loc[df_loc['location_name'] == selected_loc].sort_values(by='record_date', ascending=False)
                        dates_at_loc = loc_filtered['record_date'].unique()
                        for date_str in dates_at_loc:
                            count = len(loc_filtered[loc_filtered['record_date'] == date_str])
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"### 📅 {date_str}")
                                    st.caption(f"🎧 検出: {count} 件")
                                with col2:
                                    st.button("詳細", on_click=go_to_date_detail, args=(date_str,), key=f"btn_{selected_loc}_{date_str}")
                            st.divider()

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
                
                # 📍 NEW: BirdNETのCSVを一括登録する管理画面
                # 📍 修正箇所: 📁 データ登録タブ
                # 📍 修正箇所: 📁 データ登録タブ
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
                        
                    # 📍 修正: CSVとWAVの両方を待ち受けるUploader
                    uploaded_csvs = st.file_uploader("📄 BirdNETのCSVを選択 (複数OK)", type=["csv"], accept_multiple_files=True)
                    uploaded_wavs = st.file_uploader("🎵 録音データ(WAV)を選択 (複数OK)", type=["wav"], accept_multiple_files=True)
                    
                    if st.button("🚀 DB & Storageへ一括登録", use_container_width=True):
                        if not loc_name_input:
                            st.warning("⚠️ 場所の名前を入力してくれ！")
                        elif not uploaded_csvs and not uploaded_wavs:
                            st.warning("⚠️ CSVかWAVファイルを選んでくれ！")
                        else:
                            with st.spinner("Supabaseへ同期中..."):
                                try:
                                    # 1. WAVファイルをStorageへ直接アップロード
                                    if uploaded_wavs:
                                        for wav_file in uploaded_wavs:
                                            supabase.storage.from_(BUCKET_NAME).upload(
                                                wav_file.name, 
                                                wav_file.getvalue(), 
                                                file_options={"upsert": "true"}
                                            )
                                        st.success(f"🎵 {len(uploaded_wavs)} 個のWAVファイルをStorageにアップロードしたぜ！")

                                    # 2. CSVデータをパースしてDBへ登録
                                    if uploaded_csvs:
                                        all_data = []
                                        for uploaded_csv in uploaded_csvs:
                                            df_csv = pd.read_csv(uploaded_csv)
                                            df_csv = df_csv.rename(columns={'Start (s)': 'start_sec', 'End (s)': 'end_sec', 'Scientific name': 'scientific_name', 'Common name': 'common_name', 'Confidence': 'confidence'})
                                            if 'File' in df_csv.columns:
                                                df_csv['wav_filename'] = df_csv['File'].apply(lambda x: os.path.basename(str(x)))
                                                df_csv = df_csv.drop(columns=['File'])
                                            df_csv['location_name'] = loc_name_input
                                            df_csv['latitude'] = lat_input
                                            df_csv['longitude'] = lon_input
                                            all_data.append(df_csv)
                                            
                                        if all_data:
                                            final_df = pd.concat(all_data, ignore_index=True).replace({float('nan'): None})
                                            supabase.table("detections").insert(final_df.to_dict(orient='records')).execute()
                                            st.success(f"🔥 {len(final_df)} 件の解析データをDBに刻み込んだぜ！")
                                            
                                    st.rerun()  # 画面をリフレッシュ
                                except Exception as e:
                                    st.error(f"システムエラー: {e}")

        # --- 🐦 鳥の詳細画面 ---
        elif st.session_state.page == 'bird_detail':
            # 🔥 疑似タブナビゲーション（緑の強調なし）
            col_nav1, col_nav2 = st.columns(2)
            with col_nav1:
                st.button("🐦 鳥から探す", on_click=go_to_main, type="tertiary", use_container_width=True, key="nav_bird_detail_1")
            with col_nav2:
                st.button("📍 場所から探す", on_click=go_to_main, type="tertiary", use_container_width=True, key="nav_bird_detail_2")
            st.markdown("<hr style='margin-top: -10px; margin-bottom: 16px; border-color: #444;'>", unsafe_allow_html=True)
            
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
                            
                            col_meta1, col_meta2 = st.columns([1, 1])
                            with col_meta1:
                                st.markdown(f"**信頼度:** `{confidence_pct}%`")
                                st.markdown(f"**再生時間:** `{duration}秒`")
                            with col_meta2:
                                st.button(f"📅 {row['record_date']}", on_click=go_to_date_detail, args=(row['record_date'],), key=f"link_date_{index}")
                                loc_name = row['location_name'] if pd.notna(row['location_name']) else "場所不明"
                                st.button(f"📍 {loc_name}", on_click=go_to_loc_detail, args=(loc_name,), key=f"link_loc_{index}")
                            
                            # 表示されたものは無条件で物理カット＆自動ロード
                            try:
                                with st.spinner("Loading Audio..."):
                                    sliced_audio, actual_start, actual_end = get_sliced_remote_wav(public_url, float(row['start_sec']), float(row['end_sec']))
                                st.audio(sliced_audio, format="audio/wav")
                            except Exception as e:
                                st.error(f"ロードエラー: {e}")
                        st.divider()
                    
                    # 10件追加ロードボタン
                    if current_limit < len(bird_data):
                        if st.button("🔽 さらに10件読み込む", use_container_width=True, key=f"btn_load_more_bird_{target_bird}"):
                            st.session_state[count_key] += 10
                            st.rerun()

        # --- 📅 日付の詳細画面 ---
        elif st.session_state.page == 'date_detail':
            # 🔥 疑似タブナビゲーション（緑の強調なし）
            col_nav1, col_nav2 = st.columns(2)
            with col_nav1:
                st.button("🐦 鳥から探す", on_click=go_to_main, type="tertiary", use_container_width=True, key="nav_date_detail_1")
            with col_nav2:
                st.button("📍 場所から探す", on_click=go_to_main, type="tertiary", use_container_width=True, key="nav_date_detail_2")
            st.markdown("<hr style='margin-top: -10px; margin-bottom: 16px; border-color: #444;'>", unsafe_allow_html=True)
            
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
                st.markdown(f"### 🐦 その日の鳥 (計 {len(day_data)} 件)")
                
                day_data_sorted = day_data.sort_values(by='confidence', ascending=False)
                
                # --- 🔥 上位10件をスライスして表示 ---
                current_limit = st.session_state[count_key]
                day_data_to_show = day_data_sorted.head(current_limit)
                
                for index, row in day_data_to_show.iterrows():
                    with st.container():
                        col1, col2 = st.columns([3, 2])
                        wav_filename = row['wav_filename']
                        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(wav_filename)
                        
                        with col1:
                            confidence_pct = int(row['confidence'] * 100)
                            st.button(f"**{row['common_name']}**", on_click=go_to_bird_detail, args=(row['common_name'],), key=f"link_bird_date_{index}")
                            st.progress(row['confidence'], text=f"信頼度: {confidence_pct}%")
                            
                        with col2:
                            # 表示された10件は無条件で物理カット＆自動ロード
                            try:
                                with st.spinner("Loading..."):
                                    sliced_audio, actual_start, actual_end = get_sliced_remote_wav(public_url, float(row['start_sec']), float(row['end_sec']))
                                st.audio(sliced_audio, format="audio/wav")
                            except Exception as e:
                                st.error(f"ロードエラー: {e}")
                    st.divider()
                
                # --- 🔥 10件追加ロードボタン ---
                if current_limit < len(day_data):
                    if st.button("🔽 さらに10件読み込む", use_container_width=True, key=f"btn_load_more_date_{target_date}"):
                        st.session_state[count_key] += 10
                        st.rerun()

        elif st.session_state.page == 'loc_detail':
            # 🔥 疑似タブナビゲーション（緑の強調なし）
            col_nav1, col_nav2 = st.columns(2)
            with col_nav1:
                st.button("🐦 鳥から探す", on_click=go_to_main, type="tertiary", use_container_width=True, key="nav_loc_detail_1")
            with col_nav2:
                st.button("📍 場所から探す", on_click=go_to_main, type="tertiary", use_container_width=True, key="nav_loc_detail_2")
            st.markdown("<hr style='margin-top: -10px; margin-bottom: 16px; border-color: #444;'>", unsafe_allow_html=True)
            
            target_loc = st.session_state.selected_loc
            
            st.markdown(f"## 📍 {target_loc} の記録")
            loc_data = df_all[df_all['location_name'] == target_loc].sort_values(by='record_date', ascending=False)
            if not loc_data.empty:
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